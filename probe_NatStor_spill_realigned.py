import os, re, gc, pickle
import numpy as np
import pandas as pd
from os import path
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- reproducible paths: resolve repo root (the dir containing utils.py) ---
from pathlib import Path as _Path
_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())
import os as _os; _os.chdir(str(PROJECT_ROOT))
from utils import test_model_Ridge, significance_testing, base_plot, decompose_sfl

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
CACHE_DIR = "embeddings"
RESULTS_CSV = "results_natstor_sentencewise.csv"
SEED = 0
os.makedirs(CACHE_DIR, exist_ok=True)

natstor = pd.read_csv("datasets/natstor.tsv", sep="\t").sort_values(["item","zone"]).reset_index(drop=True)
grouped = natstor.groupby("item", sort=True)
stories_words = grouped["word"].apply(list).tolist()
stories_rts   = grouped["gmeanItemRT"].apply(list).tolist()

print(f"#stories: {len(stories_words)}")
for i, (w, r) in enumerate(zip(stories_words, stories_rts), 1):
    if len(w) != len(r):
        print(f"[WARN] Story {i}: {len(w)} words vs {len(r)} RTs")

_end_pat = re.compile(r'[\.!?](?:["\')\]]*)$')
def split_sentences(words, rts):
    out_w, out_rt, cur_w, cur_rt = [], [], [], []
    for w, rt in zip(words, rts):
        cur_w.append(w); cur_rt.append(rt)
        if _end_pat.search(w):
            out_w.append(cur_w); out_rt.append(cur_rt)
            cur_w, cur_rt = [], []
    if cur_w:
        out_w.append(cur_w); out_rt.append(cur_rt)
    return out_w, out_rt

stories_sents_words, stories_sents_rts = [], []
for w, r in zip(stories_words, stories_rts):
    sw, sr = split_sentences(w, r)
    stories_sents_words.append(sw)
    stories_sents_rts.append(sr)

natst_sent_words = [s for story in stories_sents_words for s in story]
natst_sent_rts   = [s for story in stories_sents_rts   for s in story]

freq_tbl = pd.read_excel("subtlex.xlsx")
freq_tbl.columns = [str(c).strip() for c in freq_tbl.columns]
freq_pos = freq_tbl.loc[freq_tbl["FREQcount"] > 0, ["Word","FREQcount"]].copy()
logf_map = dict(zip(freq_pos["Word"], np.log(freq_pos["FREQcount"].astype(float))))
oov_logf = float(np.min(list(logf_map.values())))
_punct = re.compile(r"[^\w'-]+", flags=re.UNICODE)

def get_f(word: str) -> float:
    return logf_map.get(_punct.sub("", word), oov_logf)

def get_l(word: str) -> int:
    return len(_punct.sub("", word))

def get_surprisal(prompt):
    inputs = toker(prompt, return_tensors="pt")
    input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]
    outputs = model(**inputs, labels=input_ids)
    logits = outputs.logits
    logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2))
    return [-item[0] for item in logprobs.tolist()[0]]

def tok_maker(tokens, sep, cased=False):
    plain = " ".join(tokens)
    sub = [re.sub(sep, "", t) for t in toker.tokenize(plain)]
    groups = []
    for element in (tokens if cased else [t.lower() for t in tokens]):
        tmp = []
        while "".join(tmp) != element:
            tmp.append(sub.pop(0))
        groups.append(tmp)
    return groups

def get_surprisal_tokens(tokens, sep, cased=False):
    s = get_surprisal(" ".join(tokens))
    groups = tok_maker(tokens, sep, cased)
    out, k = [], 0
    for g in groups[1:]:  # skip first word
        if len(g) == 1:
            out.append(s[k]); k += 1
        else:
            out.append(sum(s[k:k+len(g)])); k += len(g)
    return out  # length = len(tokens)-1

def word_embeddings_for_sentence(words, tokenizer, model, sep, cased=True):
    if not words:
        H = model.config.n_embd
        return np.empty((0, H), dtype=np.float32)
    enc = tokenizer(" ".join(words), return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model(enc["input_ids"], output_hidden_states=True)
    last = out.hidden_states[-1][0].detach().cpu().numpy()  # [T, H]
    groups = tok_maker(words, sep, cased=cased)
    n_subtok = sum(len(g) for g in groups)
    if n_subtok != last.shape[0]:
        raise RuntimeError(f"Token alignment mismatch: subtoks={n_subtok} vs last.T={last.shape[0]}")
    rows, k = [], 0
    for g in groups:
        L = len(g)
        rows.append(last[k] if L == 1 else np.mean(last[k:k+L], axis=0))
        k += L
    E = np.vstack(rows).astype(np.float32)
    assert E.shape[0] == len(words)
    return E

_debug_printed = 0
def build_sentence_windows(surp, logf, length, embs, rt_sent, words):
    k_minus_1 = len(surp)  # surprisal for words[1..k-1]
    if k_minus_1 == 0:
        return np.empty((0,9)), np.empty((0, 3*embs.shape[1])), np.empty((0,))
    rows_b, rows_e, rows_y, idx_triplets = [], [], [], []
    for j in range(2, k_minus_1):  # j indexes surp/logf/len; target word i=j+1
        rows_b.append([surp[j],logf[j],length[j],
                       surp[j-1],logf[j-1],length[j-1],
                       surp[j-2],logf[j-2],length[j-2]])
        w_im2, w_im1, w_i = j-1, j, j+1
        rows_e.append(np.concatenate([embs[w_im2], embs[w_im1], embs[w_i]]))
        rows_y.append(rt_sent[w_i])
        idx_triplets.append((w_im2, w_im1, w_i))
    global _debug_printed
    if _debug_printed < 3 and idx_triplets:
        w_im2, w_im1, w_i = idx_triplets[-1]
        print(f"[spill-check] predict RT(word={w_i}, '{words[w_i]}') from ('{words[w_im2]}','{words[w_im1]}','{words[w_i]}')")
        _debug_printed += 1
    return np.asarray(rows_b, float), np.asarray(rows_e, np.float32), np.asarray(rows_y, float)

def drop_nonfinite_rows(X_list, y):
    mask = np.isfinite(y)
    for X in X_list:
        mask &= np.isfinite(X).all(axis=1)
    return [X[mask] for X in X_list], y[mask]

def sanity_banner(name, arr):
    print(f"{name}: shape={getattr(arr,'shape',None)} finite={np.isfinite(arr).mean():.4f} "
          f"min={np.nanmin(arr):.4g} max={np.nanmax(arr):.4g}")

def quick_corr_checks(X_base, y, label=""):
    cols = ["s_i","logf_i","len_i","s_im1","logf_im1","len_im1","s_im2","logf_im2","len_im2"]
    print(f"\n[quick-corr] {label}  n={len(y)}")
    for i, c in enumerate(cols):
        x = X_base[:, i]
        r_p, p_p = pearsonr(x, y); r_s, p_s = spearmanr(x, y)
        mu, sd = float(np.mean(x)), float(np.std(x))
        print(f"rt ~ {c:9s}: r={r_p:+.3f} (p={p_p:.1e}) | ρ={r_s:+.3f} (p={p_s:.1e}) | mean={mu:.3g}, sd={sd:.3g}")
    r_sf, p_sf = pearsonr(X_base[:,0], X_base[:,1])
    r_lf, p_lf = pearsonr(X_base[:,2], X_base[:,1])
    print(f"s_i vs logf_i: r={r_sf:+.3f} (p={p_sf:.1e})  | len_i vs logf_i: r={r_lf:+.3f} (p={p_lf:.1e})")

def decile_means(x, y, q=10):
    qs = np.linspace(0, 1, q+1)
    edges = np.quantile(x, qs, method="nearest")
    bins = np.digitize(x, edges[1:-1], right=True)
    return np.array([y[bins==k].mean() if np.any(bins==k) else np.nan for k in range(q)])

def cache_paths(tag):
    return (f"{CACHE_DIR}/{tag}_sentence_cache.pkl", f"{CACHE_DIR}/{tag}_features.pkl")

def load_sentence_cache(tag):
    p, _ = cache_paths(tag)
    if path.isfile(p):
        with open(p, "rb") as fh: return pickle.load(fh)
    return None

def save_sentence_cache(tag, surprisals_list, embeddings_list):
    p, _ = cache_paths(tag)
    with open(p, "wb") as fh:
        pickle.dump({'surprisals': surprisals_list, 'embeddings': embeddings_list}, fh, protocol=pickle.HIGHEST_PROTOCOL)

def load_feature_cache(tag):
    _, p = cache_paths(tag)
    if path.isfile(p):
        with open(p, "rb") as fh: return pickle.load(fh)
    return None

def save_feature_cache(tag, X_base, X_emb, y):
    _, p = cache_paths(tag)
    with open(p, "wb") as fh:
        pickle.dump({'X_base': X_base, 'X_emb': X_emb, 'y': y}, fh, protocol=pickle.HIGHEST_PROTOCOL)

MODELS = [
    ("openai-gpt",  "</w>", False, "gpt"),
    ("gpt2",        "Ġ",    True,  "gpt2_small"),
    ("gpt2-medium", "Ġ",    True,  "gpt2_med"),
    ("gpt2-large",  "Ġ",    True,  "gpt2_large"),
    ("gpt2-xl",     "Ġ",    True,  "gpt2_xl"),
]

results_rows = []

for model_name, sep, cased, tag in MODELS:
    print(f"\n=== {model_name} ({tag}) ===")
    # 1) Try full feature cache
    feat_cached = load_feature_cache(tag)
    if feat_cached is not None:
        X_base = feat_cached['X_base']; X_emb = feat_cached['X_emb']; y_nat = feat_cached['y']
        print(f"[{tag}] loaded cached features: X_base={X_base.shape} X_emb={X_emb.shape} y={y_nat.shape}")
    else:
        sent_cached = load_sentence_cache(tag)
        if sent_cached is None:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            toker = AutoTokenizer.from_pretrained(model_name)
            model.eval()
            surprisals_list, embeddings_list = [], []
            for words in tqdm(natst_sent_words, total=len(natst_sent_words), desc=f"{tag} build-cache", leave=False):
                if not words:
                    surprisals_list.append(None); embeddings_list.append(None); continue
                try:
                    s_sent = get_surprisal_tokens(words, sep=sep, cased=cased)
                    E_sent = word_embeddings_for_sentence(words, toker, model, sep=sep, cased=cased)
                except Exception:
                    surprisals_list.append(None); embeddings_list.append(None); continue
                surprisals_list.append(s_sent); embeddings_list.append(E_sent)
            save_sentence_cache(tag, surprisals_list, embeddings_list)
            del model, toker
            gc.collect()
        else:
            surprisals_list = sent_cached['surprisals']
            embeddings_list = sent_cached['embeddings']
            print(f"[{tag}] loaded cached sentence-level data")

        Xb_list, Xe_list, y_list = [], [], []
        for words, rt_sent, s_sent, E_sent in tqdm(
            zip(natst_sent_words, natst_sent_rts, surprisals_list, embeddings_list),
            total=len(natst_sent_words), desc=f"{tag} windows", leave=False):
            if s_sent is None or E_sent is None or len(words) < 4:
                continue
            logf_sent = [get_f(t) for t in words[1:]]
            len_sent  = [get_l(t) for t in words[1:]]
            Xb, Xe, yy = build_sentence_windows(s_sent, logf_sent, len_sent, E_sent, rt_sent, words)
            if Xb.size:
                Xb_list.append(Xb); Xe_list.append(Xe); y_list.append(yy)

        if not Xb_list:
            raise RuntimeError(f"No usable samples for {tag}")

        X_base = np.vstack(Xb_list)
        X_emb  = np.vstack(Xe_list)
        y_nat  = np.concatenate(y_list)
        (X_base, X_emb), y_nat = drop_nonfinite_rows([X_base, X_emb], y_nat)
        save_feature_cache(tag, X_base, X_emb, y_nat)

    sanity_banner("X_base", X_base)
    sanity_banner("X_emb",  X_emb)
    sanity_banner("y_nat",  y_nat)
    quick_corr_checks(X_base, y_nat, label=tag)
    for i, c in enumerate(["s_i","logf_i","len_i"]):
        dm = decile_means(X_base[:, i], y_nat, q=10)
        print(f"deciles rt|{c}: " + " ".join(f"{m:.1f}" for m in dm))

    print(f"[{tag}] samples: {X_base.shape[0]} | dim_emb: {X_emb.shape[1]} | y: {y_nat.shape[0]}")

    m_base, sd_base = test_model_Ridge(X_base,                 y_nat, 10, f"natstor_{tag}_base")
    m_act,  sd_act  = test_model_Ridge(X_emb,                  y_nat, 10, f"natstor_{tag}")
    m_all,  sd_all  = test_model_Ridge(np.hstack([X_base,X_emb]), y_nat, 10, f"natstor_{tag}_all")
    print(f"[{tag}] act={m_act:.4f}±{sd_act:.4f} | base={m_base:.4f}±{sd_base:.4f} | all={m_all:.4f}±{sd_all:.4f}")

    results_rows.append([tag, m_act, sd_act, m_base, sd_base, m_all, sd_all])

df_natstor = pd.DataFrame(results_rows, columns=[
    "model_tag", "act_natstor", "act_natstor_sd",
    "base_natstor", "base_natstor_sd",
    "all_natstor", "all_natstor_sd"
])

p_base_vs_act = []
p_all_vs_base = []
p_all_vs_act  = []
for tag in df_natstor["model_tag"]:
    p_base_vs_act.append(significance_testing(f"natstor_{tag}_base", f"natstor_{tag}"))
    p_all_vs_base.append(significance_testing(f"natstor_{tag}_all",  f"natstor_{tag}_base"))
    p_all_vs_act.append(significance_testing(f"natstor_{tag}_all",   f"natstor_{tag}"))

df_natstor["p_base_vs_act"] = p_base_vs_act
df_natstor["p_all_vs_base"] = p_all_vs_base
df_natstor["p_all_vs_act"]  = p_all_vs_act

df_natstor.to_csv(RESULTS_CSV, index=False)
print("\n=== FINAL (NatStor) ===")
print(df_natstor)

base_plot(df_natstor, "act_natstor", "base_natstor", "all_natstor")
