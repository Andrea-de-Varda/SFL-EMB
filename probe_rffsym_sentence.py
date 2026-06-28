#!/usr/bin/env python3
"""
SYMMETRIC RFF comparison for the sentence/word brain datasets (Pereira, Tuckute,
UCL-N400). Routes BOTH EMB and SFL through the identical pipeline
  per-item -> standardize(train) -> RBFSampler(D, gamma) -> pool-per-unit -> ridge
with gamma nested-CV-tuned independently for each side (so EMB gets the same tunable
knob SFL has). Predictors per model:
  base     SFL(3) linear                  act      EMB(D) linear (the originals)
  sfl_rff  SFL  -> RFF(D) -> pool          emb_rff  EMB -> RFF(D) -> pool
  all_rff  concat(emb_rff, sfl_rff)
Results go to results_rffsym/ (TEMP -- nothing in the main results/ is touched).

Usage (neurox env):  python probe_rffsym_sentence.py {pereira|control|n400}
"""
import sys, itertools
from os import chdir, path, makedirs
from pathlib import Path as _Path
import numpy as np
import pandas as pd

_here = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
PROJECT_ROOT = next(p for p in (_here, *_here.parents) if (p / "utils.py").is_file())
OUTDIR = str(PROJECT_ROOT / "results_rffsym")
ROW_INDEX = ["GPT", "GPT-2$_{124M}$", "GPT-2$_{355M}$", "GPT-2$_{774M}$", "GPT-2$_{1.5B}$"]
PREDS = ["base", "act", "sfl_rff", "emb_rff", "all_rff"]

def get_layer(vec, layer, dim):
    return list(vec[layer * dim:(layer + 1) * dim])

def emb_tokens(act_list, layer, dim):
    """per-TOKEN embeddings (drop first token/sentence-initial) + token->sentence index."""
    feats, idx = [], []
    for s, act_sent in enumerate(act_list):
        for v in act_sent[1:]:
            feats.append(get_layer(v, layer, dim)); idx.append(s)
    return np.array(feats), np.array(idx)

def emb_sentence_mean(act_list, layer, dim):
    return np.array([get_layer(act_sent[1:].mean(axis=0), layer, dim) for act_sent in act_list])

def load_dataset(name, R):
    """Return dict per model-key with all arrays needed. Branch per dataset."""
    import neurox.data.loader as dl
    MODELS_STD = [("gpt", "s_GPT", 12, 768), ("small", "s_small", 12, 768),
                  ("med", "s_med", 24, 1024), ("large", "s_large", 36, 1280),
                  ("xl", "s_xl", 48, 1600)]
    if name in ("pereira", "control"):
        if name == "pereira":
            chdir(str(PROJECT_ROOT / "pereira_1"))
            meta = pd.read_csv("pereira_averaged_with_passage.csv")
            y = meta["EffectSize"].to_numpy(float); groups = meta["passage_group"].to_numpy()
            actf = {"gpt": "activations_gpt.json", "small": "activations_gpt2_small.json",
                    "med": "activations_gpt2_medium.json", "large": "activations_gpt2_large.json",
                    "xl": "activations_gpt2_xl.json"}
        else:
            chdir(str(PROJECT_ROOT / "control_fmri"))
            control = pd.read_csv("data/brain-lang-data_participant_20230728.csv")
            dc = control[control["roi"] == "lang_LH_netw"].groupby("sentence").agg(
                {"response_target": "mean", "cond": "max"})
            y = dc["response_target"].to_numpy(float); groups = None
            actf = {"gpt": "activations_gpt.json", "small": "activations_gpt2_small.json",
                    "med": "activations_gpt2_medium.json", "large": "activations_gpt2_large.json",
                    "xl": "activations_gpt2_xl.json"}
        df = pd.read_csv("SFL.csv"); n = len(y)
        sfl_idx = df["sentnum"].to_numpy(int)
        def pool_sfl(col):
            return np.array([list(df[df.sentnum == s][[col, "len", "freq"]].mean()) for s in range(n)])
        out = {}
        for key, scol, layer, dim in MODELS_STD:
            act = dl.load_activations(actf[key], dim, is_brnn=False)[0]
            et, ei = emb_tokens(act, layer, dim)
            out[key] = dict(dim=dim, y=y, groups=groups, pooling="sum",
                            base=pool_sfl(scol), act=emb_sentence_mean(act, layer, dim),
                            emb_feats=et, emb_idx=ei,
                            sfl_feats=df[[scol, "len", "freq"]].to_numpy(float), sfl_idx=sfl_idx)
            del act
        return out
    elif name == "n400":
        chdir(str(PROJECT_ROOT))
        MODELS = [("gpt", "s_GPT", 12, 768), ("small", "s_GPT2_small", 12, 768),
                  ("med", "s_GPT2_med", 24, 1024), ("large", "s_GPT2_large", 36, 1280),
                  ("xl", "s_GPT2_xl", 48, 1600)]
        actf = {"gpt": "activations_gpt.json", "small": "activations_gpt2_small.json",
                "med": "activations_gpt2_medium.json", "large": "activations_gpt2_large.json",
                "xl": "activations_gpt2_xl.json"}
        df_all = pd.read_csv("UCL_reading_all.csv")
        df_noinit = df_all.dropna(subset=["s_GPT2_small"])
        filtr = np.array([("." not in w) and ("," not in w) for w in df_noinit["word"]])
        y = np.asarray(df_noinit["N400"], float)[filtr]
        valid = ~np.isnan(y); y = y[valid]; n = len(y)
        available = set(df_all.sent_nr - 1)
        def get_act_eye(act_list, layer, dim):
            out = []
            for i in sorted(available):
                for v in act_list[i][1:]:
                    out.append(get_layer(v, layer, dim))
            return np.array(out)
        out = {}
        for key, scol, layer, dim in MODELS:
            act = dl.load_activations(actf[key], dim, is_brnn=False)[0]
            emb = get_act_eye(act, layer, dim)
            assert len(emb) == len(df_noinit), f"{len(emb)} vs {len(df_noinit)}"
            emb = emb[filtr][valid]; del act
            sfl = np.asarray(df_noinit[[scol, "logfreq", "length"]], float)[filtr][valid]
            out[key] = dict(dim=dim, y=y, groups=None, pooling="sum",
                            base=sfl, act=emb,
                            emb_feats=emb, emb_idx=np.arange(n),
                            sfl_feats=sfl, sfl_idx=np.arange(n))
        return out
    raise SystemExit("dataset must be pereira|control|n400")

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "pereira"
    mode = sys.argv[2] if len(sys.argv) > 2 else "rff"   # rff (RFF-up) | pca (PCA-down)
    sys.path.insert(0, str(PROJECT_ROOT))
    import rff_matched as R
    outdir = str(PROJECT_ROOT / ("results_rffsym" if mode == "rff" else "results_embpca"))
    makedirs(outdir, exist_ok=True); R.RESULTS_DIR = outdir
    preds = PREDS if mode == "rff" else ["base", "act", "emb_pca", "all_pca"]
    SUF = {"base": "_base", "act": "_act", "sfl_rff": "_sflrff", "emb_rff": "_embrff",
           "all_rff": "_allrff", "emb_pca": "_embpca", "all_pca": "_allpca"}
    tag = "rffsym" if mode == "rff" else "embpca"
    prefix = {"pereira": "pereira", "control": "control", "n400": "UCLn400"}[name]
    data = load_dataset(name, R)

    cols = {c: [] for c in preds}; cols.update({c + "_sd": [] for c in preds})
    keys = ["gpt", "small", "med", "large", "xl"]
    for key in keys:
        D = data[key]; dim = D["dim"]
        print(f"\n=== {name} {key} (D={dim}) [{mode}] ===", flush=True)
        m = {"base": R.score_matrix(D["base"], D["y"], f"{prefix}_{key}_base", groups=D["groups"]),
             "act": R.score_matrix(D["act"], D["y"], f"{prefix}_{key}_act", groups=D["groups"])}
        if mode == "rff":
            e, s, a = R.score_rff_pair(D["emb_feats"], D["emb_idx"], D["sfl_feats"], D["sfl_idx"],
                                       D["y"], dim, D["pooling"], groups=D["groups"],
                                       name_emb=f"{prefix}_{key}_embrff", name_sfl=f"{prefix}_{key}_sflrff",
                                       name_all=f"{prefix}_{key}_allrff")
            m["emb_rff"], m["sfl_rff"], m["all_rff"] = e, s, a
        else:
            ep, ap = R.score_pca_pair(D["act"], D["base"], D["y"], k=D["base"].shape[1],
                                      groups=D["groups"], name_emb=f"{prefix}_{key}_embpca",
                                      name_all=f"{prefix}_{key}_allpca")
            m["emb_pca"], m["all_pca"] = ep, ap
        for c in preds:
            cols[c].append(m[c][0]); cols[c + "_sd"].append(m[c][1])
        print({c: round(m[c][0], 4) for c in preds}, flush=True)

    order = [x for c in preds for x in (c, c + "_sd")]
    df = pd.DataFrame({c: cols[c] for c in order}, index=ROW_INDEX)
    for a, b in itertools.combinations(preds, 2):
        df[f"p__{a}__{b}"] = [R.significance_testing(f"{prefix}_{k}{SUF[a]}", f"{prefix}_{k}{SUF[b]}") for k in keys]
    df.to_csv(f"{outdir}/{prefix}_{tag}.csv", index=False)
    print(f"\nWROTE {outdir}/{prefix}_{tag}.csv")
    print(df[preds].round(4))

if __name__ == "__main__":
    main()
