"""Embedding + surprisal extraction with vanilla HuggingFace transformers.

Replaces the old ``neurox`` dependency. For a whitespace-tokenized sentence it
produces, per word:
  * the final-layer (or any layer) hidden state, **mean-pooled** over the word's
    sub-word tokens  (== neurox ``aggregation="average"``);
  * word surprisal, **summed** over the word's sub-word tokens (nats).

Sub-word -> word alignment uses the fast tokenizer's ``word_ids()`` on the
full-sentence tokenization (robust to BPE merges), rather than re-tokenizing each
word. The original GPT / GPT-2 tokenizers add no special tokens, so every token
maps to a word; any ``None`` (special) positions are skipped defensively.

Convention: the sentence-initial word (index 0) has no left context, so its
surprisal is ``nan`` and downstream code drops it (matching the paper's
"sentence-initial words are dropped").

Long passages (story datasets) exceed the context window; use
:func:`Extractor.chunked_word_features` for those (overlapping windows, stitched).
"""
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import MODELS

_PUNCT = re.compile(r'[\.\,\:\-\?\!\)\(\"]')


def extract_words(model_key, sentences, drop_first=True, chunked=False,
                  cache=None, device=None, n_splits=2, overlap=50, layer="last",
                  pool="mean"):
    """Per-word embeddings + surprisal for a list of sentences, stacked.

    ``sentences`` is a list of word-lists (one per sentence/passage). Returns
    ``(emb, surp)`` with ``emb`` of shape (N, D) and ``surp`` of shape (N,),
    concatenated across sentences with the sentence-initial word dropped when
    ``drop_first`` (the paper convention; surprisal there is undefined anyway).

    ``chunked=True`` uses overlapping windows for long passages (story datasets).
    ``cache`` is an optional .npz path: if it exists it is loaded, else computed
    and saved, so the GPU extraction is decoupled from downstream fitting.
    """
    if cache is not None:
        cache = Path(cache)
        if cache.exists():
            z = np.load(cache)
            return z["emb"], z["surp"]

    ex = Extractor(model_key, device)
    embs, surps = [], []
    for words in sentences:
        words = list(words)
        if chunked:
            import math
            # size chunks from the model's real context window so sub-word tokens
            # stay comfortably under it (openai-gpt = 512 positions, gpt2 = 1024).
            # ctx//2 words -> ~1.5 tokens/word keeps us well inside the limit.
            ctx = (getattr(ex.lm.config, "n_positions", None)
                   or getattr(ex.lm.config, "max_position_embeddings", 1024))
            max_words = max(64, ctx // 2)
            ns = max(n_splits, math.ceil(len(words) / max_words))
            e = ex.chunked_word_features(words, layer=layer, n_splits=ns, overlap=overlap)
            s = ex.chunked_word_surprisal(words, n_splits=ns, overlap=overlap)
        else:
            e = ex.word_embeddings(words, layer=layer, reduce=pool)
            s = ex.word_surprisal(words)
        if drop_first:
            e, s = e[1:], s[1:]
        embs.append(e)
        surps.append(s)
    emb = np.vstack(embs)
    surp = np.concatenate(surps)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, emb=emb, surp=surp)
    return emb, surp


def extract_all_layers(model_key, sentences, drop_first=True, device=None):
    """Per-word embeddings for every layer, stacked across sentences (short-sentence
    datasets only; used by the A4 layer sweep). Returns ``{layer: (N, dim)}``."""
    ex = Extractor(model_key, device)
    per_layer = None
    for words in sentences:
        d = ex.word_embeddings_all([str(w) for w in words])
        if per_layer is None:
            per_layer = {k: [] for k in d}
        for k, v in d.items():
            per_layer[k].append(v[1:] if drop_first else v)
    return {k: np.vstack(v) for k, v in per_layer.items()}


class Extractor:
    def __init__(self, model_key, device=None):
        if model_key not in MODELS:
            raise KeyError(f"{model_key!r} not in {list(MODELS)}")
        self.key = model_key
        self.cfg = MODELS[model_key]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        add_prefix = self.cfg["sep"] == "Ġ"  # GPT-2 BPE needs a leading space
        self.tok = AutoTokenizer.from_pretrained(self.cfg["hf"], add_prefix_space=add_prefix)
        self.lm = (AutoModelForCausalLM
                   .from_pretrained(self.cfg["hf"], output_hidden_states=True)
                   .to(self.device).eval())
        self.dim = self.cfg["dim"]

    # ------------------------------------------------------------------ #
    def _encode(self, words):
        words = [str(w) for w in words]   # CSV tokens may parse as numeric
        enc = self.tok(words, is_split_into_words=True, return_tensors="pt",
                       truncation=False)
        word_ids = enc.word_ids(0)
        return enc, word_ids

    @torch.no_grad()
    def word_embeddings(self, words, layer="last", reduce="mean"):
        """(n_words, dim) sub-word-pooled hidden states for ``words`` (mean or sum)."""
        enc, word_ids = self._encode(words)
        hs = self.lm(**{k: v.to(self.device) for k, v in enc.items()}).hidden_states
        li = self.cfg["last_layer"] if layer == "last" else layer
        h = hs[li][0].cpu().numpy()  # (n_tokens, dim)
        return self._pool_by_word(h, word_ids, len(words), reduce=reduce)

    @torch.no_grad()
    def word_surprisal(self, words):
        """(n_words,) summed sub-word surprisal in nats; index 0 is nan (no context)."""
        enc, word_ids = self._encode(words)
        input_ids = enc["input_ids"].to(self.device)
        logits = self.lm(input_ids=input_ids).logits[0]              # (T, V)
        logp = torch.log_softmax(logits[:-1], dim=-1)               # predict t+1 from t
        tok_surp = (-logp[torch.arange(logits.shape[0] - 1),
                          input_ids[0, 1:]]).cpu().numpy()           # surprisal of tok 1..T-1
        # tok_surp[i] is the surprisal of token i+1; token 0 has none.
        surp_by_word = np.full(len(words), np.nan)
        acc = np.zeros(len(words))
        seen = np.zeros(len(words), dtype=bool)
        for tpos in range(1, logits.shape[0]):
            w = word_ids[tpos]
            if w is None:
                continue
            acc[w] += tok_surp[tpos - 1]
            seen[w] = True
        surp_by_word[seen] = acc[seen]
        surp_by_word[0] = np.nan  # sentence-initial word is dropped downstream
        return surp_by_word

    @torch.no_grad()
    def word_embeddings_all(self, words):
        """Per-word mean-pooled hidden states for EVERY layer, in one model pass.
        Returns ``{layer_index: (n_words, dim)}`` (for the layer sweep, A4)."""
        enc, word_ids = self._encode(words)
        hs = self.lm(**{k: v.to(self.device) for k, v in enc.items()}).hidden_states
        return {li: self._pool_by_word(hs[li][0].cpu().numpy(), word_ids, len(words))
                for li in range(len(hs))}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pool_by_word(token_mat, word_ids, n_words, reduce="mean"):
        dim = token_mat.shape[1]
        out = np.zeros((n_words, dim), dtype=np.float64)
        cnt = np.zeros(n_words, dtype=np.float64)
        for tpos, w in enumerate(word_ids):
            if w is None:
                continue
            out[w] += token_mat[tpos]
            cnt[w] += 1
        nz = cnt > 0
        if reduce == "mean":
            out[nz] /= cnt[nz, None]
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _chunks(words, n_splits, overlap):
        """Overlapping word-chunks (left context ``overlap`` words) for a long text."""
        import math

        chunk_size = math.ceil(len(words) / n_splits)
        n_splits = n_splits + math.ceil(overlap * n_splits / chunk_size)
        chunks = []
        for s in range(n_splits):
            if s == 0:
                chunks.append(words[0:chunk_size])
            elif s == n_splits - 1:
                chunks.append(words[s * chunk_size - overlap * s:])
            else:
                start = s * chunk_size - overlap * s
                chunks.append(words[start:start + chunk_size])
        return [c for c in chunks if len(c)]

    @torch.no_grad()
    def chunked_word_features(self, words, layer="last", n_splits=2, overlap=50):
        """Per-word embeddings for a long passage via overlapping windows, stitched
        keeping only the non-overlap tail of later chunks. Returns (n_words, dim).
        Mirrors ``natstor_fmri/get_natstor_embeddings.py``.
        """
        embs = [self.word_embeddings(c, layer=layer)
                for c in self._chunks(words, n_splits, overlap)]
        out = [embs[0]] + [e[overlap:] for e in embs[1:]]
        return np.vstack(out)

    @torch.no_grad()
    def chunked_word_surprisal(self, words, n_splits=2, overlap=50):
        """Summed sub-word surprisal for a long passage via overlapping windows.
        Overlap words take the earlier chunk's value (more left context). (n_words,)."""
        surps = [self.word_surprisal(c) for c in self._chunks(words, n_splits, overlap)]
        out = [surps[0]] + [s[overlap:] for s in surps[1:]]
        return np.concatenate(out)
