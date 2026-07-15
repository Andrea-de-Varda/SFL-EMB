#!/usr/bin/env python3
"""Audit: every fitted number in the manuscript vs a fresh recomputation.

Re-derives each reported quantity from results/*.csv (and the appendix CSVs) and
asserts that the value actually present in emnlp2023.tex matches. This is the
"double check" pass -- it reads the .tex, so it catches transcription slips as well
as stale numbers.

Scope: Results + SI values that depend on the fitting pipeline. Data-only values
(Table 1 N / participants / reliability, Figure 2) are checked for internal
consistency instead, since the pipeline cannot change them.

    python figures/verify_paper_numbers.py

Exit status is non-zero if anything fails.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "6a56bd55cbb8310ac4c1e6db/emnlp2023-latex/emnlp2023.tex"

ET = ["provo_fpass", "meco_fpass", "zuco_fpass", "ucl_et_fpass"]
SPR = ["ucl_spr", "brown", "natstor_spr"]
MAZE = ["maze_maze"]
BRAIN = {"Pereira2018": "pereira", "Tuckute2024": "control",
         "NatStor_fMRI": "natstorfmri", "Wehbe2014": "HP"}
ERP = "ucl_n400_n400"

_cache = {}
def load(stem):
    if stem not in _cache:
        _cache[stem] = pd.read_csv(ROOT / f"results/{stem}_results.csv")
    return _cache[stem]


def grp(stems, a, b):
    v = []
    for s in stems:
        d = load(s)
        v.extend((d[a] - d[b]).tolist())
    return float(np.mean(v))


def ns(p):
    return min(p, 1 - p) > 0.025


def main():
    tex = TEX.read_text()
    checks, fails = [], []

    def chk(label, expected, present=None, tol=5e-5):
        """expected = recomputed value; assert the .tex contains it."""
        s = f"{expected:.4f}"
        found = s in tex or s.lstrip("-") in tex if present is None else present
        ok = bool(found)
        checks.append((label, s, ok))
        if not ok:
            fails.append(f"{label}: recomputed {s} NOT found in emnlp2023.tex")

    # ---- Results: behavioral ----
    chk("ET d(SFL-EMB)", grp(ET, "sfl", "emb"))
    chk("ET d(ALL-SFL)", grp(ET, "all", "sfl"))
    chk("SPR d(SFL-EMB)", grp(SPR, "sfl", "emb"))
    chk("SPR d(ALL-SFL)", grp(SPR, "all", "sfl"))
    chk("Maze d(SFL-EMB)", grp(MAZE, "sfl", "emb"))
    chk("Maze d(ALL-SFL)", grp(MAZE, "all", "sfl"))
    # ---- Results: brain ----
    for k, s in BRAIN.items():
        chk(f"{k} d(SFL-EMB)", grp([s], "sfl", "emb"))
        chk(f"{k} d(ALL-EMB)", grp([s], "all", "emb"))
    chk("N400 d(SFL-EMB)", grp([ERP], "sfl", "emb"))
    chk("N400 d(ALL-EMB)", grp([ERP], "all", "emb"))

    # ---- SI: dimensionality control (RFF) ----
    B5 = {"Pereira2018": "pereira", "Tuckute2024": "control", "NatStor_fMRI": "natstorfmri",
          "Wehbe2014": "HP", "UCL_N400": ERP}
    d_emb_rff = {k: grp([s], "emb", "sfl_rff") for k, s in B5.items()}
    pos = {k: v for k, v in d_emb_rff.items() if v > 0}
    chk("RFF mean d(EMB-SFLRFF) over the 4 where EMB wins", float(np.mean(list(pos.values()))))
    for k, v in d_emb_rff.items():
        chk(f"RFF {k} d(EMB-SFLRFF)", v)
    chk("RFF mean d(RFF+EMB - RFF) all 5",
        float(np.mean([grp([s], "sfl_rff_emb", "sfl_rff") for s in B5.values()])))
    for k, s in B5.items():
        chk(f"RFF {k} d(RFF+EMB - RFF)", grp([s], "sfl_rff_emb", "sfl_rff"))

    # ---- SI: Pereira length control (only if the manuscript still cites it) ----
    lc = ROOT / "results/pereira_length_control.csv"
    if lc.exists() and "explicit predictor" in tex:
        d = pd.read_csv(lc)
        chk("length control: length adds to SFL_RFF",
            float(d.length_sfl_rff.mean() - d.sfl_rff.mean()))
        chk("length control: length adds to EMB",
            float(d.length_emb.mean() - d.emb.mean()))
        chk("length control: length+SFL_RFF", float(d.length_sfl_rff.mean()))
        chk("length control: length+EMB", float(d.length_emb.mean()))
    else:
        checks.append(("length control (not cited in the manuscript -- skipped)", "n/a", True))

    # ---- SI: layer sweep ----
    ls = ROOT / "appendix/layer_sweep.csv"
    if ls.exists():
        df = pd.read_csv(ls)
        best = int(df[df.dataset == "UCL_ET"].set_index("layer")["emb_r"].idxmax())
        checks.append((f"layer sweep: best held-in ET layer = {best}",
                       str(best), f"layer {best}" in tex or f"layer~{best}" in tex))
        for ds in ["Provo", "Meco-en", "ZuCo-2"]:
            sub = df[(df.dataset == ds) & (df.layer == best)]
            if len(sub):
                r = sub.iloc[0]
                chk(f"layerwise {ds} SFL r", float(r.sfl_r))
                chk(f"layerwise {ds} SFL SD", float(r.sfl_sd))
                chk(f"layerwise {ds} EMB r", float(r.emb_r))
                chk(f"layerwise {ds} EMB SD", float(r.emb_sd))
        # best held-in SPR layer + the last-layer comparison
        spr = df[df.dataset == "UCL_SPR"].set_index("layer")["emb_r"]
        if len(spr):
            bl = int(spr.idxmax())
            checks.append((f"layer sweep: best held-in SPR layer = {bl}", str(bl),
                           f"layer {bl}" in tex))
            chk("layerwise UCL_SPR best-layer EMB r", float(spr.loc[bl]))
            chk("layerwise UCL_SPR last-layer EMB r", float(spr.loc[int(spr.index.max())]))
    else:
        checks.append(("layer sweep csv", "MISSING", False))
        fails.append("appendix/layer_sweep.csv missing -- run appendix/layer_sweep.py")

    # ---- SI: held-out SPR evaluation at the best held-in SPR layer ----
    ho = ROOT / "appendix/layer_sweep_spr_heldout.csv"
    if ho.exists():
        for _, r in pd.read_csv(ho).iterrows():
            chk(f"layerwise {r.dataset} SFL r", float(r.sfl_r))
            chk(f"layerwise {r.dataset} SFL SD", float(r.sfl_sd))
            chk(f"layerwise {r.dataset} EMB r", float(r.emb_r))
            chk(f"layerwise {r.dataset} EMB SD", float(r.emb_sd))
    else:
        checks.append(("held-out SPR csv", "MISSING", False))
        fails.append("appendix/layer_sweep_spr_heldout.csv missing")

    # ---- data-only: internal consistency (pipeline cannot change these) ----
    rows = re.findall(r"&\s*([\d,]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*\\\\", tex)
    rel = [float(r) for _, _, r in rows]
    ok = len(rel) == 13 and abs(np.mean(rel) - 0.51) < 0.005 and min(rel) == 0.16 and max(rel) == 0.93
    checks.append(("Table 1 reliability -> 'mean 0.51, range 0.16-0.93'",
                   f"mean={np.mean(rel):.4f}", ok))
    if not ok:
        fails.append("Table 1 reliability no longer matches the Results summary")
    fd = (TEX.parent / "fig-datasets.tex").read_text()
    pairs = re.findall(r"\$N=([\d{},]+),\\ r=(\.\d+)\$", fd)
    t1 = {n.replace(",", ""): r for n, _, r in rows}
    bad = [n for n, r in pairs
           if n.replace("{,}", "") not in t1
           or abs(float(r) - float(t1[n.replace('{,}', '')])) > 1e-9]
    checks.append((f"Figure 2 vs Table 1 ({len(pairs)} datasets)", "consistent", not bad))
    if bad:
        fails.append(f"Figure 2 disagrees with Table 1 for N = {bad}")

    w = max(len(c[0]) for c in checks)
    for label, val, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<{w}}  {val}")
    print(f"\n{sum(1 for c in checks if c[2])}/{len(checks)} checks passed")
    if fails:
        print("\nFAILURES:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("All reported numbers match a fresh recomputation.")


if __name__ == "__main__":
    main()
