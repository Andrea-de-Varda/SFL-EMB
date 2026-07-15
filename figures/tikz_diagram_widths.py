#!/usr/bin/env python3
"""Line widths for the SFL-vs-EMB summary diagram in figure.tex (panel A, bottom right).

Each dataset node is joined to whichever predictor wins it, with the line width
proportional to the size of the gap:

    width = |mean_over_5_models(r_SFL - r_EMB)| * K,      K = 20
    side  = SFL if (r_SFL - r_EMB) > 0 else EMB

K = 20 is not arbitrary: it is recovered exactly from the original figure (old width
/ old published delta = 20.000 +/- 0.01 across all six datasets whose deltas the
paper reported). The `*1.5` in the .tex is a separate display factor, left untouched.

Prints the \\draw lines to paste into
`6a56bd55cbb8310ac4c1e6db/emnlp2023-latex/figures/figure.tex`.

    python figures/tikz_diagram_widths.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
K = 20.0

# tikz node name -> results/<stem>_results.csv ; order as in the .tex
NODES = [
    ("provo", "provo_fpass"), ("meco", "meco_fpass"), ("zuco", "zuco_fpass"),
    ("ucl1", "ucl_et_fpass"), ("ucl2", "ucl_spr"), ("brown", "brown"),
    ("natst1", "natstor_spr"), ("natst2", "maze_maze"), ("ucln4", "ucl_n400_n400"),
    ("pereira", "pereira"), ("control", "control"), ("natst3", "natstorfmri"),
    ("hp", "HP"),
]


def delta(stem):
    d = pd.read_csv(ROOT / f"results/{stem}_results.csv")
    return float((d["sfl"] - d["emb"]).mean())


def main():
    for node, stem in NODES:
        dl = delta(stem)
        w = abs(dl) * K
        if dl > 0:   # SFL wins -> leaves the SFL node heading left
            print(f"    \\draw[line width={w!r}*1.5, color=gray!50] (SFL) to[out=180,in=0] ({node});")
        else:        # EMB wins -> leaves the EMB node heading right
            print(f"    \\draw[line width={w!r}*1.5, color=gray!50] (EMB) to[out=0,in=180] ({node});")


if __name__ == "__main__":
    main()
