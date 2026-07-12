#!/usr/bin/env python3
"""
Build passage labels for Pereira2018 and attach them to pereira_averaged.csv, so
the SFL-vs-EMB analysis (probe_Pereira_passages.py) can hold out WHOLE passages in
cross-validation instead of letting sentences from one passage leak across folds.

Reads only the small string columns of Pereira.parquet.gzip (Experiment, Stim,
Sentence), de-duplicated at the Arrow level per row group, so the 1.7 GB / 339M-row
file is never materialised. Stim is "<passage>_sent<k>" (e.g. "Banana_sent1" in
Experiment 2, "passage3_sent1" in Experiment 3); passage_group = "<Experiment>|<passage>"
keeps passages from the two experiments in separate groups.

Output: pereira_1/pereira_averaged_with_passage.csv (original columns + experiment,
passage_index, passage_group). Matching to pereira_averaged.csv is on normalised
sentence text (strip surrounding quotes/whitespace, collapse whitespace, lowercase).
"""
import re
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

HERE = Path(__file__).resolve().parent
PARQUET = HERE / "Pereira.parquet.gzip"
AVG = HERE / "pereira_averaged.csv"
OUT = HERE / "pereira_averaged_with_passage.csv"

def norm(s: str) -> str:
    s = str(s).strip()
    while len(s) >= 2 and s[0] in "\"'" and s[-1] in "\"'":
        s = s[1:-1].strip()
    return re.sub(r"\s+", " ", s).strip().lower()

def main():
    pf = pq.ParquetFile(PARQUET)
    stim_map = {}  # stim -> (experiment, raw_sentence); distinct via Arrow group_by
    for i in range(pf.num_row_groups):
        tbl = pf.read_row_group(i, columns=["Experiment", "Stim", "Sentence"])
        d = tbl.group_by(["Experiment", "Stim", "Sentence"]).aggregate([]).to_pandas()
        for e, st, se in zip(d["Experiment"], d["Stim"], d["Sentence"]):
            stim_map.setdefault(st, (e, se))
    print(f"{len(stim_map)} unique Stim")

    pmap = pd.DataFrame([
        {"experiment": e, "passage_index": st.split("_sent")[0],
         "passage_group": f"{e}|{st.split('_sent')[0]}", "norm_sent": norm(se)}
        for st, (e, se) in stim_map.items()])
    assert int(pmap.groupby("norm_sent")["passage_group"].nunique().gt(1).sum()) == 0, \
        "a sentence maps to >1 passage_group"

    avg = pd.read_csv(AVG)
    avg["norm_sent"] = avg["Sentence"].map(norm)
    key = pmap.drop_duplicates("norm_sent").set_index("norm_sent")[
        ["experiment", "passage_index", "passage_group"]]
    merged = avg.join(key, on="norm_sent")
    matched = int(merged["passage_group"].notna().sum())
    print(f"matched {matched}/{len(avg)} sentences; "
          f"{merged['passage_group'].nunique()} passages")
    unmatched = merged[merged["passage_group"].isna()]["Sentence"].tolist()
    for s in unmatched:
        print("  UNMATCHED:", repr(s))
    if matched != len(avg):
        raise SystemExit("unmatched sentences remain; fix normalisation before writing")
    merged.drop(columns=["norm_sent"]).to_csv(OUT, index=False)
    print(f"WROTE {OUT}")

if __name__ == "__main__":
    main()
