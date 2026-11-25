"""The paper's key statistical tests, at hospital level (each of 23 sites weighted equally).

Uses methods that ran on all five folds. Writes results/<out>/key_tests.md.
    python scripts/key_tests.py --cases results/final/all_cases.csv --out results/final
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

FOLDS = [0, 1, 2, 3, 4]
SIZES = [0, 10, 30, 10**6]
SIZE_LABELS = ["<=10", "11-30", ">30"]


def holm(p: dict) -> dict:
    keys, out, run = sorted(p, key=p.get), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * p[k]))
        out[k] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="results/final/all_cases.csv")
    ap.add_argument("--out", default="results/final")
    a = ap.parse_args()
    df = pd.read_csv(a.cases)
    L = []

    def sm(het, m):
        x = df[(df.het == het) & (df.method == m) & df.fold.isin(FOLDS)]
        if x.fold.nunique() != len(FOLDS):
            raise SystemExit(f"{het}/{m} is missing folds")
        return x.groupby("site").agg(d=("dice_mean", "mean"), n=("n_train_site", "mean"))

    def paired(a_, b_):
        j = a_.join(b_, rsuffix="_b").dropna()
        d = j.d - j.d_b
        return d.mean(), wilcoxon(j.d, j.d_b).pvalue, int((d > 0).sum()), int((d < 0).sum())

    L += ["# Key tests (hospital level, 5 folds)", "",
          "Each hospital's Dice is the mean over all its test cases across the five folds; "
          "tests are paired two-sided Wilcoxon over the 23 hospitals.", ""]

    # 1. Interaction: head vs stem across settings
    L += ["## 1. Local head minus local stem, by setting", "",
          "| Setting | Gap (Dice) | Hospitals favouring head | p (gap = 0) |", "| --- | --: | --: | --: |"]
    gaps = {}
    for het in ["site", "modality", "site_modality"]:
        g = sm(het, "out_local_1").d - sm(het, "in_local_1").d
        gaps[het] = g
        L.append(f"| {het} | {g.mean():+.4f} | {int((g > 0).sum())}/23 | {wilcoxon(g).pvalue:.2g} |")
    ch = gaps["site_modality"] - gaps["site"]
    L += ["", f"Change in gap from real hospitals to the same hospitals with missing sequences: "
          f"{ch.mean():+.4f} (p = {wilcoxon(gaps['site'], gaps['site_modality']).pvalue:.2g}).", ""]
    gs = sm("site", "out_local_1+ft").d - sm("site", "in_local_1+ft").d
    gm = sm("site_modality", "out_local_1+ft").d - sm("site_modality", "in_local_1+ft").d
    L += [f"With fine-tuning: {gs.mean():+.4f} ({int((gs > 0).sum())}/23) vs {gm.mean():+.4f} "
          f"({int((gm > 0).sum())}/23), change p = {wilcoxon(gs, gm).pvalue:.2g}.", ""]

    # 2. split + fine-tuning vs fine-tuning
    res = {(h, m): paired(sm(h, m), sm(h, "fedavg+ft"))
           for h in ["site", "modality", "site_modality"]
           for m in ["norm_local+ft", "in_local_1+ft", "out_local_1+ft"]}
    adj = holm({k: v[1] for k, v in res.items()})
    L += ["## 2. Split + fine-tuning vs fine-tuning alone", "",
          "| Setting | Method | Δ Dice | Better / worse | p | Holm p (9 tests) |",
          "| --- | --- | --: | --: | --: | --: |"]
    for (h, m), (d, p, b, w) in res.items():
        L.append(f"| {h} | `{m}` | {d:+.4f} | {b} / {w} | {p:.2g} | {adj[(h, m)]:.2g} |")
    j = sm("site", "out_local_1+ft").join(sm("site", "fedavg+ft"), rsuffix="_r")
    j["delta"] = j.d - j.d_r
    j["size"] = pd.cut(j.n, SIZES, labels=SIZE_LABELS)
    L += ["", "`out_local_1+ft` minus `fedavg+ft`, real hospitals, by training cases per hospital:", "",
          "| Hospital size | Hospitals | Mean Δ | Improved |", "| --- | --: | --: | --: |"]
    for s, g in j.groupby("size", observed=True):
        L.append(f"| {s} | {len(g)} | {g.delta.mean():+.4f} | {int((g.delta > 0).sum())} |")
    L.append("")

    # 3. Single-sequence hospitals
    m2 = df[df.het.isin(["modality", "site_modality"]) & df.fold.isin(FOLDS)].copy()
    m2["n_mod"] = m2.modalities.str.count(r"\+") + 1
    one = m2[m2.n_mod == 1]
    L += ["## 3. Cases from hospitals holding one MRI sequence", "",
          "| Setting | Method | vs fedavg | p | vs fedavg+ft | p | Cases |",
          "| --- | --- | --: | --: | --: | --: | --: |"]
    for h in ["modality", "site_modality"]:
        base = one[(one.het == h) & (one.method == "fedavg")].set_index(["case", "fold"]).dice_mean
        ft = one[(one.het == h) & (one.method == "fedavg+ft")].set_index(["case", "fold"]).dice_mean
        for m in ["norm_local", "in_local_1", "norm_local+ft", "in_local_1+ft"]:
            x = one[(one.het == h) & (one.method == m)].set_index(["case", "fold"]).dice_mean
            k = pd.concat([x, base, ft], axis=1, join="inner")
            k.columns = ["x", "b", "f"]
            L.append(f"| {h} | `{m}` | {k.x.mean() - k.b.mean():+.3f} | {wilcoxon(k.x, k.b).pvalue:.2g} "
                     f"| {k.x.mean() - k.f.mean():+.3f} | {wilcoxon(k.x, k.f).pvalue:.2g} | {len(k)} |")
    L.append("")

    # 4. Cost of personalizing large blocks, by hospital size
    L += ["## 4. Δ Dice vs fedavg+ft by hospital size (≤10 / 11–30 / >30 training cases)", "",
          "| Setting | " + " | ".join(["norm", "head", "stem", "encoder", "decoder", "local"]) + " |",
          "| --- |" + " --: |" * 6]
    for h in ["site", "modality", "site_modality"]:
        r = sm(h, "fedavg+ft")
        cells = []
        for m in ["norm_local", "out_local_1", "in_local_1", "encoder_local", "decoder_local", "local"]:
            x = sm(h, m).join(r, rsuffix="_r")
            x["size"] = pd.cut(x.n, SIZES, labels=SIZE_LABELS)
            v = x.groupby("size", observed=True).apply(lambda s: (s.d - s.d_r).mean())
            cells.append(" / ".join(f"{t:+.3f}" for t in v))
        L.append(f"| {h} | " + " | ".join(cells) + " |")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "key_tests.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
