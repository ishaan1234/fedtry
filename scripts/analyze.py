"""Collect finished runs into the study's results tables.

For each heterogeneity setting and method (folds and seeds pooled), reports
per-case Dice, per-site Dice, HD95, and a paired Wilcoxon signed-rank test
against the reference method (fedavg+ft if present, else fedavg), Holm-
corrected within each setting. Works on partial grids.

    python scripts/analyze.py --runs runs --out results
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REG = ("WT", "TC", "ET")


def holm(p: dict[str, float]) -> dict[str, float]:
    items = sorted((v, k) for k, v in p.items() if not np.isnan(v))
    out, running, m = {k: np.nan for k in p}, 0.0, len(items)
    for i, (v, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * v))
        out[k] = running
    return out


def load(runs: str) -> pd.DataFrame:
    frames = []
    for s in glob.glob(f"{runs}/*/summary.json"):
        d = Path(s).parent
        if (d / "test_cases.csv").exists():
            frames.append(pd.read_csv(d / "test_cases.csv"))
    if not frames:
        raise SystemExit(f"no finished runs under {runs}/")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    df = load(a.runs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_cases.csv", index=False)
    key = ["case", "fold", "seed"]

    site = (df.groupby(["het", "method", "site"])
              .agg(n_test=("case", "size"), n_train=("n_train_site", "first"),
                   modalities=("modalities", "first"), dice=("dice_mean", "mean"))
              .reset_index())
    site.to_csv(out / "per_site.csv", index=False)

    lines = ["# Results", "",
             f"{df.groupby(['het', 'method']).ngroups} method x setting cells, "
             f"{len(df)} test predictions. Dice is the mean over WT, TC and ET unless named. "
             "HD95 is in mm. p is a paired two-sided Wilcoxon test on per-case mean Dice "
             "against the reference, Holm-corrected within each setting.", ""]
    best_rows = []
    for het, g in df.groupby("het"):
        methods = sorted(g["method"].unique())
        ref = "fedavg+ft" if "fedavg+ft" in methods else ("fedavg" if "fedavg" in methods else methods[0])
        r = g[g.method == ref].set_index(key)
        rs = site[(site.het == het) & (site.method == ref)].set_index("site")["dice"]
        rows, pvals = [], {}
        for m in methods:
            x = g[g.method == m]
            xs = site[(site.het == het) & (site.method == m)].set_index("site")["dice"]
            paired = x.set_index(key)[["dice_mean"]].join(r[["dice_mean"]], rsuffix="_ref", how="inner")
            diff = paired["dice_mean"] - paired["dice_mean_ref"]
            p = np.nan
            if m != ref and len(paired) >= 6 and np.any(diff != 0):
                p = float(wilcoxon(paired["dice_mean"], paired["dice_mean_ref"]).pvalue)
            pvals[m] = p
            common = xs.index.intersection(rs.index)
            rows.append(dict(
                method=m, cases=len(x),
                dice=x["dice_mean"].mean(),
                **{f"dice_{k}": x[f"dice_{k}"].mean() for k in REG},
                hd95=np.nanmean(x[[f"hd95_{k}" for k in REG]].to_numpy()) if f"hd95_WT" in x else np.nan,
                site_mean=xs.mean(), worst_site=xs.min(),
                delta=diff.mean() if len(paired) else np.nan,
                better=int((xs[common] > rs[common]).sum()),
                worse=int((xs[common] < rs[common]).sum()), n_sites=len(common)))
        adj = holm(pvals)
        t = pd.DataFrame(rows)
        t["p_holm"] = t["method"].map(adj)
        t = t.sort_values("dice", ascending=False)
        t.to_csv(out / f"table_{het}.csv", index=False)
        best_rows.append(dict(het=het, best=t.iloc[0]["method"], dice=t.iloc[0]["dice"], ref=ref))

        lines += [f"## {het}", "", f"Reference: `{ref}`.", "",
                  "| Method | Cases | Dice | WT | TC | ET | HD95 | Site mean | Worst site "
                  "| Δ vs ref | Sites better / worse | p (Holm) |",
                  "| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |"]
        for _, x in t.iterrows():
            pv = "" if np.isnan(x.p_holm) else (f"{x.p_holm:.1e}" if x.p_holm < 1e-3 else f"{x.p_holm:.3f}")
            dl = "" if x.method == ref or np.isnan(x.delta) else f"{x.delta:+.4f}"
            bw = "" if x.method == ref else f"{x.better} / {x.worse} of {x.n_sites}"
            lines.append(f"| `{x.method}` | {x.cases} | {x.dice:.4f} | {x.dice_WT:.4f} | {x.dice_TC:.4f} "
                         f"| {x.dice_ET:.4f} | {x.hd95:.2f} | {x.site_mean:.4f} | {x.worst_site:.4f} "
                         f"| {dl} | {bw} | {pv} |")
        lines.append("")

    lines += ["## Best method per setting", "", "| Setting | Best | Dice | Reference |",
              "| --- | --- | --: | --- |"]
    lines += [f"| {b['het']} | `{b['best']}` | {b['dice']:.4f} | `{b['ref']}` |" for b in best_rows]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}/summary.md, table_*.csv, per_site.csv, all_cases.csv")


if __name__ == "__main__":
    main()
