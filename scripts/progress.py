"""How far a grid has got: finished, running (with current round), waiting, and an ETA.

    python scripts/progress.py                                 # round 2, from runs/
    python scripts/progress.py --grid experiments/pilot.txt
    python scripts/progress.py --runs results/runs             # from what reached GitHub
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pflseg.train import parse_args, run_name  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="experiments/round2.txt")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--parallel", type=int, default=3, help="jobs running at once, for the ETA")
    a = ap.parse_args()

    grid = [line.split() for line in open(a.grid) if line.strip()]
    names = [run_name(parse_args(["--data_dir", "-", "--partition", "-"] + g)) for g in grid]
    done, running, waiting, hours = [], [], [], []
    for n in names:
        d = Path(a.runs) / n
        if (d / "summary.json").exists():
            done.append(n)
            hours.append(json.loads((d / "summary.json").read_text())["hours"])
        elif (d / "train_log.csv").exists():
            rows = list(csv.DictReader(open(d / "train_log.csv")))
            running.append((n, int(rows[-1]["round"]), float(rows[-1]["seconds"])) if rows else (n, 0, 0))
        else:
            waiting.append(n)

    total = len(names)
    print(f"{a.grid}: {len(done)}/{total} finished ({100 * len(done) / total:.0f}%), "
          f"{len(running)} started, {len(waiting)} waiting")
    for n, r, s in running:
        print(f"  running  {n}: round {r}/300, {s:.0f} s/round")
    per_run = sum(hours) / len(hours) if hours else 2.1
    left = len(waiting) + sum(1 - r / 300 for _, r, _ in running)
    print(f"about {per_run:.1f} h per run; {left:.1f} runs left -> "
          f"about {left * per_run / a.parallel:.0f} h at {a.parallel} at a time")


if __name__ == "__main__":
    main()
