"""Federated training with partial sharing, then per-site testing.

One run = one split x one heterogeneity setting x one fold. Each site ends with
its own model: the shared blocks from the server plus its own local blocks,
taken from the round where its validation Dice was best. Optionally each site
then fine-tunes the whole model locally (the "+ft" baseline).

    python -m pflseg.train --data_dir data/fets2022_prep \
        --partition partitions/fets2022/partitioning_1.csv \
        --split decoder_local --het site --fold 0

Outputs in runs/<run name>/: config.json, train_log.csv, val_log.csv,
test_cases.csv (per case, per region Dice and HD95) and summary.json.
Interrupted runs resume from runs/<run name>/ckpt.pt.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from monai.losses import DiceLoss
from torch.utils.data import DataLoader

from .data import (MODALITIES, REGIONS, PatchDataset, RoundBatches, VolumeCache,
                   assign_modalities, augment, cap_training, channel_mask,
                   labels_to_regions, load_index, make_clients, split_folds)
from .evaluate import dice_scores, hd95_scores, predict_regions
from .model import build_unet
from .splits import all_splits


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--split", required=True, choices=sorted(all_splits()))
    ap.add_argument("--het", default="site", choices=["site", "modality", "site_modality"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="training seed")
    ap.add_argument("--data_seed", type=int, default=0,
                    help="seed for folds, random partitions and modality assignment")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lr_decay", type=float, default=0.995, help="per round")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=12.0)
    ap.add_argument("--val_every", type=int, default=10)
    ap.add_argument("--val_overlap", type=float, default=0.25)
    ap.add_argument("--test_overlap", type=float, default=0.5)
    ap.add_argument("--finetune_epochs", type=int, default=0)
    ap.add_argument("--finetune_lr", type=float, default=1e-3)
    ap.add_argument("--client_cap", type=int, default=0,
                    help="cap on training cases per site (0 = no cap)")
    ap.add_argument("--min_cases", type=int, default=0,
                    help="drop sites with fewer cases than this")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", choices=["bf16", "off"], default="bf16")
    ap.add_argument("--no_hd95", action="store_true")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="auto")
    return ap.parse_args(argv)


LOCK_STALE = 1800  # seconds


def run_name(a) -> str:
    m = a.split + (f"+ft{a.finetune_epochs}" if a.finetune_epochs else "")
    cap = f"__cap{a.client_cap}" if a.client_cap else ""
    tag = f"__{a.tag}" if a.tag else ""
    return f"{a.het}__{m}{cap}__f{a.fold}__s{a.seed}{tag}"


def _append(path: Path, row: dict):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main(argv=None):
    a = parse_args(argv)
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                       if a.device == "auto" else a.device)
    amp = a.amp == "bf16" and dev.type == "cuda"
    torch.backends.cudnn.benchmark = dev.type == "cuda"
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    run_dir = Path(a.out) / run_name(a)
    if (run_dir / "summary.json").exists():
        print(f"[skip] {run_dir} already finished")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Refuse to run if another job is working on this run right now. The lock is
    # refreshed every step; one untouched for LOCK_STALE seconds is from a dead job.
    lock = run_dir / "lock"
    if lock.exists() and time.time() - lock.stat().st_mtime < LOCK_STALE:
        print(f"[busy] {run_dir} is being run by another job ({lock.read_text().strip()})")
        return run_dir
    lock.write_text(f"{os.environ.get('SLURM_JOB_ID', 'local')} {os.uname().nodename}\n")

    def touch():
        os.utime(lock)

    # ---------------------------------------------------------------- setup
    index = load_index(a.data_dir)
    clients = make_clients(a.partition, index, a.het, seed=a.data_seed)
    if a.min_cases:
        clients = {k: v for k, v in clients.items() if len(v) >= a.min_cases}
    splits = split_folds(clients, a.fold, a.n_folds, seed=a.data_seed)
    cap_training(splits, a.client_cap, seed=a.data_seed)
    splits = {k: s for k, s in splits.items() if s.train}
    names = list(splits)
    mods = assign_modalities(names, a.het, seed=a.data_seed)
    masks = {k: channel_mask(mods[k], dev) for k in names}
    weights = {k: float(len(splits[k].train)) for k in names}
    split = all_splits()[a.split]

    cfg = dict(vars(a), run_name=run_name(a), device=str(dev),
               clients={k: dict(n_train=len(s.train), n_val=len(s.val), n_test=len(s.test),
                                modalities=[MODALITIES[i] for i in mods[k]])
                        for k, s in splits.items()})
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"run {run_dir.name}: {len(names)} sites, "
          f"{sum(len(s.train) for s in splits.values())} train / "
          f"{sum(len(s.val) for s in splits.values())} val / "
          f"{sum(len(s.test) for s in splits.values())} test cases, device {dev}")

    net = build_unet().to(dev)
    pnames = [n for n, _ in net.named_parameters()]
    shared = sorted(split.shared_names(pnames))
    local = sorted(split.local_names(pnames))
    init = {n: p.detach().clone() for n, p in net.named_parameters()}
    print(f"split {a.split}: {len(shared)} shared tensors, {len(local)} local tensors")

    train_cases = sorted({c for s in splits.values() for c in s.train})
    pos = {c: i for i, c in enumerate(train_cases)}
    sampler = RoundBatches()
    loader = DataLoader(PatchDataset(a.data_dir, train_cases, a.patch), batch_sampler=sampler,
                        num_workers=a.workers, persistent_workers=a.workers > 0,
                        pin_memory=dev.type == "cuda",
                        prefetch_factor=4 if a.workers > 0 else None)
    cache = VolumeCache(a.data_dir)
    dice_loss = DiceLoss(sigmoid=True, smooth_nr=1.0, smooth_dr=1.0)
    bce = torch.nn.BCEWithLogitsLoss()

    def load_model(full: dict):
        with torch.no_grad():
            for n, p in net.named_parameters():
                p.copy_(full[n])

    def snapshot() -> dict:
        return {n: p.detach().clone() for n, p in net.named_parameters()}

    def site_model(k) -> dict:
        m = dict(global_shared)
        m.update(client_local[k])
        return m

    def epoch_plan(k) -> list[tuple[str, list[int]]]:
        idx = [pos[c] for c in splits[k].train]
        random.shuffle(idx)
        return [(k, idx[i:i + a.batch]) for i in range(0, len(idx), a.batch)]

    def run_plan(plan, lr, on_end):
        """Train through a schedule of (site, batch) in order; returns mean loss."""
        sampler.batches = [b for _, b in plan]
        owners = [k for k, _ in plan]
        losses, cur, opt = [], None, None
        net.train()
        for (x, y), k in zip(loader, owners):
            if k != cur:
                if cur is not None:
                    on_end(cur)
                cur = k
                load_model(start_of[k])
                opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=a.momentum,
                                      nesterov=a.momentum > 0, weight_decay=a.wd)
            x = x.to(dev, non_blocking=True).float()
            y = labels_to_regions(y.to(dev, non_blocking=True)).float()
            x, y = augment(x, y)
            x = x * masks[k]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                logits = net(x)
            loss = dice_loss(logits.float(), y) + bce(logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
            opt.step()
            losses.append(loss.item())
            touch()
        if cur is not None:
            on_end(cur)
        return float(np.mean(losses)) if losses else float("nan")

    def evaluate_cases(k, model: dict, cases, overlap, with_hd):
        load_model(model)
        rows = []
        for c in cases:
            touch()
            img, seg = cache.get(c)
            x = torch.from_numpy(img).to(dev).float()[None] * masks[k]
            gt = labels_to_regions(torch.from_numpy(seg).to(dev)[None])[0]
            pred = predict_regions(net, x, a.patch, overlap, amp)
            d = dice_scores(pred, gt)
            h = hd95_scores(pred, gt) if with_hd else [float("nan")] * 3
            rows.append((c, d, h))
        return rows

    def val_score(k, model) -> float | None:
        if not splits[k].val:
            return None
        rows = evaluate_cases(k, model, splits[k].val, a.val_overlap, False)
        return float(np.mean([np.mean(d) for _, d, _ in rows]))

    # ---------------------------------------------------------------- state
    global_shared = {n: init[n].clone() for n in shared}
    client_local = {k: {n: init[n].clone() for n in local} for k in names}
    best = {k: None for k in names}
    best_score = {k: -1.0 for k in names}
    best_round = {k: -1 for k in names}
    start_round = 0
    ckpt = run_dir / "ckpt.pt"
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        start_round = st["round"]
        global_shared = {n: t.to(dev) for n, t in st["global_shared"].items()}
        client_local = {k: {n: t.to(dev) for n, t in v.items()} for k, v in st["client_local"].items()}
        best, best_score, best_round = st["best"], st["best_score"], st["best_round"]
        print(f"resumed from round {start_round}")

    def save_ckpt(r):
        st = dict(round=r,
                  global_shared={n: t.cpu() for n, t in global_shared.items()},
                  client_local={k: {n: t.cpu() for n, t in v.items()} for k, v in client_local.items()},
                  best=best, best_score=best_score, best_round=best_round)
        tmp = ckpt.with_suffix(".tmp")
        torch.save(st, tmp)
        os.replace(tmp, ckpt)

    # ---------------------------------------------------------------- federated rounds
    W = sum(weights.values())
    t0 = time.time()
    for r in range(start_round, a.rounds):
        tr = time.time()
        lr = a.lr * a.lr_decay ** r
        start_of = {k: site_model(k) for k in names}
        acc = {n: torch.zeros_like(t) for n, t in global_shared.items()}

        def end_fl(k):
            p = {n: q.detach() for n, q in net.named_parameters()}
            for n in shared:
                acc[n].add_(p[n], alpha=weights[k] / W)
            client_local[k] = {n: p[n].clone() for n in local}

        plan = [b for k in names for b in epoch_plan(k)]
        loss = run_plan(plan, lr, end_fl)
        if shared:
            global_shared = acc

        last = r == a.rounds - 1
        if (r + 1) % a.val_every == 0 or last:
            for k in names:
                m = site_model(k)
                s = val_score(k, m)
                score = -0.5 if s is None else s  # no val cases: keep the latest model
                if score >= best_score[k]:
                    best_score[k], best_round[k] = score, r + 1
                    best[k] = {n: t.cpu().clone() for n, t in m.items()}
                _append(run_dir / "val_log.csv",
                        dict(round=r + 1, site=k, val_dice=s if s is not None else ""))
            save_ckpt(r + 1)
        el = time.time() - tr
        _append(run_dir / "train_log.csv", dict(round=r + 1, lr=lr, loss=loss, seconds=round(el, 1)))
        done = r + 1 - start_round
        eta = (time.time() - t0) / done * (a.rounds - r - 1) / 3600
        print(f"round {r + 1}/{a.rounds} loss {loss:.4f} {el:.0f}s  eta {eta:.1f}h", flush=True)

    # ---------------------------------------------------------------- local fine-tuning
    finetuned = {}
    if a.finetune_epochs:
        for k in names:
            m0 = {n: t.to(dev) for n, t in best[k].items()}
            s0 = val_score(k, m0)
            if s0 is None:
                continue  # nothing to select with: keep the federated model
            cur_best, cur_score, cur_ep = best[k], s0, 0
            state = {"m": m0}
            for ep in range(1, a.finetune_epochs + 1):
                start_of = {k: state["m"]}

                def end_ft(kk):
                    state["m"] = snapshot()

                run_plan(epoch_plan(k), a.finetune_lr, end_ft)
                s = val_score(k, state["m"])
                _append(run_dir / "finetune_log.csv", dict(site=k, epoch=ep, val_dice=s))
                if s > cur_score:
                    cur_best = {n: t.cpu().clone() for n, t in state["m"].items()}
                    cur_score, cur_ep = s, ep
            best[k], finetuned[k] = cur_best, cur_ep
            print(f"finetune {k}: best epoch {cur_ep}, val {cur_score:.4f} (start {s0:.4f})")

    # ---------------------------------------------------------------- test
    test_path = run_dir / "test_cases.csv"
    if test_path.exists():
        test_path.unlink()
    all_rows = []
    for k in names:
        if not splits[k].test:
            continue
        m = {n: t.to(dev) for n, t in best[k].items()}
        for c, d, h in evaluate_cases(k, m, splits[k].test, a.test_overlap, not a.no_hd95):
            row = dict(case=c, site=k, het=a.het, split=a.split,
                       method=a.split + ("+ft" if a.finetune_epochs else ""),
                       fold=a.fold, seed=a.seed, n_train_site=len(splits[k].train),
                       modalities="+".join(MODALITIES[i] for i in mods[k]),
                       best_round=best_round[k], finetune_epoch=finetuned.get(k, 0))
            for reg, v in zip(REGIONS, d):
                row[f"dice_{reg}"] = v
            for reg, v in zip(REGIONS, h):
                row[f"hd95_{reg}"] = v
            row["dice_mean"] = float(np.mean(d))
            _append(test_path, row)
            all_rows.append(row)

    summary = dict(run_name=run_dir.name, n_test_cases=len(all_rows),
                   hours=round((time.time() - t0) / 3600, 2),
                   **{f"dice_{reg}": float(np.mean([r[f"dice_{reg}"] for r in all_rows]))
                      for reg in REGIONS},
                   dice_mean=float(np.mean([r["dice_mean"] for r in all_rows])))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if ckpt.exists():
        ckpt.unlink()
    lock.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))
    return run_dir


if __name__ == "__main__":
    main()
