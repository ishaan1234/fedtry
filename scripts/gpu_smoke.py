"""GPU smoke test: one federated round on random data, plus step timing.

Checks that the network trains on the GPU, that partial averaging works there,
and measures seconds per training step and peak memory at the study's patch
size. The timing sets how large the experiment grid can be.

    python scripts/gpu_smoke.py --patch 128 --batch 4 --steps 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pflseg.fed import aggregate, start_round  # noqa: E402
from pflseg.model import build_unet, get_params, set_params  # noqa: E402
from pflseg.splits import all_splits  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--amp", choices=["bf16", "off"], default="bf16")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}", torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    print(f"torch {torch.__version__}, patch {args.patch}^3, batch {args.batch}, amp {args.amp}")
    torch.manual_seed(0)
    use_amp = args.amp == "bf16" and dev.type == "cuda"

    # Two clients, one split, one round: proves partial averaging runs on GPU.
    split = all_splits()["decoder_local"]
    nets = [build_unet().to(dev) for _ in range(2)]
    names = [n for n, _ in nets[0].named_parameters()]
    shared = split.shared_names(names)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    x = torch.randn(args.batch, 4, args.patch, args.patch, args.patch, device=dev)
    y = (torch.rand(args.batch, 3, args.patch, args.patch, args.patch, device=dev) > 0.9).float()

    step_times = []
    for c, net in enumerate(nets):
        opt = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
        for s in range(args.steps):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            if s >= 2:  # skip warm-up steps
                step_times.append(time.perf_counter() - t)
        print(f"client {c}: final loss {loss.item():.4f}")

    params = [get_params(n) for n in nets]
    g = aggregate(params, [1.0, 1.0], shared)
    for net, p in zip(nets, params):
        set_params(net, start_round(p, g))
    after = [get_params(n) for n in nets]
    same_shared = all(torch.equal(after[0][n], after[1][n]) for n in shared)
    local = [n for n in names if n not in shared]
    differ_local = any(not torch.equal(after[0][n], after[1][n]) for n in local)
    print(f"shared blocks identical after round: {same_shared}")
    print(f"local blocks still differ: {differ_local}")

    if step_times:
        mean = sum(step_times) / len(step_times)
        print(f"seconds per training step: {mean:.3f}")
        print(f"estimated hours per 300-epoch centralized run on 1,251 cases: "
              f"{mean * 1251 / args.batch * 300 / 3600:.1f}")
    if dev.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    if not (same_shared and differ_local):
        raise SystemExit("FAIL: partial averaging did not behave as expected")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
