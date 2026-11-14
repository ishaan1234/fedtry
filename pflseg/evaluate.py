"""Whole-volume prediction and the BraTS metrics: Dice and 95% Hausdorff distance."""
from __future__ import annotations

import math
import warnings

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance

HD95_EMPTY_PRED = 373.13  # BraTS convention when a region is missed entirely


@torch.no_grad()
def predict_regions(net, img: torch.Tensor, patch: int, overlap: float,
                    amp: bool, sw_batch: int = 4) -> torch.Tensor:
    """img (1, 4, D, H, W) on device -> bool (3, D, H, W): WT, TC, ET."""
    was_training = net.training
    net.eval()
    with torch.autocast(img.device.type, dtype=torch.bfloat16, enabled=amp):
        logits = sliding_window_inference(img, (patch,) * 3, sw_batch, net,
                                          overlap=overlap, mode="gaussian")
    if was_training:
        net.train()
    return logits[0].float() > 0


def dice_scores(pred: torch.Tensor, gt: torch.Tensor) -> list[float]:
    """Per region; an empty ground truth scores 1 if the prediction is also empty."""
    out = []
    for c in range(pred.shape[0]):
        p, g = pred[c], gt[c]
        ps, gs = int(p.sum()), int(g.sum())
        if gs == 0:
            out.append(1.0 if ps == 0 else 0.0)
        else:
            out.append(2.0 * int((p & g).sum()) / (ps + gs))
    return out


def hd95_scores(pred: torch.Tensor, gt: torch.Tensor) -> list[float]:
    """Per region in voxels (= mm for BraTS); NaN when the ground truth is empty."""
    out = []
    for c in range(pred.shape[0]):
        p, g = pred[c].cpu(), gt[c].cpu()
        if not bool(g.any()):
            out.append(math.nan)
        elif not bool(p.any()):
            out.append(HD95_EMPTY_PRED)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = compute_hausdorff_distance(p[None, None].float(), g[None, None].float(),
                                               include_background=True, percentile=95)
            out.append(float(d.item()))
    return out


def mean_dice(rows) -> float:
    return float(np.mean([np.mean(r) for r in rows])) if rows else float("nan")
