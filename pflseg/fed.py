"""Federated averaging restricted to the shared part of a split."""
from __future__ import annotations

import torch


def aggregate(client_params: list[dict[str, torch.Tensor]],
              weights: list[float],
              shared_names) -> dict[str, torch.Tensor]:
    """Weighted average of the shared parameters across clients.

    Returns only the shared parameters. Local parameters never leave a client.
    """
    if len(client_params) != len(weights) or not client_params:
        raise ValueError("need one weight per client and at least one client")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    out = {}
    for name in shared_names:
        acc = torch.zeros_like(client_params[0][name], dtype=torch.float32)
        for params, w in zip(client_params, weights):
            acc += params[name].to(torch.float32) * (w / total)
        out[name] = acc.to(client_params[0][name].dtype)
    return out


def start_round(client_state: dict[str, torch.Tensor],
                global_shared: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Client parameters at the start of a round: own local part, global shared part."""
    merged = dict(client_state)
    merged.update({n: t.clone() for n, t in global_shared.items()})
    return merged
