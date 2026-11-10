"""Split specifications: which parameters each site keeps local.

A split is a set of blocks kept local, plus an optional flag that keeps every
normalization layer local regardless of block. Everything else is federated.
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import BLOCKS, block_of, is_norm_param

_ENC = ("stem", "enc1", "enc2", "enc3", "bottleneck")
_DEC = ("dec1", "dec2", "dec3", "dec4", "head")


@dataclass(frozen=True)
class Split:
    name: str
    local_blocks: frozenset[str] = frozenset()
    local_norm: bool = False

    def __post_init__(self):
        unknown = set(self.local_blocks) - set(BLOCKS)
        if unknown:
            raise ValueError(f"unknown blocks: {sorted(unknown)}")

    def is_local(self, param_name: str) -> bool:
        if self.local_norm and is_norm_param(param_name):
            return True
        return block_of(param_name) in self.local_blocks

    def shared_names(self, param_names) -> frozenset[str]:
        return frozenset(n for n in param_names if not self.is_local(n))

    def local_names(self, param_names) -> frozenset[str]:
        return frozenset(n for n in param_names if self.is_local(n))


def _mk(name, blocks=(), norm=False):
    return Split(name, frozenset(blocks), norm)


def all_splits() -> dict[str, Split]:
    """Every split in the study grid, keyed by name."""
    s = [
        _mk("fedavg"),                                   # everything shared
        _mk("local", BLOCKS),                            # nothing shared
        _mk("norm_local", norm=True),                    # FedBN-style
        _mk("bottleneck_local", ("bottleneck",)),
        # Non-prefix shapes.
        _mk("stem_head_local", ("stem", "head")),        # shared middle
        _mk("stem_norm_local", ("stem",), norm=True),
        _mk("encoder_local", _ENC),                      # shared decoder
        _mk("decoder_local", _DEC),                      # shared encoder
    ]
    # Input side local, growing deeper (LG-FedAvg style).
    for k in range(1, len(_ENC) + 1):
        s.append(_mk(f"in_local_{k}", _ENC[:k]))
    # Output side local, growing deeper (FedPer style). out_local_1 is the head only.
    for k in range(1, len(_DEC) + 1):
        s.append(_mk(f"out_local_{k}", _DEC[len(_DEC) - k:]))
    return {x.name: x for x in s}
