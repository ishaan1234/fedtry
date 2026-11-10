"""The study's network and the mapping from parameter names to blocks.

The network follows Manthe et al. (Medical Image Analysis, 2024): MONAI DynUNet,
five resolution levels (32 to 512 channels), instance normalization, LeakyReLU.
One deliberate change: normalization layers carry learnable affine parameters,
so that "normalization kept local" is a testable split.
"""
from __future__ import annotations

import torch
from monai.networks.nets import DynUNet

# Blocks in order from input to output. dec1 is the deepest decoder stage.
BLOCKS = ("stem", "enc1", "enc2", "enc3", "bottleneck",
          "dec1", "dec2", "dec3", "dec4", "head")

_PREFIX_TO_BLOCK = {
    "input_block": "stem",
    "downsamples.0": "enc1",
    "downsamples.1": "enc2",
    "downsamples.2": "enc3",
    "bottleneck": "bottleneck",
    "upsamples.0": "dec1",
    "upsamples.1": "dec2",
    "upsamples.2": "dec3",
    "upsamples.3": "dec4",
    "output_block": "head",
}


def build_unet(in_channels: int = 4, out_channels: int = 3) -> DynUNet:
    return DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=[3] * 5,
        strides=[1, 2, 2, 2, 2],
        upsample_kernel_size=[2] * 4,
        filters=[32, 64, 128, 256, 512],
        norm_name=("INSTANCE", {"affine": True}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision=False,
        res_block=False,
    )


def block_of(param_name: str) -> str:
    """Block that a parameter (as named by ``named_parameters``) belongs to."""
    for prefix, block in _PREFIX_TO_BLOCK.items():
        if param_name == prefix or param_name.startswith(prefix + "."):
            return block
    raise KeyError(f"parameter {param_name!r} is not in a known block")


def is_norm_param(param_name: str) -> bool:
    """True for the affine weight and bias of a normalization layer."""
    parts = param_name.split(".")
    return len(parts) >= 2 and parts[-2].startswith("norm")


def get_params(net: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Detached copy of every parameter, keyed by its ``named_parameters`` name.

    ``state_dict`` is avoided on purpose: DynUNet registers each block a second
    time under ``skip_layers``, so its state dict lists every tensor twice.
    """
    return {n: p.detach().clone() for n, p in net.named_parameters()}


@torch.no_grad()
def set_params(net: torch.nn.Module, params: dict[str, torch.Tensor],
               only: set[str] | frozenset[str] | None = None) -> None:
    """Copy ``params`` into the network, optionally restricted to names in ``only``."""
    for n, p in net.named_parameters():
        if n in params and (only is None or n in only):
            p.copy_(params[n])
