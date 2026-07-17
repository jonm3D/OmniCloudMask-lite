"""Small compatibility patch for non-contiguous EdgeNeXt tensors on MPS."""

from __future__ import annotations

import torch


class MPSSafeConv2d(torch.nn.Conv2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.device.type == "mps" and not input.is_contiguous():
            input = input.contiguous()
        return super().forward(input)


def patch_models_for_mps(
    models: list[torch.nn.Module],
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.nn.Module]:
    if device.type != "mps":
        return models
    for model in models:
        channels = next(
            module.in_channels
            for module in model.modules()
            if isinstance(module, torch.nn.Conv2d)
        )
        try:
            with torch.no_grad():
                model(torch.zeros((1, channels, 65, 65), device=device, dtype=dtype))
        except RuntimeError as error:
            if "view size is not compatible" not in str(error):
                raise
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    module.__class__ = MPSSafeConv2d
    return models
