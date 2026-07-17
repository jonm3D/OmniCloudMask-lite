from __future__ import annotations

import numpy as np
import pytest
import torch

from omnicloudmask_lite.inference import predict


class ConstantModel(torch.nn.Module):
    def __init__(self, cloud_class: int) -> None:
        super().__init__()
        self.cloud_class = cloud_class

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(
            (input.shape[0], 4, input.shape[2], input.shape[3]),
            dtype=input.dtype,
            device=input.device,
        )
        logits[:, self.cloud_class] = 1
        return logits


def test_predict_respects_explicit_validity() -> None:
    rgn = np.arange(3 * 64 * 72, dtype=np.float32).reshape(3, 64, 72)
    valid = np.ones((64, 72), dtype=np.uint8)
    valid[:8, :] = 0
    classes = predict(
        rgn,
        valid,
        [ConstantModel(2)],
        device=torch.device("cpu"),
        patch_size=48,
        patch_overlap=16,
        batch_size=2,
    )
    assert classes.dtype == np.uint8
    assert np.all(classes[:8] == 255)
    assert np.all(classes[8:] == 2)


def test_predict_does_not_treat_zero_reflectance_as_invalid() -> None:
    rgn = np.zeros((3, 32, 32), dtype=np.float32)
    valid = np.ones((32, 32), dtype=np.uint8)
    classes = predict(
        rgn, valid, [ConstantModel(0)], device=torch.device("cpu"), patch_size=32
    )
    assert np.all(classes == 0)


def test_predict_rejects_nonfinite_valid_values() -> None:
    rgn = np.ones((3, 32, 32), dtype=np.float32)
    rgn[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        predict(
            rgn,
            np.ones((32, 32), dtype=np.uint8),
            [ConstantModel(0)],
            device=torch.device("cpu"),
            patch_size=32,
        )
