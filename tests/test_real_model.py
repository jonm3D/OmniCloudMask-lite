from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest
import torch

from omnicloudmask_lite.inference import predict
from omnicloudmask_lite.models import load_models


def test_pinned_v4_ensemble_regression() -> None:
    model_dir = os.environ.get("OCM_MODEL_DIR")
    if not model_dir:
        pytest.skip("set OCM_MODEL_DIR to run the pinned-weight regression")
    y, x = np.mgrid[:96, :96]
    rgn = np.stack(((x + 1) / 97, (y + 1) / 97, ((x + y) % 31) / 31)).astype(
        np.float32
    )
    valid = np.ones((96, 96), dtype=np.uint8)
    valid[:4, :] = 0
    valid[:, :3] = 0
    device = torch.device("cpu")
    classes = predict(
        rgn,
        valid,
        load_models(model_dir, device=device),
        device=device,
        patch_size=96,
        patch_overlap=32,
        batch_size=1,
    )
    assert hashlib.sha256(classes.tobytes()).hexdigest() == (
        "50f4df3244e6731764c882014d59aea69050f244c10ad8de8bb6d8b0538755f1"
    )
    counts = {
        value: int(np.count_nonzero(classes == value))
        for value in (0, 1, 2, 3, 255)
    }
    assert counts == {
        0: 843,
        1: 7713,
        2: 0,
        3: 0,
        255: 660,
    }
