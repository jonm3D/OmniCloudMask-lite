from __future__ import annotations

import hashlib

import pytest

from omnicloudmask_lite.models import ModelSpec, verify_model


def test_verify_model_checks_size_and_digest(tmp_path) -> None:
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"model")
    spec = ModelSpec(
        encoder="test",
        filename=path.name,
        bytes=5,
        sha256=hashlib.sha256(b"model").hexdigest(),
    )
    verify_model(path, spec)
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="size mismatch"):
        verify_model(path, spec)
