"""Immutable model acquisition and construction."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import segmentation_models_pytorch as smp
import torch
from safetensors.torch import load_file

from .mps_patch import patch_models_for_mps


@dataclass(frozen=True)
class ModelSpec:
    encoder: str
    filename: str
    bytes: int
    sha256: str


def manifest() -> dict[str, Any]:
    resource = files("omnicloudmask_lite").joinpath("model_manifest.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def model_specs() -> tuple[ModelSpec, ...]:
    return tuple(ModelSpec(**entry) for entry in manifest()["models"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path, spec: ModelSpec) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Model file is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != spec.bytes:
        raise RuntimeError(
            f"Model size mismatch for {path.name}: {actual_size} != {spec.bytes}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != spec.sha256:
        raise RuntimeError(
            f"Model SHA-256 mismatch for {path.name}: {actual_hash} != {spec.sha256}"
        )


def fetch_models(model_dir: str | Path) -> tuple[Path, ...]:
    """Fetch the exact v4 ensemble, refusing any byte or digest mismatch."""
    target_dir = Path(model_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    model_manifest = manifest()
    revision = model_manifest["revision"]
    repository = model_manifest["repository"]
    resolved: list[Path] = []
    for spec in model_specs():
        destination = target_dir / spec.filename
        if destination.exists():
            verify_model(destination, spec)
            resolved.append(destination)
            continue
        url = (
            f"https://huggingface.co/{repository}/resolve/{revision}/"
            f"{spec.filename}?download=true"
        )
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urllib.request.urlopen(url) as response, temporary.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            verify_model(temporary, spec)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        resolved.append(destination)
    return tuple(resolved)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def load_models(
    model_dir: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> list[torch.nn.Module]:
    """Construct and load only the pinned two-model SMP v4 ensemble."""
    root = Path(model_dir).expanduser().resolve()
    models: list[torch.nn.Module] = []
    for spec in model_specs():
        path = root / spec.filename
        verify_model(path, spec)
        model = smp.Unet(
            encoder_name=spec.encoder,
            encoder_weights=None,
            in_channels=3,
            classes=4,
        )
        model.load_state_dict(load_file(path, device="cpu"))
        model.eval()
        models.append(model.to(device=device, dtype=dtype))
    return patch_models_for_mps(models, device, dtype)
