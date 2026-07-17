"""Deterministic Red/Green/NIR array inference."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch

INVALID_CLASS = np.uint8(255)


def _validate_inputs(
    rgn: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rgn = np.asarray(rgn, dtype=np.float32)
    valid = np.asarray(valid)
    if rgn.ndim != 3 or rgn.shape[0] != 3:
        raise ValueError(f"input must have shape (3, H, W); found {rgn.shape}")
    if valid.ndim != 2 or valid.shape != rgn.shape[1:]:
        raise ValueError(
            f"valid mask must have shape {rgn.shape[1:]}; found {valid.shape}"
        )
    if min(rgn.shape[1:]) < 32:
        raise ValueError("input height and width must each be at least 32 pixels")
    valid_bool = valid.astype(bool, copy=False)
    if np.any(valid_bool) and not np.all(np.isfinite(rgn[:, valid_bool])):
        raise ValueError("input contains non-finite values at valid pixels")
    return rgn, valid_bool


def _effective_patch(
    shape: tuple[int, int], size: int, overlap: int
) -> tuple[int, int]:
    if size < 32:
        raise ValueError("patch size must be at least 32 pixels")
    size = min(size, *shape)
    overlap = min(overlap, size // 2)
    if overlap < 0 or overlap >= size:
        raise ValueError(
            "patch overlap must be nonnegative and smaller than patch size"
        )
    return size, overlap


def _patch_indexes(
    height: int, width: int, size: int, overlap: int
) -> list[tuple[int, int, int, int]]:
    stride = size - overlap
    tops = list(range(0, max(height - size, 0) + 1, stride))
    lefts = list(range(0, max(width - size, 0) + 1, stride))
    if not tops or tops[-1] != height - size:
        tops.append(height - size)
    if not lefts or lefts[-1] != width - size:
        lefts.append(width - size)
    return [(top, top + size, left, left + size) for top in tops for left in lefts]


def _gradient(size: int, overlap: int, *, device: torch.device) -> torch.Tensor:
    if overlap == 0:
        return torch.ones((size, size), dtype=torch.float32, device=device)
    axis = torch.ones(size, dtype=torch.float32, device=device)
    axis[:overlap] = torch.arange(1, overlap + 1, device=device) / overlap
    axis[-overlap:] = torch.arange(overlap, 0, -1, device=device) / overlap
    return axis[:, None] * axis[None, :]


def _normalize_patch(rgn: np.ndarray, valid: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(rgn, dtype=np.float32)
    if not np.any(valid):
        return normalized
    for index, band in enumerate(rgn):
        values = band[valid]
        standard_deviation = float(values.std())
        if standard_deviation == 0:
            standard_deviation = 1.0
        normalized[index, valid] = (values - float(values.mean())) / standard_deviation
    return normalized


def _batches(
    indexes: Sequence[tuple[int, int, int, int]], batch_size: int
) -> Iterator[Sequence[tuple[int, int, int, int]]]:
    if batch_size < 1:
        raise ValueError("batch size must be at least 1")
    for start in range(0, len(indexes), batch_size):
        yield indexes[start : start + batch_size]


def predict(
    rgn: np.ndarray,
    valid: np.ndarray,
    models: Sequence[torch.nn.Module],
    *,
    device: torch.device,
    patch_size: int = 1000,
    patch_overlap: int = 300,
    batch_size: int = 1,
) -> np.ndarray:
    """Predict class values 0..3 and assign 255 wherever validity is false."""
    rgn, valid = _validate_inputs(rgn, valid)
    if not models:
        raise ValueError("at least one model is required")
    output = np.full(valid.shape, INVALID_CLASS, dtype=np.uint8)
    if not np.any(valid):
        return output

    patch_size, patch_overlap = _effective_patch(
        valid.shape, patch_size, patch_overlap
    )
    indexes = [
        index
        for index in _patch_indexes(*valid.shape, patch_size, patch_overlap)
        if np.any(valid[index[0] : index[1], index[2] : index[3]])
    ]
    gradient = _gradient(patch_size, patch_overlap, device=device)
    logits = torch.zeros((4, *valid.shape), dtype=torch.float32, device=device)
    weights = torch.zeros(valid.shape, dtype=torch.float32, device=device)

    for index_batch in _batches(indexes, batch_size):
        patches = np.stack(
            [
                _normalize_patch(
                    rgn[:, top:bottom, left:right],
                    valid[top:bottom, left:right],
                )
                for top, bottom, left, right in index_batch
            ]
        )
        tensor = torch.as_tensor(patches, dtype=torch.float32, device=device)
        predictions = []
        for model in models:
            with torch.inference_mode():
                predictions.append(model(tensor).to(dtype=torch.float32))
        ensemble = torch.stack(predictions).mean(dim=0)
        for prediction, (top, bottom, left, right) in zip(
            ensemble, index_batch, strict=True
        ):
            logits[:, top:bottom, left:right] += prediction * gradient
            weights[top:bottom, left:right] += gradient

    classes = torch.argmax(logits / weights.clamp_min(1e-12), dim=0)
    output[valid] = classes.detach().cpu().numpy().astype(np.uint8)[valid]
    return output
