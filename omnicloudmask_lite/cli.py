"""Command-line boundary used by the parent project."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .inference import predict
from .models import fetch_models, load_models, manifest, model_specs, resolve_device


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omnicloudmask-lite")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch-model")
    fetch.add_argument("--model-dir", type=Path, required=True)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--valid-mask", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--model-dir", type=Path, required=True)
    infer.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    infer.add_argument("--dtype", choices=("fp32",), default="fp32")
    infer.add_argument("--patch-size", type=int, default=1000)
    infer.add_argument("--patch-overlap", type=int, default=300)
    infer.add_argument("--batch-size", type=int, default=1)
    return parser


def _infer(arguments: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    device = resolve_device(arguments.device)
    input_path = arguments.input.expanduser().resolve()
    valid_path = arguments.valid_mask.expanduser().resolve()
    output_path = arguments.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgn = np.load(input_path, allow_pickle=False)
    valid = np.load(valid_path, allow_pickle=False)
    models = load_models(arguments.model_dir, device=device, dtype=torch.float32)
    classes = predict(
        rgn,
        valid,
        models,
        device=device,
        patch_size=arguments.patch_size,
        patch_overlap=arguments.patch_overlap,
        batch_size=arguments.batch_size,
    )
    np.save(output_path, classes, allow_pickle=False)
    counts = {
        str(value): int(np.count_nonzero(classes == value))
        for value in (0, 1, 2, 3, 255)
    }
    model_manifest = manifest()
    record: dict[str, object] = {
        "schema_version": 1,
        "runtime_version": __version__,
        "input": str(input_path),
        "input_sha256": _hash(input_path),
        "valid_mask": str(valid_path),
        "valid_mask_sha256": _hash(valid_path),
        "output": str(output_path),
        "output_sha256": _hash(output_path),
        "shape": list(classes.shape),
        "class_counts": counts,
        "model_repository": model_manifest["repository"],
        "model_revision": model_manifest["revision"],
        "model_sha256": [spec.sha256 for spec in model_specs()],
        "device": str(device),
        "dtype": arguments.dtype,
        "patch_size": arguments.patch_size,
        "patch_overlap": arguments.patch_overlap,
        "batch_size": arguments.batch_size,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }
    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "fetch-model":
        paths = fetch_models(arguments.model_dir)
        result: object = {"models": [str(path) for path in paths]}
    else:
        result = _infer(arguments)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
