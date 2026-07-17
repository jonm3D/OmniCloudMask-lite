# OmniCloudMask-lite

This is the deliberately narrow OmniCloudMask v4 inference runtime vendored for
`multimethod-sdb`. It accepts only an explicit Red/Green/NIR NumPy array and an
explicit validity mask. Geospatial I/O, sensor adapters, QA composition, and
pipeline orchestration belong to the parent project.

The model ensemble and Hugging Face revision are immutable in
`omnicloudmask_lite/model_manifest.json`. Class values are:

| Value | Meaning |
|---:|---|
| 0 | clear |
| 1 | thick cloud |
| 2 | thin cloud |
| 3 | cloud shadow |
| 255 | invalid / not inferred |

## Commands

```console
uv sync --frozen
uv run omnicloudmask-lite fetch-model --model-dir .models/ocm-v4
uv run omnicloudmask-lite infer \
  --input rgn.npy --valid-mask valid.npy --output classes.npy \
  --model-dir .models/ocm-v4 --device auto --dtype fp32 \
  --patch-size 1000 --patch-overlap 300 --batch-size 1
```

The `infer` command writes a `uint8` class array and an adjacent JSON record.
Inputs are never interpreted as rasters and are never modified.

The original project is Copyright (c) 2022 Nick Wright and is distributed under
the MIT license retained in this repository.
