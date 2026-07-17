"""Narrow, pinned OmniCloudMask v4 inference API."""

from .inference import predict
from .models import fetch_models, load_models, resolve_device

__all__ = ["fetch_models", "load_models", "predict", "resolve_device"]
__version__ = "0.1.0"
