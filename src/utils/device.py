"""Device selection utilities.

Centralises the "pick the best available torch device" logic that was
previously duplicated across training/system_config.py, training/train_rnn.py,
training/neural_trainer.py, and training/meta_controller_trainer.py.

The functions degrade gracefully when torch is not installed, returning "cpu"
so that import-time defaults (e.g. dataclass fields) don't crash on optional
neural extras being absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.observability.logging import get_logger

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


def get_default_device_str() -> str:
    """Return the best available torch device as a string.

    Selection order: CUDA > MPS (Apple Silicon) > CPU. Falls back to "cpu"
    when torch isn't installed so this can be used in dataclass `default_factory`
    expressions without forcing the neural extras dependency.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve a device specification to a concrete torch.device.

    Behaviour:
    - None or "auto": pick the best available device (see get_default_device_str).
    - Explicit "cuda*"/"mps"/"cpu": honour the request, but fall back to CPU with
      a warning if the requested accelerator isn't available — prevents crashes
      from misconfigured environments.

    Raises ImportError if torch is not installed (callers requesting a
    torch.device necessarily need torch).
    """
    import torch  # noqa: PLC0415 — local import; torch is an optional dep

    if requested is None or requested == "auto":
        return torch.device(get_default_device_str())

    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("Requested device %r unavailable; falling back to CPU", requested)
        return torch.device("cpu")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        logger.warning("Requested device 'mps' unavailable; falling back to CPU")
        return torch.device("cpu")

    return torch.device(requested)
