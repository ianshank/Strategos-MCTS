"""GPU utilities for memory tracking, pre-flight checks, and CUDA management.

Provides hardware introspection, GPU memory limit enforcement, and context managers
for tracking peak memory usage during training and inference passes.
"""

from __future__ import annotations

import types
from typing import Any

from src.config.constants import (
    DEFAULT_CUDA_MEMORY_FRACTION,
    MAX_CUDA_MEMORY_FRACTION,
    MIN_CUDA_MEMORY_FRACTION,
    MIN_GPU_MEMORY_GB,
)
from src.observability.logging import get_logger

logger = get_logger(__name__)


def get_gpu_info() -> dict[str, Any]:
    """Retrieve detailed CUDA GPU info safely.

    Returns dict with keys: cuda_available, device_count, device_name,
    memory_total_gb, memory_allocated_gb, memory_reserved_gb, capability.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return {
            "cuda_available": False,
            "device_count": 0,
            "device_name": "N/A",
            "memory_total_gb": 0.0,
            "memory_allocated_gb": 0.0,
            "memory_reserved_gb": 0.0,
            "capability": (0, 0),
        }

    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "device_count": 0,
            "device_name": "N/A",
            "memory_total_gb": 0.0,
            "memory_allocated_gb": 0.0,
            "memory_reserved_gb": 0.0,
            "capability": (0, 0),
        }

    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    total_mem = props.total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
    reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)

    return {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": props.name,
        "memory_total_gb": round(total_mem, 2),
        "memory_allocated_gb": round(allocated, 2),
        "memory_reserved_gb": round(reserved, 2),
        "capability": (props.major, props.minor),
    }


def check_gpu_ready(min_memory_gb: float = MIN_GPU_MEMORY_GB) -> bool:
    """Pre-flight check to verify if a CUDA GPU is available and has sufficient free memory."""
    info = get_gpu_info()
    if not info["cuda_available"]:
        logger.info("GPU pre-flight check: CUDA is not available")
        return False

    try:
        import torch  # noqa: PLC0415

        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        free_mem = free_bytes / (1024**3)
    except Exception as err:
        logger.warning("GPU pre-flight check: Failed to query CUDA memory info: %s", err)
        return False

    if free_mem < min_memory_gb:
        logger.warning(
            "GPU pre-flight check: Available memory (%.2f GB) is below required minimum (%.2f GB)",
            free_mem,
            min_memory_gb,
        )
        return False

    logger.info(
        "GPU pre-flight check PASSED: %s with %.2f GB free memory",
        info["device_name"],
        free_mem,
    )
    return True


def set_cuda_memory_fraction(fraction: float = DEFAULT_CUDA_MEMORY_FRACTION) -> None:
    """Set the maximum fraction of GPU memory PyTorch can allocate."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return

    if torch.cuda.is_available():
        clamped_fraction = max(MIN_CUDA_MEMORY_FRACTION, min(MAX_CUDA_MEMORY_FRACTION, fraction))
        try:

            torch.cuda.set_per_process_memory_fraction(clamped_fraction)
            logger.info("Set CUDA per-process memory fraction to %.2f", clamped_fraction)
        except Exception as err:
            logger.warning("Failed to set CUDA memory fraction: %s", err)


class GPUMemoryTracker:
    """Context manager for tracking peak GPU memory usage during code execution."""

    def __init__(self, name: str = "operation") -> None:
        self.name = name
        self.peak_allocated_mb: float = 0.0
        self.peak_reserved_mb: float = 0.0
        self.start_allocated_mb: float = 0.0

    def __enter__(self) -> GPUMemoryTracker:
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return self

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_allocated_mb = torch.cuda.memory_allocated() / (1024**2)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return

        if torch.cuda.is_available():
            self.peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
            self.peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
            logger.debug(
                "[%s] GPU Peak Allocated: %.2f MB (Reserved: %.2f MB, Net Delta: %.2f MB)",
                self.name,
                self.peak_allocated_mb,
                self.peak_reserved_mb,
                self.peak_allocated_mb - self.start_allocated_mb,
            )
