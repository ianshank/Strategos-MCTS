"""Distributed data parallel utilities."""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from src.observability.logging import get_logger

logger = get_logger(__name__)


def is_distributed() -> bool:
    """Return True if torch.distributed is initialized."""
    import torch.distributed as dist

    return bool(dist.is_available() and dist.is_initialized())


def init_distributed(backend: str = "nccl") -> bool:
    """Initialize the process group for distributed training.

    Returns:
        bool: True if distributed training was successfully initialized or already running, False otherwise.
    """
    import torch.distributed as dist

    if is_distributed():
        return True

    try:
        dist.init_process_group(backend=backend)
        logger.info(
            "Distributed process group initialized",
            extra={"backend": backend, "rank": get_rank(), "world_size": get_world_size()},
        )
        return True
    except (ValueError, RuntimeError) as e:
        logger.warning("Failed to initialize distributed process group: %s", e)
        return False


def cleanup_distributed() -> None:
    """Destroy the distributed process group."""
    import torch.distributed as dist

    if is_distributed():
        dist.destroy_process_group()


def get_rank() -> int:
    """Return the global rank of the current process."""
    if is_distributed():
        import torch.distributed as dist

        return int(dist.get_rank())
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """Return the local rank of the current process."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size(default: int = 1) -> int:
    """Return the world size of the distributed setup."""
    if is_distributed():
        import torch.distributed as dist

        return int(dist.get_world_size())
    return int(os.environ.get("WORLD_SIZE", str(default)))


def is_main_process() -> bool:
    """Return True if this is the main process (rank 0)."""
    return get_rank() == 0


def wrap_ddp(module: nn.Module, device: str | torch.device) -> nn.Module:
    """Wrap a neural network in DistributedDataParallel if running in a distributed context."""
    if not is_distributed():
        return module

    from torch.nn.parallel import DistributedDataParallel as DDP

    device_str = str(device)
    device_ids = [get_local_rank()] if device_str.startswith("cuda") else None

    try:
        wrapped: nn.Module = DDP(module, device_ids=device_ids)
        return wrapped
    except Exception as e:
        logger.error("Failed to wrap model in DDP: %s", e)
        return module


def unwrap_model(module: nn.Module) -> nn.Module:
    """Extract the base module, removing any DDP wrappers."""
    if hasattr(module, "module"):
        from typing import cast

        return cast(nn.Module, module.module)
    return module
