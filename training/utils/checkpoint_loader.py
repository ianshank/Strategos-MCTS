import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_checkpoint_safe(checkpoint_path: str | Path, map_location: str = "cpu") -> Any:
    """
    Load a PyTorch checkpoint safely, handling weights_only and numpy types.

    Args:
        checkpoint_path: Path to the checkpoint file.
        map_location: Device to map location to.

    Returns:
        Loaded checkpoint dictionary.
    """
    checkpoint_path = str(checkpoint_path)

    # List of safe globals for numpy
    safe_globals = [
        np._core.multiarray.scalar,
        np.dtype,
        np.dtypes.Float64DType,
        np.dtypes.Float32DType,
        np.dtypes.Int64DType,
        np.dtypes.Int32DType,
    ]

    try:
        # Attempt to load with weights_only=True (safer, default in 2.6+)
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals(safe_globals)

        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)

    except (RuntimeError, ImportError, AttributeError) as e:
        # Fallback to standard load if safe globals fail or older torch version
        logger.warning(f"Safe load failed for {checkpoint_path} ({e}), trying unsafe load")
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        raise
