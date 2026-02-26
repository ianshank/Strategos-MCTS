"""
Neural Meta-Controller package for Multi-Agent MCTS Framework.

This package provides the base infrastructure for neural network-based
meta-controllers that dynamically select which agent to route queries to.

Includes Assembly Theory integration for hybrid neural+rule-based routing.

Note: This module requires PyTorch for neural components.
If PyTorch is not installed, only base classes will be available.
"""

import contextlib
import logging
from typing import TYPE_CHECKING

_logger = logging.getLogger(__name__)

# Track what's available
_TORCH_AVAILABLE = False
_BERT_AVAILABLE = False
_FEATURE_EXTRACTOR_AVAILABLE = False
_ASSEMBLY_AVAILABLE = False

# Check if torch is available
try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _logger.debug("PyTorch not available - neural meta-controllers will be limited")

# Always import base classes (no torch dependency)
from src.agents.meta_controller.base import (
    AbstractMetaController,
    MetaControllerFeatures,
    MetaControllerPrediction,
)

# Build __all__ with always-available exports
__all__ = [
    "AbstractMetaController",
    "MetaControllerFeatures",
    "MetaControllerPrediction",
    "is_torch_available",
    "is_rnn_available",
    "is_bert_available",
    "is_assembly_router_available",
]


def is_torch_available() -> bool:
    """Check if PyTorch is available."""
    return _TORCH_AVAILABLE


def is_rnn_available() -> bool:
    """Check if RNN meta-controller is available (requires PyTorch)."""
    return _TORCH_AVAILABLE


def is_bert_available() -> bool:
    """Check if BERT meta-controller is available (requires PyTorch + transformers)."""
    return _BERT_AVAILABLE


def is_assembly_router_available() -> bool:
    """Check if assembly router is available (requires networkx)."""
    return _ASSEMBLY_AVAILABLE


# Import torch-dependent components if available
if _TORCH_AVAILABLE:
    from src.agents.meta_controller.rnn_controller import (
        RNNMetaController,
        RNNMetaControllerModel,
    )
    from src.agents.meta_controller.utils import (
        features_to_tensor,
        features_to_text,
        normalize_features,
        one_hot_encode_agent,
    )

    __all__.extend(
        [
            "normalize_features",
            "one_hot_encode_agent",
            "features_to_tensor",
            "features_to_text",
            "RNNMetaController",
            "RNNMetaControllerModel",
        ]
    )

    # Try importing BERT controller (requires transformers/peft)
    try:
        from src.agents.meta_controller.bert_controller import BERTMetaController

        _BERT_AVAILABLE = True
        __all__.append("BERTMetaController")
    except ImportError:
        _logger.debug("BERT meta-controller not available - transformers/peft not installed")

    # Try importing hybrid controller
    try:
        from src.agents.meta_controller.hybrid_controller import (
            HybridMetaController,
            HybridPrediction,
        )

        __all__.extend(["HybridMetaController", "HybridPrediction"])
    except ImportError:
        _logger.debug("Hybrid meta-controller not available")

# Try importing feature extractor (may have additional dependencies)
try:
    from src.agents.meta_controller.feature_extractor import (
        FeatureExtractor,
        FeatureExtractorConfig,
    )

    _FEATURE_EXTRACTOR_AVAILABLE = True
    __all__.extend(["FeatureExtractor", "FeatureExtractorConfig"])
except ImportError:
    _logger.debug("Feature extractor not available")

# Try importing Assembly Theory components (requires networkx)
try:
    from src.agents.meta_controller.assembly_router import (
        AssemblyRouter,
        RoutingDecision,
    )

    _ASSEMBLY_AVAILABLE = True
    __all__.extend(["AssemblyRouter", "RoutingDecision"])
except ImportError:
    _logger.debug("Assembly router not available - networkx not installed")

# TYPE_CHECKING imports for static analysis
# Note: These are guarded with try/except to handle cases where
# dependencies are not installed but type checking is still run
if TYPE_CHECKING:
    with contextlib.suppress(ImportError):
        from src.agents.meta_controller.rnn_controller import (  # noqa: F401
            RNNMetaController,
            RNNMetaControllerModel,
        )
        from src.agents.meta_controller.utils import (  # noqa: F401
            features_to_tensor,
            features_to_text,
            normalize_features,
            one_hot_encode_agent,
        )

    with contextlib.suppress(ImportError):
        from src.agents.meta_controller.bert_controller import BERTMetaController  # noqa: F401

    with contextlib.suppress(ImportError):
        from src.agents.meta_controller.hybrid_controller import (  # noqa: F401
            HybridMetaController,
            HybridPrediction,
        )

    with contextlib.suppress(ImportError):
        from src.agents.meta_controller.feature_extractor import (  # noqa: F401
            FeatureExtractor,
            FeatureExtractorConfig,
        )

    with contextlib.suppress(ImportError):
        from src.agents.meta_controller.assembly_router import (  # noqa: F401
            AssemblyRouter,
            RoutingDecision,
        )
