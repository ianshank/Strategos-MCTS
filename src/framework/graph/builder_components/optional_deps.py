"""Optional heavy dependencies for GraphBuilder mixins.

Module-level names are annotated as ``Any`` and default to ``None`` so
ImportError fallbacks type-check cleanly (no per-assignment ``# type: ignore``).
Mixins import from here instead of duplicating try/except import blocks.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Meta-controller (optional)
# ---------------------------------------------------------------------------

AbstractMetaController: Any = None
MetaControllerFeatures: Any = None
RNNMetaController: Any = None
BERTMetaController: Any = None
MetaControllerConfig: Any = None
MetaControllerConfigLoader: Any = None
_META_CONTROLLER_AVAILABLE: bool = False

try:
    from src.agents.meta_controller.base import (
        AbstractMetaController as _AbstractMetaController,
    )
    from src.agents.meta_controller.base import (
        MetaControllerFeatures as _MetaControllerFeatures,
    )
    from src.agents.meta_controller.bert_controller import BERTMetaController as _BERTMetaController
    from src.agents.meta_controller.config_loader import (
        MetaControllerConfig as _MetaControllerConfig,
    )
    from src.agents.meta_controller.config_loader import (
        MetaControllerConfigLoader as _MetaControllerConfigLoader,
    )
    from src.agents.meta_controller.rnn_controller import RNNMetaController as _RNNMetaController

    AbstractMetaController = _AbstractMetaController
    MetaControllerFeatures = _MetaControllerFeatures
    RNNMetaController = _RNNMetaController
    BERTMetaController = _BERTMetaController
    MetaControllerConfig = _MetaControllerConfig
    MetaControllerConfigLoader = _MetaControllerConfigLoader
    _META_CONTROLLER_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Neuro-symbolic (optional)
# ---------------------------------------------------------------------------

NeuroSymbolicConfig: Any = None
SymbolicReasoningAgent: Any = None
SymbolicAgentGraphExtension: Any = None
SymbolicAgentNodeConfig: Any = None
NeuroSymbolicMCTSIntegration: Any = None
NeuroSymbolicMCTSConfig: Any = None
ConstraintSystem: Any = None
ConstraintConfig: Any = None
_NEURO_SYMBOLIC_AVAILABLE: bool = False

try:
    from src.neuro_symbolic import (
        ConstraintSystem as _ConstraintSystem,
    )
    from src.neuro_symbolic import (
        NeuroSymbolicConfig as _NeuroSymbolicConfig,
    )
    from src.neuro_symbolic import (
        NeuroSymbolicMCTSConfig as _NeuroSymbolicMCTSConfig,
    )
    from src.neuro_symbolic import (
        NeuroSymbolicMCTSIntegration as _NeuroSymbolicMCTSIntegration,
    )
    from src.neuro_symbolic import (
        SymbolicAgentGraphExtension as _SymbolicAgentGraphExtension,
    )
    from src.neuro_symbolic import (
        SymbolicAgentNodeConfig as _SymbolicAgentNodeConfig,
    )
    from src.neuro_symbolic import (
        SymbolicReasoningAgent as _SymbolicReasoningAgent,
    )
    from src.neuro_symbolic.config import ConstraintConfig as _ConstraintConfig

    NeuroSymbolicConfig = _NeuroSymbolicConfig
    SymbolicReasoningAgent = _SymbolicReasoningAgent
    SymbolicAgentGraphExtension = _SymbolicAgentGraphExtension
    SymbolicAgentNodeConfig = _SymbolicAgentNodeConfig
    NeuroSymbolicMCTSIntegration = _NeuroSymbolicMCTSIntegration
    NeuroSymbolicMCTSConfig = _NeuroSymbolicMCTSConfig
    ConstraintSystem = _ConstraintSystem
    ConstraintConfig = _ConstraintConfig
    _NEURO_SYMBOLIC_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    "AbstractMetaController",
    "MetaControllerFeatures",
    "RNNMetaController",
    "BERTMetaController",
    "MetaControllerConfig",
    "MetaControllerConfigLoader",
    "_META_CONTROLLER_AVAILABLE",
    "NeuroSymbolicConfig",
    "SymbolicReasoningAgent",
    "SymbolicAgentGraphExtension",
    "SymbolicAgentNodeConfig",
    "NeuroSymbolicMCTSIntegration",
    "NeuroSymbolicMCTSConfig",
    "ConstraintSystem",
    "ConstraintConfig",
    "_NEURO_SYMBOLIC_AVAILABLE",
]
