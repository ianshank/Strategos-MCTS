"""
Agents module for the LangGraph Multi-Agent MCTS framework.

This module provides the core agent implementations:
- HRM (Hierarchical Reasoning Module): DeBERTa-based adaptive computation time agent
- TRM (Task Refinement Module): Iterative refinement with deep supervision
- Hybrid Agent: Cost-optimized blending of neural and LLM decisions

All agents follow the dependency injection pattern and use configuration
from Pydantic Settings (no hardcoded values).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# Lazy imports to avoid circular dependencies and optional dependency issues
_logger = logging.getLogger(__name__)

# Track what's available
_HRM_AVAILABLE = False
_TRM_AVAILABLE = False
_HYBRID_AVAILABLE = False

# Try importing HRM components
try:
    from src.agents.hrm_agent import (
        AdaptiveComputationTime,
        HModule,
        HRMAgent,
        HRMLoss,
        HRMOutput,
        LModule,
        SubProblem,
        create_hrm_agent,
    )
    _HRM_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"HRM agent not available: {e}")
    # Create placeholder types for type checking
    if TYPE_CHECKING:
        from src.agents.hrm_agent import (  # noqa: F401
            AdaptiveComputationTime,
            HModule,
            HRMAgent,
            HRMLoss,
            HRMOutput,
            LModule,
            SubProblem,
            create_hrm_agent,
        )

# Try importing TRM components
try:
    from src.agents.trm_agent import (
        DeepSupervisionHead,
        RecursiveBlock,
        TRMAgent,
        TRMLoss,
        TRMOutput,
        TRMRefinementWrapper,
        create_trm_agent,
    )
    _TRM_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"TRM agent not available: {e}")
    if TYPE_CHECKING:
        from src.agents.trm_agent import (  # noqa: F401
            DeepSupervisionHead,
            RecursiveBlock,
            TRMAgent,
            TRMLoss,
            TRMOutput,
            TRMRefinementWrapper,
            create_trm_agent,
        )

# Try importing Hybrid components
try:
    from src.agents.hybrid_agent import (
        CostSavings,
        DecisionMetadata,
        DecisionSource,
        HybridAgent,
        HybridConfig,
        create_hybrid_agent,
    )
    _HYBRID_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"Hybrid agent not available: {e}")
    if TYPE_CHECKING:
        from src.agents.hybrid_agent import (  # noqa: F401
            CostSavings,
            DecisionMetadata,
            DecisionSource,
            HybridAgent,
            HybridConfig,
            create_hybrid_agent,
        )


def is_hrm_available() -> bool:
    """Check if HRM agent is available (requires PyTorch)."""
    return _HRM_AVAILABLE


def is_trm_available() -> bool:
    """Check if TRM agent is available (requires PyTorch)."""
    return _TRM_AVAILABLE


def is_hybrid_available() -> bool:
    """Check if Hybrid agent is available."""
    return _HYBRID_AVAILABLE


# Build __all__ dynamically based on what's available
__all__ = [
    # Availability checks
    "is_hrm_available",
    "is_trm_available",
    "is_hybrid_available",
]

if _HRM_AVAILABLE:
    __all__.extend([
        # HRM Agent
        "AdaptiveComputationTime",
        "HModule",
        "HRMAgent",
        "HRMLoss",
        "HRMOutput",
        "LModule",
        "SubProblem",
        "create_hrm_agent",
    ])

if _TRM_AVAILABLE:
    __all__.extend([
        # TRM Agent
        "DeepSupervisionHead",
        "RecursiveBlock",
        "TRMAgent",
        "TRMLoss",
        "TRMOutput",
        "TRMRefinementWrapper",
        "create_trm_agent",
    ])

if _HYBRID_AVAILABLE:
    __all__.extend([
        # Hybrid Agent
        "CostSavings",
        "DecisionMetadata",
        "DecisionSource",
        "HybridAgent",
        "HybridConfig",
        "create_hybrid_agent",
    ])
