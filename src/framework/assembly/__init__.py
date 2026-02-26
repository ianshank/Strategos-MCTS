"""
Assembly Theory integration for LangGraph Multi-Agent MCTS Framework.

This module implements Assembly Theory concepts for quantifying complexity
and guiding agent decisions based on construction pathways and reusability.

Key Components:
- AssemblyIndexCalculator: Core complexity measurement
- ConceptExtractor: Extract concepts and build dependency graphs
- AssemblyGraph: Specialized graph structure for assembly pathways
- SubstructureLibrary: Pattern reuse tracking
- AssemblyFeatureExtractor: Feature extraction for ML models

Based on:
Assembly Theory by Cronin, Walker, et al.

Note: This module requires networkx for graph operations.
If networkx is not installed, imports will fail gracefully.
"""

import logging
from typing import TYPE_CHECKING

_logger = logging.getLogger(__name__)

# Check if networkx is available
_NETWORKX_AVAILABLE = False
try:
    import networkx  # noqa: F401

    _NETWORKX_AVAILABLE = True
except ImportError:
    _logger.debug("networkx not available - assembly module will be limited")

# Only import if networkx is available
if _NETWORKX_AVAILABLE:
    from .calculator import AssemblyIndexCalculator
    from .concept_extractor import Concept, ConceptExtractor
    from .config import AssemblyConfig
    from .features import AssemblyFeatureExtractor, AssemblyFeatures
    from .graph import AssemblyGraph, AssemblyNode
    from .substructure_library import Match, SubstructureLibrary

    __all__ = [
        "AssemblyIndexCalculator",
        "ConceptExtractor",
        "Concept",
        "AssemblyGraph",
        "AssemblyNode",
        "SubstructureLibrary",
        "Match",
        "AssemblyFeatureExtractor",
        "AssemblyFeatures",
        "AssemblyConfig",
        "is_assembly_available",
    ]
else:
    # Provide stub exports for type checking
    if TYPE_CHECKING:
        from .calculator import AssemblyIndexCalculator
        from .concept_extractor import Concept, ConceptExtractor
        from .config import AssemblyConfig
        from .features import AssemblyFeatureExtractor, AssemblyFeatures
        from .graph import AssemblyGraph, AssemblyNode
        from .substructure_library import Match, SubstructureLibrary

    __all__ = ["is_assembly_available"]


def is_assembly_available() -> bool:
    """Check if assembly module is fully available (requires networkx)."""
    return _NETWORKX_AVAILABLE


__version__ = "0.1.0"
