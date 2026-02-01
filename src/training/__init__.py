"""
Training Module for Multi-Agent MCTS.

Provides:
- Experiment tracking (Braintrust, W&B)
- Training pipelines
- Model evaluation with statistical rigor
- Artifact management
"""

from __future__ import annotations

# Check PyTorch availability for conditional imports
_HAS_TORCH = False
try:
    import torch as _torch

    _HAS_TORCH = True
except ImportError:
    pass


# Always available exports
__all__ = [
    # Experiment tracking
    "BraintrustTracker",
    "WandBTracker",
    "UnifiedExperimentTracker",
    "TrainingMetrics",
    "ExperimentConfig",
]


# Experiment tracker (doesn't require torch)
from .experiment_tracker import (
    BraintrustTracker,
    ExperimentConfig,
    TrainingMetrics,
    UnifiedExperimentTracker,
    WandBTracker,
)

# Conditional imports for PyTorch-dependent modules
if _HAS_TORCH:
    from .evaluation_service import (
        EvaluationConfig,
        EvaluationResult,
        EvaluationService,
        EvaluationStrategy,
        GameOutcome,
        GameResult,
        MetricsHistory,
        create_evaluation_service,
    )

    __all__.extend(
        [
            "EvaluationService",
            "EvaluationConfig",
            "EvaluationResult",
            "EvaluationStrategy",
            "GameOutcome",
            "GameResult",
            "MetricsHistory",
            "create_evaluation_service",
        ]
    )


__version__ = "1.0.0"
