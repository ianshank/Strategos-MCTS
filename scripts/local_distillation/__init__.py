"""Hygienic NeuralMCTS distillation helpers (Connect Four teacher labels).

This package does not import ``src.framework.mcts.llm_guided``. Teacher search is
:class:`~src.framework.mcts.neural_mcts.NeuralMCTS` only.
"""

from __future__ import annotations

from scripts.local_distillation.eval_arms import (
    COMMITTED_RESULTS_RELATIVE_PATH,
    PRIMARY_ENDPOINT,
)
from scripts.local_distillation.schema import SCHEMA_VERSION

__all__ = ["COMMITTED_RESULTS_RELATIVE_PATH", "PRIMARY_ENDPOINT", "SCHEMA_VERSION"]
