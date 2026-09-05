"""Structural typing for UnifiedTrainingOrchestrator mixin hosts."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OrchestratorHost(Protocol):
    """Attrs mixins expect on the composed UnifiedTrainingOrchestrator."""

    device: Any
    config: Any
    monitor: Any
    self_play_collector: Any
    initial_state_fn: Any
    replay_buffer: Any
    policy_value_net: Any
    pv_optimizer: Any
    pv_loss_fn: Any
    pv_scheduler: Any
    scaler: Any
    hrm_agent: Any
    hrm_optimizer: Any
    hrm_loss_fn: Any
    trm_agent: Any
    trm_optimizer: Any
    trm_loss_fn: Any
    best_model_path: Any
    best_win_rate: float
    checkpoint_dir: Any
    current_iteration: int
    mcts: Any

    def _compute_gradient_norm(self, model: Any) -> float: ...
