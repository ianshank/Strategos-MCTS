# mypy: disable-error-code="attr-defined,misc,assignment"
from typing import Any

from src.api.exceptions import TrainingError
from src.observability.logging import get_structured_logger

logger = get_structured_logger(__name__)


def _strict_training_errors() -> bool:
    """
    Resolve whether training-step failures should raise (vs. return zero metrics).

    Reads ``Settings.TRAINING_STRICT_ERRORS`` defensively so the orchestrator still works
    when full settings validation is unavailable; defaults to ``False`` (current behavior).
    """
    try:
        from src.config.settings import get_settings

        return bool(get_settings().TRAINING_STRICT_ERRORS)
    except Exception:
        return False


def _handle_training_failure(stage: str, reason: str, zero_metrics: dict[str, float]) -> dict[str, Any]:
    """
    Centralize the strict-vs-degraded decision for a failed training step.

    When ``TRAINING_STRICT_ERRORS`` is enabled, raise a ``TrainingError`` so a failed step
    can never masquerade as a successful one with zero loss. Otherwise emit a structured
    ``training_step_degraded`` warning (so the silent degradation is observable) and return
    the zero-filled metrics with a degraded flag set to True.
    """
    if _strict_training_errors():
        raise TrainingError(user_message=f"Training step '{stage}' failed", internal_details=reason, stage=stage)
    logger.warning(
        "Training step degraded; returning zero metrics", event="training_step_degraded", stage=stage, reason=reason
    )
    result = zero_metrics.copy()
    result["degraded"] = True
    return result
