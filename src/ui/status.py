"""
Runtime status reporting for the demo UI.

The UI previously advertised "REAL trained models" unconditionally, in a
repository where every checkpoint is a Git-LFS pointer stub. That is a claim no
command reproduces (CHARTER NG-3). These helpers report what is actually loaded,
so the banner tracks reality instead of intent.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from src.config.constants import GIT_LFS_REMEDIATION
from src.models.checkpoints import CheckpointReport, CheckpointStatus, inspect_checkpoint

__all__ = [
    "DEFAULT_CHECKPOINTS",
    "checkpoint_status_banner",
    "inspect_configured_checkpoints",
    "resolve_checkpoint_paths",
]

# Repository-relative default locations, used when the matching settings field is
# unset. Kept here rather than inline at the call site so the UI, the loader and
# the tests all agree on where weights are expected to live.
DEFAULT_CHECKPOINTS: Mapping[str, str] = {
    "RNN meta-controller": "models/rnn_meta_controller.pt",
    "BERT LoRA adapter": "models/bert_lora/final_model",
}

_SETTINGS_FIELD_BY_LABEL: Mapping[str, str] = {
    "RNN meta-controller": "RNN_MODEL_PATH",
    "BERT LoRA adapter": "BERT_MODEL_PATH",
}


def resolve_checkpoint_paths(settings: object | None, repo_root: Path) -> dict[str, Path]:
    """
    Resolve each checkpoint's path, preferring configured overrides.

    Args:
        settings: Settings object, or ``None`` when unconfigured.
        repo_root: Directory the packaged defaults are relative to.

    Returns:
        Mapping of human label to resolved path.
    """
    resolved: dict[str, Path] = {}
    for label, default_rel in DEFAULT_CHECKPOINTS.items():
        override = None
        field = _SETTINGS_FIELD_BY_LABEL.get(label)
        if settings is not None and field:
            override = getattr(settings, field, None)
        resolved[label] = Path(override) if override else repo_root / default_rel
    return resolved


def inspect_configured_checkpoints(settings: object | None, repo_root: Path) -> dict[str, CheckpointReport]:
    """Classify every configured checkpoint. Never raises."""
    return {label: inspect_checkpoint(path) for label, path in resolve_checkpoint_paths(settings, repo_root).items()}


def checkpoint_status_banner(reports: Mapping[str, CheckpointReport]) -> str:
    """
    Render a Markdown banner describing which weights actually loaded.

    Args:
        reports: Mapping of label to its inspection report.

    Returns:
        Markdown. States trained-model operation only when every checkpoint is
        genuinely readable; otherwise names each problem and the remedy.
    """
    if not reports:
        return "**Model status unknown** — no checkpoints were configured."

    unusable = {label: report for label, report in reports.items() if not report.is_ok}

    if not unusable:
        return "✅ **Running on trained weights.** All model checkpoints loaded successfully."

    lines = [
        "⚠️ **Reduced mode — running on untrained weights.**",
        "",
        "Routing decisions below come from randomly-initialized models and are not meaningful:",
        "",
    ]
    for label, report in unusable.items():
        lines.append(f"- **{label}**: {report.detail}")

    if any(r.status is CheckpointStatus.LFS_POINTER for r in unusable.values()):
        lines += ["", f"Fetch the real weights with `{GIT_LFS_REMEDIATION}`, then restart."]

    return "\n".join(lines)
