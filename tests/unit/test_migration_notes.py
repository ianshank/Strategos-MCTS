"""Verify docs/MIGRATION_NOTES.md documents every externally observable change.

Maps to spec ``strategos_langgraph_hardening`` AC-6: the note must exist and enumerate strict
initial-state validation, the mcts_root summary change, and the new versioned persistence
formats, while affirming the legacy results artifact and training checkpoints are unaffected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_NOTES = Path(__file__).resolve().parents[2] / "docs" / "MIGRATION_NOTES.md"


@pytest.fixture(scope="module")
def notes_text() -> str:
    assert _NOTES.exists(), "docs/MIGRATION_NOTES.md is missing"
    return _NOTES.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "marker",
    [
        "strategos_langgraph_hardening",
        "StateValidationError",  # strict initial-state validation
        "allow_extra_keys",
        "thread_id",
        "mcts_root",  # state content change
        "schema_version",  # versioned persistence formats
        "results.jsonl",  # benchmark run store
        "run.json",
        "GRAPH_TRACE_DIR",  # trace files
        "GRAPH_CHECKPOINT_BACKEND=sqlite",  # sqlite checkpoints
        "benchmark_results.json",  # legacy artifact affirmed unaffected
        "ckpt_iter_",  # training checkpoints affirmed unaffected
    ],
)
def test_migration_notes_document_change(notes_text: str, marker: str) -> None:
    assert marker in notes_text, f"MIGRATION_NOTES.md must document '{marker}'"
