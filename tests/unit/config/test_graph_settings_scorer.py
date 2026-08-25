"""Tests for the GRAPH_MCTS_CANDIDATE_SCORER setting.

Covers strategos_subgoal_scoring_seam AC-2: the scorer is a bounded Pydantic Settings
enum defaulting to 'identity', and an invalid value is rejected at construction (no
silent fallback).
"""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from src.config.constants import CANDIDATE_SCORER_IDENTITY, CANDIDATE_SCORER_VALUE, DEFAULT_CANDIDATE_SCORER
from src.config.graph_settings import (
    GraphHardeningSettings,
    get_graph_hardening_settings,
    reset_graph_hardening_settings,
)


def test_default_scorer_is_identity_from_constants():  # AC-2
    settings = GraphHardeningSettings()
    assert settings.GRAPH_MCTS_CANDIDATE_SCORER == DEFAULT_CANDIDATE_SCORER == CANDIDATE_SCORER_IDENTITY


def test_value_scorer_is_accepted():  # AC-2
    settings = GraphHardeningSettings(GRAPH_MCTS_CANDIDATE_SCORER=CANDIDATE_SCORER_VALUE)
    assert settings.GRAPH_MCTS_CANDIDATE_SCORER == CANDIDATE_SCORER_VALUE


def test_unknown_scorer_is_rejected_at_construction():  # AC-2
    with pytest.raises(ValidationError):
        GraphHardeningSettings(GRAPH_MCTS_CANDIDATE_SCORER="bogus")


def test_cached_accessor_and_reset(monkeypatch):
    reset_graph_hardening_settings()
    monkeypatch.setenv("GRAPH_MCTS_CANDIDATE_SCORER", CANDIDATE_SCORER_VALUE)
    reset_graph_hardening_settings()
    try:
        assert get_graph_hardening_settings().GRAPH_MCTS_CANDIDATE_SCORER == CANDIDATE_SCORER_VALUE
    finally:
        monkeypatch.delenv("GRAPH_MCTS_CANDIDATE_SCORER", raising=False)
        reset_graph_hardening_settings()
