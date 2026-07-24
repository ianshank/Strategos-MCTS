"""Flag-gating integration tests for the risk-averse subgoal penalty (AC-2).

Verifies that IntegratedFramework wires the risk scorer only when the flag is ON, and keeps the
byte-for-byte-baseline identity scorer when OFF (default).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from src.config.graph_settings import reset_graph_hardening_settings
from src.framework.mcts.risk_scoring import RiskAverseSubgoalScorer
from src.framework.mcts.scoring import IdentityCandidateScorer


def _build_framework():
    from src.framework.graph.integrated import IntegratedFramework

    return IntegratedFramework(model_adapter=MagicMock(), logger=logging.getLogger("test.integ.penalty"))


@pytest.fixture(autouse=True)
def _clean_settings_cache(monkeypatch):
    # Isolate each test from ambient env / cached settings.
    for key in ("ENABLE_UNCERTAINTY_SUBGOAL_PENALTY", "SUBGOAL_UNCERTAINTY_LAMBDA"):
        monkeypatch.delenv(key, raising=False)
    reset_graph_hardening_settings()
    yield
    reset_graph_hardening_settings()


class TestFlagGating:
    def test_flag_off_keeps_identity_baseline(self):  # AC-2
        framework = _build_framework()
        assert isinstance(framework.graph_builder.candidate_scorer, IdentityCandidateScorer)

    def test_flag_on_selects_risk_scorer_with_lambda(self, monkeypatch):  # AC-2
        monkeypatch.setenv("ENABLE_UNCERTAINTY_SUBGOAL_PENALTY", "true")
        monkeypatch.setenv("SUBGOAL_UNCERTAINTY_LAMBDA", "2.5")
        reset_graph_hardening_settings()

        scorer = _build_framework().graph_builder.candidate_scorer
        assert isinstance(scorer, RiskAverseSubgoalScorer)
        assert scorer.lambda_weight == 2.5

    async def test_flag_off_node_runs_and_is_deterministic(self):  # AC-2
        # Two OFF frameworks produce identical node output on the same seed (the identity/baseline path).
        state = {"query": "penalty-off determinism query"}
        first = await _build_framework().graph_builder._mcts_simulator_node(dict(state))
        reset_graph_hardening_settings()
        second = await _build_framework().graph_builder._mcts_simulator_node(dict(state))
        assert first == second
