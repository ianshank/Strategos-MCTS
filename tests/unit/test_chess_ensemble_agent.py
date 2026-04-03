"""Unit tests for src/games/chess/ensemble_agent.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from src.games.chess.config import AgentType
from src.games.chess.ensemble_agent import (
    AgentResponse,
    ChessStateEncoder,
    EnsembleResponse,
)


def _make_agent_response(
    agent_type: AgentType = AgentType.HRM,
    move: str = "e2e4",
    confidence: float = 0.8,
    value_estimate: float = 0.5,
    move_probabilities: dict | None = None,
) -> AgentResponse:
    return AgentResponse(
        agent_type=agent_type,
        move=move,
        confidence=confidence,
        value_estimate=value_estimate,
        move_probabilities=move_probabilities or {"e2e4": 0.6, "d2d4": 0.3, "g1f3": 0.1},
        thinking_time_ms=10.0,
    )


@pytest.mark.unit
class TestAgentResponse:
    def test_creation(self):
        resp = _make_agent_response()
        assert resp.agent_type == AgentType.HRM
        assert resp.move == "e2e4"
        assert resp.confidence == 0.8

    def test_extra_info_default(self):
        resp = _make_agent_response()
        assert resp.extra_info == {}

    def test_extra_info_custom(self):
        resp = AgentResponse(
            agent_type=AgentType.MCTS,
            move="e2e4",
            confidence=0.9,
            value_estimate=0.7,
            move_probabilities={},
            thinking_time_ms=100.0,
            extra_info={"num_simulations": 800},
        )
        assert resp.extra_info["num_simulations"] == 800


@pytest.mark.unit
class TestEnsembleResponse:
    def test_creation(self):
        routing = MagicMock()
        resp = EnsembleResponse(
            best_move="e2e4",
            move_probabilities={"e2e4": 0.7, "d2d4": 0.3},
            value_estimate=0.6,
            confidence=0.7,
            routing_decision=routing,
            agent_responses={},
            ensemble_method="weighted_vote",
            thinking_time_ms=50.0,
        )
        assert resp.best_move == "e2e4"
        assert resp.ensemble_method == "weighted_vote"


@pytest.mark.unit
class TestChessStateEncoder:
    def test_forward(self):
        encoder = ChessStateEncoder(input_channels=12, output_dim=64, hidden_dim=128)
        x = torch.randn(2, 12, 8, 8)
        out = encoder(x)
        assert out.shape == (2, 64)

    def test_single_batch(self):
        encoder = ChessStateEncoder(input_channels=6, output_dim=32, hidden_dim=64)
        x = torch.randn(1, 6, 8, 8)
        out = encoder(x)
        assert out.shape == (1, 32)


@pytest.mark.unit
class TestChessEnsembleAgentCombineMethods:
    """Test the combination methods without initializing the full agent."""

    def _make_agent_with_config(self, combination_method="weighted_vote"):
        """Create a minimal mock of ChessEnsembleAgent for combination testing."""
        agent = MagicMock()
        agent.config = MagicMock()
        agent.config.ensemble.combination_method = combination_method

        # Import the actual methods
        from src.games.chess.ensemble_agent import ChessEnsembleAgent

        agent._combine_responses = ChessEnsembleAgent._combine_responses.__get__(agent)
        agent._weighted_vote_combination = ChessEnsembleAgent._weighted_vote_combination.__get__(agent)
        agent._max_confidence_combination = ChessEnsembleAgent._max_confidence_combination.__get__(agent)
        agent._bayesian_combination = ChessEnsembleAgent._bayesian_combination.__get__(agent)
        return agent

    def test_weighted_vote(self):
        agent = self._make_agent_with_config("weighted_vote")
        responses = {
            "hrm": _make_agent_response(AgentType.HRM, "e2e4", 0.8, 0.5, {"e2e4": 0.7, "d2d4": 0.3}),
            "trm": _make_agent_response(AgentType.TRM, "d2d4", 0.6, 0.4, {"e2e4": 0.3, "d2d4": 0.7}),
            "mcts": _make_agent_response(AgentType.MCTS, "e2e4", 0.9, 0.6, {"e2e4": 0.8, "d2d4": 0.2}),
        }
        weights = {"hrm": 0.3, "trm": 0.3, "mcts": 0.4}
        best_move, probs, value = agent._combine_responses(responses, weights)
        assert best_move in ("e2e4", "d2d4")
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_max_confidence(self):
        agent = self._make_agent_with_config("max_confidence")
        responses = {
            "hrm": _make_agent_response(AgentType.HRM, "e2e4", 0.3, 0.5),
            "trm": _make_agent_response(AgentType.TRM, "d2d4", 0.9, 0.4),
        }
        best_move, probs, value = agent._combine_responses(responses, {})
        assert best_move == "d2d4"
        assert value == 0.4

    def test_bayesian(self):
        agent = self._make_agent_with_config("bayesian")
        responses = {
            "hrm": _make_agent_response(AgentType.HRM, "e2e4", 0.8, 0.6, {"e2e4": 0.9, "d2d4": 0.1}),
            "mcts": _make_agent_response(AgentType.MCTS, "d2d4", 0.5, 0.3, {"e2e4": 0.2, "d2d4": 0.8}),
        }
        weights = {"hrm": 0.5, "mcts": 0.5}
        best_move, probs, value = agent._combine_responses(responses, weights)
        assert best_move in ("e2e4", "d2d4")
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_unknown_method_defaults_to_weighted_vote(self):
        agent = self._make_agent_with_config("unknown")
        responses = {
            "hrm": _make_agent_response(AgentType.HRM, "e2e4", 0.8, 0.5, {"e2e4": 1.0}),
        }
        weights = {"hrm": 1.0}
        best_move, probs, value = agent._combine_responses(responses, weights)
        assert best_move == "e2e4"

    def test_weighted_vote_normalizes(self):
        agent = self._make_agent_with_config("weighted_vote")
        responses = {
            "hrm": _make_agent_response(AgentType.HRM, "e2e4", 0.5, 0.5, {"e2e4": 0.5, "d2d4": 0.5}),
        }
        weights = {"hrm": 0.5}
        _, probs, _ = agent._combine_responses(responses, weights)
        assert abs(sum(probs.values()) - 1.0) < 0.01
