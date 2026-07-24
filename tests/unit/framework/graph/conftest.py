"""Shared fixtures for the graph MCTS candidate-scoring seam tests."""

from __future__ import annotations

import logging

import pytest

from src.framework.graph.builder import GraphBuilder
from src.framework.mcts.config import ConfigPreset, create_preset_config


@pytest.fixture
def make_graph_builder():
    """Factory building a minimal GraphBuilder for exercising ``_mcts_simulator_node``.

    Agents/adapters are unused by the MCTS node, so they are ``None``; a small
    fixed-seed MCTS config keeps runs fast and deterministic.
    """

    def _make(candidate_scorer=None):
        return GraphBuilder(
            hrm_agent=None,
            trm_agent=None,
            model_adapter=None,
            logger=logging.getLogger("test.graph.seam"),
            mcts_config=create_preset_config(ConfigPreset.BALANCED),
            candidate_scorer=candidate_scorer,
        )

    return _make


@pytest.fixture
def mcts_state():
    """Minimal AgentState-shaped input for the MCTS node."""
    return {"query": "What is the best next action for this task?"}
