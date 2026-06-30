"""Unit tests for MCTS early-termination wiring through the graph builder (Phase 4.1).

The early-termination *logic* lives in ``mcts/core.py`` and is covered by
``test_mcts_early_termination.py``. These tests pin the *wiring*: the graph builder
must pass the ``MCTSConfig`` thresholds into ``MCTSEngine.search`` and gate
value-convergence stopping behind ``MCTSConfig.enable_early_termination`` (default off,
preserving historical behavior).
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, Mock

import pytest

from src.framework.mcts.config import MCTSConfig

pytestmark = [pytest.mark.unit]


def _make_builder(mcts_config):
    """Construct a GraphBuilder with lightweight mocks and a patched search engine."""
    from src.framework.graph.builder import GraphBuilder

    builder = GraphBuilder(
        hrm_agent=Mock(),
        trm_agent=Mock(),
        model_adapter=Mock(),
        logger=logging.getLogger("test.early_termination_wiring"),
        mcts_config=mcts_config,
    )
    # Patch only the search call; keep the real engine for clear_cache()/rng/tree stats.
    builder.mcts_engine.search = AsyncMock(
        return_value=(
            "action_A",
            {
                "early_stopped": True,
                "termination_reason": "value_converged",
                "iterations_run": 5,
                "iterations": 5,
                "cache_hit_rate": 0.0,
                "best_action_visits": 3,
                "best_action_value": 0.5,
            },
        )
    )
    return builder


@pytest.mark.asyncio
async def test_builder_passes_config_thresholds_to_search():
    """Builder forwards every MCTSConfig early-termination threshold into search()."""
    config = MCTSConfig(enable_early_termination=True, early_stop_threshold=0.02, early_stop_patience=7)
    builder = _make_builder(config)

    await builder._mcts_simulator_node({"query": "test query"})

    kwargs = builder.mcts_engine.search.call_args.kwargs
    assert kwargs["early_stop_threshold"] == 0.02  # enabled -> config value flows through
    assert kwargs["early_stop_patience"] == 7
    assert kwargs["early_termination_threshold"] == config.early_termination_threshold
    assert kwargs["min_iterations_before_termination"] == config.min_iterations_before_termination


@pytest.mark.asyncio
async def test_disabled_flag_passes_zero_threshold():
    """With the flag off (default), value-convergence is disabled via early_stop_threshold=0.0."""
    config = MCTSConfig(enable_early_termination=False, early_stop_threshold=0.05)
    builder = _make_builder(config)

    await builder._mcts_simulator_node({"query": "test query"})

    kwargs = builder.mcts_engine.search.call_args.kwargs
    assert kwargs["early_stop_threshold"] == 0.0  # gated off -> historical behavior preserved
