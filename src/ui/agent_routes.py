"""
Routing table and result shaping for the meta-controller demo UI.

The meta-controller picks an agent by name; this module maps that name to how the
query should actually be executed, and turns the framework's ``QueryResult`` into
the reasoning trace the UI renders.

It lives under ``src/`` rather than in the root ``app.py`` deliberately: ``app.py``
sits outside ``[tool.coverage.run] source = ["src"]``, so logic defined there is
invisible to the 85% gate by construction. Anything here is measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "AGENT_ROUTES",
    "AgentRouteSpec",
    "default_route",
    "derive_reasoning_steps",
]


class _QueryResultLike(Protocol):
    """Structural view of ``src.api.framework_service.QueryResult``.

    Declared structurally so this module carries no import-time dependency on the
    API layer (and stays testable without it).
    """

    agents_used: list[str]
    mcts_stats: dict[str, Any] | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AgentRouteSpec:
    """How a meta-controller decision maps onto a framework call."""

    key: str
    label: str
    use_mcts: bool
    description: str = ""


# The three routes the meta-controller can select. ``use_mcts`` is the only real
# behavioural difference at the framework boundary: HRM and TRM are single-shot
# reasoning passes, while the MCTS route runs tree search.
AGENT_ROUTES: dict[str, AgentRouteSpec] = {
    "hrm": AgentRouteSpec(
        key="hrm",
        label="HRM (Hierarchical Reasoning)",
        use_mcts=False,
        description="Decomposes a problem into hierarchical subproblems.",
    ),
    "trm": AgentRouteSpec(
        key="trm",
        label="TRM (Iterative Refinement)",
        use_mcts=False,
        description="Refines a candidate answer through recursive passes.",
    ),
    "mcts": AgentRouteSpec(
        key="mcts",
        label="MCTS (Monte Carlo Tree Search)",
        use_mcts=True,
        description="Explores alternatives via UCB1-guided tree search.",
    ),
}


def default_route() -> AgentRouteSpec:
    """Route used when the controller returns an unrecognized agent name."""
    return AGENT_ROUTES["hrm"]


def derive_reasoning_steps(result: _QueryResultLike) -> list[str]:
    """
    Build a reasoning trace from what the framework actually reported.

    Prefers the framework's own ``reasoning_steps`` metadata; otherwise
    reconstructs a trace from the agents that ran and any MCTS statistics. The
    previous implementation returned a fixed list of prose regardless of what
    executed, so the trace described the architecture rather than the run.

    Args:
        result: The framework's query result.

    Returns:
        Human-readable steps. Empty when the framework reported nothing, so
        callers can distinguish "no trace" from an invented one.
    """
    metadata = getattr(result, "metadata", None) or {}

    reported = metadata.get("reasoning_steps")
    if isinstance(reported, list) and reported:
        return [str(step) for step in reported]

    steps: list[str] = []

    agents = getattr(result, "agents_used", None) or []
    if agents:
        steps.append(f"Agents executed: {', '.join(str(a) for a in agents)}")

    mcts_stats = getattr(result, "mcts_stats", None)
    if mcts_stats:
        iterations = mcts_stats.get("iterations")
        if iterations is not None:
            steps.append(f"MCTS iterations: {iterations}")
        nodes = mcts_stats.get("nodes_explored") or mcts_stats.get("total_nodes")
        if nodes is not None:
            steps.append(f"Nodes explored: {nodes}")
        best = mcts_stats.get("best_action")
        if best is not None:
            steps.append(f"Best action: {best}")

    if metadata.get("rag_context_used"):
        steps.append("Retrieved supporting context via RAG")

    return steps
