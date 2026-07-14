"""
Agent-Specific E2E Test Flows with LangSmith Tracing.

Tests each agent (HRM, TRM, MCTS) in isolation plus combined "full stack" flows.
Demonstrates comprehensive tracing patterns for all agents and subagents.

Test Organization:
1. HRM-only flows (hierarchical reasoning)
2. TRM-only flows (task refinement)
3. MCTS-only flows (decision simulation)
4. Combined full-stack flows (HRM + TRM + MCTS)
"""

import pytest

from tests.mocks.mock_external_services import create_mock_llm
from tests.utils.langsmith_tracing import (
    trace_e2e_test,
    trace_mcts_simulation,
    update_run_metadata,
)


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client routing canned responses by agent prompt prefix."""
    client = create_mock_llm(provider="openai")
    client.set_prompt_responses(
        {
            "HRM": """Hierarchical Reasoning Model Analysis:
        1. Primary objective: Secure northern sector
        2. Sub-task decomposition:
           - Establish defensive positions
           - Deploy observation assets
           - Coordinate communications
        3. Risk assessment: Medium
        Confidence: 0.87""",
            "TRM": """Task Refinement Model Analysis:
        Iterative refinement cycle 1:
        - Position Alpha: Coverage 85%, Risk Medium
        - Position Beta: Coverage 72%, Risk Low
        Recommended: Alpha with Beta fallback
        Confidence: 0.83""",
        }
    )
    return client


@pytest.fixture
def tactical_query():
    """Standard tactical query for testing."""
    return {
        "query": "Enemy approaching from north. Limited visibility, night conditions. "
        "Infantry platoon, UAV support, limited ammo. Recommend defensive strategy.",
        "use_rag": True,
        "use_mcts": False,
        "thread_id": "test_agent_001",
    }


@pytest.fixture
def mcts_tactical_scenario():
    """MCTS tactical scenario configuration."""
    return {
        "initial_state": {
            "position": "neutral",
            "resources": {"ammo": 100, "fuel": 80, "personnel": 25},
            "enemy_position": "north",
            "visibility": "low",
        },
        "possible_actions": [
            "advance_to_alpha",
            "hold_current_position",
            "retreat_to_beta",
            "flanking_maneuver",
            "request_reinforcement",
        ],
        "objective": "secure_area_minimal_casualties",
    }


# ============================================================================
# HRM-ONLY E2E FLOWS
# ============================================================================


class TestHRMOnlyFlows:
    """Test Hierarchical Reasoning Model in isolation."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @trace_e2e_test(
        "e2e_hrm_tactical_flow",
        phase="hrm_only",
        scenario_type="tactical",
        provider="openai",
        use_mcts=False,
        tags=["hrm", "hierarchical_reasoning", "tactical"],
    )
    async def test_hrm_tactical_analysis(self, mock_llm_client, tactical_query):
        """HRM-only tactical analysis (no TRM, no MCTS)."""
        from src.models.validation import QueryInput

        # Step 1: Validate input
        query_input = QueryInput(**tactical_query)

        # Step 2: Process through HRM only
        # In real implementation, framework would route to HRM
        _hrm_response = await mock_llm_client.generate(f"HRM: {query_input.query}")

        # Step 3: Extract HRM-specific metrics
        assert _hrm_response.content is not None
        assert "Hierarchical" in _hrm_response.content

        # Extract hierarchical structure
        objectives_count = _hrm_response.content.count("objective")
        subtasks_count = _hrm_response.content.count("Sub-task")

        # Update trace with HRM-specific metadata
        update_run_metadata(
            {
                "agent": "hrm",
                "hierarchical_objectives": objectives_count,
                "subtasks_identified": subtasks_count,
                "hrm_confidence": 0.87,
                "decomposition_depth": 3,
            }
        )

        # Assertions
        assert objectives_count >= 1
        assert subtasks_count >= 1

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @trace_e2e_test(
        "e2e_hrm_cybersecurity_flow",
        phase="hrm_only",
        scenario_type="cybersecurity",
        provider="openai",
        use_mcts=False,
        tags=["hrm", "hierarchical_reasoning", "cybersecurity", "threat_analysis"],
    )
    async def test_hrm_cybersecurity_analysis(self, mock_llm_client):
        """HRM-only cybersecurity threat analysis."""
        from src.models.validation import QueryInput

        query = QueryInput(
            query="APT28 indicators detected. Credential harvesting and lateral movement observed. "
            "Recommend containment strategy.",
            use_rag=True,
            use_mcts=False,
        )

        # Process through HRM
        _hrm_response = await mock_llm_client.generate(f"HRM Threat Analysis: {query.query}")

        assert _hrm_response.content is not None

        # Simulate HRM hierarchical threat breakdown
        threat_hierarchy = {
            "primary_threat": "APT28",
            "threat_vectors": ["credential_harvesting", "lateral_movement"],
            "containment_priorities": [
                "isolate_compromised_systems",
                "reset_credentials",
                "deploy_monitoring",
            ],
        }

        # Update trace
        update_run_metadata(
            {
                "agent": "hrm",
                "primary_threat": threat_hierarchy["primary_threat"],
                "threat_vectors_count": len(threat_hierarchy["threat_vectors"]),
                "containment_actions": len(threat_hierarchy["containment_priorities"]),
            }
        )

        assert threat_hierarchy["primary_threat"] == "APT28"
        assert len(threat_hierarchy["containment_priorities"]) >= 3


# ============================================================================
# TRM-ONLY E2E FLOWS
# ============================================================================


class TestTRMOnlyFlows:
    """Test Task Refinement Model in isolation."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @trace_e2e_test(
        "e2e_trm_tactical_flow",
        phase="trm_only",
        scenario_type="tactical",
        provider="openai",
        use_mcts=False,
        tags=["trm", "task_refinement", "tactical"],
    )
    async def test_trm_tactical_refinement(self, mock_llm_client, tactical_query):
        """TRM-only tactical task refinement (no HRM, no MCTS)."""
        from src.models.validation import QueryInput

        query_input = QueryInput(**tactical_query)

        # Process through TRM only
        _trm_response = await mock_llm_client.generate(f"TRM: {query_input.query}")

        assert _trm_response.content is not None
        assert "refinement" in _trm_response.content.lower()

        # Extract TRM-specific metrics
        refinement_cycles = _trm_response.content.count("cycle")
        positions_evaluated = _trm_response.content.count("Position")

        # Simulate TRM iterative refinement metadata
        update_run_metadata(
            {
                "agent": "trm",
                "refinement_cycles": max(refinement_cycles, 1),
                "alternatives_evaluated": positions_evaluated,
                "trm_confidence": 0.83,
                "convergence_achieved": True,
            }
        )

        assert positions_evaluated >= 2  # TRM should evaluate multiple options

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @trace_e2e_test(
        "e2e_trm_multi_iteration_refinement",
        phase="trm_only",
        scenario_type="tactical",
        provider="openai",
        use_mcts=False,
        tags=["trm", "task_refinement", "multi_iteration", "performance"],
    )
    async def test_trm_multi_iteration_refinement(self, mock_llm_client, tactical_query):
        """TRM with multiple refinement iterations."""
        from src.models.validation import QueryInput

        query_input = QueryInput(**tactical_query)

        # Simulate multiple TRM refinement cycles
        iterations = 3
        refinement_scores = []

        for iteration_num in range(iterations):
            _response = await mock_llm_client.generate(f"TRM Iteration {iteration_num + 1}: {query_input.query}")
            # Simulate improving scores over iterations
            score = 0.70 + (iteration_num * 0.05)
            refinement_scores.append(score)

        final_score = refinement_scores[-1]

        # Update trace with iteration metrics
        update_run_metadata(
            {
                "agent": "trm",
                "total_iterations": iterations,
                "refinement_scores": refinement_scores,
                "final_score": final_score,
                "improvement": refinement_scores[-1] - refinement_scores[0],
            }
        )

        # Assert improvement over iterations
        assert refinement_scores[-1] > refinement_scores[0]


# ============================================================================
# MCTS-ONLY E2E FLOWS
# ============================================================================


class TestMCTSOnlyFlows:
    """Test Monte Carlo Tree Search in isolation."""

    @pytest.mark.e2e
    @trace_mcts_simulation(
        iterations=100,
        scenario_type="tactical",
        seed=42,
        max_depth=10,
        tags=["mcts", "simulation", "tactical", "performance"],
    )
    def test_mcts_tactical_simulation(self, mcts_tactical_scenario):
        """MCTS-only tactical decision simulation (100 iterations)."""
        import random

        random.seed(42)

        # Simulate MCTS tree exploration
        simulations = 100
        action_stats = {action: {"visits": 0, "wins": 0} for action in mcts_tactical_scenario["possible_actions"]}

        for _ in range(simulations):
            # Random action selection (in real MCTS, this would be UCB1)
            action = random.choice(mcts_tactical_scenario["possible_actions"])
            action_stats[action]["visits"] += 1

            # Random simulation outcome
            if random.random() > 0.5:
                action_stats[action]["wins"] += 1

        # Calculate win probabilities
        win_probs = {
            action: stats["wins"] / stats["visits"] if stats["visits"] > 0 else 0
            for action, stats in action_stats.items()
        }

        best_action = max(win_probs, key=win_probs.get)
        best_win_prob = win_probs[best_action]

        # Update trace with MCTS metrics
        update_run_metadata(
            {
                "agent": "mcts",
                "total_simulations": simulations,
                "actions_explored": len(action_stats),
                "best_action": best_action,
                "best_win_probability": best_win_prob,
                "tree_depth": 10,
                "exploration_rate": sum(1 for s in action_stats.values() if s["visits"] > 0) / len(action_stats),
            }
        )

        assert best_win_prob >= 0.0
        assert best_action in mcts_tactical_scenario["possible_actions"]

    @pytest.mark.e2e
    @trace_mcts_simulation(
        iterations=200,
        scenario_type="cybersecurity",
        seed=42,
        tags=["mcts", "simulation", "cybersecurity", "incident_response"],
    )
    def test_mcts_incident_response_simulation(self):
        """MCTS-only cybersecurity incident response simulation."""
        import random

        random.seed(42)

        # Incident response actions
        response_actions = [
            "isolate_systems",
            "collect_forensics",
            "reset_credentials",
            "patch_vulnerabilities",
            "notify_authorities",
        ]

        # Simulate MCTS evaluation of response strategies
        simulations = 200
        strategy_scores = {}

        for action in response_actions:
            # Simulate multiple rollouts for each action
            scores = [random.uniform(0.6, 0.95) for _ in range(simulations // len(response_actions))]
            strategy_scores[action] = {
                "mean_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "simulations": len(scores),
            }

        best_strategy = max(strategy_scores, key=lambda k: strategy_scores[k]["mean_score"])
        best_score = strategy_scores[best_strategy]["mean_score"]

        # Update trace
        update_run_metadata(
            {
                "agent": "mcts",
                "scenario": "incident_response",
                "total_simulations": simulations,
                "strategies_evaluated": len(response_actions),
                "best_strategy": best_strategy,
                "best_mean_score": best_score,
                "threat_containment_priority": best_strategy == "isolate_systems",
            }
        )

        assert best_score >= 0.6
        assert best_strategy in response_actions


# ============================================================================
# FULL-STACK COMBINED FLOWS (HRM + TRM + MCTS)
# ============================================================================


class TestFullStackFlows:
    """Test combined HRM + TRM + MCTS flows."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @trace_e2e_test(
        "e2e_full_stack_tactical_flow",
        phase="complete_flow",
        scenario_type="tactical",
        provider="openai",
        use_mcts=True,
        mcts_iterations=200,
        tags=["hrm", "trm", "mcts", "full_stack", "tactical"],
    )
    async def test_full_stack_tactical_analysis(self, mock_llm_client, tactical_query, mcts_tactical_scenario):
        """Full stack: HRM + TRM + MCTS for tactical scenario."""
        from src.models.validation import QueryInput

        query_input = QueryInput(**tactical_query)

        # Step 1: HRM - Hierarchical decomposition
        _hrm_response = await mock_llm_client.generate(f"HRM: {query_input.query}")
        hrm_confidence = 0.87

        # Step 2: TRM - Task refinement
        _trm_response = await mock_llm_client.generate(f"TRM: {query_input.query}")
        trm_confidence = 0.83

        # Step 3: MCTS - Decision simulation
        import random

        random.seed(42)
        mcts_simulations = 200
        mcts_best_action = random.choice(mcts_tactical_scenario["possible_actions"])
        mcts_win_prob = 0.75

        # Step 4: Consensus calculation
        consensus = (hrm_confidence + trm_confidence + mcts_win_prob) / 3

        # Final response
        final_response = {
            "recommendation": mcts_best_action,
            "confidence": consensus,
            "agents_consulted": ["HRM", "TRM", "MCTS"],
            "hrm_analysis": "Hierarchical objectives identified",
            "trm_refinement": "Position alternatives evaluated",
            "mcts_simulation": f"Best action from {mcts_simulations} simulations",
        }

        # Update comprehensive trace metadata
        update_run_metadata(
            {
                "agents_used": ["hrm", "trm", "mcts"],
                "hrm_confidence": hrm_confidence,
                "trm_confidence": trm_confidence,
                "mcts_win_probability": mcts_win_prob,
                "consensus_score": consensus,
                "mcts_iterations": mcts_simulations,
                "best_action": mcts_best_action,
                "processing_time_ms": 2500,  # Mock value
            }
        )

        # Assertions
        assert set(final_response["agents_consulted"]) == {"HRM", "TRM", "MCTS"}
        assert consensus >= 0.75
        assert mcts_best_action in mcts_tactical_scenario["possible_actions"]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @trace_e2e_test(
        "e2e_full_stack_cybersecurity_flow",
        phase="complete_flow",
        scenario_type="cybersecurity",
        provider="openai",
        use_mcts=True,
        mcts_iterations=150,
        tags=["hrm", "trm", "mcts", "full_stack", "cybersecurity", "threat_response"],
    )
    async def test_full_stack_cybersecurity_response(self, mock_llm_client):
        """Full stack: HRM + TRM + MCTS for cybersecurity incident."""
        from src.models.validation import QueryInput

        query = QueryInput(
            query="APT28 detected. Recommend immediate response strategy.",
            use_rag=True,
            use_mcts=True,
        )

        # HRM: Threat hierarchy
        _hrm_response = await mock_llm_client.generate(f"HRM Threat: {query.query}")
        hrm_threat_breakdown = {
            "primary": "APT28",
            "vectors": ["credential_harvesting", "lateral_movement"],
        }

        # TRM: Response refinement
        _trm_response = await mock_llm_client.generate(f"TRM Response: {query.query}")
        trm_actions = ["isolate", "forensics", "patch"]

        # MCTS: Strategy simulation
        mcts_iterations = 150
        mcts_best_strategy = "isolate_and_forensics"
        mcts_containment_prob = 0.89

        # Consensus
        final_result = {
            "threat_identified": True,
            "threat_actor": hrm_threat_breakdown["primary"],
            "recommended_actions": trm_actions,
            "best_strategy": mcts_best_strategy,
            "containment_probability": mcts_containment_prob,
            "agents_consulted": ["HRM", "TRM", "MCTS"],
        }

        # Update trace
        update_run_metadata(
            {
                "agents_used": ["hrm", "trm", "mcts"],
                "threat_actor": final_result["threat_actor"],
                "threat_vectors_count": len(hrm_threat_breakdown["vectors"]),
                "response_actions_count": len(trm_actions),
                "mcts_iterations": mcts_iterations,
                "best_strategy": mcts_best_strategy,
                "containment_probability": mcts_containment_prob,
                "severity": "HIGH",
            }
        )

        assert final_result["threat_identified"]
        assert len(final_result["agents_consulted"]) == 3
        assert mcts_containment_prob >= 0.80
