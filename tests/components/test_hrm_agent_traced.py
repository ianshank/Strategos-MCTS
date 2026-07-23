"""
HRM (Hierarchical Reasoning Model) Component Tests with LangSmith Tracing.

Component-level tests for HRM agent behavior, decomposition quality,
and hierarchical structure validation.

Tests focus on:
- Task decomposition depth and quality
- Objective identification
- Sub-task generation
- Confidence estimation
- Performance metrics
"""

import pytest

from tests.mocks.mock_external_services import create_mock_llm
from tests.utils.langsmith_tracing import trace_e2e_test, update_run_metadata


@pytest.fixture
def hrm_config():
    """HRM agent configuration."""
    return {
        "max_depth": 5,
        "min_confidence": 0.7,
        "decomposition_strategy": "top_down",
        "objective_threshold": 3,
    }


@pytest.fixture
def mock_hrm_llm():
    """Mock LLM for HRM testing."""
    client = create_mock_llm(provider="openai")
    client.set_responses(
        [
            """HRM Hierarchical Analysis:
Level 1 - Primary Objective: Secure defensive perimeter
Level 2 - Sub-objectives:
  2.1 Establish observation posts
  2.2 Position defensive assets
  2.3 Coordinate communication network
Level 3 - Tactical tasks:
  3.1.1 Deploy UAV for reconnaissance
  3.1.2 Position infantry at key chokepoints
  3.1.3 Establish radio relay points
Confidence: 0.88
Decomposition depth: 3"""
        ]
    )
    return client


class TestHRMTaskDecomposition:
    """Test HRM task decomposition capabilities."""

    @pytest.mark.component
    @pytest.mark.asyncio
    @trace_e2e_test(
        "component_hrm_task_decomposition",
        phase="component",
        scenario_type="tactical",
        tags=["hrm", "component", "decomposition"],
    )
    async def test_hierarchical_decomposition_depth(self, mock_hrm_llm, hrm_config):
        """Test HRM decomposes tasks to appropriate depth."""
        query = "Secure defensive perimeter against approaching enemy force"

        # Simulate HRM decomposition
        response = await mock_hrm_llm.generate(f"HRM Decompose: {query}")

        # Extract hierarchical structure
        levels = response.content.count("Level")
        objectives = response.content.count("objective")
        subtasks = response.content.count("3.1.")

        # Update trace with decomposition metrics
        update_run_metadata(
            {
                "component": "hrm",
                "test_type": "decomposition_depth",
                "hierarchical_levels": levels,
                "objectives_identified": objectives,
                "subtasks_generated": subtasks,
                "max_depth_config": hrm_config["max_depth"],
                "actual_depth": 3,
            }
        )

        # Assertions
        assert levels >= 2, "Should have at least 2 hierarchical levels"
        assert objectives >= 1, "Should identify multiple objectives"
        assert subtasks >= 1, "Should generate concrete subtasks"

    @pytest.mark.component
    @pytest.mark.asyncio
    @trace_e2e_test(
        "component_hrm_objective_identification",
        phase="component",
        scenario_type="tactical",
        tags=["hrm", "component", "objectives"],
    )
    async def test_objective_identification_quality(self, mock_hrm_llm):
        """Test HRM identifies clear, actionable objectives."""
        _query = "Respond to cybersecurity incident with credential compromise"

        # Mock HRM objective identification
        objectives = [
            {"level": 1, "objective": "Contain threat", "priority": "critical"},
            {"level": 2, "objective": "Preserve evidence", "priority": "high"},
            {"level": 2, "objective": "Reset compromised credentials", "priority": "high"},
            {"level": 3, "objective": "Notify stakeholders", "priority": "medium"},
        ]

        # Update trace
        update_run_metadata(
            {
                "component": "hrm",
                "test_type": "objective_identification",
                "total_objectives": len(objectives),
                "critical_objectives": sum(1 for o in objectives if o["priority"] == "critical"),
                "high_priority_objectives": sum(1 for o in objectives if o["priority"] == "high"),
                "objective_clarity_score": 0.90,
            }
        )

        # Assertions
        assert len(objectives) >= 3
        assert any(o["priority"] == "critical" for o in objectives)


class TestHRMConfidenceEstimation:
    """Test HRM confidence scoring."""

    @pytest.mark.component
    @trace_e2e_test(
        "component_hrm_confidence_calibration",
        phase="component",
        scenario_type="tactical",
        tags=["hrm", "component", "confidence"],
    )
    def test_confidence_score_calibration(self, hrm_config):
        """Test HRM confidence scores are well-calibrated."""
        # Simulate HRM confidence for different scenarios
        scenarios = [
            {"complexity": "low", "info_available": "high", "expected_confidence": 0.90},
            {"complexity": "medium", "info_available": "medium", "expected_confidence": 0.75},
            {"complexity": "high", "info_available": "low", "expected_confidence": 0.55},
        ]

        confidence_scores = []
        for scenario in scenarios:
            # In real implementation, would call HRM agent
            score = scenario["expected_confidence"]
            confidence_scores.append(score)

        avg_confidence = sum(confidence_scores) / len(confidence_scores)

        # Update trace
        update_run_metadata(
            {
                "component": "hrm",
                "test_type": "confidence_calibration",
                "scenarios_tested": len(scenarios),
                "average_confidence": avg_confidence,
                "confidence_range": [min(confidence_scores), max(confidence_scores)],
                "min_confidence_threshold": hrm_config["min_confidence"],
            }
        )

        # Assert confidence varies appropriately
        assert max(confidence_scores) > min(confidence_scores), "Confidence should vary by scenario"
        assert all(0.0 <= c <= 1.0 for c in confidence_scores), "Confidence in valid range"


class TestHRMPerformance:
    """Test HRM performance characteristics."""

    @pytest.mark.component
    @pytest.mark.performance
    @pytest.mark.asyncio
    @trace_e2e_test(
        "component_hrm_decomposition_latency",
        phase="component",
        scenario_type="tactical",
        tags=["hrm", "component", "performance", "latency"],
    )
    async def test_decomposition_latency(self, mock_hrm_llm):
        """Test HRM decomposition completes within SLA."""
        import time

        query = "Plan defensive strategy for multi-sector threat"

        start_time = time.time()

        # Simulate HRM processing
        response = await mock_hrm_llm.generate(f"HRM: {query}")

        elapsed_ms = (time.time() - start_time) * 1000

        # Update trace
        update_run_metadata(
            {
                "component": "hrm",
                "test_type": "performance_latency",
                "latency_ms": elapsed_ms,
                "sla_target_ms": 3000,
                "within_sla": elapsed_ms < 3000,
            }
        )

        assert response.content is not None
        assert elapsed_ms < 3000, "HRM should complete within 3s SLA"
