"""
Comprehensive Tests for Continual Learning Module

Tests all components of the production feedback loop:
- DataQualityValidator
- ProductionInteractionLogger
- FailurePatternAnalyzer
- ActiveLearningSelector
- IncrementalRetrainingPipeline
- FeedbackCollector (legacy)
- IncrementalTrainer (EWC)
- DriftDetector
- ABTestFramework
"""

from pathlib import Path
import tempfile
import time

import numpy as np
import pytest

from training.continual_learning import (
    ABTestFramework,
    ActiveLearningCandidate,
    ActiveLearningSelector,
    DataQualityValidator,
    DriftDetector,
    FailurePatternAnalyzer,
    FeedbackCollector,
    FeedbackSample,
    IncrementalRetrainingPipeline,
    ProductionInteraction,
    ProductionInteractionLogger,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_config(temp_dir):
    """Test configuration."""
    return {
        "enabled": True,
        "storage": str(Path(temp_dir) / "logs"),
        "use_sqlite": True,
        "use_compression": True,
        "buffer_size": 10,
        "sanitize_pii": True,
        "use_langsmith": False,
        "min_query_length": 3,
        "max_query_length": 5000,
        "min_response_length": 1,
        "max_response_length": 10000,
        "blocked_patterns": [],
        "min_cluster_size": 3,
        "similarity_threshold": 0.7,
        "selection_strategy": "uncertainty",
        "schedule": "daily",
        "min_new_samples": 5,
        "validation_split": 0.2,
        "enable_ab_test": False,
    }


@pytest.fixture
def sample_interaction():
    """Sample production interaction."""
    return ProductionInteraction(
        interaction_id="test_001",
        timestamp=time.time(),
        session_id="session_123",
        user_query="What is MCTS?",
        agent_selected="HRM",
        agent_confidence=0.85,
        response="MCTS stands for Monte Carlo Tree Search, a heuristic search algorithm...",
        user_feedback_score=4.5,
        thumbs_up_down="up",
        latency_ms=1200.0,
        tokens_used=150,
        cost=0.002,
        retrieval_quality=0.92,
    )


@pytest.fixture
def sample_interaction_with_pii():
    """Sample interaction containing PII."""
    return ProductionInteraction(
        interaction_id="test_pii_001",
        timestamp=time.time(),
        session_id="session_456",
        user_query="My email is test@example.com and phone is 555-123-4567",
        response="I can help you with that.",
        latency_ms=1000.0,
    )


# =============================================================================
# DataQualityValidator Tests
# =============================================================================


class TestDataQualityValidator:
    """Test data quality validation and PII removal."""

    def test_initialization(self, test_config):
        """Test validator initialization."""
        validator = DataQualityValidator(test_config)
        assert validator.sanitize_pii is True
        assert validator.min_query_length == 3

    def test_validate_valid_interaction(self, test_config, sample_interaction):
        """Test validation of valid interaction."""
        validator = DataQualityValidator(test_config)
        is_valid, issues = validator.validate_interaction(sample_interaction)
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_short_query(self, test_config, sample_interaction):
        """Test validation fails for short query."""
        validator = DataQualityValidator(test_config)
        sample_interaction.user_query = "Hi"
        is_valid, issues = validator.validate_interaction(sample_interaction)
        assert is_valid is False
        assert any("too short" in issue for issue in issues)

    def test_detect_pii(self, test_config):
        """Test PII detection."""
        validator = DataQualityValidator(test_config)
        text = "Contact me at user@example.com or call 555-123-4567"
        pii_found = validator._detect_pii(text)
        assert "email" in pii_found
        assert "phone" in pii_found

    def test_remove_pii(self, test_config):
        """Test PII removal."""
        validator = DataQualityValidator(test_config)
        text = "Email: user@example.com, Phone: 555-123-4567"
        sanitized = validator._remove_pii(text)
        assert "user@example.com" not in sanitized
        assert "555-123-4567" not in sanitized
        assert "[REDACTED_EMAIL]" in sanitized
        assert "[REDACTED_PHONE]" in sanitized

    def test_sanitize_interaction(self, test_config, sample_interaction_with_pii):
        """Test full interaction sanitization."""
        validator = DataQualityValidator(test_config)
        sanitized = validator.sanitize_interaction(sample_interaction_with_pii)
        assert "test@example.com" not in sanitized.user_query
        assert "555-123-4567" not in sanitized.user_query
        assert "[REDACTED_EMAIL]" in sanitized.user_query
        assert "[REDACTED_PHONE]" in sanitized.user_query

    def test_coherence_check(self, test_config):
        """Test coherence checking."""
        validator = DataQualityValidator(test_config)
        assert validator._is_coherent("This is a valid response.") is True
        assert validator._is_coherent("") is False
        assert validator._is_coherent("test test test test test test") is False  # Too repetitive


# =============================================================================
# ProductionInteractionLogger Tests
# =============================================================================


class TestProductionInteractionLogger:
    """Test production interaction logging."""

    @pytest.mark.asyncio
    async def test_initialization(self, test_config):
        """Test logger initialization."""
        logger = ProductionInteractionLogger(test_config)
        assert logger.enabled is True
        assert logger.use_sqlite is True
        assert logger.storage_path.exists()

    @pytest.mark.asyncio
    async def test_log_interaction(self, test_config, sample_interaction):
        """Test logging single interaction."""
        logger = ProductionInteractionLogger(test_config)
        success = await logger.log_interaction(sample_interaction)
        assert success is True
        assert len(logger.interaction_buffer) == 1

    @pytest.mark.asyncio
    async def test_buffer_flush(self, test_config, sample_interaction):
        """Test buffer flushing to storage."""
        test_config["buffer_size"] = 3
        logger = ProductionInteractionLogger(test_config)

        # Log enough to trigger flush
        for i in range(3):
            interaction = ProductionInteraction(
                interaction_id=f"test_{i}",
                timestamp=time.time(),
                session_id="session_test",
                user_query=f"Query {i}",
                response=f"Response {i}",
                latency_ms=1000.0,
            )
            await logger.log_interaction(interaction)

        # Buffer should be flushed
        assert len(logger.interaction_buffer) == 0

    @pytest.mark.asyncio
    async def test_query_interactions(self, test_config, sample_interaction):
        """Test querying logged interactions."""
        logger = ProductionInteractionLogger(test_config)

        # Log some interactions
        for i in range(5):
            interaction = ProductionInteraction(
                interaction_id=f"test_{i}",
                timestamp=time.time(),
                session_id="session_test",
                user_query=f"Query {i}",
                response=f"Response {i}",
                latency_ms=1000.0,
                user_feedback_score=3.0 + i * 0.5,
            )
            await logger.log_interaction(interaction)

        # Flush to database
        await logger._flush_buffer()

        # Query all
        results = logger.query_interactions(limit=10)
        assert len(results) == 5

        # Query with feedback filter
        results = logger.query_interactions(min_feedback_score=4.0, limit=10)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_statistics(self, test_config, sample_interaction):
        """Test statistics collection."""
        logger = ProductionInteractionLogger(test_config)

        for i in range(5):
            interaction = ProductionInteraction(
                interaction_id=f"test_{i}",
                timestamp=time.time(),
                session_id="session_test",
                user_query=f"Query {i}",
                response=f"Response {i}",
                latency_ms=1000.0,
                user_feedback_score=4.0,
            )
            await logger.log_interaction(interaction)

        stats = logger.get_statistics()
        assert stats["total_logged"] == 5
        assert stats["buffer_size"] == 5


# =============================================================================
# FailurePatternAnalyzer Tests
# =============================================================================


class TestFailurePatternAnalyzer:
    """Test failure pattern identification."""

    def test_initialization(self, test_config):
        """Test analyzer initialization."""
        analyzer = FailurePatternAnalyzer(test_config)
        assert analyzer.min_cluster_size == 3

    def test_identify_low_rated_responses(self, test_config):
        """Test low rating pattern detection."""
        analyzer = FailurePatternAnalyzer(test_config)

        # Create interactions with low ratings
        interactions = [
            {
                "interaction_id": f"low_{i}",
                "user_feedback_score": 2.0,
                "user_query": f"Query {i}",
            }
            for i in range(5)
        ]

        patterns = analyzer._identify_low_rated_responses(interactions)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "low_rating"
        assert patterns[0].frequency == 5

    def test_identify_poor_retrieval(self, test_config):
        """Test poor retrieval pattern detection."""
        analyzer = FailurePatternAnalyzer(test_config)

        interactions = [
            {
                "interaction_id": f"poor_{i}",
                "retrieval_failed": True,
                "user_query": f"Query {i}",
            }
            for i in range(4)
        ]

        patterns = analyzer._identify_poor_retrieval(interactions)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "poor_retrieval"

    def test_identify_slow_responses(self, test_config):
        """Test slow response pattern detection."""
        analyzer = FailurePatternAnalyzer(test_config)

        interactions = [
            {
                "interaction_id": f"slow_{i}",
                "latency_ms": 6000.0,  # >5s threshold
                "user_query": f"Query {i}",
            }
            for i in range(3)
        ]

        patterns = analyzer._identify_slow_responses(interactions)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "slow"

    def test_analyze_failures(self, test_config):
        """Test full failure analysis."""
        analyzer = FailurePatternAnalyzer(test_config)

        # Mix of failure types
        interactions = [
            {"interaction_id": f"low_{i}", "user_feedback_score": 2.0, "user_query": f"Query {i}"} for i in range(3)
        ]
        interactions.extend(
            [{"interaction_id": f"slow_{i}", "latency_ms": 6000.0, "user_query": f"Query {i}"} for i in range(3)]
        )

        patterns = analyzer.analyze_failures(interactions)
        assert len(patterns) >= 2  # At least low_rating and slow patterns

    def test_get_summary(self, test_config):
        """Test pattern summary."""
        analyzer = FailurePatternAnalyzer(test_config)

        interactions = [
            {"interaction_id": f"low_{i}", "user_feedback_score": 2.0, "user_query": f"Query {i}"} for i in range(3)
        ]

        analyzer.analyze_failures(interactions)
        summary = analyzer.get_summary()

        assert summary["total_patterns"] > 0
        assert "pattern_types" in summary


# =============================================================================
# ActiveLearningSelector Tests
# =============================================================================


class TestActiveLearningSelector:
    """Test active learning sample selection."""

    def test_initialization(self, test_config):
        """Test selector initialization."""
        selector = ActiveLearningSelector(test_config)
        assert selector.selection_strategy == "uncertainty"

    def test_uncertainty_sampling(self, test_config):
        """Test uncertainty-based sampling."""
        selector = ActiveLearningSelector(test_config)

        # High and low confidence interactions
        interactions = [
            {
                "interaction_id": f"uncertain_{i}",
                "agent_confidence": 0.5,  # Highly uncertain
                "user_query": f"Query {i}",
            }
            for i in range(5)
        ]
        interactions.extend(
            [
                {
                    "interaction_id": f"certain_{i}",
                    "agent_confidence": 0.95,  # Very certain
                    "user_query": f"Query {i}",
                }
                for i in range(5)
            ]
        )

        candidates = selector._uncertainty_sampling(interactions, budget=5)
        assert len(candidates) == 5
        # Should prioritize uncertain samples
        assert all("uncertain" in c.interaction_id for c in candidates)

    def test_failure_prioritization(self, test_config):
        """Test failure-based prioritization."""
        selector = ActiveLearningSelector(test_config)

        interactions = [
            {
                "interaction_id": f"failure_{i}",
                "error_occurred": True,
                "user_query": f"Query {i}",
            }
            for i in range(3)
        ]
        interactions.extend(
            [
                {
                    "interaction_id": f"success_{i}",
                    "error_occurred": False,
                    "user_query": f"Query {i}",
                }
                for i in range(3)
            ]
        )

        candidates = selector._failure_prioritization(interactions, budget=3)
        assert len(candidates) == 3
        assert all("failure" in c.interaction_id for c in candidates)

    def test_select_for_annotation(self, test_config):
        """Test full selection workflow."""
        selector = ActiveLearningSelector(test_config)

        interactions = [
            {
                "interaction_id": f"int_{i}",
                "agent_confidence": np.random.random(),
                "user_query": f"Query {i}",
            }
            for i in range(20)
        ]

        candidates = selector.select_for_annotation(interactions, budget=5)
        assert len(candidates) == 5
        assert all(isinstance(c, ActiveLearningCandidate) for c in candidates)


# =============================================================================
# IncrementalRetrainingPipeline Tests
# =============================================================================


class TestIncrementalRetrainingPipeline:
    """Test incremental retraining pipeline."""

    def test_initialization(self, test_config):
        """Test pipeline initialization."""
        pipeline = IncrementalRetrainingPipeline(test_config)
        assert pipeline.schedule == "daily"
        assert pipeline.min_new_samples == 5

    @pytest.mark.asyncio
    async def test_retrain_insufficient_samples(self, test_config):
        """Test retraining skips with insufficient samples."""
        pipeline = IncrementalRetrainingPipeline(test_config)

        new_data = [{"interaction_id": f"test_{i}", "user_query": f"Query {i}"} for i in range(2)]

        result = await pipeline.retrain(new_data)
        assert result["status"] == "skipped"
        assert result["reason"] == "insufficient_samples"

    @pytest.mark.asyncio
    async def test_retrain_success(self, test_config):
        """Test successful retraining."""
        pipeline = IncrementalRetrainingPipeline(test_config)

        # Sufficient samples with feedback
        new_data = [
            {
                "interaction_id": f"test_{i}",
                "user_query": f"Query {i}",
                "user_feedback_score": 4.0,
            }
            for i in range(10)
        ]

        result = await pipeline.retrain(new_data)
        assert result["status"] == "completed"
        assert len(result["steps"]) > 0

    def test_prepare_training_data(self, test_config):
        """Test training data preparation."""
        pipeline = IncrementalRetrainingPipeline(test_config)

        new_data = [
            {
                "interaction_id": f"test_{i}",
                "user_query": f"Query {i}",
                "user_feedback_score": 4.0,
            }
            for i in range(10)
        ]

        train_data, val_data = pipeline._prepare_training_data(new_data)
        assert len(train_data) > 0
        assert len(val_data) > 0
        assert len(train_data) + len(val_data) == len(new_data)

    def test_should_retrain(self, test_config):
        """Test retrain triggering logic."""
        from datetime import datetime, timedelta

        pipeline = IncrementalRetrainingPipeline(test_config)

        # Test with insufficient samples
        last_retrain = datetime.now() - timedelta(days=2)
        assert pipeline.should_retrain(last_retrain, 3) is False

        # Test with sufficient samples and time
        assert pipeline.should_retrain(last_retrain, 10) is True

        # Test with sufficient samples but recent retrain
        last_retrain = datetime.now() - timedelta(hours=1)
        assert pipeline.should_retrain(last_retrain, 10) is False


# =============================================================================
# Legacy Component Tests
# =============================================================================


class TestFeedbackCollector:
    """Test legacy feedback collector."""

    def test_initialization(self, test_config):
        """Test collector initialization."""
        collector = FeedbackCollector(test_config)
        assert collector.buffer_size > 0

    def test_add_feedback(self, test_config):
        """Test adding feedback samples."""
        collector = FeedbackCollector(test_config)
        collector.sample_rate = 1.0  # Always sample

        feedback = FeedbackSample(
            sample_id="fb_001",
            input_data={"query": "Test"},
            model_output="Response",
            user_feedback="positive",
            corrected_output=None,
            timestamp=time.time(),
        )

        collector.add_feedback(feedback)
        assert len(collector.feedback_buffer) == 1
        assert collector.statistics["positive"] == 1


class TestDriftDetector:
    """Test drift detection."""

    def test_initialization(self, test_config):
        """Test detector initialization."""
        detector = DriftDetector(test_config)
        assert detector.window_size > 0

    def test_set_reference_distribution(self, test_config):
        """Test setting reference distribution."""
        detector = DriftDetector(test_config)
        reference = np.random.randn(100, 5)
        detector.set_reference_distribution(reference)
        assert detector.reference_distribution is not None

    def test_drift_detection(self, test_config):
        """Test drift is detected."""
        detector = DriftDetector(test_config)
        detector.window_size = 100

        # Set reference
        reference = np.random.randn(100, 5)
        detector.set_reference_distribution(reference)

        # Add samples with drift
        for _i in range(100):
            sample = np.random.randn(5) + 2.0  # Shifted distribution
            detector.add_sample(sample)

        # Should detect drift
        assert len(detector.drift_history) > 0


class TestABTestFramework:
    """Test A/B testing framework."""

    def test_initialization(self, test_config):
        """Test framework initialization."""
        framework = ABTestFramework(test_config)
        assert framework.traffic_split >= 0

    def test_create_test(self, test_config):
        """Test creating A/B test."""
        framework = ABTestFramework(test_config)

        test_id = framework.create_test("test_experiment", model_a="v1", model_b="v2", metric_fn=lambda _x, _y: 1.0)

        assert test_id in framework.tests
        assert framework.tests[test_id]["status"] == "running"

    def test_assign_group(self, test_config):
        """Test group assignment."""
        framework = ABTestFramework(test_config)
        test_id = framework.create_test("test_experiment", model_a="v1", model_b="v2", metric_fn=lambda _x, _y: 1.0)

        # Should consistently assign same request to same group
        group1 = framework.assign_group(test_id, "request_123")
        group2 = framework.assign_group(test_id, "request_123")
        assert group1 == group2


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Test integration of all components."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, test_config, sample_interaction):
        """Test complete continual learning workflow."""
        # 1. Log interactions
        logger = ProductionInteractionLogger(test_config)

        interactions_data = []
        for i in range(10):
            interaction = ProductionInteraction(
                interaction_id=f"test_{i}",
                timestamp=time.time(),
                session_id="session_test",
                user_query=f"Query {i}",
                response=f"Response {i}",
                latency_ms=np.random.randint(1000, 3000),
                user_feedback_score=np.random.uniform(1, 5),
                agent_confidence=np.random.random(),
            )
            await logger.log_interaction(interaction)
            interactions_data.append(
                {
                    "interaction_id": interaction.interaction_id,
                    "user_query": interaction.user_query,
                    "user_feedback_score": interaction.user_feedback_score,
                    "latency_ms": interaction.latency_ms,
                    "agent_confidence": interaction.agent_confidence,
                }
            )

        await logger._flush_buffer()

        # 2. Analyze failures
        analyzer = FailurePatternAnalyzer(test_config)
        patterns = analyzer.analyze_failures(interactions_data)
        assert isinstance(patterns, list)

        # 3. Select samples for annotation
        selector = ActiveLearningSelector(test_config)
        candidates = selector.select_for_annotation(interactions_data, budget=3)
        assert len(candidates) <= 3

        # 4. Trigger retraining
        pipeline = IncrementalRetrainingPipeline(test_config)
        result = await pipeline.retrain(interactions_data)
        assert "status" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
