"""
Unit tests for ParallelMCTSConfig and ParallelMCTSStats.

Tests configuration validation, serialization, and backwards compatibility.
"""

import pytest

pytest.importorskip("numpy", reason="numpy required for MCTS framework")

from src.framework.mcts.parallel_mcts import ParallelMCTSConfig, ParallelMCTSStats


class TestParallelMCTSConfig:
    """Tests for ParallelMCTSConfig dataclass."""

    def test_default_values(self):
        """Test that default values are sensible."""
        config = ParallelMCTSConfig()

        assert config.num_workers == 4
        assert config.virtual_loss_value == 3.0
        assert config.adaptive_virtual_loss is True
        assert config.virtual_loss_min == 1.0
        assert config.virtual_loss_max == 10.0
        assert config.virtual_loss_increase_rate == 1.1
        assert config.virtual_loss_decrease_rate == 0.9
        assert config.collision_history_size == 100
        assert config.collision_rate_high_threshold == 0.3
        assert config.collision_rate_low_threshold == 0.1
        assert config.exploration_weight == 1.414
        assert config.seed == 42
        assert config.lock_timeout_seconds is None

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = ParallelMCTSConfig(
            num_workers=8,
            virtual_loss_value=5.0,
            adaptive_virtual_loss=False,
            collision_history_size=200,
            exploration_weight=2.0,
            seed=123,
            lock_timeout_seconds=5.0,
        )

        assert config.num_workers == 8
        assert config.virtual_loss_value == 5.0
        assert config.adaptive_virtual_loss is False
        assert config.collision_history_size == 200
        assert config.exploration_weight == 2.0
        assert config.seed == 123
        assert config.lock_timeout_seconds == 5.0

    def test_validation_passes_for_valid_config(self):
        """Test that validation passes for valid configuration."""
        config = ParallelMCTSConfig()
        # Should not raise
        config.validate()

    def test_validation_fails_for_invalid_num_workers(self):
        """Test validation fails for invalid num_workers."""
        config = ParallelMCTSConfig(num_workers=0)
        with pytest.raises(ValueError, match="num_workers must be >= 1"):
            config.validate()

    def test_validation_fails_for_negative_virtual_loss(self):
        """Test validation fails for negative virtual loss."""
        config = ParallelMCTSConfig(virtual_loss_value=-1.0)
        with pytest.raises(ValueError, match="virtual_loss_value must be >= 0"):
            config.validate()

    def test_validation_fails_for_invalid_virtual_loss_bounds(self):
        """Test validation fails when max < min."""
        config = ParallelMCTSConfig(virtual_loss_min=10.0, virtual_loss_max=5.0)
        with pytest.raises(ValueError, match="virtual_loss_max must be >= virtual_loss_min"):
            config.validate()

    def test_validation_fails_for_invalid_collision_thresholds(self):
        """Test validation fails for invalid collision thresholds."""
        # High threshold should be > low threshold
        config = ParallelMCTSConfig(
            collision_rate_low_threshold=0.5,
            collision_rate_high_threshold=0.3,
        )
        with pytest.raises(ValueError, match="collision thresholds"):
            config.validate()

    def test_validation_fails_for_invalid_collision_history_size(self):
        """Test validation fails for invalid collision history size."""
        config = ParallelMCTSConfig(collision_history_size=0)
        with pytest.raises(ValueError, match="collision_history_size must be >= 1"):
            config.validate()

    def test_validation_fails_for_negative_exploration_weight(self):
        """Test validation fails for negative exploration weight."""
        config = ParallelMCTSConfig(exploration_weight=-0.5)
        with pytest.raises(ValueError, match="exploration_weight must be >= 0"):
            config.validate()

    def test_validation_fails_for_invalid_lock_timeout(self):
        """Test validation fails for non-positive lock timeout."""
        config = ParallelMCTSConfig(lock_timeout_seconds=0)
        with pytest.raises(ValueError, match="lock_timeout_seconds must be None or > 0"):
            config.validate()

        config_negative = ParallelMCTSConfig(lock_timeout_seconds=-1.0)
        with pytest.raises(ValueError, match="lock_timeout_seconds must be None or > 0"):
            config_negative.validate()

    def test_validation_passes_for_none_lock_timeout(self):
        """Test validation passes when lock_timeout_seconds is None."""
        config = ParallelMCTSConfig(lock_timeout_seconds=None)
        config.validate()  # Should not raise

    def test_validation_passes_for_positive_lock_timeout(self):
        """Test validation passes for positive lock timeout."""
        config = ParallelMCTSConfig(lock_timeout_seconds=5.0)
        config.validate()  # Should not raise


class TestParallelMCTSStats:
    """Tests for ParallelMCTSStats dataclass."""

    def test_default_values(self):
        """Test that default values are correct."""
        stats = ParallelMCTSStats()

        assert stats.total_simulations == 0
        assert stats.total_duration == 0.0
        assert stats.thread_simulations == {}
        assert stats.collision_count == 0
        assert stats.lock_wait_time == 0.0
        assert stats.avg_tree_depth == 0.0
        assert stats.effective_parallelism == 0.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        stats = ParallelMCTSStats(
            total_simulations=100,
            total_duration=5.5,
            thread_simulations={0: 30, 1: 35, 2: 35},
            collision_count=10,
            lock_wait_time=0.5,
            avg_tree_depth=8.5,
            effective_parallelism=2.8,
        )

        result = stats.to_dict()

        assert result["total_simulations"] == 100
        assert result["total_duration"] == 5.5
        assert result["thread_simulations"] == {0: 30, 1: 35, 2: 35}
        assert result["collision_count"] == 10
        assert result["lock_wait_time"] == 0.5
        assert result["avg_tree_depth"] == 8.5
        assert result["effective_parallelism"] == 2.8

    def test_to_dict_returns_copy(self):
        """Test that to_dict returns a copy of thread_simulations."""
        stats = ParallelMCTSStats(thread_simulations={0: 10})
        result = stats.to_dict()

        # Modify the returned dict
        result["thread_simulations"][0] = 999

        # Original should be unchanged
        assert stats.thread_simulations[0] == 10


class TestParallelMCTSConfigIntegration:
    """Integration tests for config with engine."""

    def test_config_used_by_engine(self):
        """Test that config is properly used by ParallelMCTSEngine."""
        from src.framework.mcts.parallel_mcts import ParallelMCTSEngine

        config = ParallelMCTSConfig(
            num_workers=2,
            virtual_loss_value=4.0,
            exploration_weight=2.0,
            seed=99,
        )

        engine = ParallelMCTSEngine(config=config)

        assert engine.num_workers == 2
        assert engine.virtual_loss_value == 4.0
        assert engine.exploration_weight == 2.0
        assert engine.seed == 99
        assert engine._config is config

    def test_backwards_compatible_initialization(self):
        """Test that engine can be created with individual parameters."""
        from src.framework.mcts.parallel_mcts import ParallelMCTSEngine

        # Old-style initialization (deprecated but should still work)
        engine = ParallelMCTSEngine(
            num_workers=3,
            virtual_loss_value=2.5,
            exploration_weight=1.5,
            seed=42,
        )

        assert engine.num_workers == 3
        assert engine.virtual_loss_value == 2.5
        assert engine.exploration_weight == 1.5
        assert engine.seed == 42
        # Config should be created automatically
        assert engine._config is not None
        assert engine._config.num_workers == 3
