"""
Comprehensive unit tests for Component Factory module.

Tests cover:
- TrainerFactory: HRMTrainer, TRMTrainer, SelfPlayEvaluator, ReplayBuffer creation
- MetricsFactory: PerformanceMonitor, ExperimentTracker, MetricsCollector creation
- DataLoaderFactory: DABStepLoader, PRIMUSLoader, CombinedDatasetLoader creation
- ComponentRegistry: Singleton pattern, lazy initialization, cache management

Focus areas:
- Factory pattern correctness
- Configuration propagation from settings
- Singleton/caching behavior
- Lazy initialization
- Dependency injection patterns
"""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pydantic", reason="pydantic required for settings/benchmark")

# ============================================================================
# Test Constants (avoid hardcoded values in tests)
# ============================================================================

TEST_BATCH_SIZE = 16
TEST_BUFFER_CAPACITY = 5000
TEST_WINDOW_SIZE = 50
TEST_MCTS_ITERATIONS = 25
TEST_CACHE_DIR = "/tmp/test_cache"
TEST_PROJECT_NAME = "test-project"
TEST_API_KEY = "test-api-key-12345"
TEST_DEVICE = "cpu"


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_settings():
    """Create mock settings for factory tests."""
    settings = MagicMock()
    settings.MCTS_MAX_PARALLEL_ROLLOUTS = 2
    settings.MCTS_IMPL.value = "baseline"
    settings.MCTS_ITERATIONS = TEST_MCTS_ITERATIONS
    settings.S3_BUCKET = None
    settings.S3_PREFIX = "test-prefix"
    settings.WANDB_PROJECT = TEST_PROJECT_NAME
    settings.WANDB_ENTITY = "test-entity"
    settings.WANDB_MODE = "offline"
    settings.get_api_key = MagicMock(return_value=TEST_API_KEY)
    settings.get_braintrust_api_key = MagicMock(return_value=TEST_API_KEY)
    settings.get_wandb_api_key = MagicMock(return_value=TEST_API_KEY)
    return settings


@pytest.fixture
def mock_logger():
    """Create mock logger for factory tests."""
    logger = MagicMock(spec=logging.Logger)
    logger.info = MagicMock()
    logger.debug = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    return logger


@pytest.fixture
def mock_hrm_agent():
    """Create mock HRM agent for trainer tests."""
    agent = MagicMock()
    agent.parameters = MagicMock(return_value=[])
    agent.forward = MagicMock()
    return agent


@pytest.fixture
def mock_trm_agent():
    """Create mock TRM agent for trainer tests."""
    agent = MagicMock()
    agent.parameters = MagicMock(return_value=[])
    agent.forward = MagicMock()
    return agent


@pytest.fixture
def mock_optimizer():
    """Create mock PyTorch optimizer."""
    optimizer = MagicMock()
    optimizer.param_groups = [{"lr": 0.001}]
    return optimizer


@pytest.fixture
def mock_loss_fn():
    """Create mock loss function."""
    loss_fn = MagicMock()
    loss_fn.return_value = MagicMock(item=MagicMock(return_value=0.5))
    return loss_fn


@pytest.fixture
def mock_mcts():
    """Create mock NeuralMCTS instance."""
    mcts = MagicMock()
    mcts.search = MagicMock()
    return mcts


@pytest.fixture
def trainer_config():
    """Create TrainerConfig for testing."""
    from src.framework.component_factory import TrainerConfig

    return TrainerConfig(
        batch_size=TEST_BATCH_SIZE,
        num_batches=5,
        gradient_clip_norm=1.0,
        device=TEST_DEVICE,
        buffer_capacity=TEST_BUFFER_CAPACITY,
        mcts_iterations=TEST_MCTS_ITERATIONS,
    )


@pytest.fixture
def metrics_config():
    """Create MetricsConfig for testing."""
    from src.framework.component_factory import MetricsConfig

    return MetricsConfig(
        window_size=TEST_WINDOW_SIZE,
        enable_gpu_monitoring=False,
        alert_threshold_ms=500.0,
        project_name=TEST_PROJECT_NAME,
        offline_mode=True,
    )


@pytest.fixture
def data_loader_config():
    """Create DataLoaderConfig for testing."""
    from src.framework.component_factory import DataLoaderConfig

    return DataLoaderConfig(
        cache_dir=TEST_CACHE_DIR,
        max_samples=100,
        streaming=False,
    )


@pytest.fixture(autouse=True)
def clear_factory_caches():
    """Clear all factory caches before and after each test."""
    from src.framework.component_factory import (
        ComponentRegistry,
        DataLoaderFactory,
        MetricsFactory,
        TrainerFactory,
    )

    # Clear before test
    TrainerFactory.clear_singleton_cache()
    MetricsFactory.clear_singleton_cache()
    DataLoaderFactory.clear_singleton_cache()
    ComponentRegistry.reset_instance()

    yield

    # Clear after test
    TrainerFactory.clear_singleton_cache()
    MetricsFactory.clear_singleton_cache()
    DataLoaderFactory.clear_singleton_cache()
    ComponentRegistry.reset_instance()


# ============================================================================
# TrainerFactory Tests
# ============================================================================


@pytest.mark.unit
class TestTrainerFactory:
    """Test suite for TrainerFactory class."""

    def test_factory_initialization(self, mock_settings, mock_logger):
        """Test trainer factory initializes with correct parameters."""
        from src.framework.component_factory import TrainerFactory

        factory = TrainerFactory(
            settings=mock_settings,
            logger=mock_logger,
        )

        assert factory._settings is mock_settings
        assert factory._logger is mock_logger
        assert factory._config is not None

    def test_factory_initialization_default_settings(self):
        """Test trainer factory uses default settings when not provided."""
        from src.framework.component_factory import TrainerFactory

        with patch("src.framework.component_factory.trainer_factory.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.MCTS_MAX_PARALLEL_ROLLOUTS = 4
            mock_settings.MCTS_IMPL.value = "baseline"
            mock_settings.MCTS_ITERATIONS = 100
            mock_settings.S3_BUCKET = None
            mock_get_settings.return_value = mock_settings

            factory = TrainerFactory()

            mock_get_settings.assert_called_once()
            assert factory._settings is mock_settings

    def test_create_hrm_trainer_default(
        self, mock_settings, mock_logger, trainer_config, mock_hrm_agent, mock_optimizer, mock_loss_fn
    ):
        """Test creating HRM trainer with default configuration."""
        import sys

        from src.framework.component_factory import TrainerFactory

        factory = TrainerFactory(settings=mock_settings, logger=mock_logger, config=trainer_config)

        # Create mock module with mock classes
        mock_trainer_module = MagicMock()
        mock_hrm_trainer_class = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_hrm_trainer_class.return_value = mock_trainer_instance
        mock_trainer_module.HRMTrainer = mock_hrm_trainer_class
        mock_trainer_module.HRMTrainingConfig = MagicMock(return_value=MagicMock())

        # Inject mock module before factory tries to import
        with patch.dict(sys.modules, {"src.training.agent_trainer": mock_trainer_module}):
            trainer = factory.create_hrm_trainer(
                agent=mock_hrm_agent,
                optimizer=mock_optimizer,
                loss_fn=mock_loss_fn,
            )

            # Verify trainer was created
            mock_hrm_trainer_class.assert_called_once()
            assert trainer is mock_trainer_instance

            # Verify logging occurred
            mock_logger.info.assert_called()

    def test_create_trm_trainer_default(
        self, mock_settings, mock_logger, trainer_config, mock_trm_agent, mock_optimizer, mock_loss_fn
    ):
        """Test creating TRM trainer with default configuration."""
        import sys

        from src.framework.component_factory import TrainerFactory

        factory = TrainerFactory(settings=mock_settings, logger=mock_logger, config=trainer_config)

        # Create mock module with mock classes
        mock_trainer_module = MagicMock()
        mock_trm_trainer_class = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trm_trainer_class.return_value = mock_trainer_instance
        mock_trainer_module.TRMTrainer = mock_trm_trainer_class
        mock_trainer_module.TRMTrainingConfig = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"src.training.agent_trainer": mock_trainer_module}):
            trainer = factory.create_trm_trainer(
                agent=mock_trm_agent,
                optimizer=mock_optimizer,
                loss_fn=mock_loss_fn,
            )

            mock_trm_trainer_class.assert_called_once()
            assert trainer is mock_trainer_instance

    def test_create_self_play_evaluator(self, mock_settings, mock_logger, trainer_config, mock_mcts):
        """Test creating self-play evaluator."""
        import sys

        from src.framework.component_factory import TrainerFactory

        factory = TrainerFactory(settings=mock_settings, logger=mock_logger, config=trainer_config)

        mock_initial_state_fn = MagicMock()

        # Create mock module with mock classes
        mock_trainer_module = MagicMock()
        mock_evaluator_class = MagicMock()
        mock_evaluator_instance = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator_instance
        mock_config_class = MagicMock(return_value=MagicMock())
        mock_trainer_module.SelfPlayEvaluator = mock_evaluator_class
        mock_trainer_module.EvaluationConfig = mock_config_class

        with patch.dict(sys.modules, {"src.training.agent_trainer": mock_trainer_module}):
            evaluator = factory.create_self_play_evaluator(
                mcts=mock_mcts,
                initial_state_fn=mock_initial_state_fn,
            )

            mock_evaluator_class.assert_called_once()
            assert evaluator is mock_evaluator_instance

            # Verify configuration used defaults from trainer_config
            assert mock_config_class.called

    def test_create_self_play_evaluator_custom_config(self, mock_settings, mock_logger, trainer_config, mock_mcts):
        """Test creating self-play evaluator with custom configuration."""
        import sys

        from src.framework.component_factory import TrainerFactory

        factory = TrainerFactory(settings=mock_settings, logger=mock_logger, config=trainer_config)

        custom_num_games = 50
        custom_temperature = 0.5

        # Create mock module with mock classes
        mock_trainer_module = MagicMock()
        mock_evaluator_class = MagicMock()
        mock_config_class = MagicMock()
        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance
        mock_trainer_module.SelfPlayEvaluator = mock_evaluator_class
        mock_trainer_module.EvaluationConfig = mock_config_class

        with patch.dict(sys.modules, {"src.training.agent_trainer": mock_trainer_module}):
            factory.create_self_play_evaluator(
                mcts=mock_mcts,
                initial_state_fn=MagicMock(),
                num_games=custom_num_games,
                temperature=custom_temperature,
            )

            # Verify custom values were passed
            call_kwargs = mock_config_class.call_args.kwargs
            assert call_kwargs.get("num_games") == custom_num_games
            assert call_kwargs.get("temperature") == custom_temperature

    def test_create_replay_buffer_uniform(self, mock_settings, mock_logger, trainer_config):
        """Test creating uniform replay buffer."""
        import sys

        from src.framework.component_factory import TrainerFactory

        factory = TrainerFactory(settings=mock_settings, logger=mock_logger, config=trainer_config)

        # Create mock module
        mock_buffer_module = MagicMock()
        mock_buffer_class = MagicMock()
        mock_buffer_instance = MagicMock()
        mock_buffer_class.return_value = mock_buffer_instance
        mock_buffer_module.ReplayBuffer = mock_buffer_class
        mock_buffer_module.PrioritizedReplayBuffer = MagicMock()
        mock_buffer_module.AugmentedReplayBuffer = MagicMock()

        with patch.dict(sys.modules, {"src.training.replay_buffer": mock_buffer_module}):
            buffer = factory.create_replay_buffer(
                buffer_type="uniform",
                capacity=TEST_BUFFER_CAPACITY,
                use_singleton=False,
            )

            mock_buffer_class.assert_called_once_with(capacity=TEST_BUFFER_CAPACITY)
            assert buffer is mock_buffer_instance

    def test_create_replay_buffer_prioritized(self, mock_settings, mock_logger, trainer_config):
        """Test creating prioritized replay buffer."""
        import sys

        from src.framework.component_factory import TrainerFactory

        factory = TrainerFactory(settings=mock_settings, logger=mock_logger, config=trainer_config)

        custom_alpha = 0.7
        custom_beta_start = 0.5

        # Create mock module
        mock_buffer_module = MagicMock()
        mock_buffer_class = MagicMock()
        mock_buffer_instance = MagicMock()
        mock_buffer_class.return_value = mock_buffer_instance
        mock_buffer_module.ReplayBuffer = MagicMock()
        mock_buffer_module.PrioritizedReplayBuffer = mock_buffer_class
        mock_buffer_module.AugmentedReplayBuffer = MagicMock()

        with patch.dict(sys.modules, {"src.training.replay_buffer": mock_buffer_module}):
            buffer = factory.create_replay_buffer(
                buffer_type="prioritized",
                capacity=TEST_BUFFER_CAPACITY,
                alpha=custom_alpha,
                beta_start=custom_beta_start,
                use_singleton=False,
            )

            mock_buffer_class.assert_called_once()
            call_kwargs = mock_buffer_class.call_args.kwargs
            assert call_kwargs["capacity"] == TEST_BUFFER_CAPACITY
            assert call_kwargs["alpha"] == custom_alpha
            assert call_kwargs["beta_start"] == custom_beta_start
            assert buffer is mock_buffer_instance

    def test_create_replay_buffer_singleton_caching(self, mock_settings, mock_logger, trainer_config):
        """Test that replay buffers are cached when use_singleton=True."""
        import sys

        from src.framework.component_factory import TrainerFactory

        # Clear any existing cache
        TrainerFactory.clear_singleton_cache()

        factory = TrainerFactory(settings=mock_settings, logger=mock_logger, config=trainer_config)

        # Create mock module
        mock_buffer_module = MagicMock()
        mock_buffer_class = MagicMock()
        mock_buffer_instance = MagicMock()
        mock_buffer_class.return_value = mock_buffer_instance
        mock_buffer_module.ReplayBuffer = mock_buffer_class
        mock_buffer_module.PrioritizedReplayBuffer = MagicMock()
        mock_buffer_module.AugmentedReplayBuffer = MagicMock()

        with patch.dict(sys.modules, {"src.training.replay_buffer": mock_buffer_module}):
            # First call - should create new buffer
            buffer1 = factory.create_replay_buffer(
                buffer_type="uniform",
                capacity=TEST_BUFFER_CAPACITY,
                use_singleton=True,
            )

            # Second call - should return cached buffer
            buffer2 = factory.create_replay_buffer(
                buffer_type="uniform",
                capacity=TEST_BUFFER_CAPACITY,
                use_singleton=True,
            )

            # Should only be called once due to caching
            assert mock_buffer_class.call_count == 1
            assert buffer1 is buffer2

    def test_create_replay_buffer_invalid_type(self, mock_settings, mock_logger, trainer_config):
        """Test that invalid buffer type raises ValueError."""
        import sys

        from src.framework.component_factory import TrainerFactory

        factory = TrainerFactory(settings=mock_settings, logger=mock_logger, config=trainer_config)

        # Create mock module
        mock_buffer_module = MagicMock()
        mock_buffer_module.ReplayBuffer = MagicMock()
        mock_buffer_module.PrioritizedReplayBuffer = MagicMock()
        mock_buffer_module.AugmentedReplayBuffer = MagicMock()

        with patch.dict(sys.modules, {"src.training.replay_buffer": mock_buffer_module}):
            with pytest.raises(ValueError, match="Unknown buffer_type"):
                factory.create_replay_buffer(
                    buffer_type="invalid_type",
                    use_singleton=False,
                )

    def test_clear_singleton_cache(self, mock_settings, mock_logger, trainer_config):
        """Test clearing singleton cache."""
        import sys

        from src.framework.component_factory import TrainerFactory

        # Clear any existing cache
        TrainerFactory.clear_singleton_cache()

        factory = TrainerFactory(settings=mock_settings, logger=mock_logger, config=trainer_config)

        # Create mock module
        mock_buffer_module = MagicMock()
        mock_buffer_class = MagicMock()
        mock_buffer_class.return_value = MagicMock()
        mock_buffer_module.ReplayBuffer = mock_buffer_class
        mock_buffer_module.PrioritizedReplayBuffer = MagicMock()
        mock_buffer_module.AugmentedReplayBuffer = MagicMock()

        with patch.dict(sys.modules, {"src.training.replay_buffer": mock_buffer_module}):
            # Create a cached buffer
            factory.create_replay_buffer(buffer_type="uniform", use_singleton=True)
            assert mock_buffer_class.call_count == 1

            # Clear cache
            TrainerFactory.clear_singleton_cache()

            # Create again - should make a new instance
            factory.create_replay_buffer(buffer_type="uniform", use_singleton=True)
            assert mock_buffer_class.call_count == 2


# ============================================================================
# MetricsFactory Tests
# ============================================================================


@pytest.mark.unit
class TestMetricsFactory:
    """Test suite for MetricsFactory class."""

    def test_factory_initialization(self, mock_settings, mock_logger, metrics_config):
        """Test metrics factory initializes correctly."""
        from src.framework.component_factory import MetricsFactory

        factory = MetricsFactory(
            settings=mock_settings,
            logger=mock_logger,
            config=metrics_config,
        )

        assert factory._settings is mock_settings
        assert factory._logger is mock_logger
        assert factory._config is metrics_config

    def test_create_performance_monitor(self, mock_settings, mock_logger, metrics_config):
        """Test creating performance monitor."""
        import sys

        from src.framework.component_factory import MetricsFactory

        factory = MetricsFactory(settings=mock_settings, logger=mock_logger, config=metrics_config)

        # Create mock module
        mock_monitor_module = MagicMock()
        mock_monitor_class = MagicMock()
        mock_monitor_instance = MagicMock()
        mock_monitor_class.return_value = mock_monitor_instance
        mock_monitor_module.PerformanceMonitor = mock_monitor_class

        with patch.dict(sys.modules, {"src.training.performance_monitor": mock_monitor_module}):
            monitor = factory.create_performance_monitor(
                window_size=TEST_WINDOW_SIZE,
                enable_gpu_monitoring=False,
                use_singleton=False,
            )

            mock_monitor_class.assert_called_once_with(
                window_size=TEST_WINDOW_SIZE,
                enable_gpu_monitoring=False,
                alert_threshold_ms=metrics_config.alert_threshold_ms,
            )
            assert monitor is mock_monitor_instance

    def test_create_performance_monitor_singleton(self, mock_settings, mock_logger, metrics_config):
        """Test performance monitor singleton caching."""
        import sys

        from src.framework.component_factory import MetricsFactory

        # Clear cache
        MetricsFactory.clear_singleton_cache()

        factory = MetricsFactory(settings=mock_settings, logger=mock_logger, config=metrics_config)

        # Create mock module
        mock_monitor_module = MagicMock()
        mock_monitor_class = MagicMock()
        mock_monitor_instance = MagicMock()
        mock_monitor_class.return_value = mock_monitor_instance
        mock_monitor_module.PerformanceMonitor = mock_monitor_class

        with patch.dict(sys.modules, {"src.training.performance_monitor": mock_monitor_module}):
            # First call
            monitor1 = factory.create_performance_monitor(use_singleton=True)

            # Second call - should return cached
            monitor2 = factory.create_performance_monitor(use_singleton=True)

            assert mock_monitor_class.call_count == 1
            assert monitor1 is monitor2

    def test_create_experiment_tracker_braintrust(self, mock_settings, mock_logger, metrics_config):
        """Test creating Braintrust experiment tracker."""
        import sys

        from src.framework.component_factory import MetricsFactory

        factory = MetricsFactory(settings=mock_settings, logger=mock_logger, config=metrics_config)

        # Create mock module
        mock_tracker_module = MagicMock()
        mock_tracker_class = MagicMock()
        mock_tracker_instance = MagicMock()
        mock_tracker_class.return_value = mock_tracker_instance
        mock_tracker_module.BraintrustTracker = mock_tracker_class
        mock_tracker_module.WandBTracker = MagicMock()
        mock_tracker_module.UnifiedExperimentTracker = MagicMock()

        with patch.dict(sys.modules, {"src.training.experiment_tracker": mock_tracker_module}):
            tracker = factory.create_experiment_tracker(
                platform="braintrust",
                project_name=TEST_PROJECT_NAME,
                use_singleton=False,
            )

            mock_tracker_class.assert_called_once()
            call_kwargs = mock_tracker_class.call_args.kwargs
            assert call_kwargs["project_name"] == TEST_PROJECT_NAME
            assert tracker is mock_tracker_instance

    def test_create_experiment_tracker_wandb(self, mock_settings, mock_logger, metrics_config):
        """Test creating W&B experiment tracker."""
        import sys

        from src.framework.component_factory import MetricsFactory

        factory = MetricsFactory(settings=mock_settings, logger=mock_logger, config=metrics_config)

        # Create mock module
        mock_tracker_module = MagicMock()
        mock_tracker_class = MagicMock()
        mock_tracker_instance = MagicMock()
        mock_tracker_class.return_value = mock_tracker_instance
        mock_tracker_module.BraintrustTracker = MagicMock()
        mock_tracker_module.WandBTracker = mock_tracker_class
        mock_tracker_module.UnifiedExperimentTracker = MagicMock()

        with patch.dict(sys.modules, {"src.training.experiment_tracker": mock_tracker_module}):
            tracker = factory.create_experiment_tracker(
                platform="wandb",
                project_name=TEST_PROJECT_NAME,
                entity="test-entity",
                use_singleton=False,
            )

            mock_tracker_class.assert_called_once()
            call_kwargs = mock_tracker_class.call_args.kwargs
            assert call_kwargs["project_name"] == TEST_PROJECT_NAME
            assert call_kwargs["entity"] == "test-entity"
            assert tracker is mock_tracker_instance

    def test_create_experiment_tracker_unified(self, mock_settings, mock_logger, metrics_config):
        """Test creating unified experiment tracker."""
        import sys

        from src.framework.component_factory import MetricsFactory

        factory = MetricsFactory(settings=mock_settings, logger=mock_logger, config=metrics_config)

        # Create mock module
        mock_tracker_module = MagicMock()
        mock_tracker_class = MagicMock()
        mock_tracker_instance = MagicMock()
        mock_tracker_class.return_value = mock_tracker_instance
        mock_tracker_module.BraintrustTracker = MagicMock()
        mock_tracker_module.WandBTracker = MagicMock()
        mock_tracker_module.UnifiedExperimentTracker = mock_tracker_class

        with patch.dict(sys.modules, {"src.training.experiment_tracker": mock_tracker_module}):
            tracker = factory.create_experiment_tracker(
                platform="unified",
                use_singleton=False,
            )

            mock_tracker_class.assert_called_once()
            assert tracker is mock_tracker_instance

    def test_create_experiment_tracker_invalid_platform(self, mock_settings, mock_logger, metrics_config):
        """Test that invalid platform raises ValueError."""
        import sys

        from src.framework.component_factory import MetricsFactory

        factory = MetricsFactory(settings=mock_settings, logger=mock_logger, config=metrics_config)

        # Create mock module
        mock_tracker_module = MagicMock()
        mock_tracker_module.BraintrustTracker = MagicMock()
        mock_tracker_module.WandBTracker = MagicMock()
        mock_tracker_module.UnifiedExperimentTracker = MagicMock()

        with patch.dict(sys.modules, {"src.training.experiment_tracker": mock_tracker_module}):
            with pytest.raises(ValueError, match="Unknown platform"):
                factory.create_experiment_tracker(
                    platform="invalid_platform",
                    use_singleton=False,
                )

    def test_create_metrics_collector(self, mock_settings, mock_logger, metrics_config):
        """Test creating metrics collector with all components."""
        import sys

        from src.framework.component_factory import MetricsFactory

        factory = MetricsFactory(settings=mock_settings, logger=mock_logger, config=metrics_config)

        # Create mock modules
        mock_monitor_module = MagicMock()
        mock_monitor_module.PerformanceMonitor = MagicMock(return_value=MagicMock())
        mock_tracker_module = MagicMock()
        mock_tracker_module.UnifiedExperimentTracker = MagicMock(return_value=MagicMock())
        mock_tracker_module.BraintrustTracker = MagicMock()
        mock_tracker_module.WandBTracker = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "src.training.performance_monitor": mock_monitor_module,
                "src.training.experiment_tracker": mock_tracker_module,
            },
        ):
            collector = factory.create_metrics_collector(
                include_performance=True,
                include_experiment_tracking=True,
                tracking_platform="unified",
            )

            assert collector is not None
            assert collector._monitor is not None
            assert collector._tracker is not None

    def test_create_metrics_collector_performance_only(self, mock_settings, mock_logger, metrics_config):
        """Test creating metrics collector with performance only."""
        import sys

        from src.framework.component_factory import MetricsFactory

        factory = MetricsFactory(settings=mock_settings, logger=mock_logger, config=metrics_config)

        # Create mock module
        mock_monitor_module = MagicMock()
        mock_monitor_module.PerformanceMonitor = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"src.training.performance_monitor": mock_monitor_module}):
            collector = factory.create_metrics_collector(
                include_performance=True,
                include_experiment_tracking=False,
            )

            assert collector._monitor is not None
            assert collector._tracker is None

    def test_clear_singleton_cache(self, mock_settings, mock_logger, metrics_config):
        """Test clearing metrics factory singleton cache."""
        import sys

        from src.framework.component_factory import MetricsFactory

        # Clear any existing cache at start
        MetricsFactory.clear_singleton_cache()

        # Create two separate factory instances
        factory1 = MetricsFactory(settings=mock_settings, logger=mock_logger, config=metrics_config)
        factory2 = MetricsFactory(settings=mock_settings, logger=mock_logger, config=metrics_config)

        # Create mock modules for each call
        mock_monitor_module1 = MagicMock()
        mock_monitor_class1 = MagicMock()
        mock_instance1 = MagicMock()
        mock_monitor_class1.return_value = mock_instance1
        mock_monitor_module1.PerformanceMonitor = mock_monitor_class1

        with patch.dict(sys.modules, {"src.training.performance_monitor": mock_monitor_module1}):
            # Create cached monitor with first factory
            monitor1 = factory1.create_performance_monitor(use_singleton=True)
            assert mock_monitor_class1.call_count == 1
            assert monitor1 is mock_instance1

        # Clear cache
        MetricsFactory.clear_singleton_cache()

        # Create new mock for second call
        mock_monitor_module2 = MagicMock()
        mock_monitor_class2 = MagicMock()
        mock_instance2 = MagicMock()
        mock_monitor_class2.return_value = mock_instance2
        mock_monitor_module2.PerformanceMonitor = mock_monitor_class2

        with patch.dict(sys.modules, {"src.training.performance_monitor": mock_monitor_module2}):
            # Create new monitor with second factory - should create new instance
            monitor2 = factory2.create_performance_monitor(use_singleton=True)
            assert mock_monitor_class2.call_count == 1
            assert monitor2 is mock_instance2

        # Verify they are different instances
        assert monitor1 is not monitor2


# ============================================================================
# DataLoaderFactory Tests
# ============================================================================


@pytest.mark.unit
class TestDataLoaderFactory:
    """Test suite for DataLoaderFactory class."""

    def test_factory_initialization(self, mock_settings, mock_logger, data_loader_config):
        """Test data loader factory initializes correctly."""
        from src.framework.component_factory import DataLoaderFactory

        factory = DataLoaderFactory(
            settings=mock_settings,
            logger=mock_logger,
            config=data_loader_config,
        )

        assert factory._settings is mock_settings
        assert factory._logger is mock_logger
        assert factory._config is data_loader_config

    def test_create_dabstep_loader(self, mock_settings, mock_logger, data_loader_config):
        """Test creating DABStep dataset loader."""
        from src.framework.component_factory import DataLoaderFactory

        factory = DataLoaderFactory(settings=mock_settings, logger=mock_logger, config=data_loader_config)

        with patch("src.data.dataset_loader.DABStepLoader") as mock_loader_class:
            mock_loader_instance = MagicMock()
            mock_loader_class.return_value = mock_loader_instance

            loader = factory.create_dabstep_loader(
                cache_dir=TEST_CACHE_DIR,
                use_singleton=False,
            )

            mock_loader_class.assert_called_once_with(cache_dir=TEST_CACHE_DIR)
            assert loader is mock_loader_instance

    def test_create_primus_loader(self, mock_settings, mock_logger, data_loader_config):
        """Test creating PRIMUS dataset loader."""
        from src.framework.component_factory import DataLoaderFactory

        factory = DataLoaderFactory(settings=mock_settings, logger=mock_logger, config=data_loader_config)

        with patch("src.data.dataset_loader.PRIMUSLoader") as mock_loader_class:
            mock_loader_instance = MagicMock()
            mock_loader_class.return_value = mock_loader_instance

            loader = factory.create_primus_loader(
                cache_dir=TEST_CACHE_DIR,
                use_singleton=False,
            )

            mock_loader_class.assert_called_once_with(cache_dir=TEST_CACHE_DIR)
            assert loader is mock_loader_instance

    def test_create_combined_loader(self, mock_settings, mock_logger, data_loader_config):
        """Test creating combined dataset loader."""
        from src.framework.component_factory import DataLoaderFactory

        factory = DataLoaderFactory(settings=mock_settings, logger=mock_logger, config=data_loader_config)

        with patch("src.data.dataset_loader.CombinedDatasetLoader") as mock_loader_class:
            mock_loader_instance = MagicMock()
            mock_loader_class.return_value = mock_loader_instance

            loader = factory.create_combined_loader(
                cache_dir=TEST_CACHE_DIR,
                use_singleton=False,
            )

            mock_loader_class.assert_called_once_with(cache_dir=TEST_CACHE_DIR)
            assert loader is mock_loader_instance

    def test_loader_singleton_caching(self, mock_settings, mock_logger, data_loader_config):
        """Test that loaders are cached when use_singleton=True."""
        from src.framework.component_factory import DataLoaderFactory

        factory = DataLoaderFactory(settings=mock_settings, logger=mock_logger, config=data_loader_config)

        with patch("src.data.dataset_loader.DABStepLoader") as mock_loader_class:
            mock_loader_instance = MagicMock()
            mock_loader_class.return_value = mock_loader_instance

            # First call
            loader1 = factory.create_dabstep_loader(use_singleton=True)

            # Second call - should return cached
            loader2 = factory.create_dabstep_loader(use_singleton=True)

            assert mock_loader_class.call_count == 1
            assert loader1 is loader2

    def test_load_dataset_dabstep(self, mock_settings, mock_logger, data_loader_config):
        """Test load_dataset convenience method for DABStep."""
        from src.framework.component_factory import DataLoaderFactory

        factory = DataLoaderFactory(settings=mock_settings, logger=mock_logger, config=data_loader_config)

        with patch("src.data.dataset_loader.DABStepLoader") as mock_loader_class:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = [MagicMock(), MagicMock()]
            mock_loader_class.return_value = mock_loader_instance

            samples = factory.load_dataset("dabstep", split="train")

            mock_loader_instance.load.assert_called_once_with(split="train")
            assert len(samples) == 2

    def test_load_dataset_invalid_name(self, mock_settings, mock_logger, data_loader_config):
        """Test that invalid dataset name raises ValueError."""
        from src.framework.component_factory import DataLoaderFactory

        factory = DataLoaderFactory(settings=mock_settings, logger=mock_logger, config=data_loader_config)

        with pytest.raises(ValueError, match="Unknown dataset_name"):
            factory.load_dataset("invalid_dataset")

    def test_clear_singleton_cache(self, mock_settings, mock_logger, data_loader_config):
        """Test clearing data loader factory singleton cache."""
        from src.framework.component_factory import DataLoaderFactory

        factory = DataLoaderFactory(settings=mock_settings, logger=mock_logger, config=data_loader_config)

        with patch("src.data.dataset_loader.DABStepLoader") as mock_loader_class:
            mock_loader_class.return_value = MagicMock()

            # Create cached loader
            factory.create_dabstep_loader(use_singleton=True)
            assert mock_loader_class.call_count == 1

            # Clear cache
            DataLoaderFactory.clear_singleton_cache()

            # Create again
            factory.create_dabstep_loader(use_singleton=True)
            assert mock_loader_class.call_count == 2


# ============================================================================
# ComponentRegistry Tests
# ============================================================================


@pytest.mark.unit
class TestComponentRegistry:
    """Test suite for ComponentRegistry class."""

    def test_singleton_pattern(self, mock_settings):
        """Test that ComponentRegistry follows singleton pattern."""
        from src.framework.component_factory import ComponentRegistry

        # Reset to ensure clean state
        ComponentRegistry.reset_instance()

        registry1 = ComponentRegistry.get_instance(settings=mock_settings)
        registry2 = ComponentRegistry.get_instance()

        assert registry1 is registry2

    def test_singleton_thread_safety(self, mock_settings):
        """Test singleton pattern is thread-safe."""
        from src.framework.component_factory import ComponentRegistry

        ComponentRegistry.reset_instance()

        instances = []
        errors = []

        def get_instance():
            try:
                instance = ComponentRegistry.get_instance(settings=mock_settings)
                instances.append(instance)
            except Exception as e:
                errors.append(e)

        # Create multiple threads trying to get instance simultaneously
        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(instances) == 10
        # All instances should be the same
        assert all(inst is instances[0] for inst in instances)

    def test_factory_lazy_initialization(self, mock_settings, mock_logger):
        """Test that factories are lazily initialized."""
        from src.framework.component_factory import ComponentRegistry

        registry = ComponentRegistry(settings=mock_settings, logger=mock_logger)

        # Initially, factories should be None
        assert registry._trainer_factory is None
        assert registry._metrics_factory is None
        assert registry._data_loader_factory is None

        # Access trainers property - should initialize trainer factory
        trainers = registry.trainers
        assert registry._trainer_factory is not None
        assert trainers is registry._trainer_factory

        # Other factories still None
        assert registry._metrics_factory is None
        assert registry._data_loader_factory is None

        # Access metrics property
        metrics = registry.metrics
        assert registry._metrics_factory is not None
        assert metrics is registry._metrics_factory

        # Access data_loaders property
        data_loaders = registry.data_loaders
        assert registry._data_loader_factory is not None
        assert data_loaders is registry._data_loader_factory

    def test_factory_reuse(self, mock_settings, mock_logger):
        """Test that factory instances are reused after initialization."""
        from src.framework.component_factory import ComponentRegistry

        registry = ComponentRegistry(settings=mock_settings, logger=mock_logger)

        # Get trainers twice
        trainers1 = registry.trainers
        trainers2 = registry.trainers

        assert trainers1 is trainers2

    def test_clear_caches(self, mock_settings, mock_logger):
        """Test clearing all caches across factories."""
        import sys

        from src.framework.component_factory import ComponentRegistry

        registry = ComponentRegistry(settings=mock_settings, logger=mock_logger)

        # Initialize all factories
        _ = registry.trainers
        _ = registry.metrics
        _ = registry.data_loaders

        # Create mock modules
        mock_buffer_module = MagicMock()
        mock_buffer_module.ReplayBuffer = MagicMock(return_value=MagicMock())
        mock_buffer_module.PrioritizedReplayBuffer = MagicMock()
        mock_buffer_module.AugmentedReplayBuffer = MagicMock()

        mock_monitor_module = MagicMock()
        mock_monitor_module.PerformanceMonitor = MagicMock(return_value=MagicMock())

        mock_loader_module = MagicMock()
        mock_loader_module.DABStepLoader = MagicMock(return_value=MagicMock())
        mock_loader_module.PRIMUSLoader = MagicMock()
        mock_loader_module.CombinedDatasetLoader = MagicMock()

        # Create some cached components
        with patch.dict(
            sys.modules,
            {
                "src.training.replay_buffer": mock_buffer_module,
                "src.training.performance_monitor": mock_monitor_module,
                "src.data.dataset_loader": mock_loader_module,
            },
        ):
            registry.trainers.create_replay_buffer(use_singleton=True)
            registry.metrics.create_performance_monitor(use_singleton=True)
            registry.data_loaders.create_dabstep_loader(use_singleton=True)

            # Clear all caches
            registry.clear_caches()

        # Verify logging occurred
        mock_logger.info.assert_called()

    def test_reset_instance(self, mock_settings):
        """Test resetting singleton instance."""
        from src.framework.component_factory import ComponentRegistry

        registry1 = ComponentRegistry.get_instance(settings=mock_settings)

        ComponentRegistry.reset_instance()

        registry2 = ComponentRegistry.get_instance(settings=mock_settings)

        assert registry1 is not registry2

    def test_settings_propagation(self, mock_settings, mock_logger):
        """Test that settings are propagated to child factories."""
        from src.framework.component_factory import ComponentRegistry

        registry = ComponentRegistry(settings=mock_settings, logger=mock_logger)

        assert registry.trainers._settings is mock_settings
        assert registry.metrics._settings is mock_settings
        assert registry.data_loaders._settings is mock_settings


# ============================================================================
# Configuration Classes Tests
# ============================================================================


@pytest.mark.unit
class TestTrainerConfig:
    """Test suite for TrainerConfig dataclass."""

    def test_default_values(self):
        """Test TrainerConfig default values."""
        from src.framework.component_factory import TrainerConfig

        config = TrainerConfig()

        assert config.batch_size == 32
        assert config.num_batches == 10
        assert config.gradient_clip_norm == 1.0
        assert config.use_mixed_precision is False
        assert config.device == "cpu"

    def test_from_settings(self, mock_settings):
        """Test creating TrainerConfig from settings."""
        from src.framework.component_factory import TrainerConfig

        config = TrainerConfig.from_settings(mock_settings)

        # batch_size = MCTS_MAX_PARALLEL_ROLLOUTS * 8
        assert config.batch_size == mock_settings.MCTS_MAX_PARALLEL_ROLLOUTS * 8
        assert config.mcts_iterations == mock_settings.MCTS_ITERATIONS

    def test_custom_values(self):
        """Test TrainerConfig with custom values."""
        from src.framework.component_factory import TrainerConfig

        config = TrainerConfig(
            batch_size=64,
            num_batches=20,
            gradient_clip_norm=0.5,
            device="cuda",
        )

        assert config.batch_size == 64
        assert config.num_batches == 20
        assert config.gradient_clip_norm == 0.5
        assert config.device == "cuda"


@pytest.mark.unit
class TestMetricsConfig:
    """Test suite for MetricsConfig dataclass."""

    def test_default_values(self):
        """Test MetricsConfig default values."""
        from src.framework.component_factory import MetricsConfig

        config = MetricsConfig()

        assert config.window_size == 100
        assert config.enable_gpu_monitoring is True
        assert config.alert_threshold_ms == 1000.0
        assert config.project_name == "langgraph-mcts"

    def test_from_settings(self, mock_settings):
        """Test creating MetricsConfig from settings."""
        from src.framework.component_factory import MetricsConfig

        config = MetricsConfig.from_settings(mock_settings)

        assert config.project_name == mock_settings.WANDB_PROJECT
        assert config.wandb_entity == mock_settings.WANDB_ENTITY


@pytest.mark.unit
class TestDataLoaderConfig:
    """Test suite for DataLoaderConfig dataclass."""

    def test_default_values(self):
        """Test DataLoaderConfig default values."""
        from src.framework.component_factory import DataLoaderConfig

        config = DataLoaderConfig()

        assert config.cache_dir is None
        assert config.max_samples is None
        assert config.streaming is True
        assert config.include_instruct is True
        assert config.batch_size == 32

    def test_from_settings(self, mock_settings):
        """Test creating DataLoaderConfig from settings."""
        from src.framework.component_factory import DataLoaderConfig

        config = DataLoaderConfig.from_settings(mock_settings)

        assert config.cache_dir is not None
        assert config.batch_size == mock_settings.MCTS_MAX_PARALLEL_ROLLOUTS * 8


# ============================================================================
# MetricsCollector Tests
# ============================================================================


@pytest.mark.unit
class TestMetricsCollector:
    """Test suite for MetricsCollector class."""

    def test_initialization(self, mock_logger):
        """Test MetricsCollector initialization."""
        from src.framework.component_factory import MetricsCollector

        mock_monitor = MagicMock()
        mock_tracker = MagicMock()

        collector = MetricsCollector(
            performance_monitor=mock_monitor,
            experiment_tracker=mock_tracker,
            logger=mock_logger,
        )

        assert collector._monitor is mock_monitor
        assert collector._tracker is mock_tracker
        assert collector._logger is mock_logger

    def test_log_metric(self, mock_logger):
        """Test logging a single metric."""
        from src.framework.component_factory import MetricsCollector

        mock_tracker = MagicMock()
        mock_tracker.log = MagicMock()

        collector = MetricsCollector(
            experiment_tracker=mock_tracker,
            logger=mock_logger,
        )

        collector.log_metric("test_metric", 0.75, step=10)

        # Tracker should be called
        assert mock_tracker.log.called or mock_tracker.log_metric.called

    def test_log_inference(self, mock_logger):
        """Test logging inference timing."""
        from src.framework.component_factory import MetricsCollector

        mock_monitor = MagicMock()

        collector = MetricsCollector(
            performance_monitor=mock_monitor,
            logger=mock_logger,
        )

        collector.log_inference(total_time_ms=45.5)

        mock_monitor.log_inference.assert_called_once_with(45.5)

    def test_get_stats(self, mock_logger):
        """Test getting collected statistics."""
        from src.framework.component_factory import MetricsCollector

        mock_monitor = MagicMock()
        mock_monitor.get_stats.return_value = {"latency_mean": 50.0}
        mock_monitor.get_current_memory.return_value = {"cpu": 1.5}

        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = {"total_steps": 100}

        collector = MetricsCollector(
            performance_monitor=mock_monitor,
            experiment_tracker=mock_tracker,
            logger=mock_logger,
        )

        stats = collector.get_stats()

        assert "performance" in stats
        assert "memory" in stats
        assert "experiment" in stats

    def test_get_stats_no_components(self, mock_logger):
        """Test getting stats when no components are configured."""
        from src.framework.component_factory import MetricsCollector

        collector = MetricsCollector(logger=mock_logger)

        stats = collector.get_stats()

        assert stats == {}


# ============================================================================
# Convenience Functions Tests
# ============================================================================


@pytest.mark.unit
class TestConvenienceFunctions:
    """Test suite for module-level convenience functions."""

    def test_create_trainer_factory(self, mock_settings):
        """Test create_trainer_factory convenience function."""
        from src.framework.component_factory import create_trainer_factory

        with patch("src.framework.component_factory.registry.get_settings", return_value=mock_settings):
            factory = create_trainer_factory(settings=mock_settings)

            assert factory is not None
            assert factory._settings is mock_settings

    def test_create_metrics_factory(self, mock_settings):
        """Test create_metrics_factory convenience function."""
        from src.framework.component_factory import create_metrics_factory

        with patch("src.framework.component_factory.registry.get_settings", return_value=mock_settings):
            factory = create_metrics_factory(settings=mock_settings)

            assert factory is not None
            assert factory._settings is mock_settings

    def test_create_data_loader_factory(self, mock_settings):
        """Test create_data_loader_factory convenience function."""
        from src.framework.component_factory import create_data_loader_factory

        with patch("src.framework.component_factory.registry.get_settings", return_value=mock_settings):
            factory = create_data_loader_factory(settings=mock_settings)

            assert factory is not None
            assert factory._settings is mock_settings

    def test_get_component_registry(self, mock_settings):
        """Test get_component_registry convenience function."""
        from src.framework.component_factory import get_component_registry

        with patch("src.framework.component_factory.registry.get_settings", return_value=mock_settings):
            registry = get_component_registry(settings=mock_settings)

            assert registry is not None


# ============================================================================
# Protocol Tests
# ============================================================================


@pytest.mark.unit
class TestProtocols:
    """Test suite for protocol definitions."""

    def test_trainer_protocol(self):
        """Test TrainerProtocol can be used for type checking."""

        # Create a mock that implements the protocol
        mock_trainer = MagicMock()
        mock_trainer.train_step = MagicMock()
        mock_trainer.train_epoch = MagicMock()

        # This should not raise - duck typing with protocol
        assert hasattr(mock_trainer, "train_step")
        assert hasattr(mock_trainer, "train_epoch")

    def test_metrics_collector_protocol(self):
        """Test MetricsCollectorProtocol can be used for type checking."""

        mock_collector = MagicMock()
        mock_collector.log_metric = MagicMock()
        mock_collector.get_stats = MagicMock()

        assert hasattr(mock_collector, "log_metric")
        assert hasattr(mock_collector, "get_stats")

    def test_data_loader_protocol(self):
        """Test DataLoaderProtocol can be used for type checking."""

        mock_loader = MagicMock()
        mock_loader.load = MagicMock()
        mock_loader.get_statistics = MagicMock()

        assert hasattr(mock_loader, "load")
        assert hasattr(mock_loader, "get_statistics")


# ============================================================================
# Additional Coverage Tests - Augmented Buffer, Singleton Hits, DataLoader
# convenience methods, MetricsCollector branches, Protocol isinstance checks
# ============================================================================


@pytest.mark.unit
class TestTrainerFactoryAugmentedBuffer:
    """Tests for augmented replay buffer creation."""

    def test_create_replay_buffer_augmented(self, mock_settings, mock_logger, trainer_config):
        """Test creating augmented replay buffer with augmentation function."""
        import sys

        from src.framework.component_factory import TrainerFactory

        factory = TrainerFactory(settings=mock_settings, logger=mock_logger, config=trainer_config)

        mock_augmentation_fn = MagicMock()

        mock_buffer_module = MagicMock()
        mock_augmented_class = MagicMock()
        mock_augmented_instance = MagicMock()
        mock_augmented_class.return_value = mock_augmented_instance
        mock_buffer_module.ReplayBuffer = MagicMock()
        mock_buffer_module.PrioritizedReplayBuffer = MagicMock()
        mock_buffer_module.AugmentedReplayBuffer = mock_augmented_class

        with patch.dict(sys.modules, {"src.training.replay_buffer": mock_buffer_module}):
            buffer = factory.create_replay_buffer(
                buffer_type="augmented",
                capacity=TEST_BUFFER_CAPACITY,
                augmentation_fn=mock_augmentation_fn,
                use_singleton=False,
            )

            mock_augmented_class.assert_called_once_with(
                capacity=TEST_BUFFER_CAPACITY,
                augmentation_fn=mock_augmentation_fn,
            )
            assert buffer is mock_augmented_instance


@pytest.mark.unit
class TestExperimentTrackerSingletonCacheHit:
    """Tests for experiment tracker singleton cache hit path."""

    def test_experiment_tracker_singleton_caching(self, mock_settings, mock_logger, metrics_config):
        """Test that experiment trackers are cached and returned on second call."""
        import sys

        from src.framework.component_factory import MetricsFactory

        MetricsFactory.clear_singleton_cache()

        factory = MetricsFactory(settings=mock_settings, logger=mock_logger, config=metrics_config)

        mock_tracker_module = MagicMock()
        mock_bt_class = MagicMock()
        mock_bt_instance = MagicMock()
        mock_bt_class.return_value = mock_bt_instance
        mock_tracker_module.BraintrustTracker = mock_bt_class
        mock_tracker_module.WandBTracker = MagicMock()
        mock_tracker_module.UnifiedExperimentTracker = MagicMock()

        with patch.dict(sys.modules, {"src.training.experiment_tracker": mock_tracker_module}):
            # First call creates new tracker
            tracker1 = factory.create_experiment_tracker(
                platform="braintrust",
                project_name=TEST_PROJECT_NAME,
                use_singleton=True,
            )

            # Second call should return cached tracker
            tracker2 = factory.create_experiment_tracker(
                platform="braintrust",
                project_name=TEST_PROJECT_NAME,
                use_singleton=True,
            )

            assert mock_bt_class.call_count == 1
            assert tracker1 is tracker2


@pytest.mark.unit
class TestMetricsCollectorBranches:
    """Tests for MetricsCollector branch coverage."""

    def test_log_metric_with_log_metric_attr(self, mock_logger):
        """Test log_metric when tracker has log_metric attribute."""
        from src.framework.component_factory import MetricsCollector

        mock_tracker = MagicMock()
        mock_tracker.log_metric = MagicMock()
        # Ensure hasattr(mock_tracker, "log_metric") returns True
        collector = MetricsCollector(experiment_tracker=mock_tracker, logger=mock_logger)

        collector.log_metric("loss", 0.5, step=1)

        mock_tracker.log_metric.assert_called_once_with("loss", 0.5, step=1)

    def test_log_metric_with_log_attr_only(self, mock_logger):
        """Test log_metric when tracker only has log attribute (no log_metric)."""
        from src.framework.component_factory import MetricsCollector

        mock_tracker = MagicMock(spec=[])
        mock_tracker.log = MagicMock()
        # Remove log_metric so hasattr returns False
        collector = MetricsCollector(experiment_tracker=mock_tracker, logger=mock_logger)

        collector.log_metric("loss", 0.5, step=1)

        mock_tracker.log.assert_called_once_with({"loss": 0.5}, step=1)

    def test_log_metric_no_tracker(self, mock_logger):
        """Test log_metric when no tracker is configured."""
        from src.framework.component_factory import MetricsCollector

        collector = MetricsCollector(logger=mock_logger)

        # Should not raise
        collector.log_metric("loss", 0.5, step=1)

    def test_log_training_step(self, mock_logger):
        """Test log_training_step with tracker that has log_training_step."""
        import sys

        from src.framework.component_factory import MetricsCollector

        mock_tracker = MagicMock()
        mock_tracker.log_training_step = MagicMock()
        mock_monitor = MagicMock()

        collector = MetricsCollector(
            performance_monitor=mock_monitor,
            experiment_tracker=mock_tracker,
            logger=mock_logger,
        )

        mock_metrics_module = MagicMock()
        mock_training_metrics_class = MagicMock()
        mock_training_metrics_instance = MagicMock()
        mock_training_metrics_class.return_value = mock_training_metrics_instance
        mock_metrics_module.TrainingMetrics = mock_training_metrics_class

        with patch.dict(sys.modules, {"src.training.experiment_tracker": mock_metrics_module}):
            collector.log_training_step(epoch=1, loss=0.5, accuracy=0.9, learning_rate=0.001, step=10)

        mock_tracker.log_training_step.assert_called_once()
        mock_monitor.log_loss.assert_called_once()

    def test_log_training_step_with_log_metrics(self, mock_logger):
        """Test log_training_step when tracker has log_metrics instead of log_training_step."""
        import sys

        from src.framework.component_factory import MetricsCollector

        # Create tracker without log_training_step but with log_metrics
        mock_tracker = MagicMock(spec=[])
        mock_tracker.log_metrics = MagicMock()

        collector = MetricsCollector(
            experiment_tracker=mock_tracker,
            logger=mock_logger,
        )

        mock_metrics_module = MagicMock()
        mock_training_metrics_class = MagicMock()
        mock_training_metrics_instance = MagicMock()
        mock_training_metrics_class.return_value = mock_training_metrics_instance
        mock_metrics_module.TrainingMetrics = mock_training_metrics_class

        with patch.dict(sys.modules, {"src.training.experiment_tracker": mock_metrics_module}):
            collector.log_training_step(epoch=1, loss=0.5)

        mock_tracker.log_metrics.assert_called_once()

    def test_log_training_step_no_tracker_no_monitor(self, mock_logger):
        """Test log_training_step with no tracker and no monitor."""
        import sys

        from src.framework.component_factory import MetricsCollector

        collector = MetricsCollector(logger=mock_logger)

        mock_metrics_module = MagicMock()
        mock_metrics_module.TrainingMetrics = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"src.training.experiment_tracker": mock_metrics_module}):
            # Should not raise
            collector.log_training_step(epoch=1, loss=0.5)

    def test_log_inference_no_monitor(self, mock_logger):
        """Test log_inference when no monitor is configured."""
        from src.framework.component_factory import MetricsCollector

        collector = MetricsCollector(logger=mock_logger)

        # Should not raise
        collector.log_inference(total_time_ms=45.5)

    def test_log_memory(self, mock_logger):
        """Test log_memory with monitor."""
        from src.framework.component_factory import MetricsCollector

        mock_monitor = MagicMock()

        collector = MetricsCollector(
            performance_monitor=mock_monitor,
            logger=mock_logger,
        )

        collector.log_memory()

        mock_monitor.log_memory.assert_called_once()

    def test_log_memory_no_monitor(self, mock_logger):
        """Test log_memory when no monitor is configured."""
        from src.framework.component_factory import MetricsCollector

        collector = MetricsCollector(logger=mock_logger)

        # Should not raise
        collector.log_memory()

    def test_print_summary(self, mock_logger):
        """Test print_summary with monitor."""
        from src.framework.component_factory import MetricsCollector

        mock_monitor = MagicMock()

        collector = MetricsCollector(
            performance_monitor=mock_monitor,
            logger=mock_logger,
        )

        collector.print_summary()

        mock_monitor.print_summary.assert_called_once()

    def test_print_summary_no_monitor(self, mock_logger):
        """Test print_summary when no monitor is configured."""
        from src.framework.component_factory import MetricsCollector

        collector = MetricsCollector(logger=mock_logger)

        # Should not raise
        collector.print_summary()

    def test_get_stats_no_tracker_summary(self, mock_logger):
        """Test get_stats when tracker has no get_summary method."""
        from src.framework.component_factory import MetricsCollector

        mock_tracker = MagicMock(spec=[])
        # No get_summary attribute

        collector = MetricsCollector(
            experiment_tracker=mock_tracker,
            logger=mock_logger,
        )

        stats = collector.get_stats()

        assert "experiment" not in stats


@pytest.mark.unit
class TestDataLoaderFactorySingletonCacheHits:
    """Tests for data loader singleton cache hit paths."""

    def test_primus_loader_singleton_caching(self, mock_settings, mock_logger, data_loader_config):
        """Test that PRIMUS loader is cached when use_singleton=True."""
        from src.framework.component_factory import DataLoaderFactory

        DataLoaderFactory.clear_singleton_cache()

        factory = DataLoaderFactory(settings=mock_settings, logger=mock_logger, config=data_loader_config)

        with patch("src.data.dataset_loader.PRIMUSLoader") as mock_loader_class:
            mock_loader_instance = MagicMock()
            mock_loader_class.return_value = mock_loader_instance

            loader1 = factory.create_primus_loader(use_singleton=True)
            loader2 = factory.create_primus_loader(use_singleton=True)

            assert mock_loader_class.call_count == 1
            assert loader1 is loader2

    def test_combined_loader_singleton_caching(self, mock_settings, mock_logger, data_loader_config):
        """Test that combined loader is cached when use_singleton=True."""
        from src.framework.component_factory import DataLoaderFactory

        DataLoaderFactory.clear_singleton_cache()

        factory = DataLoaderFactory(settings=mock_settings, logger=mock_logger, config=data_loader_config)

        with patch("src.data.dataset_loader.CombinedDatasetLoader") as mock_loader_class:
            mock_loader_instance = MagicMock()
            mock_loader_class.return_value = mock_loader_instance

            loader1 = factory.create_combined_loader(use_singleton=True)
            loader2 = factory.create_combined_loader(use_singleton=True)

            assert mock_loader_class.call_count == 1
            assert loader1 is loader2


@pytest.mark.unit
class TestDataLoaderFactoryLoadDatasetBranches:
    """Tests for load_dataset convenience method - all dataset types."""

    def test_load_dataset_primus_seed(self, mock_settings, mock_logger, data_loader_config):
        """Test load_dataset for primus_seed."""
        from src.framework.component_factory import DataLoaderFactory

        factory = DataLoaderFactory(settings=mock_settings, logger=mock_logger, config=data_loader_config)

        with patch("src.data.dataset_loader.PRIMUSLoader") as mock_loader_class:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load_seed.return_value = [MagicMock(), MagicMock(), MagicMock()]
            mock_loader_class.return_value = mock_loader_instance

            samples = factory.load_dataset("primus_seed", max_samples=50)

            mock_loader_instance.load_seed.assert_called_once_with(max_samples=50)
            assert len(samples) == 3

    def test_load_dataset_primus_instruct(self, mock_settings, mock_logger, data_loader_config):
        """Test load_dataset for primus_instruct."""
        from src.framework.component_factory import DataLoaderFactory

        factory = DataLoaderFactory(settings=mock_settings, logger=mock_logger, config=data_loader_config)

        with patch("src.data.dataset_loader.PRIMUSLoader") as mock_loader_class:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load_instruct.return_value = [MagicMock()]
            mock_loader_class.return_value = mock_loader_instance

            samples = factory.load_dataset("primus_instruct")

            mock_loader_instance.load_instruct.assert_called_once()
            assert len(samples) == 1

    def test_load_dataset_combined(self, mock_settings, mock_logger, data_loader_config):
        """Test load_dataset for combined datasets."""
        from src.framework.component_factory import DataLoaderFactory

        factory = DataLoaderFactory(settings=mock_settings, logger=mock_logger, config=data_loader_config)

        with patch("src.data.dataset_loader.CombinedDatasetLoader") as mock_loader_class:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load_all.return_value = [MagicMock(), MagicMock()]
            mock_loader_class.return_value = mock_loader_instance

            samples = factory.load_dataset("combined", max_samples=100)

            mock_loader_instance.load_all.assert_called_once_with(
                primus_max_samples=100,
                include_instruct=data_loader_config.include_instruct,
            )
            assert len(samples) == 2

    def test_load_dataset_uses_config_max_samples(self, mock_settings, mock_logger):
        """Test load_dataset falls back to config max_samples when not provided."""
        from src.framework.component_factory import DataLoaderConfig, DataLoaderFactory

        config = DataLoaderConfig(
            cache_dir=TEST_CACHE_DIR,
            max_samples=200,
        )
        factory = DataLoaderFactory(settings=mock_settings, logger=mock_logger, config=config)

        with patch("src.data.dataset_loader.PRIMUSLoader") as mock_loader_class:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load_seed.return_value = []
            mock_loader_class.return_value = mock_loader_instance

            factory.load_dataset("primus_seed")

            mock_loader_instance.load_seed.assert_called_once_with(max_samples=200)


@pytest.mark.unit
class TestProtocolIsinstance:
    """Tests for runtime_checkable protocol isinstance checks."""

    def test_component_protocol_isinstance(self):
        """Test ComponentProtocol isinstance check."""
        from src.framework.component_factory import ComponentProtocol

        class MyComponent:
            def __init__(self, **kwargs):
                pass

        assert isinstance(MyComponent(), ComponentProtocol)

    def test_trainer_protocol_isinstance(self):
        """Test TrainerProtocol isinstance check."""
        from src.framework.component_factory import TrainerProtocol

        class MyTrainer:
            async def train_step(self, *args, **kwargs):
                pass

            async def train_epoch(self, data_loader):
                return {}

        assert isinstance(MyTrainer(), TrainerProtocol)

    def test_metrics_collector_protocol_isinstance(self):
        """Test MetricsCollectorProtocol isinstance check."""
        from src.framework.component_factory import MetricsCollectorProtocol

        class MyCollector:
            def log_metric(self, name, value, **kwargs):
                pass

            def get_stats(self):
                return {}

        assert isinstance(MyCollector(), MetricsCollectorProtocol)

    def test_data_loader_protocol_isinstance(self):
        """Test DataLoaderProtocol isinstance check."""
        from src.framework.component_factory import DataLoaderProtocol

        class MyLoader:
            def load(self, split, **kwargs):
                return []

            def get_statistics(self):
                return {}

        assert isinstance(MyLoader(), DataLoaderProtocol)


@pytest.mark.unit
class TestTrainerConfigFromSettingsEdgeCases:
    """Additional edge-case tests for config from_settings."""

    def test_trainer_config_from_settings_with_s3_bucket(self):
        """Test TrainerConfig.from_settings when S3_BUCKET is set."""
        from src.framework.component_factory import TrainerConfig

        mock_settings = MagicMock()
        mock_settings.MCTS_MAX_PARALLEL_ROLLOUTS = 4
        mock_settings.MCTS_IMPL.value = "neural"
        mock_settings.MCTS_ITERATIONS = 200
        mock_settings.S3_BUCKET = "my-bucket"
        mock_settings.S3_PREFIX = "experiment-1"

        config = TrainerConfig.from_settings(mock_settings)

        assert config.device == "cuda"
        assert config.checkpoint_dir == "experiment-1"

    def test_trainer_config_from_settings_baseline_no_s3(self):
        """Test TrainerConfig.from_settings with baseline impl and no S3."""
        from src.framework.component_factory import TrainerConfig

        mock_settings = MagicMock()
        mock_settings.MCTS_MAX_PARALLEL_ROLLOUTS = 2
        mock_settings.MCTS_IMPL.value = "baseline"
        mock_settings.MCTS_ITERATIONS = 50
        mock_settings.S3_BUCKET = None
        mock_settings.S3_PREFIX = "unused"

        config = TrainerConfig.from_settings(mock_settings)

        assert config.device == "cpu"
        assert config.checkpoint_dir == "checkpoints"

    def test_metrics_config_from_settings_online_mode(self):
        """Test MetricsConfig.from_settings with online wandb mode."""
        from src.framework.component_factory import MetricsConfig

        mock_settings = MagicMock()
        mock_settings.WANDB_PROJECT = "online-project"
        mock_settings.WANDB_ENTITY = None
        mock_settings.WANDB_MODE = "online"

        config = MetricsConfig.from_settings(mock_settings)

        assert config.offline_mode is False
        assert config.wandb_entity is None


@pytest.mark.unit
class TestHRMTrainerCustomConfig:
    """Test HRM trainer creation with custom parameter overrides."""

    def test_create_hrm_trainer_all_custom_params(
        self, mock_settings, mock_logger, trainer_config, mock_hrm_agent, mock_optimizer, mock_loss_fn
    ):
        """Test creating HRM trainer with all custom overrides."""
        import sys

        from src.framework.component_factory import TrainerFactory

        factory = TrainerFactory(settings=mock_settings, logger=mock_logger, config=trainer_config)

        mock_trainer_module = MagicMock()
        mock_hrm_trainer_class = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_hrm_trainer_class.return_value = mock_trainer_instance
        mock_config_class = MagicMock()
        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance
        mock_trainer_module.HRMTrainer = mock_hrm_trainer_class
        mock_trainer_module.HRMTrainingConfig = mock_config_class

        custom_scaler = MagicMock()

        with patch.dict(sys.modules, {"src.training.agent_trainer": mock_trainer_module}):
            trainer = factory.create_hrm_trainer(
                agent=mock_hrm_agent,
                optimizer=mock_optimizer,
                loss_fn=mock_loss_fn,
                batch_size=64,
                num_batches=20,
                gradient_clip_norm=0.5,
                ponder_weight=0.05,
                consistency_weight=0.2,
                use_mixed_precision=True,
                device="cuda",
                scaler=custom_scaler,
            )

            assert trainer is mock_trainer_instance
            config_call_kwargs = mock_config_class.call_args.kwargs
            assert config_call_kwargs["batch_size"] == 64
            assert config_call_kwargs["num_batches"] == 20
            assert config_call_kwargs["gradient_clip_norm"] == 0.5
            assert config_call_kwargs["ponder_weight"] == 0.05
            assert config_call_kwargs["consistency_weight"] == 0.2
            assert config_call_kwargs["use_mixed_precision"] is True

            trainer_call_kwargs = mock_hrm_trainer_class.call_args.kwargs
            assert trainer_call_kwargs["device"] == "cuda"
            assert trainer_call_kwargs["scaler"] is custom_scaler


@pytest.mark.unit
class TestTRMTrainerCustomConfig:
    """Test TRM trainer creation with custom parameter overrides."""

    def test_create_trm_trainer_all_custom_params(
        self, mock_settings, mock_logger, trainer_config, mock_trm_agent, mock_optimizer, mock_loss_fn
    ):
        """Test creating TRM trainer with all custom overrides."""
        import sys

        from src.framework.component_factory import TrainerFactory

        factory = TrainerFactory(settings=mock_settings, logger=mock_logger, config=trainer_config)

        mock_trainer_module = MagicMock()
        mock_trm_trainer_class = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trm_trainer_class.return_value = mock_trainer_instance
        mock_config_class = MagicMock()
        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance
        mock_trainer_module.TRMTrainer = mock_trm_trainer_class
        mock_trainer_module.TRMTrainingConfig = mock_config_class

        with patch.dict(sys.modules, {"src.training.agent_trainer": mock_trainer_module}):
            trainer = factory.create_trm_trainer(
                agent=mock_trm_agent,
                optimizer=mock_optimizer,
                loss_fn=mock_loss_fn,
                batch_size=128,
                num_batches=30,
                gradient_clip_norm=2.0,
                supervision_weight_decay=0.8,
                use_mixed_precision=True,
                device="cuda:1",
                scaler=MagicMock(),
            )

            assert trainer is mock_trainer_instance
            config_call_kwargs = mock_config_class.call_args.kwargs
            assert config_call_kwargs["batch_size"] == 128
            assert config_call_kwargs["supervision_weight_decay"] == 0.8


@pytest.mark.unit
class TestConvenienceFunctionsNoSettings:
    """Test convenience functions when no settings are provided (use defaults)."""

    def test_create_trainer_factory_no_settings(self):
        """Test create_trainer_factory without explicit settings."""
        from src.framework.component_factory import create_trainer_factory

        mock_settings = MagicMock()
        mock_settings.MCTS_MAX_PARALLEL_ROLLOUTS = 2
        mock_settings.MCTS_IMPL.value = "baseline"
        mock_settings.MCTS_ITERATIONS = 100
        mock_settings.S3_BUCKET = None

        with patch("src.framework.component_factory.registry.get_settings", return_value=mock_settings):
            factory = create_trainer_factory()
            assert factory is not None

    def test_create_metrics_factory_no_settings(self):
        """Test create_metrics_factory without explicit settings."""
        from src.framework.component_factory import create_metrics_factory

        mock_settings = MagicMock()
        mock_settings.WANDB_PROJECT = "test"
        mock_settings.WANDB_ENTITY = None
        mock_settings.WANDB_MODE = "offline"

        with patch("src.framework.component_factory.registry.get_settings", return_value=mock_settings):
            factory = create_metrics_factory()
            assert factory is not None

    def test_create_data_loader_factory_no_settings(self):
        """Test create_data_loader_factory without explicit settings."""
        from src.framework.component_factory import create_data_loader_factory

        mock_settings = MagicMock()
        mock_settings.MCTS_MAX_PARALLEL_ROLLOUTS = 2

        with patch("src.framework.component_factory.registry.get_settings", return_value=mock_settings):
            factory = create_data_loader_factory()
            assert factory is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
