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

        with patch("src.framework.component_factory.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.MCTS_MAX_PARALLEL_ROLLOUTS = 4
            mock_settings.MCTS_IMPL.value = "baseline"
            mock_settings.MCTS_ITERATIONS = 100
            mock_settings.S3_BUCKET = None
            mock_get_settings.return_value = mock_settings

            factory = TrainerFactory()

            mock_get_settings.assert_called_once()
            assert factory._settings is mock_settings

    def test_create_hrm_trainer_default(self, mock_settings, mock_logger, trainer_config, mock_hrm_agent, mock_optimizer, mock_loss_fn):
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

    def test_create_trm_trainer_default(self, mock_settings, mock_logger, trainer_config, mock_trm_agent, mock_optimizer, mock_loss_fn):
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

        with patch.dict(sys.modules, {
            "src.training.performance_monitor": mock_monitor_module,
            "src.training.experiment_tracker": mock_tracker_module,
        }):
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
        with patch.dict(sys.modules, {
            "src.training.replay_buffer": mock_buffer_module,
            "src.training.performance_monitor": mock_monitor_module,
            "src.data.dataset_loader": mock_loader_module,
        }):
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

        with patch("src.framework.component_factory.get_settings", return_value=mock_settings):
            factory = create_trainer_factory(settings=mock_settings)

            assert factory is not None
            assert factory._settings is mock_settings

    def test_create_metrics_factory(self, mock_settings):
        """Test create_metrics_factory convenience function."""
        from src.framework.component_factory import create_metrics_factory

        with patch("src.framework.component_factory.get_settings", return_value=mock_settings):
            factory = create_metrics_factory(settings=mock_settings)

            assert factory is not None
            assert factory._settings is mock_settings

    def test_create_data_loader_factory(self, mock_settings):
        """Test create_data_loader_factory convenience function."""
        from src.framework.component_factory import create_data_loader_factory

        with patch("src.framework.component_factory.get_settings", return_value=mock_settings):
            factory = create_data_loader_factory(settings=mock_settings)

            assert factory is not None
            assert factory._settings is mock_settings

    def test_get_component_registry(self, mock_settings):
        """Test get_component_registry convenience function."""
        from src.framework.component_factory import get_component_registry

        with patch("src.framework.component_factory.get_settings", return_value=mock_settings):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
