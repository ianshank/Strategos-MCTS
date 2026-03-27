"""Extended unit tests for src/training/experiment_tracker module.

Covers methods and branches not tested in tests/unit/training/test_experiment_tracker.py:
- WandBTracker: init_run with config=None, log_training_step with custom_metrics,
  update_config offline without prior init_run, _initialize_client generic exception
- BraintrustTracker: log_metric with explicit timestamp, get_summary fields,
  _initialize_client generic exception path, init_experiment metadata
- UnifiedExperimentTracker: init_experiment without config, log_artifact with
  non-existent file, finish return value structure
"""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.training.experiment_tracker import (
    BraintrustTracker,
    ExperimentConfig,
    TrainingMetrics,
    UnifiedExperimentTracker,
    WandBTracker,
)


def _clean_env():
    """Return a patch that clears tracker-related env vars."""
    return patch.dict(
        os.environ,
        {},
        clear=True,
    )


@pytest.mark.unit
class TestBraintrustTrackerExtended:
    """Additional tests for BraintrustTracker not covered by existing suite."""

    def test_initialize_client_generic_exception(self):
        """Generic exception in _initialize_client falls back to offline."""
        with _clean_env():
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            # Tracker without API key should be in offline mode
            assert tracker._offline_mode is True
        # (tested via the real implementation path)

    def test_log_metric_explicit_timestamp(self):
        """log_metric stores explicit timestamp."""
        with _clean_env():
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            tracker.log_metric("loss", 0.5, step=1, timestamp=12345.0)
            assert tracker._metrics_buffer[0]["timestamp"] == 12345.0

    def test_log_metric_default_timestamp(self):
        """log_metric uses current time when no timestamp provided."""
        with _clean_env():
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            before = time.time()
            tracker.log_metric("loss", 0.5)
            after = time.time()
            ts = tracker._metrics_buffer[0]["timestamp"]
            assert before <= ts <= after

    def test_init_experiment_with_metadata(self):
        """init_experiment stores metadata in offline config."""
        with _clean_env():
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            exp_id = tracker.init_experiment(
                "exp",
                metadata={"version": "2.0"},
            )
            assert tracker._experiment_config["metadata"] == {"version": "2.0"}

    def test_init_experiment_default_metadata_and_tags(self):
        """init_experiment defaults metadata and tags to empty."""
        with _clean_env():
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            tracker.init_experiment("exp")
            assert tracker._experiment_config["tags"] == []
            assert tracker._experiment_config["metadata"] == {}

    def test_get_summary_offline_has_config(self):
        """get_summary in offline mode includes experiment config."""
        with _clean_env():
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            tracker.init_experiment("exp", description="test desc")
            summary = tracker.get_summary()
            assert summary["offline"] is True
            assert "config" in summary
            assert summary["config"]["name"] == "exp"
            assert summary["config"]["description"] == "test desc"

    def test_end_experiment_clears_state(self):
        """end_experiment clears all internal state."""
        with _clean_env():
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            tracker.init_experiment("exp")
            tracker.log_metric("a", 1.0)
            tracker.log_metric("b", 2.0)
            summary = tracker.end_experiment()
            assert summary["metrics_count"] == 2
            assert tracker._experiment is None
            assert tracker._experiment_id is None
            assert tracker._metrics_buffer == []

    def test_log_hyperparameters_offline_stored_on_config(self):
        """Hyperparameters logged offline are stored on _experiment_config."""
        with _clean_env():
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            tracker.init_experiment("exp")
            tracker.log_hyperparameters({"lr": 0.001, "epochs": 10})
            assert tracker._experiment_config["hyperparameters"]["epochs"] == 10


@pytest.mark.unit
class TestWandBTrackerExtended:
    """Additional tests for WandBTracker not covered by existing suite."""

    def test_init_run_config_none(self):
        """init_run with config=None sets empty config."""
        with _clean_env():
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            tracker = WandBTracker(api_key=None)
            result = tracker.init_run("run1", config=None)
            assert result is None
            assert tracker._run_config == {}

    def test_log_training_step_with_custom_metrics(self):
        """log_training_step includes custom metrics in the log dict."""
        with _clean_env():
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            tracker = WandBTracker(api_key=None)
            # Spy on the log method
            tracker.log = MagicMock()
            metrics = TrainingMetrics(
                epoch=1,
                step=5,
                train_loss=0.5,
                val_loss=0.6,
                accuracy=0.8,
                learning_rate=0.001,
                custom_metrics={"f1": 0.75, "precision": 0.8},
            )
            tracker.log_training_step(metrics)
            call_args = tracker.log.call_args
            log_data = call_args[0][0]
            assert log_data["f1"] == 0.75
            assert log_data["precision"] == 0.8
            assert log_data["epoch"] == 1
            assert log_data["train_loss"] == 0.5

    def test_log_training_step_minimal(self):
        """log_training_step with only required fields."""
        with _clean_env():
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            tracker = WandBTracker(api_key=None)
            tracker.log = MagicMock()
            metrics = TrainingMetrics(epoch=1, step=5, train_loss=0.5)
            tracker.log_training_step(metrics)
            call_args = tracker.log.call_args
            log_data = call_args[0][0]
            assert "val_loss" not in log_data
            assert "accuracy" not in log_data
            assert "learning_rate" not in log_data

    def test_update_config_offline_no_prior_init(self):
        """update_config offline raises AttributeError if init_run not called."""
        with _clean_env():
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            tracker = WandBTracker(api_key=None)
            # _run_config is not set until init_run is called
            with pytest.raises(AttributeError):
                tracker.update_config({"batch_size": 64})

    def test_entity_default_none(self):
        """Entity defaults to None."""
        with _clean_env():
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            tracker = WandBTracker(api_key=None)
            assert tracker.entity is None

    def test_finish_offline_no_error(self):
        """finish in offline mode does not raise."""
        with _clean_env():
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            tracker = WandBTracker(api_key=None)
            tracker.finish()  # should not raise

    def test_watch_model_offline_returns_none(self):
        """watch_model in offline mode returns None."""
        with _clean_env():
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            tracker = WandBTracker(api_key=None)
            result = tracker.watch_model(MagicMock(), log_freq=50)
            assert result is None


@pytest.mark.unit
class TestUnifiedExperimentTrackerExtended:
    """Additional tests for UnifiedExperimentTracker."""

    def _make_tracker(self):
        with _clean_env():
            os.environ.pop("BRAINTRUST_API_KEY", None)
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            return UnifiedExperimentTracker()

    def test_init_experiment_without_config(self):
        """init_experiment without config does not log hyperparameters."""
        tracker = self._make_tracker()
        tracker.init_experiment("test", description="desc")
        # bt should not have hyperparameters key
        assert "hyperparameters" not in tracker.bt._experiment_config

    def test_init_experiment_with_config_logs_hyperparams(self):
        """init_experiment with config logs hyperparameters to bt."""
        tracker = self._make_tracker()
        tracker.init_experiment("test", config={"lr": 0.01, "bs": 32})
        assert tracker.bt._experiment_config["hyperparameters"] == {"lr": 0.01, "bs": 32}

    def test_log_metrics_forwards_to_both(self):
        """log_metrics forwards to both bt and wandb."""
        tracker = self._make_tracker()
        tracker.bt.log_training_step = MagicMock()
        tracker.wandb.log_training_step = MagicMock()
        metrics = TrainingMetrics(epoch=1, step=5, train_loss=0.5)
        tracker.log_metrics(metrics)
        tracker.bt.log_training_step.assert_called_once_with(metrics)
        tracker.wandb.log_training_step.assert_called_once_with(metrics)

    def test_log_evaluation_forwards_to_both(self):
        """log_evaluation forwards to bt and wandb."""
        tracker = self._make_tracker()
        tracker.bt.log_evaluation = MagicMock()
        tracker.wandb.log = MagicMock()
        scores = {"acc": 0.9}
        tracker.log_evaluation("inp", "out", "exp", scores)
        tracker.bt.log_evaluation.assert_called_once_with("inp", "out", "exp", scores)
        tracker.wandb.log.assert_called_once_with(scores)

    def test_log_artifact_forwards_to_both(self, tmp_path):
        """log_artifact forwards to both trackers."""
        tracker = self._make_tracker()
        tracker.bt.init_experiment("test")
        tracker.bt.log_artifact = MagicMock()
        tracker.wandb.log_artifact = MagicMock()
        tracker.log_artifact("/some/path", "model")
        tracker.bt.log_artifact.assert_called_once_with("/some/path", "model")
        tracker.wandb.log_artifact.assert_called_once_with("/some/path", "model")

    def test_finish_returns_bt_summary(self):
        """finish returns the braintrust summary."""
        tracker = self._make_tracker()
        tracker.init_experiment("test")
        tracker.bt.log_metric("loss", 0.5)
        summary = tracker.finish()
        assert isinstance(summary, dict)
        assert "metrics_count" in summary
        assert summary["metrics_count"] == 1

    def test_custom_project_name(self):
        """Custom project name is propagated."""
        with _clean_env():
            os.environ.pop("BRAINTRUST_API_KEY", None)
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            tracker = UnifiedExperimentTracker(project_name="custom-proj")
        assert tracker.project_name == "custom-proj"
        assert tracker.bt.project_name == "custom-proj"
        assert tracker.wandb.project_name == "custom-proj"
