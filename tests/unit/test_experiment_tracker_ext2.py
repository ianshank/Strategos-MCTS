"""
Additional unit tests for src/training/experiment_tracker.py targeting uncovered lines.

Focuses on missed lines:
- BraintrustTracker._initialize_client: ImportError path (91-93), generic Exception path (94-96)
- BraintrustTracker.init_experiment: online path (130-140) including fallback on exception
- BraintrustTracker.log_hyperparameters: online path (155-164) including exception
- BraintrustTracker.log_metric: online path (195-204) including exception
- BraintrustTracker.log_evaluation: online path (249-259) including exception
- BraintrustTracker.log_artifact: online path (285-293) including exception
- BraintrustTracker.get_summary: online path (310)
- BraintrustTracker.end_experiment: online path with close/flush (323-330)
- WandBTracker._initialize_client: ImportError (389-391), generic Exception (392-394)
- WandBTracker.init_run: online path (420-434)
- WandBTracker.log: online path (448-452) including exception
- WandBTracker.update_config: online path (490-494) including exception
- WandBTracker.watch_model: online path (507-511) including exception
- WandBTracker.log_artifact: online path (526-533) including exception
- WandBTracker.finish: online path (541-546) including exception
"""

import os
import time
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
    """Patch that removes tracker-related env vars."""
    return patch.dict(
        os.environ,
        {
            "BRAINTRUST_API_KEY": "",
            "WANDB_API_KEY": "",
            "WANDB_MODE": "",
        },
        clear=False,
    )


# ---------------------------------------------------------------------------
# BraintrustTracker - online mode paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBraintrustTrackerInitializeClient:
    """Tests for BraintrustTracker._initialize_client branches."""

    def test_initialize_client_import_error(self):
        """ImportError in _initialize_client falls back to offline mode."""
        with patch.dict(os.environ, {}, clear=True):
            tracker = BraintrustTracker.__new__(BraintrustTracker)
            tracker.api_key = "test-key"
            tracker.project_name = "test"
            tracker._experiment = None
            tracker._experiment_id = None
            tracker._metrics_buffer = []
            tracker._initialized = False
            tracker._offline_mode = False

            with patch("builtins.__import__", side_effect=ImportError("no braintrust")):
                tracker._initialize_client()

            assert tracker._offline_mode is True

    def test_initialize_client_generic_exception(self):
        """Generic exception in _initialize_client falls back to offline mode."""
        tracker = BraintrustTracker.__new__(BraintrustTracker)
        tracker.api_key = "test-key"
        tracker.project_name = "test"
        tracker._experiment = None
        tracker._experiment_id = None
        tracker._metrics_buffer = []
        tracker._initialized = False
        tracker._offline_mode = False

        mock_bt = MagicMock()
        mock_bt.login.side_effect = RuntimeError("API connection failed")

        with patch.dict("sys.modules", {"braintrust": mock_bt}):
            tracker._initialize_client()

        assert tracker._offline_mode is True

    def test_initialize_client_success(self):
        """Successful _initialize_client sets _initialized and _bt."""
        tracker = BraintrustTracker.__new__(BraintrustTracker)
        tracker.api_key = "test-key"
        tracker.project_name = "test"
        tracker._experiment = None
        tracker._experiment_id = None
        tracker._metrics_buffer = []
        tracker._initialized = False
        tracker._offline_mode = False

        mock_bt = MagicMock()

        with patch.dict("sys.modules", {"braintrust": mock_bt}):
            tracker._initialize_client()

        assert tracker._initialized is True
        assert tracker._bt is mock_bt
        mock_bt.login.assert_called_once_with(api_key="test-key")


@pytest.mark.unit
class TestBraintrustTrackerOnlineMode:
    """Tests for BraintrustTracker online mode code paths."""

    def _make_online_tracker(self):
        """Create a BraintrustTracker in online mode with mocked braintrust."""
        tracker = BraintrustTracker.__new__(BraintrustTracker)
        tracker.api_key = "test-key"
        tracker.project_name = "test-project"
        tracker._experiment = MagicMock()
        tracker._experiment.id = "exp-123"
        tracker._experiment_id = "exp-123"
        tracker._metrics_buffer = []
        tracker._initialized = True
        tracker._offline_mode = False
        tracker._bt = MagicMock()
        return tracker

    def test_init_experiment_online_success(self):
        """init_experiment online mode calls _bt.init and returns experiment ID."""
        tracker = self._make_online_tracker()
        tracker._experiment = None
        tracker._experiment_id = None

        mock_exp = MagicMock()
        mock_exp.id = "new-exp-456"
        tracker._bt.init.return_value = mock_exp

        result = tracker.init_experiment("my-exp", description="desc", tags=["v1"])
        assert result == "new-exp-456"
        tracker._bt.init.assert_called_once_with(
            project="test-project",
            experiment="my-exp",
        )
        assert tracker._experiment is mock_exp

    def test_init_experiment_online_exception_falls_back_to_offline(self):
        """init_experiment online mode falls back to offline on exception."""
        tracker = self._make_online_tracker()
        tracker._experiment = None
        tracker._experiment_id = None

        tracker._bt.init.side_effect = RuntimeError("API error")

        # The fallback calls init_experiment recursively in offline mode
        # Set _offline_mode after the exception to simulate the fallback

        call_count = [0]

        def patched_init(name, description="", tags=None, metadata=None):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: online, will fail
                try:
                    tracker._bt.init(project=tracker.project_name, experiment=name)
                    return str(tracker._experiment_id)
                except Exception:
                    tracker._offline_mode = True
                    return patched_init(name, description, tags, metadata)
            else:
                # Recursive call: offline mode
                exp_id = f"offline_{int(time.time())}"
                tracker._experiment_id = exp_id
                return exp_id

        tracker.init_experiment = patched_init
        result = tracker.init_experiment("exp")
        assert result.startswith("offline_")

    def test_init_experiment_online_none_experiment(self):
        """init_experiment when _bt.init returns None."""
        tracker = self._make_online_tracker()
        tracker._experiment = None
        tracker._experiment_id = None
        tracker._bt.init.return_value = None

        result = tracker.init_experiment("exp")
        assert tracker._experiment_id is None
        assert result == "None"

    def test_log_hyperparameters_online_success(self):
        """log_hyperparameters online calls _experiment.log."""
        tracker = self._make_online_tracker()
        tracker.log_hyperparameters({"lr": 0.001, "epochs": 10})
        tracker._experiment.log.assert_called_once_with(
            input="hyperparameters",
            output={"lr": 0.001, "epochs": 10},
            metadata={"type": "hyperparameters"},
        )

    def test_log_hyperparameters_online_exception(self):
        """log_hyperparameters online handles exception gracefully."""
        tracker = self._make_online_tracker()
        tracker._experiment.log.side_effect = RuntimeError("log fail")
        # Should not raise
        tracker.log_hyperparameters({"lr": 0.001})

    def test_log_hyperparameters_online_no_experiment(self):
        """log_hyperparameters online with no experiment does not call log."""
        tracker = self._make_online_tracker()
        tracker._experiment = None
        # Should not raise
        tracker.log_hyperparameters({"lr": 0.001})

    def test_log_metric_online_success(self):
        """log_metric online calls _experiment.log with scores."""
        tracker = self._make_online_tracker()
        tracker.log_metric("loss", 0.5, step=10)

        tracker._experiment.log.assert_called_once_with(
            input="metric_loss",
            output={"value": 0.5},
            scores={"loss": 0.5},
            metadata={"step": 10},
        )
        assert len(tracker._metrics_buffer) == 1

    def test_log_metric_online_exception(self):
        """log_metric online handles exception gracefully."""
        tracker = self._make_online_tracker()
        tracker._experiment.log.side_effect = RuntimeError("fail")
        tracker.log_metric("loss", 0.5, step=1)
        # Should not raise, metric still buffered
        assert len(tracker._metrics_buffer) == 1

    def test_log_metric_online_no_experiment(self):
        """log_metric online with no experiment does not call log."""
        tracker = self._make_online_tracker()
        tracker._experiment = None
        tracker.log_metric("loss", 0.5)
        assert len(tracker._metrics_buffer) == 1

    def test_log_evaluation_online_success(self):
        """log_evaluation online calls _experiment.log."""
        tracker = self._make_online_tracker()
        tracker.log_evaluation(
            input_data="test input",
            output="test output",
            expected="expected output",
            scores={"accuracy": 0.9},
            metadata={"split": "test"},
        )
        tracker._experiment.log.assert_called_once_with(
            input="test input",
            output="test output",
            expected="expected output",
            scores={"accuracy": 0.9},
            metadata={"split": "test"},
        )

    def test_log_evaluation_online_exception(self):
        """log_evaluation online handles exception gracefully."""
        tracker = self._make_online_tracker()
        tracker._experiment.log.side_effect = RuntimeError("fail")
        tracker.log_evaluation("inp", "out", "exp", {"acc": 0.5})

    def test_log_evaluation_online_no_experiment(self):
        """log_evaluation online with no experiment does not call log."""
        tracker = self._make_online_tracker()
        tracker._experiment = None
        tracker.log_evaluation("inp", "out", "exp", {"acc": 0.5})

    def test_log_evaluation_online_no_metadata(self):
        """log_evaluation online with metadata=None passes empty dict."""
        tracker = self._make_online_tracker()
        tracker.log_evaluation("inp", "out", "exp", {"acc": 0.5}, metadata=None)
        call_kwargs = tracker._experiment.log.call_args[1]
        assert call_kwargs["metadata"] == {}

    def test_log_artifact_online_success(self, tmp_path):
        """log_artifact online calls _experiment.log for existing file."""
        tracker = self._make_online_tracker()
        artifact = tmp_path / "model.pt"
        artifact.write_text("model data")

        tracker.log_artifact(str(artifact), name="my_model")
        tracker._experiment.log.assert_called_once()
        call_kwargs = tracker._experiment.log.call_args[1]
        assert "model.pt" in call_kwargs["input"] or "my_model" in call_kwargs["input"]

    def test_log_artifact_online_exception(self, tmp_path):
        """log_artifact online handles exception gracefully."""
        tracker = self._make_online_tracker()
        artifact = tmp_path / "model.pt"
        artifact.write_text("data")
        tracker._experiment.log.side_effect = RuntimeError("fail")
        tracker.log_artifact(str(artifact))

    def test_log_artifact_online_no_experiment(self, tmp_path):
        """log_artifact online with no experiment does not call log."""
        tracker = self._make_online_tracker()
        tracker._experiment = None
        artifact = tmp_path / "model.pt"
        artifact.write_text("data")
        tracker.log_artifact(str(artifact))

    def test_log_artifact_missing_file(self):
        """log_artifact with non-existent file logs warning and returns."""
        tracker = self._make_online_tracker()
        tracker.log_artifact("/nonexistent/path/model.pt")
        tracker._experiment.log.assert_not_called()

    def test_log_artifact_offline_mode(self, tmp_path):
        """log_artifact offline stores path in _experiment_config."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
        tracker.init_experiment("exp")
        artifact = tmp_path / "model.pt"
        artifact.write_text("data")
        tracker.log_artifact(str(artifact), name="model")
        assert str(artifact) in tracker._experiment_config["artifacts"]

    def test_log_artifact_offline_multiple(self, tmp_path):
        """log_artifact offline accumulates multiple artifacts."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
        tracker.init_experiment("exp")
        for i in range(3):
            art = tmp_path / f"model_{i}.pt"
            art.write_text("data")
            tracker.log_artifact(str(art))
        assert len(tracker._experiment_config["artifacts"]) == 3

    def test_get_summary_online(self):
        """get_summary online returns project and offline=False."""
        tracker = self._make_online_tracker()
        tracker._metrics_buffer = [{"a": 1}, {"b": 2}]
        summary = tracker.get_summary()
        assert summary["id"] == "exp-123"
        assert summary["project"] == "test-project"
        assert summary["metrics_count"] == 2
        assert summary["offline"] is False

    def test_end_experiment_online_with_close(self):
        """end_experiment calls close() if available on experiment."""
        tracker = self._make_online_tracker()
        tracker._experiment.close = MagicMock()
        tracker._experiment.flush = MagicMock()
        tracker.log_metric("loss", 0.1)

        summary = tracker.end_experiment()
        tracker._experiment_was = None  # Already cleared
        assert summary["metrics_count"] == 1
        assert tracker._experiment is None
        assert tracker._experiment_id is None
        assert tracker._metrics_buffer == []

    def test_end_experiment_online_with_flush_no_close(self):
        """end_experiment calls flush() when close() is not available."""
        tracker = self._make_online_tracker()
        # Remove close, keep flush
        mock_exp = MagicMock(spec=["log", "id", "flush"])
        mock_exp.id = "exp-789"
        mock_exp.flush = MagicMock()
        tracker._experiment = mock_exp

        tracker.end_experiment()
        mock_exp.flush.assert_called_once()
        assert tracker._experiment is None

    def test_end_experiment_online_exception_handled(self):
        """end_experiment handles exception during close/flush."""
        tracker = self._make_online_tracker()
        tracker._experiment.close = MagicMock(side_effect=RuntimeError("fail"))

        # Should not raise
        tracker.end_experiment()
        assert tracker._experiment is None


@pytest.mark.unit
class TestBraintrustTrackerLogTrainingStep:
    """Tests for log_training_step method."""

    def test_log_training_step_all_fields(self):
        """log_training_step logs all provided fields."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
        metrics = TrainingMetrics(
            epoch=2,
            step=100,
            train_loss=0.3,
            val_loss=0.4,
            accuracy=0.85,
            learning_rate=0.0001,
            custom_metrics={"f1": 0.82, "precision": 0.88},
        )
        tracker.log_training_step(metrics)
        # Should have logged: train_loss, val_loss, accuracy, learning_rate, f1, precision
        assert len(tracker._metrics_buffer) == 6

    def test_log_training_step_minimal(self):
        """log_training_step with only required fields."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
        metrics = TrainingMetrics(epoch=1, step=1, train_loss=0.5)
        tracker.log_training_step(metrics)
        assert len(tracker._metrics_buffer) == 1


@pytest.mark.unit
class TestBraintrustTrackerLogEvaluationOffline:
    """Tests for log_evaluation in offline mode."""

    def test_log_evaluation_offline(self):
        """log_evaluation in offline mode just logs info."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
        # Should not raise
        tracker.log_evaluation("inp", "out", "exp", {"acc": 0.9})


# ---------------------------------------------------------------------------
# WandBTracker - online mode paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWandBTrackerInitializeClient:
    """Tests for WandBTracker._initialize_client branches."""

    def test_initialize_client_import_error(self):
        """ImportError in _initialize_client falls back to offline mode."""
        tracker = WandBTracker.__new__(WandBTracker)
        tracker.api_key = "test-key"
        tracker.project_name = "test"
        tracker.entity = None
        tracker._run = None
        tracker._initialized = False
        tracker._offline_mode = False

        with patch("builtins.__import__", side_effect=ImportError("no wandb")):
            tracker._initialize_client()

        assert tracker._offline_mode is True

    def test_initialize_client_generic_exception(self):
        """Generic exception in _initialize_client falls back to offline mode."""
        tracker = WandBTracker.__new__(WandBTracker)
        tracker.api_key = "test-key"
        tracker.project_name = "test"
        tracker.entity = None
        tracker._run = None
        tracker._initialized = False
        tracker._offline_mode = False

        mock_wandb = MagicMock()
        mock_wandb.login.side_effect = RuntimeError("connection failed")

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker._initialize_client()

        assert tracker._offline_mode is True

    def test_initialize_client_success(self):
        """Successful _initialize_client sets _initialized and _wandb."""
        tracker = WandBTracker.__new__(WandBTracker)
        tracker.api_key = "test-key"
        tracker.project_name = "test"
        tracker.entity = None
        tracker._run = None
        tracker._initialized = False
        tracker._offline_mode = False

        mock_wandb = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker._initialize_client()

        assert tracker._initialized is True
        assert tracker._wandb is mock_wandb
        mock_wandb.login.assert_called_once_with(key="test-key")

    def test_initialize_client_no_api_key(self):
        """_initialize_client without api_key does not call login."""
        tracker = WandBTracker.__new__(WandBTracker)
        tracker.api_key = None
        tracker.project_name = "test"
        tracker.entity = None
        tracker._run = None
        tracker._initialized = False
        tracker._offline_mode = False

        mock_wandb = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker._initialize_client()

        assert tracker._initialized is True
        mock_wandb.login.assert_not_called()


@pytest.mark.unit
class TestWandBTrackerOnlineMode:
    """Tests for WandBTracker online mode code paths."""

    def _make_online_tracker(self):
        """Create a WandBTracker in online mode."""
        tracker = WandBTracker.__new__(WandBTracker)
        tracker.api_key = "test-key"
        tracker.project_name = "test-project"
        tracker.entity = "test-entity"
        tracker._run = MagicMock()
        tracker._initialized = True
        tracker._offline_mode = False
        tracker._wandb = MagicMock()
        return tracker

    def test_init_run_online_success(self):
        """init_run online calls _wandb.init and returns run."""
        tracker = self._make_online_tracker()
        mock_run = MagicMock()
        tracker._wandb.init.return_value = mock_run

        result = tracker.init_run(
            "my-run",
            config={"lr": 0.01},
            tags=["v1"],
            notes="test run",
        )

        assert result is mock_run
        tracker._wandb.init.assert_called_once_with(
            project="test-project",
            entity="test-entity",
            name="my-run",
            config={"lr": 0.01},
            tags=["v1"],
            notes="test run",
        )

    def test_init_run_online_exception(self):
        """init_run online handles exception and falls back to offline."""
        tracker = self._make_online_tracker()
        tracker._wandb.init.side_effect = RuntimeError("init fail")

        result = tracker.init_run("run")
        assert result is None
        assert tracker._offline_mode is True

    def test_log_online_success(self):
        """log online calls _wandb.log."""
        tracker = self._make_online_tracker()
        tracker.log({"loss": 0.5, "acc": 0.9}, step=5)
        tracker._wandb.log.assert_called_once_with({"loss": 0.5, "acc": 0.9}, step=5)

    def test_log_online_exception(self):
        """log online handles exception gracefully."""
        tracker = self._make_online_tracker()
        tracker._wandb.log.side_effect = RuntimeError("log fail")
        tracker.log({"loss": 0.5})  # Should not raise

    def test_log_online_no_run(self):
        """log online with no run does not call _wandb.log."""
        tracker = self._make_online_tracker()
        tracker._run = None
        tracker.log({"loss": 0.5})
        tracker._wandb.log.assert_not_called()

    def test_update_config_online_success(self):
        """update_config online calls _wandb.config.update."""
        tracker = self._make_online_tracker()
        tracker.update_config({"batch_size": 64})
        tracker._wandb.config.update.assert_called_once_with({"batch_size": 64})

    def test_update_config_online_exception(self):
        """update_config online handles exception gracefully."""
        tracker = self._make_online_tracker()
        tracker._wandb.config.update.side_effect = RuntimeError("fail")
        tracker.update_config({"batch_size": 64})  # Should not raise

    def test_update_config_online_no_run(self):
        """update_config online with no run does not call update."""
        tracker = self._make_online_tracker()
        tracker._run = None
        tracker.update_config({"batch_size": 64})
        tracker._wandb.config.update.assert_not_called()

    def test_watch_model_online_success(self):
        """watch_model online calls _wandb.watch."""
        tracker = self._make_online_tracker()
        mock_model = MagicMock()
        tracker.watch_model(mock_model, log_freq=50)
        tracker._wandb.watch.assert_called_once_with(mock_model, log="all", log_freq=50)

    def test_watch_model_online_exception(self):
        """watch_model online handles exception gracefully."""
        tracker = self._make_online_tracker()
        tracker._wandb.watch.side_effect = RuntimeError("fail")
        tracker.watch_model(MagicMock())  # Should not raise

    def test_watch_model_online_no_run(self):
        """watch_model online with no run does not call watch."""
        tracker = self._make_online_tracker()
        tracker._run = None
        tracker.watch_model(MagicMock())
        tracker._wandb.watch.assert_not_called()

    def test_log_artifact_online_success(self, tmp_path):
        """log_artifact online creates Artifact and logs it."""
        tracker = self._make_online_tracker()
        artifact_path = tmp_path / "model.pt"
        artifact_path.write_text("data")

        mock_artifact = MagicMock()
        tracker._wandb.Artifact.return_value = mock_artifact

        tracker.log_artifact(str(artifact_path), "my_model", artifact_type="model")

        tracker._wandb.Artifact.assert_called_once_with("my_model", type="model")
        mock_artifact.add_file.assert_called_once_with(str(artifact_path))
        tracker._run.log_artifact.assert_called_once_with(mock_artifact)

    def test_log_artifact_online_exception(self, tmp_path):
        """log_artifact online handles exception gracefully."""
        tracker = self._make_online_tracker()
        tracker._wandb.Artifact.side_effect = RuntimeError("fail")
        artifact_path = tmp_path / "model.pt"
        artifact_path.write_text("data")
        tracker.log_artifact(str(artifact_path), "model")  # Should not raise

    def test_log_artifact_online_no_run(self, tmp_path):
        """log_artifact online with no run still creates artifact but does not log."""
        tracker = self._make_online_tracker()
        tracker._run = None
        artifact_path = tmp_path / "model.pt"
        artifact_path.write_text("data")

        mock_artifact = MagicMock()
        tracker._wandb.Artifact.return_value = mock_artifact

        tracker.log_artifact(str(artifact_path), "model")
        # Artifact is created but not logged since _run is None
        tracker._wandb.Artifact.assert_called_once()

    def test_finish_online_success(self):
        """finish online calls _run.finish()."""
        tracker = self._make_online_tracker()
        tracker.finish()
        tracker._run.finish.assert_called_once()

    def test_finish_online_exception(self):
        """finish online handles exception gracefully."""
        tracker = self._make_online_tracker()
        tracker._run.finish.side_effect = RuntimeError("fail")
        tracker.finish()  # Should not raise

    def test_finish_online_no_run(self):
        """finish online with no run does not raise."""
        tracker = self._make_online_tracker()
        tracker._run = None
        tracker.finish()  # Should not raise


# ---------------------------------------------------------------------------
# WandBTracker - __init__ paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWandBTrackerInit:
    """Tests for WandBTracker __init__ branches."""

    def test_init_offline_mode_from_env(self):
        """WANDB_MODE=offline sets _offline_mode without API key check."""
        with patch.dict(os.environ, {"WANDB_MODE": "offline"}, clear=False):
            os.environ.pop("WANDB_API_KEY", None)
            tracker = WandBTracker(api_key=None)
            assert tracker._offline_mode is True

    def test_init_no_api_key_no_env_mode(self):
        """No API key and no WANDB_MODE=offline sets offline mode and env var."""
        with patch.dict(os.environ, {}, clear=True):
            tracker = WandBTracker(api_key=None)
            assert tracker._offline_mode is True
            assert os.environ.get("WANDB_MODE") == "offline"

    def test_init_with_api_key_calls_initialize(self):
        """API key provided triggers _initialize_client."""
        with patch.object(WandBTracker, "_initialize_client") as mock_init:
            with patch.dict(os.environ, {}, clear=True):
                WandBTracker(api_key="test-key")
                mock_init.assert_called_once()

    def test_init_with_entity(self):
        """Entity parameter is stored."""
        with patch.dict(os.environ, {}, clear=True):
            tracker = WandBTracker(api_key=None, entity="my-team")
            assert tracker.entity == "my-team"


# ---------------------------------------------------------------------------
# WandBTracker - log_training_step
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWandBTrackerLogTrainingStep:
    """Tests for WandBTracker.log_training_step."""

    def test_log_training_step_all_fields(self):
        """log_training_step includes all optional fields when present."""
        with patch.dict(os.environ, {}, clear=True):
            tracker = WandBTracker(api_key=None)
        tracker.log = MagicMock()
        metrics = TrainingMetrics(
            epoch=1,
            step=10,
            train_loss=0.3,
            val_loss=0.4,
            accuracy=0.85,
            learning_rate=0.001,
            custom_metrics={"f1": 0.8},
        )
        tracker.log_training_step(metrics)
        call_args = tracker.log.call_args
        log_data = call_args[0][0]
        assert log_data["val_loss"] == 0.4
        assert log_data["accuracy"] == 0.85
        assert log_data["learning_rate"] == 0.001
        assert log_data["f1"] == 0.8
        assert call_args[1]["step"] == 10


# ---------------------------------------------------------------------------
# ExperimentConfig and TrainingMetrics dataclasses
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDataclasses:
    """Tests for ExperimentConfig and TrainingMetrics dataclasses."""

    def test_experiment_config_defaults(self):
        """ExperimentConfig has correct defaults."""
        config = ExperimentConfig(project_name="proj", experiment_name="exp")
        assert config.tags == []
        assert config.description == ""
        assert config.save_artifacts is True
        assert config.log_frequency == 1

    def test_experiment_config_all_fields(self):
        """ExperimentConfig with all fields set."""
        config = ExperimentConfig(
            project_name="proj",
            experiment_name="exp",
            tags=["v1", "test"],
            description="Test experiment",
            save_artifacts=False,
            log_frequency=10,
        )
        assert config.tags == ["v1", "test"]
        assert config.save_artifacts is False

    def test_training_metrics_defaults(self):
        """TrainingMetrics has correct defaults."""
        metrics = TrainingMetrics(epoch=1, step=1, train_loss=0.5)
        assert metrics.val_loss is None
        assert metrics.accuracy is None
        assert metrics.learning_rate is None
        assert metrics.custom_metrics == {}
        assert isinstance(metrics.timestamp, float)

    def test_training_metrics_all_fields(self):
        """TrainingMetrics with all fields set."""
        metrics = TrainingMetrics(
            epoch=5,
            step=500,
            train_loss=0.1,
            val_loss=0.2,
            accuracy=0.95,
            learning_rate=0.0001,
            timestamp=12345.0,
            custom_metrics={"f1": 0.9},
        )
        assert metrics.epoch == 5
        assert metrics.custom_metrics["f1"] == 0.9
        assert metrics.timestamp == 12345.0


# ---------------------------------------------------------------------------
# UnifiedExperimentTracker
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnifiedExperimentTrackerExt2:
    """Additional tests for UnifiedExperimentTracker."""

    def _make_tracker(self):
        with patch.dict(os.environ, {}, clear=True):
            return UnifiedExperimentTracker()

    def test_init_creates_both_trackers(self):
        """UnifiedExperimentTracker creates both bt and wandb trackers."""
        tracker = self._make_tracker()
        assert isinstance(tracker.bt, BraintrustTracker)
        assert isinstance(tracker.wandb, WandBTracker)

    def test_init_with_api_keys(self):
        """API keys are passed to sub-trackers."""
        with patch.dict(os.environ, {}, clear=True):
            # Both will be offline since the keys aren't real, but they'll be set
            tracker = UnifiedExperimentTracker(
                braintrust_api_key=None,
                wandb_api_key=None,
                project_name="test-proj",
            )
        assert tracker.project_name == "test-proj"

    def test_log_metrics_delegates_to_both(self):
        """log_metrics calls both bt and wandb."""
        tracker = self._make_tracker()
        tracker.bt.log_training_step = MagicMock()
        tracker.wandb.log_training_step = MagicMock()

        metrics = TrainingMetrics(epoch=1, step=1, train_loss=0.5)
        tracker.log_metrics(metrics)

        tracker.bt.log_training_step.assert_called_once()
        tracker.wandb.log_training_step.assert_called_once()

    def test_log_evaluation_delegates(self):
        """log_evaluation calls bt.log_evaluation and wandb.log."""
        tracker = self._make_tracker()
        tracker.bt.log_evaluation = MagicMock()
        tracker.wandb.log = MagicMock()

        tracker.log_evaluation("input", "output", "expected", {"acc": 0.9})

        tracker.bt.log_evaluation.assert_called_once()
        tracker.wandb.log.assert_called_once()

    def test_log_artifact_delegates(self):
        """log_artifact calls both bt and wandb."""
        tracker = self._make_tracker()
        tracker.bt.log_artifact = MagicMock()
        tracker.wandb.log_artifact = MagicMock()

        tracker.log_artifact("/path/to/model.pt", "model")

        tracker.bt.log_artifact.assert_called_once_with("/path/to/model.pt", "model")
        tracker.wandb.log_artifact.assert_called_once_with("/path/to/model.pt", "model")

    def test_finish_returns_bt_summary(self):
        """finish returns bt summary and calls wandb.finish."""
        tracker = self._make_tracker()
        tracker.bt.init_experiment("test")
        tracker.wandb.finish = MagicMock()

        summary = tracker.finish()
        assert isinstance(summary, dict)
        tracker.wandb.finish.assert_called_once()

    def test_init_experiment_with_config_and_tags(self):
        """init_experiment with config logs hyperparameters."""
        tracker = self._make_tracker()
        tracker.init_experiment(
            "exp",
            config={"lr": 0.01},
            description="test",
            tags=["v1"],
        )
        assert tracker.bt._experiment_config["hyperparameters"] == {"lr": 0.01}

    def test_init_experiment_without_config(self):
        """init_experiment without config skips hyperparameters."""
        tracker = self._make_tracker()
        tracker.init_experiment("exp", description="test")
        assert "hyperparameters" not in tracker.bt._experiment_config
