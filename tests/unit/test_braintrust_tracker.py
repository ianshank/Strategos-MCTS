"""Unit tests for BraintrustTracker and related components."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure braintrust mock is in sys.modules before the tracker module loads
_mock_braintrust = MagicMock()
sys.modules["braintrust"] = _mock_braintrust

# Force reimport so the module picks up BRAINTRUST_AVAILABLE = True
import importlib

import src.observability.braintrust_tracker as bt_module

importlib.reload(bt_module)

from src.observability.braintrust_tracker import (  # noqa: E402
    BraintrustContextManager,
    BraintrustTracker,
    create_training_tracker,
)

# Verify the module sees braintrust as available
assert bt_module.BRAINTRUST_AVAILABLE is True, "BRAINTRUST_AVAILABLE should be True after mocking"


@pytest.mark.unit
class TestBraintrustTrackerInit:
    """Tests for BraintrustTracker initialization."""

    def test_default_project_name(self):
        tracker = BraintrustTracker(auto_init=False)
        assert tracker.project_name == "neural-meta-controller"

    def test_custom_project_name(self):
        tracker = BraintrustTracker(project_name="my-project", auto_init=False)
        assert tracker.project_name == "my-project"

    def test_api_key_from_param(self):
        tracker = BraintrustTracker(api_key="test-key", auto_init=False)
        assert tracker._api_key == "test-key"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("BRAINTRUST_API_KEY", "env-key")
        tracker = BraintrustTracker(auto_init=False)
        assert tracker._api_key == "env-key"

    def test_initial_state(self):
        tracker = BraintrustTracker(auto_init=False)
        assert tracker._experiment is None
        assert tracker._current_span is None
        assert tracker._metrics_buffer == []

    def test_auto_init_calls_initialize_when_key_present(self):
        """auto_init=True with an API key should call _initialize."""
        with patch.object(BraintrustTracker, "_initialize") as mock_init:
            BraintrustTracker(api_key="test-key", auto_init=True)
            mock_init.assert_called_once()

    def test_auto_init_skipped_without_key(self):
        with patch.object(BraintrustTracker, "_initialize") as mock_init:
            BraintrustTracker(api_key=None, auto_init=True)
            mock_init.assert_not_called()

    def test_initialize_calls_braintrust_login(self):
        """_initialize should call braintrust.login with the API key."""
        tracker = BraintrustTracker(api_key="my-key", auto_init=False)
        tracker._initialize()
        _mock_braintrust.login.assert_called_with(api_key="my-key")
        assert tracker._is_initialized is True


@pytest.mark.unit
class TestBraintrustTrackerIsAvailable:
    """Tests for the is_available property."""

    def test_available_when_initialized_with_key(self):
        tracker = BraintrustTracker(api_key="test-key", auto_init=False)
        tracker._is_initialized = True
        assert tracker.is_available is True

    def test_not_available_without_key(self):
        tracker = BraintrustTracker(auto_init=False)
        tracker._is_initialized = True
        assert tracker.is_available is False

    def test_not_available_when_not_initialized(self):
        tracker = BraintrustTracker(api_key="test-key", auto_init=False)
        assert tracker.is_available is False


@pytest.mark.unit
class TestBraintrustTrackerStartExperiment:
    """Tests for start_experiment."""

    def _make_available_tracker(self):
        tracker = BraintrustTracker(api_key="test-key", auto_init=False)
        tracker._is_initialized = True
        return tracker

    def test_returns_none_when_not_available(self):
        tracker = BraintrustTracker(auto_init=False)
        result = tracker.start_experiment("test")
        assert result is None

    @patch("src.observability.braintrust_tracker.braintrust")
    def test_start_experiment_with_name(self, mock_bt):
        mock_bt.init.return_value = MagicMock()
        tracker = self._make_available_tracker()
        result = tracker.start_experiment("my_experiment")
        mock_bt.init.assert_called_once_with(
            project="neural-meta-controller",
            experiment="my_experiment",
            metadata={},
        )
        assert result is not None
        assert tracker._experiment is not None

    @patch("src.observability.braintrust_tracker.braintrust")
    def test_start_experiment_auto_name(self, mock_bt):
        mock_bt.init.return_value = MagicMock()
        tracker = self._make_available_tracker()
        tracker.start_experiment()
        call_kwargs = mock_bt.init.call_args[1]
        assert call_kwargs["experiment"].startswith("meta_controller_training_")

    @patch("src.observability.braintrust_tracker.braintrust")
    def test_start_experiment_with_metadata(self, mock_bt):
        mock_bt.init.return_value = MagicMock()
        tracker = self._make_available_tracker()
        meta = {"lr": 0.001}
        tracker.start_experiment("exp", metadata=meta)
        call_kwargs = mock_bt.init.call_args[1]
        assert call_kwargs["metadata"] == meta

    @patch("src.observability.braintrust_tracker.braintrust")
    def test_start_experiment_handles_exception(self, mock_bt):
        mock_bt.init.side_effect = RuntimeError("API error")
        tracker = self._make_available_tracker()
        result = tracker.start_experiment("exp")
        assert result is None


@pytest.mark.unit
class TestBraintrustTrackerLogging:
    """Tests for logging methods when not available (buffer mode)."""

    def _make_unavailable_tracker(self):
        return BraintrustTracker(auto_init=False)

    def test_log_hyperparameters_buffers(self):
        tracker = self._make_unavailable_tracker()
        tracker.log_hyperparameters({"lr": 0.001})
        assert len(tracker._metrics_buffer) == 1
        assert tracker._metrics_buffer[0]["type"] == "hyperparameters"
        assert tracker._metrics_buffer[0]["data"] == {"lr": 0.001}

    def test_log_training_step_buffers(self):
        tracker = self._make_unavailable_tracker()
        tracker.log_training_step(epoch=1, step=10, loss=0.5, metrics={"acc": 0.9})
        buf = tracker._metrics_buffer[0]
        assert buf["type"] == "training_step"
        assert buf["epoch"] == 1
        assert buf["step"] == 10
        assert buf["loss"] == 0.5
        assert buf["metrics"] == {"acc": 0.9}

    def test_log_training_step_buffers_default_metrics(self):
        tracker = self._make_unavailable_tracker()
        tracker.log_training_step(epoch=0, step=0, loss=1.0)
        buf = tracker._metrics_buffer[0]
        assert buf["metrics"] == {}

    def test_log_epoch_summary_buffers(self):
        tracker = self._make_unavailable_tracker()
        tracker.log_epoch_summary(
            epoch=1, train_loss=0.5, val_loss=0.3,
            train_accuracy=0.8, val_accuracy=0.85,
            additional_metrics={"f1": 0.82},
        )
        buf = tracker._metrics_buffer[0]
        assert buf["type"] == "epoch_summary"
        assert buf["train_loss"] == 0.5
        assert buf["val_loss"] == 0.3
        assert buf["val_accuracy"] == 0.85

    def test_log_evaluation_buffers(self):
        tracker = self._make_unavailable_tracker()
        tracker.log_evaluation(
            eval_type="test",
            predictions=["a", "b"],
            ground_truth=["a", "c"],
            metrics={"accuracy": 0.5},
        )
        buf = tracker._metrics_buffer[0]
        assert buf["type"] == "evaluation"
        assert buf["num_samples"] == 2

    def test_log_model_prediction_buffers(self):
        tracker = self._make_unavailable_tracker()
        tracker.log_model_prediction(
            input_features={"x": 1},
            prediction="agent_a",
            confidence=0.95,
            ground_truth="agent_a",
        )
        buf = tracker._metrics_buffer[0]
        assert buf["type"] == "prediction"
        assert buf["confidence"] == 0.95

    def test_log_model_artifact_buffers(self):
        tracker = self._make_unavailable_tracker()
        tracker.log_model_artifact(
            model_path="/tmp/model.pt",
            model_type="rnn",
            metrics={"accuracy": 0.9},
            metadata={"version": "1.0"},
        )
        buf = tracker._metrics_buffer[0]
        assert buf["type"] == "model_artifact"
        assert buf["model_path"] == "/tmp/model.pt"


@pytest.mark.unit
class TestBraintrustTrackerWithExperiment:
    """Tests for logging methods when experiment is active."""

    def _make_active_tracker(self):
        tracker = BraintrustTracker(api_key="test-key", auto_init=False)
        tracker._is_initialized = True
        tracker._experiment = MagicMock()
        return tracker

    def test_log_hyperparameters_calls_experiment(self):
        tracker = self._make_active_tracker()
        tracker.log_hyperparameters({"lr": 0.001})
        tracker._experiment.log.assert_called_once_with(metadata={"lr": 0.001})

    def test_log_training_step_calls_experiment(self):
        tracker = self._make_active_tracker()
        tracker.log_training_step(epoch=1, step=5, loss=0.3, metrics={"acc": 0.9})
        tracker._experiment.log.assert_called_once_with(
            input={"epoch": 1, "step": 5},
            output={"loss": 0.3},
            scores={"acc": 0.9},
        )

    def test_log_epoch_summary_calls_experiment(self):
        tracker = self._make_active_tracker()
        tracker.log_epoch_summary(epoch=2, train_loss=0.4, val_loss=0.35)
        tracker._experiment.log.assert_called_once()
        call_kwargs = tracker._experiment.log.call_args[1]
        assert call_kwargs["scores"]["train_loss"] == 0.4
        assert call_kwargs["scores"]["val_loss"] == 0.35

    def test_log_epoch_summary_optional_fields(self):
        tracker = self._make_active_tracker()
        tracker.log_epoch_summary(epoch=1, train_loss=0.5)
        call_kwargs = tracker._experiment.log.call_args[1]
        assert "val_loss" not in call_kwargs["scores"]
        assert "train_accuracy" not in call_kwargs["scores"]
        assert "val_accuracy" not in call_kwargs["scores"]

    def test_log_epoch_summary_all_fields(self):
        tracker = self._make_active_tracker()
        tracker.log_epoch_summary(
            epoch=1, train_loss=0.5, val_loss=0.4,
            train_accuracy=0.8, val_accuracy=0.85,
            additional_metrics={"f1": 0.82},
        )
        call_kwargs = tracker._experiment.log.call_args[1]
        assert call_kwargs["scores"]["train_accuracy"] == 0.8
        assert call_kwargs["scores"]["val_accuracy"] == 0.85
        assert call_kwargs["scores"]["f1"] == 0.82

    def test_log_evaluation_calls_experiment(self):
        tracker = self._make_active_tracker()
        tracker.log_evaluation("test", ["a", "b"], ["a", "c"], {"acc": 0.5})
        tracker._experiment.log.assert_called_once()
        call_kwargs = tracker._experiment.log.call_args[1]
        assert call_kwargs["input"]["eval_type"] == "test"
        assert call_kwargs["input"]["num_samples"] == 2

    def test_log_model_prediction_correct_match(self):
        tracker = self._make_active_tracker()
        tracker.log_model_prediction({"x": 1}, "a", 0.9, ground_truth="a")
        call_kwargs = tracker._experiment.log.call_args[1]
        assert call_kwargs["scores"]["correct"] == 1.0
        assert call_kwargs["scores"]["confidence"] == 0.9

    def test_log_model_prediction_incorrect_match(self):
        tracker = self._make_active_tracker()
        tracker.log_model_prediction({"x": 1}, "a", 0.9, ground_truth="b")
        call_kwargs = tracker._experiment.log.call_args[1]
        assert call_kwargs["scores"]["correct"] == 0.0

    def test_log_model_prediction_no_ground_truth(self):
        tracker = self._make_active_tracker()
        tracker.log_model_prediction({"x": 1}, "a", 0.9)
        call_kwargs = tracker._experiment.log.call_args[1]
        assert "correct" not in call_kwargs["scores"]
        assert call_kwargs["scores"]["confidence"] == 0.9

    def test_log_model_artifact_calls_experiment(self):
        tracker = self._make_active_tracker()
        tracker.log_model_artifact("/tmp/m.pt", "rnn", {"acc": 0.9})
        tracker._experiment.log.assert_called_once()
        call_kwargs = tracker._experiment.log.call_args[1]
        assert call_kwargs["input"]["model_path"] == "/tmp/m.pt"
        assert call_kwargs["output"]["saved"] is True

    def test_log_model_artifact_with_metadata(self):
        tracker = self._make_active_tracker()
        tracker.log_model_artifact("/tmp/m.pt", "rnn", {"acc": 0.9}, metadata={"v": "1"})
        call_kwargs = tracker._experiment.log.call_args[1]
        assert call_kwargs["metadata"] == {"v": "1"}

    def test_log_handles_exception_gracefully(self):
        tracker = self._make_active_tracker()
        tracker._experiment.log.side_effect = RuntimeError("fail")
        # None of these should raise
        tracker.log_hyperparameters({"lr": 0.001})
        tracker.log_training_step(1, 1, 0.5)
        tracker.log_epoch_summary(1, 0.5)
        tracker.log_evaluation("test", ["a"], ["a"], {"acc": 1.0})
        tracker.log_model_prediction({"x": 1}, "a", 0.9, "a")
        tracker.log_model_artifact("/tmp/m.pt", "rnn", {"acc": 0.9})


@pytest.mark.unit
class TestBraintrustTrackerEndExperiment:
    """Tests for end_experiment."""

    def test_end_when_not_available(self):
        tracker = BraintrustTracker(auto_init=False)
        assert tracker.end_experiment() is None

    def test_end_when_no_experiment(self):
        tracker = BraintrustTracker(api_key="k", auto_init=False)
        tracker._is_initialized = True
        assert tracker.end_experiment() is None

    def test_end_returns_url(self):
        tracker = BraintrustTracker(api_key="k", auto_init=False)
        tracker._is_initialized = True
        mock_exp = MagicMock()
        mock_summary = MagicMock()
        mock_summary.experiment_url = "https://braintrust.dev/exp/123"
        mock_exp.summarize.return_value = mock_summary
        tracker._experiment = mock_exp

        url = tracker.end_experiment()
        assert url == "https://braintrust.dev/exp/123"
        assert tracker._experiment is None

    def test_end_returns_none_when_no_url_attr(self):
        tracker = BraintrustTracker(api_key="k", auto_init=False)
        tracker._is_initialized = True
        mock_exp = MagicMock()
        mock_summary = MagicMock(spec=[])  # No attributes
        mock_exp.summarize.return_value = mock_summary
        tracker._experiment = mock_exp

        url = tracker.end_experiment()
        assert url is None

    def test_end_handles_exception(self):
        tracker = BraintrustTracker(api_key="k", auto_init=False)
        tracker._is_initialized = True
        mock_exp = MagicMock()
        mock_exp.summarize.side_effect = RuntimeError("fail")
        tracker._experiment = mock_exp

        assert tracker.end_experiment() is None


@pytest.mark.unit
class TestBraintrustTrackerBuffer:
    """Tests for buffer operations."""

    def test_get_buffered_metrics_returns_copy(self):
        tracker = BraintrustTracker(auto_init=False)
        tracker.log_hyperparameters({"lr": 0.001})
        buf = tracker.get_buffered_metrics()
        assert len(buf) == 1
        buf.clear()
        assert len(tracker.get_buffered_metrics()) == 1

    def test_clear_buffer(self):
        tracker = BraintrustTracker(auto_init=False)
        tracker.log_hyperparameters({"lr": 0.001})
        tracker.clear_buffer()
        assert tracker.get_buffered_metrics() == []

    def test_multiple_items_buffered(self):
        tracker = BraintrustTracker(auto_init=False)
        tracker.log_hyperparameters({"lr": 0.001})
        tracker.log_training_step(1, 1, 0.5)
        tracker.log_epoch_summary(1, 0.5)
        assert len(tracker.get_buffered_metrics()) == 3


@pytest.mark.unit
class TestBraintrustContextManager:
    """Tests for BraintrustContextManager."""

    def test_enter_creates_tracker_and_starts_experiment(self):
        with patch.object(BraintrustTracker, "start_experiment") as mock_start:
            cm = BraintrustContextManager(
                project_name="test-proj",
                experiment_name="exp1",
                api_key="key",
                metadata={"k": "v"},
            )
            tracker = cm.__enter__()
            assert isinstance(tracker, BraintrustTracker)
            mock_start.assert_called_once_with(
                experiment_name="exp1",
                metadata={"k": "v"},
            )

    def test_exit_ends_experiment(self):
        with patch.object(BraintrustTracker, "start_experiment"):
            with patch.object(BraintrustTracker, "end_experiment", return_value="http://url") as mock_end:
                cm = BraintrustContextManager(api_key="key")
                cm.__enter__()
                cm.__exit__(None, None, None)
                mock_end.assert_called_once()
                assert cm.experiment_url == "http://url"

    def test_exit_returns_false(self):
        with patch.object(BraintrustTracker, "start_experiment"):
            with patch.object(BraintrustTracker, "end_experiment"):
                cm = BraintrustContextManager()
                cm.__enter__()
                assert cm.__exit__(None, None, None) is False

    def test_used_as_context_manager(self):
        with patch.object(BraintrustTracker, "start_experiment"):
            with patch.object(BraintrustTracker, "end_experiment"):
                with BraintrustContextManager() as tracker:
                    assert isinstance(tracker, BraintrustTracker)


@pytest.mark.unit
class TestCreateTrainingTracker:
    """Tests for create_training_tracker factory."""

    def test_returns_tracker(self):
        tracker = create_training_tracker()
        assert isinstance(tracker, BraintrustTracker)

    def test_default_project_name(self):
        tracker = create_training_tracker()
        assert tracker.project_name == "neural-meta-controller"

    def test_starts_experiment_when_available(self):
        with patch.object(BraintrustTracker, "start_experiment") as mock_start:
            with patch.object(
                BraintrustTracker, "is_available",
                new_callable=lambda: property(lambda self: True),
            ):
                tracker = create_training_tracker(model_type="bert", config={"epochs": 10})
                mock_start.assert_called_once()
                call_kwargs = mock_start.call_args[1]
                assert call_kwargs["experiment_name"].startswith("bert_training_")
                assert call_kwargs["metadata"]["model_type"] == "bert"
                assert call_kwargs["metadata"]["epochs"] == 10

    def test_does_not_start_experiment_when_not_available(self):
        with patch.object(BraintrustTracker, "start_experiment") as mock_start:
            with patch.object(
                BraintrustTracker, "is_available",
                new_callable=lambda: property(lambda self: False),
            ):
                create_training_tracker()
                mock_start.assert_not_called()
