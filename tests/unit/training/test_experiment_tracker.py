"""Unit tests for src/training/experiment_tracker module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.training.experiment_tracker import (
    BraintrustTracker,
    ExperimentConfig,
    TrainingMetrics,
    UnifiedExperimentTracker,
    WandBTracker,
)


@pytest.mark.unit
class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_defaults(self):
        cfg = ExperimentConfig(project_name="proj", experiment_name="exp1")
        assert cfg.project_name == "proj"
        assert cfg.experiment_name == "exp1"
        assert cfg.tags == []
        assert cfg.description == ""
        assert cfg.save_artifacts is True
        assert cfg.log_frequency == 1

    def test_custom_values(self):
        cfg = ExperimentConfig(
            project_name="p",
            experiment_name="e",
            tags=["v1", "test"],
            description="desc",
            save_artifacts=False,
            log_frequency=10,
        )
        assert cfg.tags == ["v1", "test"]
        assert cfg.save_artifacts is False
        assert cfg.log_frequency == 10


@pytest.mark.unit
class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_required_fields(self):
        m = TrainingMetrics(epoch=1, step=10, train_loss=0.5)
        assert m.epoch == 1
        assert m.step == 10
        assert m.train_loss == 0.5
        assert m.val_loss is None
        assert m.accuracy is None
        assert m.learning_rate is None
        assert isinstance(m.timestamp, float)
        assert m.custom_metrics == {}

    def test_all_fields(self):
        m = TrainingMetrics(
            epoch=2,
            step=20,
            train_loss=0.3,
            val_loss=0.4,
            accuracy=0.85,
            learning_rate=0.001,
            custom_metrics={"f1": 0.9},
        )
        assert m.val_loss == 0.4
        assert m.accuracy == 0.85
        assert m.learning_rate == 0.001
        assert m.custom_metrics["f1"] == 0.9


@pytest.mark.unit
class TestBraintrustTracker:
    """Tests for BraintrustTracker."""

    def test_init_no_api_key_offline_mode(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove BRAINTRUST_API_KEY if set
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            assert tracker._offline_mode is True
            assert tracker._initialized is False

    def test_init_with_api_key_import_error(self):
        with patch("builtins.__import__", side_effect=ImportError("no braintrust")):
            tracker = BraintrustTracker(api_key="fake-key")
            # Should fall back to offline mode when import fails
            assert tracker._offline_mode is True

    def test_init_experiment_offline(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            exp_id = tracker.init_experiment("test_exp", description="testing", tags=["v1"])

            assert exp_id.startswith("offline_")
            assert tracker._experiment_id == exp_id
            assert tracker._experiment_config["name"] == "test_exp"
            assert tracker._experiment_config["description"] == "testing"
            assert tracker._experiment_config["tags"] == ["v1"]

    def test_log_hyperparameters_offline(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            tracker.init_experiment("test_exp")
            params = {"lr": 0.001, "batch_size": 32}
            tracker.log_hyperparameters(params)

            assert tracker._experiment_config["hyperparameters"] == params

    def test_log_metric_offline(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            tracker.log_metric("loss", 0.5, step=1)

            assert len(tracker._metrics_buffer) == 1
            assert tracker._metrics_buffer[0]["name"] == "loss"
            assert tracker._metrics_buffer[0]["value"] == 0.5

    def test_log_metric_auto_step(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            tracker.log_metric("loss", 0.5)
            tracker.log_metric("loss", 0.4)

            assert tracker._metrics_buffer[0]["step"] == 0
            assert tracker._metrics_buffer[1]["step"] == 1

    def test_log_training_step_offline(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            metrics = TrainingMetrics(
                epoch=1,
                step=5,
                train_loss=0.5,
                val_loss=0.6,
                accuracy=0.8,
                learning_rate=0.001,
                custom_metrics={"f1": 0.75},
            )
            tracker.log_training_step(metrics)

            # Should log train_loss, val_loss, accuracy, learning_rate, f1
            assert len(tracker._metrics_buffer) == 5
            names = [m["name"] for m in tracker._metrics_buffer]
            assert "train_loss" in names
            assert "val_loss" in names
            assert "accuracy" in names
            assert "learning_rate" in names
            assert "f1" in names

    def test_log_training_step_minimal(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            metrics = TrainingMetrics(epoch=1, step=5, train_loss=0.5)
            tracker.log_training_step(metrics)

            # Should only log train_loss
            assert len(tracker._metrics_buffer) == 1

    def test_log_evaluation_offline(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            # Should not raise
            tracker.log_evaluation("input", "output", "expected", {"acc": 0.9})

    def test_log_artifact_nonexistent_path(self, tmp_path):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            tracker.init_experiment("test")
            # Should warn and return without error
            tracker.log_artifact(tmp_path / "nonexistent.pt", name="model")

    def test_log_artifact_offline(self, tmp_path):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            tracker.init_experiment("test")

            artifact = tmp_path / "model.pt"
            artifact.write_text("fake model")
            tracker.log_artifact(artifact, name="model")

            assert str(artifact) in tracker._experiment_config["artifacts"]

    def test_get_summary_offline(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            tracker.init_experiment("test_exp")
            tracker.log_metric("loss", 0.5)

            summary = tracker.get_summary()
            assert summary["offline"] is True
            assert summary["metrics_count"] == 1
            assert summary["id"] is not None

    def test_end_experiment_offline(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            tracker.init_experiment("test_exp")
            tracker.log_metric("loss", 0.5)

            summary = tracker.end_experiment()
            assert summary["metrics_count"] == 1
            assert tracker._experiment is None
            assert tracker._experiment_id is None
            assert tracker._metrics_buffer == []

    def test_project_name_default(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None)
            assert tracker.project_name == "mcts-neural-meta-controller"

    def test_project_name_custom(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            tracker = BraintrustTracker(api_key=None, project_name="my-project")
            assert tracker.project_name == "my-project"


@pytest.mark.unit
class TestWandBTracker:
    """Tests for WandBTracker."""

    def test_init_no_api_key_offline(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            tracker = WandBTracker(api_key=None)
            assert tracker._offline_mode is True

    def test_init_wandb_mode_offline(self):
        with patch.dict(os.environ, {"WANDB_MODE": "offline"}, clear=True):
            tracker = WandBTracker(api_key=None)
            assert tracker._offline_mode is True

    def test_init_with_api_key_import_error(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("WANDB_MODE", None)
            with patch("builtins.__import__", side_effect=ImportError("no wandb")):
                tracker = WandBTracker(api_key="fake-key")
                assert tracker._offline_mode is True

    def test_init_run_offline(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            tracker = WandBTracker(api_key=None)
            result = tracker.init_run("test_run", config={"lr": 0.01})
            assert result is None
            assert tracker._run_config == {"lr": 0.01}

    def test_log_offline(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            tracker = WandBTracker(api_key=None)
            # Should not raise
            tracker.log({"loss": 0.5}, step=1)

    def test_log_training_step_offline(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            tracker = WandBTracker(api_key=None)
            metrics = TrainingMetrics(
                epoch=1,
                step=5,
                train_loss=0.5,
                val_loss=0.6,
                accuracy=0.8,
                learning_rate=0.001,
            )
            # Should not raise
            tracker.log_training_step(metrics)

    def test_update_config_offline(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            tracker = WandBTracker(api_key=None)
            tracker.init_run("test", config={"lr": 0.01})
            tracker.update_config({"batch_size": 64})
            assert tracker._run_config["batch_size"] == 64
            assert tracker._run_config["lr"] == 0.01

    def test_watch_model_offline(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            tracker = WandBTracker(api_key=None)
            # Should return without error
            tracker.watch_model(MagicMock())

    def test_log_artifact_offline(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            tracker = WandBTracker(api_key=None)
            # Should not raise
            tracker.log_artifact("/fake/path", "model", "model")

    def test_finish_offline(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            tracker = WandBTracker(api_key=None)
            # Should not raise
            tracker.finish()

    def test_project_name(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            tracker = WandBTracker(api_key=None, project_name="custom")
            assert tracker.project_name == "custom"

    def test_entity(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            tracker = WandBTracker(api_key=None, entity="my-team")
            assert tracker.entity == "my-team"


@pytest.mark.unit
class TestUnifiedExperimentTracker:
    """Tests for UnifiedExperimentTracker."""

    def _make_tracker(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            os.environ.pop("WANDB_API_KEY", None)
            os.environ.pop("WANDB_MODE", None)
            return UnifiedExperimentTracker()

    def test_init(self):
        tracker = self._make_tracker()
        assert isinstance(tracker.bt, BraintrustTracker)
        assert isinstance(tracker.wandb, WandBTracker)
        assert tracker.project_name == "mcts-neural-meta-controller"

    def test_init_experiment(self):
        tracker = self._make_tracker()
        tracker.init_experiment(
            "test",
            config={"lr": 0.01},
            description="desc",
            tags=["v1"],
        )
        # bt should have experiment initialized
        assert tracker.bt._experiment_id is not None
        assert tracker.bt._experiment_config["hyperparameters"] == {"lr": 0.01}

    def test_log_metrics(self):
        tracker = self._make_tracker()
        metrics = TrainingMetrics(epoch=1, step=5, train_loss=0.5)
        tracker.log_metrics(metrics)
        # bt should have metrics logged
        assert len(tracker.bt._metrics_buffer) == 1

    def test_log_evaluation(self):
        tracker = self._make_tracker()
        tracker.init_experiment("test")
        # Should not raise
        tracker.log_evaluation("input", "output", "expected", {"acc": 0.9})

    def test_log_artifact(self, tmp_path):
        tracker = self._make_tracker()
        tracker.bt.init_experiment("test")
        artifact = tmp_path / "model.pt"
        artifact.write_text("fake")
        tracker.log_artifact(str(artifact), "model")

    def test_finish(self):
        tracker = self._make_tracker()
        tracker.init_experiment("test")
        tracker.bt.log_metric("loss", 0.5)
        summary = tracker.finish()

        assert summary["metrics_count"] == 1
        assert tracker.bt._experiment is None
