"""
End-to-End Integration Tests for Neural Meta-Controller System.

Tests the complete integration of:
- RNN and BERT meta-controllers
- Training pipelines with data generation
- Braintrust experiment tracking (optional)
- Pinecone vector storage (optional)
- Configuration loading
- Feature extraction and normalization
"""

from pathlib import Path
import tempfile
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required")

from src.agents.meta_controller.base import (
    MetaControllerFeatures,
    MetaControllerPrediction,
)
from src.agents.meta_controller.bert_controller import BERTMetaController
from src.agents.meta_controller.rnn_controller import RNNMetaController
from src.agents.meta_controller.utils import normalize_features
from src.training.data_generator import MetaControllerDataGenerator
from src.training.train_rnn import RNNTrainer


@pytest.fixture
def sample_state() -> dict[str, Any]:
    """Create a sample agent state for testing."""
    return {
        "hrm_confidence": 0.85,
        "trm_confidence": 0.72,
        "mcts_value": 0.68,
        "consensus_score": 0.75,
        "last_agent": "hrm",
        "iteration": 3,
        "query_length": 250,
        "rag_context": "Sample context",
    }


@pytest.fixture
def multiple_features() -> list:
    """Create multiple feature instances for batch testing."""
    return [
        MetaControllerFeatures(
            hrm_confidence=0.9,
            trm_confidence=0.6,
            mcts_value=0.7,
            consensus_score=0.8,
            last_agent="hrm",
            iteration=1,
            query_length=100,
            has_rag_context=True,
        ),
        MetaControllerFeatures(
            hrm_confidence=0.5,
            trm_confidence=0.9,
            mcts_value=0.4,
            consensus_score=0.6,
            last_agent="trm",
            iteration=2,
            query_length=200,
            has_rag_context=False,
        ),
        MetaControllerFeatures(
            hrm_confidence=0.3,
            trm_confidence=0.4,
            mcts_value=0.95,
            consensus_score=0.5,
            last_agent="mcts",
            iteration=5,
            query_length=500,
            has_rag_context=True,
        ),
    ]


class TestEndToEndRNNPipeline:
    """End-to-end tests for RNN meta-controller pipeline."""

    def test_complete_training_and_inference_cycle(self):
        """Test complete cycle: data generation -> training -> inference."""
        # Generate synthetic data
        generator = MetaControllerDataGenerator(seed=42)
        features_list, labels_list = generator.generate_balanced_dataset(
            num_samples_per_class=50  # Small for fast testing
        )

        assert len(features_list) == 150
        assert len(labels_list) == 150

        # Convert to tensors
        X, y = generator.to_tensor_dataset(features_list, labels_list)
        assert X.shape == (150, 10)
        assert y.shape == (150,)

        # Split dataset
        splits = generator.split_dataset(X, y, train_ratio=0.7, val_ratio=0.15)

        # Create and train model
        trainer = RNNTrainer(
            hidden_dim=32,  # Small for testing
            num_layers=1,
            dropout=0.1,
            lr=1e-3,
            batch_size=16,
            epochs=3,  # Few epochs for testing
            early_stopping_patience=2,
            seed=42,
        )

        # Train
        history = trainer.train(
            train_data=(splits["X_train"], splits["y_train"]),
            val_data=(splits["X_val"], splits["y_val"]),
        )

        assert "train_losses" in history
        assert "val_losses" in history
        assert "val_accuracies" in history
        assert len(history["train_losses"]) > 0

        # Create controller and use trained model
        controller = RNNMetaController(name="TrainedRNN", seed=42)
        controller.model = trainer.model

        # Make predictions
        for features in features_list[:5]:
            prediction = controller.predict(features)
            assert isinstance(prediction, MetaControllerPrediction)
            assert prediction.agent in ["hrm", "trm", "mcts"]
            assert 0.0 <= prediction.confidence <= 1.0

    def test_model_save_load_roundtrip(self, multiple_features):
        """Test model saving and loading preserves behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "rnn_model.pt"

            # Create and train controller
            controller = RNNMetaController(name="Original", seed=123)

            # Get predictions before save
            predictions_before = [controller.predict(f) for f in multiple_features]

            # Save
            controller.save_model(str(model_path))
            assert model_path.exists()

            # Create new controller and load
            loaded_controller = RNNMetaController(name="Loaded", seed=456)
            loaded_controller.load_model(str(model_path))

            # Get predictions after load
            predictions_after = [loaded_controller.predict(f) for f in multiple_features]

            # Verify identical predictions
            for before, after in zip(predictions_before, predictions_after, strict=True):
                assert before.agent == after.agent
                assert before.confidence == pytest.approx(after.confidence, abs=1e-6)

    def test_training_with_checkpointing(self):
        """Test that model checkpointing works during training."""
        generator = MetaControllerDataGenerator(seed=42)
        features_list, labels_list = generator.generate_balanced_dataset(30)
        X, y = generator.to_tensor_dataset(features_list, labels_list)
        splits = generator.split_dataset(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            trainer = RNNTrainer(
                hidden_dim=16,
                epochs=2,
                batch_size=8,
                seed=42,
            )

            _history = trainer.train(
                train_data=(splits["X_train"], splits["y_train"]),
                val_data=(splits["X_val"], splits["y_val"]),
                save_path=str(checkpoint_path),
            )

            # Verify checkpoint was created
            assert checkpoint_path.exists()

            # Verify model can be loaded - must match hidden_dim from training
            new_controller = RNNMetaController(name="FromCheckpoint", seed=42, hidden_dim=16)
            new_controller.load_model(str(checkpoint_path))

            # Verify it can make predictions
            test_features = features_list[0]
            prediction = new_controller.predict(test_features)
            assert isinstance(prediction, MetaControllerPrediction)

    def test_evaluation_metrics(self):
        """Test that evaluation produces comprehensive metrics."""
        generator = MetaControllerDataGenerator(seed=42)
        features_list, labels_list = generator.generate_balanced_dataset(30)
        X, y = generator.to_tensor_dataset(features_list, labels_list)
        splits = generator.split_dataset(X, y)

        trainer = RNNTrainer(
            hidden_dim=16,
            epochs=2,
            batch_size=8,
            seed=42,
        )

        trainer.train(
            train_data=(splits["X_train"], splits["y_train"]),
            val_data=(splits["X_val"], splits["y_val"]),
        )

        test_loader = trainer.create_dataloader(splits["X_test"], splits["y_test"], shuffle=False)
        results = trainer.evaluate(test_loader)

        # Check all metrics are present
        assert "loss" in results
        assert "accuracy" in results
        assert "per_class_metrics" in results
        assert "confusion_matrix" in results
        assert "total_samples" in results

        # Check per-class metrics
        for agent in ["hrm", "trm", "mcts"]:
            assert agent in results["per_class_metrics"]
            metrics = results["per_class_metrics"][agent]
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics
            assert "support" in metrics

        # Verify confusion matrix shape
        assert len(results["confusion_matrix"]) == 3
        assert all(len(row) == 3 for row in results["confusion_matrix"])


class TestEndToEndBERTPipeline:
    """End-to-end tests for BERT meta-controller pipeline."""

    @pytest.mark.slow
    def test_bert_prediction_cycle(self, multiple_features):
        """Test BERT controller prediction cycle with caching."""
        controller = BERTMetaController(name="BERTTest", seed=42)

        # Clear cache
        controller.clear_cache()
        assert controller.get_cache_info()["cache_size"] == 0

        # Make predictions
        predictions = []
        for features in multiple_features:
            pred = controller.predict(features)
            predictions.append(pred)
            assert isinstance(pred, MetaControllerPrediction)

        # Check cache was populated
        cache_info = controller.get_cache_info()
        assert cache_info["cache_size"] == len(multiple_features)

        # Repeat predictions - should use cache
        for i, features in enumerate(multiple_features):
            pred2 = controller.predict(features)
            assert pred2.agent == predictions[i].agent
            assert pred2.confidence == pytest.approx(predictions[i].confidence, abs=1e-6)

        # Cache size should remain same
        assert controller.get_cache_info()["cache_size"] == len(multiple_features)

    @pytest.mark.slow
    def test_bert_lora_parameter_efficiency(self):
        """Test that LoRA adapters are parameter efficient."""
        controller = BERTMetaController(name="LoRATest", seed=42, use_lora=True)

        params = controller.get_trainable_parameters()

        # LoRA should have much fewer trainable params than total
        assert params["trainable_percentage"] < 50.0
        assert params["trainable_params"] < params["total_params"]

        # Should still have some trainable params
        assert params["trainable_params"] > 0

    @pytest.mark.slow
    def test_bert_save_load_adapter(self, multiple_features):
        """Test BERT LoRA adapter save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "bert_adapter"

            controller = BERTMetaController(name="Original", seed=42)

            # Get predictions
            predictions_before = [controller.predict(f) for f in multiple_features]

            # Save adapter
            controller.save_model(str(adapter_path))

            # Load into new controller
            loaded_controller = BERTMetaController(name="Loaded", seed=99)
            loaded_controller.load_model(str(adapter_path))

            # Verify predictions match
            predictions_after = [loaded_controller.predict(f) for f in multiple_features]

            for before, after in zip(predictions_before, predictions_after, strict=True):
                assert before.agent == after.agent
                assert before.confidence == pytest.approx(after.confidence, abs=1e-6)


class TestDataGeneration:
    """Tests for synthetic data generation."""

    def test_balanced_dataset_generation(self):
        """Test that balanced dataset generation works correctly."""
        generator = MetaControllerDataGenerator(seed=42)
        features_list, labels_list = generator.generate_balanced_dataset(100)

        # Check total samples
        assert len(features_list) == 300
        assert len(labels_list) == 300

        # Check balance
        label_counts = {"hrm": 0, "trm": 0, "mcts": 0}
        for label in labels_list:
            label_counts[label] += 1

        assert label_counts["hrm"] == 100
        assert label_counts["trm"] == 100
        assert label_counts["mcts"] == 100

    def test_dataset_save_load(self):
        """Test saving and loading datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.json"

            generator = MetaControllerDataGenerator(seed=42)
            features_list, labels_list = generator.generate_balanced_dataset(10)

            # Save
            generator.save_dataset(features_list, labels_list, str(dataset_path))
            assert dataset_path.exists()

            # Load
            loaded_features, loaded_labels = generator.load_dataset(str(dataset_path))

            # Verify
            assert len(loaded_features) == len(features_list)
            assert len(loaded_labels) == len(labels_list)

            for original, loaded in zip(features_list, loaded_features, strict=True):
                assert original.hrm_confidence == loaded.hrm_confidence
                assert original.iteration == loaded.iteration

    def test_tensor_conversion(self):
        """Test feature to tensor conversion."""
        generator = MetaControllerDataGenerator(seed=42)
        features_list, labels_list = generator.generate_balanced_dataset(10)

        X, y = generator.to_tensor_dataset(features_list, labels_list)

        assert X.dtype == torch.float32
        assert y.dtype == torch.long
        assert X.shape[1] == 10  # Feature dimension
        assert X.shape[0] == y.shape[0]

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same data."""
        gen1 = MetaControllerDataGenerator(seed=123)
        gen2 = MetaControllerDataGenerator(seed=123)

        features1, labels1 = gen1.generate_balanced_dataset(10)
        features2, labels2 = gen2.generate_balanced_dataset(10)

        # Should be identical
        for f1, f2 in zip(features1, features2, strict=True):
            assert f1.hrm_confidence == f2.hrm_confidence
            assert f1.iteration == f2.iteration

        assert labels1 == labels2


class TestConfigurationIntegration:
    """Tests for configuration and settings integration."""

    def test_yaml_config_loading(self):
        """Test that YAML configuration can be loaded."""
        from src.agents.meta_controller.config_loader import MetaControllerConfigLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # Write test config
            config_content = """
meta_controller:
  enabled: true
  type: "rnn"
  fallback_to_rule_based: true
  rnn:
    hidden_dim: 128
    num_layers: 2
    model_path: null
  bert:
    model_name: "prajjwal1/bert-mini"
    use_lora: true
    lora_r: 8
    lora_alpha: 32
"""
            config_path.write_text(config_content)

            # Load config
            config = MetaControllerConfigLoader.load_from_yaml(str(config_path))

            assert config.enabled is True
            assert config.type == "rnn"
            assert config.fallback_to_rule_based is True
            assert config.rnn.hidden_dim == 128
            assert config.rnn.num_layers == 2
            assert config.bert.lora_r == 8

    def test_settings_api_key_masking(self):
        """Test that API keys are properly masked in safe_dict."""
        import os

        from src.config.settings import Settings

        # Create settings with test values
        os.environ["LLM_PROVIDER"] = "lmstudio"
        os.environ["LMSTUDIO_BASE_URL"] = "http://localhost:1234/v1"
        os.environ.pop("OPENAI_API_KEY", None)

        settings = Settings()
        safe_data = settings.safe_dict()

        # Keys should be masked if present
        if "OPENAI_API_KEY" in safe_data and safe_data["OPENAI_API_KEY"]:
            assert safe_data["OPENAI_API_KEY"] == "***MASKED***"
        if "ANTHROPIC_API_KEY" in safe_data and safe_data["ANTHROPIC_API_KEY"]:
            assert safe_data["ANTHROPIC_API_KEY"] == "***MASKED***"
        if "BRAINTRUST_API_KEY" in safe_data and safe_data["BRAINTRUST_API_KEY"]:
            assert safe_data["BRAINTRUST_API_KEY"] == "***MASKED***"
        if "PINECONE_API_KEY" in safe_data and safe_data["PINECONE_API_KEY"]:
            assert safe_data["PINECONE_API_KEY"] == "***MASKED***"


class TestBraintrustIntegration:
    """Tests for Braintrust experiment tracking integration."""

    def test_tracker_without_api_key(self):
        """Test that tracker works gracefully without API key."""
        try:
            from src.observability.braintrust_tracker import BraintrustTracker

            tracker = BraintrustTracker(api_key=None, auto_init=False)

            # Should not be available
            assert not tracker.is_available

            # Should buffer operations
            tracker.log_hyperparameters({"lr": 0.001})
            assert len(tracker.get_buffered_metrics()) == 1

            tracker.log_epoch_summary(1, train_loss=0.5, val_loss=0.4)
            assert len(tracker.get_buffered_metrics()) == 2

            tracker.clear_buffer()
            assert len(tracker.get_buffered_metrics()) == 0

        except ImportError:
            pytest.skip("Braintrust not installed")

    def test_tracker_context_manager(self):
        """Test context manager pattern for experiment tracking."""
        try:
            from src.observability.braintrust_tracker import BraintrustContextManager

            with BraintrustContextManager(
                project_name="test-project",
                api_key=None,  # Will not connect
            ) as tracker:
                assert tracker is not None
                tracker.log_hyperparameters({"test": "value"})

        except ImportError:
            pytest.skip("Braintrust not installed")


class TestPineconeIntegration:
    """Tests for Pinecone vector storage integration."""

    def test_store_without_connection(self, multiple_features):
        """Test that store works gracefully without connection."""
        try:
            from src.agents.meta_controller.base import MetaControllerPrediction
            from src.storage.pinecone_store import PineconeVectorStore

            store = PineconeVectorStore(api_key=None, host=None, auto_init=False)

            # Should not be available
            assert not store.is_available

            # Create a prediction
            prediction = MetaControllerPrediction(
                agent="hrm",
                confidence=0.85,
                probabilities={"hrm": 0.85, "trm": 0.10, "mcts": 0.05},
            )

            # Should buffer the operation
            result = store.store_prediction(multiple_features[0], prediction)
            assert result is None  # Returns None when not available

            # Check buffer
            buffered = store.get_buffered_operations()
            assert len(buffered) == 1
            assert buffered[0]["type"] == "store_prediction"

            store.clear_buffer()
            assert len(store.get_buffered_operations()) == 0

        except ImportError:
            pytest.skip("Pinecone not installed")

    def test_vector_normalization_consistency(self, multiple_features):
        """Test that normalized features are consistent for vector storage."""
        for features in multiple_features:
            vector = normalize_features(features)

            # All values should be in [0, 1]
            assert all(0.0 <= v <= 1.0 for v in vector)

            # Length should be 10
            assert len(vector) == 10

            # Same features should produce same vector
            vector2 = normalize_features(features)
            assert vector == vector2


class TestFullIntegration:
    """Full system integration tests."""

    def test_complete_workflow(self):
        """Test complete workflow from data generation to prediction."""
        # 1. Generate data
        generator = MetaControllerDataGenerator(seed=42)
        features_list, labels_list = generator.generate_balanced_dataset(20)

        # 2. Convert to tensors
        X, y = generator.to_tensor_dataset(features_list, labels_list)

        # 3. Split
        splits = generator.split_dataset(X, y)

        # 4. Train RNN model
        trainer = RNNTrainer(
            hidden_dim=16,
            epochs=2,
            batch_size=8,
            seed=42,
        )

        history = trainer.train(
            train_data=(splits["X_train"], splits["y_train"]),
            val_data=(splits["X_val"], splits["y_val"]),
        )

        # 5. Evaluate
        test_loader = trainer.create_dataloader(splits["X_test"], splits["y_test"], shuffle=False)
        eval_results = trainer.evaluate(test_loader)

        # 6. Create controller for inference
        controller = RNNMetaController(name="Integrated", seed=42)
        controller.model = trainer.model

        # 7. Make predictions on new data
        new_features = MetaControllerFeatures(
            hrm_confidence=0.7,
            trm_confidence=0.8,
            mcts_value=0.6,
            consensus_score=0.7,
            last_agent="trm",
            iteration=4,
            query_length=300,
            has_rag_context=True,
        )

        prediction = controller.predict(new_features)

        # Verify everything worked
        assert len(history["train_losses"]) > 0
        assert eval_results["accuracy"] >= 0.0
        assert prediction.agent in ["hrm", "trm", "mcts"]
        assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-6)

    def test_multiple_controllers_comparison(self, multiple_features):
        """Test that different controllers can be compared."""
        rnn_controller = RNNMetaController(name="RNN", seed=42)

        # Get RNN predictions
        rnn_predictions = [rnn_controller.predict(f) for f in multiple_features]

        # All predictions should be valid
        for pred in rnn_predictions:
            assert isinstance(pred, MetaControllerPrediction)
            assert pred.agent in ["hrm", "trm", "mcts"]
            assert 0.0 <= pred.confidence <= 1.0
            assert sum(pred.probabilities.values()) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.slow
    def test_bert_rnn_comparison(self, multiple_features):
        """Compare BERT and RNN controller outputs."""
        rnn_controller = RNNMetaController(name="RNN", seed=42)
        bert_controller = BERTMetaController(name="BERT", seed=42)

        for features in multiple_features:
            rnn_pred = rnn_controller.predict(features)
            bert_pred = bert_controller.predict(features)

            # Both should produce valid predictions
            assert rnn_pred.agent in ["hrm", "trm", "mcts"]
            assert bert_pred.agent in ["hrm", "trm", "mcts"]

            # Probabilities should sum to 1
            assert sum(rnn_pred.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
            assert sum(bert_pred.probabilities.values()) == pytest.approx(1.0, abs=1e-6)

            # They might not agree on agent selection, but that's expected
            # with different architectures and untrained weights
