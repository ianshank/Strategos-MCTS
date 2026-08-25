#!/usr/bin/env python3
"""
Neural Meta-Controller Demo

Demonstrates the complete Neural Meta-Controller system including:
- RNN and BERT-based agent routing
- Synthetic data generation and training
- Feature extraction and normalization
- Braintrust experiment tracking (optional)
- Pinecone vector storage (optional)
- Configuration management

Usage:
    python demos/neural_meta_controller_demo.py
"""

from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.meta_controller.base import (  # noqa: E402
    MetaControllerFeatures,
    MetaControllerPrediction,
)
from src.agents.meta_controller.bert_controller import BERTMetaController  # noqa: E402
from src.agents.meta_controller.config_loader import (  # noqa: E402
    MetaControllerConfigLoader,
)
from src.agents.meta_controller.rnn_controller import RNNMetaController  # noqa: E402
from src.agents.meta_controller.utils import (  # noqa: E402
    features_to_text,
    normalize_features,
)
from src.training.data_generator import MetaControllerDataGenerator  # noqa: E402
from src.training.train_rnn import RNNTrainer  # noqa: E402


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str) -> None:
    """Print a formatted subheader."""
    print(f"\n--- {title} ---")


def demo_feature_extraction() -> None:
    """Demonstrate feature extraction from agent state."""
    print_header("Feature Extraction & Normalization")

    # Create sample features
    features = MetaControllerFeatures(
        hrm_confidence=0.85,
        trm_confidence=0.72,
        mcts_value=0.68,
        consensus_score=0.75,
        last_agent="hrm",
        iteration=3,
        query_length=250,
        has_rag_context=True,
    )

    print("Input Features:")
    print(f"  HRM Confidence: {features.hrm_confidence}")
    print(f"  TRM Confidence: {features.trm_confidence}")
    print(f"  MCTS Value: {features.mcts_value}")
    print(f"  Consensus Score: {features.consensus_score}")
    print(f"  Last Agent: {features.last_agent}")
    print(f"  Iteration: {features.iteration}")
    print(f"  Query Length: {features.query_length}")
    print(f"  Has RAG Context: {features.has_rag_context}")

    print_subheader("Normalized 10-D Vector")
    normalized = normalize_features(features)
    labels = [
        "hrm_conf",
        "trm_conf",
        "mcts_val",
        "consensus",
        "last_hrm",
        "last_trm",
        "last_mcts",
        "iter_norm",
        "query_norm",
        "has_rag",
    ]
    for i, (label, value) in enumerate(zip(labels, normalized, strict=True)):
        print(f"  [{i}] {label}: {value:.4f}")

    print_subheader("Text Representation (for BERT)")
    text = features_to_text(features)
    print(text)


def demo_rnn_controller() -> None:
    """Demonstrate RNN-based meta-controller."""
    print_header("RNN Meta-Controller")

    # Create controller
    print("Creating RNN Meta-Controller...")
    controller = RNNMetaController(name="DemoRNN", seed=42, hidden_dim=64)
    print(f"  Name: {controller.name}")
    print(f"  Device: {controller.device}")
    print(f"  Hidden Dimension: {controller.hidden_dim}")
    print(f"  Number of Layers: {controller.num_layers}")

    # Test predictions
    print_subheader("Predictions on Sample States")

    test_cases = [
        ("High HRM confidence", MetaControllerFeatures(0.95, 0.4, 0.3, 0.8, "hrm", 1, 100, True)),
        ("High TRM confidence", MetaControllerFeatures(0.3, 0.92, 0.4, 0.7, "trm", 2, 200, False)),
        ("High MCTS value", MetaControllerFeatures(0.4, 0.5, 0.88, 0.6, "mcts", 5, 500, True)),
        ("Balanced state", MetaControllerFeatures(0.6, 0.6, 0.6, 0.6, "none", 3, 300, True)),
    ]

    for name, features in test_cases:
        prediction = controller.predict(features)
        print(f"\n  {name}:")
        print(f"    Selected Agent: {prediction.agent.upper()}")
        print(f"    Confidence: {prediction.confidence:.4f}")
        print(
            f"    Probabilities: HRM={prediction.probabilities['hrm']:.3f}, "
            f"TRM={prediction.probabilities['trm']:.3f}, "
            f"MCTS={prediction.probabilities['mcts']:.3f}"
        )


def demo_bert_controller() -> None:
    """Demonstrate BERT-based meta-controller with LoRA."""
    print_header("BERT Meta-Controller with LoRA")

    print("Creating BERT Meta-Controller (this may take a moment)...")
    controller = BERTMetaController(name="DemoBERT", seed=42, use_lora=True)

    print(f"  Name: {controller.name}")
    print(f"  Device: {controller.device}")
    print(f"  Model: {controller.model_name}")
    print(f"  Use LoRA: {controller.use_lora}")
    print(f"  LoRA r: {controller.lora_r}")
    print(f"  LoRA alpha: {controller.lora_alpha}")

    # Show parameter efficiency
    print_subheader("Parameter Efficiency (LoRA)")
    params = controller.get_trainable_parameters()
    print(f"  Total Parameters: {params['total_params']:,}")
    print(f"  Trainable Parameters: {params['trainable_params']:,}")
    print(f"  Trainable Percentage: {params['trainable_percentage']:.2f}%")
    print(f"  Memory Savings: ~{100 - params['trainable_percentage']:.1f}%")

    # Test prediction with caching
    print_subheader("Prediction with Tokenization Caching")
    controller.clear_cache()

    features = MetaControllerFeatures(0.7, 0.8, 0.6, 0.75, "trm", 4, 350, True)

    start = datetime.now()
    pred1 = controller.predict(features)
    first_time = (datetime.now() - start).total_seconds()

    start = datetime.now()
    _ = controller.predict(features)  # Test cached prediction
    second_time = (datetime.now() - start).total_seconds()

    print(f"  First prediction time: {first_time:.4f}s")
    print(f"  Cached prediction time: {second_time:.4f}s")
    print(f"  Speedup: {first_time / second_time:.2f}x")

    cache_info = controller.get_cache_info()
    print(f"  Cache size: {cache_info['cache_size']} entries")

    print(f"\n  Selected Agent: {pred1.agent.upper()}")
    print(f"  Confidence: {pred1.confidence:.4f}")


def demo_data_generation() -> None:
    """Demonstrate synthetic data generation."""
    print_header("Synthetic Data Generation")

    generator = MetaControllerDataGenerator(seed=42)

    print("Generating balanced dataset (100 samples per class)...")
    features_list, labels_list = generator.generate_balanced_dataset(100)

    # Analyze distribution
    label_counts = {"hrm": 0, "trm": 0, "mcts": 0}
    for label in labels_list:
        label_counts[label] += 1

    print(f"  Total samples: {len(features_list)}")
    print(f"  HRM samples: {label_counts['hrm']}")
    print(f"  TRM samples: {label_counts['trm']}")
    print(f"  MCTS samples: {label_counts['mcts']}")

    # Convert to tensors
    print_subheader("Tensor Conversion")
    X, y = generator.to_tensor_dataset(features_list, labels_list)
    print(f"  Feature tensor shape: {X.shape}")
    print(f"  Label tensor shape: {y.shape}")
    print(f"  Feature dtype: {X.dtype}")
    print(f"  Label dtype: {y.dtype}")

    # Split dataset
    print_subheader("Dataset Splitting (70/15/15)")
    splits = generator.split_dataset(X, y, train_ratio=0.7, val_ratio=0.15)
    print(f"  Training set: {splits['X_train'].shape[0]} samples")
    print(f"  Validation set: {splits['X_val'].shape[0]} samples")
    print(f"  Test set: {splits['X_test'].shape[0]} samples")


def demo_training_pipeline() -> None:
    """Demonstrate the training pipeline."""
    print_header("Training Pipeline Demo")

    # Generate small dataset for quick demo
    print("Generating training data...")
    generator = MetaControllerDataGenerator(seed=42)
    features_list, labels_list = generator.generate_balanced_dataset(30)
    X, y = generator.to_tensor_dataset(features_list, labels_list)
    splits = generator.split_dataset(X, y)

    # Create trainer
    print_subheader("Model Configuration")
    trainer = RNNTrainer(
        hidden_dim=32,
        num_layers=1,
        dropout=0.1,
        lr=1e-3,
        batch_size=16,
        epochs=3,  # Short for demo
        early_stopping_patience=2,
        seed=42,
    )

    # Train model
    print_subheader("Training Progress")
    history = trainer.train(
        train_data=(splits["X_train"], splits["y_train"]),
        val_data=(splits["X_val"], splits["y_val"]),
    )

    # Show results
    print_subheader("Training Results")
    print(f"  Total epochs: {history['total_epochs']}")
    print(f"  Best epoch: {history['best_epoch']}")
    print(f"  Best validation loss: {history['best_val_loss']:.4f}")
    print(f"  Best validation accuracy: {history['best_val_accuracy']:.4f}")
    print(f"  Stopped early: {history['stopped_early']}")

    # Evaluate
    print_subheader("Test Set Evaluation")
    test_loader = trainer.create_dataloader(splits["X_test"], splits["y_test"], shuffle=False)
    results = trainer.evaluate(test_loader)

    print(f"  Test accuracy: {results['accuracy']:.4f}")
    print(f"  Test loss: {results['loss']:.4f}")
    print("\n  Per-class F1 scores:")
    for agent, metrics in results["per_class_metrics"].items():
        print(f"    {agent}: {metrics['f1_score']:.4f}")


def demo_braintrust_integration() -> None:
    """Demonstrate Braintrust experiment tracking."""
    print_header("Braintrust Experiment Tracking (Optional)")

    try:
        from src.observability.braintrust_tracker import (
            BRAINTRUST_AVAILABLE,
            BraintrustTracker,
        )

        if not BRAINTRUST_AVAILABLE:
            print("  Braintrust not installed. Install with: pip install braintrust")
            return

        print("  Creating Braintrust tracker...")
        tracker = BraintrustTracker(project_name="neural-meta-controller", auto_init=True)

        if tracker.is_available:
            print("  [OK] Braintrust connection established")
            print("    Project: neural-meta-controller")
            print("    Ready for experiment tracking")
        else:
            print("  [FAIL] Braintrust not configured (missing API key)")
            print("    Set BRAINTRUST_API_KEY in .env file")

        # Show buffering capability
        print_subheader("Offline Buffering Capability")
        test_tracker = BraintrustTracker(api_key=None, auto_init=False)
        test_tracker.log_hyperparameters({"learning_rate": 0.001})
        test_tracker.log_epoch_summary(1, train_loss=0.5, val_loss=0.4)

        print(f"  Buffered operations: {len(test_tracker.get_buffered_metrics())}")
        print("  Operations will be sent when connection is available")

    except ImportError:
        print("  Braintrust module not found. Install with: pip install braintrust")


def demo_pinecone_integration() -> None:
    """Demonstrate Pinecone vector storage."""
    print_header("Pinecone Vector Storage (Optional)")

    try:
        from src.storage.pinecone_store import (
            PINECONE_AVAILABLE,
            PineconeVectorStore,
        )

        if not PINECONE_AVAILABLE:
            print("  Pinecone not installed. Install with: pip install pinecone")
            return

        print("  Creating Pinecone vector store...")
        store = PineconeVectorStore(namespace="demo", auto_init=True)

        if store.is_available:
            print("  [OK] Pinecone connection established")
            stats = store.get_stats()
            print(f"    Total vectors: {stats.get('total_vectors', 0)}")
            print(f"    Vector dimension: {store.VECTOR_DIMENSION}")
        else:
            print("  [FAIL] Pinecone not configured (missing API key/host)")
            print("    Set PINECONE_API_KEY and PINECONE_HOST in .env file")

        # Show vector storage concept
        print_subheader("Vector Storage Concept")
        features = MetaControllerFeatures(0.7, 0.8, 0.6, 0.75, "trm", 4, 350, True)
        vector = normalize_features(features)

        print(f"  Feature vector (10D): {[f'{v:.3f}' for v in vector[:5]]}...")
        print("  This vector can be stored and queried for similar decisions")
        print("  Enables retrieval-augmented routing based on historical patterns")

        # Show buffering
        print_subheader("Offline Buffering")
        test_store = PineconeVectorStore(api_key=None, host=None, auto_init=False)
        prediction = MetaControllerPrediction("hrm", 0.85, {"hrm": 0.85, "trm": 0.10, "mcts": 0.05})
        test_store.store_prediction(features, prediction)

        print(f"  Buffered operations: {len(test_store.get_buffered_operations())}")
        print("  Operations will be stored when connection is available")

    except ImportError:
        print("  Pinecone module not found. Install with: pip install pinecone")


def demo_configuration() -> None:
    """Demonstrate configuration management."""
    print_header("Configuration Management")

    # Get default config
    default_config = MetaControllerConfigLoader.get_default_config()

    print("Default Configuration:")
    print(f"  Enabled: {default_config.enabled}")
    print(f"  Type: {default_config.type}")
    print(f"  Fallback to rule-based: {default_config.fallback_to_rule_based}")

    print_subheader("RNN Configuration")
    print(f"  Hidden dimension: {default_config.rnn.hidden_dim}")
    print(f"  Number of layers: {default_config.rnn.num_layers}")
    print(f"  Dropout: {default_config.rnn.dropout}")

    print_subheader("BERT Configuration")
    print(f"  Model name: {default_config.bert.model_name}")
    print(f"  Use LoRA: {default_config.bert.use_lora}")
    print(f"  LoRA r: {default_config.bert.lora_r}")
    print(f"  LoRA alpha: {default_config.bert.lora_alpha}")

    # Validate config
    print_subheader("Configuration Validation")
    try:
        MetaControllerConfigLoader.validate(default_config)
        print("  [OK] Default configuration is valid")
    except ValueError as e:
        print(f"  [FAIL] Configuration error: {e}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("    NEURAL META-CONTROLLER DEMONSTRATION")
    print("    LangGraph Multi-Agent MCTS Framework")
    print("=" * 70)

    demos = [
        ("Feature Extraction", demo_feature_extraction),
        ("RNN Controller", demo_rnn_controller),
        ("BERT Controller", demo_bert_controller),
        ("Data Generation", demo_data_generation),
        ("Training Pipeline", demo_training_pipeline),
        ("Braintrust Integration", demo_braintrust_integration),
        ("Pinecone Integration", demo_pinecone_integration),
        ("Configuration", demo_configuration),
    ]

    print(f"\nRunning {len(demos)} demonstrations...\n")

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n[ERROR] Error in {name} demo: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("    DEMO COMPLETE")
    print("=" * 70)
    print("\nFor more information, see:")
    print("  - README.md")
    print("  - tests/test_meta_controller.py")
    print("  - tests/test_integration_e2e.py")
    print("  - src/agents/meta_controller/")
    print()


if __name__ == "__main__":
    main()
