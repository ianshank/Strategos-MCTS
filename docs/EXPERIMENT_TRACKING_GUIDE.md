# Experiment Tracking Integration Guide

## Overview

The LangGraph Multi-Agent MCTS framework supports two experiment tracking platforms:

1. **Braintrust** - Fully integrated with custom tracker module
2. **Weights & Biases (Wandb)** - Basic support via HuggingFace integration

Both platforms enable tracking of training metrics, hyperparameters, and model performance for the Neural Meta-Controller.

## Braintrust Integration

### Features

- **Full Integration**: Custom `BraintrustTracker` class in `src/observability/braintrust_tracker.py`
- **Automatic Buffering**: Metrics are buffered when offline and synced when connected
- **Context Manager**: Easy experiment management with context managers
- **Training Integration**: Built into RNN training pipeline with `--use_braintrust` flag

### Setup

1. **Install Package**:
```bash
pip install braintrust
```

2. **Get API Key**:
   - Sign up at [https://www.braintrust.dev/](https://www.braintrust.dev/)
   - Get your API key from the dashboard

3. **Configure Environment**:
```bash
# Option A: Environment variable
export BRAINTRUST_API_KEY="your-api-key-here"

# Option B: .env file
echo "BRAINTRUST_API_KEY=your-api-key-here" >> .env
```

### Usage

#### In Training Scripts

```bash
# Train RNN with Braintrust tracking
python src/training/train_rnn.py \
    --use_braintrust \
    --experiment_name "rnn_experiment_v1" \
    --epochs 50
```

#### Programmatic Usage

```python
from src.observability.braintrust_tracker import BraintrustTracker

# Initialize tracker
tracker = BraintrustTracker(project_name="neural-meta-controller")

# Start experiment
tracker.start_experiment(
    experiment_name="my_experiment",
    metadata={"model": "rnn", "version": "1.0"}
)

# Log hyperparameters
tracker.log_hyperparameters({
    "learning_rate": 0.001,
    "batch_size": 32,
    "hidden_dim": 64
})

# Log training metrics
tracker.log_epoch_summary(
    epoch=1,
    train_loss=0.5,
    val_loss=0.4,
    train_accuracy=0.85,
    val_accuracy=0.82
)

# End experiment
url = tracker.end_experiment()
print(f"View experiment: {url}")
```

#### Context Manager

```python
from src.observability.braintrust_tracker import BraintrustContextManager

with BraintrustContextManager(
    project_name="neural-meta-controller",
    experiment_name="training_run_1"
) as tracker:
    # Training code here
    tracker.log_hyperparameters(config)
    for epoch in range(num_epochs):
        # Train...
        tracker.log_epoch_summary(epoch, metrics)
```

### Features

- **Experiment Tracking**: Automatic versioning and comparison
- **Metric Logging**: Loss, accuracy, per-class metrics
- **Hyperparameter Tracking**: All training configurations
- **Model Artifacts**: Track saved model paths and performance
- **Offline Support**: Automatic buffering when disconnected
- **Training Integration**: Native support in RNN training script

## Weights & Biases (Wandb) Integration

### Features

- **Standard Tracking**: Full experiment tracking capabilities
- **Visualization**: Rich UI for metric visualization
- **HuggingFace Support**: Native integration with Transformers library
- **Offline Mode**: Log locally and sync later

### Setup

1. **Install Package**:
```bash
pip install wandb
```

2. **Login/Configure**:
```bash
# Interactive login
wandb login

# Or set API key
export WANDB_API_KEY="your-api-key-here"
```

3. **Get API Key**:
   - Sign up at [https://wandb.ai/](https://wandb.ai/)
   - Get key from [https://wandb.ai/settings](https://wandb.ai/settings)

### Usage

#### Basic Usage

```python
import wandb

# Initialize run
run = wandb.init(
    project="langgraph-mcts",
    name="meta_controller_training",
    config={
        "learning_rate": 0.001,
        "architecture": "RNN",
        "dataset": "synthetic"
    }
)

# Log metrics
wandb.log({
    "epoch": epoch,
    "train_loss": train_loss,
    "val_loss": val_loss,
    "accuracy": accuracy
})

# Log summary
wandb.summary["best_accuracy"] = best_accuracy

# Finish run
wandb.finish()
```

#### With HuggingFace Trainer

Modify `src/training/train_bert_lora.py`:

```python
training_args = TrainingArguments(
    output_dir=output_dir,
    report_to="wandb",  # Enable wandb
    run_name="bert_meta_controller",
    logging_steps=10,
    # ... other args
)
```

#### Offline Mode

```python
# For offline logging
run = wandb.init(
    project="langgraph-mcts",
    mode="offline"
)

# Later sync with:
# wandb sync [run_directory]
```

## Comparison

| Feature | Braintrust | Wandb |
|---------|------------|--------|
| Installation | `pip install braintrust` | `pip install wandb` |
| RNN Training Integration | ✅ Full integration | ❌ Manual setup needed |
| BERT Training Integration | ❌ Not integrated | ✅ HuggingFace native |
| Offline Buffering | ✅ Automatic | ✅ With offline mode |
| Custom Tracker Class | ✅ `BraintrustTracker` | ❌ Use standard API |
| Context Manager | ✅ Available | ❌ Use standard API |
| Visualization | ✅ Web UI | ✅ Rich web UI |
| Free Tier | ✅ Available | ✅ Available |

## Recommendations

### For RNN Meta-Controller Training
Use **Braintrust** - it's fully integrated:
```bash
python src/training/train_rnn.py --use_braintrust
```

### For BERT Meta-Controller Training
Use **Wandb** - it's natively supported by HuggingFace:
```python
# In train_bert_lora.py
report_to="wandb"
```

### For Custom Experiments
Either platform works well. Choose based on:
- **Braintrust**: If you want to use the provided tracker class
- **Wandb**: If you prefer more visualization options

## Verification

Run the verification script to check your setup:

```bash
python scripts/verification/verify_braintrust_wandb_integration.py
```

This will:
1. Check package installation
2. Verify API key configuration
3. Test basic functionality
4. Show integration status

## Advanced Features

### Braintrust Advanced

```python
# Log individual predictions
tracker.log_model_prediction(
    input_features={
        "task_complexity": 0.8,
        "computational_intensity": 0.6
    },
    prediction="trm",
    confidence=0.85,
    ground_truth="trm"
)

# Log model artifacts
tracker.log_model_artifact(
    model_path="models/best_model.pt",
    model_type="rnn",
    metrics={"accuracy": 0.92, "f1": 0.89}
)
```

### Wandb Advanced

```python
# Log artifacts
wandb.log_artifact("model.pt", type="model")

# Log tables
table = wandb.Table(
    columns=["epoch", "accuracy", "loss"],
    data=[[1, 0.8, 0.5], [2, 0.85, 0.4]]
)
wandb.log({"results": table})

# Log images/plots
wandb.log({"confusion_matrix": wandb.Image(plt)})
```

## Troubleshooting

### Braintrust Issues

1. **"braintrust not installed"**
   ```bash
   pip install braintrust
   ```

2. **"API key not found"**
   ```bash
   export BRAINTRUST_API_KEY="your-key"
   ```

3. **Connection errors**
   - Check internet connection
   - Verify API key is valid
   - Check Braintrust service status

### Wandb Issues

1. **"wandb not installed"**
   ```bash
   pip install wandb
   ```

2. **Authentication errors**
   ```bash
   wandb login
   ```

3. **Sync issues in offline mode**
   ```bash
   wandb sync wandb/offline-run-[id]
   ```

## Environment Variables

Add to your `.env` file:

```env
# Braintrust
BRAINTRUST_API_KEY=your-braintrust-api-key

# Wandb
WANDB_API_KEY=your-wandb-api-key
WANDB_PROJECT=langgraph-mcts
WANDB_ENTITY=your-username-or-team
```

## Best Practices

1. **Use meaningful experiment names**: Include model type, dataset, and key params
2. **Log all hyperparameters**: Makes experiments reproducible
3. **Track both training and validation metrics**: Monitor overfitting
4. **Use tags**: Organize experiments by type, model, dataset
5. **Log model artifacts**: Keep track of best models
6. **Document runs**: Add notes about what changed
