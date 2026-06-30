# Meta-Controller Training (Phase 5.3)

The neural meta-controller routes each query to the best agent (HRM / TRM / MCTS). This
guide covers the **learning loop** that improves routing from observed decisions, added in
Phase 5.3.

## Components

| Piece | Location |
|---|---|
| Feature extraction | `src/agents/meta_controller/feature_extractor.py` → `MetaControllerFeatures` |
| Controllers (models) | `src/agents/meta_controller/{rnn_controller,bert_controller}.py` |
| **Data collection + train/validate** | `src/training/meta_controller_data_collector.py` |
| Orchestrator (curriculum/calibration) | `src/training/meta_controller_trainer.py` |

## Pipeline

1. **Collect** labeled routing decisions during self-play / inference. Each decision is the
   agent chosen for a state plus the realized outcome (e.g. terminal reward, downstream
   confidence). Features are vectorized deterministically by `features_to_vector(...)`
   (`FEATURE_DIM = 13`: agent confidences, MCTS value, consensus, iteration, normalized
   query length, RAG flags, technical flag, and a one-hot of the last agent).

   ```python
   from src.training.meta_controller_data_collector import MetaControllerDataCollector

   collector = MetaControllerDataCollector()
   # During self-play, once the best agent in hindsight is known:
   collector.record_features(features, agent="hrm", outcome=terminal_reward)
   ```

2. **Train + validate** any logits-producing controller model (RNN/BERT/linear). The loop is
   seeded and reproducible, splits a held-out validation set, and reports accuracy against a
   majority-class baseline:

   ```python
   from src.training.meta_controller_data_collector import train_and_validate, AGENT_LABELS, FEATURE_DIM
   from torch import nn

   model = nn.Linear(FEATURE_DIM, len(AGENT_LABELS))  # or an RNN/BERT controller
   report = train_and_validate(model, collector, epochs=20, learning_rate=1e-2, seed=42)
   print(report.val_accuracy, report.baseline_accuracy, report.accuracy_lift)
   ```

   `MetaControllerTrainingReport` carries `final_train_loss`, `val_accuracy`,
   `baseline_accuracy`, and `accuracy_lift` (validation accuracy minus the majority baseline).

## Conventions & guarantees

- **Reproducible**: `train_and_validate(..., seed=...)` seeds torch/numpy and uses a seeded
  shuffle; identical model init + seed → identical results.
- **Decoupled**: the loop accepts any `nn.Module` mapping `(batch, FEATURE_DIM)` →
  `(batch, len(AGENT_LABELS))` logits, so it composes with the existing controllers without
  importing them.
- **No hardcoded values**: agent label space (`AGENT_LABELS`), feature width (`FEATURE_DIM`),
  and the query-length normalization are named module constants.

## Measuring routing quality

Report `accuracy_lift` (vs the majority-class baseline) on a held-out split. For end-to-end
decision-quality lift of the *system* (not just routing), use the policy-comparison benchmark
(`src/benchmark/policy_comparison.py`, Phase 5.4).
