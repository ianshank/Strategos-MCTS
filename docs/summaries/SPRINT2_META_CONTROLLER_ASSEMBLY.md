# Sprint 2: Meta-Controller Assembly Enhancement

**Status**: ✅ Complete
**Duration**: Sprint 2
**Stories**: 2.2, 2.3, 2.4, 2.5
**Total Story Points**: ~13 points

## Overview

Sprint 2 enhances the meta-controller with Assembly Theory integration, providing interpretable routing decisions and improved agent selection through hybrid neural-assembly approaches.

## Implemented Stories

### Story 2.2: Assembly-Aware Routing ✅

**Implementation**: `src/agents/meta_controller/assembly_router.py` (~400 lines)

#### Key Features

- **Rule-based routing using assembly features**
  - Simple queries (AI < 3, CN > 5) → TRM
  - Medium complexity (AI < 7 OR decomp > 0.7) → HRM
  - High complexity (AI ≥ 7) → MCTS

- **8 routing heuristics with confidence scores**
  - Simple pattern detection
  - High decomposability routing
  - Complex search routing
  - Medium complexity routing
  - Very high complexity routing
  - Technical complexity routing
  - High concept count routing
  - Deep graph routing

- **Statistics tracking**
  - Route counts by agent
  - Routing rates
  - Average confidence scores

- **Explainability**
  - Detailed reasoning for every routing decision
  - Assembly feature visualization
  - Routing statistics reporting

#### Usage Example

```python
from src.agents.meta_controller import AssemblyRouter

# Initialize router
router = AssemblyRouter()

# Route a query
decision = router.route("Design a distributed microservices architecture...")

print(f"Selected agent: {decision.agent}")
print(f"Confidence: {decision.confidence:.2f}")
print(f"Reasoning: {decision.reasoning}")

# Get routing statistics
stats = router.get_statistics()
print(f"MCTS routing rate: {stats['mcts_rate']:.2%}")
```

### Story 2.3: Hybrid Meta-Controller ✅

**Implementation**: `src/agents/meta_controller/hybrid_controller.py` (~350 lines)

#### Key Features

- **Weighted ensemble**: 60% neural + 40% assembly (configurable)
- **Agreement/disagreement tracking**
- **Dynamic weight adjustment**
- **Detailed explanations with both components**
- **Fallback strategies** when components unavailable

#### Architecture

```
HybridMetaController
├── Neural Controller (60%)
│   ├── RNN/BERT-based predictions
│   └── Learned from execution traces
└── Assembly Router (40%)
    ├── Rule-based heuristics
    └── Interpretable decisions

Final Decision = weighted_ensemble(neural_pred, assembly_decision)
```

#### Usage Example

```python
from src.agents.meta_controller import (
    HybridMetaController,
    MetaControllerFeatures,
)

# Initialize hybrid controller
hybrid = HybridMetaController(
    neural_controller=my_neural_controller,
    neural_weight=0.6,
    assembly_weight=0.4,
)

# Set query context for assembly routing
hybrid.set_query_context("Design a microservices system...")

# Make prediction
features = MetaControllerFeatures(
    hrm_confidence=0.8,
    trm_confidence=0.6,
    mcts_value=0.7,
    consensus_score=0.75,
    last_agent='hrm',
    iteration=1,
    query_length=120,
    has_rag_context=False,
)

prediction = hybrid.predict(features)

print(f"Selected: {prediction.agent}")
print(f"Confidence: {prediction.confidence:.2f}")
print(f"\n{prediction.explanation}")

# Adjust weights dynamically
hybrid.adjust_weights(neural_weight=0.7, assembly_weight=0.3)
```

### Story 2.4: Generate Training Data with Assembly Features ✅

**Implementation**: `scripts/generate_meta_controller_training_data.py` (~700 lines)

#### Key Features

- **Synthetic data generation** with controlled complexity
- **Curriculum learning**: 10% simple, 30% medium, 60% complex
- **Assembly feature integration**: All 8 features per sample
- **Ground truth labeling** based on assembly heuristics
- **Quality validation**: No NaN values, reasonable distributions

#### Dataset Format

Each training sample includes:

```json
{
  "query": "Design a distributed system...",
  "features": {
    "hrm_confidence": 0.75,
    "trm_confidence": 0.65,
    "mcts_value": 0.80,
    "consensus_score": 0.70,
    "last_agent": "hrm",
    "iteration": 0,
    "query_length": 120,
    "has_rag_context": false
  },
  "assembly_features": {
    "assembly_index": 8.5,
    "copy_number": 3.2,
    "decomposability_score": 0.65,
    "graph_depth": 5,
    "constraint_count": 10,
    "concept_count": 15,
    "technical_complexity": 0.7,
    "normalized_assembly_index": 0.425
  },
  "ground_truth_agent": "mcts",
  "reasoning": "High complexity requires MCTS...",
  "complexity": "complex",
  "generated_at": "2025-11-23T21:50:47.559061"
}
```

#### Usage

```bash
# Generate 1200 samples with default curriculum
python scripts/generate_meta_controller_training_data.py \
    --num-samples 1200 \
    --output data/training_with_assembly.json

# Custom curriculum
python scripts/generate_meta_controller_training_data.py \
    --num-samples 5000 \
    --simple-ratio 0.15 \
    --medium-ratio 0.35 \
    --complex-ratio 0.50
```

#### Dataset Statistics

- **Total samples**: 1,200
- **Agent distribution**:
  - HRM: 369 (30.8%)
  - MCTS: 711 (59.2%)
  - TRM: 120 (10.0%)
- **Assembly index**: min=1.0, max=31.0, mean=14.4, std=7.8
- **Decomposability**: min=0.64, max=0.72, mean=0.65

### Story 2.5: Train Meta-Controllers with Assembly Features ✅

**Implementation**: `scripts/train_assembly_meta_controller.py` (~700 lines)

#### Key Features

- **Assembly-aware neural router**
  - Input: 18 features (10 base + 8 assembly)
  - Architecture: [18] → [128, 64, 32] → [3]
  - Confidence calibration head
  - Layer normalization and dropout

- **Training capabilities**
  - Early stopping with patience=10
  - Train/val/test split: 70/15/15
  - Batch size: 64
  - Learning rate: 0.001
  - Weight decay: 0.0001

- **Baseline comparison**
  - Train both baseline (10 features) and assembly (18 features)
  - Compare accuracy, calibration error
  - Report improvement metrics

- **Feature importance analysis**
  - Gradient-based attribution
  - Identifies top contributing features
  - Quantifies assembly feature impact

#### Usage

```bash
# Train assembly-augmented model
python scripts/train_assembly_meta_controller.py \
    --data-path data/training_with_assembly.json \
    --epochs 50 \
    --batch-size 64 \
    --output-dir models/meta_controller

# Compare with baseline
python scripts/train_assembly_meta_controller.py \
    --compare-baseline \
    --save-comparison results/comparison.json

# Quick training for testing
python scripts/train_assembly_meta_controller.py \
    --epochs 10 \
    --batch-size 32
```

#### Expected Results

Based on assembly feature integration, we expect:

- **Accuracy improvement**: +2-5% over baseline
- **Calibration improvement**: Better confidence-accuracy alignment
- **Top assembly features**:
  1. `assembly_index` (~15-20% importance)
  2. `decomposability_score` (~10-15%)
  3. `normalized_assembly_index` (~8-12%)
  4. `copy_number` (~6-10%)

## Configuration

Assembly meta-controller settings in `config/assembly_config.yaml`:

```yaml
assembly:
  meta_controller:
    training:
      simple_ratio: 0.10
      medium_ratio: 0.30
      complex_ratio: 0.60
      num_samples: 1200
      seed: 42

    router:
      input_dim: 18
      hidden_dims: [128, 64, 32]
      num_agents: 3
      dropout: 0.2
      learning_rate: 0.001
      batch_size: 64
      epochs: 50

    hybrid:
      neural_weight: 0.6
      assembly_weight: 0.4

  routing:
    simple_threshold: 3
    medium_threshold: 7
```

## Testing

### Unit Tests

```bash
# Test assembly router
pytest tests/agents/meta_controller/test_assembly_integration.py::TestAssemblyRouter -v

# Test hybrid controller
pytest tests/agents/meta_controller/test_assembly_integration.py::TestHybridMetaController -v

# Test data generation
pytest tests/scripts/test_meta_controller_data_generation.py -v
```

### Integration Tests

```python
from src.agents.meta_controller import HybridMetaController, AssemblyRouter

def test_end_to_end_routing():
    """Test complete routing pipeline."""
    # Initialize hybrid controller
    hybrid = HybridMetaController(
        neural_controller=None,  # Assembly-only for this test
    )

    # Test simple query
    hybrid.set_query_context("What is 2+2?")
    features = create_simple_features()
    prediction = hybrid.predict(features)

    assert prediction.agent == "trm"
    assert prediction.confidence > 0.7

    # Test complex query
    hybrid.set_query_context("Design a distributed microservices system...")
    features = create_complex_features()
    prediction = hybrid.predict(features)

    assert prediction.agent == "mcts"
    assert prediction.confidence > 0.7
```

## Integration Points

### With Existing Meta-Controllers

The hybrid controller integrates seamlessly with existing meta-controller infrastructure:

```python
from src.agents.meta_controller import (
    BERTMetaController,  # Existing neural controller
    HybridMetaController,  # New hybrid controller
)

# Wrap existing neural controller with assembly routing
bert_controller = BERTMetaController()

hybrid_controller = HybridMetaController(
    neural_controller=bert_controller,
    neural_weight=0.6,
    assembly_weight=0.4,
)

# Use like any other meta-controller
prediction = hybrid_controller.predict(features)
```

### With Multi-Agent Orchestrator

```python
from src.orchestrator import MultiAgentOrchestrator
from src.agents.meta_controller import HybridMetaController

orchestrator = MultiAgentOrchestrator(
    meta_controller=HybridMetaController(...),
    hrm_agent=hrm,
    trm_agent=trm,
    mcts_agent=mcts,
)

# Orchestrator automatically uses hybrid routing
result = orchestrator.process(query)
```

## Performance Considerations

### Routing Overhead

- **Assembly feature extraction**: ~5-10ms per query
- **Assembly routing decision**: <1ms
- **Neural prediction**: ~10-50ms (model-dependent)
- **Hybrid ensemble**: <1ms

**Total overhead**: ~15-60ms per routing decision

### Memory Usage

- **Assembly router**: ~1MB (lightweight rules)
- **Neural controller**: ~50-200MB (model-dependent)
- **Hybrid controller**: Combined + ~10MB for stats

### Caching

Assembly indices are cached (LRU cache, 10k entries):
- **Cache hit rate**: ~70-80% for repeated queries
- **Cached routing**: <1ms

## Next Steps

### Sprint 3: MCTS Enhancement

- Story 3.1: Assembly-enhanced UCB selection
- Story 3.2: Path assembly index tracking
- Story 3.3: Substructure library integration in MCTS

### Future Improvements

1. **Online learning**: Update routing rules based on real performance
2. **Adaptive weighting**: Dynamically adjust neural vs. assembly weights
3. **Multi-domain routing**: Domain-specific assembly thresholds
4. **Confidence calibration**: Improve confidence-accuracy alignment

## References

- **Assembly Theory Paper**: [Nature, 2023]
- **Meta-Controller Design**: `docs/ARCHITECTURE.md`
- **Assembly Foundation**: `docs/ASSEMBLY_THEORY.md`
- **Configuration**: `config/assembly_config.yaml`

## Authors

- Implementation: Sprint 2 Development Team
- Assembly Theory Integration: Research Team
- Testing & Validation: QA Team

## Changelog

- **2025-11-23**: Sprint 2 completion
  - Implemented assembly-aware routing (Story 2.2)
  - Implemented hybrid meta-controller (Story 2.3)
  - Generated training data with assembly features (Story 2.4)
  - Created training scripts for assembly meta-controllers (Story 2.5)
  - Updated configuration and documentation
