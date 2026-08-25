"""
Continual Learning System Demo

Demonstrates the complete production feedback loop for continuous improvement:
1. Logging production interactions with privacy preservation
2. Analyzing failure patterns
3. Selecting valuable examples for annotation
4. Incremental model retraining
5. A/B testing new models

Usage:
    python training/examples/continual_learning_demo.py
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import time
import uuid

import numpy as np
import yaml

from training.continual_learning import (
    ABTestFramework,
    ActiveLearningSelector,
    DriftDetector,
    FailurePatternAnalyzer,
    IncrementalRetrainingPipeline,
    ProductionInteraction,
    ProductionInteractionLogger,
)


def load_config():
    """Load configuration."""
    config_path = Path("training/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["continual_learning"]


def generate_sample_interactions(num_samples: int = 100) -> list[ProductionInteraction]:
    """
    Generate synthetic production interactions for demo.

    Args:
        num_samples: Number of interactions to generate

    Returns:
        List of production interactions
    """
    print(f"\nGenerating {num_samples} sample interactions...")

    agents = ["HRM", "TRM", "MCTS"]
    queries = [
        "What is Monte Carlo Tree Search?",
        "How does hierarchical reasoning work?",
        "Explain task refinement in AI systems",
        "How to implement multi-agent systems?",
        "What are the benefits of MCTS for planning?",
        "Compare different search algorithms",
        "How to optimize agent coordination?",
        "Explain the UCB formula in MCTS",
        "What is the difference between DFS and MCTS?",
        "How to handle uncertainty in planning?",
    ]

    interactions = []

    for i in range(num_samples):
        # Randomly select parameters
        agent = np.random.choice(agents)
        query = np.random.choice(queries)
        confidence = np.random.beta(8, 2)  # Biased toward high confidence

        # Generate latency (some slow outliers)
        latency = np.random.gamma(2, 500) + 500  # 500-5000ms range

        # Generate feedback (mostly positive, some failures)
        if np.random.random() < 0.15:  # 15% failures
            feedback_score = np.random.uniform(1.0, 2.5)
            thumbs = "down"
            hallucination = np.random.random() < 0.3
            retrieval_failed = np.random.random() < 0.4
        else:  # 85% success
            feedback_score = np.random.uniform(3.5, 5.0)
            thumbs = "up"
            hallucination = False
            retrieval_failed = False

        interaction = ProductionInteraction(
            interaction_id=str(uuid.uuid4()),
            timestamp=time.time() - (num_samples - i) * 60,  # Spread over time
            session_id=f"session_{i % 20}",  # 20 different sessions
            user_query=query,
            agent_selected=agent,
            agent_confidence=confidence,
            response=f"Detailed response about {query.lower()}...",
            user_feedback_score=feedback_score,
            thumbs_up_down=thumbs,
            latency_ms=latency,
            tokens_used=int(np.random.uniform(50, 500)),
            cost=np.random.uniform(0.001, 0.01),
            hallucination_detected=hallucination,
            retrieval_failed=retrieval_failed,
            retrieval_quality=np.random.beta(8, 2),
        )

        interactions.append(interaction)

    print(f"Generated {len(interactions)} interactions")
    return interactions


async def demo_production_logging(config, interactions):
    """
    Demonstrate production interaction logging.

    Args:
        config: Configuration dict
        interactions: List of interactions to log
    """
    print("\n" + "=" * 80)
    print("1. PRODUCTION INTERACTION LOGGING")
    print("=" * 80)

    # Initialize logger
    logger_config = config["logging"]
    logger = ProductionInteractionLogger(logger_config)

    print(f"\nLogging {len(interactions)} interactions...")

    # Log all interactions
    for interaction in interactions:
        await logger.log_interaction(interaction)

    # Flush remaining buffer
    await logger._flush_buffer()

    # Show statistics
    stats = logger.get_statistics()
    print("\nLogging Statistics:")
    print(f"  Total logged: {stats['total_logged']}")
    print(f"  Total in DB: {stats.get('total_in_db', 0)}")
    print(f"  Avg feedback score: {stats.get('avg_feedback_score', 0):.2f}/5.0")
    print(f"  Avg latency: {stats.get('avg_latency_ms', 0):.0f}ms")
    print(f"  Validation failures: {stats['validation_failures']}")
    print(f"  PII sanitized: {stats['pii_sanitized']}")

    # Demo querying
    print("\nQuerying low-rated interactions (< 3 stars):")
    low_rated = logger.query_interactions(max_feedback_score=3.0, limit=5)
    print(f"  Found {len(low_rated)} low-rated interactions")

    return logger


def demo_failure_analysis(config, logger):
    """
    Demonstrate failure pattern analysis.

    Args:
        config: Configuration dict
        logger: Production interaction logger
    """
    print("\n" + "=" * 80)
    print("2. FAILURE PATTERN ANALYSIS")
    print("=" * 80)

    # Get all interactions from logger
    interactions = logger.query_interactions(limit=1000)

    # Initialize analyzer
    analyzer_config = config.get("failure_analysis", {})
    analyzer = FailurePatternAnalyzer(analyzer_config)

    print(f"\nAnalyzing {len(interactions)} interactions for failure patterns...")

    # Analyze failures
    patterns = analyzer.analyze_failures(interactions)

    print(f"\nIdentified {len(patterns)} failure patterns:")
    for i, pattern in enumerate(patterns, 1):
        print(f"\n  Pattern {i}: {pattern.pattern_type.upper()}")
        print(f"    Frequency: {pattern.frequency} occurrences")
        print(f"    Severity: {pattern.severity:.2f}/1.0")
        print(f"    Description: {pattern.description}")
        print(f"    Suggested fix: {pattern.suggested_fix}")

    # Show summary
    summary = analyzer.get_summary()
    print("\nPattern Summary:")
    print(f"  Total patterns: {summary['total_patterns']}")
    print(f"  Pattern types: {summary.get('pattern_types', {})}")
    print(f"  Total failures: {summary.get('total_failures_analyzed', 0)}")
    print(f"  Avg severity: {summary.get('avg_severity', 0):.2f}")

    return analyzer


def demo_active_learning(config, logger):
    """
    Demonstrate active learning sample selection.

    Args:
        config: Configuration dict
        logger: Production interaction logger
    """
    print("\n" + "=" * 80)
    print("3. ACTIVE LEARNING SAMPLE SELECTION")
    print("=" * 80)

    # Get interactions
    interactions = logger.query_interactions(limit=1000)

    # Initialize selector
    selector_config = config.get("active_learning", {})
    selector = ActiveLearningSelector(selector_config)

    budget = selector_config.get("budget_per_cycle", 10)
    strategy = selector_config.get("selection_strategy", "hybrid")

    print(f"\nSelecting {budget} samples using '{strategy}' strategy...")

    # Select candidates
    candidates = selector.select_for_annotation(interactions, budget=budget)

    print(f"\nSelected {len(candidates)} candidates for annotation:")
    for i, candidate in enumerate(candidates[:5], 1):  # Show first 5
        print(f"\n  Candidate {i}:")
        print(f"    ID: {candidate.interaction_id}")
        print(f"    Priority: {candidate.priority_score:.3f}")
        print(f"    Reason: {candidate.selection_reason}")

    if len(candidates) > 5:
        print(f"\n  ... and {len(candidates) - 5} more candidates")

    # Show selection breakdown
    selection_reasons = {}
    for c in candidates:
        selection_reasons[c.selection_reason] = selection_reasons.get(c.selection_reason, 0) + 1

    print("\nSelection breakdown:")
    for reason, count in selection_reasons.items():
        print(f"  {reason}: {count} samples")

    return candidates


async def demo_incremental_retraining(config, logger):
    """
    Demonstrate incremental model retraining.

    Args:
        config: Configuration dict
        logger: Production interaction logger
    """
    print("\n" + "=" * 80)
    print("4. INCREMENTAL MODEL RETRAINING")
    print("=" * 80)

    # Get high-quality interactions (with feedback)
    interactions = logger.query_interactions(limit=1000)

    # Initialize pipeline
    retraining_config = config.get("retraining", {})
    pipeline = IncrementalRetrainingPipeline(retraining_config)

    print(f"\nPreparing to retrain with {len(interactions)} interactions...")
    print(f"  Schedule: {pipeline.schedule}")
    print(f"  Min samples required: {pipeline.min_new_samples}")

    # Check if should retrain
    last_retrain = datetime.now() - timedelta(days=8)  # 8 days ago
    should_retrain = pipeline.should_retrain(last_retrain, len(interactions))

    print(f"\nShould retrain? {should_retrain}")
    print(f"  Last retrain: {last_retrain.strftime('%Y-%m-%d %H:%M')}")
    print(f"  New samples: {len(interactions)}")

    if should_retrain:
        print("\nStarting retraining pipeline...")
        result = await pipeline.retrain(interactions)

        print(f"\nRetraining result: {result['status']}")
        if result["status"] == "completed":
            print(f"  Timestamp: {result['timestamp']}")
            print(f"  Samples used: {result['num_samples']}")
            print("\n  Pipeline steps:")
            for step in result["steps"]:
                print(f"    {step['step']}: {step['status']}")
                if "metrics" in step:
                    print(f"      Metrics: {step['metrics']}")
    else:
        print("\nRetraining conditions not met (skipping)")

    return pipeline


def demo_drift_detection(config, interactions):
    """
    Demonstrate data drift detection.

    Args:
        config: Configuration dict
        interactions: List of interactions
    """
    print("\n" + "=" * 80)
    print("5. DATA DRIFT DETECTION")
    print("=" * 80)

    # Initialize detector
    drift_config = config.get("drift_detection", {})
    detector = DriftDetector(drift_config)

    print(f"\nDetection method: {detector.detection_method}")
    print(f"Window size: {detector.window_size}")
    print(f"Threshold: {detector.threshold}")

    # Create feature vectors from interactions
    # (In practice, these would be embeddings or extracted features)
    print("\nExtracting features from interactions...")

    reference_features = []
    current_features = []

    for i, interaction in enumerate(interactions):
        # Simple features: confidence, latency (normalized), feedback
        features = [
            interaction.agent_confidence,
            min(interaction.latency_ms / 10000.0, 1.0),  # Normalize
            (interaction.user_feedback_score or 3.0) / 5.0,  # Normalize
        ]

        if i < len(interactions) // 2:
            reference_features.append(features)
        else:
            current_features.append(features)

    # Set reference distribution
    reference = np.array(reference_features)
    detector.set_reference_distribution(reference)

    print(f"Set reference distribution: {len(reference_features)} samples")

    # Add current samples and check for drift
    print(f"\nChecking {len(current_features)} current samples for drift...")

    drift_detected = False
    for features in current_features:
        report = detector.add_sample(np.array(features))
        if report:
            drift_detected = True

    if drift_detected:
        summary = detector.get_drift_summary()
        print("\nDrift detected!")
        print(f"  Total drifts: {summary['total_drifts']}")
        print(f"  Avg severity: {summary.get('avg_severity', 0):.3f}")

        if "recent_drifts" in summary:
            print("\n  Recent drifts:")
            for drift in summary["recent_drifts"]:
                print(f"    Type: {drift['type']}, Severity: {drift['severity']:.3f}")
    else:
        print("\nNo significant drift detected")

    return detector


def demo_ab_testing(config):
    """
    Demonstrate A/B testing framework.

    Args:
        config: Configuration dict
    """
    print("\n" + "=" * 80)
    print("6. A/B TESTING FRAMEWORK")
    print("=" * 80)

    # Initialize framework
    ab_config = config.get("ab_testing", {})
    framework = ABTestFramework(ab_config)

    print(f"\nTraffic split: {framework.traffic_split * 100:.0f}% to treatment")
    print(f"Min samples: {framework.min_samples}")
    print(f"Confidence level: {framework.confidence_level}")

    # Create test
    def simple_metric(input_data, output):
        """Simple success metric."""
        return np.random.random()  # Dummy metric

    test_id = framework.create_test(
        test_name="model_v2_vs_v1",
        model_a="production_v1",
        model_b="retrained_v2",
        metric_fn=simple_metric,
    )

    print(f"\nCreated A/B test: {test_id}")

    # Simulate traffic
    print("\nSimulating traffic...")
    num_requests = 1200

    for i in range(num_requests):
        request_id = f"req_{i}"
        group = framework.assign_group(test_id, request_id)

        # Simulate metric (B slightly better)
        if group == "A":
            metric = np.random.normal(0.70, 0.1)
        else:
            metric = np.random.normal(0.75, 0.1)  # 5% better

        framework.record_result(test_id, group, {"req": i}, "output", metric)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{num_requests} requests processed...")

    # Get status
    status = framework.get_test_status(test_id)
    print("\nTest Status:")
    print(f"  Status: {status['status']}")
    print(f"  Control samples: {status['samples_control']}")
    print(f"  Treatment samples: {status['samples_treatment']}")

    if "result" in status:
        result = status["result"]
        print("\nTest Result:")
        print(f"  Control mean: {result['mean_control']:.4f}")
        print(f"  Treatment mean: {result['mean_treatment']:.4f}")
        print(f"  Improvement: {result['improvement'] * 100:.2f}%")
        print(f"  P-value: {result['p_value']:.4f}")
        print(f"  Significant: {result['is_significant']}")
        print(f"  Recommendation: {result['recommendation']}")

    # End test
    final_result = framework.end_test(test_id)
    print(f"\nTest completed: {final_result['recommendation']}")

    return framework


async def main():
    """Run complete continual learning demo."""
    print("=" * 80)
    print("CONTINUAL LEARNING SYSTEM DEMONSTRATION")
    print("Production Feedback Loop for Continuous Improvement")
    print("=" * 80)

    # Load configuration
    print("\nLoading configuration...")
    config = load_config()

    # Generate sample data
    interactions = generate_sample_interactions(num_samples=200)

    # 1. Production logging
    logger = await demo_production_logging(config, interactions)

    # 2. Failure analysis
    analyzer = demo_failure_analysis(config, logger)

    # 3. Active learning
    candidates = demo_active_learning(config, logger)

    # 4. Incremental retraining
    await demo_incremental_retraining(config, logger)

    # 5. Drift detection
    demo_drift_detection(config, interactions)

    # 6. A/B testing
    demo_ab_testing(config)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nContinual Learning System Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("  1. Production interaction logging with privacy preservation")
    print("  2. Systematic failure pattern identification")
    print("  3. Active learning sample selection")
    print("  4. Incremental model retraining pipeline")
    print("  5. Data drift detection")
    print("  6. A/B testing for model deployment")

    stats = logger.get_statistics()
    print("\nProduction Metrics:")
    print(f"  Interactions logged: {stats['total_logged']}")
    print(f"  Avg satisfaction: {stats.get('avg_feedback_score', 0):.2f}/5.0")
    print(f"  Avg latency: {stats.get('avg_latency_ms', 0):.0f}ms")
    print(f"  Failure patterns: {len(analyzer.identified_patterns)}")
    print(f"  Samples for annotation: {len(candidates)}")

    print("\nNext Steps:")
    print("  1. Annotate selected samples")
    print("  2. Retrain models with new data")
    print("  3. A/B test improved models")
    print("  4. Deploy winners to production")
    print("  5. Monitor and iterate")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
