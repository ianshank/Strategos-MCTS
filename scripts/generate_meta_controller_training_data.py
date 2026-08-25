#!/usr/bin/env python3
"""
Generate Meta-Controller Training Data with Assembly Features (Story 2.4).

This script generates synthetic training data for meta-controllers that includes
both traditional features and assembly theory features for improved routing decisions.

Usage:
    # Generate 1000 samples with default curriculum
    python scripts/generate_meta_controller_training_data.py --num-samples 1000

    # Generate with specific output location
    python scripts/generate_meta_controller_training_data.py \\
        --num-samples 5000 \\
        --output data/meta_controller_training.json

    # Adjust curriculum distribution
    python scripts/generate_meta_controller_training_data.py \\
        --num-samples 2000 \\
        --simple-ratio 0.15 \\
        --medium-ratio 0.35 \\
        --complex-ratio 0.50

Data Format:
    Each training sample includes:
    - query: Natural language query text
    - features: Meta-controller features (HRM/TRM/MCTS confidences, etc.)
    - assembly_features: 8 assembly theory features
    - ground_truth_agent: Correct agent selection (hrm, trm, or mcts)
    - reasoning: Explanation of routing decision
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.framework.assembly import (  # noqa: E402
    AssemblyFeatureExtractor,
    AssemblyFeatures,
)

logger = logging.getLogger(__name__)


# Query templates by complexity level
QUERY_TEMPLATES = {
    "simple": [
        "What is {concept}?",
        "How do I {action}?",
        "Explain {term} briefly",
        "Show me an example of {feature}",
        "What does {function} do?",
        "Define {term}",
        "How to {simple_task}?",
        "What is the purpose of {component}?",
        "Explain {concept} in simple terms",
        "Give me a quick overview of {topic}",
    ],
    "medium": [
        "Implement a {algorithm} in Python with {constraint}",
        "Design a {system_type} system for {use_case}",
        "How do I optimize {component} for {goal}?",
        "Compare {method_a} with {method_b} for {problem}",
        "What are the best practices for {task} in {domain}?",
        "Explain how {system} handles {challenge}",
        "Build a {component} that supports {features}",
        "How do you integrate {tech_a} with {tech_b}?",
        "What are the tradeoffs between {approach_a} and {approach_b}?",
        "Debug this code that {problem_description}",
    ],
    "complex": [
        "Design a distributed {system} with {feature_1}, {feature_2}, and {feature_3} "
        "that handles {challenge_1} and {challenge_2} while maintaining {constraint}",
        "Implement a multi-agent {algorithm} system that coordinates {num_agents} agents "
        "with hierarchical task decomposition, consensus mechanisms, and fault tolerance",
        "Build a scalable microservices architecture for {domain} with API gateway, "
        "service mesh, event-driven communication, distributed caching, monitoring, "
        "and observability across multiple availability zones",
        "Create a production-ready {system} that integrates {tech_1}, {tech_2}, and {tech_3} "
        "with comprehensive error handling, retry logic, circuit breakers, and graceful degradation",
        "Design an end-to-end machine learning pipeline for {use_case} including data ingestion, "
        "preprocessing, feature engineering, model training, hyperparameter tuning, deployment, "
        "monitoring, and automated retraining with A/B testing capabilities",
        "Implement a real-time {system} with {constraint_1}, {constraint_2}, and {constraint_3} "
        "that processes {data_type} data at {scale} while ensuring {quality_attribute_1}, "
        "{quality_attribute_2}, and {quality_attribute_3}",
    ],
}

# Vocabularies for template filling
VOCABULARIES = {
    "concept": ["MCTS", "UCB1", "PUCT", "LangGraph", "state machine", "tree search"],
    "action": ["install dependencies", "run tests", "deploy the app", "configure the system"],
    "term": ["exploration-exploitation", "backpropagation", "policy network", "value function"],
    "feature": ["async/await", "recursion", "list comprehension", "decorator"],
    "function": ["map", "filter", "reduce", "zip"],
    "simple_task": ["read a file", "parse JSON", "make HTTP request", "handle errors"],
    "component": ["router", "controller", "middleware", "service", "adapter"],
    "topic": ["neural networks", "distributed systems", "REST APIs", "graph algorithms"],
    "algorithm": ["binary search", "merge sort", "depth-first search", "A* pathfinding"],
    "constraint": ["O(n log n) time", "constant space", "thread-safety", "memory efficiency"],
    "system_type": ["real-time", "distributed", "event-driven", "microservices"],
    "use_case": ["e-commerce", "streaming analytics", "IoT data processing", "recommendation engine"],
    "goal": ["performance", "scalability", "reliability", "maintainability"],
    "method_a": ["MCTS", "greedy search", "beam search", "A*"],
    "method_b": ["minimax", "alpha-beta pruning", "Monte Carlo", "heuristic search"],
    "problem": ["pathfinding", "optimization", "scheduling", "resource allocation"],
    "task": ["API design", "database modeling", "error handling", "testing"],
    "domain": ["web development", "data engineering", "machine learning", "cloud architecture"],
    "system": ["load balancer", "message queue", "cache", "database"],
    "challenge": ["high latency", "network partitions", "data inconsistency", "race conditions"],
    "tech_a": ["Redis", "Kafka", "PostgreSQL", "Elasticsearch"],
    "tech_b": ["RabbitMQ", "MongoDB", "Cassandra", "S3"],
    "approach_a": ["synchronous", "SQL", "monolithic", "REST"],
    "approach_b": ["asynchronous", "NoSQL", "microservices", "GraphQL"],
    "problem_description": ["throws KeyError", "has memory leak", "fails under load", "produces incorrect results"],
    "feature_1": ["auto-scaling", "load balancing", "data replication"],
    "feature_2": ["health checks", "circuit breakers", "rate limiting"],
    "feature_3": ["monitoring", "logging", "distributed tracing"],
    "challenge_1": ["network failures", "data corruption", "peak traffic"],
    "challenge_2": ["partial outages", "cascading failures", "data races"],
    "num_agents": ["5", "10", "20"],
    "tech_1": ["Kubernetes", "Docker", "Terraform"],
    "tech_2": ["Prometheus", "Grafana", "ELK stack"],
    "tech_3": ["Istio", "Envoy", "Linkerd"],
    "constraint_1": ["sub-100ms latency", "99.99% availability", "ACID transactions"],
    "constraint_2": ["horizontal scalability", "fault tolerance", "eventual consistency"],
    "constraint_3": ["cost optimization", "security compliance", "multi-region deployment"],
    "data_type": ["streaming", "time-series", "graph", "geospatial"],
    "scale": ["1M requests/second", "10TB/day", "1B events/hour"],
    "quality_attribute_1": ["consistency", "availability", "partition tolerance"],
    "quality_attribute_2": ["security", "performance", "reliability"],
    "quality_attribute_3": ["observability", "maintainability", "cost-efficiency"],
}


def fill_template(template: str) -> str:
    """Fill template with random vocabulary."""
    filled = template

    # Extract placeholders
    import re

    placeholders = re.findall(r"\{(\w+)\}", template)

    for placeholder in placeholders:
        if placeholder in VOCABULARIES:
            replacement = random.choice(VOCABULARIES[placeholder])
            filled = filled.replace(f"{{{placeholder}}}", replacement, 1)

    return filled


def generate_query(complexity: str) -> str:
    """
    Generate a query of specified complexity.

    Args:
        complexity: One of 'simple', 'medium', 'complex'

    Returns:
        Generated query string
    """
    template = random.choice(QUERY_TEMPLATES[complexity])
    return fill_template(template)


def select_ground_truth_agent(
    assembly_features: AssemblyFeatures,
    complexity: str,
) -> tuple[str, str]:
    """
    Select ground truth agent based on assembly features and complexity.

    Uses routing heuristics:
    - Simple (AI < 3, CN > 5): TRM
    - Medium (AI < 7 OR decomp > 0.7): HRM
    - Complex (AI >= 7): MCTS

    Args:
        assembly_features: Extracted assembly features
        complexity: Query complexity level

    Returns:
        Tuple of (agent_name, reasoning)
    """
    ai = assembly_features.assembly_index
    cn = assembly_features.copy_number
    decomp = assembly_features.decomposability_score

    # Simple query routing
    if complexity == "simple" or (ai < 3 and cn > 5):
        return "trm", (
            f"Simple query (AI={ai:.1f}, CN={cn:.1f}) with high pattern reuse "
            "routes to TRM for fast, iterative refinement"
        )

    # High decomposability → HRM
    if decomp > 0.7:
        return "hrm", (
            f"High decomposability (score={decomp:.2f}) indicates query can be "
            "broken into subtasks, routing to HRM for hierarchical decomposition"
        )

    # Medium complexity
    if complexity == "medium" or ai < 7:
        return "hrm", (
            f"Medium complexity (AI={ai:.1f}) routes to HRM for " "structured hierarchical reasoning and task breakdown"
        )

    # Complex query → MCTS
    return "mcts", (
        f"High complexity (AI={ai:.1f}) with low decomposability (score={decomp:.2f}) "
        "requires MCTS for exploratory tree search and optimization"
    )


def generate_meta_controller_features(
    assembly_features: AssemblyFeatures,
    ground_truth_agent: str,
    iteration: int = 0,
) -> dict[str, Any]:
    """
    Generate realistic meta-controller features.

    Simulates agent confidences based on assembly features and ground truth.

    Args:
        assembly_features: Assembly features
        ground_truth_agent: The correct agent for this query
        iteration: Current iteration number

    Returns:
        Dictionary of meta-controller features
    """
    # Base confidences with some noise
    ai = assembly_features.assembly_index
    decomp = assembly_features.decomposability_score

    # TRM confidence: high for simple, low assembly index
    trm_base = max(0.1, 1.0 - ai / 10.0)
    trm_confidence = min(0.95, max(0.05, trm_base + np.random.normal(0, 0.1)))

    # HRM confidence: high for medium complexity and high decomposability
    hrm_base = decomp * (1.0 - abs(ai - 5) / 10.0)
    hrm_confidence = min(0.95, max(0.05, hrm_base + np.random.normal(0, 0.1)))

    # MCTS confidence: high for complex queries
    mcts_base = min(1.0, ai / 10.0)
    mcts_confidence = min(0.95, max(0.05, mcts_base + np.random.normal(0, 0.1)))

    # Boost ground truth agent confidence
    if ground_truth_agent == "trm":
        trm_confidence = min(0.95, trm_confidence + 0.2)
    elif ground_truth_agent == "hrm":
        hrm_confidence = min(0.95, hrm_confidence + 0.2)
    else:  # mcts
        mcts_confidence = min(0.95, mcts_confidence + 0.2)

    # MCTS value estimate (0-1)
    mcts_value = mcts_confidence * np.random.uniform(0.7, 1.0)

    # Consensus score: higher when confidences align
    confidences = [trm_confidence, hrm_confidence, mcts_confidence]
    consensus_score = 1.0 - np.std(confidences)

    return {
        "hrm_confidence": float(hrm_confidence),
        "trm_confidence": float(trm_confidence),
        "mcts_value": float(mcts_value),
        "consensus_score": float(consensus_score),
        "last_agent": random.choice(["hrm", "trm", "mcts", "none"]),
        "iteration": iteration,
        "query_length": assembly_features.concept_count * 10,  # Approximate
        "has_rag_context": random.random() < 0.3,  # 30% have RAG context
    }


def generate_training_sample(
    complexity: str,
    feature_extractor: AssemblyFeatureExtractor,
) -> dict[str, Any]:
    """
    Generate a single training sample.

    Args:
        complexity: Query complexity level (simple, medium, complex)
        feature_extractor: Assembly feature extractor

    Returns:
        Training sample dictionary
    """
    # Generate query
    query = generate_query(complexity)

    # Extract assembly features
    assembly_features = feature_extractor.extract(query)

    # Select ground truth agent
    ground_truth_agent, reasoning = select_ground_truth_agent(assembly_features, complexity)

    # Generate meta-controller features
    mc_features = generate_meta_controller_features(assembly_features, ground_truth_agent)

    return {
        "query": query,
        "features": mc_features,
        "assembly_features": {
            "assembly_index": float(assembly_features.assembly_index),
            "copy_number": float(assembly_features.copy_number),
            "decomposability_score": float(assembly_features.decomposability_score),
            "graph_depth": int(assembly_features.graph_depth),
            "constraint_count": int(assembly_features.constraint_count),
            "concept_count": int(assembly_features.concept_count),
            "technical_complexity": float(assembly_features.technical_complexity),
            "normalized_assembly_index": float(assembly_features.normalized_assembly_index),
        },
        "ground_truth_agent": ground_truth_agent,
        "reasoning": reasoning,
        "complexity": complexity,
        "generated_at": datetime.utcnow().isoformat(),
    }


def validate_dataset(samples: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """
    Validate generated dataset.

    Checks:
    - No NaN values in features
    - Reasonable distributions
    - All required fields present

    Args:
        samples: List of training samples

    Returns:
        Tuple of (is_valid, list of errors)
    """
    errors = []

    if len(samples) == 0:
        errors.append("Dataset is empty")
        return False, errors

    # Check each sample
    for i, sample in enumerate(samples):
        # Required fields
        required_fields = ["query", "features", "assembly_features", "ground_truth_agent"]
        for field in required_fields:
            if field not in sample:
                errors.append(f"Sample {i}: Missing required field '{field}'")

        # Check for NaN in assembly features
        if "assembly_features" in sample:
            for key, value in sample["assembly_features"].items():
                if isinstance(value, float) and np.isnan(value):
                    errors.append(f"Sample {i}: NaN value in assembly_features.{key}")

        # Check for NaN in meta-controller features
        if "features" in sample:
            for key, value in sample["features"].items():
                if isinstance(value, float) and np.isnan(value):
                    errors.append(f"Sample {i}: NaN value in features.{key}")

    # Distribution checks
    if not errors:
        assembly_indices = [s["assembly_features"]["assembly_index"] for s in samples]
        agents = [s["ground_truth_agent"] for s in samples]

        # Assembly index distribution
        ai_min, ai_max = min(assembly_indices), max(assembly_indices)
        ai_mean = np.mean(assembly_indices)

        logger.info(f"Assembly Index: min={ai_min:.1f}, max={ai_max:.1f}, mean={ai_mean:.1f}")

        if ai_min == ai_max:
            errors.append("Assembly index has no variance (all same value)")

        # Agent distribution
        from collections import Counter

        agent_counts = Counter(agents)

        logger.info(f"Agent distribution: {dict(agent_counts)}")

        if len(agent_counts) < 3:
            errors.append("Not all three agents represented in dataset")

    return len(errors) == 0, errors


def generate_dataset(
    num_samples: int = 1000,
    simple_ratio: float = 0.10,
    medium_ratio: float = 0.30,
    complex_ratio: float = 0.60,
    output_file: str = "data/training_with_assembly.json",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate complete training dataset.

    Args:
        num_samples: Total number of samples to generate
        simple_ratio: Proportion of simple queries (AI 1-3)
        medium_ratio: Proportion of medium queries (AI 4-6)
        complex_ratio: Proportion of complex queries (AI 7-10)
        output_file: Output file path
        seed: Random seed

    Returns:
        List of training samples
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Validate ratios
    total_ratio = simple_ratio + medium_ratio + complex_ratio
    if abs(total_ratio - 1.0) > 0.01:
        logger.warning(f"Ratios sum to {total_ratio}, normalizing to 1.0")
        simple_ratio /= total_ratio
        medium_ratio /= total_ratio
        complex_ratio /= total_ratio

    # Calculate sample counts
    num_simple = int(num_samples * simple_ratio)
    num_medium = int(num_samples * medium_ratio)
    num_complex = num_samples - num_simple - num_medium  # Remainder

    logger.info("=" * 70)
    logger.info(f"Generating {num_samples} training samples")
    logger.info("=" * 70)
    logger.info(f"  Simple (AI 1-3):   {num_simple:5d} ({simple_ratio:.0%})")
    logger.info(f"  Medium (AI 4-6):   {num_medium:5d} ({medium_ratio:.0%})")
    logger.info(f"  Complex (AI 7-10): {num_complex:5d} ({complex_ratio:.0%})")
    logger.info("=" * 70)

    # Initialize feature extractor
    feature_extractor = AssemblyFeatureExtractor()

    # Generate samples by complexity
    samples = []

    complexities = ["simple"] * num_simple + ["medium"] * num_medium + ["complex"] * num_complex
    random.shuffle(complexities)

    for i, complexity in enumerate(complexities):
        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{num_samples} samples...")

        sample = generate_training_sample(complexity, feature_extractor)
        samples.append(sample)

    logger.info(f"Generated {len(samples)} samples")

    # Validate dataset
    logger.info("Validating dataset...")
    is_valid, errors = validate_dataset(samples)

    if not is_valid:
        logger.error("Dataset validation failed:")
        for error in errors[:10]:  # Show first 10 errors
            logger.error(f"  - {error}")
        if len(errors) > 10:
            logger.error(f"  ... and {len(errors) - 10} more errors")
        raise ValueError("Dataset validation failed")

    logger.info("✓ Dataset validation passed")

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    logger.info(f"✓ Saved {len(samples)} samples to {output_path}")

    # Print statistics
    print_statistics(samples)

    return samples


def print_statistics(samples: list[dict[str, Any]]) -> None:
    """Print dataset statistics."""
    from collections import Counter

    # Agent distribution
    agents = [s["ground_truth_agent"] for s in samples]
    agent_counts = Counter(agents)

    # Assembly index stats
    assembly_indices = [s["assembly_features"]["assembly_index"] for s in samples]
    ai_stats = {
        "min": min(assembly_indices),
        "max": max(assembly_indices),
        "mean": np.mean(assembly_indices),
        "std": np.std(assembly_indices),
    }

    # Decomposability stats
    decomp_scores = [s["assembly_features"]["decomposability_score"] for s in samples]
    decomp_stats = {
        "min": min(decomp_scores),
        "max": max(decomp_scores),
        "mean": np.mean(decomp_scores),
    }

    # Complexity distribution
    complexities = [s["complexity"] for s in samples]
    complexity_counts = Counter(complexities)

    logger.info("\n" + "=" * 70)
    logger.info("Dataset Statistics")
    logger.info("=" * 70)
    logger.info(f"Total samples: {len(samples)}")
    logger.info("")
    logger.info("Agent Distribution:")
    for agent, count in sorted(agent_counts.items()):
        logger.info(f"  {agent.upper():5s}: {count:5d} ({count/len(samples):.1%})")
    logger.info("")
    logger.info("Assembly Index:")
    logger.info(f"  Min:  {ai_stats['min']:.2f}")
    logger.info(f"  Max:  {ai_stats['max']:.2f}")
    logger.info(f"  Mean: {ai_stats['mean']:.2f}")
    logger.info(f"  Std:  {ai_stats['std']:.2f}")
    logger.info("")
    logger.info("Decomposability Score:")
    logger.info(f"  Min:  {decomp_stats['min']:.3f}")
    logger.info(f"  Max:  {decomp_stats['max']:.3f}")
    logger.info(f"  Mean: {decomp_stats['mean']:.3f}")
    logger.info("")
    logger.info("Complexity Distribution:")
    for complexity, count in sorted(complexity_counts.items()):
        logger.info(f"  {complexity.capitalize():8s}: {count:5d} ({count/len(samples):.1%})")
    logger.info("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate meta-controller training data with assembly features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to generate (default: 1000)",
    )
    parser.add_argument(
        "--simple-ratio",
        type=float,
        default=0.10,
        help="Proportion of simple queries (default: 0.10)",
    )
    parser.add_argument(
        "--medium-ratio",
        type=float,
        default=0.30,
        help="Proportion of medium queries (default: 0.30)",
    )
    parser.add_argument(
        "--complex-ratio",
        type=float,
        default=0.60,
        help="Proportion of complex queries (default: 0.60)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training_with_assembly.json",
        help="Output file path (default: data/training_with_assembly.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Generate dataset
    try:
        start_time = datetime.utcnow()

        samples = generate_dataset(
            num_samples=args.num_samples,
            simple_ratio=args.simple_ratio,
            medium_ratio=args.medium_ratio,
            complex_ratio=args.complex_ratio,
            output_file=args.output,
            seed=args.seed,
        )

        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"\n✓ Generation completed in {duration:.1f} seconds")
        logger.info(f"  Rate: {len(samples) / duration:.1f} samples/second")
        logger.info(f"  Output: {args.output}")

    except KeyboardInterrupt:
        logger.info("\n⚠ Generation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n✗ Generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
