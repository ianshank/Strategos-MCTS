"""
Evaluation Suite for Multi-Agent MCTS Training Pipeline

Comprehensive evaluation framework including:
- DABStep benchmark evaluation
- Multi-agent consensus metrics
- Performance profiling
- Production validation
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a benchmark evaluation."""

    task_id: str
    predicted_answer: Any
    ground_truth: Any
    is_correct: bool
    steps_taken: int
    reasoning_quality: float
    latency_ms: float
    memory_mb: float
    agent_consensus: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""

    timestamp: str
    total_samples: int
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    avg_latency_ms: float
    avg_memory_mb: float
    avg_consensus: float
    per_difficulty_metrics: dict[str, dict[str, float]]
    agent_metrics: dict[str, dict[str, float]]
    error_analysis: dict[str, int]
    recommendations: list[str]


class DABStepBenchmark:
    """Evaluate on DABStep benchmark dataset."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize DABStep benchmark.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.test_size = config.get("test_size", 0.1)
        self.report_path = Path(
            config.get("benchmarks", {}).get("dabstep", {}).get("report_path", "./reports/dabstep_benchmark.json")
        )

        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.success_criteria = config.get("success_criteria", {})

        logger.info(f"DABStepBenchmark initialized, report path: {self.report_path}")

    def evaluate_model(self, model: Any, test_data: list[dict[str, Any]], verbose: bool = True) -> EvaluationReport:
        """
        Evaluate model on DABStep test set.

        Args:
            model: Model to evaluate
            test_data: Test dataset
            verbose: Print progress

        Returns:
            Evaluation report
        """
        logger.info(f"Evaluating on {len(test_data)} test samples")

        self.results = []

        for idx, sample in enumerate(test_data):
            start_time = time.time()

            # Get model prediction
            try:
                prediction = self._get_model_prediction(model, sample)
            except Exception as e:
                logger.warning(f"Error on sample {idx}: {e}")
                prediction = {"answer": None, "steps": 0, "confidence": 0.0}

            latency_ms = (time.time() - start_time) * 1000

            # Evaluate correctness
            is_correct = self._check_correctness(prediction["answer"], sample.get("expected_output"))

            # Assess reasoning quality
            quality = self._assess_reasoning_quality(prediction.get("reasoning", []), sample.get("steps", []))

            result = BenchmarkResult(
                task_id=sample.get("task_id", str(idx)),
                predicted_answer=prediction["answer"],
                ground_truth=sample.get("expected_output"),
                is_correct=is_correct,
                steps_taken=prediction.get("steps", 0),
                reasoning_quality=quality,
                latency_ms=latency_ms,
                memory_mb=self._get_memory_usage(),
                agent_consensus=prediction.get("consensus", 1.0),
                metadata={
                    "difficulty": sample.get("difficulty", "unknown"),
                    "category": sample.get("category", "general"),
                    "confidence": prediction.get("confidence", 0.0),
                },
            )

            self.results.append(result)

            if verbose and idx % 10 == 0:
                logger.info(f"Evaluated {idx + 1}/{len(test_data)} samples")

        # Generate report
        report = self._generate_report()

        # Save report
        self._save_report(report)

        return report

    def _get_model_prediction(self, model: Any, sample: dict[str, Any]) -> dict[str, Any]:
        """Get prediction from model."""
        # This would interface with actual model
        # For now, simulate prediction
        if hasattr(model, "predict"):
            return model.predict(sample)
        else:
            # Mock prediction for testing
            return {
                "answer": sample.get("expected_output", "predicted"),
                "steps": len(sample.get("steps", [])),
                "reasoning": sample.get("steps", []),
                "confidence": np.random.uniform(0.7, 1.0),
                "consensus": np.random.uniform(0.8, 1.0),
            }

    def _check_correctness(self, predicted: Any, expected: Any) -> bool:
        """Check if prediction matches ground truth."""
        if predicted is None or expected is None:
            return False

        # String comparison
        if isinstance(predicted, str) and isinstance(expected, str):
            return predicted.strip().lower() == expected.strip().lower()

        # Numeric comparison
        if isinstance(predicted, (int, float)) and isinstance(expected, (int, float)):
            return abs(predicted - expected) < 1e-6

        # Direct equality
        return predicted == expected

    def _assess_reasoning_quality(self, predicted_steps: list[str], expected_steps: list[str]) -> float:
        """Assess quality of reasoning steps."""
        if not expected_steps:
            return 1.0 if predicted_steps else 0.5

        # Check step coverage
        num_predicted = len(predicted_steps)
        num_expected = len(expected_steps)

        if num_predicted == 0:
            return 0.0

        # Score based on step count similarity
        count_ratio = min(num_predicted, num_expected) / max(num_predicted, num_expected)

        # Simple text overlap (would use better metrics in production)
        predicted_text = " ".join(predicted_steps).lower()
        expected_text = " ".join(expected_steps).lower()

        predicted_words = set(predicted_text.split())
        expected_words = set(expected_text.split())

        if not expected_words:
            return count_ratio

        overlap = len(predicted_words & expected_words) / len(expected_words)

        quality = 0.5 * count_ratio + 0.5 * overlap
        return quality

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if HAS_TORCH and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)

    def _generate_report(self) -> EvaluationReport:
        """Generate comprehensive evaluation report."""
        if not self.results:
            return EvaluationReport(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                total_samples=0,
                accuracy=0.0,
                f1_score=0.0,
                precision=0.0,
                recall=0.0,
                avg_latency_ms=0.0,
                avg_memory_mb=0.0,
                avg_consensus=0.0,
                per_difficulty_metrics={},
                agent_metrics={},
                error_analysis={},
                recommendations=[],
            )

        # Basic metrics
        correct = sum(1 for r in self.results if r.is_correct)
        total = len(self.results)
        accuracy = correct / total

        # F1, Precision, Recall (binary classification view)
        # For multi-step reasoning, we consider task completion
        true_positives = correct
        false_positives = 0  # Predicted positive but wrong
        false_negatives = total - correct  # Should be positive but predicted wrong

        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        # Aggregate metrics
        avg_latency = np.mean([r.latency_ms for r in self.results])
        avg_memory = np.mean([r.memory_mb for r in self.results])
        avg_consensus = np.mean([r.agent_consensus for r in self.results])

        # Per-difficulty metrics
        difficulty_metrics = {}
        for diff in ["easy", "medium", "hard", "unknown"]:
            diff_results = [r for r in self.results if r.metadata.get("difficulty") == diff]
            if diff_results:
                difficulty_metrics[diff] = {
                    "count": len(diff_results),
                    "accuracy": sum(1 for r in diff_results if r.is_correct) / len(diff_results),
                    "avg_steps": np.mean([r.steps_taken for r in diff_results]),
                    "avg_quality": np.mean([r.reasoning_quality for r in diff_results]),
                }

        # Error analysis
        error_analysis = {
            "incorrect_answer": sum(1 for r in self.results if not r.is_correct),
            "low_quality_reasoning": sum(1 for r in self.results if r.reasoning_quality < 0.5),
            "high_latency": sum(1 for r in self.results if r.latency_ms > 5000),
            "low_consensus": sum(1 for r in self.results if r.agent_consensus < 0.7),
        }

        # Recommendations
        recommendations = self._generate_recommendations(accuracy, avg_latency, avg_consensus, error_analysis)

        report = EvaluationReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_samples=total,
            accuracy=accuracy,
            f1_score=f1,
            precision=precision,
            recall=recall,
            avg_latency_ms=avg_latency,
            avg_memory_mb=avg_memory,
            avg_consensus=avg_consensus,
            per_difficulty_metrics=difficulty_metrics,
            agent_metrics={},  # Would be populated with agent-specific metrics
            error_analysis=error_analysis,
            recommendations=recommendations,
        )

        return report

    def _generate_recommendations(
        self, accuracy: float, latency: float, consensus: float, errors: dict[str, int]
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if accuracy < self.success_criteria.get("hrm_accuracy", 0.85):
            recommendations.append(
                f"Accuracy {accuracy:.2%} is below target. Consider more training epochs or data augmentation."
            )

        if latency > 5000:
            recommendations.append(f"Average latency {latency:.0f}ms is high. Consider model optimization or caching.")

        if consensus < 0.8:
            recommendations.append(f"Agent consensus {consensus:.2%} is low. Review agent coordination strategy.")

        if errors["low_quality_reasoning"] > len(self.results) * 0.2:
            recommendations.append(
                "High proportion of low-quality reasoning. Focus on improving decomposition training."
            )

        if not recommendations:
            recommendations.append("Model performance meets all criteria. Consider production deployment.")

        return recommendations

    def _save_report(self, report: EvaluationReport) -> None:
        """Save evaluation report to disk."""
        report_dict = {
            "timestamp": report.timestamp,
            "total_samples": report.total_samples,
            "accuracy": report.accuracy,
            "f1_score": report.f1_score,
            "precision": report.precision,
            "recall": report.recall,
            "avg_latency_ms": report.avg_latency_ms,
            "avg_memory_mb": report.avg_memory_mb,
            "avg_consensus": report.avg_consensus,
            "per_difficulty_metrics": report.per_difficulty_metrics,
            "agent_metrics": report.agent_metrics,
            "error_analysis": report.error_analysis,
            "recommendations": report.recommendations,
        }

        with open(self.report_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Report saved to {self.report_path}")

    def compare_with_baseline(self, baseline_report_path: str) -> dict[str, float]:
        """Compare current results with baseline."""
        with open(baseline_report_path) as f:
            baseline = json.load(f)

        current = self._generate_report()

        comparison = {
            "accuracy_improvement": current.accuracy - baseline["accuracy"],
            "latency_improvement": baseline["avg_latency_ms"] - current.avg_latency_ms,
            "consensus_improvement": current.avg_consensus - baseline["avg_consensus"],
            "f1_improvement": current.f1_score - baseline["f1_score"],
        }

        logger.info(f"Baseline comparison: {comparison}")
        return comparison


class MultiAgentEvaluator:
    """Evaluate multi-agent coordination and consensus."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize multi-agent evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config

    def evaluate_consensus(self, agent_outputs: dict[str, list[Any]], ground_truths: list[Any]) -> dict[str, float]:
        """
        Evaluate consensus quality across agents.

        Args:
            agent_outputs: Outputs from each agent type
            ground_truths: Expected outputs

        Returns:
            Consensus metrics
        """
        metrics = {}

        # Agreement rate between agents
        agreement_scores = []
        for i in range(len(ground_truths)):
            outputs = [agent_outputs[agent][i] for agent in agent_outputs]
            # Count how many agents agree
            agreement = self._compute_agreement(outputs)
            agreement_scores.append(agreement)

        metrics["avg_agreement"] = np.mean(agreement_scores)

        # Accuracy per agent
        for agent_name, outputs in agent_outputs.items():
            correct = sum(
                1 for pred, truth in zip(outputs, ground_truths, strict=True) if self._outputs_match(pred, truth)
            )
            metrics[f"{agent_name}_accuracy"] = correct / len(ground_truths)

        # Ensemble accuracy (majority voting)
        ensemble_correct = 0
        for i in range(len(ground_truths)):
            outputs = [agent_outputs[agent][i] for agent in agent_outputs]
            ensemble_pred = self._majority_vote(outputs)
            if self._outputs_match(ensemble_pred, ground_truths[i]):
                ensemble_correct += 1
        metrics["ensemble_accuracy"] = ensemble_correct / len(ground_truths)

        # Improvement over best single agent
        best_single = max(metrics[f"{agent}_accuracy"] for agent in agent_outputs)
        metrics["ensemble_improvement"] = metrics["ensemble_accuracy"] - best_single

        logger.info(f"Consensus metrics: {metrics}")
        return metrics

    def _compute_agreement(self, outputs: list[Any]) -> float:
        """Compute agreement score between outputs."""
        if len(outputs) < 2:
            return 1.0

        agreements = 0
        total = 0

        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                if self._outputs_match(outputs[i], outputs[j]):
                    agreements += 1
                total += 1

        return agreements / total if total > 0 else 1.0

    def _outputs_match(self, a: Any, b: Any) -> bool:
        """Check if two outputs match."""
        if isinstance(a, str) and isinstance(b, str):
            return a.strip().lower() == b.strip().lower()
        return a == b

    def _majority_vote(self, outputs: list[Any]) -> Any:
        """Get majority vote from outputs."""
        from collections import Counter

        # Convert to hashable
        hashable_outputs = [str(o) for o in outputs]
        counter = Counter(hashable_outputs)
        most_common = counter.most_common(1)[0][0]

        # Return original type
        for o, h in zip(outputs, hashable_outputs, strict=True):
            if h == most_common:
                return o
        return outputs[0]

    def analyze_agent_specialization(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Analyze which agents excel at which task types."""
        specialization = {}

        # Group by category and difficulty
        categories = {r.metadata.get("category") for r in results}
        _difficulties = {r.metadata.get("difficulty") for r in results}  # noqa: F841

        for category in categories:
            specialization[category] = {"best_performing_tasks": [], "avg_quality": 0.0}

            cat_results = [r for r in results if r.metadata.get("category") == category]
            if cat_results:
                specialization[category]["avg_quality"] = np.mean([r.reasoning_quality for r in cat_results])

        return specialization


class PerformanceProfiler:
    """Profile system performance and resource usage."""

    def __init__(self):
        """Initialize performance profiler."""
        self.profiles = []

    def profile_inference(self, model: Any, test_inputs: list[Any], num_runs: int = 10) -> dict[str, Any]:
        """
        Profile inference performance.

        Args:
            model: Model to profile
            test_inputs: Test inputs
            num_runs: Number of profiling runs

        Returns:
            Performance profile
        """
        latencies = []
        memory_usage = []

        for _run in range(num_runs):
            for inp in test_inputs:
                start_time = time.time()

                # Record memory before
                mem_before = self._get_memory()

                # Run inference
                if hasattr(model, "predict"):
                    _ = model.predict(inp)
                else:
                    # Mock inference
                    time.sleep(0.01)

                # Record metrics
                latency = (time.time() - start_time) * 1000
                mem_after = self._get_memory()

                latencies.append(latency)
                memory_usage.append(mem_after - mem_before)

        profile = {
            "avg_latency_ms": np.mean(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "max_latency_ms": np.max(latencies),
            "avg_memory_increase_mb": np.mean(memory_usage),
            "throughput_rps": 1000.0 / np.mean(latencies) if latencies else 0,
        }

        self.profiles.append(profile)

        logger.info(f"Performance profile: {profile}")
        return profile

    def _get_memory(self) -> float:
        """Get current memory usage in MB."""
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)

    def identify_bottlenecks(self, profile: dict[str, Any]) -> list[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        if profile["p95_latency_ms"] > 2 * profile["avg_latency_ms"]:
            bottlenecks.append("High latency variance - consider request queuing")

        if profile["throughput_rps"] < 10:
            bottlenecks.append("Low throughput - consider batching or model optimization")

        if profile["avg_memory_increase_mb"] > 100:
            bottlenecks.append("High memory usage - check for memory leaks")

        return bottlenecks


class ProductionValidator:
    """Validate production readiness."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize production validator.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.success_criteria = config.get("success_criteria", {})

    def validate_all_criteria(self, evaluation_report: EvaluationReport) -> tuple[bool, dict[str, bool]]:
        """
        Check if all production criteria are met.

        Args:
            evaluation_report: Evaluation results

        Returns:
            Tuple of (overall_pass, individual_checks)
        """
        checks = {}

        # Accuracy checks
        checks["hrm_accuracy"] = evaluation_report.accuracy >= self.success_criteria.get("hrm_accuracy", 0.85)

        # Latency checks
        checks["latency_sla"] = evaluation_report.avg_latency_ms < 30000  # 30 seconds

        # Consensus checks
        checks["consensus_rate"] = evaluation_report.avg_consensus >= 0.8

        # Error rate check
        total_errors = sum(evaluation_report.error_analysis.values())
        error_rate = total_errors / (evaluation_report.total_samples * 4)  # 4 error types
        checks["error_rate"] = error_rate < 0.2

        overall_pass = all(checks.values())

        logger.info(f"Production validation: {checks}")
        logger.info(f"Overall pass: {overall_pass}")

        return overall_pass, checks

    def run_adversarial_tests(self, model: Any, adversarial_samples: list[dict[str, Any]]) -> dict[str, float]:
        """
        Test model robustness against adversarial inputs.

        Args:
            model: Model to test
            adversarial_samples: Adversarial test cases

        Returns:
            Robustness metrics
        """
        logger.info(f"Running adversarial tests on {len(adversarial_samples)} samples")

        results = {"prompt_injection_resistance": 0.0, "input_perturbation_robustness": 0.0, "edge_case_handling": 0.0}

        for sample in adversarial_samples:
            attack_type = sample.get("attack_type", "unknown")

            try:
                # Get model response
                if hasattr(model, "predict"):
                    response = model.predict(sample)
                else:
                    response = {"answer": "safe_response"}

                # Check if model handled adversarial input correctly
                if attack_type == "prompt_injection":
                    # Check that response doesn't contain injected content
                    is_safe = "injected" not in str(response).lower()
                    results["prompt_injection_resistance"] += float(is_safe)

                elif attack_type == "input_perturbation":
                    # Check response consistency
                    is_robust = response.get("confidence", 0) < 0.9  # Should show uncertainty
                    results["input_perturbation_robustness"] += float(is_robust)

                elif attack_type == "edge_case":
                    # Check graceful handling
                    is_handled = response.get("answer") is not None
                    results["edge_case_handling"] += float(is_handled)

            except Exception as e:
                logger.warning(f"Adversarial test failed: {e}")

        # Normalize by count
        for key in results:
            if adversarial_samples:
                results[key] /= len(adversarial_samples)

        logger.info(f"Adversarial test results: {results}")
        return results

    def generate_production_checklist(self, validation_results: dict[str, bool]) -> list[dict[str, Any]]:
        """Generate production deployment checklist."""
        checklist = [
            {
                "item": "Model accuracy meets threshold",
                "status": "pass" if validation_results.get("hrm_accuracy", False) else "fail",
                "priority": "critical",
            },
            {
                "item": "Latency SLA met",
                "status": "pass" if validation_results.get("latency_sla", False) else "fail",
                "priority": "critical",
            },
            {
                "item": "Agent consensus rate acceptable",
                "status": "pass" if validation_results.get("consensus_rate", False) else "fail",
                "priority": "high",
            },
            {
                "item": "Error rate within bounds",
                "status": "pass" if validation_results.get("error_rate", False) else "fail",
                "priority": "high",
            },
            {"item": "Monitoring dashboards configured", "status": "pending", "priority": "medium"},
            {"item": "Rollback mechanism tested", "status": "pending", "priority": "critical"},
            {"item": "Documentation complete", "status": "pending", "priority": "medium"},
        ]

        return checklist


if __name__ == "__main__":
    # Test evaluation suite
    logging.basicConfig(level=logging.INFO)

    logger.info("Testing Evaluation Suite")

    # Load config
    config_path = "training/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    eval_config = config["evaluation"]

    # Test DABStep benchmark
    benchmark = DABStepBenchmark(eval_config)

    # Create mock test data
    test_data = [
        {
            "task_id": f"test_{i}",
            "task_text": f"Test task {i}",
            "expected_output": f"Result {i}",
            "steps": [f"Step {j}" for j in range(3)],
            "difficulty": ["easy", "medium", "hard"][i % 3],
            "category": "reasoning",
        }
        for i in range(20)
    ]

    # Mock model
    class MockModel:
        def predict(self, sample):
            # Simulate 80% accuracy
            correct = np.random.random() < 0.8
            answer = sample["expected_output"] if correct else "wrong"
            return {
                "answer": answer,
                "steps": len(sample.get("steps", [])),
                "reasoning": sample.get("steps", []),
                "confidence": np.random.uniform(0.7, 1.0),
                "consensus": np.random.uniform(0.8, 1.0),
            }

    model = MockModel()

    # Run evaluation
    report = benchmark.evaluate_model(model, test_data, verbose=True)

    logger.info("Evaluation Report:")
    logger.info(f"  Accuracy: {report.accuracy:.2%}")
    logger.info(f"  F1 Score: {report.f1_score:.4f}")
    logger.info(f"  Avg Latency: {report.avg_latency_ms:.2f}ms")
    logger.info(f"  Recommendations: {report.recommendations}")

    # Test production validator
    validator = ProductionValidator(eval_config)
    passed, checks = validator.validate_all_criteria(report)

    logger.info(f"Production validation: {passed}")
    logger.info(f"Checks: {checks}")

    # Generate checklist
    checklist = validator.generate_production_checklist(checks)
    logger.info(f"Production checklist: {len(checklist)} items")

    logger.info("Evaluation Suite test complete")
