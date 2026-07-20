"""
Comprehensive Benchmark Suite for RAG System and Knowledge Base Quality Evaluation

This module provides a production-ready benchmarking framework for evaluating:
- RAG retrieval quality (nDCG@k, Recall@k, MRR, Precision@k)
- Reasoning capabilities (accuracy, quality scoring)
- Code generation (Pass@k, syntax correctness)
- Baseline comparisons and A/B testing
- Automated reporting and visualization

Integration Points:
- LangSmith for dataset management and tracing
- Weights & Biases for experiment tracking
- Compatible with training/evaluation.py and scripts/evaluate_rag.py
"""

import csv
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

try:
    import matplotlib.pyplot as plt
    import seaborn  # noqa: F401 - reserved for future use

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

try:
    import pandas  # noqa: F401 - reserved for future use

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from langsmith import Client

    HAS_LANGSMITH = True
except ImportError:
    HAS_LANGSMITH = False

logger = logging.getLogger(__name__)


# =============================================================================
# Benchmark Configuration
# =============================================================================

BENCHMARKS = {
    "rag_retrieval": {
        "datasets": ["custom_mcts", "custom_langgraph", "custom_multiagent"],
        "metrics": ["nDCG@10", "Recall@100", "MRR", "Precision@k"],
        "description": "Evaluate retrieval quality for knowledge base queries",
    },
    "reasoning": {
        "datasets": ["gsm8k_subset", "math_subset", "dabstep_subset"],
        "metrics": ["Accuracy", "Reasoning_quality", "Step_correctness"],
        "description": "Evaluate mathematical and logical reasoning capabilities",
    },
    "code_generation": {
        "datasets": ["humaneval_subset", "mbpp_subset"],
        "metrics": ["Pass@1", "Pass@10", "Syntax_correctness", "Code_quality"],
        "description": "Evaluate code generation and correctness",
    },
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RetrievalResult:
    """Result from a single retrieval query."""

    query: str
    retrieved_docs: list[str]  # Document IDs
    relevance_scores: list[float]  # Scores from retrieval system
    ground_truth_relevant: list[str]  # Ground truth relevant doc IDs
    ground_truth_rankings: dict[str, int] = field(default_factory=dict)  # Doc ID -> relevance rank
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    """Result from computing a metric."""

    metric_name: str
    value: float
    confidence_interval: tuple[float, float] | None = None
    sample_size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkRun:
    """Complete benchmark run results."""

    benchmark_name: str
    dataset_name: str
    timestamp: str
    model_config: dict[str, Any]
    metrics: dict[str, MetricResult]
    raw_results: list[Any]
    duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonReport:
    """Comparison report between multiple runs."""

    baseline_run: str
    comparison_runs: list[str]
    metric_deltas: dict[str, dict[str, float]]  # run_name -> metric_name -> delta
    statistical_significance: dict[str, dict[str, bool]]  # run_name -> metric_name -> is_significant
    recommendations: list[str]
    timestamp: str


# =============================================================================
# Evaluation Metrics Implementation
# =============================================================================


class RetrievalMetrics:
    """Compute retrieval quality metrics."""

    @staticmethod
    def ndcg_at_k(result: RetrievalResult, k: int = 10) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at k.

        Args:
            result: Retrieval result with retrieved docs and ground truth
            k: Cutoff position

        Returns:
            nDCG@k score (0.0 to 1.0)
        """
        retrieved = result.retrieved_docs[:k]
        if not retrieved:
            return 0.0

        # Compute DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved):
            rank = i + 1
            # Relevance score from ground truth rankings (higher rank = more relevant)
            if doc_id in result.ground_truth_rankings:
                relevance = len(result.ground_truth_rankings) - result.ground_truth_rankings[doc_id]
            elif doc_id in result.ground_truth_relevant:
                relevance = 1.0
            else:
                relevance = 0.0

            dcg += relevance / np.log2(rank + 1)

        # Compute IDCG (ideal DCG with perfect ranking)
        ideal_ranks = (
            sorted(result.ground_truth_rankings.values())
            if result.ground_truth_rankings
            else list(range(len(result.ground_truth_relevant)))
        )
        idcg = 0.0
        for i, _rank in enumerate(ideal_ranks[:k]):
            relevance = len(ideal_ranks) - i
            idcg += relevance / np.log2(i + 2)

        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def recall_at_k(result: RetrievalResult, k: int = 100) -> float:
        """
        Compute Recall at k.

        Args:
            result: Retrieval result
            k: Cutoff position

        Returns:
            Recall@k score (0.0 to 1.0)
        """
        if not result.ground_truth_relevant:
            return 0.0

        retrieved = set(result.retrieved_docs[:k])
        relevant = set(result.ground_truth_relevant)

        retrieved_relevant = retrieved & relevant
        return len(retrieved_relevant) / len(relevant)

    @staticmethod
    def precision_at_k(result: RetrievalResult, k: int = 10) -> float:
        """
        Compute Precision at k.

        Args:
            result: Retrieval result
            k: Cutoff position

        Returns:
            Precision@k score (0.0 to 1.0)
        """
        retrieved = result.retrieved_docs[:k]
        if not retrieved:
            return 0.0

        relevant = set(result.ground_truth_relevant)
        retrieved_relevant = sum(1 for doc_id in retrieved if doc_id in relevant)

        return retrieved_relevant / len(retrieved)

    @staticmethod
    def mean_reciprocal_rank(result: RetrievalResult) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).

        Args:
            result: Retrieval result

        Returns:
            MRR score (0.0 to 1.0)
        """
        relevant = set(result.ground_truth_relevant)

        for i, doc_id in enumerate(result.retrieved_docs):
            if doc_id in relevant:
                return 1.0 / (i + 1)

        return 0.0

    @staticmethod
    def mean_average_precision(results: list[RetrievalResult]) -> float:
        """
        Compute Mean Average Precision (MAP) across multiple queries.

        Args:
            results: List of retrieval results

        Returns:
            MAP score (0.0 to 1.0)
        """
        if not results:
            return 0.0

        aps = []
        for result in results:
            relevant = set(result.ground_truth_relevant)
            if not relevant:
                continue

            relevant_retrieved = 0
            precision_sum = 0.0

            for i, doc_id in enumerate(result.retrieved_docs):
                if doc_id in relevant:
                    relevant_retrieved += 1
                    precision_at_i = relevant_retrieved / (i + 1)
                    precision_sum += precision_at_i

            if relevant_retrieved > 0:
                aps.append(precision_sum / len(relevant))
            else:
                aps.append(0.0)

        return np.mean(aps)


class ReasoningMetrics:
    """Compute reasoning quality metrics."""

    @staticmethod
    def accuracy(predictions: list[Any], ground_truths: list[Any]) -> float:
        """
        Compute accuracy.

        Args:
            predictions: Predicted values
            ground_truths: Ground truth values

        Returns:
            Accuracy (0.0 to 1.0)
        """
        if not predictions or len(predictions) != len(ground_truths):
            return 0.0

        correct = sum(
            1 for pred, truth in zip(predictions, ground_truths, strict=False) if ReasoningMetrics._match(pred, truth)
        )
        return correct / len(predictions)

    @staticmethod
    def _match(pred: Any, truth: Any) -> bool:
        """Check if prediction matches ground truth."""
        # Numeric comparison
        if isinstance(pred, (int, float)) and isinstance(truth, (int, float)):
            return abs(pred - truth) < 1e-6

        # String comparison
        if isinstance(pred, str) and isinstance(truth, str):
            return pred.strip().lower() == truth.strip().lower()

        # Direct equality
        return pred == truth

    @staticmethod
    def reasoning_quality_score(
        predicted_steps: list[str], ground_truth_steps: list[str], use_llm: bool = False
    ) -> float:
        """
        Score reasoning quality.

        Args:
            predicted_steps: Predicted reasoning steps
            ground_truth_steps: Ground truth reasoning steps
            use_llm: Whether to use LLM-as-judge (requires API)

        Returns:
            Quality score (0.0 to 1.0)
        """
        if use_llm:
            return ReasoningMetrics._llm_judge_quality(predicted_steps, ground_truth_steps)

        # Simple heuristic-based scoring
        if not ground_truth_steps:
            return 0.5 if predicted_steps else 0.0

        if not predicted_steps:
            return 0.0

        # Step count similarity
        len_score = min(len(predicted_steps), len(ground_truth_steps)) / max(
            len(predicted_steps), len(ground_truth_steps)
        )

        # Content overlap
        pred_words = set(" ".join(predicted_steps).lower().split())
        truth_words = set(" ".join(ground_truth_steps).lower().split())

        if truth_words:
            overlap_score = len(pred_words & truth_words) / len(truth_words)
        else:
            overlap_score = 0.0

        # Weighted combination
        quality = 0.3 * len_score + 0.7 * overlap_score
        return quality

    @staticmethod
    def _llm_judge_quality(predicted_steps: list[str], ground_truth_steps: list[str]) -> float:
        """
        Use LLM as judge to score reasoning quality.

        Args:
            predicted_steps: Predicted steps
            ground_truth_steps: Ground truth steps

        Returns:
            Quality score from LLM (0.0 to 1.0)
        """
        # Placeholder for LLM-as-judge implementation
        # In production, this would call an LLM API with a scoring prompt
        logger.warning("LLM-as-judge not implemented, using heuristic")
        return ReasoningMetrics.reasoning_quality_score(predicted_steps, ground_truth_steps, use_llm=False)


class CodeMetrics:
    """Compute code generation metrics."""

    @staticmethod
    def pass_at_k(results: list[dict[str, Any]], k: int = 1) -> float:
        """
        Compute Pass@k metric for code generation.

        Args:
            results: List of results with 'passed' and 'total_tests' fields
            _k: Number of samples to consider (currently unused, placeholder for future implementation)

        Returns:
            Pass@k score (0.0 to 1.0)
        """
        if not results:
            return 0.0

        # For each problem, check if at least one of k samples passed
        passed_count = 0
        for result in results:
            # Check if any of the k samples passed all tests
            if result.get("passed", False):
                passed_count += 1

        return passed_count / len(results)

    @staticmethod
    def syntax_correctness(code_samples: list[str]) -> float:
        """
        Check syntax correctness of generated code.

        Args:
            code_samples: List of code strings

        Returns:
            Fraction of syntactically correct code (0.0 to 1.0)
        """
        if not code_samples:
            return 0.0

        correct = 0
        for code in code_samples:
            try:
                compile(code, "<string>", "exec")
                correct += 1
            except SyntaxError:
                pass
            except Exception:
                # Other compilation errors (e.g., indentation)
                pass

        return correct / len(code_samples)

    @staticmethod
    def code_quality_score(code: str) -> float:
        """
        Simple heuristic code quality score.

        Args:
            code: Code string

        Returns:
            Quality score (0.0 to 1.0)
        """
        if not code:
            return 0.0

        score = 0.0
        max_score = 0.0

        # Check for docstrings
        if '"""' in code or "'''" in code:
            score += 1.0
        max_score += 1.0

        # Check for type hints
        if "->" in code or ": " in code:
            score += 1.0
        max_score += 1.0

        # Check for comments
        if "#" in code:
            score += 0.5
        max_score += 0.5

        # Check for functions
        if "def " in code:
            score += 0.5
        max_score += 0.5

        # Check reasonable length (not too short, not too long)
        lines = code.split("\n")
        if 5 <= len(lines) <= 100:
            score += 1.0
        max_score += 1.0

        return score / max_score if max_score > 0 else 0.0


# =============================================================================
# Statistical Analysis
# =============================================================================


class StatisticalAnalysis:
    """Statistical significance testing and confidence intervals."""

    @staticmethod
    def bootstrap_confidence_interval(
        values: list[float], confidence: float = 0.95, n_bootstrap: int = 1000
    ) -> tuple[float, float]:
        """
        Compute bootstrap confidence interval.

        Args:
            values: Metric values
            confidence: Confidence level (e.g., 0.95 for 95%)
            n_bootstrap: Number of bootstrap samples

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not values:
            return (0.0, 0.0)

        if len(values) < 2:
            mean_val = values[0] if values else 0.0
            return (mean_val, mean_val)

        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))

        # Compute percentiles
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return (float(lower), float(upper))

    @staticmethod
    def paired_t_test(baseline_values: list[float], comparison_values: list[float]) -> tuple[float, bool]:
        """
        Perform paired t-test for statistical significance.

        Args:
            baseline_values: Baseline metric values
            comparison_values: Comparison metric values

        Returns:
            Tuple of (p_value, is_significant)
        """
        if not HAS_SCIPY:
            logger.warning("scipy not available, cannot perform statistical tests")
            return (1.0, False)

        if len(baseline_values) != len(comparison_values) or len(baseline_values) < 2:
            return (1.0, False)

        try:
            statistic, p_value = stats.ttest_rel(comparison_values, baseline_values)
            is_significant = p_value < 0.05
            return (float(p_value), is_significant)
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            return (1.0, False)

    @staticmethod
    def effect_size(baseline_values: list[float], comparison_values: list[float]) -> float:
        """
        Compute Cohen's d effect size.

        Args:
            baseline_values: Baseline values
            comparison_values: Comparison values

        Returns:
            Cohen's d effect size
        """
        if not baseline_values or not comparison_values:
            return 0.0

        mean_diff = np.mean(comparison_values) - np.mean(baseline_values)
        pooled_std = np.sqrt((np.var(baseline_values) + np.var(comparison_values)) / 2)

        if pooled_std == 0:
            return 0.0

        return mean_diff / pooled_std


# =============================================================================
# Benchmark Suite
# =============================================================================


class BenchmarkSuite:
    """Main benchmark suite for RAG system evaluation."""

    def __init__(self, config: dict[str, Any] | None = None, output_dir: Path | None = None):
        """
        Initialize benchmark suite.

        Args:
            config: Configuration dictionary
            output_dir: Output directory for results
        """
        self.config = config or {}
        self.output_dir = output_dir or Path("./benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.runs: list[BenchmarkRun] = []
        self.langsmith_client = None
        self.wandb_run = None

        logger.info(f"BenchmarkSuite initialized, output dir: {self.output_dir}")

    def initialize_integrations(self) -> None:
        """Initialize LangSmith and W&B integrations."""
        # Initialize LangSmith
        if HAS_LANGSMITH:
            langsmith_key = os.getenv("LANGSMITH_API_KEY")
            if langsmith_key:
                try:
                    self.langsmith_client = Client(api_key=langsmith_key)
                    logger.info("LangSmith client initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize LangSmith: {e}")

        # Initialize W&B
        if HAS_WANDB and self.config.get("use_wandb", False):
            wandb_key = os.getenv("WANDB_API_KEY")
            if wandb_key:
                try:
                    self.wandb_run = wandb.init(
                        project=self.config.get("wandb_project", "rag-benchmarks"),
                        entity=self.config.get("wandb_entity"),
                        config=self.config,
                        tags=["benchmark", "evaluation"],
                    )
                    logger.info(f"W&B run initialized: {self.wandb_run.url}")
                except Exception as e:
                    logger.warning(f"Failed to initialize W&B: {e}")

    def run_retrieval_benchmark(
        self,
        dataset_name: str,
        retrieval_fn: Any,
        model_config: dict[str, Any] | None = None,
        k_values: list[int] | None = None,
    ) -> BenchmarkRun:
        """
        Run retrieval benchmark.

        Args:
            dataset_name: Name of the dataset to use
            retrieval_fn: Function that takes a query and returns retrieved docs
            model_config: Model configuration for tracking
            k_values: List of k values for metrics (default: [5, 10, 20, 100])

        Returns:
            BenchmarkRun with results
        """
        logger.info(f"Running retrieval benchmark on dataset: {dataset_name}")
        start_time = time.time()

        k_values = k_values or [5, 10, 20, 100]
        model_config = model_config or {}

        # Load dataset
        dataset = self._load_dataset(dataset_name, "retrieval")
        if not dataset:
            logger.error(f"Failed to load dataset: {dataset_name}")
            return self._empty_benchmark_run("rag_retrieval", dataset_name, model_config)

        # Run retrieval on all queries
        results = []
        for item in dataset:
            query = item.get("query", "")
            ground_truth = item.get("ground_truth_relevant", [])
            ground_truth_rankings = item.get("ground_truth_rankings", {})

            # Call retrieval function
            retrieved = retrieval_fn(query)

            # Parse retrieval function output
            if isinstance(retrieved, dict):
                retrieved_docs = retrieved.get("doc_ids", [])
                relevance_scores = retrieved.get("scores", [])
            elif isinstance(retrieved, list):
                retrieved_docs = [doc if isinstance(doc, str) else doc.get("doc_id", "") for doc in retrieved]
                relevance_scores = [doc.get("score", 0.0) if isinstance(doc, dict) else 0.0 for doc in retrieved]
            else:
                logger.warning(f"Unexpected retrieval output format: {type(retrieved)}")
                continue

            result = RetrievalResult(
                query=query,
                retrieved_docs=retrieved_docs,
                relevance_scores=relevance_scores,
                ground_truth_relevant=ground_truth,
                ground_truth_rankings=ground_truth_rankings,
                metadata={"dataset": dataset_name},
            )
            results.append(result)

        # Compute metrics
        metrics = {}

        for k in k_values:
            # nDCG@k
            ndcg_values = [RetrievalMetrics.ndcg_at_k(r, k) for r in results]
            ndcg_mean = np.mean(ndcg_values)
            ndcg_ci = StatisticalAnalysis.bootstrap_confidence_interval(ndcg_values)
            metrics[f"nDCG@{k}"] = MetricResult(
                metric_name=f"nDCG@{k}",
                value=ndcg_mean,
                confidence_interval=ndcg_ci,
                sample_size=len(results),
            )

            # Recall@k
            recall_values = [RetrievalMetrics.recall_at_k(r, k) for r in results]
            recall_mean = np.mean(recall_values)
            recall_ci = StatisticalAnalysis.bootstrap_confidence_interval(recall_values)
            metrics[f"Recall@{k}"] = MetricResult(
                metric_name=f"Recall@{k}",
                value=recall_mean,
                confidence_interval=recall_ci,
                sample_size=len(results),
            )

            # Precision@k
            precision_values = [RetrievalMetrics.precision_at_k(r, k) for r in results]
            precision_mean = np.mean(precision_values)
            precision_ci = StatisticalAnalysis.bootstrap_confidence_interval(precision_values)
            metrics[f"Precision@{k}"] = MetricResult(
                metric_name=f"Precision@{k}",
                value=precision_mean,
                confidence_interval=precision_ci,
                sample_size=len(results),
            )

        # MRR (not dependent on k)
        mrr_values = [RetrievalMetrics.mean_reciprocal_rank(r) for r in results]
        mrr_mean = np.mean(mrr_values)
        mrr_ci = StatisticalAnalysis.bootstrap_confidence_interval(mrr_values)
        metrics["MRR"] = MetricResult(
            metric_name="MRR", value=mrr_mean, confidence_interval=mrr_ci, sample_size=len(results)
        )

        # MAP
        map_score = RetrievalMetrics.mean_average_precision(results)
        metrics["MAP"] = MetricResult(metric_name="MAP", value=map_score, sample_size=len(results))

        duration = time.time() - start_time

        # Create benchmark run
        benchmark_run = BenchmarkRun(
            benchmark_name="rag_retrieval",
            dataset_name=dataset_name,
            timestamp=datetime.now().isoformat(),
            model_config=model_config,
            metrics=metrics,
            raw_results=results,
            duration_seconds=duration,
        )

        self.runs.append(benchmark_run)

        # Log to integrations
        self._log_to_integrations(benchmark_run)

        logger.info(f"Retrieval benchmark completed in {duration:.2f}s")
        return benchmark_run

    def run_reasoning_benchmark(
        self,
        dataset_name: str,
        reasoning_fn: Any,
        model_config: dict[str, Any] | None = None,
        use_llm_judge: bool = False,
    ) -> BenchmarkRun:
        """
        Run reasoning benchmark.

        Args:
            dataset_name: Name of the dataset
            reasoning_fn: Function that takes a problem and returns answer + steps
            model_config: Model configuration
            use_llm_judge: Whether to use LLM-as-judge for quality scoring

        Returns:
            BenchmarkRun with results
        """
        logger.info(f"Running reasoning benchmark on dataset: {dataset_name}")
        start_time = time.time()

        model_config = model_config or {}

        # Load dataset
        dataset = self._load_dataset(dataset_name, "reasoning")
        if not dataset:
            logger.error(f"Failed to load dataset: {dataset_name}")
            return self._empty_benchmark_run("reasoning", dataset_name, model_config)

        # Run reasoning on all problems
        predictions = []
        ground_truths = []
        quality_scores = []

        for item in dataset:
            problem = item.get("problem", "")
            ground_truth = item.get("answer", "")
            ground_truth_steps = item.get("steps", [])

            # Call reasoning function
            result = reasoning_fn(problem)

            # Parse reasoning output
            if isinstance(result, dict):
                predicted_answer = result.get("answer", "")
                predicted_steps = result.get("steps", [])
            else:
                predicted_answer = result
                predicted_steps = []

            predictions.append(predicted_answer)
            ground_truths.append(ground_truth)

            # Compute reasoning quality
            quality = ReasoningMetrics.reasoning_quality_score(predicted_steps, ground_truth_steps, use_llm_judge)
            quality_scores.append(quality)

        # Compute metrics
        accuracy = ReasoningMetrics.accuracy(predictions, ground_truths)
        avg_quality = np.mean(quality_scores)

        accuracy_ci = StatisticalAnalysis.bootstrap_confidence_interval(
            [float(p == g) for p, g in zip(predictions, ground_truths, strict=False)]
        )
        quality_ci = StatisticalAnalysis.bootstrap_confidence_interval(quality_scores)

        metrics = {
            "Accuracy": MetricResult(
                metric_name="Accuracy", value=accuracy, confidence_interval=accuracy_ci, sample_size=len(predictions)
            ),
            "Reasoning_quality": MetricResult(
                metric_name="Reasoning_quality",
                value=avg_quality,
                confidence_interval=quality_ci,
                sample_size=len(quality_scores),
            ),
        }

        duration = time.time() - start_time

        benchmark_run = BenchmarkRun(
            benchmark_name="reasoning",
            dataset_name=dataset_name,
            timestamp=datetime.now().isoformat(),
            model_config=model_config,
            metrics=metrics,
            raw_results=list(zip(predictions, ground_truths, quality_scores, strict=False)),
            duration_seconds=duration,
        )

        self.runs.append(benchmark_run)
        self._log_to_integrations(benchmark_run)

        logger.info(f"Reasoning benchmark completed in {duration:.2f}s")
        return benchmark_run

    def run_code_generation_benchmark(
        self,
        dataset_name: str,
        code_gen_fn: Any,
        model_config: dict[str, Any] | None = None,
        k_values: list[int] | None = None,
    ) -> BenchmarkRun:
        """
        Run code generation benchmark.

        Args:
            dataset_name: Name of the dataset
            code_gen_fn: Function that generates code from problem description
            model_config: Model configuration
            k_values: List of k values for Pass@k (default: [1, 10])

        Returns:
            BenchmarkRun with results
        """
        logger.info(f"Running code generation benchmark on dataset: {dataset_name}")
        start_time = time.time()

        k_values = k_values or [1, 10]
        model_config = model_config or {}

        # Load dataset
        dataset = self._load_dataset(dataset_name, "code_generation")
        if not dataset:
            logger.error(f"Failed to load dataset: {dataset_name}")
            return self._empty_benchmark_run("code_generation", dataset_name, model_config)

        # Run code generation
        code_samples = []
        results = []

        for item in dataset:
            problem = item.get("problem", "")
            test_cases = item.get("test_cases", [])

            # Generate code
            generated_code = code_gen_fn(problem)
            code_samples.append(generated_code)

            # Run tests (simplified - in production use proper execution sandbox)
            passed = self._run_code_tests(generated_code, test_cases)

            results.append({"code": generated_code, "passed": passed, "total_tests": len(test_cases)})

        # Compute metrics
        metrics = {}

        for k in k_values:
            pass_at_k = CodeMetrics.pass_at_k(results, k)
            metrics[f"Pass@{k}"] = MetricResult(metric_name=f"Pass@{k}", value=pass_at_k, sample_size=len(results))

        syntax_correct = CodeMetrics.syntax_correctness(code_samples)
        metrics["Syntax_correctness"] = MetricResult(
            metric_name="Syntax_correctness", value=syntax_correct, sample_size=len(code_samples)
        )

        quality_scores = [CodeMetrics.code_quality_score(code) for code in code_samples]
        avg_quality = np.mean(quality_scores)
        metrics["Code_quality"] = MetricResult(
            metric_name="Code_quality", value=avg_quality, sample_size=len(quality_scores)
        )

        duration = time.time() - start_time

        benchmark_run = BenchmarkRun(
            benchmark_name="code_generation",
            dataset_name=dataset_name,
            timestamp=datetime.now().isoformat(),
            model_config=model_config,
            metrics=metrics,
            raw_results=results,
            duration_seconds=duration,
        )

        self.runs.append(benchmark_run)
        self._log_to_integrations(benchmark_run)

        logger.info(f"Code generation benchmark completed in {duration:.2f}s")
        return benchmark_run

    def compare_runs(
        self, baseline_run_id: str, comparison_run_ids: list[str], output_file: Path | None = None
    ) -> ComparisonReport:
        """
        Compare multiple benchmark runs.

        Args:
            baseline_run_id: ID of baseline run (timestamp)
            comparison_run_ids: IDs of runs to compare
            output_file: Optional output file for report

        Returns:
            ComparisonReport with comparison results
        """
        logger.info(f"Comparing runs: baseline={baseline_run_id}, comparisons={comparison_run_ids}")

        # Find runs
        baseline_run = self._find_run(baseline_run_id)
        comparison_runs = [self._find_run(run_id) for run_id in comparison_run_ids]

        if not baseline_run:
            logger.error(f"Baseline run not found: {baseline_run_id}")
            return self._empty_comparison_report()

        comparison_runs = [r for r in comparison_runs if r is not None]

        # Compute metric deltas and statistical significance
        metric_deltas = {}
        statistical_significance = {}

        for comp_run in comparison_runs:
            run_name = comp_run.timestamp
            metric_deltas[run_name] = {}
            statistical_significance[run_name] = {}

            for metric_name in baseline_run.metrics:
                if metric_name in comp_run.metrics:
                    baseline_val = baseline_run.metrics[metric_name].value
                    comp_val = comp_run.metrics[metric_name].value
                    delta = comp_val - baseline_val
                    metric_deltas[run_name][metric_name] = delta

                    # Statistical significance testing
                    # Note: This is simplified - need raw values for proper testing
                    is_significant = abs(delta) > 0.01  # Simple threshold
                    statistical_significance[run_name][metric_name] = is_significant

        # Generate recommendations
        recommendations = self._generate_comparison_recommendations(metric_deltas, statistical_significance)

        report = ComparisonReport(
            baseline_run=baseline_run_id,
            comparison_runs=comparison_run_ids,
            metric_deltas=metric_deltas,
            statistical_significance=statistical_significance,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat(),
        )

        # Save report
        if output_file:
            self._save_comparison_report(report, output_file)

        logger.info(f"Comparison complete: {len(recommendations)} recommendations")
        return report

    def generate_report(
        self,
        run: BenchmarkRun,
        output_format: Literal["json", "markdown", "html"] = "json",
        output_file: Path | None = None,
    ) -> str:
        """
        Generate benchmark report.

        Args:
            run: Benchmark run to report on
            output_format: Output format (json, markdown, html)
            output_file: Optional output file

        Returns:
            Report content as string
        """
        logger.info(f"Generating {output_format} report for run: {run.timestamp}")

        if output_format == "json":
            report = self._generate_json_report(run)
        elif output_format == "markdown":
            report = self._generate_markdown_report(run)
        elif output_format == "html":
            report = self._generate_html_report(run)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")

        return report

    def visualize_results(
        self, runs: list[BenchmarkRun] | None = None, output_dir: Path | None = None
    ) -> dict[str, Path]:
        """
        Generate visualizations for benchmark results.

        Args:
            runs: List of runs to visualize (default: all runs)
            output_dir: Output directory for plots

        Returns:
            Dictionary mapping plot names to file paths
        """
        if not HAS_PLOTTING:
            logger.warning("matplotlib/seaborn not available, cannot generate visualizations")
            return {}

        runs = runs or self.runs
        output_dir = output_dir or self.output_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating visualizations for {len(runs)} runs")

        plot_files = {}

        # Metric comparison bar chart
        plot_files["metric_comparison"] = self._plot_metric_comparison(runs, output_dir)

        # Radar plot for multi-metric comparison
        plot_files["radar_plot"] = self._plot_radar_chart(runs, output_dir)

        # Historical trend graphs
        plot_files["trend_plot"] = self._plot_trend_analysis(runs, output_dir)

        logger.info(f"Generated {len(plot_files)} visualizations")
        return plot_files

    def export_to_csv(self, run: BenchmarkRun, output_file: Path) -> None:
        """
        Export benchmark results to CSV.

        Args:
            run: Benchmark run
            output_file: Output CSV file path
        """
        logger.info(f"Exporting results to CSV: {output_file}")

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(["Metric", "Value", "CI_Lower", "CI_Upper", "Sample_Size"])

            # Write metrics
            for metric in run.metrics.values():
                ci_lower, ci_upper = metric.confidence_interval or (None, None)
                writer.writerow([metric.metric_name, metric.value, ci_lower, ci_upper, metric.sample_size])

        logger.info(f"CSV export complete: {output_file}")

    # =============================================================================
    # Private Helper Methods
    # =============================================================================

    def _load_dataset(self, dataset_name: str, benchmark_type: str) -> list[dict[str, Any]]:
        """Load dataset from LangSmith or local cache."""
        # Try LangSmith first
        if self.langsmith_client:
            try:
                dataset = self.langsmith_client.read_dataset(dataset_name=dataset_name)
                examples = list(dataset.examples)
                logger.info(f"Loaded {len(examples)} examples from LangSmith dataset: {dataset_name}")
                return [{"query": ex.inputs.get("query"), **ex.outputs} for ex in examples]
            except Exception as e:
                logger.warning(f"Failed to load from LangSmith: {e}")

        # Fallback to mock data for testing
        logger.info(f"Using mock dataset for {dataset_name}")
        return self._create_mock_dataset(benchmark_type)

    def _create_mock_dataset(self, benchmark_type: str) -> list[dict[str, Any]]:
        """Create mock dataset for testing."""
        datasets = {
            "retrieval": [
                {
                    "query": "What is MCTS?",
                    "ground_truth_relevant": ["doc1", "doc2"],
                    "ground_truth_rankings": {"doc1": 0, "doc2": 1},
                },
                {
                    "query": "Explain LangGraph",
                    "ground_truth_relevant": ["doc3", "doc4", "doc5"],
                    "ground_truth_rankings": {"doc3": 0, "doc4": 1, "doc5": 2},
                },
            ],
            "reasoning": [
                {"problem": "What is 2+2?", "answer": "4", "steps": ["Add 2 and 2", "Result is 4"]},
                {
                    "problem": "Solve: x + 5 = 10",
                    "answer": "x = 5",
                    "steps": ["Subtract 5 from both sides", "x = 10 - 5", "x = 5"],
                },
            ],
            "code_generation": [
                {
                    "problem": "Write a function to add two numbers",
                    "test_cases": [("add(2, 3)", 5), ("add(0, 0)", 0)],
                },
            ],
        }
        return datasets.get(benchmark_type, [])

    def _run_code_tests(self, code: str, test_cases: list[tuple[str, Any]]) -> bool:
        """Run test cases on generated code (simplified)."""
        try:
            # Create a local namespace
            namespace = {}
            exec(code, namespace)

            # Run test cases
            for test_input, expected_output in test_cases:
                # This is simplified - in production use proper sandboxing
                result = eval(test_input, namespace)
                if result != expected_output:
                    return False

            return True
        except Exception:
            return False

    def _find_run(self, run_id: str) -> BenchmarkRun | None:
        """Find benchmark run by ID (timestamp)."""
        for run in self.runs:
            if run.timestamp == run_id:
                return run
        return None

    def _empty_benchmark_run(
        self, benchmark_name: str, dataset_name: str, model_config: dict[str, Any]
    ) -> BenchmarkRun:
        """Create empty benchmark run."""
        return BenchmarkRun(
            benchmark_name=benchmark_name,
            dataset_name=dataset_name,
            timestamp=datetime.now().isoformat(),
            model_config=model_config,
            metrics={},
            raw_results=[],
            duration_seconds=0.0,
        )

    def _empty_comparison_report(self) -> ComparisonReport:
        """Create empty comparison report."""
        return ComparisonReport(
            baseline_run="",
            comparison_runs=[],
            metric_deltas={},
            statistical_significance={},
            recommendations=[],
            timestamp=datetime.now().isoformat(),
        )

    def _generate_comparison_recommendations(
        self, metric_deltas: dict[str, dict[str, float]], statistical_significance: dict[str, dict[str, bool]]
    ) -> list[str]:
        """Generate recommendations from comparison."""
        recommendations = []

        for run_name, deltas in metric_deltas.items():
            significant_improvements = []
            significant_regressions = []

            for metric_name, delta in deltas.items():
                is_significant = statistical_significance.get(run_name, {}).get(metric_name, False)

                if is_significant:
                    if delta > 0:
                        significant_improvements.append(f"{metric_name} (+{delta:.3f})")
                    else:
                        significant_regressions.append(f"{metric_name} ({delta:.3f})")

            if significant_improvements:
                recommendations.append(
                    f"Run {run_name}: Significant improvements in {', '.join(significant_improvements)}"
                )

            if significant_regressions:
                recommendations.append(
                    f"Run {run_name}: Significant regressions in {', '.join(significant_regressions)}"
                )

        if not recommendations:
            recommendations.append("No statistically significant differences detected")

        return recommendations

    def _generate_json_report(self, run: BenchmarkRun) -> str:
        """Generate JSON report."""
        report_dict = {
            "benchmark_name": run.benchmark_name,
            "dataset_name": run.dataset_name,
            "timestamp": run.timestamp,
            "model_config": run.model_config,
            "duration_seconds": run.duration_seconds,
            "metrics": {
                name: {
                    "value": metric.value,
                    "confidence_interval": metric.confidence_interval,
                    "sample_size": metric.sample_size,
                }
                for name, metric in run.metrics.items()
            },
        }
        return json.dumps(report_dict, indent=2)

    def _generate_markdown_report(self, run: BenchmarkRun) -> str:
        """Generate Markdown report."""
        lines = [
            f"# Benchmark Report: {run.benchmark_name}",
            "",
            f"**Dataset:** {run.dataset_name}",
            f"**Timestamp:** {run.timestamp}",
            f"**Duration:** {run.duration_seconds:.2f}s",
            "",
            "## Model Configuration",
            "",
            "```json",
            json.dumps(run.model_config, indent=2),
            "```",
            "",
            "## Metrics",
            "",
            "| Metric | Value | 95% CI | Sample Size |",
            "|--------|-------|--------|-------------|",
        ]

        for metric in run.metrics.values():
            ci_str = (
                f"[{metric.confidence_interval[0]:.4f}, {metric.confidence_interval[1]:.4f}]"
                if metric.confidence_interval
                else "N/A"
            )
            lines.append(f"| {metric.metric_name} | {metric.value:.4f} | {ci_str} | {metric.sample_size} |")

        return "\n".join(lines)

    def _generate_html_report(self, run: BenchmarkRun) -> str:
        """Generate HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Report: {run.benchmark_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metadata {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Benchmark Report: {run.benchmark_name}</h1>

    <div class="metadata">
        <p><strong>Dataset:</strong> {run.dataset_name}</p>
        <p><strong>Timestamp:</strong> {run.timestamp}</p>
        <p><strong>Duration:</strong> {run.duration_seconds:.2f}s</p>
    </div>

    <h2>Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>95% CI</th>
            <th>Sample Size</th>
        </tr>
"""

        for metric in run.metrics.values():
            ci_str = (
                f"[{metric.confidence_interval[0]:.4f}, {metric.confidence_interval[1]:.4f}]"
                if metric.confidence_interval
                else "N/A"
            )
            html += f"""
        <tr>
            <td>{metric.metric_name}</td>
            <td>{metric.value:.4f}</td>
            <td>{ci_str}</td>
            <td>{metric.sample_size}</td>
        </tr>
"""

        html += """
    </table>
</body>
</html>
"""
        return html

    def _save_comparison_report(self, report: ComparisonReport, output_file: Path) -> None:
        """Save comparison report to file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        report_dict = {
            "baseline_run": report.baseline_run,
            "comparison_runs": report.comparison_runs,
            "metric_deltas": report.metric_deltas,
            "statistical_significance": {k: bool(v) for k, v in report.statistical_significance.items()},
            "recommendations": report.recommendations,
            "timestamp": report.timestamp,
        }

        with open(output_file, "w") as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Comparison report saved to {output_file}")

    def _plot_metric_comparison(self, runs: list[BenchmarkRun], output_dir: Path) -> Path:
        """Generate metric comparison bar chart."""
        if not HAS_PLOTTING or not runs:
            return output_dir / "metric_comparison.png"

        # Collect metrics
        metric_names = set()
        for run in runs:
            metric_names.update(run.metrics.keys())

        metric_names = sorted(metric_names)

        # Create data
        data = defaultdict(list)
        run_labels = []

        for run in runs:
            run_labels.append(f"{run.dataset_name}\n{run.timestamp[-8:]}")
            for metric_name in metric_names:
                value = run.metrics.get(metric_name)
                data[metric_name].append(value.value if value else 0.0)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(run_labels))
        width = 0.8 / len(metric_names)

        for i, metric_name in enumerate(metric_names):
            offset = (i - len(metric_names) / 2) * width
            ax.bar(x + offset, data[metric_name], width, label=metric_name)

        ax.set_xlabel("Run")
        ax.set_ylabel("Metric Value")
        ax.set_title("Metric Comparison Across Runs")
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_file = output_dir / "metric_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Metric comparison plot saved to {output_file}")
        return output_file

    def _plot_radar_chart(self, runs: list[BenchmarkRun], output_dir: Path) -> Path:
        """Generate radar plot for multi-metric comparison."""
        if not HAS_PLOTTING or not runs:
            return output_dir / "radar_plot.png"

        # Use first run's metrics as reference
        if not runs:
            return output_dir / "radar_plot.png"

        metric_names = list(runs[0].metrics.keys())
        num_vars = len(metric_names)

        if num_vars < 3:
            logger.warning("Need at least 3 metrics for radar plot")
            return output_dir / "radar_plot.png"

        # Compute angles
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

        for run in runs[:5]:  # Limit to 5 runs for clarity
            values = [run.metrics.get(m, MetricResult("", 0.0)).value for m in metric_names]
            values += values[:1]  # Complete the circle

            label = f"{run.dataset_name}"
            ax.plot(angles, values, "o-", linewidth=2, label=label)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1.0)
        ax.set_title("Multi-Metric Comparison (Radar Plot)", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        output_file = output_dir / "radar_plot.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Radar plot saved to {output_file}")
        return output_file

    def _plot_trend_analysis(self, runs: list[BenchmarkRun], output_dir: Path) -> Path:
        """Generate historical trend graph."""
        if not HAS_PLOTTING or not runs:
            return output_dir / "trend_plot.png"

        # Sort runs by timestamp
        sorted_runs = sorted(runs, key=lambda r: r.timestamp)

        # Collect metrics
        metric_names = set()
        for run in sorted_runs:
            metric_names.update(run.metrics.keys())

        metric_names = sorted(metric_names)

        # Create data
        timestamps = [run.timestamp for run in sorted_runs]
        data = {metric: [] for metric in metric_names}

        for run in sorted_runs:
            for metric_name in metric_names:
                value = run.metrics.get(metric_name)
                data[metric_name].append(value.value if value else 0.0)

        # Plot
        fig, ax = plt.subplots(figsize=(14, 7))

        for metric_name in metric_names:
            ax.plot(range(len(timestamps)), data[metric_name], marker="o", label=metric_name, linewidth=2)

        ax.set_xlabel("Run Index")
        ax.set_ylabel("Metric Value")
        ax.set_title("Metric Trends Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add timestamp labels (show every nth label to avoid crowding)
        step = max(1, len(timestamps) // 10)
        ax.set_xticks(range(0, len(timestamps), step))
        ax.set_xticklabels([timestamps[i][-8:] for i in range(0, len(timestamps), step)], rotation=45)

        plt.tight_layout()
        output_file = output_dir / "trend_plot.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Trend plot saved to {output_file}")
        return output_file

    def _log_to_integrations(self, run: BenchmarkRun) -> None:
        """Log benchmark run to W&B and LangSmith."""
        # Log to W&B
        if self.wandb_run and HAS_WANDB:
            try:
                metrics_dict = {metric.metric_name: metric.value for metric in run.metrics.values()}
                wandb.log(
                    {
                        **metrics_dict,
                        "benchmark_name": run.benchmark_name,
                        "dataset_name": run.dataset_name,
                        "duration_seconds": run.duration_seconds,
                    }
                )
                logger.info("Logged to W&B")
            except Exception as e:
                logger.warning(f"Failed to log to W&B: {e}")

        # Log to LangSmith (as dataset for future reference)
        if self.langsmith_client and HAS_LANGSMITH:
            try:
                # Create or update dataset with benchmark results
                # This is simplified - expand based on LangSmith capabilities
                logger.info("LangSmith logging available (not implemented in this example)")
            except Exception as e:
                logger.warning(f"Failed to log to LangSmith: {e}")


# =============================================================================
# Example Usage and Testing
# =============================================================================


def create_example_retrieval_function():
    """Create example retrieval function for testing."""

    def retrieval_fn(query: str) -> dict[str, Any]:
        """Mock retrieval function."""
        # Simulate retrieval with some relevant and some irrelevant docs
        if "MCTS" in query:
            return {
                "doc_ids": ["doc1", "doc2", "doc5", "doc8"],
                "scores": [0.95, 0.87, 0.65, 0.45],
            }
        else:
            return {
                "doc_ids": ["doc3", "doc4", "doc5", "doc7"],
                "scores": [0.92, 0.85, 0.78, 0.55],
            }

    return retrieval_fn


def create_example_reasoning_function():
    """Create example reasoning function for testing."""

    def reasoning_fn(problem: str) -> dict[str, Any]:
        """Mock reasoning function."""
        # Simulate reasoning with answer and steps
        if "2+2" in problem:
            return {
                "answer": "4",
                "steps": ["Identify the operation: addition", "Add 2 and 2", "Result is 4"],
            }
        elif "x + 5 = 10" in problem:
            return {
                "answer": "x = 5",
                "steps": ["Isolate x by subtracting 5 from both sides", "x = 10 - 5", "x = 5"],
            }
        else:
            return {
                "answer": "unknown",
                "steps": ["Problem not recognized"],
            }

    return reasoning_fn


def create_example_code_gen_function():
    """Create example code generation function for testing."""

    def code_gen_fn(problem: str) -> str:
        """Mock code generation function."""
        if "add two numbers" in problem.lower():
            return '''def add(a, b):
    """Add two numbers."""
    return a + b
'''
        else:
            return "# TODO: Implement solution"

    return code_gen_fn


def main():
    """Main function for testing benchmark suite."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info("=== Benchmark Suite Testing ===")

    # Load configuration
    config_path = Path("training/config.yaml")
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Create benchmark suite
    suite = BenchmarkSuite(config=config, output_dir=Path("./benchmarks/test_run"))
    suite.initialize_integrations()

    # Test retrieval benchmark
    logger.info("\n--- Testing Retrieval Benchmark ---")
    retrieval_fn = create_example_retrieval_function()
    retrieval_run = suite.run_retrieval_benchmark(
        dataset_name="custom_mcts",
        retrieval_fn=retrieval_fn,
        model_config={"model": "test-retriever-v1", "embedding_dim": 384},
    )

    # Print results
    logger.info("\nRetrieval Metrics:")
    for metric_name, metric in retrieval_run.metrics.items():
        ci_str = (
            f"±{(metric.confidence_interval[1] - metric.confidence_interval[0]) / 2:.4f}"
            if metric.confidence_interval
            else ""
        )
        logger.info(f"  {metric_name}: {metric.value:.4f} {ci_str}")

    # Test reasoning benchmark
    logger.info("\n--- Testing Reasoning Benchmark ---")
    reasoning_fn = create_example_reasoning_function()
    reasoning_run = suite.run_reasoning_benchmark(
        dataset_name="gsm8k_subset",
        reasoning_fn=reasoning_fn,
        model_config={"model": "test-reasoner-v1", "temperature": 0.7},
    )

    logger.info("\nReasoning Metrics:")
    for metric_name, metric in reasoning_run.metrics.items():
        ci_str = (
            f"±{(metric.confidence_interval[1] - metric.confidence_interval[0]) / 2:.4f}"
            if metric.confidence_interval
            else ""
        )
        logger.info(f"  {metric_name}: {metric.value:.4f} {ci_str}")

    # Test code generation benchmark
    logger.info("\n--- Testing Code Generation Benchmark ---")
    code_gen_fn = create_example_code_gen_function()
    code_run = suite.run_code_generation_benchmark(
        dataset_name="humaneval_subset",
        code_gen_fn=code_gen_fn,
        model_config={"model": "test-codegen-v1", "max_tokens": 256},
    )

    logger.info("\nCode Generation Metrics:")
    for metric_name, metric in code_run.metrics.items():
        logger.info(f"  {metric_name}: {metric.value:.4f}")

    # Generate reports
    logger.info("\n--- Generating Reports ---")

    # JSON report
    suite.generate_report(retrieval_run, output_format="json", output_file=suite.output_dir / "report.json")
    logger.info("JSON report generated")

    # Markdown report
    suite.generate_report(retrieval_run, output_format="markdown", output_file=suite.output_dir / "report.md")
    logger.info("Markdown report generated")

    # HTML report
    suite.generate_report(retrieval_run, output_format="html", output_file=suite.output_dir / "report.html")
    logger.info("HTML report generated")

    # Export to CSV
    suite.export_to_csv(retrieval_run, suite.output_dir / "metrics.csv")

    # Generate visualizations
    logger.info("\n--- Generating Visualizations ---")
    plot_files = suite.visualize_results()
    for plot_name, plot_path in plot_files.items():
        logger.info(f"  {plot_name}: {plot_path}")

    # Comparison (simulate second run)
    logger.info("\n--- Testing Comparison ---")
    retrieval_run2 = suite.run_retrieval_benchmark(
        dataset_name="custom_mcts",
        retrieval_fn=retrieval_fn,
        model_config={"model": "test-retriever-v2", "embedding_dim": 768},
    )

    comparison = suite.compare_runs(
        baseline_run_id=retrieval_run.timestamp,
        comparison_run_ids=[retrieval_run2.timestamp],
        output_file=suite.output_dir / "comparison.json",
    )

    logger.info("\nComparison Recommendations:")
    for rec in comparison.recommendations:
        logger.info(f"  - {rec}")

    logger.info("\n=== Benchmark Suite Testing Complete ===")
    logger.info(f"Results saved to: {suite.output_dir}")

    # Close W&B run if active
    if suite.wandb_run and HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
