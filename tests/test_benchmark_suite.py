"""
Unit Tests for Benchmark Suite

Tests core functionality of the benchmark suite including:
- Metric computation
- Statistical analysis
- Report generation
- Comparison logic
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

# Mock numpy if not available
try:
    import numpy as np
except ImportError:
    # Create mock numpy for testing
    class MockNumpy:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0.0

        @staticmethod
        def percentile(values, p):
            if not values:
                return 0.0
            sorted_vals = sorted(values)
            idx = int(len(sorted_vals) * p / 100)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]

        @staticmethod
        def var(values):
            if not values:
                return 0.0
            mean_val = sum(values) / len(values)
            return sum((x - mean_val) ** 2 for x in values) / len(values)

        @staticmethod
        def sqrt(x):
            return x**0.5

        @staticmethod
        def log2(x):
            import math

            return math.log2(x)

        @staticmethod
        def max(values):
            return max(values) if values else 0.0

        class random:
            @staticmethod
            def choice(values, size, _replace):
                import random

                return [random.choice(values) for _ in range(size)]

            @staticmethod
            def uniform(low, high):
                import random

                return random.uniform(low, high)

            @staticmethod
            def random():
                import random

                return random.random()

    np = MockNumpy()


from training.benchmark_suite import (
    BenchmarkSuite,
    CodeMetrics,
    ReasoningMetrics,
    RetrievalMetrics,
    RetrievalResult,
    StatisticalAnalysis,
)


class TestRetrievalMetrics(unittest.TestCase):
    """Test retrieval metrics computation."""

    def setUp(self):
        """Set up test data."""
        self.result = RetrievalResult(
            query="test query",
            retrieved_docs=["doc1", "doc2", "doc5", "doc8"],
            relevance_scores=[0.95, 0.87, 0.65, 0.45],
            ground_truth_relevant=["doc1", "doc2", "doc3"],
            ground_truth_rankings={"doc1": 0, "doc2": 1, "doc3": 2},
        )

    def test_recall_at_k(self):
        """Test Recall@k computation."""
        # At k=4, we retrieved doc1 and doc2 (2 out of 3 relevant)
        recall = RetrievalMetrics.recall_at_k(self.result, k=4)
        self.assertAlmostEqual(recall, 2 / 3, places=2)

        # At k=2, we retrieved doc1 and doc2 (2 out of 3 relevant)
        recall = RetrievalMetrics.recall_at_k(self.result, k=2)
        self.assertAlmostEqual(recall, 2 / 3, places=2)

        # At k=1, we retrieved only doc1 (1 out of 3 relevant)
        recall = RetrievalMetrics.recall_at_k(self.result, k=1)
        self.assertAlmostEqual(recall, 1 / 3, places=2)

    def test_precision_at_k(self):
        """Test Precision@k computation."""
        # At k=4, we have 2 relevant out of 4 retrieved
        precision = RetrievalMetrics.precision_at_k(self.result, k=4)
        self.assertAlmostEqual(precision, 2 / 4, places=2)

        # At k=2, we have 2 relevant out of 2 retrieved
        precision = RetrievalMetrics.precision_at_k(self.result, k=2)
        self.assertAlmostEqual(precision, 2 / 2, places=2)

    def test_mrr(self):
        """Test Mean Reciprocal Rank computation."""
        # First relevant document (doc1) is at position 0
        mrr = RetrievalMetrics.mean_reciprocal_rank(self.result)
        self.assertAlmostEqual(mrr, 1.0, places=2)

        # Test with first relevant at position 2
        result2 = RetrievalResult(
            query="test",
            retrieved_docs=["doc5", "doc8", "doc1", "doc2"],
            relevance_scores=[0.9, 0.8, 0.7, 0.6],
            ground_truth_relevant=["doc1", "doc2"],
        )
        mrr2 = RetrievalMetrics.mean_reciprocal_rank(result2)
        self.assertAlmostEqual(mrr2, 1 / 3, places=2)

    def test_ndcg_at_k(self):
        """Test nDCG@k computation."""
        ndcg = RetrievalMetrics.ndcg_at_k(self.result, k=10)
        # Should be > 0 and <= 1
        self.assertGreater(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)

    def test_map(self):
        """Test Mean Average Precision."""
        results = [self.result]
        map_score = RetrievalMetrics.mean_average_precision(results)
        self.assertGreater(map_score, 0.0)
        self.assertLessEqual(map_score, 1.0)


class TestReasoningMetrics(unittest.TestCase):
    """Test reasoning metrics computation."""

    def test_accuracy(self):
        """Test accuracy computation."""
        predictions = ["4", "5", "6"]
        ground_truths = ["4", "5", "7"]

        accuracy = ReasoningMetrics.accuracy(predictions, ground_truths)
        self.assertAlmostEqual(accuracy, 2 / 3, places=2)

    def test_accuracy_numeric(self):
        """Test accuracy with numeric values."""
        predictions = [4.0, 5.0, 6.0]
        ground_truths = [4.0, 5.0, 7.0]

        accuracy = ReasoningMetrics.accuracy(predictions, ground_truths)
        self.assertAlmostEqual(accuracy, 2 / 3, places=2)

    def test_reasoning_quality_score(self):
        """Test reasoning quality scoring."""
        predicted = ["Step 1: Add numbers", "Step 2: Get result"]
        expected = ["Add 2 and 2", "Result is 4"]

        quality = ReasoningMetrics.reasoning_quality_score(predicted, expected)
        self.assertGreater(quality, 0.0)
        self.assertLessEqual(quality, 1.0)

    def test_match_function(self):
        """Test the match helper function."""
        # String matching
        self.assertTrue(ReasoningMetrics._match("hello", "HELLO"))
        self.assertTrue(ReasoningMetrics._match("  test  ", "test"))
        self.assertFalse(ReasoningMetrics._match("hello", "world"))

        # Numeric matching
        self.assertTrue(ReasoningMetrics._match(4.0, 4.0))
        self.assertTrue(ReasoningMetrics._match(4.0000001, 4.0))
        self.assertFalse(ReasoningMetrics._match(4.0, 5.0))


class TestCodeMetrics(unittest.TestCase):
    """Test code generation metrics."""

    def test_syntax_correctness(self):
        """Test syntax correctness checking."""
        valid_code = [
            "def add(a, b):\n    return a + b",
            "x = 1 + 2",
            "print('hello')",
        ]

        invalid_code = ["def add(a, b)\n    return a + b", "x = 1 +", "print('hello]"]

        # All valid
        score = CodeMetrics.syntax_correctness(valid_code)
        self.assertEqual(score, 1.0)

        # All invalid
        score = CodeMetrics.syntax_correctness(invalid_code)
        self.assertEqual(score, 0.0)

        # Mix
        mixed = valid_code[:2] + invalid_code[:1]
        score = CodeMetrics.syntax_correctness(mixed)
        self.assertAlmostEqual(score, 2 / 3, places=2)

    def test_pass_at_k(self):
        """Test Pass@k computation."""
        results = [
            {"passed": True, "total_tests": 5},
            {"passed": False, "total_tests": 5},
            {"passed": True, "total_tests": 5},
        ]

        pass_at_1 = CodeMetrics.pass_at_k(results, k=1)
        self.assertAlmostEqual(pass_at_1, 2 / 3, places=2)

    def test_code_quality_score(self):
        """Test code quality scoring."""
        good_code = '''def add(a: int, b: int) -> int:
    """Add two numbers."""
    # Implementation
    return a + b
'''

        bad_code = "x=1+2"

        good_score = CodeMetrics.code_quality_score(good_code)
        bad_score = CodeMetrics.code_quality_score(bad_code)

        self.assertGreater(good_score, bad_score)
        self.assertGreater(good_score, 0.5)


class TestStatisticalAnalysis(unittest.TestCase):
    """Test statistical analysis functions."""

    def test_bootstrap_confidence_interval(self):
        """Test bootstrap CI computation."""
        values = [0.8, 0.85, 0.82, 0.88, 0.81, 0.86, 0.83, 0.87, 0.84, 0.85]

        lower, upper = StatisticalAnalysis.bootstrap_confidence_interval(values, confidence=0.95, n_bootstrap=100)

        # CI should contain the mean
        mean = sum(values) / len(values)
        self.assertLessEqual(lower, mean)
        self.assertGreaterEqual(upper, mean)

        # CI should be reasonable
        self.assertGreater(upper - lower, 0.0)
        self.assertLess(upper - lower, 0.5)

    def test_effect_size(self):
        """Test Cohen's d effect size computation."""
        baseline = [0.7, 0.72, 0.71, 0.73, 0.69]
        improved = [0.8, 0.82, 0.81, 0.83, 0.79]

        effect = StatisticalAnalysis.effect_size(baseline, improved)

        # Should show positive effect
        self.assertGreater(effect, 0.0)


class TestBenchmarkSuite(unittest.TestCase):
    """Test benchmark suite functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test benchmark suite initialization."""
        suite = BenchmarkSuite(output_dir=self.output_dir)
        self.assertTrue(self.output_dir.exists())
        self.assertEqual(len(suite.runs), 0)

    def test_retrieval_benchmark(self):
        """Test retrieval benchmark execution."""
        suite = BenchmarkSuite(output_dir=self.output_dir)

        def mock_retrieval(query: str):
            return {"doc_ids": ["doc1", "doc2", "doc3"], "scores": [0.9, 0.8, 0.7]}

        run = suite.run_retrieval_benchmark(
            dataset_name="test_dataset", retrieval_fn=mock_retrieval, model_config={"model": "test"}
        )

        # Check run was recorded
        self.assertEqual(len(suite.runs), 1)
        self.assertEqual(run.benchmark_name, "rag_retrieval")
        self.assertEqual(run.dataset_name, "test_dataset")

        # Check metrics exist
        self.assertIn("nDCG@10", run.metrics)
        self.assertIn("Recall@100", run.metrics)
        self.assertIn("MRR", run.metrics)

    def test_report_generation_json(self):
        """Test JSON report generation."""
        suite = BenchmarkSuite(output_dir=self.output_dir)

        def mock_retrieval(query: str):
            return {"doc_ids": ["doc1"], "scores": [0.9]}

        run = suite.run_retrieval_benchmark(dataset_name="test", retrieval_fn=mock_retrieval, model_config={})

        # Generate JSON report
        output_file = self.output_dir / "report.json"
        report = suite.generate_report(run, output_format="json", output_file=output_file)

        # Check file created
        self.assertTrue(output_file.exists())

        # Check valid JSON
        report_dict = json.loads(report)
        self.assertIn("benchmark_name", report_dict)
        self.assertIn("metrics", report_dict)

    def test_report_generation_markdown(self):
        """Test Markdown report generation."""
        suite = BenchmarkSuite(output_dir=self.output_dir)

        def mock_retrieval(query: str):
            return {"doc_ids": ["doc1"], "scores": [0.9]}

        run = suite.run_retrieval_benchmark(dataset_name="test", retrieval_fn=mock_retrieval, model_config={})

        # Generate Markdown report
        output_file = self.output_dir / "report.md"
        report = suite.generate_report(run, output_format="markdown", output_file=output_file)

        # Check file created
        self.assertTrue(output_file.exists())

        # Check Markdown formatting
        self.assertIn("# Benchmark Report", report)
        self.assertIn("## Metrics", report)

    def test_csv_export(self):
        """Test CSV export functionality."""
        suite = BenchmarkSuite(output_dir=self.output_dir)

        def mock_retrieval(query: str):
            return {"doc_ids": ["doc1"], "scores": [0.9]}

        run = suite.run_retrieval_benchmark(dataset_name="test", retrieval_fn=mock_retrieval, model_config={})

        # Export to CSV
        output_file = self.output_dir / "metrics.csv"
        suite.export_to_csv(run, output_file)

        # Check file created
        self.assertTrue(output_file.exists())

        # Check content
        with open(output_file) as f:
            content = f.read()
            self.assertIn("Metric,Value", content)

    def test_comparison(self):
        """Test run comparison functionality."""
        suite = BenchmarkSuite(output_dir=self.output_dir)

        def mock_retrieval(query: str):
            return {"doc_ids": ["doc1", "doc2"], "scores": [0.9, 0.8]}

        # Run baseline
        run1 = suite.run_retrieval_benchmark(dataset_name="test", retrieval_fn=mock_retrieval, model_config={"v": 1})

        # Run comparison
        run2 = suite.run_retrieval_benchmark(dataset_name="test", retrieval_fn=mock_retrieval, model_config={"v": 2})

        # Compare
        comparison = suite.compare_runs(
            baseline_run_id=run1.timestamp,
            comparison_run_ids=[run2.timestamp],
            output_file=self.output_dir / "comparison.json",
        )

        # Check comparison
        self.assertEqual(comparison.baseline_run, run1.timestamp)
        self.assertIn(run2.timestamp, comparison.comparison_runs)
        self.assertTrue(len(comparison.recommendations) > 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRetrievalMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestReasoningMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestCodeMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestBenchmarkSuite))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import sys

    sys.exit(run_tests())
