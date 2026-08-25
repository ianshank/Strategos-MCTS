"""
Embedding Model Benchmarking Tool

Compares different embedding models on RAG evaluation datasets.
Measures: retrieval precision, recall, nDCG@10, latency, cost.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from training.advanced_embeddings import BaseEmbedder, EmbedderFactory

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from embedding benchmark."""

    model_name: str
    dimension: int
    precision_at_k: dict[int, float]  # k -> precision
    recall_at_k: dict[int, float]  # k -> recall
    ndcg_at_k: dict[int, float]  # k -> nDCG
    mrr: float
    avg_latency_ms: float
    total_time_s: float
    cache_hit_rate: float
    num_queries: int
    embeddings_generated: int
    metadata: dict[str, Any] = field(default_factory=dict)


class EmbeddingBenchmark:
    """Benchmark embedding models on retrieval tasks."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize benchmark.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.k_values = config.get("k_values", [1, 5, 10, 20])
        self.test_queries = []
        self.corpus = []
        self.ground_truth = {}

    def load_evaluation_dataset(self, dataset_path: str | Path) -> None:
        """
        Load evaluation dataset.

        Expected format:
        {
            "queries": [{"id": "q1", "text": "query text"}],
            "corpus": [{"id": "d1", "text": "document text"}],
            "ground_truth": {"q1": ["d1", "d2"]}  # relevant docs for each query
        }

        Args:
            dataset_path: Path to evaluation dataset JSON
        """
        dataset_path = Path(dataset_path)
        with open(dataset_path) as f:
            data = json.load(f)

        self.test_queries = data["queries"]
        self.corpus = data["corpus"]
        self.ground_truth = data["ground_truth"]

        logger.info(f"Loaded evaluation dataset: {len(self.test_queries)} queries, {len(self.corpus)} documents")

    def create_synthetic_dataset(
        self, num_queries: int = 50, num_docs: int = 1000, relevant_docs_per_query: int = 5
    ) -> None:
        """
        Create synthetic evaluation dataset.

        Args:
            num_queries: Number of test queries
            num_docs: Number of documents in corpus
            relevant_docs_per_query: Number of relevant docs per query
        """
        # Synthetic cybersecurity queries and documents
        query_templates = [
            "How to detect {threat} in network traffic?",
            "What are the indicators of {attack} attacks?",
            "Best practices for preventing {vulnerability}",
            "Mitigating {technique} in enterprise environments",
            "Understanding {framework} framework for threat analysis",
        ]

        doc_templates = [
            "This document discusses {topic} detection methods and best practices.",
            "Analysis of {topic} techniques used by threat actors.",
            "Security controls for preventing {topic} in corporate networks.",
            "{topic} indicators and response procedures.",
            "Comprehensive guide to {topic} mitigation strategies.",
        ]

        topics = [
            "malware",
            "phishing",
            "ransomware",
            "SQL injection",
            "XSS",
            "DDoS",
            "APT",
            "privilege escalation",
            "lateral movement",
            "data exfiltration",
            "zero-day exploits",
            "MITRE ATT&CK",
            "threat intelligence",
            "intrusion detection",
            "vulnerability management",
        ]

        import random

        random.seed(42)
        np.random.seed(42)

        # Generate queries
        for i in range(num_queries):
            template = random.choice(query_templates)
            topic = random.choice(topics)
            query_text = template.format(
                threat=topic, attack=topic, vulnerability=topic, technique=topic, framework=topic
            )
            self.test_queries.append({"id": f"q{i}", "text": query_text})

        # Generate corpus
        for i in range(num_docs):
            template = random.choice(doc_templates)
            topic = random.choice(topics)
            doc_text = template.format(topic=topic)
            self.corpus.append({"id": f"d{i}", "text": doc_text})

        # Generate ground truth (random relevant docs)
        for query in self.test_queries:
            relevant_indices = np.random.choice(num_docs, size=relevant_docs_per_query, replace=False)
            self.ground_truth[query["id"]] = [f"d{idx}" for idx in relevant_indices]

        logger.info(f"Created synthetic dataset: {num_queries} queries, {num_docs} documents")

    def benchmark_embedder(self, embedder: BaseEmbedder) -> BenchmarkResult:
        """
        Benchmark a single embedder.

        Args:
            embedder: Embedder to benchmark

        Returns:
            BenchmarkResult
        """
        if not self.test_queries or not self.corpus:
            raise ValueError(
                "No evaluation dataset loaded. Call load_evaluation_dataset or create_synthetic_dataset first."
            )

        logger.info(f"Benchmarking embedder: {embedder.model_name}")

        # Embed corpus
        corpus_start = time.time()
        corpus_texts = [doc["text"] for doc in self.corpus]
        corpus_result = embedder.embed_with_cache(corpus_texts)
        corpus_embeddings = corpus_result.embeddings
        corpus_time = time.time() - corpus_start

        # Embed queries and retrieve
        query_latencies = []
        all_precisions = {k: [] for k in self.k_values}
        all_recalls = {k: [] for k in self.k_values}
        all_ndcgs = {k: [] for k in self.k_values}
        all_mrrs = []
        total_cache_hits = 0
        total_embeddings = 0

        query_start = time.time()
        for query in self.test_queries:
            q_start = time.time()

            # Embed query
            query_result = embedder.embed_with_cache([query["text"]])
            query_embedding = query_result.embeddings[0]

            # Retrieve similar documents
            similarities = np.dot(corpus_embeddings, query_embedding)
            ranked_indices = np.argsort(similarities)[::-1]
            ranked_doc_ids = [self.corpus[idx]["id"] for idx in ranked_indices]

            # Get ground truth
            relevant_docs = set(self.ground_truth.get(query["id"], []))

            # Calculate metrics at different k values
            for k in self.k_values:
                retrieved_k = ranked_doc_ids[:k]
                relevant_retrieved = len(set(retrieved_k) & relevant_docs)

                # Precision@k
                precision = relevant_retrieved / k if k > 0 else 0
                all_precisions[k].append(precision)

                # Recall@k
                recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0
                all_recalls[k].append(recall)

                # nDCG@k
                dcg = 0.0
                idcg = 0.0
                for i, doc_id in enumerate(retrieved_k):
                    if doc_id in relevant_docs:
                        dcg += 1.0 / np.log2(i + 2)
                for i in range(min(k, len(relevant_docs))):
                    idcg += 1.0 / np.log2(i + 2)
                ndcg = dcg / idcg if idcg > 0 else 0
                all_ndcgs[k].append(ndcg)

            # MRR
            mrr = 0.0
            for i, doc_id in enumerate(ranked_doc_ids):
                if doc_id in relevant_docs:
                    mrr = 1.0 / (i + 1)
                    break
            all_mrrs.append(mrr)

            q_latency = (time.time() - q_start) * 1000
            query_latencies.append(q_latency)

            # Track cache stats
            total_cache_hits += query_result.metadata.get("cached_count", 0)
            total_embeddings += 1

        query_time = time.time() - query_start
        total_time = corpus_time + query_time

        # Aggregate metrics
        precision_at_k = {k: np.mean(all_precisions[k]) for k in self.k_values}
        recall_at_k = {k: np.mean(all_recalls[k]) for k in self.k_values}
        ndcg_at_k = {k: np.mean(all_ndcgs[k]) for k in self.k_values}
        mrr = np.mean(all_mrrs)
        avg_latency = np.mean(query_latencies)
        cache_hit_rate = total_cache_hits / total_embeddings if total_embeddings > 0 else 0

        result = BenchmarkResult(
            model_name=embedder.model_name,
            dimension=embedder.dimension,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            mrr=mrr,
            avg_latency_ms=avg_latency,
            total_time_s=total_time,
            cache_hit_rate=cache_hit_rate,
            num_queries=len(self.test_queries),
            embeddings_generated=len(self.corpus) + len(self.test_queries),
            metadata={
                "corpus_time_s": corpus_time,
                "query_time_s": query_time,
                "corpus_cache_hit_rate": corpus_result.metadata.get("cache_hit_rate", 0),
            },
        )

        logger.info(f"Benchmark complete for {embedder.model_name}")
        logger.info(f"  Precision@10: {precision_at_k.get(10, 0):.4f}")
        logger.info(f"  Recall@10: {recall_at_k.get(10, 0):.4f}")
        logger.info(f"  nDCG@10: {ndcg_at_k.get(10, 0):.4f}")
        logger.info(f"  MRR: {mrr:.4f}")
        logger.info(f"  Avg latency: {avg_latency:.2f}ms")

        return result

    def compare_embedders(self, embedder_configs: list[dict[str, Any]]) -> list[BenchmarkResult]:
        """
        Compare multiple embedders.

        Args:
            embedder_configs: List of embedder configurations

        Returns:
            List of BenchmarkResults
        """
        results = []

        for config in embedder_configs:
            try:
                embedder = EmbedderFactory.create_embedder(config)
                if embedder.is_available():
                    result = self.benchmark_embedder(embedder)
                    results.append(result)
                else:
                    logger.warning(f"Embedder {config.get('model')} not available, skipping...")
            except Exception as e:
                logger.error(f"Failed to benchmark {config.get('model')}: {e}")

        return results

    def generate_report(self, results: list[BenchmarkResult], output_path: str | Path) -> None:
        """
        Generate comparison report.

        Args:
            results: List of BenchmarkResults
            output_path: Path to save report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create report
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_queries": results[0].num_queries if results else 0,
            "num_corpus_docs": len(self.corpus),
            "k_values": self.k_values,
            "results": [],
        }

        for result in results:
            report["results"].append(
                {
                    "model_name": result.model_name,
                    "dimension": result.dimension,
                    "precision_at_k": result.precision_at_k,
                    "recall_at_k": result.recall_at_k,
                    "ndcg_at_k": result.ndcg_at_k,
                    "mrr": result.mrr,
                    "avg_latency_ms": result.avg_latency_ms,
                    "total_time_s": result.total_time_s,
                    "cache_hit_rate": result.cache_hit_rate,
                    "metadata": result.metadata,
                }
            )

        # Save JSON report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {output_path}")

        # Print summary
        self._print_summary(results)

    def _print_summary(self, results: list[BenchmarkResult]) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("EMBEDDING MODEL BENCHMARK SUMMARY")
        print("=" * 80)

        # Sort by nDCG@10
        results_sorted = sorted(results, key=lambda r: r.ndcg_at_k.get(10, 0), reverse=True)

        print(f"\n{'Model':<40} {'Dim':<6} {'P@10':<8} {'R@10':<8} {'nDCG@10':<9} {'MRR':<8} {'Latency':<10}")
        print("-" * 80)

        for result in results_sorted:
            print(
                f"{result.model_name:<40} "
                f"{result.dimension:<6} "
                f"{result.precision_at_k.get(10, 0):<8.4f} "
                f"{result.recall_at_k.get(10, 0):<8.4f} "
                f"{result.ndcg_at_k.get(10, 0):<9.4f} "
                f"{result.mrr:<8.4f} "
                f"{result.avg_latency_ms:<10.2f}ms"
            )

        print("-" * 80)

        # Calculate improvements
        if len(results_sorted) > 1:
            best = results_sorted[0]
            baseline = results_sorted[-1]

            improvements = {
                "precision@10": (
                    (best.precision_at_k.get(10, 0) - baseline.precision_at_k.get(10, 0))
                    / baseline.precision_at_k.get(10, 0.0001)
                    * 100
                ),
                "recall@10": (
                    (best.recall_at_k.get(10, 0) - baseline.recall_at_k.get(10, 0))
                    / baseline.recall_at_k.get(10, 0.0001)
                    * 100
                ),
                "ndcg@10": (
                    (best.ndcg_at_k.get(10, 0) - baseline.ndcg_at_k.get(10, 0))
                    / baseline.ndcg_at_k.get(10, 0.0001)
                    * 100
                ),
                "mrr": ((best.mrr - baseline.mrr) / baseline.mrr * 100) if baseline.mrr > 0 else 0,
            }

            print(f"\nBest Model: {best.model_name}")
            print(f"Baseline: {baseline.model_name}")
            print("\nImprovements:")
            for metric, improvement in improvements.items():
                print(f"  {metric}: {improvement:+.2f}%")

        print("=" * 80 + "\n")


def main():
    """Run benchmark from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark embedding models")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, help="Path to evaluation dataset JSON")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset")
    parser.add_argument("--num-queries", type=int, default=50, help="Number of synthetic queries")
    parser.add_argument("--num-docs", type=int, default=1000, help="Number of synthetic documents")
    parser.add_argument("--output", type=str, default="./reports/embedding_benchmark.json", help="Output report path")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Initialize benchmark
    benchmark_config = {"k_values": [1, 5, 10, 20]}
    benchmark = EmbeddingBenchmark(benchmark_config)

    # Load or create dataset
    if args.dataset:
        benchmark.load_evaluation_dataset(args.dataset)
    elif args.synthetic:
        benchmark.create_synthetic_dataset(num_queries=args.num_queries, num_docs=args.num_docs)
    else:
        # Default: create small synthetic dataset
        benchmark.create_synthetic_dataset(num_queries=20, num_docs=200)

    # Create embedder configs
    embeddings_config = config["rag"]["embeddings"]
    embedder_configs = []

    # Primary model
    primary_config = {
        "model": embeddings_config["model"],
        "provider": embeddings_config.get("provider"),
        "dimension": embeddings_config["dimension"],
        "batch_size": embeddings_config["batch_size"],
        "cache_dir": embeddings_config["cache_dir"],
        "cache_enabled": embeddings_config["cache_enabled"],
    }
    embedder_configs.append(primary_config)

    # Fallback models
    for fallback in embeddings_config.get("fallback_models", []):
        fallback_config = {
            "model": fallback["model"],
            "provider": fallback.get("provider"),
            "dimension": fallback["dimension"],
            "batch_size": embeddings_config["batch_size"],
            "cache_dir": embeddings_config["cache_dir"],
            "cache_enabled": embeddings_config["cache_enabled"],
        }
        embedder_configs.append(fallback_config)

    # Run benchmark
    results = benchmark.compare_embedders(embedder_configs)

    # Generate report
    if results:
        benchmark.generate_report(results, args.output)
    else:
        logger.error("No embedders were successfully benchmarked")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
