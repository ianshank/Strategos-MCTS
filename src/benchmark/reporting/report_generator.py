"""
Markdown report generator for benchmark comparisons.

Produces structured comparison reports with summary tables,
per-task analysis, and key findings.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from src.benchmark.config.benchmark_settings import BenchmarkSettings, ReportConfig, get_benchmark_settings
from src.benchmark.evaluation.models import BenchmarkResult
from src.benchmark.reporting.metrics_aggregator import MetricsAggregator


class ReportGenerator:
    """
    Generates markdown benchmark comparison reports.

    Example:
        >>> generator = ReportGenerator(config=report_config)
        >>> report = generator.generate(results)
        >>> generator.save(report)
    """

    def __init__(
        self,
        config: ReportConfig | None = None,
        settings: BenchmarkSettings | None = None,
        aggregator: MetricsAggregator | None = None,
    ) -> None:
        """
        Initialize report generator.

        Args:
            config: Report configuration
            settings: Full benchmark settings (for report metadata)
            aggregator: Metrics aggregator instance
        """
        self._settings = settings or get_benchmark_settings()
        self._config = config or self._settings.report
        self._aggregator = aggregator or MetricsAggregator()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def generate(
        self,
        results: list[BenchmarkResult],
        title: str | None = None,
    ) -> str:
        """
        Generate a complete markdown comparison report.

        Args:
            results: All benchmark results
            title: Report title (defaults to benchmark_name from settings)

        Returns:
            Markdown report as string
        """
        report_title = title or f"{self._settings.benchmark_name} Benchmark Report"
        sections = [
            self._generate_header(report_title, results),
            self._generate_summary_table(results),
            self._generate_per_task_analysis(results),
            self._generate_scoring_comparison(results),
            self._generate_cost_analysis(results),
            self._generate_key_findings(results),
        ]

        report = "\n\n".join(sections)
        self._logger.info("Generated benchmark report (%d chars)", len(report))
        return report

    def save(self, report: str, output_dir: str | Path | None = None) -> Path:
        """
        Save report to file.

        Args:
            report: Markdown report content
            output_dir: Output directory

        Returns:
            Path to saved report file
        """
        out_dir = Path(output_dir or self._config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        report_path = out_dir / self._config.report_filename
        report_path.write_text(report)

        self._logger.info("Report saved to %s", report_path)
        return report_path

    def _generate_header(self, title: str, results: list[BenchmarkResult]) -> str:
        """Generate report header with metadata."""
        systems = sorted({r.system for r in results})
        tasks = sorted({r.task_id for r in results})

        lines = [
            f"# {title}",
            "",
            f"**Date:** {time.strftime('%Y-%m-%d')}",
            f"**Systems Compared:** {', '.join(systems)}",
            f"**Tasks Evaluated:** {len(tasks)}",
            f"**Total Results:** {len(results)}",
        ]

        return "\n".join(lines)

    def _generate_summary_table(self, results: list[BenchmarkResult]) -> str:
        """Generate summary comparison table."""
        systems = sorted({r.system for r in results})
        if len(systems) < 2:
            return "## Summary\n\n*Insufficient systems for comparison.*"

        comparisons = self._aggregator.compare_systems(results, systems[0], systems[1])

        lines = [
            "## Summary Comparison",
            "",
            f"| Metric | {systems[0]} | {systems[1]} | Winner |",
            "|--------|" + "|".join(["-------"] * 3) + "|",
        ]

        for comp in comparisons:
            lines.append(
                f"| {comp.metric_name} | {comp.system_a.mean:.2f} | {comp.system_b.mean:.2f} | {comp.winner} |"
            )

        return "\n".join(lines)

    def _generate_per_task_analysis(self, results: list[BenchmarkResult]) -> str:
        """Generate per-task analysis section."""
        by_task = self._aggregator.aggregate_by_task(results)

        lines = ["## Per-Task Analysis", ""]

        for task_id in sorted(by_task.keys()):
            task_systems = by_task[task_id]
            task_desc = ""
            for r in results:
                if r.task_id == task_id:
                    task_desc = r.task_description
                    break

            lines.append(f"### Task {task_id}: {task_desc}")
            lines.append("")

            system_names = sorted(task_systems.keys())
            header = "| Metric | " + " | ".join(system_names) + " |"
            separator = "|--------|" + "|".join(["-------"] * len(system_names)) + "|"
            lines.append(header)
            lines.append(separator)

            # Key metrics to show per task
            metric_keys = ["average_score", "latency_ms", "total_tokens", "cost_usd"]
            for metric in metric_keys:
                row = f"| {metric} |"
                for system in system_names:
                    metrics = task_systems.get(system, {})
                    stats = metrics.get(metric)
                    if stats and stats.count > 0:
                        row += f" {stats.mean:.2f} |"
                    else:
                        row += " N/A |"
                lines.append(row)

            lines.append("")

        return "\n".join(lines)

    def _generate_scoring_comparison(self, results: list[BenchmarkResult]) -> str:
        """Generate detailed scoring comparison."""
        by_system = self._aggregator.aggregate_by_system(results)
        systems = sorted(by_system.keys())

        lines = ["## Scoring Breakdown", ""]

        scoring_metrics = [
            "task_completion",
            "reasoning_depth",
            "accuracy",
            "coherence",
            "average_score",
        ]

        header = "| Dimension | " + " | ".join(systems) + " |"
        separator = "|-----------|" + "|".join(["-------"] * len(systems)) + "|"
        lines.append(header)
        lines.append(separator)

        for metric in scoring_metrics:
            row = f"| {metric} |"
            for system in systems:
                metrics = by_system.get(system, {})
                stats = metrics.get(metric)
                if stats and stats.count > 0:
                    row += f" {stats.mean:.2f} (+/-{stats.std_dev:.2f}) |"
                else:
                    row += " N/A |"
            lines.append(row)

        return "\n".join(lines)

    def _generate_cost_analysis(self, results: list[BenchmarkResult]) -> str:
        """Generate cost analysis section."""
        by_system: dict[str, list[BenchmarkResult]] = {}
        for r in results:
            by_system.setdefault(r.system, []).append(r)

        lines = ["## Cost Analysis", ""]

        for system in sorted(by_system.keys()):
            sys_results = by_system[system]
            total_cost = sum(r.estimated_cost_usd for r in sys_results)
            total_tokens = sum(r.total_tokens for r in sys_results)
            avg_cost = total_cost / len(sys_results) if sys_results else 0

            lines.append(f"### {system}")
            lines.append(f"- Total cost: ${total_cost:.4f}")
            lines.append(f"- Average cost per task: ${avg_cost:.4f}")
            lines.append(f"- Total tokens: {total_tokens:,}")
            lines.append("")

        return "\n".join(lines)

    def _generate_key_findings(self, results: list[BenchmarkResult]) -> str:
        """Generate key findings summary."""
        systems = sorted({r.system for r in results})

        lines = ["## Key Findings", ""]

        if len(systems) >= 2:
            comparisons = self._aggregator.compare_systems(results, systems[0], systems[1])

            # Find which system wins more metrics
            wins: dict[str, int] = dict.fromkeys(systems, 0)
            for comp in comparisons:
                if comp.winner in wins:
                    wins[comp.winner] += 1

            for system, count in sorted(wins.items(), key=lambda x: -x[1]):
                lines.append(f"- **{system}** wins on {count}/{len(comparisons)} metrics")

            # Highlight specific areas
            for comp in comparisons:
                if comp.metric_name == "average_score" and comp.winner != "tie":
                    diff = abs(comp.system_a.mean - comp.system_b.mean)
                    lines.append(f"- Quality advantage: **{comp.winner}** leads by {diff:.2f} points on average score")
                elif comp.metric_name == "latency_ms" and comp.winner != "tie":
                    faster = comp.winner
                    lines.append(f"- Speed advantage: **{faster}** has lower average latency")

        else:
            lines.append("*Single system evaluated. No comparison available.*")

        return "\n".join(lines)
