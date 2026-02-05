"""
Tests for benchmark CLI runner.

Validates argument parsing, configuration overrides,
dry run mode, and overall CLI flow.
"""

from __future__ import annotations

import argparse

from src.benchmark.cli import _dry_run, _print_summary, apply_cli_overrides, build_parser
from src.benchmark.config.benchmark_settings import BenchmarkSettings, reset_benchmark_settings
from src.benchmark.factory import BenchmarkFactory


class TestBuildParser:
    """Test argument parser construction."""

    def test_parser_creation(self) -> None:
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parse_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.systems is None
        assert args.tasks is None
        assert args.iterations is None
        assert args.output_dir is None
        assert not args.no_scoring
        assert not args.dry_run
        assert args.log_level == "INFO"
        assert args.save_results is True
        assert not args.no_report

    def test_parse_systems(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--systems", "langgraph_mcts", "vertex_adk"])
        assert args.systems == ["langgraph_mcts", "vertex_adk"]

    def test_parse_tasks(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--tasks", "A1", "A2", "B1"])
        assert args.tasks == ["A1", "A2", "B1"]

    def test_parse_iterations(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--iterations", "5"])
        assert args.iterations == 5

    def test_parse_output_dir(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--output-dir", "/tmp/results"])
        assert args.output_dir == "/tmp/results"

    def test_parse_no_scoring(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--no-scoring"])
        assert args.no_scoring is True

    def test_parse_dry_run(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_parse_log_level(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

    def test_parse_no_report(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--no-report"])
        assert args.no_report is True

    def test_parse_combined(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--systems",
                "langgraph_mcts",
                "--tasks",
                "A1",
                "--iterations",
                "3",
                "--no-scoring",
                "--log-level",
                "DEBUG",
            ]
        )
        assert args.systems == ["langgraph_mcts"]
        assert args.tasks == ["A1"]
        assert args.iterations == 3
        assert args.no_scoring is True
        assert args.log_level == "DEBUG"


class TestApplyCliOverrides:
    """Test CLI argument overrides on settings."""

    def setup_method(self) -> None:
        reset_benchmark_settings()
        self.settings = BenchmarkSettings()

    def test_override_iterations(self) -> None:
        args = argparse.Namespace(
            iterations=5,
            output_dir=None,
            no_scoring=False,
        )
        apply_cli_overrides(self.settings, args)
        assert self.settings.run.num_iterations == 5

    def test_override_output_dir(self) -> None:
        args = argparse.Namespace(
            iterations=None,
            output_dir="/tmp/benchmark_out",
            no_scoring=False,
        )
        apply_cli_overrides(self.settings, args)
        assert self.settings.report.output_dir == "/tmp/benchmark_out"

    def test_override_no_scoring(self) -> None:
        args = argparse.Namespace(
            iterations=None,
            output_dir=None,
            no_scoring=True,
        )
        apply_cli_overrides(self.settings, args)
        assert not self.settings.scoring.enabled

    def test_no_overrides(self) -> None:
        original_iterations = self.settings.run.num_iterations
        args = argparse.Namespace(
            iterations=None,
            output_dir=None,
            no_scoring=False,
        )
        apply_cli_overrides(self.settings, args)
        assert self.settings.run.num_iterations == original_iterations


class TestDryRun:
    """Test dry run output."""

    def setup_method(self) -> None:
        reset_benchmark_settings()
        self.settings = BenchmarkSettings()

    def test_dry_run_returns_zero(self, capsys) -> None:
        factory = BenchmarkFactory(settings=self.settings)
        args = argparse.Namespace(systems=None, tasks=None)
        result = _dry_run(factory, args)
        assert result == 0

        captured = capsys.readouterr()
        assert "Dry Run" in captured.out
        assert "Systems:" in captured.out
        assert "Tasks" in captured.out

    def test_dry_run_with_specific_systems(self, capsys) -> None:
        factory = BenchmarkFactory(settings=self.settings)
        args = argparse.Namespace(systems=["langgraph_mcts"], tasks=None)
        result = _dry_run(factory, args)
        assert result == 0

        captured = capsys.readouterr()
        assert "langgraph_mcts" in captured.out

    def test_dry_run_with_specific_tasks(self, capsys) -> None:
        factory = BenchmarkFactory(settings=self.settings)
        args = argparse.Namespace(systems=None, tasks=["A1", "A2"])
        result = _dry_run(factory, args)
        assert result == 0

        captured = capsys.readouterr()
        assert "A1" in captured.out
        assert "A2" in captured.out


class TestPrintSummary:
    """Test summary printing."""

    def test_print_summary(self, capsys) -> None:
        summary = {
            "run_id": "test123",
            "total_results": 4,
            "systems": {
                "sys_a": {
                    "total_tasks": 2,
                    "successful": 2,
                    "errors": 0,
                    "avg_latency_ms": 1500.0,
                    "avg_score": 4.2,
                    "total_cost_usd": 0.05,
                },
                "sys_b": {
                    "total_tasks": 2,
                    "successful": 1,
                    "errors": 1,
                    "avg_latency_ms": 800.0,
                    "avg_score": 3.5,
                    "total_cost_usd": 0.02,
                },
            },
        }

        _print_summary(summary)

        captured = capsys.readouterr()
        assert "test123" in captured.out
        assert "sys_a" in captured.out
        assert "sys_b" in captured.out
        assert "4.20" in captured.out
        assert "3.50" in captured.out
