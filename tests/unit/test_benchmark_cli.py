"""Unit tests for src/benchmark/cli.py."""

from __future__ import annotations

import argparse
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.benchmark.cli import (
    _dry_run,
    _print_summary,
    apply_cli_overrides,
    build_parser,
    configure_logging,
    main,
    run_benchmark,
)


@pytest.mark.unit
class TestBuildParser:
    """Tests for CLI argument parser construction."""

    def test_parser_returns_argparse(self):
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_default_args(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.systems is None
        assert args.tasks is None
        assert args.iterations is None
        assert args.output_dir is None
        assert args.no_scoring is False
        assert args.dry_run is False
        assert args.log_level == "INFO"
        assert args.no_save_results is False
        assert args.no_report is False

    def test_systems_arg(self):
        parser = build_parser()
        args = parser.parse_args(["--systems", "langgraph_mcts", "vertex_adk"])
        assert args.systems == ["langgraph_mcts", "vertex_adk"]

    def test_tasks_arg(self):
        parser = build_parser()
        args = parser.parse_args(["--tasks", "A1", "A2", "B1"])
        assert args.tasks == ["A1", "A2", "B1"]

    def test_iterations_arg(self):
        parser = build_parser()
        args = parser.parse_args(["--iterations", "5"])
        assert args.iterations == 5

    def test_output_dir_arg(self):
        parser = build_parser()
        args = parser.parse_args(["--output-dir", "/tmp/results"])
        assert args.output_dir == "/tmp/results"

    def test_no_scoring_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--no-scoring"])
        assert args.no_scoring is True

    def test_dry_run_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_log_level_choices(self):
        parser = build_parser()
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            args = parser.parse_args(["--log-level", level])
            assert args.log_level == level

    def test_no_save_results_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--no-save-results"])
        assert args.no_save_results is True

    def test_no_report_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--no-report"])
        assert args.no_report is True

    def test_combined_args(self):
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
                "--dry-run",
                "--log-level",
                "DEBUG",
            ]
        )
        assert args.systems == ["langgraph_mcts"]
        assert args.tasks == ["A1"]
        assert args.iterations == 3
        assert args.no_scoring is True
        assert args.dry_run is True
        assert args.log_level == "DEBUG"


@pytest.mark.unit
class TestConfigureLogging:
    """Tests for logging configuration."""

    @patch("src.benchmark.cli.logging.basicConfig")
    def test_configure_logging_info(self, mock_basic_config):
        configure_logging("INFO")
        mock_basic_config.assert_called_once()
        call_kwargs = mock_basic_config.call_args
        assert call_kwargs[1]["level"] == 20  # logging.INFO

    @patch("src.benchmark.cli.logging.basicConfig")
    def test_configure_logging_debug(self, mock_basic_config):
        configure_logging("DEBUG")
        mock_basic_config.assert_called_once()
        call_kwargs = mock_basic_config.call_args
        assert call_kwargs[1]["level"] == 10  # logging.DEBUG


@pytest.mark.unit
class TestApplyCliOverrides:
    """Tests for CLI overrides applied to settings."""

    def test_iterations_override(self):
        settings = MagicMock()
        args = argparse.Namespace(
            iterations=5,
            output_dir=None,
            no_scoring=False,
        )
        apply_cli_overrides(settings, args)
        assert settings._run is not None

    def test_output_dir_override(self):
        settings = MagicMock()
        args = argparse.Namespace(
            iterations=None,
            output_dir="/tmp/out",
            no_scoring=False,
        )
        apply_cli_overrides(settings, args)
        assert settings._report is not None

    def test_no_scoring_override(self):
        settings = MagicMock()
        args = argparse.Namespace(
            iterations=None,
            output_dir=None,
            no_scoring=True,
        )
        apply_cli_overrides(settings, args)
        assert settings._scoring is not None

    def test_no_overrides(self):
        settings = MagicMock()
        original_run = settings._run
        original_report = settings._report
        original_scoring = settings._scoring
        args = argparse.Namespace(
            iterations=None,
            output_dir=None,
            no_scoring=False,
        )
        apply_cli_overrides(settings, args)
        # Nothing should be reset when no overrides
        # _run, _report, _scoring should stay as original mock values
        assert settings._run == original_run
        assert settings._report == original_report
        assert settings._scoring == original_scoring


@pytest.mark.unit
class TestRunBenchmark:
    """Tests for the async run_benchmark function."""

    @patch("src.benchmark.cli.reset_benchmark_settings")
    @patch("src.benchmark.cli.get_benchmark_settings")
    @patch("src.benchmark.cli.set_correlation_id")
    @patch("src.benchmark.cli.BenchmarkFactory")
    def test_benchmark_disabled_returns_1(self, mock_factory_cls, mock_set_cid, mock_get_settings, mock_reset):
        settings = MagicMock()
        settings.benchmark_enabled = False
        mock_get_settings.return_value = settings

        args = argparse.Namespace(
            iterations=None,
            output_dir=None,
            no_scoring=False,
            dry_run=False,
            systems=None,
            tasks=None,
            no_save_results=False,
            no_report=False,
        )
        result = asyncio.run(run_benchmark(args))
        assert result == 1

    @patch("src.benchmark.cli.reset_benchmark_settings")
    @patch("src.benchmark.cli.get_benchmark_settings")
    @patch("src.benchmark.cli.set_correlation_id")
    @patch("src.benchmark.cli.BenchmarkFactory")
    def test_dry_run_returns_0(self, mock_factory_cls, mock_set_cid, mock_get_settings, mock_reset):
        settings = MagicMock()
        settings.benchmark_enabled = True
        mock_get_settings.return_value = settings

        factory_instance = MagicMock()
        mock_factory_cls.return_value = factory_instance

        # Mock dry run dependencies
        adapter_factory = MagicMock()
        adapter_factory.get_available_systems.return_value = ["langgraph_mcts"]
        factory_instance.create_adapter_factory.return_value = adapter_factory

        registry = MagicMock()
        task = MagicMock()
        task.task_id = "A1"
        task.description = "Test task"
        task.category.value = "reasoning"
        registry.get_all.return_value = [task]
        registry.get.return_value = task
        factory_instance.create_task_registry.return_value = registry

        settings.run.num_iterations = 1
        settings.scoring.enabled = True
        settings.report.output_dir = "/tmp"
        factory_instance.settings = settings

        args = argparse.Namespace(
            iterations=None,
            output_dir=None,
            no_scoring=False,
            dry_run=True,
            systems=None,
            tasks=None,
            no_save_results=False,
            no_report=False,
        )
        result = asyncio.run(run_benchmark(args))
        assert result == 0

    @patch("src.benchmark.cli.reset_benchmark_settings")
    @patch("src.benchmark.cli.get_benchmark_settings")
    @patch("src.benchmark.cli.set_correlation_id")
    @patch("src.benchmark.cli.BenchmarkFactory")
    def test_harness_creation_failure_returns_1(self, mock_factory_cls, mock_set_cid, mock_get_settings, mock_reset):
        settings = MagicMock()
        settings.benchmark_enabled = True
        mock_get_settings.return_value = settings

        factory_instance = MagicMock()
        factory_instance.create_harness.side_effect = RuntimeError("boom")
        mock_factory_cls.return_value = factory_instance

        args = argparse.Namespace(
            iterations=None,
            output_dir=None,
            no_scoring=False,
            dry_run=False,
            systems=["langgraph_mcts"],
            tasks=None,
            no_save_results=False,
            no_report=False,
        )
        result = asyncio.run(run_benchmark(args))
        assert result == 1

    @patch("src.benchmark.cli.reset_benchmark_settings")
    @patch("src.benchmark.cli.get_benchmark_settings")
    @patch("src.benchmark.cli.set_correlation_id")
    @patch("src.benchmark.cli.BenchmarkFactory")
    def test_no_results_returns_2(self, mock_factory_cls, mock_set_cid, mock_get_settings, mock_reset):
        settings = MagicMock()
        settings.benchmark_enabled = True
        mock_get_settings.return_value = settings

        factory_instance = MagicMock()
        mock_factory_cls.return_value = factory_instance

        harness = MagicMock()
        harness.run = AsyncMock(return_value=[])
        factory_instance.create_harness.return_value = harness

        args = argparse.Namespace(
            iterations=None,
            output_dir=None,
            no_scoring=False,
            dry_run=False,
            systems=None,
            tasks=None,
            no_save_results=False,
            no_report=False,
        )
        result = asyncio.run(run_benchmark(args))
        assert result == 2

    @patch("src.benchmark.cli.reset_benchmark_settings")
    @patch("src.benchmark.cli.get_benchmark_settings")
    @patch("src.benchmark.cli.set_correlation_id")
    @patch("src.benchmark.cli.BenchmarkFactory")
    def test_successful_run_returns_0(self, mock_factory_cls, mock_set_cid, mock_get_settings, mock_reset):
        settings = MagicMock()
        settings.benchmark_enabled = True
        settings.benchmark_name = "test"
        settings.report.output_dir = "/tmp/out"
        mock_get_settings.return_value = settings

        factory_instance = MagicMock()
        mock_factory_cls.return_value = factory_instance

        harness = MagicMock()
        harness.run = AsyncMock(return_value=[{"result": "ok"}])
        harness.summary.return_value = {
            "run_id": "test-123",
            "total_results": 1,
            "systems": {
                "langgraph_mcts": {
                    "successful": 1,
                    "total_tasks": 1,
                    "avg_latency_ms": 100.0,
                    "avg_score": 0.9,
                    "total_cost_usd": 0.001,
                },
            },
        }
        harness.save_results.return_value = "/tmp/out/results.json"
        factory_instance.create_harness.return_value = harness

        report_gen = MagicMock()
        report_gen.generate.return_value = "# Report"
        report_gen.save.return_value = "/tmp/out/report.md"
        factory_instance.create_report_generator.return_value = report_gen

        args = argparse.Namespace(
            iterations=None,
            output_dir=None,
            no_scoring=False,
            dry_run=False,
            systems=None,
            tasks=None,
            no_save_results=False,
            no_report=False,
        )
        result = asyncio.run(run_benchmark(args))
        assert result == 0
        harness.save_results.assert_called_once()
        report_gen.generate.assert_called_once()

    @patch("src.benchmark.cli.reset_benchmark_settings")
    @patch("src.benchmark.cli.get_benchmark_settings")
    @patch("src.benchmark.cli.set_correlation_id")
    @patch("src.benchmark.cli.BenchmarkFactory")
    def test_execution_failure_returns_1(self, mock_factory_cls, mock_set_cid, mock_get_settings, mock_reset):
        settings = MagicMock()
        settings.benchmark_enabled = True
        mock_get_settings.return_value = settings

        factory_instance = MagicMock()
        mock_factory_cls.return_value = factory_instance

        harness = MagicMock()
        harness.run = AsyncMock(side_effect=RuntimeError("execution failed"))
        factory_instance.create_harness.return_value = harness

        args = argparse.Namespace(
            iterations=None,
            output_dir=None,
            no_scoring=False,
            dry_run=False,
            systems=None,
            tasks=None,
            no_save_results=False,
            no_report=False,
        )
        result = asyncio.run(run_benchmark(args))
        assert result == 1


@pytest.mark.unit
class TestPrintSummary:
    """Tests for _print_summary helper."""

    def test_print_summary_outputs(self, capsys):
        summary = {
            "run_id": "test-abc",
            "total_results": 2,
            "systems": {
                "sys1": {
                    "successful": 1,
                    "total_tasks": 2,
                    "avg_latency_ms": 150.0,
                    "avg_score": 0.85,
                    "total_cost_usd": 0.005,
                },
            },
        }
        _print_summary(summary)
        captured = capsys.readouterr()
        assert "test-abc" in captured.out
        assert "Total Results: 2" in captured.out
        assert "sys1" in captured.out
        assert "150ms" in captured.out

    def test_print_summary_no_systems(self, capsys):
        summary = {"run_id": "empty", "total_results": 0}
        _print_summary(summary)
        captured = capsys.readouterr()
        assert "empty" in captured.out


@pytest.mark.unit
class TestDryRun:
    """Tests for _dry_run helper."""

    def test_dry_run_with_specified_systems(self, capsys):
        factory = MagicMock()
        adapter_factory = MagicMock()
        adapter_factory.get_available_systems.return_value = ["langgraph_mcts", "vertex_adk"]
        factory.create_adapter_factory.return_value = adapter_factory

        registry = MagicMock()
        task = MagicMock()
        task.task_id = "A1"
        task.description = "Test"
        task.category.value = "reasoning"
        registry.get_all.return_value = [task]
        registry.get.return_value = task
        factory.create_task_registry.return_value = registry

        settings = MagicMock()
        settings.run.num_iterations = 2
        settings.scoring.enabled = True
        settings.report.output_dir = "/tmp"
        factory.settings = settings

        args = argparse.Namespace(systems=["langgraph_mcts"], tasks=None)
        result = _dry_run(factory, args)
        assert result == 0
        captured = capsys.readouterr()
        assert "langgraph_mcts" in captured.out
        assert "available" in captured.out

    def test_dry_run_task_not_found(self, capsys):
        factory = MagicMock()
        adapter_factory = MagicMock()
        adapter_factory.get_available_systems.return_value = []
        factory.create_adapter_factory.return_value = adapter_factory

        registry = MagicMock()
        registry.get.side_effect = KeyError("not found")
        factory.create_task_registry.return_value = registry

        settings = MagicMock()
        settings.run.num_iterations = 1
        settings.scoring.enabled = False
        settings.report.output_dir = "/tmp"
        factory.settings = settings

        args = argparse.Namespace(systems=None, tasks=["MISSING"])
        result = _dry_run(factory, args)
        assert result == 0
        captured = capsys.readouterr()
        assert "NOT FOUND" in captured.out


@pytest.mark.unit
class TestMain:
    """Tests for main entry point."""

    @patch("src.benchmark.cli.sys.exit")
    @patch("src.benchmark.cli.asyncio.run", return_value=0)
    @patch("src.benchmark.cli.configure_logging")
    @patch("src.benchmark.cli.build_parser")
    def test_main_calls_sys_exit(self, mock_build_parser, mock_logging, mock_async_run, mock_exit):
        parser = MagicMock()
        parser.parse_args.return_value = argparse.Namespace(log_level="INFO")
        mock_build_parser.return_value = parser

        main()
        mock_exit.assert_called_once_with(0)

    @patch("src.benchmark.cli.sys.exit")
    @patch("src.benchmark.cli.asyncio.run", return_value=1)
    @patch("src.benchmark.cli.configure_logging")
    @patch("src.benchmark.cli.build_parser")
    def test_main_exit_code_1(self, mock_build_parser, mock_logging, mock_async_run, mock_exit):
        parser = MagicMock()
        parser.parse_args.return_value = argparse.Namespace(log_level="INFO")
        mock_build_parser.return_value = parser

        main()
        mock_exit.assert_called_once_with(1)
