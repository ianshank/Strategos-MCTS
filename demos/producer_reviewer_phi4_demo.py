"""Producer-Reviewer demo wired to LM Studio + (by default) the configured reasoning preset.

CLI defaults are env-driven; nothing is hardcoded. Designed for sequential,
latency-bounded inference on a single local model (e.g. a 14B reasoning
model on Tesla P40 / 24GB) — producer and reviewer share a single LLM
client and execute strictly serially.

Usage:
    python -m demos.producer_reviewer_phi4_demo --task A1 [--rounds 3]
    HARNESS_BENCHMARK_TASK_ID=A2 python -m demos.producer_reviewer_phi4_demo

Exit codes:
    0  — pipeline completed successfully (`outcome.success is True`)
    1  — pipeline ran but the final outcome was not successful
    2  — usage error (e.g. no task id supplied, unknown task id)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence

from src.adapters.llm.base import LLMClient
from src.benchmark.harness_bridge import BenchmarkTaskAdapter
from src.benchmark.tasks.task_sets import ALL_TASKS
from src.config.settings import Settings, get_settings
from src.framework.harness.agents import LLMProducerAgent, LLMReviewerAgent
from src.framework.harness.factories import HarnessFactory
from src.framework.harness.settings import HarnessSettings, get_harness_settings
from src.framework.harness.topology.base import AgentOutcome
from src.framework.harness.topology.producer_reviewer import ProducerReviewerTopology

logger = logging.getLogger("demos.producer_reviewer_phi4_demo")

# Exit codes — kept module-level for tests and humans to introspect.
EXIT_OK: int = 0
EXIT_PIPELINE_FAILURE: int = 1
EXIT_USAGE_ERROR: int = 2

DEFAULT_TASK_ID_HELP: str = "Benchmark task id (e.g. A1, A2, A3). Default: HARNESS_BENCHMARK_TASK_ID env."


def _known_task_ids() -> list[str]:
    """Return the list of known benchmark task ids for help / error text."""
    return [bt.task_id for bt in ALL_TASKS]


def build_parser(
    *,
    settings: Settings | None = None,
    harness_settings: HarnessSettings | None = None,
) -> argparse.ArgumentParser:
    """Construct the demo's argument parser with env-driven defaults.

    All numeric / string defaults trace to :class:`Settings` or
    :class:`HarnessSettings`; nothing is hardcoded.
    """
    s = settings or get_settings()
    hs = harness_settings or get_harness_settings()

    parser = argparse.ArgumentParser(
        prog="producer_reviewer_phi4_demo",
        description=(
            "Run the producer-reviewer pipeline against an LM Studio reasoning "
            "model on a chosen benchmark task. Defaults are env-driven."
        ),
    )
    parser.add_argument(
        "--task",
        default=hs.BENCHMARK_TASK_ID,
        help=DEFAULT_TASK_ID_HELP,
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=hs.PRODUCER_REVIEWER_ROUNDS,
        help=("Maximum producer/reviewer rounds. " "Default: HarnessSettings.PRODUCER_REVIEWER_ROUNDS."),
    )
    parser.add_argument(
        "--producer-max-tokens",
        type=int,
        default=hs.PRODUCER_MAX_TOKENS,
        help="Max tokens for the producer's draft. Default: HarnessSettings.PRODUCER_MAX_TOKENS.",
    )
    parser.add_argument(
        "--reviewer-max-tokens",
        type=int,
        default=hs.REVIEWER_MAX_TOKENS,
        help="Max tokens for the reviewer's review. Default: HarnessSettings.REVIEWER_MAX_TOKENS.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=s.LMSTUDIO_TEMPERATURE,
        help=(
            "Sampling temperature override. Default: Settings.LMSTUDIO_TEMPERATURE "
            "(may be None to defer to preset / DEFAULT_LMSTUDIO_TEMPERATURE)."
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_only",
        action="store_true",
        help="Emit JSON-only output (suppresses human prose) for machine consumption.",
    )
    parser.add_argument(
        "--log-level",
        default=str(s.LOG_LEVEL.value),
        help="Log level. Default: Settings.LOG_LEVEL.",
    )
    return parser


async def run_pipeline(
    *,
    llm: LLMClient,
    task_id: str,
    rounds: int,
    producer_max_tokens: int | None = None,
    reviewer_max_tokens: int | None = None,
    temperature: float | None = None,
) -> AgentOutcome:
    """Pure orchestration. Easily testable with a fake LLMClient.

    Args:
        llm: LLM client shared by both the producer and the reviewer.
        task_id: Benchmark task id (case-insensitive).
        rounds: Maximum producer/reviewer rounds.
        producer_max_tokens: Override for the producer's max-token budget. When
            ``None`` the agent's class default applies.
        reviewer_max_tokens: Override for the reviewer's max-token budget. When
            ``None`` the agent's class default applies.
        temperature: Optional sampling temperature shared by both agents.

    Returns:
        The aggregated :class:`AgentOutcome` from the topology run.

    Raises:
        KeyError: when ``task_id`` does not match any known benchmark task.
    """
    logger.info(
        "run_pipeline start task_id=%s rounds=%d producer_max_tokens=%s " "reviewer_max_tokens=%s temperature=%s",
        task_id,
        rounds,
        producer_max_tokens,
        reviewer_max_tokens,
        temperature,
    )
    adapter = BenchmarkTaskAdapter()
    bt = adapter.lookup(task_id)
    task = adapter.to_task(bt)

    producer_kwargs: dict[str, object] = {"llm": llm, "temperature": temperature}
    if producer_max_tokens is not None:
        producer_kwargs["max_tokens"] = producer_max_tokens
    reviewer_kwargs: dict[str, object] = {"llm": llm, "temperature": temperature}
    if reviewer_max_tokens is not None:
        reviewer_kwargs["max_tokens"] = reviewer_max_tokens

    producer = LLMProducerAgent(**producer_kwargs)  # type: ignore[arg-type]
    reviewer = LLMReviewerAgent(**reviewer_kwargs)  # type: ignore[arg-type]
    topology = ProducerReviewerTopology(name="producer_reviewer", max_rounds=rounds)
    outcome = await topology.run(task, [producer, reviewer])
    logger.info(
        "run_pipeline done task_id=%s success=%s confidence=%.3f winner=%s",
        task_id,
        outcome.success,
        outcome.confidence,
        outcome.agent_name,
    )
    return outcome


def _outcome_to_payload(outcome: AgentOutcome) -> dict[str, object]:
    """Convert an :class:`AgentOutcome` into a JSON-serialisable dict."""
    return {
        "agent_name": outcome.agent_name,
        "response": outcome.response,
        "confidence": outcome.confidence,
        "success": outcome.success,
        "error": outcome.error,
        "metadata": outcome.metadata,
    }


def format_outcome(outcome: AgentOutcome, *, json_only: bool = False) -> str:
    """Render the final :class:`AgentOutcome` for stdout.

    With ``json_only=True`` the result is a single JSON object (machine
    readable). Otherwise a short human header precedes a pretty-printed JSON
    block summarising the outcome and any intermediate agent history.
    """
    payload = _outcome_to_payload(outcome)
    intermediate_obj = outcome.metadata.get("intermediate", []) if outcome.metadata else []
    intermediate: list[dict[str, object]] = (
        [dict(item) for item in intermediate_obj if isinstance(item, dict)]
        if isinstance(intermediate_obj, list)
        else []
    )

    if json_only:
        return json.dumps(payload, indent=2, sort_keys=True, default=str)

    lines: list[str] = []
    status = "SUCCESS" if outcome.success else "FAILURE"
    lines.append(f"=== producer-reviewer pipeline: {status} ===")
    lines.append(f"agent_name: {outcome.agent_name}")
    lines.append(f"confidence: {outcome.confidence:.3f}")
    if outcome.error:
        lines.append(f"error: {outcome.error}")
    if intermediate:
        lines.append("")
        lines.append("intermediate history:")
        for idx, item in enumerate(intermediate, start=1):
            name = item.get("agent_name", "?")
            ok = item.get("success", False)
            conf = item.get("confidence", 0.0)
            lines.append(f"  [{idx}] {name} success={ok} confidence={conf}")
    lines.append("")
    lines.append("payload:")
    lines.append(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return "\n".join(lines)


def _print(message: str, *, stream: object | None = None) -> None:
    """Tiny indirection so tests can capture output via ``capsys``."""
    target = stream if stream is not None else sys.stdout
    print(message, file=target)  # type: ignore[arg-type]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns an integer exit code."""
    settings = get_settings()
    harness_settings = get_harness_settings()
    parser = build_parser(settings=settings, harness_settings=harness_settings)
    args = parser.parse_args(argv)

    logging.basicConfig(level=str(args.log_level).upper())

    task_id = args.task
    if not task_id:
        _print(
            "error: no benchmark task id supplied. Pass --task or set "
            "HARNESS_BENCHMARK_TASK_ID. Available task ids: " + ", ".join(_known_task_ids()),
            stream=sys.stderr,
        )
        return EXIT_USAGE_ERROR

    factory = HarnessFactory(settings=settings, harness_settings=harness_settings)
    try:
        llm = factory.create_llm()
    except Exception as exc:  # noqa: BLE001 - surface as friendly CLI error
        _print(f"error: failed to construct LLM client: {exc}", stream=sys.stderr)
        return EXIT_PIPELINE_FAILURE

    try:
        outcome = asyncio.run(
            run_pipeline(
                llm=llm,
                task_id=task_id,
                rounds=args.rounds,
                producer_max_tokens=args.producer_max_tokens,
                reviewer_max_tokens=args.reviewer_max_tokens,
                temperature=args.temperature,
            )
        )
    except KeyError as exc:
        _print(
            "error: unknown benchmark task id. "
            f"{exc.args[0] if exc.args else exc}. Available: {', '.join(_known_task_ids())}",
            stream=sys.stderr,
        )
        return EXIT_USAGE_ERROR

    _print(format_outcome(outcome, json_only=bool(args.json_only)))
    return EXIT_OK if outcome.success else EXIT_PIPELINE_FAILURE


if __name__ == "__main__":  # pragma: no cover - CLI dispatch
    sys.exit(main())
