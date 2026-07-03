"""``harness`` console-script entry point.

Subcommands:

* ``harness run`` — full loop against a spec or inline goal.
* ``harness dry-run`` — parse spec, build plan, exit without LLM calls.
* ``harness replay`` — replay a recorded cassette through the runner.
* ``harness validate-spec`` — validate one or more SPEC.md files against the
  spec schema v2 (frontmatter ``id``/``status`` lifecycle, ``AC-n:`` criterion
  IDs, no inline done-markers); errors exit 1. Ad-hoc AGENTS.md-style files do
  not carry the v2 frontmatter and will not validate — ``run``/``dry-run``
  still accept them.

The CLI uses ``argparse`` to avoid pulling in optional dependencies (no
``click``/``typer`` at runtime). All defaults come from
:class:`HarnessSettings`; flags only override.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from src.framework.harness.factories import HarnessFactory
from src.framework.harness.intent import SpecLoader, SpecValidator
from src.framework.harness.outcomes import Terminal
from src.framework.harness.planner import HeuristicPlanner
from src.framework.harness.settings import HarnessSettings

# Length of the goal excerpt shown on validate-spec success lines.
_GOAL_PREVIEW_CHARS = 80


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Strategos agent harness CLI.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level (DEBUG, INFO, WARNING).")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the harness against a spec or inline goal.")
    run.add_argument("--spec", type=Path, help="Path to SPEC.md / AGENTS.md.")
    run.add_argument("--goal", help="Inline goal string (used if --spec not provided).")
    run.add_argument("--max-iterations", type=int, default=None)
    run.add_argument("--memory-root", type=Path, default=None)
    run.add_argument("--output-dir", type=Path, default=None)
    run.add_argument("--ralph", action="store_true", help="Run via the Ralph outer loop.")
    run.add_argument("--shell-allow", action="append", default=[], help="Add an entry to the shell allowlist.")
    run.add_argument("--json", action="store_true", help="Emit JSON-only stdout.")

    dry = sub.add_parser("dry-run", help="Parse the spec and print the planned task; no LLM calls.")
    dry.add_argument("--spec", type=Path, required=True)

    replay = sub.add_parser("replay", help="Replay a recorded cassette directory.")
    replay.add_argument("--cassette-dir", type=Path, required=True)
    replay.add_argument("--spec", type=Path)
    replay.add_argument("--goal")

    val = sub.add_parser(
        "validate-spec",
        help="Validate SPEC.md files against the spec schema v2; any error exits 1.",
    )
    val.add_argument("paths", type=Path, nargs="+", metavar="path")

    return parser


def _apply_settings_overrides(args: argparse.Namespace) -> HarnessSettings:
    """Promote a few CLI flags into ``HARNESS_*`` env vars before instantiation."""
    if getattr(args, "max_iterations", None) is not None:
        os.environ["HARNESS_MAX_ITERATIONS"] = str(args.max_iterations)
    if getattr(args, "memory_root", None) is not None:
        os.environ["HARNESS_MEMORY_ROOT"] = str(args.memory_root)
    if getattr(args, "output_dir", None) is not None:
        os.environ["HARNESS_OUTPUT_DIR"] = str(args.output_dir)
    return HarnessSettings()


def _resolve_intent(args: argparse.Namespace) -> str | dict[str, object]:
    if getattr(args, "spec", None):
        spec = SpecLoader().load(args.spec)
        return {
            "id": f"cli-{args.spec.stem}",
            "goal": spec.goal or f"Execute spec at {args.spec}",
            "acceptance_criteria": spec.criteria_payload(),
            "constraints": list(spec.constraints),
            "metadata": {"spec_path": str(args.spec)},
        }
    goal = getattr(args, "goal", None)
    if goal:
        return str(goal)
    raise SystemExit("error: must supply --spec or --goal")


async def _cmd_run(args: argparse.Namespace) -> int:
    hs = _apply_settings_overrides(args)
    factory = HarnessFactory(harness_settings=hs)
    # ``replay`` reuses this path but its subparser omits the run-only flags,
    # so read them defensively rather than assuming they exist on the namespace.
    shell_allow = getattr(args, "shell_allow", None)
    runner = factory.create_runner(shell_allowlist=shell_allow or None)
    intent = _resolve_intent(args)

    payload: dict[str, object]
    if getattr(args, "ralph", False):
        loop = factory.create_ralph(runner, spec_path=args.spec)
        result = await loop.run()
        ralph_accepted = result.status in {"accepted", "done"}
        payload = {
            "status": result.status,
            "accepted": ralph_accepted,
            "rounds": result.rounds,
            "stuck_kind": result.stuck_kind,
            "outcome": result.last_run.outcome.kind if result.last_run else None,
            "confidence": result.last_run.confidence if result.last_run else 0.0,
        }
    else:
        run_result = await runner.run(intent)
        accepted = isinstance(run_result.outcome, Terminal) and run_result.outcome.accepted
        payload = {
            "outcome": run_result.outcome.kind,
            "accepted": accepted,
            "iterations": run_result.iterations,
            "duration_ms": round(run_result.duration_ms, 2),
            "confidence": run_result.confidence,
            "metadata": run_result.metadata,
        }

    if getattr(args, "json", False):
        sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")
    else:
        sys.stdout.write(f"outcome={payload['outcome']}\n")
        for k, v in payload.items():
            if k == "outcome":
                continue
            sys.stdout.write(f"{k}={v}\n")

    return 0 if payload.get("accepted") else 2


async def _cmd_dry_run(args: argparse.Namespace) -> int:
    spec = SpecLoader().load(args.spec)
    intent = {
        "id": "dry-run",
        "goal": spec.goal,
        "acceptance_criteria": spec.criteria_payload(),
        "constraints": list(spec.constraints),
    }
    from src.framework.harness.intent import DefaultIntentNormalizer

    task = await DefaultIntentNormalizer().normalize(intent, HarnessSettings())
    plan = await HeuristicPlanner().plan(task)
    payload = {
        "task_id": task.id,
        "goal": task.goal,
        "criteria": [c.description for c in task.acceptance_criteria],
        "plan_summary": plan.summary,
        "plan_steps": [{"id": s.id, "description": s.description} for s in plan.steps],
    }
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return 0


async def _cmd_replay(args: argparse.Namespace) -> int:
    os.environ["HARNESS_REPLAY_DIR"] = str(args.cassette_dir)
    return await _cmd_run(args)


def _cmd_validate_spec(args: argparse.Namespace) -> int:
    """Validate every given path; report all issues, exit 1 if any error."""
    issues = SpecValidator().validate_paths(args.paths)
    for issue in issues:
        sys.stderr.write(issue.render() + "\n")
    failing = {issue.path for issue in issues if issue.severity == "error"}
    warn_counts: dict[str, int] = {}
    for issue in issues:
        if issue.severity == "warning":
            warn_counts[issue.path] = warn_counts.get(issue.path, 0) + 1
    loader = SpecLoader()
    for path in args.paths:
        if str(path) in failing:
            continue
        spec = loader.load(path)
        # Surface per-file warning counts on stdout so "ok with warnings" is
        # visible in the summary, not only on stderr (exit code stays 0).
        warn_note = f" warnings={warn_counts[str(path)]}" if str(path) in warn_counts else ""
        sys.stdout.write(
            f"ok: {path}: id='{spec.id}' status={spec.status} "
            f"goal='{spec.goal[:_GOAL_PREVIEW_CHARS]}' criteria={len(spec.acceptance_criteria)}{warn_note}\n"
        )
    return 1 if failing else 0


def main(argv: list[str] | None = None) -> int:
    """Entry point invoked by the ``harness`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    if args.command == "run":
        return asyncio.run(_cmd_run(args))
    if args.command == "dry-run":
        return asyncio.run(_cmd_dry_run(args))
    if args.command == "replay":
        return asyncio.run(_cmd_replay(args))
    if args.command == "validate-spec":
        return _cmd_validate_spec(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
