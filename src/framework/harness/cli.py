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
import glob
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

from src.framework.harness.factories import HarnessFactory
from src.framework.harness.intent import SpecLoader, SpecParseError, SpecValidator
from src.framework.harness.intent.spec_scaffold import SpecScaffoldError, scaffold_spec
from src.framework.harness.intent.spec_trace import run_trace
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

    new = sub.add_parser("spec-new", help="Scaffold a schema-v2 spec; refuses on module overlap with open specs.")
    new.add_argument("--id", required=True, dest="spec_id", help="Spec id (lowercase/digits/_; becomes <id>.SPEC.md).")
    new.add_argument("--module", required=True, help="Repo-relative path prefix the spec governs (e.g. src/api/).")
    new.add_argument("--goal", default="", help="One-line goal (a TODO placeholder is written when omitted).")
    new.add_argument("--specs-dir", type=Path, default=Path("specs"))

    status = sub.add_parser("spec-status", help="Print a spec's lifecycle status; --require exits 1 on mismatch.")
    status.add_argument("spec_id", metavar="id")
    status.add_argument("--require", default=None, help="Exit 1 unless the status equals this value.")
    status.add_argument("--specs-dir", type=Path, default=Path("specs"))

    trace = sub.add_parser(
        "spec-trace",
        help="CI spec-traceability: PR diffs touching src/ need an approved spec/<id> branch or a No-Spec trailer.",
    )
    trace.add_argument("--base-ref", required=True, help="Base ref to diff against (e.g. origin/main).")
    trace.add_argument("--head-ref", default="HEAD")
    trace.add_argument("--branch", required=True, help="Head branch name (github.head_ref; PR checkouts are detached).")
    trace.add_argument(
        "--allow-unmapped-verified",
        action="store_true",
        help="Soften the verified-flip AC/test mapping rule to a warning.",
    )

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
    expanded_paths: list[Path] = []
    for path in args.paths:
        path_str = str(path)
        if any(char in path_str for char in ("*", "?")):
            matches = [Path(m) for m in sorted(glob.glob(path_str))]
            if matches:
                expanded_paths.extend(matches)
            else:
                expanded_paths.append(path)
        else:
            expanded_paths.append(path)

    report = SpecValidator().validate_paths(expanded_paths)
    for issue in report.issues:
        sys.stderr.write(issue.render() + "\n")
    failing = {issue.path for issue in report.errors()}
    warn_counts: dict[str, int] = {}
    for issue in report.issues:
        if issue.severity == "warning":
            warn_counts[issue.path] = warn_counts.get(issue.path, 0) + 1
    for path in expanded_paths:
        spec = report.specs.get(str(path))
        if str(path) in failing or spec is None:
            continue
        # Surface per-file warning counts on stdout so "ok with warnings" is
        # visible in the summary, not only on stderr (exit code stays 0).
        warn_note = f" warnings={warn_counts[str(path)]}" if str(path) in warn_counts else ""
        sys.stdout.write(
            f"ok: {path}: id='{spec.id}' status={spec.status} "
            f"goal='{spec.goal[:_GOAL_PREVIEW_CHARS]}' criteria={len(spec.acceptance_criteria)}{warn_note}\n"
        )
    return 1 if failing else 0


def _cmd_spec_new(args: argparse.Namespace) -> int:
    try:
        path = scaffold_spec(args.specs_dir, args.spec_id, args.module, args.goal)
    except SpecScaffoldError as exc:
        sys.stderr.write(f"error: spec-new: {exc}\n")
        return 1
    sys.stdout.write(f"created: {path} (status=draft) — fill Goal/ACs, then spec-review gates draft->approved\n")
    return 0


def _cmd_spec_status(args: argparse.Namespace) -> int:
    path = args.specs_dir / f"{args.spec_id}.SPEC.md"
    try:
        spec = SpecLoader().load(path)
    except SpecParseError as exc:
        sys.stderr.write(f"error: spec-status: {exc}\n")
        return 1
    status = spec.status or "<none>"
    sys.stdout.write(f"id={args.spec_id} status={status}\n")
    if args.require and spec.status != args.require:
        sys.stderr.write(f"error: spec-status: spec '{args.spec_id}' is '{status}', required '{args.require}'\n")
        return 1
    return 0


def _cmd_spec_trace(args: argparse.Namespace) -> int:
    try:
        result = run_trace(
            Path.cwd(),
            base_ref=args.base_ref,
            head_ref=args.head_ref,
            branch=args.branch,
            allow_unmapped_verified=args.allow_unmapped_verified,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        # Operational failure (bad ref, not a repo, hung/missing git) — report
        # like the other subcommands instead of a traceback; exit 1 fails CI.
        detail = (getattr(exc, "stderr", None) or "").strip()
        sys.stderr.write(f"error: spec-trace: {exc}{f' ({detail})' if detail else ''}\n")
        return 1
    for message in result.messages:
        stream = sys.stdout if result.ok else sys.stderr
        stream.write(f"spec-trace: {message}\n")
    sys.stdout.write(f"spec-trace: {'OK' if result.ok else 'FAILED'}\n")
    return 0 if result.ok else 1


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
    if args.command == "spec-new":
        return _cmd_spec_new(args)
    if args.command == "spec-status":
        return _cmd_spec_status(args)
    if args.command == "spec-trace":
        return _cmd_spec_trace(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
