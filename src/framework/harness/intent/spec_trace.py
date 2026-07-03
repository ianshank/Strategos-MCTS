"""CI spec-traceability engine behind ``harness spec-trace`` (SDD Phase 1).

Enforces the rules of ``docs/plans/SDD_PLUGIN_EXTRACTION_PLAN.md`` §3 on a
pull request's diff:

(a) a PR that changes nothing under ``src/`` needs no spec reference;
(b) otherwise a ``No-Spec: <reason>`` commit trailer exempts it (hotfix
    channel — the reason is echoed into the log);
(c) otherwise the branch must be ``spec/<id>`` and ``specs/<id>.SPEC.md``
    must be ``approved`` on the base branch (the completing PR may flip it to
    ``implemented`` in its own diff; a spec therefore cannot be authored and
    implemented in one PR — the trailer is the escape hatch);
(d) independently of (a)–(c): any flip to ``verified`` in the diff requires,
    per ``AC-n``, a same-line co-occurrence of the spec id and the
    word-bounded AC token somewhere under ``tests/**/*.py`` (e.g. a test
    docstring ``Covers my_spec AC-1``). Rule (d) is deliberately NOT
    short-circuited by (a) or (b): the canonical verified-flip PR touches
    only specs/ + tests/, and an exemption trailer must not bypass it.

Stated limitation (matches §3): ``src/`` diffs are not scoped against the
spec's ``module`` prefix.

The decision logic is a pure function (:func:`evaluate_trace`); the git layer
gathers its inputs with plain ``git`` subprocess calls so CI needs nothing
beyond the clone (``fetch-depth: 0``) and the installed harness.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from src.framework.harness.intent.spec_loader import SpecLoader
from src.framework.harness.intent.spec_scaffold import SPEC_ID_PATTERN

SPEC_BRANCH_RE: Final[re.Pattern[str]] = re.compile(rf"^spec/({SPEC_ID_PATTERN})$")
# Trailer scanned over concatenated `git log --format=%B` output → MULTILINE.
NO_SPEC_TRAILER_RE: Final[re.Pattern[str]] = re.compile(r"^No-Spec:\s*(\S.*)$", re.MULTILINE)
_SPEC_FILE_RE: Final[re.Pattern[str]] = re.compile(rf"^specs/({SPEC_ID_PATTERN})\.SPEC\.md$")

_REMEDIATION: Final[str] = (
    "PRs touching src/** must run on a spec/<id> branch whose specs/<id>.SPEC.md is "
    "'approved' on the base branch, OR carry a commit trailer 'No-Spec: <reason>'. "
    "See docs/plans/SDD_PLUGIN_EXTRACTION_PLAN.md §3."
)


@dataclass(frozen=True)
class VerifiedFlip:
    """A spec whose status becomes ``verified`` in this diff."""

    spec_id: str
    unmapped_criteria: tuple[str, ...]


@dataclass(frozen=True)
class TraceResult:
    """Outcome of the traceability evaluation."""

    ok: bool
    messages: tuple[str, ...]


def evaluate_trace(
    *,
    branch: str,
    changed_files: Sequence[str],
    base_status: str | None,
    head_status: str | None,
    trailers: Sequence[str],
    verified_flips: Sequence[VerifiedFlip],
    allow_unmapped_verified: bool = False,
) -> TraceResult:
    """Pure rule evaluation; all repo state is supplied by the caller."""
    problems: list[str] = []
    notes: list[str] = []

    src_changed = any(f.startswith("src/") for f in changed_files)
    if not src_changed:
        notes.append("no src/ changes; spec reference not required")
    elif trailers:
        notes.append(f"No-Spec exemption: {trailers[0]}")
    else:
        match = SPEC_BRANCH_RE.match(branch)
        if not match:
            problems.append(f"branch '{branch}' is not a spec/<id> branch and no No-Spec trailer found. {_REMEDIATION}")
        elif base_status != "approved":
            hint = ""
            if base_status == "implemented":
                hint = (
                    " (this spec already completed — follow-up src/ work on its branch needs a "
                    "'No-Spec: <reason>' trailer)"
                )
            problems.append(
                f"spec '{match.group(1)}' is '{base_status or 'absent'}' on the base branch, "
                f"not 'approved'{hint}. {_REMEDIATION}"
            )
        elif head_status not in {"approved", "implemented"}:
            problems.append(
                f"spec '{match.group(1)}' has head status '{head_status or 'absent'}'; only the "
                "approved->implemented flip may land in an implementing PR (verified flips are a "
                "separate, test-mapped PR)."
            )

    # Rule (d): unconditional — evaluated on every PR, never bypassed by (a)/(b).
    for flip in verified_flips:
        if flip.unmapped_criteria:
            message = (
                f"spec '{flip.spec_id}' flips to 'verified' but criteria "
                f"[{', '.join(flip.unmapped_criteria)}] have no test mapping: add a line under "
                f"tests/**/*.py containing both '{flip.spec_id}' and the AC id "
                f"(e.g. a docstring 'Covers {flip.spec_id} {flip.unmapped_criteria[0]}')."
            )
            if allow_unmapped_verified:
                notes.append(f"warning (softened): {message}")
            else:
                problems.append(message)

    return TraceResult(ok=not problems, messages=tuple(problems + notes))


def run_trace(
    repo_root: Path,
    base_ref: str,
    head_ref: str,
    branch: str,
    allow_unmapped_verified: bool = False,
) -> TraceResult:
    """Gather inputs from git and evaluate the trace rules."""
    changed = _git(repo_root, "diff", "--name-only", "--no-renames", f"{base_ref}...{head_ref}").splitlines()
    trailers = NO_SPEC_TRAILER_RE.findall(_git(repo_root, "log", f"{base_ref}..{head_ref}", "--format=%B"))

    base_status: str | None = None
    head_status: str | None = None
    branch_match = SPEC_BRANCH_RE.match(branch)
    if branch_match:
        spec_id = branch_match.group(1)
        base_status = _status_at_ref(repo_root, base_ref, spec_id)
        head_status = _status_in_worktree(repo_root, spec_id)

    flips = _verified_flips(repo_root, base_ref, changed)
    return evaluate_trace(
        branch=branch,
        changed_files=changed,
        base_status=base_status,
        head_status=head_status,
        trailers=trailers,
        verified_flips=flips,
        allow_unmapped_verified=allow_unmapped_verified,
    )


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _status_at_ref(repo_root: Path, ref: str, spec_id: str) -> str | None:
    try:
        text = _git(repo_root, "show", f"{ref}:specs/{spec_id}.SPEC.md")
    except subprocess.CalledProcessError:
        return None
    return SpecLoader().parse(text).status or None


def _status_in_worktree(repo_root: Path, spec_id: str) -> str | None:
    path = repo_root / "specs" / f"{spec_id}.SPEC.md"
    if not path.is_file():
        return None
    return SpecLoader().load(path).status or None


def _verified_flips(repo_root: Path, base_ref: str, changed_files: Sequence[str]) -> list[VerifiedFlip]:
    loader = SpecLoader()
    flips: list[VerifiedFlip] = []
    for changed in changed_files:
        file_match = _SPEC_FILE_RE.match(changed)
        if not file_match:
            continue
        spec_id = file_match.group(1)
        path = repo_root / changed
        if not path.is_file():  # deleted in this diff
            continue
        head_spec = loader.load(path)
        if head_spec.status != "verified" or _status_at_ref(repo_root, base_ref, spec_id) == "verified":
            continue
        criterion_ids = [c.id for c in head_spec.criteria]
        unmapped = _unmapped_criteria(repo_root, spec_id, criterion_ids)
        flips.append(VerifiedFlip(spec_id=spec_id, unmapped_criteria=unmapped))
    return flips


def _unmapped_criteria(repo_root: Path, spec_id: str, criterion_ids: Sequence[str]) -> tuple[str, ...]:
    """Criteria lacking a same-line spec-id + word-bounded AC-token line under tests/.

    Single pass over ``tests/**/*.py`` for the whole criterion set (not one
    walk per criterion): each file is read at most once, per flip. No caching
    across calls — the worktree may change between invocations in one process.
    """
    remaining = {cid: re.compile(rf"\b{re.escape(cid)}\b") for cid in criterion_ids}
    tests_dir = repo_root / "tests"
    if not tests_dir.is_dir() or not remaining:
        return tuple(criterion_ids)
    for test_file in tests_dir.rglob("*.py"):
        try:
            text = test_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        for line in text.splitlines():
            if spec_id not in line:
                continue
            for cid in [c for c, token in remaining.items() if token.search(line)]:
                del remaining[cid]
            if not remaining:
                return ()
    return tuple(cid for cid in criterion_ids if cid in remaining)


__all__ = [
    "NO_SPEC_TRAILER_RE",
    "SPEC_BRANCH_RE",
    "TraceResult",
    "VerifiedFlip",
    "evaluate_trace",
    "run_trace",
]
