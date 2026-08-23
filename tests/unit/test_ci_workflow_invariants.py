"""Structural invariants for the GitHub Actions workflows.

These tests exist because every defect they check for was found *in the tree*, not
hypothesised: jobs with no ``timeout-minutes`` inheriting the 360-minute default on a
build that had already hung; a summary job that printed a job's result without gating
on it, so a bandit HIGH finding merged green; ``|| true`` swallowing an entire e2e
suite; and workflows with no ``concurrency`` group leaving ten overlapping 45-minute
image builds in flight after a push burst.

Each check is therefore a *regression gate*, not a style preference. They are
deliberately derived from the workflow files rather than from a hardcoded list of job
names, so adding a job to a workflow automatically brings it under the invariant
instead of quietly escaping it — a hardcoded list would rot on the first new job and
is the failure mode these tests exist to prevent.

Spec: ``specs/hygiene_ci_mechanical.SPEC.md`` (AC-1, AC-3, AC-5, AC-8, AC-9, AC-11, AC-12).
The supply-chain and privilege section at the end of the file comes instead from
``specs/evidence_claim_ledger.SPEC.md`` AC-8.
Each section below is headed with the AC it enforces.
"""

from __future__ import annotations

import ast
import configparser
import json
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import pytest

# Imported normally, not via importorskip. PyYAML is a declared `dev` dependency, and a
# regression gate that silently skips itself when a dependency goes missing is exactly
# the failure mode this module exists to catch.
import yaml

from src.config.constants import LLM_PROVIDER_CREDENTIAL_ENV_VARS
from src.tools.action_pins import audit as pin_audit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

# Most invariants below are derived from *every* workflow, so they need no filename.
# These two are named because the invariant is genuinely about that specific pipeline:
# the main gate, and the one whose image matrix is expensive enough to need a paths
# filter. Named once so a rename fails in test_named_workflows_resolve with a clear
# message instead of five KeyErrors in unrelated tests.
CI_WORKFLOW = "ci.yml"
EXPENSIVE_WORKFLOW = "docker-deployment.yml"

# Upper bound on any single job. Not a performance target — a backstop that turns a
# hung job into a failed job. The GitHub default is 360 minutes, which is long enough
# that a hang looks like a slow run for most of a working day.
MAX_TIMEOUT_MINUTES = 90

# Commands whose EXIT CODE is itself the gate. Deliberately excludes `bandit`,
# `pip-audit` and `trivy`: those exit non-zero on *any* finding at *any* severity, so
# their `|| true` is correct design — a separate parsing/gating step decides, and that
# step is asserted elsewhere (test_image_scan_can_actually_fail, and the missing-report
# guards in ci.yml). Adding them here would flag working code and train people to
# ignore this test.
RESULT_BEARING_COMMANDS = (
    "pytest",
    "mypy",
    "ruff",
    "black",
    "gitleaks",
    "harness",
    # `pip` covers both dependency installs and the training-manifest resolve gate in
    # build-check. A disarmed `pip install` is never correct: it turns a failed install
    # into a green step whose later commands then fail for unrelated-looking reasons.
    # The resolve gate in particular is the only thing standing between a Dependabot
    # bump and an unresolvable manifest, so it must not be deletable without notice.
    #
    # This does NOT flag `pip-audit ... || true` (ci.yml:220): the word-boundary
    # lookahead below excludes a following hyphen, and pip-audit's suppression is the
    # correct design documented above — a separate step parses its report and gates.
    "pip",
)

# Shell constructs that discard a command's exit status. `|| true` is the obvious one
# and `|| :` is the same thing spelled with the null builtin. Both are matched on the
# command's own line.
_DISARMING_SUFFIXES = (r"\|\|\s*true\b", r"\|\|\s*:\s*$", r"\|\|\s*:\s")
_DISARMING_RE = re.compile("|".join(_DISARMING_SUFFIXES))

# `set +e` is a different shape of the same defect: it disarms every *subsequent*
# command in the step rather than the one it shares a line with, so a suffix regex
# cannot see it. GitHub runs `run:` blocks under `bash -e`, so `set +e` turns the rest
# of a multi-command step into best-effort — the step then reports the exit status of
# whatever happened to run last. Tracked as a state machine over the step's lines by
# _disarmed_commands().
#
# Parsed rather than pattern-matched, because errexit has two spellings and a regex
# for the short one silently misses the long one: `set +o errexit` disarms exactly as
# `set +e` does. The lookbehind keeps `offset=+e` and `--dataset +e` from matching.
_SET_LINE_RE = re.compile(r"(?<![\w-])set\s+([-+][^\n;&|]*)")


def _errexit_change(line: str) -> bool | None:
    """Whether a ``set`` line disables errexit (True), enables it (False), or is unrelated (None).

    Handles every spelling bash accepts, since they are equivalent in effect:

    * short flags, alone or bundled — ``set -e``, ``set +e``, ``set -euo pipefail``
    * the long option — ``set -o errexit``, ``set +o errexit``

    Unrelated options are ignored, so ``set -o pipefail`` and ``set -x`` return None
    and leave the current state alone rather than falsely re-arming.
    """
    match = _SET_LINE_RE.search(line)
    if not match:
        return None
    state: bool | None = None
    tokens = match.group(1).split()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in ("-o", "+o"):
            if index + 1 < len(tokens) and tokens[index + 1] == "errexit":
                state = token.startswith("+")
                index += 2
                continue
        elif token.startswith(("-", "+")) and "e" in token[1:]:
            state = token.startswith("+")
        index += 1
    return state


# A command name only counts when it appears as a *word*. Substring matching flagged
# `rm -rf .pytest_cache || true` (contains "pytest"), `docker rm blackbox || true`
# (contains "black") and `curl -F file=@pytest-report.xml || true` — all legitimate
# cleanup or upload steps. Word boundaries with an explicit guard against `-`/`.`/`_`
# neighbours fix that without needing an allowlist.
_COMMAND_WORD_RE = re.compile(
    r"(?<![\w./-])(?:" + "|".join(re.escape(c) for c in RESULT_BEARING_COMMANDS) + r")(?![\w.-])"
)

# One shell line can hold several commands. `set +e; pytest tests/unit/` disables
# errexit and then runs a suppressed test suite on that same line — treating the line
# as a single unit meant the `set +e` branch consumed it and the `pytest` after the
# separator was never examined at all.
#
# `;` and `&&` split; `||` deliberately does NOT, because `pytest … || true` is one
# command plus its disarming tail and splitting it would destroy the very pattern
# _DISARMING_RE exists to match.
_SEGMENT_SPLIT_RE = re.compile(r";|&&")

# The API suites whose three-layer suppression AC-5 removed. Named once so the
# suppression invariant and its message stay in step.
PROTECTED_API_SUITES = ("test_rest_server.py", "test_inference_server.py")

# pytest accepts both `--ignore=path` and `--ignore path`; matching only the first
# spelling let the second slip a suite back out of collection.
_IGNORE_ARG_RE = re.compile(r"--ignore(?:=|\s+)(\S+)")


def _disarmed_commands(run: str) -> list[str]:
    """Lines in a ``run:`` block where a result-bearing command cannot fail the step.

    Two independent disarming mechanisms, both reported:

    * A per-line suffix (``|| true``, ``|| :``) that swallows *that* command.
    * An in-scope ``set +e``, which swallows every command *after* it. A matching
      ``set -e`` re-arms, so a step that deliberately brackets one tolerant region is
      not flagged for the commands outside it.
    """
    offenders: list[str] = []
    errexit_disabled = False
    # Join backslash continuations first: a `|| true` on the wrapped tail of a
    # command still disarms it, and the two belong to the same logical line.
    logical = re.sub(r"\\\s*\n\s*", " ", run)
    for line in logical.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # a comment mentioning `|| true` is not a disarmed command
        for segment in (part.strip() for part in _SEGMENT_SPLIT_RE.split(stripped)):
            if not segment:
                continue
            change = _errexit_change(segment)
            if change is not None:
                errexit_disabled = change
                continue
            if not _COMMAND_WORD_RE.search(segment):
                continue
            if _DISARMING_RE.search(segment) or errexit_disabled:
                offenders.append(segment)
    return offenders


def _workflow_files() -> list[Path]:
    """Every workflow definition, discovered rather than listed.

    A hardcoded list would silently stop covering a newly added workflow — which is
    the class of gap this module exists to close.
    """
    files = sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))
    assert files, f"no workflow files found under {WORKFLOW_DIR}"
    return files


def _load(path: Path) -> dict[str, Any]:
    """Parse a workflow file into its YAML mapping."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _literal_collect_ignores() -> set[str]:
    """Paths added to conftest's ``collect_ignore_glob`` by a literal list.

    Parsed from the AST, not matched textually. conftest carries a comment that names
    ``test_inference_server.py`` while explaining why its torch guard exists, and a
    substring check read that comment as a suppression — this test failed on its own
    first draft for exactly that reason.

    ``_require_or_ignore()`` is invisible here by construction: it appends through a
    *variable* inside the function, so only unconditional literal additions surface.
    That is the intended distinction — the helper fires only when an optional
    dependency is genuinely absent, and STRICT_OPTIONAL_DEPS turns it into a hard
    error under CI.
    """
    tree = ast.parse((REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        target = node.target if isinstance(node, ast.AugAssign) else None
        if isinstance(target, ast.Name) and target.id == "collect_ignore_glob":
            for element in ast.walk(node.value):
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    found.add(element.value)
    return found


def _load_pyproject() -> dict[str, Any]:
    """Parse ``pyproject.toml``.

    ``tomllib`` is stdlib only from 3.11, while ``pyproject.toml`` itself declares
    ``requires-python = ">=3.10"``. Importing it unconditionally made this module
    fail at *collection* time on the declared baseline — and a regression gate that
    cannot be collected enforces nothing. The ``tomli`` backport is declared in the
    ``dev`` extra behind a ``python_version < "3.11"`` marker, so it costs nothing on
    the 3.11 CI runners.
    """
    try:
        import tomllib as toml_reader
    except ModuleNotFoundError:  # pragma: no cover - only on Python 3.10
        import tomli as toml_reader  # type: ignore[import-not-found]

    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return toml_reader.load(fh)


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return the ``on:`` block.

    PyYAML follows YAML 1.1, where the bare key ``on`` is parsed as the boolean
    ``True`` rather than the string ``"on"``. Both spellings are checked so this
    helper keeps working if the workflows ever quote the key.
    """
    return workflow.get("on") or workflow.get(True) or {}


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    """A job's steps, skipping any non-mapping entry so a malformed step cannot crash a test."""
    return [s for s in job.get("steps", []) or [] if isinstance(s, dict)]


def _all_jobs() -> list[tuple[str, str, dict[str, Any]]]:
    """Every (workflow filename, job id, job body) triple across all workflows."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for path in _workflow_files():
        for job_id, job in (_load(path).get("jobs") or {}).items():
            out.append((path.name, job_id, job))
    return out


def _job_ids() -> list[str]:
    """Stable ``workflow::job`` labels, so a parametrized failure names the job directly."""
    return [f"{wf}::{job_id}" for wf, job_id, _ in _all_jobs()]


def _gated_job_list(body: str) -> set[str]:
    """Names a summary job actually gates on.

    Two legitimate forms, both counted:

    1. Membership in the ``JOBS="..."`` list the status loop iterates.
    2. An explicit conditional — ``[ "${NAME}" != "success" ]`` or ``case "${NAME}"``.
       Needed because some results warrant different semantics than the loop's
       ``success|skipped``: a *skipped* ``check-secrets`` means the credential probe
       never ran, which is a failure, whereas a skipped test job is fine.

    Comment lines are stripped first. Searching the raw body let a name mentioned only
    in a ``#`` comment satisfy the check — mutation testing showed that was a live
    route to the exact regression this invariant exists to prevent.
    """
    code = "\n".join(line for line in body.splitlines() if not line.strip().startswith("#"))

    gated: set[str] = set()
    match = re.search(r'JOBS=(["\'])(.*?)\1', code, re.S)
    if match:
        # Collapse backslash-continuations, then split on whitespace.
        gated |= set(re.sub(r"\\\s*\n\s*", " ", match.group(2)).split())

    # Explicitly tested names: `[ "${NAME}" ...`, `[[ "${NAME}" ...`, `case "${NAME}"`.
    gated |= set(re.findall(r'(?:\[\[?|case)\s+"?\$\{([A-Z_][A-Z0-9_]*)\}"?', code))
    return gated


def _summary_jobs() -> list[tuple[str, dict[str, Any]]]:
    """Every job across all workflows that acts as a pipeline summary/gate.

    Derived from the workflows rather than naming ``ci.yml``'s summary explicitly, so
    a second workflow that grows a summary job is held to the same contract. The
    e2e workflow's summary was written by the same change and initially was not.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for wf, job_id, job in _all_jobs():
        if job_id != "summary":
            continue
        if any("JOBS=" in str(s.get("run", "")) for s in _steps(job)):
            out.append((f"{wf}::{job_id}", job))
    return out


# --------------------------------------------------------------------------- AC-8


@pytest.mark.unit
@pytest.mark.parametrize(("workflow", "job_id", "job"), _all_jobs(), ids=_job_ids())
def test_every_job_declares_a_timeout(workflow: str, job_id: str, job: dict[str, Any]) -> None:
    """AC-8: no job may inherit the 360-minute default.

    Baseline when this was written: 0 of 23 jobs declared a timeout.
    """
    timeout = job.get("timeout-minutes")
    assert timeout is not None, (
        f"{workflow}::{job_id} has no timeout-minutes, so it inherits GitHub's "
        f"360-minute default. A hung job would burn six hours of runner time before "
        f"being killed. Add an explicit budget."
    )
    assert isinstance(timeout, int) and 0 < timeout <= MAX_TIMEOUT_MINUTES, (
        f"{workflow}::{job_id} declares timeout-minutes={timeout!r}, outside the "
        f"sane range 1..{MAX_TIMEOUT_MINUTES}. If this job genuinely needs longer, "
        f"raise MAX_TIMEOUT_MINUTES deliberately and say why."
    )


# --------------------------------------------------------------------------- AC-9


@pytest.mark.unit
@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_every_workflow_has_a_concurrency_group(path: Path) -> None:
    """AC-9: a push burst must supersede in-flight runs, not stack them."""
    workflow = _load(path)
    concurrency = workflow.get("concurrency")
    assert concurrency, (
        f"{path.name} declares no concurrency group. A burst of pushes to the same "
        f"ref leaves every run in flight simultaneously; this repository has already "
        f"seen ~10 overlapping 30-45 minute image builds from one dependabot batch."
    )
    group = concurrency.get("group") if isinstance(concurrency, dict) else None
    assert group and "github.ref" in str(group), (
        f"{path.name} concurrency group {group!r} must be keyed on github.ref so that "
        f"runs for different branches do not cancel each other."
    )
    # A group alone only serializes: without cancel-in-progress the superseded runs
    # queue up and still execute, which is the cost this invariant exists to avoid.
    cancel = concurrency.get("cancel-in-progress") if isinstance(concurrency, dict) else None
    assert cancel is True, (
        f"{path.name} declares cancel-in-progress={cancel!r}. A concurrency group "
        f"without it queues superseded runs instead of cancelling them, so a push "
        f"burst still pays for every obsolete run — it just pays serially."
    )


@pytest.mark.unit
def test_expensive_workflow_filters_pull_requests_by_path() -> None:
    """AC-9: the Docker workflow must not run its image matrix on docs-only PRs.

    The ``push`` trigger has always carried a ``paths`` filter; the ``pull_request``
    trigger did not, so a two-file Markdown PR ran the full ``Dockerfile.train``
    matrix — the single most expensive job in the repository.
    """
    triggers = _triggers(_load(WORKFLOW_DIR / EXPENSIVE_WORKFLOW))
    push_paths = set((triggers.get("push") or {}).get("paths") or [])
    pr_paths = set((triggers.get("pull_request") or {}).get("paths") or [])

    assert pr_paths, "docker-deployment.yml pull_request trigger has no paths filter"
    assert push_paths <= pr_paths, (
        "docker-deployment.yml pull_request paths filter is narrower than its push "
        f"filter; missing {sorted(push_paths - pr_paths)}. The two should agree, or a "
        "change that triggers a build on push will silently not be built on the PR "
        "that introduces it."
    )


# --------------------------------------------------------------------------- AC-1


@pytest.mark.unit
@pytest.mark.parametrize(("label", "summary"), _summary_jobs(), ids=[lbl for lbl, _ in _summary_jobs()])
def test_summary_gates_every_job_it_depends_on(label: str, summary: dict[str, Any]) -> None:
    """AC-1: the reported set and the enforced set must be identical.

    The original summary job listed ``security-scan`` and ``dependency-audit`` in
    ``needs``, printed their results, and then omitted them from the failure
    condition — so a bandit HIGH finding printed a red line and merged green.
    """
    needs = summary["needs"]
    if isinstance(needs, str):
        needs = [needs]
    assert needs, f"{label} must declare a needs list"

    step = next(s for s in _steps(summary) if "run" in s)
    env = step.get("env") or {}
    body = step["run"]

    # 1. Every needed job is surfaced as an env var bound to that job's result.
    bound = {
        job for job in needs for value in env.values() if re.search(rf"needs\.{re.escape(job)}\.result", str(value))
    }
    missing = set(needs) - bound
    assert not missing, (
        f"{label} depends on {sorted(missing)} but never reads their .result "
        f"into an env var, so they cannot be gated on."
    )

    # 2. Every such env var is named in the shell list the gate iterates over. This is
    #    what makes reported == enforced structurally, rather than by coincidence.
    #    Matched against the extracted JOBS="..." assignment ONLY — searching the whole
    #    body let a mere `#` comment mentioning the name satisfy the check, which
    #    mutation testing showed was a live route to the exact regression this
    #    prevents.
    gated_list = _gated_job_list(body)
    result_vars = {name for name, value in env.items() if re.search(r"needs\.[\w-]+\.result", str(value))}
    ungated = {name for name in result_vars if name not in gated_list}
    assert not ungated, (
        f"{label} reads {sorted(ungated)} but never checks them in the gate "
        f"body. A job that is printed but not enforced is worse than one that is "
        f"absent, because the red line in the log implies it was checked."
    )

    # 3. The gate must be able to fail. Two equivalent spellings are accepted: a
    #    literal `exit 1`, or an accumulator (`status=1` … `exit "${status}"`). Demanding
    #    the literal would reject the accumulator form, which is the better one for a
    #    loop that wants to report every offender before exiting.
    has_literal_exit = re.search(r"\bexit\s+1\b", body)
    has_accumulator = re.search(r"\bstatus=1\b", body) and re.search(r'\bexit\s+"?\$\{status\}"?', body)
    assert has_literal_exit or has_accumulator, (
        f"{label} has no failing exit path: expected either `exit 1` or a "
        f'`status=1` … `exit "${{status}}"` accumulator.'
    )


@pytest.mark.unit
def test_ci_summary_covers_the_test_bearing_jobs() -> None:
    """AC-1: the jobs that actually run tests must be gated.

    ``chess-tests`` and ``integration-test`` were absent from ``needs`` entirely, so
    neither could block a merge no matter how it finished.
    """
    jobs = _load(WORKFLOW_DIR / CI_WORKFLOW)["jobs"]
    needs = set(jobs["summary"]["needs"])
    for required in ("test", "chess-tests", "integration-test", "security-scan", "dependency-audit"):
        assert required in needs, f"ci.yml summary must depend on the {required!r} job"


# --------------------------------------------------------------------------- AC-3


@pytest.mark.unit
@pytest.mark.parametrize(("workflow", "job_id", "job"), _all_jobs(), ids=_job_ids())
def test_no_result_bearing_step_swallows_its_exit_code(workflow: str, job_id: str, job: dict[str, Any]) -> None:
    """AC-3: a check that cannot fail is not a check.

    Narrowly scoped to commands whose exit code *is* the signal, matched on word
    boundaries. Cleanup and best-effort reporting steps legitimately use ``|| true``
    (``rm -rf .pytest_cache || true``) and must not be flagged just because a path
    happens to contain a command name.
    """
    for step in _steps(job):
        run = step.get("run")
        if not run:
            continue

        # `continue-on-error` disarms the whole step regardless of what the shell does,
        # so it is the same defect as `|| true` wearing a different hat.
        assert not (step.get("continue-on-error") and _COMMAND_WORD_RE.search(run)), (
            f"{workflow}::{job_id} step {step.get('name')!r} runs a result-bearing "
            f"command under continue-on-error, so its result cannot fail the job."
        )

        offenders = _disarmed_commands(run)
        assert not offenders, (
            f"{workflow}::{job_id} step {step.get('name')!r} runs a result-bearing "
            f"command but discards its exit status: {offenders}. Remove the "
            f"`|| true` / `|| :` / `set +e`, or gate the whole job on a precondition "
            f"instead."
        )


# -------------------------------------------------------------------------- AC-12


@pytest.mark.unit
def test_image_scan_can_actually_fail() -> None:
    """AC-12: at least one image scan must be able to block.

    The single Trivy step carried *both* ``continue-on-error: true`` and
    ``exit-code: '0'`` — it could not fail anything while still costing ~3 minutes a
    run. The split is intentional: an advisory scan feeds the Security tab and may
    flake, while a separate blocking scan gates on CRITICAL.
    """
    docker_build = _load(WORKFLOW_DIR / CI_WORKFLOW)["jobs"]["docker-build"]
    scans = [s for s in _steps(docker_build) if "trivy-action" in str(s.get("uses", ""))]
    assert scans, "expected at least one Trivy scan step in the docker-build job"

    def blocking(step: dict[str, Any]) -> bool:
        """True when this scan step can actually fail its job.

        Both conditions are required: a non-zero ``exit-code`` makes the scanner
        report findings as failure, and the absence of ``continue-on-error`` lets
        that failure reach the job.
        """
        with_ = step.get("with") or {}
        return str(with_.get("exit-code", "0")) != "0" and not step.get("continue-on-error", False)

    assert any(blocking(s) for s in scans), (
        "no Trivy step can fail the build: every scan is either continue-on-error or "
        "exit-code 0. An advisory-only scanner is a permanently-ignored job."
    )


# -------------------------------------------------------------------------- AC-11


@pytest.mark.unit
def test_coverage_exclude_patterns_are_anchored() -> None:
    """AC-11: ``exclude_lines`` patterns are regexes matched with ``re.search``.

    An unanchored bare keyword silently removes any line *containing* it from the
    coverage denominator. ``pass`` matched 359 lines in ``src/``, 291 of which were
    docstrings ("forward pass") or dataclass fields (``num_passed``). That is the
    coverage gate moving to meet the code, which CHARTER.md NG-5 forbids.
    """
    patterns = _load_pyproject()["tool"]["coverage"]["report"]["exclude_lines"]

    # Bare single-word patterns that are also common English/identifier substrings.
    risky = {"pass", "continue", "break", "raise", "return", "..."}
    for pattern in patterns:
        stripped = pattern.strip()
        if stripped in risky:
            pytest.fail(
                f"coverage exclude_lines contains the unanchored pattern {pattern!r}. "
                f"It is applied with re.search, so it matches any line containing that "
                f"substring and silently shrinks the coverage denominator. Anchor it, "
                f"e.g. '^\\\\s*{stripped}\\\\s*$'."
            )


@pytest.mark.unit
def test_every_module_omitted_from_the_main_gate_is_measured_somewhere() -> None:
    """ui_test_coverage AC-6: a module the main gate omits must be measured by some job.

    ``[tool.coverage.run] omit`` drops ``src/games/chess/ui.py`` for a sound reason —
    the coverage-gated job installs no ``python-chess``, so the module would score 0%.
    The cost was a blind spot: the module is exercised by ~90 tests in ``ui-tests``,
    which measured nothing, so a launch-blocking defect could land inside a
    covered-looking repo. That is exactly how a TypeError in that file reached main.

    ``ui-tests`` now carries the gate via ``.coveragerc.ui``. This test fails if the
    flag is dropped, the config's scope stops covering an omitted module, or its
    threshold is quietly lowered — each of which restores the blind spot silently.

    The two ``verification/`` modules are knowingly still unmeasured; they are listed
    here so removing that exemption is a deliberate edit rather than an oversight.
    """
    known_unmeasured = {
        "src/games/chess/verification/game_verifier.py",
        "src/games/chess/verification/move_validator.py",
    }

    omitted = {
        entry
        for entry in _load_pyproject()["tool"]["coverage"]["run"]["omit"]
        if entry.startswith("src/") and entry.endswith(".py")
    }
    needing_a_gate = omitted - known_unmeasured
    assert needing_a_gate, "expected at least one omitted src module to be gated elsewhere"

    ui_job = _load(WORKFLOW_DIR / CI_WORKFLOW)["jobs"]["ui-tests"]
    runs = " ".join(str(step.get("run", "")) for step in _steps(ui_job))
    match = re.search(r"--cov-config=(\S+)", runs)
    assert match, (
        "the ui-tests job no longer passes --cov-config, so the modules the main "
        f"gate omits ({sorted(needing_a_gate)}) are measured by no job at all."
    )

    config_path = Path(__file__).resolve().parents[2] / match.group(1)
    assert config_path.is_file(), f"ui-tests references a missing coverage config: {match.group(1)}"

    # Parsed, not substring-matched: this file explains itself in prose that names the
    # very modules being checked, so `module in text` passes on the comment alone.
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")
    scope = parser.get("run", "include", fallback="") + "\n" + parser.get("run", "source", fallback="")
    scope_entries = [line.strip() for line in scope.splitlines() if line.strip()]

    for module in sorted(needing_a_gate):
        covered = any(fnmatch(module, entry) or fnmatch(f"/{module}", entry) for entry in scope_entries)
        assert covered, (
            f"{module} is omitted from the main coverage gate but matches no entry in "
            f"{match.group(1)}'s [run] scope ({scope_entries}), so nothing measures it. "
            f"Add it there, or accept the gap explicitly in known_unmeasured."
        )

    # The gate must be able to fail, and at no less than the main gate's threshold.
    main_threshold = float(_load_pyproject()["tool"]["coverage"]["report"]["fail_under"])
    ui_threshold = parser.get("report", "fail_under", fallback="")
    assert ui_threshold, f"{match.group(1)} sets no fail_under, so its coverage report cannot fail"
    assert float(ui_threshold) >= main_threshold, (
        f"{match.group(1)} gates at {ui_threshold}%, below the main gate's "
        f"{main_threshold}%. The omitted modules would be held to a weaker standard "
        f"than the code that is measured normally."
    )


# ---------------------------------------------------------------- structural sanity


@pytest.mark.unit
@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_every_needs_reference_resolves_to_a_real_job(path: Path) -> None:
    """A ``needs:`` entry naming a job that does not exist is a whole-workflow error.

    GitHub rejects the run rather than the job, so a single typo takes the entire
    workflow offline — and because nothing else in the repository parses these files,
    it would only be discovered on push. Cheap to check here.
    """
    jobs = _load(path).get("jobs") or {}
    known = set(jobs)
    for job_id, job in jobs.items():
        needs = job.get("needs") or []
        if isinstance(needs, str):
            needs = [needs]
        unknown = [n for n in needs if n not in known]
        assert not unknown, (
            f"{path.name}::{job_id} declares needs={unknown}, which name no job in "
            f"this workflow. Known jobs: {sorted(known)}."
        )


@pytest.mark.unit
@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_no_job_depends_on_itself(path: Path) -> None:
    """A self-referential ``needs:`` deadlocks the workflow."""
    for job_id, job in (_load(path).get("jobs") or {}).items():
        needs = job.get("needs") or []
        if isinstance(needs, str):
            needs = [needs]
        assert job_id not in needs, f"{path.name}::{job_id} lists itself in needs"


# --------------------------------------------------------------------------- AC-5


@pytest.mark.unit
def test_test_job_installs_the_extras_its_suites_require() -> None:
    """AC-5: the CI test job must install every extra ``tests/conftest.py`` demands.

    ``tests/conftest.py`` hard-fails collection under CI when an optional dependency
    is missing, rather than silently shrinking the suite. That is only an improvement
    if the workflow actually installs those extras — otherwise the strict guard turns
    a silent gap into a red build for the wrong reason.

    This couples the two files deliberately: dropping ``api`` from the install line is
    exactly how the API-server suites went ungated before, and nothing detected it.
    """
    conftest = (Path(__file__).resolve().parents[1] / "conftest.py").read_text(encoding="utf-8")
    required = set(re.findall(r'_require_or_ignore\(\s*"[^"]+"\s*,\s*"([^"]+)"', conftest))
    assert required, "expected tests/conftest.py to declare optional-dependency extras"

    test_job = _load(WORKFLOW_DIR / CI_WORKFLOW)["jobs"]["test"]
    install = " ".join(str(s.get("run", "")) for s in _steps(test_job) if "pip install" in str(s.get("run", "")))
    match = re.search(r'pip install\s+-e\s+"?\.\[([^\]]+)\]', install)
    assert match, f"could not find a `pip install -e .[...]` line in the test job; got: {install!r}"

    installed = {extra.strip() for extra in match.group(1).split(",")}
    missing = required - installed
    assert not missing, (
        f"tests/conftest.py requires the {sorted(missing)} extra(s) under CI, but the "
        f"ci.yml test job installs only {sorted(installed)}. Collection will abort. "
        f"Add the extra to the install line, or relax the guard in conftest.py."
    )


@pytest.mark.unit
def test_suppressed_api_suites_are_no_longer_ignored() -> None:
    """AC-5: the three suppression layers must not reappear.

    ``ci.yml`` --ignore flags, the conftest ``collect_ignore_glob``, and the coverage
    ``omit`` list previously hid the same three modules in three different places, so
    removing any one of them alone did nothing visible.

    All three layers are checked here. The first version of this test named three and
    checked two, leaving the conftest layer — the one it calls "protected" — free to
    re-suppress the suites on its own.
    """
    # Layer 1: ci.yml --ignore. Matched semantically, because pytest accepts both
    # `--ignore=path` and `--ignore path`, and only the `=` spelling was rejected.
    ci_text = (WORKFLOW_DIR / CI_WORKFLOW).read_text(encoding="utf-8")
    ignored = {Path(arg).name for arg in _IGNORE_ARG_RE.findall(ci_text)}
    clashes = ignored & set(PROTECTED_API_SUITES)
    assert not clashes, (
        f"ci.yml passes --ignore for {sorted(clashes)}. Those suites now collect and "
        f"pass; ignoring them only hides measured code."
    )

    # Layer 2: the conftest collect_ignore_glob.
    suppressed = {Path(entry).name for entry in _literal_collect_ignores()}
    clashes = suppressed & set(PROTECTED_API_SUITES)
    assert not clashes, (
        f"tests/conftest.py adds {sorted(clashes)} to collect_ignore_glob via a "
        f"literal list rather than the _require_or_ignore() guard, so it is "
        f"suppressed even when the extra is installed. That is the silent-shrink "
        f"defect AC-5 removed."
    )

    omit = _load_pyproject()["tool"]["coverage"]["run"]["omit"]
    for module in ("src/api/rest_server.py", "src/api/inference_server.py"):
        assert module not in omit, (
            f"{module} is back in the coverage omit list. Its test suite now collects "
            f"and passes, so omitting it only hides measured code — CHARTER.md NG-5."
        )


# ------------------------------------------------------------------ Makefile parity


def _makefile_text() -> str:
    """The Makefile source, read fresh so parity tests see the tree's current state."""
    return (REPO_ROOT / "Makefile").read_text(encoding="utf-8")


def _make_recipe(target: str) -> str:
    """Return the recipe body (the tab-indented lines) of a Makefile target.

    Deliberately excludes the target line itself, so a flag mentioned in the ``##``
    help text — e.g. "NOT --strict" — is not mistaken for a flag actually passed to
    the command. The first draft of the parity test below made exactly that mistake.
    """
    lines = _makefile_text().splitlines()
    body: list[str] = []
    collecting = False
    for line in lines:
        if re.match(rf"^{re.escape(target)}:", line):
            collecting = True
            continue
        if collecting:
            if line.startswith("\t"):
                body.append(line.lstrip("\t"))
            elif line.strip() == "":
                continue
            else:
                break
    return "\n".join(body)


@pytest.mark.unit
def test_makefile_exists_and_documents_its_targets() -> None:
    """The Makefile is the developer entry point; every target carries help text.

    Undocumented targets are how a Makefile becomes write-only.
    """
    text = _makefile_text()
    declared = set(re.findall(r"^\.PHONY:\s*(.*(?:\\\n.*)*)", text, re.M))
    phony = {t for block in declared for t in block.replace("\\", " ").split()}
    documented = set(re.findall(r"^([a-zA-Z_-]+):.*?## ", text, re.M))
    undocumented = phony - documented - {"help"}
    assert not undocumented, f"Makefile targets without `## ` help text: {sorted(undocumented)}"


@pytest.mark.unit
def test_makefile_gate_matches_ci_flags() -> None:
    """The Makefile must not drift from CI.

    A Makefile that quietly disagrees with the workflow is worse than no Makefile:
    it produces local green on a weaker check than the one that gates merges. These
    are the flags where a mismatch would actually change the verdict.
    """
    text = _makefile_text()
    ci_text = (WORKFLOW_DIR / CI_WORKFLOW).read_text(encoding="utf-8")

    # Line length: whatever CI checks black against, the Makefile must use.
    ci_line_length = re.search(r"black \. --check --line-length (\d+)", ci_text)
    assert ci_line_length, "could not find black's --line-length in ci.yml"
    assert f"LINE_LENGTH ?= {ci_line_length.group(1)}" in text, (
        f"Makefile LINE_LENGTH must equal ci.yml's black --line-length " f"({ci_line_length.group(1)})"
    )

    # Coverage floor: must equal the workflow's --cov-fail-under, which in turn is
    # protected from being lowered by CHARTER.md NG-5.
    ci_cov = re.search(r"--cov-fail-under=(\d+)", ci_text)
    assert ci_cov, "could not find --cov-fail-under in ci.yml"
    assert (
        f"COV_MIN     ?= {ci_cov.group(1)}" in text
    ), f"Makefile COV_MIN must equal ci.yml's --cov-fail-under ({ci_cov.group(1)})"

    # Extras: the Makefile's default install must cover what the test job installs,
    # otherwise `make test` collects a different set of tests than CI does.
    ci_extras = re.search(r'pip install -e "\.\[([^\]]+)\]"[\s\S]*?Run unit tests', ci_text)
    assert ci_extras, "could not find the test job's pip install line in ci.yml"
    required = {e.strip() for e in ci_extras.group(1).split(",")}
    mk_extras = re.search(r"EXTRAS      \?= (.+)", text)
    assert mk_extras, "Makefile does not define EXTRAS"
    have = {e.strip() for e in mk_extras.group(1).split(",")}
    assert required <= have, (
        f"Makefile EXTRAS {sorted(have)} is missing {sorted(required - have)}, which "
        f"the ci.yml test job installs. `make test` would collect fewer tests than CI."
    )

    # `make typecheck` must not silently be stricter or looser than the CI gate.
    # Checked against the recipe body only — the help text legitimately mentions
    # --strict in order to explain why it is NOT used.
    typecheck = _make_recipe("typecheck")
    assert "mypy src/" in typecheck, "Makefile `typecheck` must run `mypy src/`"
    assert "--strict" not in typecheck, (
        "Makefile typecheck must match CI's `mypy src/`; --strict reports 545 errors "
        "and is a separately-tracked ratchet (see CLAUDE.md)."
    )


# ------------------------------------------------------- the helpers' own invariants
#
# Everything above rests on the parsers in this module. That makes them a single point
# of silent failure: if `_gated_job_list` ever returned every name it saw, or
# `_disarmed_commands` returned nothing, the invariants would keep passing while
# enforcing nothing — the exact "green but not checked" defect they exist to prevent.
#
# The live workflows cannot exercise these paths, and that is the point: no workflow
# uses `set +e` (the invariant forbids it), so the branch that detects it is
# unreachable from real inputs and would otherwise be verified only by a mutation run
# that leaves nothing behind. The cases below pin that behaviour in the tree.


@pytest.mark.unit
def test_named_workflows_resolve() -> None:
    """CI_WORKFLOW / EXPENSIVE_WORKFLOW must name files that exist.

    Renaming a workflow otherwise surfaces as a confusing ``KeyError`` inside an
    unrelated assertion, several tests deep.
    """
    names = {p.name for p in _workflow_files()}
    for constant, value in (("CI_WORKFLOW", CI_WORKFLOW), ("EXPENSIVE_WORKFLOW", EXPENSIVE_WORKFLOW)):
        assert value in names, f"{constant}={value!r} names no workflow. Present: {sorted(names)}."


@pytest.mark.unit
@pytest.mark.parametrize(
    ("label", "run", "expected"),
    [
        ("plain command is armed", "pytest tests/unit/", []),
        ("|| true disarms", "pytest tests/unit/ || true", ["pytest tests/unit/ || true"]),
        ("|| : disarms", "mypy src/ || :", ["mypy src/ || :"]),
        ("|| : mid-line disarms", "mypy src/ || : # tolerated", ["mypy src/ || : # tolerated"]),
        # Word-boundary guards: each of these contains a command name as a substring
        # only, and flagging them would train people to ignore this test.
        ("cache path is not pytest", "rm -rf .pytest_cache || true", []),
        ("container name is not black", "docker rm blackbox || true", []),
        ("report filename is not pytest", "curl -F f=@pytest-report.xml || true", []),
        ("comment is not a command", "# pytest tests/ || true", []),
        # A wrapped command and its disarming tail are one logical line.
        (
            "backslash continuation",
            "pytest tests/unit/ \\\n  --tb=short || true",
            ["pytest tests/unit/  --tb=short || true"],
        ),
        # set +e: disarms every LATER command, not the line it sits on.
        ("set +e disarms what follows", "set +e\npytest tests/unit/", ["pytest tests/unit/"]),
        # Inline separators: the state change and the command it disarms share a line,
        # so treating the line as one unit let the `set +e` branch swallow both.
        ("inline ; after set +e", "set +e; pytest tests/unit/", ["pytest tests/unit/"]),
        ("inline && after set +e", "set +e && pytest tests/unit/", ["pytest tests/unit/"]),
        ("inline ; with later re-arm", "set +e; mypy src/; set -e; pytest a/", ["mypy src/"]),
        # `||` must NOT split, or `pytest … || true` stops being one matchable unit.
        ("|| is not a separator", "pytest a/ || true", ["pytest a/ || true"]),
        ("set +e leaves earlier lines armed", "pytest a/\nset +e\npytest b/", ["pytest b/"]),
        ("set -e re-arms", "set +e\nset -e\npytest tests/unit/", []),
        ("bracketed tolerant region", "pytest a/\nset +e\nmypy src/\nset -e\npytest b/", ["mypy src/"]),
        # Long-option spellings are equivalent to the short flags and must behave the
        # same; a regex for `+e` alone silently missed `+o errexit` entirely.
        ("set -o errexit re-arms", "set +e\nset -o errexit\npytest a/", []),
        ("set +o errexit disarms", "set +o errexit\npytest a/", ["pytest a/"]),
        ("set -o pipefail does not re-arm", "set +e\nset -o pipefail\npytest a/", ["pytest a/"]),
        ("set -x does not re-arm", "set +e\nset -x\npytest a/", ["pytest a/"]),
        ("set -euo pipefail re-arms", "set +e\nset -euo pipefail\npytest a/", []),
        ("set -eu counts as re-arming", "set +e\nset -eu\npytest a/", []),
        ("set +ex counts as disarming", "set +ex\npytest a/", ["pytest a/"]),
        # `offset` is a word containing "set" — the lookbehind must not match it.
        ("word containing set is not set", "offset=+e\npytest a/", []),
    ],
)
def test_disarmed_commands_detects_each_shape(label: str, run: str, expected: list[str]) -> None:
    """``_disarmed_commands`` must catch both disarming mechanisms and neither false-positive."""
    assert _disarmed_commands(run) == expected, label


@pytest.mark.unit
@pytest.mark.parametrize(
    ("label", "body", "expected"),
    [
        ("JOBS list", 'JOBS="LINT TESTS"\nfor j in ${JOBS}; do :; done', {"LINT", "TESTS"}),
        ("JOBS continuation", 'JOBS="LINT \\\n      TESTS"', {"LINT", "TESTS"}),
        ("single-bracket test", '[ "${CHECK_SECRETS}" != "success" ]', {"CHECK_SECRETS"}),
        ("double-bracket test", '[[ "${CHECK}" != "success" ]]', {"CHECK"}),
        ("case statement", 'case "${HAS_KEY}" in', {"HAS_KEY"}),
        ("both forms combine", 'JOBS="A B"\ncase "${C}" in', {"A", "B", "C"}),
        # The regression mutation testing surfaced: a name appearing only inside a
        # comment must NOT count as gated, or the summary passes while enforcing less
        # than it prints. Both shapes below yield the wrong answer if the comment
        # stripping is removed, which a comment carrying no gating construct does not.
        ("commented gate does not count", '# case "${TESTS}" in\nJOBS="LINT"', {"LINT"}),
        ("commented-out JOBS does not count", '# JOBS="A B C"\nJOBS="LINT"', {"LINT"}),
        ("commented bracket test does not count", '# [ "${OLD}" != "success" ]\nJOBS="LINT"', {"LINT"}),
        ("no gate at all", 'echo "nothing here"', set()),
    ],
)
def test_gated_job_list_counts_only_real_gates(label: str, body: str, expected: set[str]) -> None:
    """``_gated_job_list`` decides whether a summary enforces what it prints."""
    assert _gated_job_list(body) == expected, label


@pytest.mark.unit
def test_triggers_handles_the_yaml_boolean_on_key() -> None:
    """``on:`` parses as the boolean ``True`` under YAML 1.1, which PyYAML follows.

    Both spellings must work, so quoting the key in a workflow does not silently
    empty every trigger-based assertion.
    """
    assert _triggers({True: {"push": {}}}) == {"push": {}}
    assert _triggers({"on": {"push": {}}}) == {"push": {}}
    assert _triggers({}) == {}


@pytest.mark.unit
def test_steps_skips_malformed_entries() -> None:
    """A non-mapping step is skipped rather than crashing every step-walking test."""
    assert _steps({"steps": [{"run": "a"}, "not-a-step", None, {"uses": "b"}]}) == [{"run": "a"}, {"uses": "b"}]
    assert _steps({}) == []
    assert _steps({"steps": None}) == []


@pytest.mark.unit
def test_make_recipe_extracts_only_the_recipe_body() -> None:
    """``_make_recipe`` must stop at the next target and exclude the ``##`` help text.

    The first draft of the Makefile parity test matched ``--strict`` inside a help
    comment that exists precisely to say ``--strict`` is NOT used.
    """
    assert "mypy src/" in _make_recipe("typecheck")
    # The help text lives on the target line, which the recipe body excludes.
    assert "##" not in _make_recipe("typecheck")
    assert _make_recipe("no-such-target-xyz") == ""


@pytest.mark.unit
@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        ("equals form", "pytest --ignore=tests/unit/test_rest_server.py", {"tests/unit/test_rest_server.py"}),
        ("space form", "pytest --ignore tests/unit/test_rest_server.py", {"tests/unit/test_rest_server.py"}),
        ("both forms", "pytest --ignore=a.py --ignore b.py", {"a.py", "b.py"}),
        ("multiline", "pytest \\\n  --ignore=a.py \\\n  --ignore b.py", {"a.py", "b.py"}),
        ("none present", "pytest tests/unit/ --cov=src", set()),
    ],
)
def test_ignore_arg_parsing_covers_both_spellings(label: str, text: str, expected: set[str]) -> None:
    """AC-5: pytest accepts ``--ignore=path`` and ``--ignore path``.

    The suppression invariant reads whatever this finds. Matching only the ``=``
    spelling would let the space form quietly put a suite back out of collection —
    and because ``ci.yml`` currently carries no ``--ignore`` at all, that assertion
    passes vacuously either way. This is what actually pins the parsing.
    """
    assert set(_IGNORE_ARG_RE.findall(text)) == expected, label


@pytest.mark.unit
def test_literal_collect_ignores_reads_the_real_guards() -> None:
    """AC-5: the conftest layer is parsed, not assumed empty.

    A parser that silently returned nothing would make the suppression check pass no
    matter what conftest did. conftest has three genuine literal guards (chess,
    hypothesis, gradio); finding them proves the AST walk actually reaches the
    ``collect_ignore_glob`` additions, and none of them is an API suite.
    """
    found = _literal_collect_ignores()
    assert found, "parsed no collect_ignore_glob entries at all — the AST walk is not reaching them"
    # The optional-dependency guards conftest really declares, by literal list.
    assert "unit/test_chess_ensemble_checker.py" in found
    assert "ui/test_gradio_app.py" in found
    # `_require_or_ignore` appends through a variable, so its API-suite entries must
    # NOT appear here — that distinction is the whole point of the helper.
    assert not {Path(f).name for f in found} & set(PROTECTED_API_SUITES)


# ============================================================================
# Supply chain and privilege invariants
# ============================================================================
# Spec: specs/evidence_claim_ledger.SPEC.md AC-8.
#
# Every check below corresponds to a finding made while auditing this tree, not to a
# generic hardening checklist. They are grouped here because they share one theme: what
# a pull request from outside the repository is allowed to reach.


def _job_pull_request_events(workflow: dict[str, Any], job: dict[str, Any]) -> bool:
    """Whether ``job`` can execute on a pull-request event.

    Two things have to be true: the workflow must be triggered by a PR event at all, and the
    job's own ``if:`` must not exclude PR events. The second half matters — ``ci.yml``'s
    ``rag-eval`` job carries provider keys but is gated to ``workflow_dispatch``/``schedule``,
    and flagging it would be a false positive that trains people to ignore this test.
    """
    triggers = _triggers(workflow)
    pr_triggers = {"pull_request", "pull_request_target"} & set(triggers)
    if not pr_triggers:
        return False

    condition = str(job.get("if", ""))
    if not condition:
        return True

    # An explicit exclusion settles it: `github.event_name != 'pull_request'`.
    if re.search(r"github\.event_name\s*!=\s*'pull_request(_target)?'", condition):
        return False

    # An allow-list of events that does not include a PR event also settles it. This is how
    # ci.yml's rag-eval job restricts itself to workflow_dispatch and schedule.
    allowed_events = set(re.findall(r"github\.event_name\s*==\s*'([a-z_]+)'", condition))
    if allowed_events:
        return bool(allowed_events & pr_triggers)

    # Anything else (a condition about branches, needs outputs, labels, ...) does not restrict
    # the event, so the job must be assumed reachable from a pull request.
    return True


def _secret_references(node: Any) -> list[str]:
    """Every ``secrets.NAME`` referenced anywhere inside a job definition, at any depth."""
    return re.findall(r"secrets\.([A-Z0-9_]+)", json.dumps(node))


@pytest.mark.unit
@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_no_workflow_uses_pull_request_target(path: Path) -> None:
    """AC-8: ``pull_request_target`` runs with a *writable* token in the base-repo context.

    Combined with a checkout of the PR head — the pattern people reach for to make it useful —
    it executes untrusted code with write access to the repository. This tree does not use it
    today; this test is what keeps it that way, because the trigger is frequently suggested as
    the fix for "secrets are not available on fork PRs".
    """
    triggers = _triggers(_load(path))
    assert "pull_request_target" not in triggers, (
        f"{path.name} uses pull_request_target, which grants a writable token to a workflow "
        "evaluated in the base-repo context. Use `pull_request` and accept that forks have no "
        "secrets, or move the privileged half into a separate workflow_run workflow."
    )


@pytest.mark.unit
@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_every_workflow_declares_top_level_permissions(path: Path) -> None:
    """AC-8: without a ``permissions:`` block, a job inherits the repository default.

    On older repositories that default is write-all, so any compromised or merely careless
    action step can push commits, publish packages, or edit releases. Declaring the block at the
    top makes the grant explicit and makes each job-level widening visible in review.
    """
    workflow = _load(path)
    permissions = workflow.get("permissions")
    assert permissions is not None, (
        f"{path.name} declares no top-level `permissions:` block, so its jobs inherit the "
        "repository default token scope. Add `permissions: {contents: read}` and grant more "
        "only on the specific jobs that need it."
    )
    assert permissions != "write-all", f"{path.name} grants write-all at the top level"


@pytest.mark.unit
def test_top_level_permissions_are_read_only() -> None:
    """AC-8: the *default* grant is read-only; elevation is per job.

    Asserted across all workflows at once so the failure message can list every offender rather
    than surfacing them one parametrized case per run.
    """
    offenders: list[str] = []
    for path in _workflow_files():
        permissions = _load(path).get("permissions")
        if not isinstance(permissions, dict):
            continue
        for scope, level in permissions.items():
            if level not in ("read", "none"):
                offenders.append(f"{path.name}: {scope}: {level}")
    assert not offenders, (
        "top-level permissions must be read-only; move write scopes onto the individual jobs "
        f"that need them: {offenders}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("wf,job_id,job", _all_jobs(), ids=_job_ids())
def test_pull_request_jobs_receive_no_provider_credentials(wf: str, job_id: str, job: dict[str, Any]) -> None:
    """AC-8: a job that can run PR code must not hold paid provider credentials.

    This is the concrete exfiltration path, not a hypothetical one: a job that both checks out
    the pull request's code and exports ``OPENAI_API_KEY`` can be drained by a PR that adds one
    step printing its environment. ``e2e_with_langsmith.yml`` was exactly this shape, which is
    why its ``pull_request`` trigger was removed rather than its jobs re-gated one by one.
    """
    workflow = _load(WORKFLOW_DIR / wf)
    if not _job_pull_request_events(workflow, job):
        return

    exposed = sorted(set(_secret_references(job)) & set(LLM_PROVIDER_CREDENTIAL_ENV_VARS))
    assert not exposed, (
        f"{wf}::{job_id} can run on a pull_request event and references {exposed}. Code proposed "
        "in a pull request must never be handed paid provider credentials. Either gate the job on "
        "`github.event_name != 'pull_request'`, or drop the workflow's pull_request trigger."
    )


@pytest.mark.unit
def test_action_pin_ratchet_holds() -> None:
    """AC-8: unpinned action references may only decrease.

    A tag such as ``@v4`` is mutable, so a tag-pinned third-party action is an unreviewed
    code-execution path into CI. The tree is not fully pinned yet, so rather than a gate that
    would have to ship disabled, the committed baseline records where it stands and this test
    forbids the number growing. See ``src/tools/action_pins.py`` for the rules.
    """
    report = pin_audit(REPO_ROOT)
    assert report.ok, "action pin ratchet violated:\n" + "\n".join(f"  - {v}" for v in report.violations)


@pytest.mark.unit
def test_action_pin_baseline_covers_every_action_in_use() -> None:
    """AC-8: the baseline is complete, so a newly introduced action cannot slip in unpinned.

    Without this, an action absent from the baseline would be caught only by the ratchet's
    unknown-action rule; asserting completeness here makes the intent explicit and gives a
    clearer failure than a generic violation string.
    """
    report = pin_audit(REPO_ROOT)
    uncovered = sorted(set(report.counts) - set(report.baseline))
    assert (
        not uncovered
    ), f"these actions are used without a commit-SHA pin and are absent from the baseline: {uncovered}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "label,triggers,condition,expected",
    [
        ("no PR trigger at all", {"push": None}, "", False),
        ("PR trigger, unconditional job", {"pull_request": None}, "", True),
        ("PR trigger, explicitly excluded", {"pull_request": None}, "github.event_name != 'pull_request'", False),
        (
            "PR trigger, allow-list of other events",
            {"pull_request": None},
            "github.event_name == 'workflow_dispatch' || github.event_name == 'schedule'",
            False,
        ),
        (
            "PR trigger, allow-list that includes it",
            {"pull_request": None},
            "github.event_name == 'pull_request'",
            True,
        ),
        (
            "PR trigger, condition unrelated to the event",
            {"pull_request": None},
            "needs.check-secrets.outputs.has_langsmith == 'true'",
            True,
        ),
        ("PR trigger, branch condition only", {"pull_request": None}, "github.ref == 'refs/heads/main'", True),
        ("pull_request_target counts as a PR trigger", {"pull_request_target": None}, "", True),
    ],
)
def test_pull_request_reachability_helper(label: str, triggers: dict, condition: str, expected: bool) -> None:
    """AC-8: the credential invariant is only as good as its reachability analysis.

    Pinned with a table because the interesting cases are the *exemptions*: if this helper were
    too eager, ci.yml's dispatch-gated rag-eval job would fail the credential check and the test
    would get deleted; if it were too lax, the check would pass vacuously. Both failure modes are
    silent without this table.
    """
    job: dict[str, Any] = {"if": condition} if condition else {}
    assert _job_pull_request_events({"on": triggers}, job) is expected, label


@pytest.mark.unit
def test_credential_invariant_would_catch_a_reintroduced_exposure() -> None:
    """AC-8: falsifies the credential check rather than trusting that it passes.

    Reconstructs the shape ``e2e_with_langsmith.yml`` had before this change — a PR-triggered
    workflow whose job exports ``OPENAI_API_KEY`` — and asserts the analysis flags it. Without
    this, a helper that silently returned "not reachable" would make the real test green forever.
    """
    workflow = {"on": {"pull_request": {"branches": ["main"]}}}
    job = {"steps": [{"name": "run", "env": {"OPENAI_API_KEY": "${{ secrets.OPENAI_API_KEY }}"}}]}

    assert _job_pull_request_events(workflow, job) is True
    exposed = set(_secret_references(job)) & set(LLM_PROVIDER_CREDENTIAL_ENV_VARS)
    assert exposed == {"OPENAI_API_KEY"}


@pytest.mark.unit
def test_secret_reference_scan_reaches_nested_definitions() -> None:
    """AC-8: secrets appear in step env, job env, and `with:` inputs — all must be visible."""
    job = {
        "env": {"A": "${{ secrets.WANDB_API_KEY }}"},
        "steps": [
            {"with": {"token": "${{ secrets.ANTHROPIC_API_KEY }}"}},
            {"env": {"B": "${{ secrets.CODECOV_TOKEN }}"}},
        ],
    }
    assert set(_secret_references(job)) == {"WANDB_API_KEY", "ANTHROPIC_API_KEY", "CODECOV_TOKEN"}


@pytest.mark.unit
def test_e2e_langsmith_workflow_has_no_pull_request_trigger() -> None:
    """AC-8: pins the specific fix, so a well-meaning "restore PR coverage" edit fails loudly.

    The generic credential invariant above would also catch it, but only while the job keeps
    referencing a provider key. This asserts the decision itself.
    """
    triggers = _triggers(_load(WORKFLOW_DIR / "e2e_with_langsmith.yml"))
    assert "pull_request" not in triggers
    assert "pull_request_target" not in triggers
    assert "push" in triggers, "removing the PR trigger must not leave the suite with no automatic trigger"
