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
Each section below is headed with the AC it enforces.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

# Imported normally, not via importorskip. PyYAML is a declared `dev` dependency, and a
# regression gate that silently skips itself when a dependency goes missing is exactly
# the failure mode this module exists to catch.
import yaml

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"

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
)

# Shell constructs that discard a command's exit status. `|| true` is the obvious one;
# `|| :` is the same thing spelled with the null builtin, and `set +e` disables the
# errexit that would otherwise propagate a failure out of a multi-command step.
_DISARMING_SUFFIXES = (r"\|\|\s*true\b", r"\|\|\s*:\s*$", r"\|\|\s*:\s")
_DISARMING_RE = re.compile("|".join(_DISARMING_SUFFIXES))

# A command name only counts when it appears as a *word*. Substring matching flagged
# `rm -rf .pytest_cache || true` (contains "pytest"), `docker rm blackbox || true`
# (contains "black") and `curl -F file=@pytest-report.xml || true` — all legitimate
# cleanup or upload steps. Word boundaries with an explicit guard against `-`/`.`/`_`
# neighbours fix that without needing an allowlist.
_COMMAND_WORD_RE = re.compile(
    r"(?<![\w./-])(?:" + "|".join(re.escape(c) for c in RESULT_BEARING_COMMANDS) + r")(?![\w.-])"
)


def _workflow_files() -> list[Path]:
    files = sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))
    assert files, f"no workflow files found under {WORKFLOW_DIR}"
    return files


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return the ``on:`` block.

    PyYAML follows YAML 1.1, where the bare key ``on`` is parsed as the boolean
    ``True`` rather than the string ``"on"``. Both spellings are checked so this
    helper keeps working if the workflows ever quote the key.
    """
    return workflow.get("on") or workflow.get(True) or {}


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in job.get("steps", []) or [] if isinstance(s, dict)]


def _all_jobs() -> list[tuple[str, str, dict[str, Any]]]:
    """Every (workflow filename, job id, job body) triple across all workflows."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for path in _workflow_files():
        for job_id, job in (_load(path).get("jobs") or {}).items():
            out.append((path.name, job_id, job))
    return out


def _job_ids() -> list[str]:
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


@pytest.mark.unit
def test_expensive_workflow_filters_pull_requests_by_path() -> None:
    """AC-9: the Docker workflow must not run its image matrix on docs-only PRs.

    The ``push`` trigger has always carried a ``paths`` filter; the ``pull_request``
    trigger did not, so a two-file Markdown PR ran the full ``Dockerfile.train``
    matrix — the single most expensive job in the repository.
    """
    triggers = _triggers(_load(WORKFLOW_DIR / "docker-deployment.yml"))
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

    # 3. The gate must actually be able to fail.
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
    jobs = _load(WORKFLOW_DIR / "ci.yml")["jobs"]
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

        offenders: list[str] = []
        # Join backslash continuations first: a `|| true` on the wrapped tail of a
        # command still disarms it, and the two belong to the same logical line.
        logical = re.sub(r"\\\s*\n\s*", " ", run)
        for line in logical.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue  # a comment mentioning `|| true` is not a disarmed command
            if _DISARMING_RE.search(stripped) and _COMMAND_WORD_RE.search(stripped):
                offenders.append(stripped)

        assert not offenders, (
            f"{workflow}::{job_id} step {step.get('name')!r} runs a result-bearing "
            f"command but discards its exit status: {offenders}. Remove the "
            f"`|| true` / `|| :`, or gate the whole job on a precondition instead."
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
    docker_build = _load(WORKFLOW_DIR / "ci.yml")["jobs"]["docker-build"]
    scans = [s for s in _steps(docker_build) if "trivy-action" in str(s.get("uses", ""))]
    assert scans, "expected at least one Trivy scan step in the docker-build job"

    def blocking(step: dict[str, Any]) -> bool:
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
    import tomllib

    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as fh:
        config = tomllib.load(fh)
    patterns = config["tool"]["coverage"]["report"]["exclude_lines"]

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

    test_job = _load(WORKFLOW_DIR / "ci.yml")["jobs"]["test"]
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
    """
    import tomllib

    root = Path(__file__).resolve().parents[2]

    ci_text = (WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")
    assert (
        "--ignore=tests/unit/test_rest_server.py" not in ci_text
    ), "ci.yml still passes --ignore for the rest_server suite"
    assert (
        "--ignore=tests/unit/test_inference_server.py" not in ci_text
    ), "ci.yml still passes --ignore for the inference_server suite"

    with (root / "pyproject.toml").open("rb") as fh:
        omit = tomllib.load(fh)["tool"]["coverage"]["run"]["omit"]
    for module in ("src/api/rest_server.py", "src/api/inference_server.py"):
        assert module not in omit, (
            f"{module} is back in the coverage omit list. Its test suite now collects "
            f"and passes, so omitting it only hides measured code — CHARTER.md NG-5."
        )


# ------------------------------------------------------------------ Makefile parity


def _makefile_text() -> str:
    return (Path(__file__).resolve().parents[2] / "Makefile").read_text(encoding="utf-8")


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
    ci_text = (WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")

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
