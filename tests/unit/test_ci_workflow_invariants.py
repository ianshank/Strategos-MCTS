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

Spec: ``specs/hygiene_ci_mechanical.SPEC.md`` (AC-1, AC-2, AC-3, AC-8, AC-9, AC-12).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is required to parse workflow files")

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"

# Upper bound on any single job. Not a performance target — a backstop that turns a
# hung job into a failed job. The GitHub default is 360 minutes, which is long enough
# that a hang looks like a slow run for most of a working day.
MAX_TIMEOUT_MINUTES = 90

# Steps whose failure must never be swallowed. `|| true` after any of these means the
# job reports success regardless of the result, which is indistinguishable from having
# no check at all.
RESULT_BEARING_COMMANDS = ("pytest", "mypy", "ruff", "black")


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
def test_ci_summary_gates_every_job_it_depends_on() -> None:
    """AC-1: the reported set and the enforced set must be identical.

    The original summary job listed ``security-scan`` and ``dependency-audit`` in
    ``needs``, printed their results, and then omitted them from the failure
    condition — so a bandit HIGH finding printed a red line and merged green.
    """
    jobs = _load(WORKFLOW_DIR / "ci.yml")["jobs"]
    summary = jobs["summary"]
    needs = summary["needs"]
    assert isinstance(needs, list) and needs, "ci.yml summary job must declare a needs list"

    step = next(s for s in _steps(summary) if "run" in s)
    env = step.get("env") or {}
    body = step["run"]

    # 1. Every needed job is surfaced as an env var bound to that job's result.
    bound = {
        job for job in needs for value in env.values() if re.search(rf"needs\.{re.escape(job)}\.result", str(value))
    }
    missing = set(needs) - bound
    assert not missing, (
        f"ci.yml summary depends on {sorted(missing)} but never reads their .result "
        f"into an env var, so they cannot be gated on."
    )

    # 2. Every such env var is named in the shell list the gate iterates over. This is
    #    what makes reported == enforced structurally, rather than by coincidence.
    result_vars = {name for name, value in env.items() if re.search(r"needs\.[\w-]+\.result", str(value))}
    ungated = {name for name in result_vars if not re.search(rf"\b{re.escape(name)}\b", body)}
    assert not ungated, (
        f"ci.yml summary reads {sorted(ungated)} but never checks them in the gate "
        f"body. A job that is printed but not enforced is worse than one that is "
        f"absent, because the red line in the log implies it was checked."
    )

    # 3. The gate must actually be able to fail.
    assert "exit 1" in body, "ci.yml summary job has no failing exit path"


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

    Narrowly scoped to commands whose exit code *is* the signal. Cleanup and
    best-effort reporting steps legitimately use ``|| true`` and are not flagged.
    """
    for step in _steps(job):
        run = step.get("run")
        if not run or not any(cmd in run for cmd in RESULT_BEARING_COMMANDS):
            continue
        offenders = [
            line.strip()
            for line in run.splitlines()
            if "|| true" in line and any(cmd in line for cmd in RESULT_BEARING_COMMANDS)
        ]
        # A trailing `|| true` on its own continuation line also disarms the command.
        if re.search(r"(?:pytest|mypy|ruff|black)[^\n]*(?:\\\s*\n[^\n]*)*\|\|\s*true", run):
            offenders.append(run.strip().splitlines()[0])
        assert not offenders, (
            f"{workflow}::{job_id} step {step.get('name')!r} runs a result-bearing "
            f"command but discards its exit status: {offenders}. Remove the "
            f"`|| true`, or gate the whole job on a precondition instead."
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


# --------------------------------------------------------------------------- AC-2


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
