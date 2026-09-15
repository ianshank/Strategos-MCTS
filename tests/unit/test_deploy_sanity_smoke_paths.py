"""Deploy sanity must not collect container docker smoke under a 60s cap.

CI Docker Deployment run 34705825988 died in ``scripts/deployment_sanity_check.py``
(``pytest tests/ -m smoke``, timeout=60). ``-m "smoke and not e2e"`` still collects
``tests/deployment/test_docker_smoke.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "deployment_sanity_check.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker-deployment.yml"

SANITY_SMOKE_PATHS = (
    "tests/e2e/test_operational_entry_points_e2e.py",
    "tests/e2e/test_local_distillation_cli_e2e.py",
    "tests/integration/test_demo_pipeline.py",
)
DOCKER_SMOKE = "tests/deployment/test_docker_smoke.py"


def _script_constants() -> tuple[list[str], int]:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    args: list[str] | None = None
    timeout: int | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id == "SANITY_SMOKE_ARGS" and isinstance(node.value, ast.List):
            args = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
        if target.id == "SANITY_SMOKE_TIMEOUT_SECONDS" and isinstance(node.value, ast.Constant):
            timeout = int(node.value.value)
    assert args is not None, "SANITY_SMOKE_ARGS missing"
    assert timeout is not None, "SANITY_SMOKE_TIMEOUT_SECONDS missing"
    return args, timeout


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _step_run(job: dict, step_name: str) -> str:
    for step in job.get("steps") or []:
        if isinstance(step, dict) and step.get("name") == step_name:
            run = step.get("run") or ""
            assert isinstance(run, str)
            return run
    raise AssertionError(f"step {step_name!r} not found")


def _step_pytest_lines(job: dict, step_name: str) -> str:
    """The step's run body with comments stripped so docs can name excluded files."""
    run = _step_run(job, step_name)
    live = [line for line in run.splitlines() if line.lstrip() and not line.lstrip().startswith("#")]
    return "\n".join(live)


def test_sanity_script_smoke_is_explicit_paths_with_roomy_timeout() -> None:
    args, timeout = _script_constants()
    for path in SANITY_SMOKE_PATHS:
        assert path in args
    assert "tests/" not in args
    assert DOCKER_SMOKE not in args
    assert timeout >= 180


def test_sanity_job_smoke_step_matches_script_paths() -> None:
    jobs = _workflow()["jobs"]
    run = _step_pytest_lines(jobs["sanity-checks"], "Run smoke tests")
    for path in SANITY_SMOKE_PATHS:
        assert path in run
    assert "pytest tests/ -m" not in run
    assert "pytest tests/ -v" not in run
    assert DOCKER_SMOKE not in run
    assert "smoke and not e2e" not in run


def test_container_smoke_job_still_owns_docker_smoke() -> None:
    jobs = _workflow()["jobs"]
    run = _step_pytest_lines(jobs["smoke-tests"], "Run smoke tests")
    assert DOCKER_SMOKE in run
    for path in SANITY_SMOKE_PATHS:
        assert path not in run
