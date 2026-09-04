"""The entry points an operator actually types, run as processes.

Two surfaces that are declared in the tree but which nothing in ``tests/`` had ever
invoked the way a user does:

* **The installed console scripts.** ``pyproject.toml [project.scripts]`` declares eight
  ``name = "module:function"`` targets. Every existing test imports the module and calls
  the function; none resolves the installed name. A typo'd or renamed target therefore
  stays invisible until someone types the command. The script list is read from
  ``pyproject.toml`` so this test covers a new script automatically rather than rotting
  against a hardcoded list.
* **The container healthcheck.** ``healthcheck.py`` is a standalone script wired as the
  image's ``HEALTHCHECK``. Its exit-code contract is unit-tested in-process
  (``tests/unit/test_healthcheck_exit_codes.py``, ``tests/unit/test_healthcheck_cuda.py``);
  what is asserted here is the contract at the process boundary, which is the only form
  Docker observes.

Both run with every provider credential stripped from the environment, so the healthcheck
takes its "not configured" path instead of calling a provider. That matters beyond
tidiness: this suite also runs in the post-merge workflow, which exports real
``OPENAI_API_KEY`` and ``ANTHROPIC_API_KEY`` secrets, and a healthcheck that saw them
would bill a live call on every push to main.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from tests.utils.device_matrix import CPU_DEVICE, CUDA_DEVICE, device_available

pytestmark = [pytest.mark.e2e, pytest.mark.smoke]

#: The healthcheck's own environment contract (``healthcheck.py``: ``REQUIRE_GPU_ENV``).
REQUIRE_GPU_ENV = "REQUIRE_GPU"

#: ``healthcheck.py`` exit codes. DEGRADED maps to 0 because Docker treats any non-zero
#: exit as unhealthy, and a CPU-only container with no optional services is operational.
HEALTHCHECK_OPERATIONAL = 0
HEALTHCHECK_UNHEALTHY = 1


def _declared_console_scripts() -> list[str]:
    """Script names from ``pyproject.toml [project.scripts]``, so the list cannot rot."""
    try:
        import tomllib as toml_reader
    except ModuleNotFoundError:  # pragma: no cover - only on Python 3.10
        import tomli as toml_reader  # type: ignore[import-not-found]

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        data = toml_reader.load(handle)
    scripts = sorted((data.get("project") or {}).get("scripts") or {})
    assert scripts, "pyproject.toml declares no [project.scripts]"
    return scripts


@pytest.mark.parametrize("script", _declared_console_scripts())
def test_declared_console_script_is_installed_and_runs(script: str, run_script) -> None:
    """Every declared console script resolves and responds to ``--help``.

    ``--help`` is the cheapest command that still proves the whole chain: the entry point
    is on PATH, its module imports, and its ``main`` is callable.
    """
    result = run_script([script, "--help"])
    assert result.returncode == 0, (
        f"the console script {script!r} declared in pyproject.toml did not run. Either it is not "
        f"installed (`pip install -e .`) or its module:function target is wrong.\n{result.describe()}"
    )
    assert result.stdout.strip(), f"{script} --help printed nothing\n{result.describe()}"


def test_a_console_script_emits_its_logs_without_polluting_stdout(tmp_path, run_script, e2e_seed) -> None:
    """A real run must be diagnosable afterwards, and its stdout must stay a data channel.

    Both halves were broken. ``main()`` never configured logging, so ``get_logger`` handed
    back an unconfigured ``mcts.*`` logger and every INFO record in the run was discarded —
    an operator saw no resolved device, no seed, no losses, no checkpoint path. And the
    obvious fix is wrong: ``setup_logging`` defaults to stdout, where ``policy-lift`` prints
    its JSON artifact, so logging there would corrupt ``policy-lift ... | jq``.

    Driven through the installed console script rather than in-process, because the defect
    lived in ``main()`` — the one function an in-process test never calls.
    """
    result = run_script(
        [
            "self-play-convergence",
            "--domain",
            "reasoning",
            "--iterations",
            "1",
            "--checkpoint-dir",
            str(tmp_path / "ck"),
            "--seed",
            str(e2e_seed),
            "--device",
            CPU_DEVICE,
            "--num-simulations",
            "2",
            "--games-per-iteration",
            "1",
        ]
    )
    assert result.returncode == 0, result.describe()

    assert result.stderr.strip(), f"the driver produced no diagnostics at all\n{result.describe()}"
    assert (
        not result.stdout.strip()
    ), f"the driver wrote to stdout, which must stay free for command output\n{result.describe()}"

    # Structured, not just present: the records carry the correlation id INV-8 requires.
    records = [json.loads(line) for line in result.stderr.splitlines() if line.startswith("{")]
    assert records, f"stderr carried no structured log records\n{result.describe()}"
    assert all("correlation_id" in record for record in records)
    assert any(record.get("level") == "INFO" for record in records), "INFO records are still being discarded"


def _run_healthcheck(run_script, repo_root: Path, *, require_gpu: bool = False):
    overrides = {REQUIRE_GPU_ENV: "1"} if require_gpu else {REQUIRE_GPU_ENV: None}
    return run_script(
        [sys.executable, str(repo_root / "healthcheck.py")],
        env_overrides=overrides,
        with_provider_key=False,
    )


def _parse_report(stdout: str) -> dict:
    """Pull the JSON report out of the script's banner-wrapped output."""
    start = stdout.find("{")
    end = stdout.rfind("}")
    assert start != -1 and end > start, f"no JSON report found in healthcheck output:\n{stdout[-2000:]}"
    return json.loads(stdout[start : end + 1])


def test_healthcheck_is_operational_without_optional_services(run_script, repo_root: Path) -> None:
    """With no credentials and no GPU required, the container reports operational (exit 0).

    This is the contract PR #165 restored: a DEGRADED container is serviceable, and Docker
    treats any non-zero exit as unhealthy, so the image could otherwise never become
    healthy on a CPU-only host. Asserted on every host, not only CPU ones — a configured
    GPU makes the report healthier, never less operational.
    """
    result = _run_healthcheck(run_script, repo_root)
    assert result.returncode == HEALTHCHECK_OPERATIONAL, (
        "the healthcheck reported the container unhealthy while only optional services "
        f"were absent; Docker would never mark this image healthy.\n{result.describe()}"
    )

    report = _parse_report(result.stdout)
    assert report["status"] in {"healthy", "degraded"}
    # ``checks`` is keyed by check name (``HealthCheckReport.to_dict``).
    checks = report["checks"]
    assert "cuda" in checks, f"the report names no cuda check: {sorted(checks)}"

    # The network guard: with no key configured the provider check must report exactly
    # that, rather than having attempted a call.
    llm_checks = [check for name, check in checks.items() if name.startswith("llm_")]
    assert llm_checks, f"the report names no llm check: {sorted(checks)}"
    assert (
        llm_checks[0]["metadata"].get("configured") is False
    ), f"the healthcheck believed a provider was configured despite a stripped environment: {llm_checks[0]}"


@pytest.mark.skipif(
    device_available(CUDA_DEVICE),
    reason="asserts the CPU-host branch of REQUIRE_GPU; this host has CUDA",
)
def test_require_gpu_makes_a_missing_accelerator_fatal(run_script, repo_root: Path) -> None:
    """``REQUIRE_GPU=1`` turns the CUDA check critical, so a CPU-only host fails."""
    result = _run_healthcheck(run_script, repo_root, require_gpu=True)
    assert result.returncode == HEALTHCHECK_UNHEALTHY, (
        f"REQUIRE_GPU=1 on a host without CUDA must be fatal, so a GPU-required deployment "
        f"cannot start on a CPU node.\n{result.describe()}"
    )

    report = _parse_report(result.stdout)
    assert report["status"] == "unhealthy"
    cuda_check = report["checks"]["cuda"]
    assert cuda_check["critical"] is True, f"REQUIRE_GPU did not make the cuda check critical: {cuda_check}"
