"""
Docker Deployment Smoke Tests
==============================

Post-deployment smoke tests to verify Docker containers are working correctly.

Tests:
- Container health checks
- GPU availability
- Service connectivity
- API endpoints
- Configuration loading

Usage:
    # Run against running containers
    pytest tests/deployment/test_docker_smoke.py -v

    # With container names
    pytest tests/deployment/test_docker_smoke.py -v --container-name=mcts-training-demo

2025 Best Practices:
- Test real deployed containers
- Verify GPU access
- Check service integration
- Validate configuration
"""

import os
import shutil
import subprocess
import time

import pytest
import requests

# Optional docker import - skip tests if not available
docker = pytest.importorskip("docker", reason="Docker SDK required for deployment tests")

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def docker_client():
    """Docker client for container interaction."""
    try:
        client = docker.from_env()
        # Verify Docker is accessible
        client.ping()
        return client
    except Exception as e:
        pytest.skip(f"Docker not available: {e}")


@pytest.fixture(scope="session")
def training_container_name():
    """Get training container name from environment."""
    return os.getenv("TRAINING_CONTAINER", "mcts-training-demo")


@pytest.fixture(scope="session")
def api_container_name():
    """Get API container name from environment."""
    return os.getenv("API_CONTAINER", "mcts-api-server")


@pytest.fixture
def running_training_container(docker_client, training_container_name):
    """Get running training container or start it if stopped."""
    try:
        container = docker_client.containers.get(training_container_name)
        if container.status != "running":
            print(f"Container {training_container_name} is {container.status}, restarting...")
            container.start()
            # Wait for container to be healthy/running
            wait_for_container_healthy(docker_client, training_container_name)
        return container
    except docker.errors.NotFound:
        pytest.skip(f"Container {training_container_name} not found. Please build and create it first.")


def wait_for_container_healthy(
    client: docker.DockerClient,
    container_name: str,
    timeout: int = 60,
) -> bool:
    """Wait for container to become healthy."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            container = client.containers.get(container_name)
            if container.status == "running":
                # Check if health check is configured
                health = container.attrs.get("State", {}).get("Health", {})
                status = health.get("Status")

                # If healthy, or if running and no health check (status is None)
                if status == "healthy" or status is None:
                    return True
        except docker.errors.NotFound:
            pass

        time.sleep(2)

    return False


def exec_in_container(
    client: docker.DockerClient,
    container_name: str,
    command: list[str],
) -> tuple[int, str]:
    """Execute command in container."""
    try:
        container = client.containers.get(container_name)
        if container.status != "running":
            # Try to restart it for the test
            container.start()
            time.sleep(5)  # Give it a moment

        exit_code, output = container.exec_run(command)
        return exit_code, output.decode("utf-8")
    except docker.errors.NotFound:
        pytest.skip(f"Container {container_name} not found")
    except Exception as e:
        return 1, str(e)


# GPU probe configuration. Named rather than inlined so the override contract is
# stated once and the timeout is not a bare magic number at the call site.
#
# Read from the environment rather than Pydantic Settings deliberately: this module
# probes the *host* before any container or application config exists, and importing
# Settings requires a configured provider key it has no business demanding. The same
# reasoning governs healthcheck.py, which reads all nine of its config values from
# os.environ and holds no `src` imports at all.
GPU_TEST_OVERRIDE_ENV = "FORCE_GPU_TESTS"
GPU_PROBE_TIMEOUT_ENV = "GPU_PROBE_TIMEOUT_SECONDS"
DEFAULT_GPU_PROBE_TIMEOUT_SECONDS = 10.0
_TRUTHY = frozenset({"1", "true", "yes"})


def _gpu_probe_timeout() -> float:
    """Seconds to allow the `nvidia-smi` probe, overridable for slow hosts."""
    raw = os.environ.get(GPU_PROBE_TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_GPU_PROBE_TIMEOUT_SECONDS
    try:
        return float(raw)
    except ValueError:
        # A malformed override must not silently disable the probe's timeout.
        return DEFAULT_GPU_PROBE_TIMEOUT_SECONDS


def _host_has_gpu() -> bool:
    """
    True when the host exposes an NVIDIA GPU that Docker can pass through.

    Checked on the host rather than in the container: a container cannot see a GPU
    the host does not have, and probing inside is what produced the original
    failure mode (``exec: "nvidia-smi": executable file not found``) instead of a
    skip.
    """
    if os.environ.get(GPU_TEST_OVERRIDE_ENV, "").strip().lower() in _TRUTHY:
        return True
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        completed = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=_gpu_probe_timeout())
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


# The GPU assertions below are real checks on a GPU host and unsatisfiable anywhere
# else. Without this guard they carried only @pytest.mark.smoke, so they FAILED
# rather than skipped on the CPU-only `ubuntu-latest` runner the smoke job uses —
# keeping the workflow red for a reason unrelated to the code under test.
# Set FORCE_GPU_TESTS=1 to run them regardless.
requires_gpu = pytest.mark.skipif(
    not _host_has_gpu(),
    reason="no NVIDIA GPU on the host (set FORCE_GPU_TESTS=1 to override)",
)


# ============================================================================
# Container Health Tests
# ============================================================================


@pytest.mark.smoke
@pytest.mark.integration
def test_training_container_running(docker_client, training_container_name):
    """Test that training container is running."""
    try:
        container = docker_client.containers.get(training_container_name)
        # If exited, restart for testing purposes
        if container.status == "exited":
            container.start()
            time.sleep(2)
            container.reload()

        assert container.status == "running", f"Container status: {container.status}"
    except docker.errors.NotFound:
        pytest.skip(f"Container {training_container_name} not found")


# How long to wait for Docker to report the container as healthy. The CI
# docker run overrides the image's HEALTHCHECK to --health-interval=10s /
# --health-start-period=5s, so the first probe lands within seconds and a
# healthy container is reported well inside this window. The previous 30s
# window was shorter than the image's default start-period (60s) and raced
# the first probe; 90s gives margin without slowing the common case.
#
# Override via HEALTH_WAIT_SECONDS env var for local debugging or slow CI runners.
_HEALTH_WAIT_SECONDS_DEFAULT = 90
_HEALTH_WAIT_SECONDS = int(os.environ.get("HEALTH_WAIT_SECONDS", str(_HEALTH_WAIT_SECONDS_DEFAULT)))


@pytest.mark.smoke
@pytest.mark.integration
def test_training_container_healthy(docker_client, training_container_name):
    """Test that training container passes the Docker HEALTHCHECK.

    The pass condition is Docker's own ``State.Health.Status == "healthy"``,
    not a direct exec of the healthcheck script. This is deliberate: if the
    test exec'd the script itself, removing or breaking the Dockerfile
    ``HEALTHCHECK`` directive would still pass, defeating the point of the
    check. The direct exec is used only to produce diagnostics on failure so
    the failure message names the real cause (e.g. the script exits non-zero)
    rather than a timing race.
    """
    try:
        container = docker_client.containers.get(training_container_name)
        if container.status == "exited":
            container.start()
    except docker.errors.NotFound:
        pytest.skip(f"Container {training_container_name} not found")

    start_time = time.time()
    healthy = False
    while time.time() - start_time < _HEALTH_WAIT_SECONDS:
        try:
            container = docker_client.containers.get(training_container_name)
            if container.status != "running":
                # The container is launched with `tail -f /dev/null`, so it must
                # stay running. An exited container — even with code 0 — is a
                # failure here: treating exit 0 as healthy would let a missing or
                # broken Dockerfile HEALTHCHECK still pass, defeating the test.
                #
                # NOTE: we fall through to time.sleep below rather than `continue`,
                # which would skip the sleep and create a tight loop hammering the
                # Docker daemon (Copilot review finding).
                pass
            elif container.attrs.get("State", {}).get("Health", {}).get("Status") == "healthy":
                # Docker's own healthcheck is the pass condition.
                healthy = True
                break
        except Exception:
            pass
        time.sleep(2)

    if healthy:
        return

    # Diagnostics only — the direct exec is NOT a pass condition. It surfaces the
    # real reason Docker never reported "healthy": the script's exit code and
    # output, plus Docker's recorded health log.
    diagnostics = _collect_health_diagnostics(docker_client, training_container_name)
    pytest.fail(
        f"Container {training_container_name} did not become healthy within {_HEALTH_WAIT_SECONDS}s.\n{diagnostics}"
    )


def _collect_health_diagnostics(client: "docker.DockerClient", container_name: str) -> str:
    """Gather container status, Docker health log and a direct healthcheck run.

    Returned as a formatted block for inclusion in a pytest.fail message. This
    is diagnostic only and never gates the test — the pass condition is Docker's
    ``State.Health.Status``.
    """
    lines = ["--- health diagnostics ---"]
    try:
        container = client.containers.get(container_name)
        state = container.attrs.get("State", {})
        lines.append(f"container status: {container.status}")
        lines.append(f"exit code: {state.get('ExitCode')}")
        health = state.get("Health")
        if health is None:
            lines.append("docker health: no HEALTHCHECK defined for this image")
        else:
            lines.append(f"docker health status: {health.get('Status')}")
            for entry in (health.get("Log") or [])[-5:]:
                lines.append(
                    f"  health log: exit={entry.get('ExitCode')} " f"out={str(entry.get('Output', '')).strip()[:200]}"
                )
    except Exception as exc:  # pragma: no cover - diagnostics best-effort
        lines.append(f"could not inspect container: {exc}")

    # Direct exec of the healthcheck script, for signal only.
    try:
        exit_code, output = exec_in_container(
            client,
            container_name,
            ["python", "/app/healthcheck.py"],
        )
        lines.append(f"direct healthcheck exec exit code: {exit_code}")
        lines.append(f"direct healthcheck output: {output.strip()[:500]}")
    except Exception as exc:  # pragma: no cover - diagnostics best-effort
        lines.append(f"could not exec healthcheck script: {exc}")

    return "\n".join(lines)


# ============================================================================
# GPU Availability Tests
# ============================================================================


@pytest.mark.smoke
@pytest.mark.integration
@requires_gpu
def test_cuda_available_in_container(docker_client, training_container_name):
    """Test that CUDA is available in training container."""
    exit_code, output = exec_in_container(
        docker_client,
        training_container_name,
        ["python", "-c", "import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count())"],
    )

    assert exit_code == 0, f"CUDA check failed: {output}"
    gpu_count = int(output.strip())
    assert gpu_count > 0, f"No GPUs available (count: {gpu_count})"


@pytest.mark.smoke
@pytest.mark.integration
@requires_gpu
def test_nvidia_smi_in_container(docker_client, training_container_name):
    """Test that nvidia-smi works in container."""
    exit_code, output = exec_in_container(
        docker_client,
        training_container_name,
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    )

    assert exit_code == 0, f"nvidia-smi failed: {output}"
    assert len(output.strip()) > 0, "nvidia-smi returned empty output"


# ============================================================================
# Configuration Tests
# ============================================================================


@pytest.mark.smoke
@pytest.mark.integration
def test_demo_config_loaded(docker_client, training_container_name):
    """Test that demo configuration is accessible."""
    exit_code, output = exec_in_container(
        docker_client,
        training_container_name,
        ["python", "-c", "import yaml; yaml.safe_load(open('training/config_local_demo.yaml')); print('OK')"],
    )

    assert exit_code == 0, f"Config load failed: {output}"
    assert "OK" in output, "Config validation failed"


@pytest.mark.smoke
@pytest.mark.integration
def test_python_imports(docker_client, training_container_name):
    """Test that required Python packages are importable."""
    # Remove tenacity from list as it might not have __version__ attribute
    packages = [
        "torch",
        "transformers",
        "yaml",
        "pydantic",
        "httpx",
    ]

    for package in packages:
        exit_code, output = exec_in_container(
            docker_client,
            training_container_name,
            ["python", "-c", f"import {package}; print('{package}', {package}.__version__)"],
        )

        assert exit_code == 0, f"Failed to import {package}: {output}"

    # Tenacity and Rich check might fail if accessed directly or no __version__, check just import
    for pkg in ["tenacity", "rich"]:
        exit_code, output = exec_in_container(
            docker_client,
            training_container_name,
            ["python", "-c", f"import {pkg}; print('{pkg} imported')"],
        )
        assert exit_code == 0, f"Failed to import {pkg}: {output}"


# ============================================================================
# Environment Variable Tests
# ============================================================================


@pytest.mark.smoke
@pytest.mark.integration
def test_required_env_vars_set(docker_client, training_container_name):
    """Test that required environment variables are set."""
    required_vars = [
        "CUDA_HOME",
        "PYTHONPATH",
        "PATH",
    ]

    for var in required_vars:
        exit_code, output = exec_in_container(
            docker_client,
            training_container_name,
            ["sh", "-c", f"echo ${var}"],
        )

        assert exit_code == 0, f"Failed to check {var}: {output}"
        assert len(output.strip()) > 0, f"Environment variable {var} is empty"


# ============================================================================
# File System Tests
# ============================================================================


@pytest.mark.smoke
@pytest.mark.integration
def test_required_directories_exist(docker_client, training_container_name):
    """Test that required directories exist in container."""
    directories = [
        "/app/src",
        "/app/training",
        "/app/scripts",
        "/app/checkpoints",
        "/app/logs",
        "/app/cache",
    ]

    for directory in directories:
        exit_code, output = exec_in_container(
            docker_client,
            training_container_name,
            ["test", "-d", directory],
        )

        assert exit_code == 0, f"Directory {directory} does not exist"


# ============================================================================
# API Container Tests (if applicable)
# ============================================================================


@pytest.mark.smoke
@pytest.mark.integration
@pytest.mark.api
def test_api_container_health_endpoint(api_container_name):
    """Test API container health endpoint."""
    try:
        # Try both standard health paths
        paths = ["/health", "/healthz", "/api/health"]
        success = False
        last_status = None

        for path in paths:
            try:
                response = requests.get(f"http://localhost:8000{path}", timeout=5)
                if response.status_code == 200:
                    success = True
                    break
                last_status = response.status_code
            except requests.exceptions.RequestException:
                continue

        if not success:
            # If we got a 404, the container is running but endpoint might be different
            # Skip if we can't find the health endpoint but can connect
            if last_status == 404:
                pytest.skip(f"API container running but health endpoint not found (tried {paths})")
            elif last_status:
                pytest.fail(f"Health check failed: {last_status}")
            else:
                pytest.skip("API container not accessible on localhost:8000")

    except requests.exceptions.ConnectionError:
        pytest.skip("API container not accessible on localhost:8000")


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.smoke
@pytest.mark.benchmark
@requires_gpu
def test_gpu_memory_available(docker_client, training_container_name):
    """Test that sufficient GPU memory is available."""
    exit_code, output = exec_in_container(
        docker_client,
        training_container_name,
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
    )

    assert exit_code == 0, f"GPU memory check failed: {output}"
    memory_mb = int(output.strip())
    assert memory_mb >= 15000, f"Insufficient GPU memory: {memory_mb}MB (need ≥15GB)"


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.smoke
@pytest.mark.integration
@pytest.mark.slow
def test_training_cli_help(docker_client, training_container_name):
    """Test that training CLI is accessible."""
    # This test executes a command inside the running container to verify the CLI is functional.
    exit_code, output = exec_in_container(
        docker_client,
        training_container_name,
        ["python", "-m", "training.cli", "--help"],
    )

    assert exit_code == 0, f"CLI help failed: {output}"
    assert "train" in output, "CLI help missing train command"
    assert "--demo" in output, "CLI help missing demo flag"


# ============================================================================
# Cleanup Tests
# ============================================================================


@pytest.mark.smoke
def test_container_logs_accessible(docker_client, training_container_name):
    """Test that container logs are accessible."""
    try:
        container = docker_client.containers.get(training_container_name)
        logs = container.logs(tail=10).decode("utf-8")
        assert len(logs) > 0, "Container logs should not be empty"
    except docker.errors.NotFound:
        pytest.skip(f"Container {training_container_name} not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "smoke"])
