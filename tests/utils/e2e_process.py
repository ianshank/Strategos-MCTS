"""Hermetic subprocess helpers for end-to-end tests.

End-to-end tests here drive the project's real entry points (``python -m <module>`` and
the installed console scripts) as a user would. Three things make that safe in CI:

1. **A hermetic environment.** :func:`hermetic_env` starts from the parent environment,
   strips every provider credential named in ``LLM_PROVIDER_CREDENTIAL_ENV_VARS`` plus the
   tracker keys, and pins the offline flags the CI ``test`` job already sets. A child
   therefore cannot reach a paid provider even when the parent shell holds a real key -
   the post-merge LangSmith workflow does exactly that.
2. **A bounded lifetime.** Every child runs in its own session (``start_new_session``)
   and is killed as a process group on timeout, so a hung ``torch.distributed``
   rendezvous cannot outlive the test or hold its port.
3. **Self-describing failures.** A :class:`ProcessResult` carries argv, exit code,
   captured streams and wall time; ``describe()`` is the assertion message, so a red log
   explains itself without re-running.

Tunables are named constants with an environment override, never inline literals.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import contextlib
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Final

from src.config.constants import LLM_PROVIDER_CREDENTIAL_ENV_VARS

logger = logging.getLogger("tests.e2e.process")

#: Per-child wall-clock budget; override for slow hosts. Kept well under pytest's global
#: 300 s so the subprocess times out first and reports its own streams.
SUBPROCESS_TIMEOUT_ENV: Final[str] = "E2E_SUBPROCESS_TIMEOUT_SECONDS"
DEFAULT_SUBPROCESS_TIMEOUT_SECONDS: Final[float] = 120.0

#: The same placeholder the unit suite and the CI test job use. It satisfies
#: ``Settings`` validation (a provider key must be present for the OpenAI provider) and
#: is never sent anywhere: every child runs with tracing off and no client is exercised.
DUMMY_OPENAI_API_KEY: Final[str] = "sk-test-key-not-real"
DUMMY_PROVIDER_ENV: Final[Mapping[str, str]] = {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": DUMMY_OPENAI_API_KEY}

#: Offline / no-tracking posture, mirroring the CI test job's ``env:`` block.
HERMETIC_ENV_DEFAULTS: Final[Mapping[str, str]] = {
    "WANDB_MODE": "disabled",
    "LANGCHAIN_TRACING_V2": "false",
    "LANGSMITH_TRACING": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONHASHSEED": "0",
}

#: Secrets a child must never inherit, whatever the parent shell holds.
#:
#: Deduplicated: ``LLM_PROVIDER_CREDENTIAL_ENV_VARS`` already carries some of these, and a
#: list with repeats invites the reader to assume it was assembled carefully when it was not.
#:
#: The cloud and tracker keys matter as much as the model-provider ones. ``aioboto3`` is a
#: *core* dependency, not an extra, so AWS credentials are plausibly present in any shell that
#: runs this suite — and this suite also runs in a post-merge workflow that exports real
#: secrets. A child that inherited them could reach S3 with production credentials.
STRIPPED_ENV_VARS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(
        tuple(LLM_PROVIDER_CREDENTIAL_ENV_VARS)
        + (
            # Model providers and gateways beyond the shared constant.
            "LANGCHAIN_API_KEY",
            "HUGGINGFACE_HUB_TOKEN",
            "HF_TOKEN",
            "BENCHMARK_ADK_GOOGLE_API_KEY",
            "GOOGLE_API_KEY",
            # Experiment trackers and vector stores.
            "PINECONE_API_KEY",
            "BRAINTRUST_API_KEY",
            # Cloud credentials — see the note above; aioboto3 is a core dependency.
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
        )
    )
)

_KILL_GRACE_SECONDS: Final[float] = 5.0


def subprocess_timeout_seconds() -> float:
    """The per-child timeout, honouring ``E2E_SUBPROCESS_TIMEOUT_SECONDS`` when it parses."""
    raw = os.environ.get(SUBPROCESS_TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_SUBPROCESS_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Ignoring malformed %s=%r; using %.0fs", SUBPROCESS_TIMEOUT_ENV, raw, DEFAULT_SUBPROCESS_TIMEOUT_SECONDS
        )
        return DEFAULT_SUBPROCESS_TIMEOUT_SECONDS
    # A malformed or non-positive override must never disable the bound.
    return value if value > 0 else DEFAULT_SUBPROCESS_TIMEOUT_SECONDS


def hermetic_env(
    *,
    repo_root: Path,
    overrides: Mapping[str, str | None] | None = None,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a child environment with secrets stripped and the offline posture pinned.

    ``overrides`` are applied last; a ``None`` value deletes the variable. ``PYTHONPATH``
    is prefixed with ``repo_root`` so ``python -m src...`` resolves from any cwd.
    """
    env = dict(os.environ if base is None else base)
    for name in STRIPPED_ENV_VARS:
        env.pop(name, None)
    env.update(HERMETIC_ENV_DEFAULTS)
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not existing_path else os.pathsep.join([str(repo_root), existing_path])
    for name, value in (overrides or {}).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return env


@dataclass(frozen=True)
class ProcessResult:
    """Everything an assertion message needs about a finished child."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def describe(self, tail_lines: int = 40) -> str:
        """A compact, log-friendly account of the run for use in assertion messages."""

        def tail(text: str) -> str:
            lines = text.strip().splitlines()
            return "\n".join(lines[-tail_lines:]) if lines else "<empty>"

        status = "TIMED OUT" if self.timed_out else f"exit {self.returncode}"
        return (
            f"$ {' '.join(self.argv)}\n"
            f"-> {status} in {self.duration_seconds:.1f}s\n"
            f"--- stdout (last {tail_lines} lines) ---\n{tail(self.stdout)}\n"
            f"--- stderr (last {tail_lines} lines) ---\n{tail(self.stderr)}"
        )


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    """SIGTERM the child's whole session, then SIGKILL whatever survives the grace period."""
    if proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def launch(argv: Sequence[str], *, env: Mapping[str, str], cwd: Path) -> subprocess.Popen[str]:
    """Start a child in its own session with both streams captured."""
    logger.debug("launch: %s (cwd=%s)", " ".join(argv), cwd)
    return subprocess.Popen(  # noqa: S603 - argv is built by the test, never from input
        list(argv),
        cwd=str(cwd),
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def wait_all(
    procs: Iterable[subprocess.Popen[str]],
    *,
    timeout: float | None = None,
    started_at: float | None = None,
) -> list[ProcessResult]:
    """Wait for every child under one shared deadline; on expiry kill them all as groups.

    Killing *all* of them matters for distributed runs: one rank blocked in a collective
    keeps the others blocked, so a per-child timeout alone would leave the port held.
    """
    budget = subprocess_timeout_seconds() if timeout is None else timeout
    t0 = time.monotonic() if started_at is None else started_at
    procs = list(procs)
    results: list[ProcessResult | None] = [None] * len(procs)
    timed_out = False
    for index, proc in enumerate(procs):
        remaining = max(0.0, budget - (time.monotonic() - t0))
        try:
            out, err = proc.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_group(proc)
            out, err = proc.communicate()
            results[index] = ProcessResult(
                tuple(proc.args), proc.returncode, out, err, time.monotonic() - t0, timed_out=True
            )
            continue
        results[index] = ProcessResult(tuple(proc.args), proc.returncode, out, err, time.monotonic() - t0)
    if timed_out:
        # A sibling may have finished normally; that is still a failed run as a group.
        for proc in procs:
            _kill_process_group(proc)
    finished = [r for r in results if r is not None]
    for result in finished:
        logger.debug("finished: %s", result.describe(tail_lines=10))
    return finished


def run_command(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: Path,
    timeout: float | None = None,
) -> ProcessResult:
    """Run one child to completion under the timeout; never raises on a non-zero exit."""
    started = time.monotonic()
    proc = launch(argv, env=env, cwd=cwd)
    return wait_all([proc], timeout=timeout, started_at=started)[0]


def run_python_module(
    module: str,
    args: Sequence[str] = (),
    *,
    env: Mapping[str, str],
    cwd: Path,
    timeout: float | None = None,
) -> ProcessResult:
    """``python -m module args...`` with the interpreter running this test session."""
    return run_command([sys.executable, "-m", module, *args], env=env, cwd=cwd, timeout=timeout)


def free_tcp_port(host: str = "127.0.0.1") -> int:
    """A currently unbound loopback port. Racy by nature; callers retry on bind failure."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
