"""Two real ranks under a real ``gloo`` process group, on CPU.

``CHARTER.md`` §5 Gate G-M1 asks for "a distributed end-to-end test green in CI on at
least two ranks", and ``specs/ddp_orchestrator.SPEC.md`` AC-4 requires rank-0 I/O
fencing. Until now every distributed test in the tree patched
``torch.distributed.init_process_group``, so no test had ever formed a process group and
the fencing was verified only against a mock.

**The observable.** Each rank is given its *own* ``--checkpoint-dir``. When the group
forms, ``SelfPlayTrainer.save_checkpoint`` returns early on every non-zero rank, so rank
1's directory stays empty. When the group does *not* form, ``is_distributed()`` is false
in both children, both believe they are the main process, and both write. So "rank 0
wrote, rank 1 did not" is a direct, falsifiable statement that the group really formed
and that fencing held — it cannot pass by accident.

That negative case is not hypothetical. ``Settings.TRAINING_BACKEND`` defaults to
``nccl``; on a CPU-only host ``init_distributed`` catches the resulting failure, returns
``False``, and ``src/training/self_play_convergence.py`` does not check the return value.
Running this pair with the default backend produces two independent single-process runs
that both write a checkpoint and both exit 0. This module therefore sets
``TRAINING_BACKEND=gloo`` explicitly, and the finding is recorded in
``docs/plans/2026-09-04-e2e-device-agnostic.md`` rather than silently worked around.

CPU-only by design: NCCL needs two GPUs, which no CI runner here has. The domain is
``reasoning`` because its MLP carries no BatchNorm buffers — DDP broadcasts buffers on
every forward, and with per-rank self-play trajectories of differing length a
buffer-carrying network can deadlock mid-search. That is a real hazard for the
orchestrator, noted in the plan; it is not what this test is for.
"""

from __future__ import annotations

from pathlib import Path
import time

import pytest

from src.training import self_play_convergence
from tests.utils.device_matrix import CPU_DEVICE
from tests.utils.e2e_process import free_tcp_port, launch, wait_all

pytestmark = [pytest.mark.e2e, pytest.mark.training]

WORLD_SIZE = 2
DOMAIN = "reasoning"
SIMULATIONS = "2"
GAMES_PER_ITERATION = "1"

#: Wall-clock budget for the whole pair. Both children share one deadline and are killed
#: as process groups on expiry: torch's default collective timeout is 30 minutes, far
#: longer than the CI job, so a rendezvous that never completes must be bounded here.
DDP_TIMEOUT_SECONDS = 240.0

#: Retries for the bind race on ``MASTER_PORT``: the port is chosen by binding and
#: releasing, so another process can take it in between.
PORT_RETRIES = 3
_PORT_IN_USE_MARKERS = ("Address already in use", "EADDRINUSE")


def _rank_env(rank: int, port: int, e2e_env) -> dict[str, str]:
    """The environment ``torchrun`` would hand a rank, plus the CPU-only pins."""
    return e2e_env(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(WORLD_SIZE),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            # Without this the settings default (nccl) is used, init fails on a CPU host,
            # the failure is swallowed, and the test would pass against two independent
            # single-process runs. See the module docstring.
            "TRAINING_BACKEND": "gloo",
            # Hide any GPU so the driver does not override --device with cuda:{local_rank}.
            "CUDA_VISIBLE_DEVICES": "",
            # torchrun sets this for nproc_per_node > 1; two default-threaded torch
            # processes oversubscribe a 2-vCPU runner badly.
            "OMP_NUM_THREADS": "1",
            # gloo otherwise resolves the hostname to choose an interface, which fails in
            # some containers.
            "GLOO_SOCKET_IFNAME": "lo",
        }
    )


def _driver_argv(checkpoint_dir: Path, seed: int) -> list[str]:
    return [
        "-m",
        self_play_convergence.__name__,
        "--domain",
        DOMAIN,
        "--iterations",
        "1",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--seed",
        str(seed),
        "--device",
        CPU_DEVICE,
        "--num-simulations",
        SIMULATIONS,
        "--games-per-iteration",
        GAMES_PER_ITERATION,
    ]


def _run_pair(tmp_path: Path, repo_root: Path, e2e_env, seed: int, attempt: int):
    """Launch both ranks concurrently under one deadline; returns (results, rank_dirs)."""
    import sys

    port = free_tcp_port()
    rank_dirs = [tmp_path / f"attempt{attempt}" / f"rank{rank}" for rank in range(WORLD_SIZE)]
    procs = []
    started = time.monotonic()
    for rank, checkpoint_dir in enumerate(rank_dirs):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        procs.append(
            launch(
                [sys.executable, *_driver_argv(checkpoint_dir, seed)],
                env=_rank_env(rank, port, e2e_env),
                cwd=repo_root,
            )
        )
    results = wait_all(procs, timeout=DDP_TIMEOUT_SECONDS, started_at=started)
    return results, rank_dirs


@pytest.mark.timeout(DDP_TIMEOUT_SECONDS + 120)
def test_two_ranks_form_a_group_and_only_rank_zero_writes(tmp_path, repo_root, e2e_env, e2e_seed) -> None:
    """Both ranks train and exit cleanly; only rank 0 writes the checkpoint and sidecar."""
    for attempt in range(PORT_RETRIES):
        results, rank_dirs = _run_pair(tmp_path, repo_root, e2e_env, e2e_seed, attempt)
        combined = "\n".join(result.stderr for result in results)
        if any(marker in combined for marker in _PORT_IN_USE_MARKERS) and attempt < PORT_RETRIES - 1:
            continue  # lost the bind race; retry on a fresh port
        break

    report = "\n\n".join(f"[rank {rank}] {result.describe()}" for rank, result in enumerate(results))

    for rank, result in enumerate(results):
        assert not result.timed_out, f"rank {rank} did not finish within {DDP_TIMEOUT_SECONDS}s\n\n{report}"
        assert result.returncode == self_play_convergence.EXIT_OK, f"rank {rank} failed\n\n{report}"

    rank_zero_files = sorted(path.name for path in rank_dirs[0].iterdir())
    rank_one_files = sorted(path.name for path in rank_dirs[1].iterdir())

    assert rank_zero_files == [
        "ckpt_iter_1.pt",
        "ckpt_iter_1.pt.meta.json",
    ], f"rank 0 must write exactly the checkpoint and its sidecar; wrote {rank_zero_files}\n\n{report}"
    assert rank_one_files == [], (
        "rank 1 wrote "
        f"{rank_one_files}. Either the process group never formed (so both ranks believed they "
        "were the main process) or rank-0 I/O fencing regressed. Both are ddp_orchestrator AC-4 "
        f"failures.\n\n{report}"
    )
