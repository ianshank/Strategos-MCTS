"""Unit tests for the M5 self-play convergence driver (spec ``m5_policy_lift``).

Exercises the driver's AC-1 mechanism (runs, writes checkpoint + sidecar, resumes with
monotonic numbering) and AC-2 (the checkpoint round-trips through the policy-lift gate's
architecture resolver via the sidecar) on the CPU-only synthetic ``reasoning`` domain.

The chess AC-1/AC-2 verification is deferred: ``python-chess`` is absent from this CI, so
the chess cases below ``skipif`` out — a reasoning run proves the mechanism but does NOT
satisfy AC-1/AC-2 for chess (roadmap: reasoning/planning lifts are smoke tests only).
"""

from __future__ import annotations

import argparse
import asyncio
import json

import pytest
import torch

from src.benchmark.policy_lift import build_network, load_architecture
from src.framework.domain_registry import DomainRegistry
from src.games.chess.registration import chess_available
from src.training.self_play_convergence import (
    EXIT_ERROR,
    EXIT_OK,
    build_parser,
    resolve_architecture,
    run,
)

pytestmark = [pytest.mark.unit]

_REASONING_STATE_DIM = 128  # ReasoningState.to_tensor() -> [STATE_FEATURE_DIM] (default 128)


def _run(argv: list[str]) -> int:
    """Parse ``argv`` through the real CLI parser and run the driver, returning its exit code."""
    args = build_parser().parse_args(argv)
    return asyncio.run(run(args))


def _tiny_reasoning_argv(checkpoint_dir, *, iterations: int = 1, resume: bool = False) -> list[str]:
    argv = [
        "--domain",
        "reasoning",
        "--iterations",
        str(iterations),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--seed",
        "0",
        "--device",
        "cpu",
        "--num-simulations",
        "2",
        "--games-per-iteration",
        "1",
    ]
    if resume:
        argv.append("--resume")
    return argv


# --------------------------------------------------------------------------- AC-1


def test_driver_writes_checkpoint_and_sidecar(tmp_path):
    checkpoint_dir = tmp_path / "ckpts"

    assert _run(_tiny_reasoning_argv(checkpoint_dir)) == EXIT_OK

    checkpoint = checkpoint_dir / "ckpt_iter_1.pt"
    sidecar = checkpoint_dir / "ckpt_iter_1.pt.meta.json"
    assert checkpoint.is_file()
    assert sidecar.is_file()

    meta = json.loads(sidecar.read_text())
    assert meta["iteration"] == 1
    assert meta["domain"] == "reasoning"
    assert meta["schema_version"] == 1
    assert meta["network"]["type"] == "mlp"
    assert meta["network"]["state_dim"] == _REASONING_STATE_DIM
    assert meta["network"]["action_size"] == DomainRegistry.action_space_size("reasoning")
    # BatchNorm-free so single-sample self-play forwards do not raise.
    assert meta["network"]["use_batch_norm"] is False


def test_resume_continues_monotonic_numbering(tmp_path):
    checkpoint_dir = tmp_path / "ckpts"

    assert _run(_tiny_reasoning_argv(checkpoint_dir, iterations=1)) == EXIT_OK
    assert (checkpoint_dir / "ckpt_iter_1.pt").is_file()

    # Resuming continues from iter 1 -> writes iter 2 (does NOT restart numbering at 1).
    assert _run(_tiny_reasoning_argv(checkpoint_dir, iterations=1, resume=True)) == EXIT_OK
    assert (checkpoint_dir / "ckpt_iter_1.pt").is_file()  # untouched
    assert (checkpoint_dir / "ckpt_iter_2.pt").is_file()  # monotonic
    assert not (checkpoint_dir / "ckpt_iter_3.pt").exists()

    meta2 = json.loads((checkpoint_dir / "ckpt_iter_2.pt.meta.json").read_text())
    assert meta2["iteration"] == 2


def test_resume_with_no_existing_checkpoint_starts_fresh(tmp_path):
    checkpoint_dir = tmp_path / "ckpts"

    # --resume against an empty dir must not fail; it starts at iteration 1.
    assert _run(_tiny_reasoning_argv(checkpoint_dir, iterations=1, resume=True)) == EXIT_OK
    assert (checkpoint_dir / "ckpt_iter_1.pt").is_file()


# ----------------------------------------------------------------------- error paths


def test_unknown_domain_exits_error(tmp_path, capsys):
    code = _run(["--domain", "does-not-exist", "--iterations", "1", "--checkpoint-dir", str(tmp_path / "c")])
    assert code == EXIT_ERROR
    assert "Unknown domain" in capsys.readouterr().err


def test_unwritable_checkpoint_dir_exits_error(tmp_path, capsys):
    # A regular file where a directory is expected: mkdir(parents=True) raises OSError.
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    checkpoint_dir = blocker / "sub"

    code = _run(_tiny_reasoning_argv(checkpoint_dir))
    assert code == EXIT_ERROR
    assert "could not write checkpoint" in capsys.readouterr().err


def test_multidim_state_domain_exits_error(tmp_path, capsys):
    """The MLP path rejects a >1-D state tensor, surfaced by run() as a clean error exit."""
    from src.framework.domain_registry import METRIC_MEAN_REWARD, register_domain

    class _TwoDState:
        def to_tensor(self) -> torch.Tensor:
            return torch.zeros(2, 3)  # 2-D -> not an MLP state vector

    register_domain("twodim-test", lambda: _TwoDState(), 4, single_agent=True, metric=METRIC_MEAN_REWARD)
    try:
        code = _run(["--domain", "twodim-test", "--iterations", "1", "--checkpoint-dir", str(tmp_path / "c")])
        assert code == EXIT_ERROR
        assert "state tensor" in capsys.readouterr().err
    finally:
        DomainRegistry._registry.pop("twodim-test", None)


# --------------------------------------------------------------------------- AC-2


def test_checkpoint_round_trips_through_gate_resolver(tmp_path):
    """AC-2: a driver checkpoint resolves via its sidecar and its state_dict loads."""
    checkpoint_dir = tmp_path / "ckpts"
    assert _run(_tiny_reasoning_argv(checkpoint_dir)) == EXIT_OK

    checkpoint = checkpoint_dir / "ckpt_iter_1.pt"
    sidecar_network = json.loads((checkpoint_dir / "ckpt_iter_1.pt.meta.json").read_text())["network"]

    spec = DomainRegistry.get("reasoning")
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # No CLI overrides -> the resolver must use the sidecar, not inference/defaults.
    ns = argparse.Namespace(network_config=None, input_dim=None, hidden_dims=None)

    resolved = load_architecture(checkpoint, ns, spec, state_dict)

    assert resolved == sidecar_network  # sidecar-driven resolution
    assert resolved["state_dim"] == _REASONING_STATE_DIM
    assert resolved["action_size"] == spec.action_space_size

    # The gate rebuilds the identical network and loads the driver's weights without error.
    rebuilt = build_network(resolved, spec, "cpu", state_dict=state_dict)
    rebuilt.load_state_dict(state_dict)


# --------------------------------------------------------------------- chess (deferred)


@pytest.mark.skipif(not chess_available(), reason="chess extra (python-chess) not installed")
def test_chess_architecture_resolves_to_conv():
    """AC-1/AC-2 for chess run only where the chess extra exists (deferred in this CI)."""
    spec = DomainRegistry.get("chess")
    arch = resolve_architecture(spec)
    assert arch["type"] == "resnet"
    assert arch["action_size"] == spec.action_space_size
