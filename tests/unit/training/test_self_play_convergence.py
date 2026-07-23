"""Unit tests for the M5 self-play convergence driver (spec ``m5_policy_lift``).

Exercises the driver's AC-1 mechanism (runs, writes checkpoint + sidecar, resumes with
monotonic numbering AND loaded weights) and AC-2 (the checkpoint round-trips through the
policy-lift gate's architecture resolver via the sidecar) on the CPU-only synthetic
``reasoning`` domain, plus argparse validation, ``main()``, and error paths.

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

from src.benchmark import policy_lift
from src.benchmark.policy_lift import build_network, load_architecture
from src.framework.domain_registry import DomainRegistry
from src.games.chess.registration import chess_available
from src.training.self_play_convergence import (
    _CHECKPOINT_RE,
    _CHECKPOINT_TEMPLATE,
    EXIT_ERROR,
    EXIT_OK,
    build_parser,
    main,
    resolve_architecture,
    run,
)

pytestmark = [pytest.mark.unit]

_REASONING_STATE_DIM = 128  # ReasoningState.to_tensor() -> [STATE_FEATURE_DIM] (default 128)


@pytest.fixture(autouse=True)
def _restore_domain_registry():
    """Snapshot/restore the process-wide DomainRegistry so a test that registers a domain
    cannot bleed into another (the registry is a class-level dict — mutations are global)."""
    snapshot = dict(DomainRegistry._registry)
    yield
    DomainRegistry._registry.clear()
    DomainRegistry._registry.update(snapshot)


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


def test_resume_from_multiple_checkpoints_picks_latest(tmp_path):
    """With several checkpoints present, --resume continues from the HIGHEST, not the first."""
    checkpoint_dir = tmp_path / "ckpts"

    assert _run(_tiny_reasoning_argv(checkpoint_dir, iterations=2)) == EXIT_OK  # writes iter 1 AND 2
    assert (checkpoint_dir / "ckpt_iter_2.pt").is_file()

    assert _run(_tiny_reasoning_argv(checkpoint_dir, iterations=1, resume=True)) == EXIT_OK
    assert (checkpoint_dir / "ckpt_iter_3.pt").is_file()  # continued past the max (2), not from 1
    assert not (checkpoint_dir / "ckpt_iter_4.pt").exists()


def test_resume_actually_loads_weights(tmp_path, monkeypatch):
    """AC-1 'weights loaded': the resumed net must carry the saved weights, not a fresh init."""
    from src.training.self_play_trainer import SelfPlayTrainer

    checkpoint_dir = tmp_path / "ckpts"
    assert _run(_tiny_reasoning_argv(checkpoint_dir, iterations=1)) == EXIT_OK
    saved = torch.load(checkpoint_dir / "ckpt_iter_1.pt", map_location="cpu", weights_only=True)

    captured: dict = {}
    real_load = SelfPlayTrainer.load_checkpoint

    def _spy(self, path):
        real_load(self, path)  # perform the real load
        captured["net"] = {key: value.clone() for key, value in self.network.state_dict().items()}

    monkeypatch.setattr(SelfPlayTrainer, "load_checkpoint", _spy)

    assert _run(_tiny_reasoning_argv(checkpoint_dir, iterations=1, resume=True)) == EXIT_OK
    assert captured, "resume must call load_checkpoint"
    for key, tensor in saved.items():
        assert torch.equal(captured["net"][key], tensor)  # loaded iter_1 weights, not a fresh seed-0 init


def test_resume_with_no_existing_checkpoint_starts_fresh(tmp_path):
    checkpoint_dir = tmp_path / "ckpts"

    # --resume against an empty dir must not fail; it starts at iteration 1.
    assert _run(_tiny_reasoning_argv(checkpoint_dir, iterations=1, resume=True)) == EXIT_OK
    assert (checkpoint_dir / "ckpt_iter_1.pt").is_file()


def test_resume_falls_back_to_resolver_when_sidecar_missing(tmp_path):
    """If a checkpoint's sidecar is absent, resume falls back to resolve_architecture cleanly."""
    checkpoint_dir = tmp_path / "ckpts"
    assert _run(_tiny_reasoning_argv(checkpoint_dir, iterations=1)) == EXIT_OK
    (checkpoint_dir / "ckpt_iter_1.pt.meta.json").unlink()  # drop the authoritative sidecar

    assert _run(_tiny_reasoning_argv(checkpoint_dir, iterations=1, resume=True)) == EXIT_OK
    assert (checkpoint_dir / "ckpt_iter_2.pt").is_file()


def test_runs_without_games_per_iteration(tmp_path):
    """The --games-per-iteration default path (None -> SelfPlayConfig default) runs cleanly."""
    checkpoint_dir = tmp_path / "ckpts"
    argv = [
        "--domain",
        "reasoning",
        "--iterations",
        "1",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--seed",
        "0",
        "--device",
        "cpu",
        "--num-simulations",
        "2",
    ]  # no --games-per-iteration
    assert _run(argv) == EXIT_OK
    assert (checkpoint_dir / "ckpt_iter_1.pt").is_file()


# ----------------------------------------------------------------- validation / errors


def test_nonpositive_iterations_rejected(tmp_path):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--domain", "reasoning", "--iterations", "0", "--checkpoint-dir", str(tmp_path)])


def test_out_of_range_seed_rejected(tmp_path):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--domain", "reasoning", "--checkpoint-dir", str(tmp_path), "--seed=-1"])


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


def test_training_failure_exits_error(tmp_path, capsys, monkeypatch):
    """A RuntimeError during training (e.g. CUDA OOM) exits cleanly, not as a raw traceback."""
    from src.training.self_play_trainer import SelfPlayTrainer

    async def _boom(self, *args, **kwargs):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(SelfPlayTrainer, "train_iteration", _boom)

    code = _run(_tiny_reasoning_argv(tmp_path / "ckpts"))
    assert code == EXIT_ERROR
    assert "training failed" in capsys.readouterr().err


def test_multidim_state_domain_exits_error(tmp_path, capsys):
    """The MLP path rejects a >1-D state tensor, surfaced by run() as a clean error exit."""
    from src.framework.domain_registry import METRIC_MEAN_REWARD, register_domain

    class _TwoDState:
        def to_tensor(self) -> torch.Tensor:
            return torch.zeros(2, 3)  # 2-D -> not an MLP state vector

    # The autouse _restore_domain_registry fixture removes this registration after the test.
    register_domain("twodim-test", lambda: _TwoDState(), 4, single_agent=True, metric=METRIC_MEAN_REWARD)
    code = _run(["--domain", "twodim-test", "--iterations", "1", "--checkpoint-dir", str(tmp_path / "c")])
    assert code == EXIT_ERROR
    assert "state tensor" in capsys.readouterr().err


# --------------------------------------------------------------------------- AC-2


def test_checkpoint_round_trips_through_gate_resolver(tmp_path):
    """AC-2: a driver checkpoint resolves via its sidecar (not inference) and its state_dict loads."""
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
    # ...and NOT shape inference: the sidecar carries "dropout", which infer never emits.
    assert "dropout" in resolved
    assert resolved != policy_lift.infer_mlp_architecture(state_dict)
    assert resolved["state_dim"] == _REASONING_STATE_DIM
    assert resolved["action_size"] == spec.action_space_size

    # The gate rebuilds the identical network and loads the driver's weights without error.
    rebuilt = build_network(resolved, spec, "cpu", state_dict=state_dict)
    rebuilt.load_state_dict(state_dict)


# --------------------------------------------------------------------------- main()


def test_main_exits_ok(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["self-play-convergence", *_tiny_reasoning_argv(tmp_path / "ckpts")])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == EXIT_OK


def test_main_exits_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "self-play-convergence",
            "--domain",
            "does-not-exist",
            "--iterations",
            "1",
            "--checkpoint-dir",
            str(tmp_path / "c"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == EXIT_ERROR


# --------------------------------------------------------------------------- helpers


def test_checkpoint_name_roundtrips():
    """The filename template and the resume regex must stay in lockstep."""
    for iteration in (1, 7, 42, 1600):
        name = _CHECKPOINT_TEMPLATE.format(n=iteration)
        match = _CHECKPOINT_RE.match(name)
        assert match is not None
        assert int(match.group(1)) == iteration


def test_cuda_memory_fraction_invoked_on_cuda_device(tmp_path, monkeypatch):
    """When device starts with cuda, set_cuda_memory_fraction is called during setup."""
    from unittest.mock import MagicMock

    import src.training.self_play_convergence as spc

    mock_set_mem = MagicMock()
    monkeypatch.setattr(spc, "set_cuda_memory_fraction", mock_set_mem)

    # Mock SelfPlayTrainer.train_iteration to avoid full run
    class DummyMetrics:
        total_loss = 0.1
        policy_loss = 0.05
        value_loss = 0.05
        games_played = 1
        examples_collected = 1
        train_steps = 1
        buffer_size = 1

    async def mock_train(self):
        return DummyMetrics()

    monkeypatch.setattr("src.training.self_play_trainer.SelfPlayTrainer.train_iteration", mock_train)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)  # keep torch from blowing up on fake device

    argv = [
        "--domain",
        "reasoning",
        "--iterations",
        "1",
        "--checkpoint-dir",
        str(tmp_path / "ckpts"),
        "--device",
        "cuda:0",
        "--num-simulations",
        "1",
        "--games-per-iteration",
        "1",
    ]
    code = _run(argv)
    assert code == EXIT_OK
    mock_set_mem.assert_called_once()


# --------------------------------------------------------------------- chess (deferred)


@pytest.mark.skipif(not chess_available(), reason="chess extra (python-chess) not installed")
def test_chess_architecture_resolves_to_conv():
    """AC-1/AC-2 for chess run only where the chess extra exists (deferred in this CI)."""
    spec = DomainRegistry.get("chess")
    arch = resolve_architecture(spec)
    assert arch["type"] == "resnet"
    assert arch["action_size"] == spec.action_space_size
