"""M5 self-play convergence driver (spec ``m5_policy_lift``, AC-1 / AC-2).

A resumable, seedable, domain-general self-play training driver that produces the
missing input for the M5 decision-quality gate: torch-safe ``state_dict`` checkpoints
paired with ``.meta.json`` architecture sidecars that
``python -m src.benchmark.policy_lift`` can resolve without manual flags.

.. code-block:: bash

    python -m src.training.self_play_convergence --domain chess \\
        --iterations 50 --checkpoint-dir checkpoints/chess --seed 0 --device cuda

Design constraints (from the spec):

- **One construction path.** The network is built through the gate's own
  :func:`src.benchmark.policy_lift.build_network` (conv/ResNet for adversarial
  ``win_rate`` domains via :func:`src.benchmark.policy_lift.chess_default_architecture`,
  MLP for single-agent ``mean_reward`` domains). The resolved architecture dict is
  written verbatim into the sidecar, so the gate reconstructs the *same* network — no
  parallel network-construction path can drift from the gate's resolver. On ``--resume``
  the checkpoint's own sidecar is the authoritative architecture source (mirroring the
  gate), so a change in defaults between runs cannot corrupt the weight load.
- **The driver never computes or asserts lift.** Measuring the >=20% gate remains the
  sole responsibility of ``src.benchmark.policy_lift``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch

from src.benchmark.policy_lift import build_network
from src.config.constants import M5_DEFAULT_MLP_HIDDEN_DIMS, M5_DEFAULT_SELF_PLAY_SIMULATIONS
from src.config.settings import get_settings
from src.framework.domain_registry import DomainRegistry, DomainSpec
from src.observability.logging import configure_cli_logging, get_logger
from src.training.self_play_trainer import SelfPlayConfig, SelfPlayTrainer
from src.training.system_config import MCTSConfig, get_default_device_str
from src.utils.gpu_utils import set_cuda_memory_fraction

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_ERROR = 2

# Checkpoints are named by the iteration they complete so ``--resume`` can find the
# latest and continue numbering monotonically. The template and pattern are paired —
# ``test_checkpoint_name_roundtrips`` guards them against drift.
_CHECKPOINT_TEMPLATE = "ckpt_iter_{n}.pt"
_CHECKPOINT_RE = re.compile(r"^ckpt_iter_(\d+)\.pt$")
SIDECAR_SUFFIX = ".meta.json"

# numpy's np.random.seed accepts a seed in [0, 2**32 - 1]; torch is more lenient but we
# validate against the stricter bound so both seeders succeed.
_MAX_SEED = 2**32 - 1

__all__ = [
    "EXIT_OK",
    "EXIT_ERROR",
    "build_parser",
    "main",
    "resolve_architecture",
    "run",
]


def _positive_int(value: str) -> int:
    """argparse type: a strictly-positive integer (``>= 1``)."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer (>= 1), got {parsed}")
    return parsed


def _seed_int(value: str) -> int:
    """argparse type: a seed within numpy's accepted range ``[0, 2**32 - 1]``."""
    parsed = int(value)
    if not 0 <= parsed <= _MAX_SEED:
        raise argparse.ArgumentTypeError(f"must be in [0, {_MAX_SEED}], got {parsed}")
    return parsed


def resolve_architecture(spec: DomainSpec) -> dict[str, Any]:
    """Resolve the network architecture dict for ``spec``, reusing the gate's builders.

    3-D state tensors (adversarial board domains like chess, connect_four, othello) use ResNet;
    1-D state vectors (single-agent reasoning/planning) use MLP.
    The returned dict is what gets persisted to the checkpoint sidecar.
    """
    tensor = spec.initial_state_fn().to_tensor()
    if tensor.dim() == 3:
        # Conv/ResNet for 3-D board domains
        return {
            "type": "resnet",
            "input_channels": int(tensor.shape[0]),
            "board_rows": int(tensor.shape[1]),
            "board_cols": int(tensor.shape[2]),
            "board_size": int(max(tensor.shape[1], tensor.shape[2])),
            "action_size": spec.action_space_size,
        }

    if tensor.dim() != 1:
        raise ValueError(
            f"Domain '{spec.name}' produces a {tensor.dim()}-D state tensor; expected 1-D for MLP or 3-D for ResNet."
        )

    # Batch-norm-free / dropout-free MLP for 1-D states
    return {
        "type": "mlp",
        "state_dim": int(tensor.shape[0]),
        "hidden_dims": list(M5_DEFAULT_MLP_HIDDEN_DIMS),
        "action_size": spec.action_space_size,
        "use_batch_norm": False,
        "dropout": 0.0,
    }


def _sidecar_architecture(checkpoint: Path) -> dict[str, Any] | None:
    """Return the ``network`` architecture dict from a checkpoint's sidecar, or ``None``.

    ``None`` means the sidecar is absent or carries no ``network`` object, in which case
    the caller falls back to :func:`resolve_architecture`.
    """
    sidecar = checkpoint.with_name(checkpoint.name + SIDECAR_SUFFIX)
    if not sidecar.is_file():
        return None
    meta = json.loads(sidecar.read_text())  # JSONDecodeError surfaces as a clean error exit
    network = meta.get("network") if isinstance(meta, dict) else None
    return network if isinstance(network, dict) else None


def _latest_checkpoint(checkpoint_dir: Path) -> tuple[int, Path] | None:
    """Return ``(iteration, path)`` for the highest-numbered checkpoint, or ``None``."""
    if not checkpoint_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for path in checkpoint_dir.iterdir():
        match = _CHECKPOINT_RE.match(path.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        if best is None or iteration > best[0]:
            best = (iteration, path)
    return best


def _self_play_config(
    games_per_iteration: int | None = None,
    use_amp: bool = False,
    compile_model: bool = False,
    pin_memory: bool = False,
) -> SelfPlayConfig:
    """Build the self-play config with options for GPU parameters."""
    cfg = SelfPlayConfig(
        use_amp=use_amp,
        compile_model=compile_model,
        pin_memory=pin_memory,
    )
    if games_per_iteration is not None:
        cfg.num_games_per_iteration = games_per_iteration
    return cfg


async def run(args: argparse.Namespace) -> int:
    """Execute the self-play driver; returns the process exit code (0 ok, 2 error)."""
    try:
        spec = DomainRegistry.get(args.domain)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return EXIT_ERROR

    # Apply profile preset if supplied
    if getattr(args, "profile", None) is not None:
        from src.training.training_config import get_training_profile

        profile_spec = get_training_profile(args.profile)
        if args.num_simulations == M5_DEFAULT_SELF_PLAY_SIMULATIONS:
            args.num_simulations = profile_spec.num_simulations
        if args.games_per_iteration is None:
            args.games_per_iteration = profile_spec.games_per_iteration
        if args.device is None:
            args.device = profile_spec.resolved_device()
        if not args.mixed_precision:
            args.mixed_precision = profile_spec.use_amp
        if not args.compile:
            args.compile = profile_spec.compile_model

    settings = get_settings()
    from src.utils import distributed

    world_size = distributed.get_world_size(default=settings.TRAINING_WORLD_SIZE)
    is_distributed_run = settings.TRAINING_DISTRIBUTED or world_size > 1

    if is_distributed_run:
        distributed.init_distributed(backend=settings.TRAINING_BACKEND)
        if torch.cuda.is_available():
            args.device = f"cuda:{distributed.get_local_rank()}"

    if args.device is None:
        args.device = "cpu"
    elif args.device == "auto":
        args.device = get_default_device_str()

    if args.device.startswith("cuda"):
        set_cuda_memory_fraction(settings.TRAINING_CUDA_MEMORY_FRACTION)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build the network + trainer and (optionally) load a resume checkpoint.
    try:
        resume_from = _latest_checkpoint(args.checkpoint_dir) if args.resume else None
        arch = _sidecar_architecture(resume_from[1]) if resume_from is not None else None
        if arch is None:
            arch = resolve_architecture(spec)

        mcts_config = MCTSConfig()
        mcts_config.num_simulations = args.num_simulations
        sp_config = _self_play_config(
            games_per_iteration=args.games_per_iteration,
            use_amp=args.mixed_precision,
            compile_model=args.compile,
            pin_memory=args.device.startswith("cuda") and get_settings().TRAINING_PIN_MEMORY,
        )
        trainer = SelfPlayTrainer(
            network=build_network(arch, spec, args.device),
            initial_state_fn=spec.initial_state_fn,
            action_space_size=spec.action_space_size,
            mcts_config=mcts_config,
            config=sp_config,
            single_agent=spec.single_agent,
            device=args.device,
            seed=args.seed,
        )

        start_iteration = 0
        if resume_from is not None:
            start_iteration, latest_path = resume_from
            trainer.load_checkpoint(latest_path)
            logger.info(
                "Resumed from checkpoint",
                extra={"iteration": start_iteration, "path": str(latest_path)},
            )
        elif args.resume:
            logger.info(
                "No checkpoint to resume from; starting fresh",
                extra={"checkpoint_dir": str(args.checkpoint_dir)},
            )
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    logger.info(
        "Self-play convergence run starting",
        extra={
            "domain": args.domain,
            "metric": spec.metric,
            "single_agent": spec.single_agent,
            "iterations": args.iterations,
            "num_simulations": args.num_simulations,
            "games_per_iteration": args.games_per_iteration,
            "device": args.device,
            "seed": args.seed,
            "checkpoint_dir": str(args.checkpoint_dir),
            "resume": args.resume,
            "start_iteration": start_iteration,
        },
    )
    logger.debug("Resolved network architecture", extra={"architecture": arch})

    for offset in range(1, args.iterations + 1):
        iteration = start_iteration + offset  # monotonic across resumes
        try:
            metrics = await trainer.train_iteration()
        except RuntimeError as exc:  # CUDA OOM, shape errors, ... — keep the trace in logs
            logger.exception("Self-play training iteration failed", extra={"iteration": iteration})
            print(f"error: training failed at iteration {iteration}: {exc}", file=sys.stderr)
            return EXIT_ERROR

        checkpoint_path = args.checkpoint_dir / _CHECKPOINT_TEMPLATE.format(n=iteration)
        try:
            trainer.save_checkpoint(
                checkpoint_path,
                metadata={
                    "network": arch,
                    "domain": args.domain,
                    "iteration": iteration,
                    "seed": args.seed,
                },
            )
        except OSError as exc:  # unwritable checkpoint dir, disk full, ...
            print(f"error: could not write checkpoint: {exc}", file=sys.stderr)
            return EXIT_ERROR

        logger.info(
            "Iteration checkpoint saved",
            extra={
                "iteration": iteration,
                "path": str(checkpoint_path),
                "total_loss": metrics.total_loss,
                "policy_loss": metrics.policy_loss,
                "value_loss": metrics.value_loss,
                "games_played": metrics.games_played,
                "examples_collected": metrics.examples_collected,
                "train_steps": metrics.train_steps,
                "buffer_size": metrics.buffer_size,
            },
        )

    logger.info(
        "Self-play convergence run complete",
        extra={
            "final_iteration": start_iteration + args.iterations,
            "checkpoint_dir": str(args.checkpoint_dir),
        },
    )

    if is_distributed_run:
        distributed.cleanup_distributed()

    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="self-play-convergence",
        description=(
            "M5 self-play convergence driver: train a policy-value network by self-play on a "
            "registered domain, writing torch-safe checkpoints plus .meta.json architecture "
            "sidecars for the policy-lift gate. Resumable and seedable; never computes lift."
        ),
    )
    parser.add_argument("--domain", required=True, help="Registered domain name (e.g. chess, reasoning)")
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=1,
        help="Self-play/train iterations to run this invocation (>= 1)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory for checkpoints and their .meta.json sidecars",
    )
    parser.add_argument("--seed", type=_seed_int, default=0, help="Seed for network init, self-play, and training")
    parser.add_argument("--device", default=None, help="Torch device (default: cpu)")
    parser.add_argument(
        "--profile",
        choices=["smoke", "dev", "full"],
        default=None,
        help="Training profile preset (smoke, dev, full)",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=False,
        help="Enable FP16 mixed precision training",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Enable PyTorch 2.0 torch.compile model compilation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in --checkpoint-dir (weights only; numbering continues)",
    )
    parser.add_argument(
        "--num-simulations",
        type=_positive_int,
        default=M5_DEFAULT_SELF_PLAY_SIMULATIONS,
        help=(
            f"MCTS simulations per move (>= 1; default: {M5_DEFAULT_SELF_PLAY_SIMULATIONS}; "
            "raise substantially for real chess convergence)"
        ),
    )
    parser.add_argument(
        "--games-per-iteration",
        type=_positive_int,
        default=None,
        help="Self-play games per iteration (>= 1; default: SelfPlayConfig's value)",
    )
    return parser


def main() -> None:
    # Without this the driver is silent: get_logger returns an unconfigured `mcts.*`
    # logger, so the resolved device, the seed, the per-iteration losses and the
    # checkpoint paths are all discarded, and a failed run cannot be diagnosed
    # afterwards. Writes to stderr so stdout stays free for the command's own output.
    configure_cli_logging()
    args = build_parser().parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
