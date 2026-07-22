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
  ``win_rate`` domains via :func:`~src.benchmark.policy_lift._chess_default_architecture`,
  MLP for single-agent ``mean_reward`` domains). The resolved architecture dict is
  written verbatim into the sidecar, so the gate reconstructs the *same* network —
  no parallel network-construction path can drift from the gate's resolver.
- **The driver never computes or asserts lift.** Measuring the >=20% gate remains the
  sole responsibility of ``src.benchmark.policy_lift``.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.benchmark.policy_lift import _chess_default_architecture, build_network
from src.config.constants import M5_DEFAULT_MLP_HIDDEN_DIMS
from src.framework.domain_registry import METRIC_WIN_RATE, DomainRegistry, DomainSpec
from src.observability.logging import get_logger
from src.training.self_play_trainer import SelfPlayConfig, SelfPlayTrainer
from src.training.system_config import MCTSConfig

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_ERROR = 2

# Checkpoints are named by the iteration they complete so ``--resume`` can find the
# latest and continue numbering monotonically.
_CHECKPOINT_TEMPLATE = "ckpt_iter_{n}.pt"
_CHECKPOINT_RE = re.compile(r"^ckpt_iter_(\d+)\.pt$")

# A deliberately tiny default: the full-run default (MCTSConfig.num_simulations = 1600)
# is far too expensive for smoke/plumbing runs, so callers opt into depth explicitly.
DEFAULT_NUM_SIMULATIONS = 16


def resolve_architecture(spec: DomainSpec) -> dict[str, Any]:
    """Resolve the network architecture dict for ``spec``, reusing the gate's builders.

    Adversarial ``win_rate`` domains (chess) use the gate's conv/ResNet default;
    single-agent ``mean_reward`` domains use an MLP sized to the domain's 1-D state
    tensor. The returned dict is what gets persisted to the checkpoint sidecar.
    """
    if spec.metric == METRIC_WIN_RATE:
        # Conv/ResNet default for adversarial board domains (chess). Reused from the
        # gate so the driver cannot build an architecture the gate cannot resolve.
        return _chess_default_architecture(spec)

    tensor = spec.initial_state_fn().to_tensor()
    if tensor.dim() != 1:
        raise ValueError(
            f"Domain '{spec.name}' produces a {tensor.dim()}-D state tensor; the MLP driver "
            "path expects a 1-D state vector. Adversarial board domains use the conv path."
        )
    # Batch-norm-free / dropout-free MLP: NeuralMCTS self-play evaluates one state at a
    # time (batch=1) and SelfPlayTrainer does not switch the net to eval() before
    # self-play, so a BatchNorm1d layer would raise on the single-sample batch. (The
    # conv path is unaffected: BatchNorm2d over an 8x8 board has 64 values per channel.)
    # This layout still round-trips through the gate's resolver — the sidecar carries
    # use_batch_norm/dropout, and policy_lift._build_mlp rebuilds the identical net.
    return {
        "type": "mlp",
        "state_dim": int(tensor.shape[0]),
        "hidden_dims": list(M5_DEFAULT_MLP_HIDDEN_DIMS),
        "action_size": spec.action_space_size,
        "use_batch_norm": False,
        "dropout": 0.0,
    }


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


async def run(args: argparse.Namespace) -> int:
    """Execute the self-play driver; returns the process exit code."""
    try:
        spec = DomainRegistry.get(args.domain)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return EXIT_ERROR

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        # ArchitectureError (from the gate) subclasses ValueError, as does the settings
        # error raised when a single-agent domain needs credentials it cannot find.
        arch = resolve_architecture(spec)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    network = build_network(arch, spec, args.device)

    mcts_config = MCTSConfig()
    mcts_config.num_simulations = args.num_simulations
    config_kwargs: dict[str, Any] = {}
    if args.games_per_iteration is not None:
        config_kwargs["num_games_per_iteration"] = args.games_per_iteration

    trainer = SelfPlayTrainer(
        network=network,
        initial_state_fn=spec.initial_state_fn,
        action_space_size=spec.action_space_size,
        mcts_config=mcts_config,
        config=SelfPlayConfig(**config_kwargs),
        single_agent=spec.single_agent,
        device=args.device,
        seed=args.seed,
    )

    start_iteration = 0
    if args.resume:
        latest = _latest_checkpoint(args.checkpoint_dir)
        if latest is None:
            logger.info(
                "No checkpoint to resume from; starting fresh",
                extra={"checkpoint_dir": str(args.checkpoint_dir)},
            )
        else:
            start_iteration, latest_path = latest
            trainer.load_checkpoint(latest_path)
            logger.info(
                "Resumed from checkpoint",
                extra={"iteration": start_iteration, "path": str(latest_path)},
            )

    try:
        for offset in range(1, args.iterations + 1):
            iteration = start_iteration + offset  # monotonic across resumes
            metrics = await trainer.train_iteration()
            checkpoint_path = args.checkpoint_dir / _CHECKPOINT_TEMPLATE.format(n=iteration)
            trainer.save_checkpoint(
                checkpoint_path,
                metadata={
                    "network": arch,
                    "domain": args.domain,
                    "iteration": iteration,
                    "seed": args.seed,
                },
            )
            logger.info(
                "Iteration checkpoint saved",
                extra={
                    "iteration": iteration,
                    "path": str(checkpoint_path),
                    "total_loss": metrics.total_loss,
                    "examples_collected": metrics.examples_collected,
                },
            )
    except OSError as exc:  # unwritable checkpoint dir, disk full, ...
        print(f"error: could not write checkpoint: {exc}", file=sys.stderr)
        return EXIT_ERROR

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
    parser.add_argument("--iterations", type=int, default=1, help="Self-play/train iterations to run this invocation")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory for checkpoints and their .meta.json sidecars",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for network init, self-play, and training")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in --checkpoint-dir (weights only; numbering continues)",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=DEFAULT_NUM_SIMULATIONS,
        help=f"MCTS simulations per move (default: {DEFAULT_NUM_SIMULATIONS}; raise for real chess convergence)",
    )
    parser.add_argument(
        "--games-per-iteration",
        type=int,
        default=None,
        help="Self-play games per iteration (default: SelfPlayConfig's value)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
