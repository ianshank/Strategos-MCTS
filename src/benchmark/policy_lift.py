"""M5 policy-lift CLI: run the baseline-vs-trained decision-quality gate.

Wires :func:`src.benchmark.policy_comparison.compare_policies` to the command line so
the M5 ">=20% decision-quality lift" acceptance criterion is reproducible:

.. code-block:: bash

    python -m src.benchmark.policy_lift --domain chess \\
        --checkpoint checkpoints/trained.pt --num-games 100 --output lift.json

The gate is the **confidence-interval lower bound** clearing the target (fail-closed),
and the exit code is the verdict: ``0`` gate met, ``1`` gate not met, ``2`` usage or
loading error — so the command can serve directly as a CI step.

Checkpoints are torch-safe ``state_dict`` files (no architecture info). The network is
reconstructed from, in priority order: ``--network-config``/``--input-dim`` CLI args, a
``<checkpoint>.meta.json`` sidecar (written by
:meth:`~src.training.self_play_trainer.SelfPlayTrainer.save_checkpoint` when metadata is
passed), shape inference for :class:`~src.models.policy_value_net.MLPPolicyValueNetwork`
state dicts, or the conv defaults for adversarial board domains (chess).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.benchmark.policy_comparison import (
    DEFAULT_MIN_BASELINE,
    DEFAULT_TARGET_LIFT_PCT,
    PolicyComparisonResult,
    compare_policies,
)
from src.config.constants import (
    M5_DEFAULT_ADVERSARIAL_BOARD_SIZE,
    M5_DEFAULT_CONFIDENCE,
    M5_DEFAULT_MLP_HIDDEN_DIMS,
)
from src.framework.domain_registry import DomainRegistry, DomainSpec
from src.models.policy_value_net import MLPPolicyValueNetwork, create_policy_value_network
from src.observability.logging import get_logger
from src.training.system_config import MCTSConfig, NeuralNetworkConfig

logger = get_logger(__name__)

EXIT_GATE_MET = 0
EXIT_GATE_NOT_MET = 1
EXIT_ERROR = 2

SIDECAR_SUFFIX = ".meta.json"

_SHARED_LINEAR_KEY = re.compile(r"^shared_network\.(\d+)\.weight$")
_POLICY_LINEAR_KEY = re.compile(r"^policy_head\.(\d+)\.weight$")


class ArchitectureError(ValueError):
    """The network architecture could not be determined for a checkpoint."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="policy-lift",
        description=(
            "M5 decision-quality gate: compare a trained checkpoint against an untrained "
            "(or explicit) baseline on a registered domain. Exit code 0 = CI lower bound "
            "clears the target, 1 = gate not met, 2 = error."
        ),
    )
    parser.add_argument("--domain", required=True, help="Registered domain name (e.g. chess, reasoning)")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trained state_dict checkpoint (.pt)")
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=None,
        help="Optional baseline checkpoint; defaults to a fresh seeded untrained network",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=None,
        help="Games per side (default: 100 for win-rate domains, 30 for mean-reward)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for baseline init and rollouts")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    parser.add_argument(
        "--max-moves",
        type=int,
        default=50,
        help="Move cap per mean-reward rollout (default: 50; adversarial games end by the domain's own rules)",
    )
    parser.add_argument(
        "--num-simulations", type=int, default=None, help="MCTS simulations per move (default: MCTSConfig)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=M5_DEFAULT_CONFIDENCE,
        choices=[0.90, 0.95, 0.99],
        help=f"Confidence level for the lift interval (default: {M5_DEFAULT_CONFIDENCE})",
    )
    parser.add_argument(
        "--target-lift",
        type=float,
        default=DEFAULT_TARGET_LIFT_PCT,
        help="Lift (%%) the CI lower bound must clear (default: 20)",
    )
    parser.add_argument(
        "--min-baseline",
        type=float,
        default=DEFAULT_MIN_BASELINE,
        help="Baseline floor below which lift is reported in absolute points (default: 0.05)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write the JSON artifact to this path")
    parser.add_argument(
        "--network-config",
        type=Path,
        default=None,
        help='JSON architecture spec, e.g. {"type": "mlp", "state_dim": 64, "hidden_dims": [512, 256]}',
    )
    parser.add_argument("--input-dim", type=int, default=None, help="MLP state dim (with --hidden-dims)")
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=None, help="MLP hidden layer sizes (with --input-dim)"
    )
    return parser


def infer_mlp_architecture(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Infer an MLPPolicyValueNetwork architecture from its (self-describing) state_dict.

    Raises:
        ArchitectureError: when the keys do not match the MLP layout (e.g. conv nets).
    """
    shared_linears = sorted(
        (int(match.group(1)), key)
        for key in state_dict
        if (match := _SHARED_LINEAR_KEY.match(key)) and state_dict[key].dim() == 2
    )
    policy_linears = sorted(
        (int(match.group(1)), key)
        for key in state_dict
        if (match := _POLICY_LINEAR_KEY.match(key)) and state_dict[key].dim() == 2
    )
    if not shared_linears or not policy_linears:
        raise ArchitectureError(
            "Checkpoint does not match the MLPPolicyValueNetwork layout; pass --network-config "
            "(or --input-dim/--hidden-dims) to describe the architecture."
        )

    first_weight = state_dict[shared_linears[0][1]]
    state_dim = int(first_weight.shape[1])
    hidden_dims = [int(state_dict[key].shape[0]) for _, key in shared_linears]
    action_size = int(state_dict[policy_linears[-1][1]].shape[0])
    use_batch_norm = any(key.startswith("shared_network.") and key.endswith(".running_mean") for key in state_dict)

    return {
        "type": "mlp",
        "state_dim": state_dim,
        "hidden_dims": hidden_dims,
        "action_size": action_size,
        "use_batch_norm": use_batch_norm,
    }


def _build_mlp(arch: dict[str, Any], state_dict: dict[str, torch.Tensor] | None) -> nn.Module:
    """Build an MLP net from an architecture dict, resolving the dropout-layer ambiguity.

    Dropout layers carry no parameters but shift the Sequential indices of later
    layers, so when loading a checkpoint we pick the dropout setting whose key set
    matches the state_dict.
    """
    dropouts = [float(arch["dropout"])] if "dropout" in arch else [0.1, 0.0]
    last_error: Exception | None = None
    for dropout in dropouts:
        network = MLPPolicyValueNetwork(
            state_dim=int(arch["state_dim"]),
            action_size=int(arch["action_size"]),
            hidden_dims=[int(dim) for dim in arch.get("hidden_dims") or M5_DEFAULT_MLP_HIDDEN_DIMS],
            use_batch_norm=bool(arch.get("use_batch_norm", True)),
            dropout=dropout,
        )
        if state_dict is None or set(network.state_dict()) == set(state_dict):
            return network
        last_error = ArchitectureError(
            "MLP layer indices do not match the checkpoint (dropout/batch-norm layout mismatch); "
            "pass --network-config with explicit 'dropout'/'use_batch_norm'."
        )
    raise last_error if last_error is not None else ArchitectureError("empty architecture candidates")


def build_network(
    arch: dict[str, Any],
    spec: DomainSpec,
    device: str,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> nn.Module:
    """Construct a network from an architecture dict (``type``: ``mlp`` | ``resnet``)."""
    arch_type = str(arch.get("type", "mlp"))
    if arch_type == "mlp":
        merged = dict(arch)
        merged.setdefault("action_size", spec.action_space_size)
        network = _build_mlp(merged, state_dict)
    elif arch_type == "resnet":
        # Unspecified fields fall back to NeuralNetworkConfig's own dataclass defaults
        # rather than re-hardcoding them here.
        config = NeuralNetworkConfig(
            num_res_blocks=int(arch.get("num_res_blocks", NeuralNetworkConfig.num_res_blocks)),
            num_channels=int(arch.get("num_channels", NeuralNetworkConfig.num_channels)),
            input_channels=int(arch["input_channels"]),
            action_size=int(arch.get("action_size", spec.action_space_size)),
        )
        board_size = int(arch.get("board_size", M5_DEFAULT_ADVERSARIAL_BOARD_SIZE))
        network = create_policy_value_network(config, board_size=board_size, device=device)
    else:
        raise ArchitectureError(f"Unknown network architecture type '{arch_type}' (expected 'mlp' or 'resnet')")
    return network.to(device)


def chess_default_architecture(spec: DomainSpec) -> dict[str, Any]:
    """Conv defaults for adversarial board domains (currently chess-shaped)."""
    try:
        from src.games.chess.config import ChessBoardConfig

        input_channels = ChessBoardConfig().total_planes
    except ImportError as exc:  # pragma: no cover - chess extra not installed
        raise ArchitectureError(
            "Cannot derive a default conv architecture without the chess extra; pass --network-config."
        ) from exc
    return {
        "type": "resnet",
        "input_channels": input_channels,
        "action_size": spec.action_space_size,
        "board_size": M5_DEFAULT_ADVERSARIAL_BOARD_SIZE,
    }


def load_architecture(
    checkpoint: Path,
    args: argparse.Namespace,
    spec: DomainSpec,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Resolve the architecture dict for a checkpoint (see module docstring for order)."""
    if args.network_config is not None:
        loaded = json.loads(args.network_config.read_text())
        if not isinstance(loaded, dict):
            raise ArchitectureError(f"--network-config must contain a JSON object, got {type(loaded).__name__}")
        return loaded
    if args.input_dim is not None:
        return {
            "type": "mlp",
            "state_dim": args.input_dim,
            "hidden_dims": args.hidden_dims or list(M5_DEFAULT_MLP_HIDDEN_DIMS),
            "action_size": spec.action_space_size,
        }
    sidecar = checkpoint.with_name(checkpoint.name + SIDECAR_SUFFIX)
    if sidecar.exists():
        meta = json.loads(sidecar.read_text())
        network_meta = meta.get("network") if isinstance(meta, dict) else None
        if isinstance(network_meta, dict):
            logger.info("Architecture loaded from sidecar", extra={"sidecar": str(sidecar)})
            return network_meta
        logger.warning("Sidecar has no 'network' object; falling back", extra={"sidecar": str(sidecar)})
    try:
        arch = infer_mlp_architecture(state_dict)
    except ArchitectureError:
        if spec.metric == "win_rate":
            return chess_default_architecture(spec)
        raise
    if arch["action_size"] != spec.action_space_size:
        raise ArchitectureError(
            f"Checkpoint policy head has {arch['action_size']} actions but domain "
            f"'{spec.name}' expects {spec.action_space_size}; wrong checkpoint or domain?"
        )
    return arch


def _load_state_dict(path: Path, device: str) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ArchitectureError(
            f"Checkpoint {path} is not a state_dict of tensors (torch-safe state_dict format required)"
        )
    return state


def result_to_dict(result: PolicyComparisonResult, run_info: dict[str, Any]) -> dict[str, Any]:
    """Serialize a comparison result plus run provenance into the JSON artifact."""
    payload = asdict(result)
    payload["meets_target"] = result.meets_target
    payload["point_meets_target"] = result.point_meets_target
    payload["run"] = run_info
    return payload


async def run(args: argparse.Namespace) -> int:
    """Execute the comparison; returns the process exit code."""
    if args.hidden_dims is not None and args.input_dim is None and args.network_config is None:
        print("error: --hidden-dims requires --input-dim (or use --network-config)", file=sys.stderr)
        return EXIT_ERROR

    try:
        spec = DomainRegistry.get(args.domain)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return EXIT_ERROR

    logger.info(
        "Policy-lift run starting",
        extra={
            "domain": args.domain,
            "metric": spec.metric,
            "checkpoint": str(args.checkpoint),
            "baseline_checkpoint": str(args.baseline_checkpoint) if args.baseline_checkpoint else None,
            "num_games": args.num_games,
            "seed": args.seed,
            "device": args.device,
        },
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        trained_state = _load_state_dict(args.checkpoint, args.device)
        arch = load_architecture(args.checkpoint, args, spec, trained_state)
        logger.debug("Network architecture resolved", extra={"architecture": arch})
        trained_network = build_network(arch, spec, args.device, state_dict=trained_state)
        trained_network.load_state_dict(trained_state)
        trained_network.eval()

        # Baseline: explicit checkpoint, or a fresh (seeded) untrained instance of the
        # same architecture — the "untrained policy" of the M5 acceptance criterion.
        # The baseline network is ALWAYS built from the trained checkpoint's resolved
        # architecture: an explicit baseline checkpoint with a different layout must
        # fail fast (load_state_dict raises), not silently compare mismatched nets.
        baseline_network = build_network(arch, spec, args.device, state_dict=trained_state)
        if args.baseline_checkpoint is not None:
            baseline_state = _load_state_dict(args.baseline_checkpoint, args.device)
            try:
                baseline_network.load_state_dict(baseline_state)
            except RuntimeError as exc:
                raise ArchitectureError(
                    f"Baseline checkpoint {args.baseline_checkpoint} does not match the trained "
                    f"checkpoint's architecture: {exc}"
                ) from exc
        baseline_network.eval()
    except (ArchitectureError, FileNotFoundError, RuntimeError, json.JSONDecodeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    mcts_config = MCTSConfig()
    if args.num_simulations is not None:
        mcts_config.num_simulations = args.num_simulations

    try:
        result = await compare_policies(
            args.domain,
            baseline_network,
            trained_network,
            num_games=args.num_games,
            mcts_config=mcts_config,
            max_moves=args.max_moves,
            device=args.device,
            confidence=args.confidence,
            min_baseline=args.min_baseline,
            target_lift_pct=args.target_lift,
        )
    except ValueError as exc:  # invalid num_games, zero-game evaluation runs, ...
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    artifact = result_to_dict(
        result,
        run_info={
            "seed": args.seed,
            "device": args.device,
            "num_simulations": mcts_config.num_simulations,
            "max_moves": args.max_moves,
            "checkpoint": str(args.checkpoint),
            "baseline_checkpoint": str(args.baseline_checkpoint) if args.baseline_checkpoint else None,
            "network": arch,
        },
    )
    rendered = json.dumps(artifact, indent=2, sort_keys=True)
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n")
        except OSError as exc:  # permission denied, path is a directory, disk full, ...
            print(rendered)
            print(f"error: could not write artifact to {args.output}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        logger.info("Lift artifact written", extra={"path": str(args.output)})
    print(rendered)

    logger.info(
        "Policy-lift run complete",
        extra={
            "domain": result.domain,
            "lift_pct": result.lift_pct,
            "lift_ci_lower_pct": result.lift_ci_lower_pct,
            "meets_target": result.meets_target,
        },
    )
    return EXIT_GATE_MET if result.meets_target else EXIT_GATE_NOT_MET


def main() -> None:
    args = build_parser().parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
