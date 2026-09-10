"""python -m scripts.local_distillation — sidecar dry-run, promotion, toy compare-arms.

Does not train a model and does not emit Connect Four lift numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from scripts.local_distillation.collector import HygienicCollector, build_mcts
from scripts.local_distillation.device import place_network
from scripts.local_distillation.eval_arms import (
    COMMITTED_RESULTS_RELATIVE_PATH,
    PRIMARY_ENDPOINT,
    compare_search_vs_no_search,
    decide_promotion,
)
from scripts.local_distillation.settings import DistillationSettings, get_distillation_settings
from scripts.local_distillation.sidecar import c4_network_architecture, validate_c4_sidecar
from scripts.local_distillation.toy_domain import ACTION_SPACE, CountingNet, TwoPlyState
from src.observability.logging import get_structured_logger
from src.utils.seeding import new_rng, resolve_seed

logger = get_structured_logger(__name__)


def _sidecar_payload(settings: DistillationSettings) -> dict[str, Any]:
    network = c4_network_architecture(settings)
    validate_c4_sidecar({"network": network}, settings)
    return {
        "network": network,
        "recurrent_enabled": settings.recurrent_enabled,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "committed_results_path": COMMITTED_RESULTS_RELATIVE_PATH,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.local_distillation")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("sidecar", help="print the C4 architecture sidecar (default)")
    promo = sub.add_parser("promote", help="accept/reject a synthetic scalar pair")
    promo.add_argument("--candidate", type=float, required=True)
    promo.add_argument("--incumbent", type=float, required=True)
    promo.add_argument("--min-delta", type=float, default=None)
    compare = sub.add_parser(
        "compare-arms",
        help="toy TwoPly search vs no-search (not a Connect Four lift)",
    )
    compare.add_argument("--simulations", type=int, default=None)
    args = parser.parse_args(argv)

    settings = get_distillation_settings()
    if args.cmd in (None, "sidecar"):
        logger.info("local distillation command", command="sidecar")
        json.dump(_sidecar_payload(settings), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.cmd == "promote":
        logger.info("local distillation command", command="promote")
        min_delta = settings.promotion_min_delta if args.min_delta is None else args.min_delta
        decision = decide_promotion(args.candidate, args.incumbent, min_delta=min_delta)
        json.dump(
            {
                "promote": decision.promote,
                "reason": decision.reason,
                "candidate": decision.candidate,
                "incumbent": decision.incumbent,
                "min_delta": decision.min_delta,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0 if decision.promote else 1
    if args.cmd == "compare-arms":
        return asyncio.run(_compare_arms(settings, simulations=args.simulations))
    parser.error(f"unknown command {args.cmd!r}")
    return 2


async def _compare_arms(settings: DistillationSettings, *, simulations: int | None) -> int:
    """Toy domain only — provenance random-weights, not a C4 golden-path result."""
    sims = simulations if simulations is not None else settings.default_simulations
    logger.info("local distillation command", command="compare-arms", domain="toy_two_ply")
    network = place_network(CountingNet(), settings.device)
    toy_settings = DistillationSettings(
        default_simulations=sims,
        temperature_threshold=settings.temperature_threshold,
        temperature_init=settings.temperature_init,
        temperature_final=settings.temperature_final,
        device=settings.device,
        wall_clock_repeat_cap=settings.wall_clock_repeat_cap,
    )
    rng = new_rng(resolve_seed(None))
    mcts = build_mcts(network, toy_settings, device=settings.device, seed=None, rng=rng, single_agent=False)
    HygienicCollector(mcts, toy_settings, action_space_size=ACTION_SPACE)
    comparison = await compare_search_vs_no_search(
        mcts,
        TwoPlyState(),
        num_simulations=sims,
        device=settings.device,
        repeat_cap=settings.wall_clock_repeat_cap,
    )
    json.dump(
        {
            "primary_endpoint": comparison.primary_endpoint,
            "no_search_expansions": comparison.no_search.expansions,
            "search_expansions": comparison.search.expansions,
            "no_search_wall_clock_s": comparison.no_search.wall_clock_s,
            "search_wall_clock_s": comparison.search.wall_clock_s,
            "no_search_repeats_in_search_budget": comparison.no_search_repeats_in_search_budget,
            "no_search_provenance": comparison.no_search.provenance,
            "search_provenance": comparison.search.provenance,
            "domain": "toy_two_ply",
            "note": "not a Connect Four lift",
        },
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
