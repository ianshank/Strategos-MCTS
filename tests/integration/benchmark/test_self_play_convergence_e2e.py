"""End-to-end integration: the self-play driver feeds the policy-lift gate (no mocks).

Builds a real (tiny) checkpoint via ``src.training.self_play_convergence`` on the synthetic
``reasoning`` domain, then runs the ``src.benchmark.policy_lift`` gate against it, asserting the
full driver -> checkpoint -> ``.meta.json`` sidecar -> gate -> artifact path. This exercises the
real neural-MCTS path end-to-end with a real (untrained) torch network — not components against
mocks. Chess (the real adversarial gate domain) is the operator's GPU run and is out of scope here.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from src.benchmark import policy_lift
from src.training import self_play_convergence

pytestmark = [pytest.mark.integration, pytest.mark.training]


def test_driver_checkpoint_feeds_policy_lift_gate(tmp_path):
    checkpoint_dir = tmp_path / "ckpts"
    driver_args = self_play_convergence.build_parser().parse_args(
        [
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
            "--games-per-iteration",
            "1",
        ]
    )
    assert asyncio.run(self_play_convergence.run(driver_args)) == self_play_convergence.EXIT_OK

    checkpoint = checkpoint_dir / "ckpt_iter_1.pt"
    assert checkpoint.is_file()

    output = tmp_path / "lift.json"
    gate_args = policy_lift.build_parser().parse_args(
        [
            "--domain",
            "reasoning",
            "--checkpoint",
            str(checkpoint),
            "--num-games",
            "2",
            "--num-simulations",
            "2",
            "--max-moves",
            "10",
            "--seed",
            "0",
            "--output",
            str(output),
        ]
    )
    exit_code = asyncio.run(policy_lift.run(gate_args))

    # The gate reaches a verdict (0 met / 1 not met) — never an architecture/loading error (2):
    # proves the driver's sidecar round-trips through the gate's resolver.
    assert exit_code in (policy_lift.EXIT_GATE_MET, policy_lift.EXIT_GATE_NOT_MET)

    artifact = json.loads(output.read_text())
    assert artifact["domain"] == "reasoning"
    assert artifact["metric"] == "mean_reward"
    # Meaningful CI relation (not just "not None", which is always true for mean_reward).
    assert artifact["lift_ci_lower_pct"] <= artifact["lift_pct"] <= artifact["lift_ci_upper_pct"]
    assert artifact["run"]["network"]["type"] == "mlp"
    assert artifact["run"]["num_simulations"] == 2
