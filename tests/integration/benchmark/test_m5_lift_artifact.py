"""AC-3 artifact schema + provenance checks for the M5 policy-lift gate.

The committed reasoning smoke artifact is asserted now, so the artifact schema/provenance path
is exercised immediately. The chess-specific acceptance checks (AC-3: domain ``chess``, >=100
games, committed at ``benchmarks/results/m5_policy_lift.json``) are guarded behind ``skipif`` until
the operator's chess run lands, so this file cannot silently bit-rot.

Deferred on purpose:

- **AC-4** (``meets_target`` is true) is NOT asserted here. It lands in ``test_m5_lift_gate.py``
  together with the spec ``verified`` flip, so a committed *not-yet-met* (exit 1) chess artifact
  never breaks CI.
- The operator produces the chess checkpoint with the ``chess`` extra + a GPU::

      python -m src.training.self_play_convergence --domain chess --iterations <N> \\
          --checkpoint-dir <dir> --seed <s> --device cuda --num-simulations <400-800>
      python -m src.benchmark.policy_lift --domain chess --checkpoint <dir>/ckpt_iter_<N>.pt \\
          --num-games 100 --output benchmarks/results/m5_policy_lift.json

Covers m5_policy_lift AC-3 (spec-trace verified-flip mapping).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.benchmark]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REASONING_ARTIFACT = _REPO_ROOT / "benchmarks" / "results" / "reasoning_smoke_lift.json"
_CHESS_ARTIFACT = _REPO_ROOT / "benchmarks" / "results" / "m5_policy_lift.json"

# The full artifact key set (PolicyComparisonResult + the two *_meets_target props + run provenance).
_ARTIFACT_KEYS = {
    "absolute_delta",
    "baseline_score",
    "confidence",
    "domain",
    "lift_ci_lower_pct",
    "lift_ci_upper_pct",
    "lift_is_absolute_fallback",
    "lift_pct",
    "meets_target",
    "metric",
    "num_games",
    "point_meets_target",
    "run",
    "target_lift_pct",
    "trained_score",
}
_RUN_KEYS = {"seed", "device", "num_simulations", "max_moves", "checkpoint", "baseline_checkpoint", "network"}


def _assert_artifact_shape(artifact: dict) -> None:
    assert set(artifact) == _ARTIFACT_KEYS
    assert set(artifact["run"]) == _RUN_KEYS
    assert isinstance(artifact["meets_target"], bool)
    assert isinstance(artifact["point_meets_target"], bool)
    assert artifact["confidence"] == 0.95
    assert artifact["target_lift_pct"] == 20.0


def test_reasoning_smoke_artifact_schema_and_provenance():
    assert _REASONING_ARTIFACT.is_file(), f"expected committed artifact at {_REASONING_ARTIFACT}"
    artifact = json.loads(_REASONING_ARTIFACT.read_text())
    _assert_artifact_shape(artifact)
    assert artifact["domain"] == "reasoning"
    assert artifact["metric"] == "mean_reward"
    assert artifact["run"]["network"]["type"] == "mlp"
    assert artifact["run"]["num_simulations"] == 16


@pytest.mark.skipif(not _CHESS_ARTIFACT.is_file(), reason="chess gate artifact not recorded yet (operator GPU run)")
def test_chess_lift_artifact_schema_and_provenance():
    """AC-3: the committed chess artifact carries the gate schema + provenance over >=100 games.

    Does NOT assert ``meets_target`` is true — that is AC-4 (``test_m5_lift_gate.py`` + the
    ``verified`` flip), kept separate so a not-yet-met chess artifact never breaks CI.
    """
    artifact = json.loads(_CHESS_ARTIFACT.read_text())
    _assert_artifact_shape(artifact)
    assert artifact["domain"] == "chess"
    assert artifact["metric"] == "win_rate"
    assert artifact["num_games"] >= 100
