"""AC-4 gate assertion: the committed chess artifact records the >=20% gate as MET.

Guarded by ``skipif`` until the operator commits the chess artifact from the GPU run, so a
not-yet-met (exit 1) artifact never breaks CI. Per the spec, this test lands together with the
``verified`` status flip (spec-trace rule (d) needs the AC mapping below).

Covers m5_policy_lift AC-4
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.benchmark]

_CHESS_ARTIFACT = Path(__file__).resolve().parents[3] / "benchmarks" / "results" / "m5_policy_lift.json"


@pytest.mark.skipif(
    not _CHESS_ARTIFACT.is_file(),
    reason="chess gate artifact not recorded yet (operator GPU run)",
)
def test_chess_lift_artifact_meets_target():
    """AC-4: the committed chess artifact's 95% CI lower bound clears +20% (meets_target true)."""
    artifact = json.loads(_CHESS_ARTIFACT.read_text())
    assert artifact["domain"] == "chess"
    assert artifact["meets_target"] is True
