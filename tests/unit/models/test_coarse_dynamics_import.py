"""Torch-optionality tests for coarse_dynamics (strategos_coarse_dynamics_mdn AC-3).

The module must import without torch (aggregator + numpy reference usable), and constructing the
MDN without torch must raise a clear error — never a silent no-op.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from src.models.coarse_dynamics import _TORCH_AVAILABLE, CoarseDynamicsMDN


def test_module_imports_without_torch():  # AC-3
    code = (
        "import importlib.util, sys\n"
        "import src.models.coarse_dynamics as m\n"
        "assert hasattr(m, 'CoarseTransitionAggregator')\n"
        "assert hasattr(m, 'mixture_variance_trace')\n"
        "if importlib.util.find_spec('torch') is None:\n"
        "    tmods = sorted(x for x in sys.modules if x == 'torch' or x.startswith('torch.'))\n"
        "    assert not tmods, tmods\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"


@pytest.mark.skipif(_TORCH_AVAILABLE, reason="verifies the torch-absent construction guard")
def test_mdn_construction_without_torch_raises():  # AC-3
    with pytest.raises(RuntimeError, match="requires PyTorch"):
        CoarseDynamicsMDN(input_dim=8)
