"""Neural MCTS search across the device matrix.

``CHARTER.md`` INV-9 promises that "CPU-only and single-GPU paths keep working", and
NG-6 forbids breaking either. Nothing in the tree measured that: before this module no
test in the repository was parametrized over devices at all, so the promise rested on CI
happening to run CPU-only.

Two properties, kept apart on purpose:

* **Per device** — a real domain state goes through a real policy-value network and a
  real search, and the search returns a usable policy. Run on every device the host has.
* **Across devices** — the network's forward pass on an accelerator agrees with the same
  weights on CPU. This is the property that actually breaks when a tensor is left on the
  wrong device, and it needs *two* devices, so it lives in its own test which collects as
  a single reasoned skip on a CPU-only host rather than silently passing.

Search reproducibility is asserted on the same device only. Cross-device bitwise equality
is not a property the framework claims (reduction order differs between kernels), and
asserting it would manufacture a failure rather than find one.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
import torch

from src.benchmark.policy_lift import build_network
from src.framework.domain_registry import DomainRegistry
from src.framework.mcts.neural_mcts import NeuralMCTS
from src.training.self_play_convergence import resolve_architecture
from src.training.system_config import MCTSConfig
from tests.utils.device_matrix import CPU_DEVICE, DeviceCase

pytestmark = [pytest.mark.e2e, pytest.mark.mcts, pytest.mark.neural]

#: An adversarial domain with a real board tensor, so the ResNet path is exercised.
DOMAIN = "connect_four"

#: Enough simulations to grow a tree past the root, few enough to stay a plumbing test.
SIMULATIONS = 8

#: Tolerance for the cross-device forward-pass comparison. Accelerator kernels reduce in
#: a different order than CPU, so exact equality is the wrong assertion; TF32 is disabled
#: alongside this so the comparison is fp32-against-fp32 rather than silently 10-bit.
CROSS_DEVICE_ATOL = 1e-4
CROSS_DEVICE_RTOL = 1e-4


def _build_search(device: str) -> tuple[NeuralMCTS, Any]:
    """A real network and search for ``DOMAIN``, both placed on ``device``."""
    spec = DomainRegistry.get(DOMAIN)
    network = build_network(resolve_architecture(spec), spec, device)
    network.eval()
    config = MCTSConfig()
    config.num_simulations = SIMULATIONS
    search = NeuralMCTS(network, config, device=device, single_agent=spec.single_agent)
    return search, spec


def _seed_everything(seed: int) -> None:
    """Seed both RNGs the search actually consumes.

    Torch alone is not enough, and that is a property of the code under test rather than
    of this test: ``NeuralMCTS`` draws its root Dirichlet noise and its stochastic action
    samples from the **process-global NumPy RNG**, so a torch-only seed leaves the search
    irreproducible. ``src/training/self_play_convergence.py`` seeds both for exactly this
    reason, so seeding both here reproduces the driver's real configuration.

    ``specs/hygiene_determinism.SPEC.md`` AC-3 (approved, unimplemented) replaces that
    global draw with an injected generator. When it lands this helper should shrink to
    passing a seeded generator into the engine, and the coupling to global state that
    makes the line below necessary disappears.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def test_search_produces_a_usable_policy_on_every_device(device_case: DeviceCase) -> None:
    """A real search on ``device`` returns a normalized policy over the legal actions."""
    search, spec = _build_search(device_case.name)
    state = spec.initial_state_fn()

    _seed_everything(0)
    action_probs, root = asyncio.run(search.search(state))

    legal = set(state.get_legal_actions())
    assert legal, "the domain's initial state reports no legal actions"
    assert set(action_probs) <= legal, (
        f"search on {device_case.name!r} returned actions outside the legal set: "
        f"{sorted(set(action_probs) - legal)}"
    )
    assert action_probs, f"search on {device_case.name!r} returned an empty policy"
    assert pytest.approx(1.0, abs=1e-6) == sum(action_probs.values())
    assert all(prob >= 0.0 for prob in action_probs.values())

    # The tree really grew: the root was visited once per simulation.
    assert (
        root.visit_count == SIMULATIONS
    ), f"expected {SIMULATIONS} root visits on {device_case.name!r}, got {root.visit_count}"
    assert root.children, "the root was never expanded"


def test_same_device_same_seed_search_is_reproducible(device_case: DeviceCase) -> None:
    """Repeating a seeded search on one device reproduces the policy exactly.

    Seeded through :func:`_seed_everything`, which is what the self-play driver does. A
    torch-only seed fails this assertion today — see that helper for why, and for the
    approved spec that removes the dependency.
    """
    search, spec = _build_search(device_case.name)
    state = spec.initial_state_fn()

    _seed_everything(0)
    first, _ = asyncio.run(search.search(state))

    # A warm evaluation cache would make the second run trivially equal, so it is cleared:
    # the assertion must cover the network path, not the dictionary in front of it.
    search.clear_cache()
    _seed_everything(0)
    second, _ = asyncio.run(search.search(state))

    assert first == second, f"a seeded search on {device_case.name!r} was not reproducible across runs in one process"


def test_accelerator_forward_pass_agrees_with_cpu(accelerator_case: DeviceCase) -> None:
    """The same weights produce the same policy and value on an accelerator as on CPU.

    Collects as one skip with a reason on a CPU-only host, which is how the report says
    "this property was not verified here" instead of staying silent about it.
    """
    # fp32 against fp32: TF32 would compare a 10-bit mantissa against a 24-bit one and
    # make the tolerance a statement about Ampere, not about this code.
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        spec = DomainRegistry.get(DOMAIN)
        architecture = resolve_architecture(spec)

        torch.manual_seed(0)
        cpu_network = build_network(architecture, spec, CPU_DEVICE)
        cpu_network.eval()

        # Same weights, moved — not a second random initialization.
        accelerator_network = build_network(architecture, spec, CPU_DEVICE)
        accelerator_network.load_state_dict(cpu_network.state_dict())
        accelerator_network = accelerator_network.to(accelerator_case.name)
        accelerator_network.eval()

        state_tensor = spec.initial_state_fn().to_tensor().unsqueeze(0)
        with torch.no_grad():
            cpu_policy, cpu_value = cpu_network(state_tensor)
            accelerator_policy, accelerator_value = accelerator_network(state_tensor.to(accelerator_case.name))

        assert torch.allclose(
            cpu_policy, accelerator_policy.cpu(), atol=CROSS_DEVICE_ATOL, rtol=CROSS_DEVICE_RTOL
        ), f"policy head diverged between cpu and {accelerator_case.name!r} beyond {CROSS_DEVICE_ATOL}"
        assert torch.allclose(
            cpu_value, accelerator_value.cpu(), atol=CROSS_DEVICE_ATOL, rtol=CROSS_DEVICE_RTOL
        ), f"value head diverged between cpu and {accelerator_case.name!r} beyond {CROSS_DEVICE_ATOL}"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
