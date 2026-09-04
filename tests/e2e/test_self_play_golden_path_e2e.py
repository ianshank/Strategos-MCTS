"""The self-play golden path, driven the way an operator drives it.

``self-play-convergence`` -> checkpoint + ``.meta.json`` sidecar -> ``--resume`` ->
``policy-lift`` -> gate artifact, as two **installed console scripts** in real
subprocesses, on every device this host provides.

What this adds over ``tests/integration/benchmark/test_self_play_convergence_e2e.py``
(which drives the same pipeline in-process, on CPU, on the synthetic ``reasoning``
domain) is exactly four things, and nothing else is duplicated here:

1. **The process boundary.** The integration test calls ``run(args)``; nothing in the
   tree invokes the ``[project.scripts]`` entry points by their installed names, so a
   broken ``module:function`` target is invisible until a user hits it.
2. **The device matrix.** The driver's ``--device`` is exercised on every available
   device, so the CUDA path is tested where CUDA exists and *reported as skipped* where
   it does not.
3. **Checkpoint portability.** The gate always runs with ``--device cpu`` against a
   checkpoint the driver may have written on an accelerator — the case that breaks when
   a ``map_location`` is coupled to the writing device.
4. **Fresh-process determinism.** ``specs/hygiene_determinism.SPEC.md`` AC-3 states a
   *fresh-process* property; an in-process double run cannot observe it.

Domain: ``connect_four``. It is adversarial (win-rate metric, so the gate takes its
Wilson-interval path) and it is the golden-path domain of
``docs/plans/EVIDENCE_FIRST_PROGRAM.md`` E3. The ``reasoning`` domain is deliberately
avoided: ``src/benchmark/policy_lift.py`` documents its reward as synthetic and gameable.

No claim is made here about *lift*. The gate's verdict at these tiny sample sizes is
meaningless as a measurement and is asserted only as "a verdict, never a loading error".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.benchmark import policy_lift
from src.training import self_play_convergence
from tests.utils.device_matrix import CPU_DEVICE, DeviceCase

pytestmark = [pytest.mark.e2e, pytest.mark.training]

#: Console scripts under test, as declared in ``pyproject.toml [project.scripts]``.
DRIVER_SCRIPT = "self-play-convergence"
GATE_SCRIPT = "policy-lift"

#: The golden-path domain (see module docstring).
DOMAIN = "connect_four"

#: Work sizes: the smallest values that still exercise every stage. Matching the sizes
#: the existing integration smoke uses, so this stays a plumbing test and not a
#: budget-consuming training run.
SIMULATIONS = "2"
GAMES_PER_ITERATION = "1"
GATE_GAMES = "2"
GATE_MAX_MOVES = "10"

#: Gate exit codes that mean "the gate reached a verdict". ``EXIT_ERROR`` (2) means it
#: could not load or rebuild the network — the failure this test exists to catch.
VERDICT_EXIT_CODES = (policy_lift.EXIT_GATE_MET, policy_lift.EXIT_GATE_NOT_MET)


def _driver_argv(checkpoint_dir: Path, device: str, seed: int, *, resume: bool = False) -> list[str]:
    argv = [
        DRIVER_SCRIPT,
        "--domain",
        DOMAIN,
        "--iterations",
        "1",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--seed",
        str(seed),
        "--device",
        device,
        "--num-simulations",
        SIMULATIONS,
        "--games-per-iteration",
        GAMES_PER_ITERATION,
    ]
    if resume:
        argv.append("--resume")
    return argv


def _checkpoint(checkpoint_dir: Path, iteration: int) -> Path:
    return checkpoint_dir / f"ckpt_iter_{iteration}.pt"


def test_golden_path_driver_resume_and_gate(tmp_path, device_case: DeviceCase, e2e_seed, run_script) -> None:
    """Driver -> checkpoint -> resume -> gate, end to end through the installed scripts."""
    checkpoint_dir = tmp_path / "checkpoints"

    first = run_script(_driver_argv(checkpoint_dir, device_case.name, e2e_seed))
    assert first.returncode == self_play_convergence.EXIT_OK, first.describe()

    checkpoint = _checkpoint(checkpoint_dir, 1)
    sidecar = checkpoint.with_name(checkpoint.name + ".meta.json")
    assert checkpoint.is_file(), f"driver reported success but wrote no checkpoint\n{first.describe()}"
    assert sidecar.is_file(), f"driver wrote no architecture sidecar\n{first.describe()}"

    # The sidecar is what lets the gate rebuild the network without guessing, so its
    # contract is asserted rather than assumed.
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["domain"] == DOMAIN
    assert metadata["iteration"] == 1
    assert metadata["seed"] == e2e_seed
    assert metadata["schema_version"] == 1
    assert metadata["network"]["type"], "sidecar records no network type"

    # --resume must continue the numbering, not restart it. A driver that silently
    # rewrote ckpt_iter_1 would lose an operator's training history.
    resumed = run_script(_driver_argv(checkpoint_dir, device_case.name, e2e_seed, resume=True))
    assert resumed.returncode == self_play_convergence.EXIT_OK, resumed.describe()
    assert _checkpoint(checkpoint_dir, 2).is_file(), (
        f"--resume did not continue iteration numbering; directory holds "
        f"{sorted(p.name for p in checkpoint_dir.iterdir())}\n{resumed.describe()}"
    )

    # The gate runs on CPU regardless of where the checkpoint was written: on an
    # accelerator host this is the portability assertion, and it is what an operator
    # running the gate on a different machine actually does.
    artifact_path = tmp_path / "lift.json"
    gate = run_script(
        [
            GATE_SCRIPT,
            "--domain",
            DOMAIN,
            "--checkpoint",
            str(_checkpoint(checkpoint_dir, 2)),
            "--num-games",
            GATE_GAMES,
            "--num-simulations",
            SIMULATIONS,
            "--max-moves",
            GATE_MAX_MOVES,
            "--seed",
            str(e2e_seed),
            "--device",
            CPU_DEVICE,
            "--output",
            str(artifact_path),
        ]
    )
    assert gate.returncode in VERDICT_EXIT_CODES, (
        f"the gate must reach a verdict (0 met / 1 not met); exit {policy_lift.EXIT_ERROR} means it "
        f"could not load or rebuild the network written on {device_case.name!r}\n{gate.describe()}"
    )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["domain"] == DOMAIN
    assert artifact["metric"] == "win_rate", "connect_four is adversarial; the gate must take its win-rate path"
    assert artifact["lift_ci_lower_pct"] <= artifact["lift_pct"] <= artifact["lift_ci_upper_pct"]

    # Provenance: the artifact must say what produced it, or it is not evidence
    # (docs/plans/EVIDENCE_FIRST_PROGRAM.md R2).
    run_info = artifact["run"]
    assert run_info["device"] == CPU_DEVICE
    assert run_info["seed"] == e2e_seed
    assert run_info["num_simulations"] == int(SIMULATIONS)
    assert Path(run_info["checkpoint"]).name == _checkpoint(checkpoint_dir, 2).name
    assert run_info["network"]["type"] == metadata["network"]["type"], "the gate rebuilt a different architecture"


def test_same_seed_in_fresh_processes_is_reproducible(tmp_path, e2e_seed, run_script) -> None:
    """Two fresh processes, one seed, identical weights.

    Bitwise equality is asserted on CPU only, and deliberately: the driver sets neither
    ``torch.backends.cudnn.deterministic`` nor ``CUBLAS_WORKSPACE_CONFIG``, so demanding
    bitwise CUDA reproducibility would be asserting a property the code does not claim.
    Covers ``hygiene_determinism`` AC-3's fresh-process half on the device where the
    guarantee holds.
    """
    digests = []
    for run_index in (0, 1):
        checkpoint_dir = tmp_path / f"run{run_index}"
        result = run_script(_driver_argv(checkpoint_dir, CPU_DEVICE, e2e_seed))
        assert result.returncode == self_play_convergence.EXIT_OK, result.describe()
        digests.append(_checkpoint(checkpoint_dir, 1).read_bytes())

    assert digests[0] == digests[1], (
        "two fresh processes with the same seed produced different weights; the driver's "
        "run is not reproducible, so no result it produces can be evidence"
    )
