"""
Tests for :mod:`src.ui.status`.

The UI header previously read "This demo uses **REAL trained models**" in a
repository where all six checkpoints are Git-LFS pointer stubs — a claim no
command reproduces (CHARTER NG-3). These tests pin the replacement: the banner
asserts trained-model operation only when the weights are genuinely readable.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config.constants import GIT_LFS_POINTER_MAGIC, GIT_LFS_REMEDIATION
from src.models.checkpoints import CheckpointReport, CheckpointStatus
from src.ui.status import (
    DEFAULT_CHECKPOINTS,
    checkpoint_status_banner,
    inspect_configured_checkpoints,
    resolve_checkpoint_paths,
)

pytestmark = pytest.mark.unit


def _ok(name: str) -> CheckpointReport:
    return CheckpointReport(Path(name), CheckpointStatus.OK, "zip-format checkpoint", 1024)


def _stub(name: str) -> CheckpointReport:
    return CheckpointReport(Path(name), CheckpointStatus.LFS_POINTER, "Git-LFS pointer stub", 130)


def _missing(name: str) -> CheckpointReport:
    return CheckpointReport(Path(name), CheckpointStatus.MISSING, "path does not exist", 0)


class TestPathResolution:
    def test_defaults_are_relative_to_the_repo_root(self, tmp_path: Path) -> None:
        paths = resolve_checkpoint_paths(None, tmp_path)

        assert set(paths) == set(DEFAULT_CHECKPOINTS)
        for path in paths.values():
            assert path.is_relative_to(tmp_path)

    def test_settings_override_wins(self, tmp_path: Path) -> None:
        settings = SimpleNamespace(RNN_MODEL_PATH="/custom/rnn.pt", BERT_MODEL_PATH=None)

        paths = resolve_checkpoint_paths(settings, tmp_path)

        assert paths["RNN meta-controller"] == Path("/custom/rnn.pt")
        # Unset override still falls back to the packaged default.
        assert paths["BERT LoRA adapter"].is_relative_to(tmp_path)

    def test_missing_attributes_do_not_raise(self, tmp_path: Path) -> None:
        assert resolve_checkpoint_paths(SimpleNamespace(), tmp_path)


class TestBanner:
    def test_all_ok_claims_trained_weights(self) -> None:
        banner = checkpoint_status_banner({"RNN": _ok("a.pt"), "BERT": _ok("b.pt")})

        assert "trained weights" in banner
        assert "Reduced mode" not in banner

    def test_any_stub_downgrades_the_claim(self) -> None:
        """One bad checkpoint must not be averaged away into a confident banner."""
        banner = checkpoint_status_banner({"RNN": _ok("a.pt"), "BERT": _stub("b.pt")})

        assert "Reduced mode" in banner
        assert "Running on trained weights" not in banner

    def test_lfs_stub_surfaces_the_remediation(self) -> None:
        banner = checkpoint_status_banner({"RNN": _stub("a.pt")})

        assert GIT_LFS_REMEDIATION in banner

    def test_non_lfs_problem_omits_the_lfs_hint(self) -> None:
        """A missing file is not fixed by `git lfs pull`; don't suggest it."""
        banner = checkpoint_status_banner({"RNN": _missing("a.pt")})

        assert GIT_LFS_REMEDIATION not in banner

    def test_each_failing_checkpoint_is_named(self) -> None:
        banner = checkpoint_status_banner({"RNN meta-controller": _stub("a.pt"), "BERT LoRA adapter": _stub("b.pt")})

        assert "RNN meta-controller" in banner
        assert "BERT LoRA adapter" in banner

    def test_empty_input_does_not_claim_success(self) -> None:
        assert "unknown" in checkpoint_status_banner({}).lower()


class TestEndToEndInspection:
    def test_repository_with_stubs_reports_reduced_mode(self, tmp_path: Path) -> None:
        rnn = tmp_path / "models"
        rnn.mkdir()
        (rnn / "rnn_meta_controller.pt").write_bytes(GIT_LFS_POINTER_MAGIC + b"\noid sha256:abc\nsize 42\n")

        banner = checkpoint_status_banner(inspect_configured_checkpoints(None, tmp_path))

        assert "Reduced mode" in banner
        assert GIT_LFS_REMEDIATION in banner

    def test_real_weights_report_success(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        (models / "bert_lora" / "final_model").mkdir(parents=True)
        for target in (
            models / "rnn_meta_controller.pt",
            models / "bert_lora" / "final_model" / "adapter_model.safetensors",
        ):
            with zipfile.ZipFile(target, "w") as archive:
                archive.writestr("data.pkl", b"payload")

        banner = checkpoint_status_banner(inspect_configured_checkpoints(None, tmp_path))

        assert "trained weights" in banner

    def test_inspection_never_raises_on_a_bare_tree(self, tmp_path: Path) -> None:
        assert inspect_configured_checkpoints(None, tmp_path)
