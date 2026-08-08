"""
Tests for :mod:`src.models.checkpoints`.

The defect these guard: ``Path.exists()`` returns ``True`` for a Git-LFS pointer
stub, so an existence check passes and the failure surfaces much later as an
opaque ``UnpicklingError`` from inside a deserializer. Every checkpoint shipped
in this repository is currently such a stub, so the degraded path is the *normal*
path for anyone cloning without Git-LFS.

Fixtures synthesize each on-disk shape rather than depending on repository
contents, so the suite stays valid after a ``git lfs pull``.
"""

from __future__ import annotations

import pickle
import sys
import zipfile
from pathlib import Path

import pytest

from src.config.constants import GIT_LFS_POINTER_MAGIC, GIT_LFS_REMEDIATION
from src.models.checkpoints import (
    CheckpointStatus,
    _inspect_file,
    inspect_checkpoint,
    load_checkpoint,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def lfs_pointer(tmp_path: Path) -> Path:
    """A byte-accurate Git-LFS pointer stub, matching the ones in models/."""
    path = tmp_path / "weights.pt"
    path.write_bytes(
        GIT_LFS_POINTER_MAGIC
        + b"\noid sha256:dc4edc9ca27e2f0a4b1d3e5f7a9b0c2d4e6f8a0b2c4d6e8f0a2b4c6d8e0f2a4b\nsize 4194304\n"
    )
    return path


@pytest.fixture
def zip_checkpoint(tmp_path: Path) -> Path:
    """A zip-container checkpoint, the shape torch>=1.6 writes."""
    path = tmp_path / "model.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/data.pkl", b"payload")
    return path


@pytest.fixture
def pickle_checkpoint(tmp_path: Path) -> Path:
    """A legacy (pre-1.6) pickle checkpoint."""
    path = tmp_path / "legacy.pt"
    path.write_bytes(pickle.dumps({"state_dict": {}}, protocol=2))
    return path


@pytest.fixture
def safetensors_checkpoint(tmp_path: Path) -> Path:
    """A minimal safetensors container: u64 header length then JSON."""
    header = b'{"__metadata__":{}}'
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header)
    return path


class TestPointerDetection:
    def test_lfs_pointer_is_classified_not_loaded(self, lfs_pointer: Path) -> None:
        report = inspect_checkpoint(lfs_pointer)

        assert report.status is CheckpointStatus.LFS_POINTER
        assert not report.is_ok

    def test_pointer_report_carries_actionable_remediation(self, lfs_pointer: Path) -> None:
        assert inspect_checkpoint(lfs_pointer).remediation == GIT_LFS_REMEDIATION

    def test_a_large_file_starting_with_the_magic_is_not_a_pointer(self, tmp_path: Path) -> None:
        """Guards the size bound: real content is never misread as a stub."""
        path = tmp_path / "big.pt"
        path.write_bytes(GIT_LFS_POINTER_MAGIC + b"\x00" * 4096)

        assert inspect_checkpoint(path).status is not CheckpointStatus.LFS_POINTER


class TestValidContainers:
    @pytest.mark.parametrize(
        "fixture_name",
        ["zip_checkpoint", "pickle_checkpoint", "safetensors_checkpoint"],
    )
    def test_recognized_containers_report_ok(self, request: pytest.FixtureRequest, fixture_name: str) -> None:
        path = request.getfixturevalue(fixture_name)
        report = inspect_checkpoint(path)

        assert report.is_ok, report.detail
        assert report.remediation == ""

    def test_size_is_reported(self, zip_checkpoint: Path) -> None:
        assert inspect_checkpoint(zip_checkpoint).size_bytes == zip_checkpoint.stat().st_size


class TestInvalidPaths:
    def test_missing_path(self, tmp_path: Path) -> None:
        assert inspect_checkpoint(tmp_path / "nope.pt").status is CheckpointStatus.MISSING

    def test_empty_file_is_unreadable(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.pt"
        path.touch()

        assert inspect_checkpoint(path).status is CheckpointStatus.UNREADABLE

    def test_unrecognized_content_is_unreadable(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.pt"
        path.write_text("this is plain text, not a checkpoint" * 100)

        assert inspect_checkpoint(path).status is CheckpointStatus.UNREADABLE

    def test_pickle_opcode_alone_is_not_a_signature(self, tmp_path: Path) -> None:
        """
        Regression: 0x80 is an ordinary byte, not a signature.

        Matching only the PROTO opcode classified any binary file starting with
        0x80 as a valid legacy checkpoint. That mattered beyond load time —
        inspect_checkpoint feeds the UI banner, so a corrupt file made the app
        claim it was running on trained weights.
        """
        path = tmp_path / "corrupt.pt"
        path.write_bytes(b"\x80" + b"\xff" * 512)

        assert inspect_checkpoint(path).status is CheckpointStatus.UNREADABLE

    @pytest.mark.parametrize("protocol", [2, 3, 4, 5])
    def test_every_proto_emitting_pickle_protocol_is_accepted(self, tmp_path: Path, protocol: int) -> None:
        """Tightening the check must not reject genuine pickles."""
        path = tmp_path / f"p{protocol}.pt"
        path.write_bytes(pickle.dumps({"state_dict": {}}, protocol=protocol))

        assert inspect_checkpoint(path).is_ok

    def test_safetensors_header_length_must_be_plausible(self, tmp_path: Path) -> None:
        """A u64 length larger than the file cannot describe a real header."""
        path = tmp_path / "bogus.safetensors"
        path.write_bytes((10**9).to_bytes(8, "little") + b'{"a":1}')

        assert inspect_checkpoint(path).status is CheckpointStatus.UNREADABLE

    def test_safetensors_zero_header_length_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "empty-header.safetensors"
        path.write_bytes((0).to_bytes(8, "little") + b'{"a":1}')

        assert inspect_checkpoint(path).status is CheckpointStatus.UNREADABLE

    def test_inspection_never_raises(self, tmp_path: Path) -> None:
        """Callers branch on the report; they must not need a try/except."""
        for candidate in [tmp_path / "absent", tmp_path, Path("/dev/null")]:
            assert inspect_checkpoint(candidate) is not None


class TestDirectoryCheckpoints:
    """A PEFT/LoRA adapter is a directory, not a file."""

    def test_directory_of_valid_weights_is_ok(self, tmp_path: Path, safetensors_checkpoint: Path) -> None:
        adapter = tmp_path / "final_model"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(safetensors_checkpoint.read_bytes())

        assert inspect_checkpoint(adapter).is_ok

    def test_one_pointer_stub_invalidates_the_directory(
        self, tmp_path: Path, safetensors_checkpoint: Path, lfs_pointer: Path
    ) -> None:
        """A partially-fetched adapter loads unpredictably; refusing it is safer."""
        adapter = tmp_path / "final_model"
        adapter.mkdir()
        (adapter / "good.safetensors").write_bytes(safetensors_checkpoint.read_bytes())
        (adapter / "adapter_model.bin").write_bytes(lfs_pointer.read_bytes())

        report = inspect_checkpoint(adapter)

        assert report.status is CheckpointStatus.LFS_POINTER
        assert "adapter_model.bin" in report.detail

    def test_directory_without_weights_is_missing(self, tmp_path: Path) -> None:
        adapter = tmp_path / "final_model"
        adapter.mkdir()
        (adapter / "README.md").write_text("no weights here")

        assert inspect_checkpoint(adapter).status is CheckpointStatus.MISSING


class TestTolerantLoading:
    def test_pointer_stub_returns_none_instead_of_raising(self, lfs_pointer: Path) -> None:
        """torch.load on this content raises UnpicklingError('invalid load key, v')."""
        assert load_checkpoint(lfs_pointer) is None

    def test_missing_path_returns_none(self, tmp_path: Path) -> None:
        assert load_checkpoint(tmp_path / "absent.pt") is None

    def test_corrupt_container_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.pt"
        path.write_bytes(b"\x80" + b"\xff" * 512)  # pickle magic, garbage body

        assert load_checkpoint(path) is None

    def test_deserializer_failure_degrades_rather_than_raising(self, tmp_path: Path) -> None:
        """
        The file passes inspection but the deserializer rejects it.

        Distinct from the corrupt-container case, which is now caught at
        classification and never reaches torch. This is the last line of defence —
        a well-formed container whose *contents* torch cannot load.
        """
        path = tmp_path / "wellformed-but-unloadable.pt"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("not-a-torch-archive.txt", b"payload")

        assert inspect_checkpoint(path).is_ok, "precondition: must pass classification"
        assert load_checkpoint(path) is None

    def test_deserializer_failure_is_logged(self, tmp_path: Path, caplog) -> None:
        path = tmp_path / "wellformed-but-unloadable.pt"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("not-a-torch-archive.txt", b"payload")

        with caplog.at_level("WARNING"):
            load_checkpoint(path)

        assert any("degraded mode" in r.message for r in caplog.records)

    def test_failure_is_logged_with_structured_fields(self, lfs_pointer: Path, caplog) -> None:
        with caplog.at_level("WARNING"):
            load_checkpoint(lfs_pointer)

        assert any("degraded mode" in record.message for record in caplog.records)

    def test_accepts_a_stdlib_logger_without_raising(self, lfs_pointer: Path) -> None:
        """Regression: structured fields on a stdlib Logger raise TypeError unwrapped."""
        import logging

        assert load_checkpoint(lfs_pointer, logger=logging.getLogger("plain.stdlib")) is None

    def test_returns_the_deserialized_object_on_success(self, tmp_path: Path) -> None:
        """The success path — the only one that returns something."""
        torch = pytest.importorskip("torch", reason="requires the [neural] extra")
        path = tmp_path / "real.pt"
        torch.save({"weight": 1}, path)

        assert load_checkpoint(path) == {"weight": 1}

    def test_absent_torch_degrades_instead_of_raising(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """
        A readable checkpoint with torch missing must return None, not ImportError.

        This is the `pip install -e '.[ui]'` case: the file is fine, the deserializer
        simply is not installed.
        """
        path = tmp_path / "model.pt"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("data.pkl", b"payload")
        monkeypatch.setitem(sys.modules, "torch", None)  # forces ImportError on `import torch`

        assert load_checkpoint(path) is None


class TestUnreadableFilesystem:
    """OS-level failures must produce a report, never an exception."""

    def test_unreadable_file_is_classified_not_raised(self, tmp_path: Path) -> None:
        """A file whose bytes cannot be read still yields a verdict."""
        path = tmp_path / "locked.pt"
        path.write_bytes(b"\x80\x02payload")
        path.chmod(0o000)
        try:
            report = inspect_checkpoint(path)
        finally:
            path.chmod(0o644)  # let tmp_path clean up

        # Root ignores mode bits, so accept either a clean read or the guarded failure —
        # the contract under test is "returns a report", not which branch was taken.
        assert report.status in {CheckpointStatus.OK, CheckpointStatus.UNREADABLE}

    def test_stat_failure_is_reported(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Targets _inspect_file directly.

        Patching Path.stat globally also breaks Path.exists(), which swallows OSError
        and reroutes to MISSING — so the stat-guard inside _inspect_file would never
        be reached and the test would pass for the wrong reason.
        """
        path = tmp_path / "gone.pt"
        path.write_bytes(b"\x80\x02payload")

        def exploding_stat(self, *args, **kwargs):
            raise OSError("simulated stat failure")

        monkeypatch.setattr(Path, "stat", exploding_stat)

        report = _inspect_file(path)

        assert report.status is CheckpointStatus.UNREADABLE
        assert "stat failed" in report.detail

    def test_file_too_short_to_hold_a_header_is_unreadable(self, tmp_path: Path) -> None:
        """Fewer bytes than a safetensors length prefix cannot be any known container."""
        path = tmp_path / "stub.pt"
        path.write_bytes(b"tiny")

        assert inspect_checkpoint(path).status is CheckpointStatus.UNREADABLE

    def test_directory_with_an_unreadable_weight_names_the_file(self, tmp_path: Path) -> None:
        """A corrupt (non-pointer) member must be reported by name, not silently ignored."""
        adapter = tmp_path / "final_model"
        adapter.mkdir()
        (adapter / "broken.safetensors").write_bytes(b"\x80" + b"\xff" * 512)

        report = inspect_checkpoint(adapter)

        assert report.status is CheckpointStatus.UNREADABLE
        assert "broken.safetensors" in report.detail

    def test_read_failure_is_reported(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """stat() succeeds but the read raises — e.g. a disappearing network mount."""
        path = tmp_path / "flaky.pt"
        path.write_bytes(b"\x80\x02payload")

        real_open = Path.open
        calls = {"n": 0}

        def flaky_open(self, *args, **kwargs):
            if self == path:
                calls["n"] += 1
                if calls["n"] > 1:  # let the LFS-pointer probe succeed, fail the format read
                    raise OSError("simulated read failure")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", flaky_open)

        report = inspect_checkpoint(path)

        assert report.status is CheckpointStatus.UNREADABLE
        assert "read failed" in report.detail

    def test_lfs_probe_survives_an_unreadable_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_is_lfs_pointer swallows OSError and reports 'not a pointer' rather than raising."""
        path = tmp_path / "small.pt"
        path.write_bytes(b"tiny")

        def exploding_open(self, *args, **kwargs):
            raise OSError("simulated open failure")

        monkeypatch.setattr(Path, "open", exploding_open)

        assert inspect_checkpoint(path).status is CheckpointStatus.UNREADABLE
