"""Unit tests for src.utils.seeding (hygiene_determinism AC-1 / AC-2).

Covers hygiene_determinism AC-1
Covers hygiene_determinism AC-2
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config.constants import DEFAULT_SEED
from src.utils.seeding import (
    NUMPY_LEGACY_SEED_MAX,
    NUMPY_LEGACY_SEED_MIN,
    new_rng,
    resolve_seed,
    set_all_seeds,
    validate_numpy_seed,
)


class TestResolveSeed:
    """Covers hygiene_determinism AC-2 — Settings.SEED / DEFAULT_SEED reuse."""

    def test_explicit_seed_wins(self) -> None:
        assert resolve_seed(7) == 7

    def test_falls_back_to_default_when_settings_unset(self) -> None:
        mock_settings = MagicMock()
        mock_settings.SEED = None
        with patch("src.config.settings.get_settings", return_value=mock_settings):
            assert resolve_seed(None) == DEFAULT_SEED

    def test_uses_settings_seed_when_configured(self) -> None:
        mock_settings = MagicMock()
        mock_settings.SEED = 99
        with patch("src.config.settings.get_settings", return_value=mock_settings):
            assert resolve_seed(None) == 99

    def test_settings_import_failure_falls_back_to_default(self) -> None:
        with patch("src.config.settings.get_settings", side_effect=RuntimeError("boom")):
            assert resolve_seed(None) == DEFAULT_SEED


class TestNewRng:
    def test_returns_generator_and_is_reproducible(self) -> None:
        a = new_rng(123)
        b = new_rng(123)
        assert isinstance(a, np.random.Generator)
        assert np.array_equal(a.random(5), b.random(5))

    def test_different_seeds_diverge(self) -> None:
        assert not np.array_equal(new_rng(1).random(5), new_rng(2).random(5))

    def test_none_seed_uses_resolve_seed(self) -> None:
        with patch("src.utils.seeding.resolve_seed", return_value=55) as mocked:
            rng = new_rng(None)
            mocked.assert_called_once_with(None)
            assert isinstance(rng, np.random.Generator)


class TestSetAllSeeds:
    """Covers hygiene_determinism AC-1 — branch coverage for set_all_seeds."""

    def test_seeds_python_and_numpy_and_returns_effective(self) -> None:
        import random as py_random

        effective = set_all_seeds(10, rank=0)
        assert effective == 10
        # Probe the legacy global RNG that set_all_seeds is responsible for seeding.
        first = float(np.random.random())  # noqa: NPY002 — asserting legacy seed took effect
        py_first = py_random.random()
        set_all_seeds(10, rank=0)
        second = float(np.random.random())  # noqa: NPY002 — asserting legacy seed took effect
        py_second = py_random.random()
        assert first == second
        assert py_first == py_second

    def test_rank_offsets_effective_seed(self) -> None:
        assert set_all_seeds(10, rank=3) == 13

    def test_skips_torch_when_import_fails(self) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002, ANN001
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import), patch("src.utils.seeding.logger") as mock_logger:
            effective = set_all_seeds(42, rank=1)
        assert effective == 43
        mock_logger.info.assert_called()
        assert "torch unavailable" in mock_logger.info.call_args[0][0]

    def test_with_torch_no_cuda(self) -> None:
        torch = pytest.importorskip("torch")
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch("src.utils.seeding.logger") as mock_logger,
        ):
            effective = set_all_seeds(5, rank=0, deterministic_torch=False)
        assert effective == 5
        mock_logger.info.assert_called()
        assert "deterministic_torch" not in mock_logger.info.call_args[0][0]

    def test_with_torch_cuda_branch(self) -> None:
        pytest.importorskip("torch")
        import torch

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "manual_seed_all") as mock_cuda_seed,
            patch("src.utils.seeding.logger"),
        ):
            effective = set_all_seeds(8, rank=2)
        assert effective == 10
        # torch.manual_seed may itself call manual_seed_all when CUDA is "available";
        # assert our explicit cuda branch fired with the effective seed.
        mock_cuda_seed.assert_any_call(10)
        assert mock_cuda_seed.call_count >= 1

    def test_deterministic_torch_sets_cudnn_flags(self) -> None:
        torch = pytest.importorskip("torch")
        if not hasattr(torch.backends, "cudnn"):
            pytest.skip("cudnn backend not present")
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch("src.utils.seeding.logger") as mock_logger,
        ):
            prev_det = torch.backends.cudnn.deterministic
            prev_bench = torch.backends.cudnn.benchmark
            try:
                effective = set_all_seeds(1, deterministic_torch=True)
                assert effective == 1
                assert torch.backends.cudnn.deterministic is True
                assert torch.backends.cudnn.benchmark is False
                assert "deterministic_torch=True" in mock_logger.info.call_args[0][0]
            finally:
                torch.backends.cudnn.deterministic = prev_det
                torch.backends.cudnn.benchmark = prev_bench

    def test_deterministic_torch_without_cudnn_attr(self) -> None:
        torch = pytest.importorskip("torch")
        mock_backends = MagicMock(spec=[])  # no cudnn attribute
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch, "backends", mock_backends),
            patch("src.utils.seeding.logger") as mock_logger,
        ):
            effective = set_all_seeds(2, deterministic_torch=True)
        assert effective == 2
        assert "deterministic_torch=True" in mock_logger.info.call_args[0][0]


class TestValidateNumpySeed:
    """NumPy legacy-safe seed bounds (0 .. 2**32 - 1)."""

    def test_accepts_bounds(self) -> None:
        assert validate_numpy_seed(NUMPY_LEGACY_SEED_MIN) == NUMPY_LEGACY_SEED_MIN
        assert validate_numpy_seed(NUMPY_LEGACY_SEED_MAX) == NUMPY_LEGACY_SEED_MAX

    def test_rejects_below_min(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            validate_numpy_seed(NUMPY_LEGACY_SEED_MIN - 1)

    def test_rejects_above_max(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            validate_numpy_seed(NUMPY_LEGACY_SEED_MAX + 1)

    def test_label_appears_in_message(self) -> None:
        with pytest.raises(ValueError, match="effective seed"):
            validate_numpy_seed(-1, label="effective seed")


class TestResolveSeedValidation:
    def test_explicit_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="seed must be in"):
            resolve_seed(NUMPY_LEGACY_SEED_MAX + 1)

    def test_settings_out_of_range_raises(self) -> None:
        mock_settings = MagicMock()
        mock_settings.SEED = -5
        with patch("src.config.settings.get_settings", return_value=mock_settings):
            with pytest.raises(ValueError, match="Settings.SEED"):
                resolve_seed(None)


class TestSetAllSeedsValidation:
    def test_effective_seed_out_of_range_raises(self) -> None:
        # base at max + rank 1 overflows the legacy range
        with pytest.raises(ValueError, match="effective seed"):
            set_all_seeds(NUMPY_LEGACY_SEED_MAX, rank=1)

    def test_negative_base_raises(self) -> None:
        with pytest.raises(ValueError, match="effective seed"):
            set_all_seeds(-1, rank=0)
