"""DistillationSettings knobs stay in sync with SCHEMA_VERSION and SEED."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from pydantic import ValidationError
import pytest

from scripts.local_distillation.schema import SCHEMA_VERSION
from scripts.local_distillation.settings import DistillationSettings
from src.utils.seeding import new_rng

pytestmark = [pytest.mark.unit]


def test_fractions_must_leave_a_test_remainder() -> None:
    with pytest.raises(ValidationError, match="remainder"):
        DistillationSettings(train_frac=0.9, val_frac=0.2)


def test_buffer_capacity_and_schema_version_match_schema_constant() -> None:
    settings = DistillationSettings()
    assert settings.schema_version == SCHEMA_VERSION
    assert settings.buffer_capacity == 10_000
    assert settings.wall_clock_repeat_cap == 1_000_000
    assert settings.device == "cpu"
    assert not hasattr(settings, "recurrent_hidden")


def test_new_rng_honors_settings_seed() -> None:
    mock_settings = MagicMock()
    mock_settings.SEED = 99
    with patch("src.config.settings.get_settings", return_value=mock_settings):
        from_none = new_rng(None)
        from_explicit = new_rng(99)
    assert np.array_equal(from_none.random(4), from_explicit.random(4))
