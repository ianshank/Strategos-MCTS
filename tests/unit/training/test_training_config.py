"""Unit tests for training profiles configuration."""

from __future__ import annotations

import pytest

from src.training.training_config import (
    TrainingProfile,
    get_training_profile,
    profile_to_configs,
)


@pytest.mark.unit
def test_get_training_profile_by_string() -> None:
    smoke = get_training_profile("smoke")
    assert smoke.iterations == 2
    assert smoke.device == "cpu"

    dev = get_training_profile("dev")
    assert dev.iterations == 20
    assert dev.use_amp is True

    full = get_training_profile("full")
    assert full.iterations == 200
    assert full.compile_model is True


@pytest.mark.unit
def test_get_training_profile_enum() -> None:
    spec = get_training_profile(TrainingProfile.SMOKE)
    assert spec.iterations == 2


@pytest.mark.unit
def test_get_training_profile_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Unknown profile"):
        get_training_profile("super_fast")


@pytest.mark.unit
def test_profile_to_configs() -> None:
    spec = get_training_profile(TrainingProfile.DEV)
    sp_cfg, mcts_cfg = profile_to_configs(spec)

    assert sp_cfg.num_games_per_iteration == spec.games_per_iteration
    assert sp_cfg.use_amp is True
    assert mcts_cfg.num_simulations == spec.num_simulations
