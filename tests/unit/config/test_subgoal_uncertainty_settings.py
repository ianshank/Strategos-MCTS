"""Tests for the risk-averse subgoal penalty settings (strategos_risk_averse_subgoal_scorer AC-3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.constants import (
    DEFAULT_SUBGOAL_UNCERTAINTY_LAMBDA,
    MAX_SUBGOAL_UNCERTAINTY_LAMBDA,
    MIN_SUBGOAL_UNCERTAINTY_LAMBDA,
)
from src.config.graph_settings import GraphHardeningSettings


def test_penalty_flag_defaults_off():  # AC-2 default
    assert GraphHardeningSettings().ENABLE_UNCERTAINTY_SUBGOAL_PENALTY is False


def test_lambda_default_sourced_from_constants():  # AC-3
    assert GraphHardeningSettings().SUBGOAL_UNCERTAINTY_LAMBDA == DEFAULT_SUBGOAL_UNCERTAINTY_LAMBDA


def test_lambda_accepts_in_range_value():  # AC-3
    settings = GraphHardeningSettings(SUBGOAL_UNCERTAINTY_LAMBDA=3.0)
    assert settings.SUBGOAL_UNCERTAINTY_LAMBDA == 3.0


def test_lambda_accepts_boundaries():  # AC-3
    assert GraphHardeningSettings(SUBGOAL_UNCERTAINTY_LAMBDA=MIN_SUBGOAL_UNCERTAINTY_LAMBDA)
    assert GraphHardeningSettings(SUBGOAL_UNCERTAINTY_LAMBDA=MAX_SUBGOAL_UNCERTAINTY_LAMBDA)


def test_negative_lambda_rejected():  # AC-3
    with pytest.raises(ValidationError):
        GraphHardeningSettings(SUBGOAL_UNCERTAINTY_LAMBDA=MIN_SUBGOAL_UNCERTAINTY_LAMBDA - 1.0)


def test_above_max_lambda_rejected():  # AC-3
    with pytest.raises(ValidationError):
        GraphHardeningSettings(SUBGOAL_UNCERTAINTY_LAMBDA=MAX_SUBGOAL_UNCERTAINTY_LAMBDA + 1.0)
