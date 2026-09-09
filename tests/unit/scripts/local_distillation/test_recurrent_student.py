"""local_distillation_contract AC-8: default-off recurrent student duck-types PolicyValueNet."""

from __future__ import annotations

import inspect

import pytest
import torch

from scripts.local_distillation.settings import DistillationSettings
from scripts.local_distillation.student import RecurrentPolicyValue, build_student
from src.models.policy_value_net import PolicyValueNetwork

pytestmark = [pytest.mark.unit]


def test_build_student_defaults_to_policy_value_network() -> None:
    student = build_student(DistillationSettings(recurrent_enabled=False))
    assert isinstance(student, PolicyValueNetwork)
    assert not isinstance(student, RecurrentPolicyValue)


def test_recurrent_enabled_returns_shared_weight_student() -> None:
    settings = DistillationSettings(recurrent_enabled=True, recurrences=3)
    student = build_student(settings)
    assert isinstance(student, RecurrentPolicyValue)
    student.eval()
    x = torch.zeros(2, settings.input_channels, settings.board_rows, settings.board_cols)
    log_probs, value = student(x)
    assert log_probs.shape == (2, settings.action_size)
    assert value.shape == (2, 1)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(value).all()
    probs = torch.exp(log_probs)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    assert float(value.min().detach()) >= -1.0 - 1e-5
    assert float(value.max().detach()) <= 1.0 + 1e-5


def test_feed_forward_outputs_are_finite() -> None:
    settings = DistillationSettings(recurrent_enabled=False, num_res_blocks=1, num_channels=8)
    student = build_student(settings)
    student.eval()
    x = torch.zeros(1, settings.input_channels, settings.board_rows, settings.board_cols)
    log_probs, value = student(x)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(value).all()


def test_recurrent_student_does_not_wrap_project_agents() -> None:
    import scripts.local_distillation.student as student_mod

    source = inspect.getsource(student_mod)
    assert "from src.agents.hrm_agent" not in source
    assert "from src.agents.trm_agent" not in source
