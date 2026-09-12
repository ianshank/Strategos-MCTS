"""LogRecord extras for meta-controllers must not use the reserved key ``name``."""

from __future__ import annotations

import logging

import pytest

from src.agents.meta_controller.utils import controller_initialized_extra

pytestmark = [pytest.mark.unit]


def test_controller_initialized_extra_avoids_logrecord_name() -> None:
    extra = controller_initialized_extra(seed=42, name="BERT")
    assert "name" not in extra
    assert extra["controller_name"] == "BERT"
    assert extra["seed"] == 42
    record = logging.LogRecord("logger", logging.INFO, __file__, 1, "msg", (), None)
    overlap = set(extra).intersection(record.__dict__)
    assert not overlap, f"extra keys collide with LogRecord: {overlap}"
