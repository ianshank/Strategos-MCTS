"""
Gating for the live-server Gradio UI tests.

These tests launch the real app and drive it through ``gradio_client``, so the
meta-controller genuinely instantiates — which means the configured BERT model
must be loadable. In an offline environment (``HF_HUB_OFFLINE=1``, the setting CI
uses to keep runs hermetic) an uncached model cannot be fetched, and the failure
surfaces as an opaque ``gradio_client.exceptions.AppError`` raised inside the
served app rather than as a missing dependency.

A pre-flight check turns that into an explicit skip naming the real cause. The
skip is deliberately narrow: it fires only when the model genuinely cannot be
resolved, so a real regression still fails rather than quietly disappearing.
"""

from __future__ import annotations

import os

import pytest


def _bert_model_is_loadable() -> tuple[bool, str]:
    """Return whether the configured BERT model can be resolved right now."""
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        return False, f"transformers is not installed ({exc})"

    from src.config.settings import Settings, get_settings

    try:
        model_name = get_settings().BERT_DEFAULT_MODEL_NAME
    except Exception:
        # Settings may be unconfigured (no provider key). Read the declared default
        # off the field rather than repeating the literal here, so this pre-flight
        # cannot drift from what the app would actually try to load.
        model_name = Settings.model_fields["BERT_DEFAULT_MODEL_NAME"].default

    try:
        AutoConfig.from_pretrained(model_name)
    except Exception as exc:
        offline = os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
        hint = " (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE is set)" if offline else ""
        return False, f"BERT model {model_name!r} is not loadable{hint}: {type(exc).__name__}"

    return True, ""


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip live-server UI tests when their model prerequisite cannot be met."""
    loadable, reason = _bert_model_is_loadable()
    if loadable:
        return

    skip = pytest.mark.skip(reason=f"live Gradio UI test requires a loadable BERT model — {reason}")
    for item in items:
        if item.path is not None and item.path.name == "test_gradio_app.py":
            item.add_marker(skip)
