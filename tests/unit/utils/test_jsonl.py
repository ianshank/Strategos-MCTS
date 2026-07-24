"""Unit tests for the shared append-only JSONL helpers (src/utils/jsonl.py)."""

from __future__ import annotations

import json

import pytest

from src.utils.jsonl import append_jsonl, iter_jsonl


def test_append_creates_parent_and_roundtrips(tmp_path):
    path = tmp_path / "nested" / "log.jsonl"
    append_jsonl(path, {"a": 1})
    append_jsonl(path, {"a": 2})
    assert [rec["a"] for rec in iter_jsonl(path)] == [1, 2]


def test_iter_missing_file_is_empty(tmp_path):
    assert list(iter_jsonl(tmp_path / "absent.jsonl")) == []


def test_non_serializable_value_uses_default(tmp_path):
    path = tmp_path / "log.jsonl"
    append_jsonl(path, {"obj": object()})  # default=str keeps it writable
    (record,) = list(iter_jsonl(path))
    assert isinstance(record["obj"], str)


def test_tolerates_torn_trailing_line(tmp_path):
    path = tmp_path / "log.jsonl"
    append_jsonl(path, {"ok": True})
    with path.open("a") as handle:
        handle.write("{partial")
    records = list(iter_jsonl(path))
    assert records == [{"ok": True}]


def test_blank_lines_skipped(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert [rec["a"] for rec in iter_jsonl(path)] == [1, 2]


def test_non_tolerant_mode_raises_on_partial(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text("{bad", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        list(iter_jsonl(path, tolerate_partial=False))
