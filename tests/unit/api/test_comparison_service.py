"""Unit tests for ComparisonService (single-shot vs MCTS, mocked LLM/pipeline)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.api.comparison_service import (
    ComparisonDisabledError,
    ComparisonResult,
    ComparisonService,
    MctsArmResult,
    SingleShotResult,
)


def _settings(enable: bool = True) -> SimpleNamespace:
    return SimpleNamespace(ENABLE_DEMO_COMPARISON=enable)


def _fake_llm_calls(n: int):
    return [SimpleNamespace(strategy=f"s{i}") for i in range(n)]


def _fake_pipeline_result(best_score: float = 0.8, llm_calls: int = 4, tree_root=object()):
    mcts_result = SimpleNamespace(
        best_strategy="decomposition",
        best_response="MCTS best answer",
        best_score=best_score,
        all_strategies={"direct": 0.5, "decomposition": best_score},
        llm_calls=_fake_llm_calls(llm_calls),
    )
    return SimpleNamespace(
        mcts_result=mcts_result,
        total_time_ms=123.4,
        tree_root=tree_root,
    )


@pytest.fixture
def mock_llm():
    """A MockLLMClient-like object whose single-shot arm is deterministic."""
    llm = MagicMock()
    llm.provider = "mock"
    return llm


@pytest.fixture
def fake_pipeline():
    pipe = MagicMock()
    pipe.run.return_value = _fake_pipeline_result()
    return pipe


def _make_service(fake_pipeline, mock_llm, settings, monkeypatch, ss=("single answer", 0.6, 50.0)):
    # Patch SingleShotRunner so its .run returns deterministic (response, score, latency)
    runner = MagicMock()
    runner.run.return_value = ss
    monkeypatch.setattr("src.api.comparison_service.SingleShotRunner", lambda _llm: runner)
    monkeypatch.setattr("src.api.comparison_service.TreeVisualizer", MagicMock(render=lambda root: "TREE"))
    return (
        ComparisonService(pipeline=fake_pipeline, llm_client=mock_llm, settings=settings),
        runner,
    )


def test_enabled_reflects_settings(fake_pipeline, mock_llm):
    svc = ComparisonService(pipeline=fake_pipeline, llm_client=mock_llm, settings=_settings(True))
    assert svc.enabled is True
    svc_off = ComparisonService(pipeline=fake_pipeline, llm_client=mock_llm, settings=_settings(False))
    assert svc_off.enabled is False


def test_compare_disabled_raises(fake_pipeline, mock_llm):
    svc = ComparisonService(pipeline=fake_pipeline, llm_client=mock_llm, settings=_settings(False))
    with pytest.raises(ComparisonDisabledError):
        svc.compare("q")


def test_compare_returns_structured_result_with_positive_delta(fake_pipeline, mock_llm, monkeypatch):
    svc, runner = _make_service(fake_pipeline, mock_llm, _settings(True), monkeypatch, ss=("ss", 0.6, 50.0))
    result = svc.compare("how to design X?")

    assert isinstance(result, ComparisonResult)
    assert isinstance(result.single_shot, SingleShotResult)
    assert isinstance(result.mcts, MctsArmResult)

    assert result.single_shot.score == 0.6
    assert result.single_shot.response == "ss"
    assert result.single_shot.latency_ms == 50.0

    assert result.mcts.best_score == 0.8
    assert result.mcts.best_strategy == "decomposition"
    assert result.mcts.llm_calls == 4
    assert result.mcts.total_time_ms == 123.4
    assert result.mcts.all_strategies == {"direct": 0.5, "decomposition": 0.8}

    # delta = 0.8 - 0.6 = 0.2 ; improvement = 0.2 / 0.6 * 100 ~= 33.3
    assert result.delta == pytest.approx(0.2, abs=1e-6)
    assert result.improvement_pct == pytest.approx(33.3, abs=0.1)
    assert result.tree == "TREE"

    runner.run.assert_called_once_with("how to design X?")
    fake_pipeline.run.assert_called_once()


def test_compare_negative_delta(fake_pipeline, mock_llm, monkeypatch):
    fake_pipeline.run.return_value = _fake_pipeline_result(best_score=0.4)
    svc, _ = _make_service(fake_pipeline, mock_llm, _settings(True), monkeypatch, ss=("ss", 0.6, 50.0))
    result = svc.compare("q")
    assert result.delta == pytest.approx(-0.2, abs=1e-6)


def test_improvement_pct_zero_when_single_shot_zero(fake_pipeline, mock_llm, monkeypatch):
    svc, _ = _make_service(fake_pipeline, mock_llm, _settings(True), monkeypatch, ss=("ss", 0.0, 10.0))
    result = svc.compare("q")
    assert result.improvement_pct == 0.0


def test_compare_without_tree(fake_pipeline, mock_llm, monkeypatch):
    svc, _ = _make_service(fake_pipeline, mock_llm, _settings(True), monkeypatch)
    result = svc.compare("q", include_tree=False)
    assert result.tree is None


def test_compare_tree_none_when_no_root(mock_llm, monkeypatch):
    pipe = MagicMock()
    pipe.run.return_value = _fake_pipeline_result(tree_root=None)
    svc, _ = _make_service(pipe, mock_llm, _settings(True), monkeypatch)
    result = svc.compare("q", include_tree=True)
    assert result.tree is None


def test_to_dict_round_trip(fake_pipeline, mock_llm, monkeypatch):
    svc, _ = _make_service(fake_pipeline, mock_llm, _settings(True), monkeypatch)
    data = svc.compare("q").to_dict()
    assert data["single_shot"]["score"] == 0.6
    assert data["mcts"]["best_strategy"] == "decomposition"
    assert data["delta"] == pytest.approx(0.2, abs=1e-6)
    assert "improvement_pct" in data
    assert data["tree"] == "TREE"


def test_static_improvement_pct():
    assert ComparisonService._improvement_pct(0.5, 0.25) == pytest.approx(50.0)
    assert ComparisonService._improvement_pct(0.0, 0.25) == 0.0


def test_mock_client_constructed_for_mock_provider(monkeypatch):
    """When no llm_client given and provider=mock, a MockLLMClient is built (no network)."""
    sentinel_pipe = MagicMock()
    svc = ComparisonService(pipeline=sentinel_pipe, provider="mock", settings=_settings(True))
    # The internal client is a real MockLLMClient with provider attribute "mock".
    assert getattr(svc._llm, "provider", None) == "mock"
