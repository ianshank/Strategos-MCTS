"""Tests for the harness hardening pass: settings-driven values, logging, and
the topology factory.

Covers behaviour added when removing hardcoded values from the harness and
wiring previously-unused settings (``TOPOLOGY``/``TOPOLOGY_*``, per-tool
timeouts, hashed-edit window, read-anchor cap, compressor budgets, planner
temperature) through ``HarnessFactory`` and ``register_builtin_tools``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from src.framework.harness.context.compressor import EpisodicCompressor
from src.framework.harness.context.injector import DefaultContextInjector
from src.framework.harness.factories import HarnessFactory
from src.framework.harness.planner import LLMPlanner
from src.framework.harness.ralph.completion import is_complete
from src.framework.harness.settings import (
    AggregationPolicy,
    HarnessPermissions,
    HarnessSettings,
    TopologyName,
)
from src.framework.harness.state import (
    AcceptanceCriterion,
    HarnessState,
    Observation,
    Plan,
    Task,
)
from src.framework.harness.tools import ToolRegistry
from src.framework.harness.tools.builtins import register_builtin_tools
from src.framework.harness.tools.builtins.fs import file_edit_hashed_tool, file_read_tool
from src.framework.harness.tools.builtins.shell import shell_tool
from src.framework.harness.tools.hashed_edit import file_sha256, window_hash
from src.framework.harness.topology import (
    AgentOutcome,
    ExpertPoolTopology,
    FanOutInTopology,
    HierarchicalTopology,
    PipelineTopology,
    ProducerReviewerTopology,
    SupervisorTopology,
    aggregate,
)
from src.framework.harness.verifier import AcceptanceCriteriaVerifier

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# EpisodicCompressor (previously untested)
# ---------------------------------------------------------------------------


class TestEpisodicCompressor:
    def test_short_text_unchanged(self) -> None:
        comp = EpisodicCompressor(max_chars=100)
        assert comp.compress("hello") == "hello"

    def test_long_text_truncated_with_marker(self) -> None:
        comp = EpisodicCompressor(max_chars=10, head_chars=3, tail_chars=3, truncation_marker="--")
        out = comp.compress("abcdefghijklmnop")
        assert out == "abc--nop"

    def test_boundary_equal_to_max_is_unchanged(self) -> None:
        comp = EpisodicCompressor(max_chars=5)
        assert comp.compress("12345") == "12345"

    def test_from_settings_uses_configured_budgets(self) -> None:
        hs = HarnessSettings(
            CONTEXT_COMPRESS_MAX_CHARS=40,
            CONTEXT_COMPRESS_HEAD_CHARS=2,
            CONTEXT_COMPRESS_TAIL_CHARS=2,
        )
        comp = EpisodicCompressor.from_settings(hs)
        assert comp.max_chars == 40
        assert comp.head_chars == 2
        assert comp.tail_chars == 2
        assert comp.compress("abcdefghij").startswith("ab")
        assert comp.compress("abcdefghij").endswith("ij")

    def test_compress_never_exceeds_max_chars(self) -> None:
        # head/tail budgets deliberately larger than the cap; output must still
        # respect max_chars (clamped around the marker).
        comp = EpisodicCompressor(max_chars=120, head_chars=500, tail_chars=500)
        out = comp.compress("x" * 1000)
        assert len(out) <= 120
        assert comp.truncation_marker in out


# ---------------------------------------------------------------------------
# New settings fields
# ---------------------------------------------------------------------------


class TestHarnessSettingsNewFields:
    def test_defaults_match_legacy_literals(self) -> None:
        hs = HarnessSettings()
        assert hs.TOOL_TEST_TIMEOUT_SECONDS == 600.0
        assert hs.TOOL_LINT_TIMEOUT_SECONDS == 120.0
        assert hs.TOOL_TYPECHECK_TIMEOUT_SECONDS == 300.0
        assert hs.FILE_READ_MAX_ANCHORS == 200
        assert hs.CONTEXT_COMPRESS_MAX_CHARS == 4000
        assert hs.CONTEXT_COMPRESS_HEAD_CHARS == 1500
        assert hs.CONTEXT_COMPRESS_TAIL_CHARS == 1500
        assert hs.TOPOLOGY_PRODUCER_REVIEWER_MAX_ROUNDS == 3
        assert hs.TOPOLOGY_SUPERVISOR_MAX_ROUNDS == 5
        assert hs.TOPOLOGY_HIERARCHICAL_GROUP_SIZE == 2
        assert hs.PLANNER_TEMPERATURE == 0.0

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARNESS_TOPOLOGY_SUPERVISOR_MAX_ROUNDS", "9")
        monkeypatch.setenv("HARNESS_PLANNER_TEMPERATURE", "0.5")
        hs = HarnessSettings()
        assert hs.TOPOLOGY_SUPERVISOR_MAX_ROUNDS == 9
        assert hs.PLANNER_TEMPERATURE == 0.5

    def test_out_of_bounds_rejected(self) -> None:
        with pytest.raises(ValueError):
            HarnessSettings(PLANNER_TEMPERATURE=5.0)
        with pytest.raises(ValueError):
            HarnessSettings(TOPOLOGY_HIERARCHICAL_GROUP_SIZE=0)


# ---------------------------------------------------------------------------
# Topology factory (realizes previously-dead TOPOLOGY / TOPOLOGY_* settings)
# ---------------------------------------------------------------------------


class TestCreateTopology:
    @pytest.mark.parametrize(
        ("name", "cls"),
        [
            (TopologyName.PIPELINE, PipelineTopology),
            (TopologyName.FAN_OUT_IN, FanOutInTopology),
            (TopologyName.EXPERT_POOL, ExpertPoolTopology),
            (TopologyName.PRODUCER_REVIEWER, ProducerReviewerTopology),
            (TopologyName.SUPERVISOR, SupervisorTopology),
            (TopologyName.HIERARCHICAL, HierarchicalTopology),
        ],
    )
    def test_each_topology_constructed(self, name: TopologyName, cls: type) -> None:
        factory = HarnessFactory(harness_settings=HarnessSettings())
        topo = factory.create_topology(name)
        assert isinstance(topo, cls)

    def test_caps_injected_from_settings(self) -> None:
        hs = HarnessSettings(
            TOPOLOGY_PRODUCER_REVIEWER_MAX_ROUNDS=7,
            TOPOLOGY_SUPERVISOR_MAX_ROUNDS=8,
            TOPOLOGY_HIERARCHICAL_GROUP_SIZE=4,
        )
        factory = HarnessFactory(harness_settings=hs)
        pr = factory.create_topology(TopologyName.PRODUCER_REVIEWER)
        sup = factory.create_topology(TopologyName.SUPERVISOR)
        hier = factory.create_topology(TopologyName.HIERARCHICAL)
        assert isinstance(pr, ProducerReviewerTopology) and pr.max_rounds == 7
        assert isinstance(sup, SupervisorTopology) and sup.max_rounds == 8
        assert isinstance(hier, HierarchicalTopology) and hier.group_size == 4

    def test_default_uses_settings_topology(self) -> None:
        hs = HarnessSettings(TOPOLOGY=TopologyName.PIPELINE)
        factory = HarnessFactory(harness_settings=hs)
        assert isinstance(factory.create_topology(), PipelineTopology)


# ---------------------------------------------------------------------------
# aggregate(): verifier_score ranking branch (previously untested)
# ---------------------------------------------------------------------------


def test_aggregate_verifier_ranked_uses_verifier_score() -> None:
    outcomes = [
        AgentOutcome(agent_name="a", response="x", success=True, confidence=0.1),
        AgentOutcome(agent_name="b", response="y", success=False, error="e", confidence=0.9),
    ]
    chosen = aggregate(outcomes, AggregationPolicy.VERIFIER_RANKED, verifier_score=0.95)
    # Successful outcome ranks above the failed one regardless of confidence.
    assert chosen.agent_name == "a"
    assert chosen.metadata["policy"] == "verifier_ranked"


# ---------------------------------------------------------------------------
# Verifier: partial scoring + invalid-regex fallback
# ---------------------------------------------------------------------------


class TestVerifier:
    @pytest.mark.asyncio
    async def test_partial_criteria_scoring(self) -> None:
        task = Task(
            id="T",
            goal="G",
            acceptance_criteria=(
                AcceptanceCriterion(id="c1", description="found-alpha", check="alpha"),
                AcceptanceCriterion(id="c2", description="missing", check="zeta"),
            ),
        )
        obs = (Observation(invocation_id="i", tool_name="t", success=True, payload="contains alpha only"),)
        result = await AcceptanceCriteriaVerifier().verify(obs, task, None)
        assert result.passed is False
        assert result.score == pytest.approx(0.5)
        assert result.failed_criteria == ("c2",)

    @pytest.mark.asyncio
    async def test_invalid_regex_falls_back_to_substring(self) -> None:
        # "(" is an invalid regex; verifier must fall back to substring match.
        task = Task(
            id="T",
            goal="G",
            acceptance_criteria=(AcceptanceCriterion(id="c1", description="d", check="a(b"),),
        )
        obs = (Observation(invocation_id="i", tool_name="t", success=True, payload="literal a(b here"),)
        result = await AcceptanceCriteriaVerifier().verify(obs, task, None)
        assert result.passed is True
        assert result.score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Builtin fs / shell tools: settings-injected params + permission/error paths
# ---------------------------------------------------------------------------


class TestBuiltinToolWiring:
    @pytest.mark.asyncio
    async def test_file_read_respects_max_anchors(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        target.write_text("\n".join(str(i) for i in range(20)), encoding="utf-8")
        perms = HarnessPermissions(READ=True)
        _, handler = file_read_tool(root=tmp_path, perms=perms, max_anchors=3, window=1)
        out = await handler({"path": "f.txt"})
        # Only 3 anchor lines emitted despite 20 source lines.
        anchor_lines = [ln for ln in out.splitlines() if ln.startswith("L")]
        assert len(anchor_lines) == 3

    @pytest.mark.asyncio
    async def test_file_read_permission_denied(self, tmp_path: Path) -> None:
        _, handler = file_read_tool(root=tmp_path, perms=HarnessPermissions(READ=False))
        assert "permission denied" in await handler({"path": "x"})

    @pytest.mark.asyncio
    async def test_file_edit_hash_mismatch_logs_and_returns(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        target.write_text("line0\nline1\n", encoding="utf-8")
        _, handler = file_edit_hashed_tool(root=tmp_path, perms=HarnessPermissions(WRITE=True))
        out = await handler(
            {
                "path": "f.txt",
                "expected_file_hash": "deadbeef",  # wrong on purpose
                "anchor_line": 0,
                "expected_window_hash": "x",
                "new_content": "new",
            }
        )
        assert out.startswith("hash_mismatch:")

    @pytest.mark.asyncio
    async def test_file_edit_success_with_injected_window(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        target.write_text("line0\nline1\n", encoding="utf-8")
        lines = target.read_text().splitlines()
        _, handler = file_edit_hashed_tool(root=tmp_path, perms=HarnessPermissions(WRITE=True), window=1)
        out = await handler(
            {
                "path": "f.txt",
                "expected_file_hash": file_sha256(target),
                "anchor_line": 0,
                "expected_window_hash": window_hash(lines, 0, 1),
                "new_content": "rewritten",
            }
        )
        assert out.startswith("ok:")
        assert target.read_text() == "rewritten"

    @pytest.mark.asyncio
    async def test_shell_allowlist_denial(self, tmp_path: Path) -> None:
        _, handler = shell_tool(cwd=tmp_path, perms=HarnessPermissions(SHELL=True), allowlist=["echo"])
        out = await handler({"argv": ["rm", "-rf", "/"]})
        assert "not in allowlist" in out

    @pytest.mark.asyncio
    async def test_shell_permission_disabled(self, tmp_path: Path) -> None:
        _, handler = shell_tool(cwd=tmp_path, perms=HarnessPermissions(SHELL=False))
        assert "permission denied" in await handler({"argv": ["echo", "hi"]})

    def test_register_builtin_tools_accepts_settings(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        register_builtin_tools(
            registry,
            root=tmp_path,
            perms=HarnessPermissions(),
            settings=HarnessSettings(TOOL_TEST_TIMEOUT_SECONDS=42.0),
        )
        names = set(registry.list_names())
        assert {"file_read", "file_edit_hashed", "shell", "test_run", "lint_run", "type_check_run"} <= names

    def test_register_builtin_tools_propagates_timeouts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Capture the timeouts/window/anchors actually forwarded to each builder
        # so this guards the settings->tool wiring, not just registration.
        import src.framework.harness.tools.builtins.registration as reg

        captured: dict[str, Any] = {}

        def _capture(key: str, real: Any) -> Any:
            def _factory(**kwargs: Any) -> Any:
                captured[key] = kwargs
                return real(**kwargs)

            return _factory

        monkeypatch.setattr(reg, "test_run_tool", _capture("test", reg.test_run_tool))
        monkeypatch.setattr(reg, "lint_run_tool", _capture("lint", reg.lint_run_tool))
        monkeypatch.setattr(reg, "type_check_run_tool", _capture("typecheck", reg.type_check_run_tool))
        monkeypatch.setattr(reg, "file_read_tool", _capture("read", reg.file_read_tool))

        reg.register_builtin_tools(
            ToolRegistry(),
            root=tmp_path,
            perms=HarnessPermissions(),
            settings=HarnessSettings(
                TOOL_TEST_TIMEOUT_SECONDS=42.0,
                TOOL_LINT_TIMEOUT_SECONDS=7.0,
                TOOL_TYPECHECK_TIMEOUT_SECONDS=9.0,
                FILE_READ_MAX_ANCHORS=11,
                HASHED_EDIT_WINDOW=3,
            ),
        )
        assert captured["test"]["timeout"] == 42.0
        assert captured["lint"]["timeout"] == 7.0
        assert captured["typecheck"]["timeout"] == 9.0
        assert captured["read"]["max_anchors"] == 11
        assert captured["read"]["window"] == 3

    def test_register_builtin_tools_defaults_settings(self, tmp_path: Path) -> None:
        # settings=None must still register every tool (defaults sourced internally).
        registry = ToolRegistry()
        register_builtin_tools(registry, root=tmp_path, perms=HarnessPermissions())
        assert "file_read" in registry.list_names()

    @pytest.mark.asyncio
    async def test_file_read_path_outside_root(self, tmp_path: Path) -> None:
        _, handler = file_read_tool(root=tmp_path, perms=HarnessPermissions(READ=True))
        out = await handler({"path": "../../etc/passwd"})
        assert out.startswith("permission error:")

    @pytest.mark.asyncio
    async def test_file_read_missing_path_arg(self, tmp_path: Path) -> None:
        _, handler = file_read_tool(root=tmp_path, perms=HarnessPermissions(READ=True))
        assert "missing required 'path'" in await handler({})

    @pytest.mark.asyncio
    async def test_file_read_not_found(self, tmp_path: Path) -> None:
        _, handler = file_read_tool(root=tmp_path, perms=HarnessPermissions(READ=True))
        assert "file not found" in await handler({"path": "ghost.txt"})

    @pytest.mark.asyncio
    async def test_file_edit_permission_denied(self, tmp_path: Path) -> None:
        _, handler = file_edit_hashed_tool(root=tmp_path, perms=HarnessPermissions(WRITE=False))
        assert "permission denied" in await handler({"path": "x"})

    @pytest.mark.asyncio
    async def test_file_edit_invalid_arguments(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("a\n", encoding="utf-8")
        _, handler = file_edit_hashed_tool(root=tmp_path, perms=HarnessPermissions(WRITE=True))
        out = await handler({"path": "f.txt", "anchor_line": "not-an-int", "new_content": "x"})
        assert out.startswith("error: invalid arguments")

    @pytest.mark.asyncio
    async def test_shell_command_not_found(self, tmp_path: Path) -> None:
        _, handler = shell_tool(cwd=tmp_path, perms=HarnessPermissions(SHELL=True))
        out = await handler({"argv": ["strategos_no_such_command_xyz"]})
        assert out.startswith("not_found:")

    @pytest.mark.asyncio
    async def test_shell_timeout(self, tmp_path: Path) -> None:
        _, handler = shell_tool(cwd=tmp_path, perms=HarnessPermissions(SHELL=True))
        # Use the Python interpreter rather than a platform-specific `sleep`
        # binary so the timeout path is exercised deterministically on any OS.
        out = await handler({"argv": [sys.executable, "-c", "import time; time.sleep(5)"], "timeout": 0.1})
        assert out.startswith("timeout:")

    @pytest.mark.asyncio
    async def test_shell_invalid_argv(self, tmp_path: Path) -> None:
        _, handler = shell_tool(cwd=tmp_path, perms=HarnessPermissions(SHELL=True))
        assert "must be a non-empty list" in await handler({"argv": "echo"})


# ---------------------------------------------------------------------------
# HarnessFactory: clock / memory / registry construction
# ---------------------------------------------------------------------------


class TestHarnessFactoryConstruction:
    def test_create_clock_deterministic(self, tmp_path: Path) -> None:
        from src.framework.harness.replay.clock import DeterministicClock

        hs = HarnessSettings(DETERMINISTIC_CLOCK=True, SEED=7)
        clock = HarnessFactory(harness_settings=hs).create_clock()
        assert isinstance(clock, DeterministicClock)

    def test_create_clock_system(self) -> None:
        from src.framework.harness.replay.clock import SystemClock

        hs = HarnessSettings(DETERMINISTIC_CLOCK=False)
        clock = HarnessFactory(harness_settings=hs).create_clock()
        assert isinstance(clock, SystemClock)

    def test_create_memory_store(self, tmp_path: Path) -> None:
        from src.framework.harness.memory.markdown import MarkdownMemoryStore

        hs = HarnessSettings(MEMORY_ROOT=tmp_path / "mem")
        store = HarnessFactory(harness_settings=hs).create_memory_store()
        assert isinstance(store, MarkdownMemoryStore)

    def test_create_tool_registry_includes_builtins(self, tmp_path: Path) -> None:
        hs = HarnessSettings(MEMORY_ROOT=tmp_path / "mem")
        registry = HarnessFactory(harness_settings=hs).create_tool_registry(memory_store=None)
        assert "file_read" in registry.list_names()


# ---------------------------------------------------------------------------
# Planner temperature wiring
# ---------------------------------------------------------------------------


class _CapturingLLM:
    """Minimal LLM stub that records the kwargs of the last generate call."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def generate(self, **kwargs: Any) -> Any:  # noqa: ANN401
        self.kwargs = kwargs
        raise RuntimeError("force fallback after capture")


class TestPlannerTemperature:
    @pytest.mark.asyncio
    async def test_temperature_forwarded_to_llm(self) -> None:
        llm = _CapturingLLM()
        planner = LLMPlanner(llm, max_tokens=100, temperature=0.42)
        task = Task(id="T", goal="G")
        # The LLM raises after capturing kwargs; planner falls back heuristically.
        plan = await planner.plan(task)
        assert llm.kwargs["temperature"] == 0.42
        assert plan.task_id == "T"  # heuristic fallback still returns a plan


# ---------------------------------------------------------------------------
# Context injector build() (direct invocation)
# ---------------------------------------------------------------------------


class _StubMemory:
    def __init__(self, index: str) -> None:
        self._index = index

    async def read_index(self) -> str:
        return self._index


class TestInjectorBuild:
    @pytest.mark.asyncio
    async def test_build_compresses_memory_and_adds_rag(self) -> None:
        async def rag(_task: Task) -> tuple[str, ...]:
            return ("snippet-1", "snippet-2")

        injector = DefaultContextInjector(
            memory=_StubMemory("x" * 50),
            compressor=EpisodicCompressor(max_chars=10, head_chars=3, tail_chars=3, truncation_marker="~"),
            rag_provider=rag,
            spec_text="SPEC",
        )
        task = Task(id="T", goal="G", acceptance_criteria=(AcceptanceCriterion(id="c", description="d"),))
        state = HarnessState(iteration=2)
        payload = await injector.build(task, Plan(task_id="T", summary="s", steps=()), state)
        assert "~" in payload.memory_excerpt  # compressed
        assert payload.rag_snippets == ("snippet-1", "snippet-2")
        assert payload.spec_excerpt == "SPEC"
        assert payload.extra["iteration"] == 2

    @pytest.mark.asyncio
    async def test_build_handles_memory_failure_gracefully(self) -> None:
        class _BoomMemory:
            async def read_index(self) -> str:
                raise RuntimeError("boom")

        injector = DefaultContextInjector(memory=_BoomMemory())
        payload = await injector.build(Task(id="T", goal="G"), None, HarnessState())
        assert payload.memory_excerpt == ""  # degraded, not raised


# ---------------------------------------------------------------------------
# Ralph completion marker detection (pure, fully covered)
# ---------------------------------------------------------------------------


class TestIsComplete:
    def test_empty_marker_is_never_complete(self) -> None:
        assert is_complete(None, "", content="anything") is False

    def test_inline_content_match(self) -> None:
        assert is_complete(None, "DONE", content="all DONE here") is True

    def test_inline_content_no_match_no_spec(self) -> None:
        assert is_complete(None, "DONE", content="still working") is False

    def test_spec_file_match(self, tmp_path: Path) -> None:
        spec = tmp_path / "spec.md"
        spec.write_text("work\n<!-- DONE -->\n", encoding="utf-8")
        assert is_complete(spec, "<!-- DONE -->") is True

    def test_spec_file_without_marker(self, tmp_path: Path) -> None:
        spec = tmp_path / "spec.md"
        spec.write_text("no marker yet", encoding="utf-8")
        assert is_complete(spec, "<!-- DONE -->") is False

    def test_missing_spec_path(self, tmp_path: Path) -> None:
        assert is_complete(tmp_path / "nope.md", "DONE") is False
