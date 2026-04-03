"""
Unit tests for Google ADK agent integration modules.

Tests cover:
- base.py: ADKConfig, ADKAgentRequest, ADKAgentResponse, ADKAgentAdapter, ADKAgentFactory
- ml_engineering.py: MLEngineeringAgent
- academic_research.py: AcademicResearchAgent
- data_engineering.py: DataEngineeringAgent
- data_science.py: DataScienceAgent
- deep_search.py: DeepSearchAgent
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

# ── Mock google.adk and google.genai before importing any ADK modules ──
_google_pkg = types.ModuleType("google")
_google_pkg.__path__ = []
_google_adk = types.ModuleType("google.adk")
_google_genai = types.ModuleType("google.genai")

sys.modules.setdefault("google", _google_pkg)
sys.modules.setdefault("google.adk", _google_adk)
sys.modules.setdefault("google.genai", _google_genai)

from src.integrations.google_adk.agents.academic_research import AcademicResearchAgent
from src.integrations.google_adk.agents.data_engineering import DataEngineeringAgent
from src.integrations.google_adk.agents.data_science import DataScienceAgent
from src.integrations.google_adk.agents.deep_search import DeepSearchAgent
from src.integrations.google_adk.agents.ml_engineering import MLEngineeringAgent
from src.integrations.google_adk.base import (
    ADKAgentAdapter,
    ADKAgentFactory,
    ADKAgentRequest,
    ADKAgentResponse,
    ADKBackend,
    ADKConfig,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _local_config(tmp_path: Path) -> ADKConfig:
    """Return an ADKConfig pointing at a temp workspace with LOCAL backend."""
    return ADKConfig(
        backend=ADKBackend.LOCAL,
        workspace_dir=str(tmp_path / "workspace"),
    )


# ── Base module tests ────────────────────────────────────────────────────


@pytest.mark.unit
class TestADKBackend:
    """Tests for ADKBackend enum."""

    def test_enum_values(self):
        assert ADKBackend.ML_DEV.value == "ml_dev"
        assert ADKBackend.VERTEX_AI.value == "vertex_ai"
        assert ADKBackend.LOCAL.value == "local"

    def test_from_string(self):
        assert ADKBackend("local") is ADKBackend.LOCAL
        assert ADKBackend("vertex_ai") is ADKBackend.VERTEX_AI


@pytest.mark.unit
class TestADKConfig:
    """Tests for ADKConfig dataclass."""

    def test_defaults(self):
        cfg = ADKConfig()
        assert cfg.project_id is None
        assert cfg.location == "us-central1"
        assert cfg.model_name == "gemini-2.0-flash-001"
        assert cfg.backend == ADKBackend.LOCAL
        assert cfg.timeout == 300
        assert cfg.max_iterations == 10
        assert cfg.temperature == 0.7
        assert cfg.enable_tracing is True
        assert cfg.enable_search is True
        assert cfg.env_vars == {}

    def test_validate_local_no_project_ok(self):
        cfg = ADKConfig(backend=ADKBackend.LOCAL)
        cfg.validate()  # should not raise

    def test_validate_vertex_requires_project(self):
        cfg = ADKConfig(backend=ADKBackend.VERTEX_AI, project_id=None)
        with pytest.raises(ValueError, match="requires GOOGLE_CLOUD_PROJECT"):
            cfg.validate()

    def test_validate_ml_dev_requires_project(self):
        cfg = ADKConfig(backend=ADKBackend.ML_DEV, project_id=None)
        with pytest.raises(ValueError, match="requires GOOGLE_CLOUD_PROJECT"):
            cfg.validate()

    def test_validate_vertex_with_project_ok(self):
        cfg = ADKConfig(backend=ADKBackend.VERTEX_AI, project_id="my-project")
        cfg.validate()

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("ADK_BACKEND", "local")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-proj")
        monkeypatch.setenv("ADK_TIMEOUT", "60")
        monkeypatch.setenv("ADK_TEMPERATURE", "0.5")
        monkeypatch.setenv("ADK_ENABLE_SEARCH", "false")

        cfg = ADKConfig.from_env()
        assert cfg.backend == ADKBackend.LOCAL
        assert cfg.project_id == "test-proj"
        assert cfg.timeout == 60
        assert cfg.temperature == 0.5
        assert cfg.enable_search is False


@pytest.mark.unit
class TestADKAgentRequest:
    """Tests for ADKAgentRequest model."""

    def test_minimal_request(self):
        req = ADKAgentRequest(query="hello")
        assert req.query == "hello"
        assert req.context == {}
        assert req.session_id is None
        assert req.parameters == {}

    def test_full_request(self):
        req = ADKAgentRequest(
            query="test",
            context={"k": "v"},
            session_id="s1",
            parameters={"p": 1},
        )
        assert req.context == {"k": "v"}
        assert req.session_id == "s1"
        assert req.parameters == {"p": 1}


@pytest.mark.unit
class TestADKAgentResponse:
    """Tests for ADKAgentResponse model."""

    def test_defaults(self):
        resp = ADKAgentResponse(result="ok")
        assert resp.result == "ok"
        assert resp.status == "success"
        assert resp.error is None
        assert resp.metadata == {}
        assert resp.artifacts == []
        assert resp.session_id is None

    def test_error_response(self):
        resp = ADKAgentResponse(result="", status="error", error="boom")
        assert resp.status == "error"
        assert resp.error == "boom"


@pytest.mark.unit
class TestADKAgentAdapter:
    """Tests for the abstract ADKAgentAdapter base class."""

    def test_cannot_instantiate_directly(self, tmp_path):
        with pytest.raises(TypeError):
            ADKAgentAdapter(_local_config(tmp_path), "test")

    def test_workspace_created(self, tmp_path):
        """Concrete subclass should create workspace on init."""

        class _Stub(ADKAgentAdapter):
            async def _agent_initialize(self):
                pass

            async def _agent_invoke(self, request):
                return ADKAgentResponse(result="ok")

        agent = _Stub(_local_config(tmp_path), "stub")
        assert Path(agent.config.workspace_dir).is_dir()
        assert agent.agent_name == "stub"
        assert agent._initialized is False

    def test_get_capabilities(self, tmp_path):
        class _Stub(ADKAgentAdapter):
            async def _agent_initialize(self):
                pass

            async def _agent_invoke(self, request):
                return ADKAgentResponse(result="ok")

        agent = _Stub(_local_config(tmp_path), "stub")
        caps = agent.get_capabilities()
        assert caps["name"] == "stub"
        assert caps["backend"] == "local"
        assert "model" in caps

    def test_invoke_calls_initialize_if_needed(self, tmp_path):
        class _Stub(ADKAgentAdapter):
            init_count = 0

            async def _agent_initialize(self):
                _Stub.init_count += 1

            async def _agent_invoke(self, request):
                return ADKAgentResponse(result="done")

        agent = _Stub(_local_config(tmp_path), "stub")
        resp = asyncio.run(agent.invoke(ADKAgentRequest(query="q")))
        assert resp.result == "done"
        assert _Stub.init_count == 1
        assert agent._initialized is True

    def test_invoke_timeout(self, tmp_path):
        cfg = _local_config(tmp_path)
        cfg.timeout = 0  # immediate timeout

        class _Slow(ADKAgentAdapter):
            async def _agent_initialize(self):
                pass

            async def _agent_invoke(self, request):
                await asyncio.sleep(10)
                return ADKAgentResponse(result="late")

        agent = _Slow(cfg, "slow")
        resp = asyncio.run(agent.invoke(ADKAgentRequest(query="q")))
        assert resp.status == "error"
        assert "timeout" in resp.error.lower()

    def test_invoke_exception_returns_error(self, tmp_path):
        class _Bad(ADKAgentAdapter):
            async def _agent_initialize(self):
                pass

            async def _agent_invoke(self, request):
                raise RuntimeError("kaboom")

        agent = _Bad(_local_config(tmp_path), "bad")
        resp = asyncio.run(agent.invoke(ADKAgentRequest(query="q")))
        assert resp.status == "error"
        assert "kaboom" in resp.error

    def test_cleanup(self, tmp_path):
        class _Stub(ADKAgentAdapter):
            async def _agent_initialize(self):
                pass

            async def _agent_invoke(self, request):
                return ADKAgentResponse(result="ok")

        agent = _Stub(_local_config(tmp_path), "stub")
        asyncio.run(agent.initialize())
        assert agent._initialized is True
        asyncio.run(agent.cleanup())
        assert agent._initialized is False


@pytest.mark.unit
class TestADKAgentFactory:
    """Tests for ADKAgentFactory."""

    def setup_method(self):
        # Clear registry between tests
        ADKAgentFactory._registry.clear()

    def test_register_and_create(self, tmp_path):
        class _Stub(ADKAgentAdapter):
            async def _agent_initialize(self):
                pass

            async def _agent_invoke(self, request):
                return ADKAgentResponse(result="ok")

        ADKAgentFactory.register("stub", _Stub)
        agent = ADKAgentFactory.create("stub", _local_config(tmp_path))
        assert isinstance(agent, _Stub)
        assert agent.agent_name == "stub"

    def test_create_unknown_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown agent type"):
            ADKAgentFactory.create("nonexistent", _local_config(tmp_path))

    def test_list_agent_types(self):
        class _A(ADKAgentAdapter):
            async def _agent_initialize(self):
                pass

            async def _agent_invoke(self, request):
                return ADKAgentResponse(result="ok")

        ADKAgentFactory.register("alpha", _A)
        ADKAgentFactory.register("beta", _A)
        assert sorted(ADKAgentFactory.list_agent_types()) == ["alpha", "beta"]


# ── ML Engineering Agent tests ───────────────────────────────────────────


@pytest.mark.unit
class TestMLEngineeringAgent:
    """Tests for MLEngineeringAgent."""

    def test_init(self, tmp_path):
        agent = MLEngineeringAgent(_local_config(tmp_path))
        assert agent.agent_name == "ml_engineering"
        assert agent.task_dir.is_dir()
        assert agent.output_dir.is_dir()
        assert "task_type" in agent.default_task_config

    def test_setup_task_environment(self, tmp_path):
        agent = MLEngineeringAgent(_local_config(tmp_path))
        agent._setup_task_environment()
        for sub in ["regression", "classification", "clustering", "forecasting"]:
            assert (agent.task_dir / sub).is_dir()

    def test_get_capabilities(self, tmp_path):
        agent = MLEngineeringAgent(_local_config(tmp_path))
        caps = agent.get_capabilities()
        assert caps["agent_type"] == "ml_engineering"
        assert "supported_tasks" in caps
        assert "supported_metrics" in caps
        assert "Tabular Regression" in caps["supported_tasks"]
        assert "rmse" in caps["supported_metrics"]

    def test_invoke_missing_data_path(self, tmp_path):
        agent = MLEngineeringAgent(_local_config(tmp_path))
        agent._initialized = True  # skip _agent_initialize (needs google.adk import)
        req = ADKAgentRequest(query="train model", parameters={})
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "error"
        assert "data_path" in resp.error

    def test_invoke_success(self, tmp_path):
        agent = MLEngineeringAgent(_local_config(tmp_path))
        agent._initialized = True
        req = ADKAgentRequest(
            query="train regression",
            parameters={
                "task_name": "house_prices",
                "data_path": "/data/train.csv",
                "metric": "rmse",
            },
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "success"
        assert "house_prices" in resp.result
        assert resp.metadata["task_name"] == "house_prices"
        assert len(resp.artifacts) == 1

    def test_generate_ml_task_plan(self, tmp_path):
        agent = MLEngineeringAgent(_local_config(tmp_path))
        plan = agent._generate_ml_task_plan(
            {
                "task_name": "t",
                "task_type": "Tabular Regression",
                "data_path": "/d",
                "metric": "rmse",
                "lower_is_better": True,
                "query": "q",
            }
        )
        assert "MLE-STAR" in plan
        assert "lower is better" in plan


# ── Academic Research Agent tests ────────────────────────────────────────


@pytest.mark.unit
class TestAcademicResearchAgent:
    """Tests for AcademicResearchAgent."""

    def test_init(self, tmp_path):
        agent = AcademicResearchAgent(_local_config(tmp_path))
        assert agent.agent_name == "academic_research"
        assert agent.papers_dir.is_dir()
        assert agent.analysis_dir.is_dir()
        assert agent.citations_dir.is_dir()
        assert agent.citation_start_date == "2023-01-01"

    def test_get_capabilities(self, tmp_path):
        agent = AcademicResearchAgent(_local_config(tmp_path))
        caps = agent.get_capabilities()
        assert caps["agent_type"] == "academic_research"
        assert "paper_analysis" in caps["features"]
        assert "citation_discovery" in caps["features"]
        assert caps["citation_date_filter"] == "2023-01-01"

    def test_invoke_full_analysis_missing_paper(self, tmp_path):
        agent = AcademicResearchAgent(_local_config(tmp_path))
        agent._initialized = True
        req = ADKAgentRequest(
            query="analyze",
            parameters={"task_type": "full_analysis"},
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "error"
        assert "paper_path" in resp.error or "paper_title" in resp.error

    def test_invoke_full_analysis_success(self, tmp_path):
        agent = AcademicResearchAgent(_local_config(tmp_path))
        agent._initialized = True
        req = ADKAgentRequest(
            query="Analyze transformer paper",
            parameters={
                "task_type": "full_analysis",
                "paper_title": "Attention Is All You Need",
            },
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "success"
        assert "Attention Is All You Need" in resp.result
        assert len(resp.artifacts) == 1

    def test_invoke_citation_discovery(self, tmp_path):
        agent = AcademicResearchAgent(_local_config(tmp_path))
        agent._initialized = True
        req = ADKAgentRequest(
            query="Find citations",
            parameters={
                "task_type": "citation_discovery",
                "paper_title": "BERT",
            },
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "success"
        assert "BERT" in resp.result

    def test_invoke_future_directions(self, tmp_path):
        agent = AcademicResearchAgent(_local_config(tmp_path))
        agent._initialized = True
        req = ADKAgentRequest(
            query="Suggest directions",
            parameters={
                "task_type": "future_directions",
                "paper_title": "GPT-4",
            },
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "success"

    def test_invoke_unknown_task_type(self, tmp_path):
        agent = AcademicResearchAgent(_local_config(tmp_path))
        agent._initialized = True
        req = ADKAgentRequest(
            query="x",
            parameters={"task_type": "unknown_type"},
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "error"
        assert "Unknown task type" in resp.error

    def test_sanitize_filename(self, tmp_path):
        agent = AcademicResearchAgent(_local_config(tmp_path))
        assert agent._sanitize_filename("Hello World!") == "Hello_World"
        # Long names are truncated to 100 chars
        long = "a" * 200
        assert len(agent._sanitize_filename(long)) == 100


# ── Data Engineering Agent tests ─────────────────────────────────────────


@pytest.mark.unit
class TestDataEngineeringAgent:
    """Tests for DataEngineeringAgent."""

    def test_init(self, tmp_path):
        agent = DataEngineeringAgent(_local_config(tmp_path))
        assert agent.agent_name == "data_engineering"
        assert agent.pipelines_dir.is_dir()
        assert agent.schemas_dir.is_dir()
        assert agent.udfs_dir.is_dir()
        assert agent.logs_dir.is_dir()

    def test_get_capabilities(self, tmp_path):
        agent = DataEngineeringAgent(_local_config(tmp_path))
        caps = agent.get_capabilities()
        assert caps["agent_type"] == "data_engineering"
        assert "pipeline_design" in caps["features"]
        assert "sqlx_generation" in caps["features"]
        assert "table" in caps["supported_transforms"]

    def test_invoke_pipeline_design(self, tmp_path):
        agent = DataEngineeringAgent(_local_config(tmp_path))
        agent._initialized = True
        req = ADKAgentRequest(
            query="Build ETL pipeline",
            parameters={
                "task_type": "pipeline_design",
                "pipeline_name": "etl_v1",
                "source_tables": ["raw_events", "raw_users"],
                "target_table": "analytics.events_enriched",
            },
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "success"
        assert "etl_v1" in resp.result
        assert resp.metadata["pipeline_name"] == "etl_v1"

    def test_invoke_sqlx_generation(self, tmp_path):
        agent = DataEngineeringAgent(_local_config(tmp_path))
        agent._initialized = True
        req = ADKAgentRequest(
            query="Create user activity table",
            parameters={
                "task_type": "sqlx_generation",
                "table_name": "user_activity",
                "transformation_type": "table",
            },
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "success"
        assert "user_activity" in resp.metadata["table_name"]

    def test_invoke_troubleshooting(self, tmp_path):
        agent = DataEngineeringAgent(_local_config(tmp_path))
        agent._initialized = True
        req = ADKAgentRequest(
            query="Pipeline fails with ref error",
            context={"error_log": "ERROR: ref not found"},
            parameters={"task_type": "troubleshooting"},
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "success"
        assert "Troubleshooting" in resp.result

    def test_invoke_optimization(self, tmp_path):
        agent = DataEngineeringAgent(_local_config(tmp_path))
        agent._initialized = True
        req = ADKAgentRequest(
            query="Speed up pipeline",
            parameters={
                "task_type": "optimization",
                "pipeline_name": "slow_pipe",
            },
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "success"
        assert "Optimization" in resp.result

    def test_invoke_schema_design(self, tmp_path):
        agent = DataEngineeringAgent(_local_config(tmp_path))
        agent._initialized = True
        req = ADKAgentRequest(
            query="Design user table",
            parameters={
                "task_type": "schema_design",
                "table_name": "users",
            },
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "success"
        assert "users" in resp.result

    def test_invoke_unknown_task_type(self, tmp_path):
        agent = DataEngineeringAgent(_local_config(tmp_path))
        agent._initialized = True
        req = ADKAgentRequest(
            query="x",
            parameters={"task_type": "invalid"},
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "error"


# ── Data Science Agent tests ─────────────────────────────────────────────


@pytest.mark.unit
class TestDataScienceAgent:
    """Tests for DataScienceAgent."""

    def test_init(self, tmp_path):
        agent = DataScienceAgent(_local_config(tmp_path))
        assert agent.agent_name == "data_science"
        assert agent.dataset_config_dir.is_dir()
        assert agent.query_history_dir.is_dir()
        assert agent.visualization_dir.is_dir()
        assert "database" in agent.sub_agents

    def test_get_capabilities(self, tmp_path):
        agent = DataScienceAgent(_local_config(tmp_path))
        # _setup_database_config is called only during _agent_initialize,
        # so set available_backends manually for capability testing
        agent.available_backends = []
        caps = agent.get_capabilities()
        assert caps["agent_type"] == "data_science"
        assert "nl2sql_translation" in caps["features"]
        assert "arima" in caps["bqml_models"]

    def test_invoke_nl2sql_no_backend(self, tmp_path):
        agent = DataScienceAgent(_local_config(tmp_path))
        agent._initialized = True
        agent.available_backends = []
        req = ADKAgentRequest(
            query="Show total sales",
            parameters={"task_type": "nl2sql", "backend": "bigquery"},
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "error"
        assert "not configured" in resp.error

    def test_invoke_nl2sql_success(self, tmp_path):
        agent = DataScienceAgent(_local_config(tmp_path))
        agent._initialized = True
        agent.available_backends = ["bigquery"]
        req = ADKAgentRequest(
            query="Show total sales by region",
            parameters={
                "task_type": "nl2sql",
                "dataset_name": "sales_db",
                "backend": "bigquery",
            },
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "success"
        assert "NL2SQL" in resp.result
        assert resp.metadata["backend"] == "bigquery"

    def test_invoke_analysis(self, tmp_path):
        agent = DataScienceAgent(_local_config(tmp_path))
        agent._initialized = True
        agent.available_backends = []
        req = ADKAgentRequest(
            query="Analyze customer churn",
            parameters={"task_type": "analysis", "analysis_type": "exploratory"},
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "success"
        assert "Analysis" in resp.result

    def test_invoke_bqml(self, tmp_path):
        agent = DataScienceAgent(_local_config(tmp_path))
        agent._initialized = True
        agent.available_backends = []
        req = ADKAgentRequest(
            query="Forecast revenue",
            parameters={
                "task_type": "bqml",
                "model_type": "arima",
                "target_column": "revenue",
            },
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "success"
        assert "ARIMA" in resp.result
        assert resp.metadata["model_type"] == "arima"

    def test_invoke_unknown_task_type(self, tmp_path):
        agent = DataScienceAgent(_local_config(tmp_path))
        agent._initialized = True
        agent.available_backends = []
        req = ADKAgentRequest(query="x", parameters={"task_type": "nope"})
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "error"


# ── Deep Search Agent tests ──────────────────────────────────────────────


@pytest.mark.unit
class TestDeepSearchAgent:
    """Tests for DeepSearchAgent."""

    def test_init(self, tmp_path):
        agent = DeepSearchAgent(_local_config(tmp_path))
        assert agent.agent_name == "deep_search"
        assert agent.research_dir.is_dir()
        assert agent.plans_dir.is_dir()
        assert agent.reports_dir.is_dir()
        assert agent.sources_dir.is_dir()
        assert agent.research_sessions == {}

    def test_get_capabilities(self, tmp_path):
        agent = DeepSearchAgent(_local_config(tmp_path))
        caps = agent.get_capabilities()
        assert caps["agent_type"] == "deep_search"
        assert "planning" in caps["workflow_phases"]
        assert "human_in_loop_planning" in caps["features"]

    def test_generate_research_id(self, tmp_path):
        agent = DeepSearchAgent(_local_config(tmp_path))
        rid = agent._generate_research_id("AI Safety")
        assert rid.startswith("aisafety-")
        # IDs should be unique
        rid2 = agent._generate_research_id("AI Safety")
        assert rid != rid2

    def test_invoke_planning_phase(self, tmp_path):
        agent = DeepSearchAgent(_local_config(tmp_path))
        agent._initialized = True
        req = ADKAgentRequest(
            query="Quantum computing advances",
            parameters={"phase": "planning"},
            session_id="test-session-1",
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "success"
        assert "Research Plan" in resp.result
        assert resp.metadata["phase"] == "planning"
        assert resp.session_id == "test-session-1"
        # Session should be stored
        assert "test-session-1" in agent.research_sessions

    def test_invoke_execution_without_plan(self, tmp_path):
        agent = DeepSearchAgent(_local_config(tmp_path))
        agent._initialized = True
        req = ADKAgentRequest(
            query="Execute research",
            parameters={"phase": "execution"},
            session_id="nonexistent",
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "error"
        assert "No research session" in resp.error

    def test_invoke_execution_after_planning(self, tmp_path):
        agent = DeepSearchAgent(_local_config(tmp_path))
        agent._initialized = True

        # Phase 1: planning
        plan_req = ADKAgentRequest(
            query="LLM reasoning",
            parameters={"phase": "planning"},
            session_id="sess-42",
        )
        plan_resp = asyncio.run(agent.invoke(plan_req))
        assert plan_resp.status == "success"

        # Phase 2: execution
        exec_req = ADKAgentRequest(
            query="Execute research plan",
            parameters={"phase": "execution"},
            session_id="sess-42",
        )
        exec_resp = asyncio.run(agent.invoke(exec_req))
        assert exec_resp.status == "success"
        assert "Research Report" in exec_resp.result
        assert exec_resp.metadata["phase"] == "execution"

    def test_invoke_full_phase(self, tmp_path):
        agent = DeepSearchAgent(_local_config(tmp_path))
        agent._initialized = True
        req = ADKAgentRequest(
            query="Climate change mitigation",
            parameters={"phase": "full"},
            session_id="full-1",
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "success"
        assert "Research Report" in resp.result

    def test_invoke_unknown_phase(self, tmp_path):
        agent = DeepSearchAgent(_local_config(tmp_path))
        agent._initialized = True
        req = ADKAgentRequest(
            query="x",
            parameters={"phase": "invalid"},
        )
        resp = asyncio.run(agent.invoke(req))
        assert resp.status == "error"
        assert "Unknown phase" in resp.error

    def test_list_and_get_research_sessions(self, tmp_path):
        agent = DeepSearchAgent(_local_config(tmp_path))
        agent._initialized = True

        assert agent.list_research_sessions() == []
        assert agent.get_research_status("nope") is None

        # Create a session via planning
        req = ADKAgentRequest(
            query="topic1",
            parameters={"phase": "planning"},
            session_id="s1",
        )
        asyncio.run(agent.invoke(req))

        sessions = agent.list_research_sessions()
        assert len(sessions) == 1
        assert sessions[0]["research_id"] == "s1"
        assert sessions[0]["topic"] == "topic1"

        status = agent.get_research_status("s1")
        assert status is not None
        assert status["status"] == "plan_pending_approval"
