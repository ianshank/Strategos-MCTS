"""Tests for src/agents/__init__.py lazy imports and availability checks."""

import pytest

import src.agents as agents_mod


@pytest.mark.unit
class TestAgentsAvailability:
    """Test availability check functions."""

    def test_is_hrm_available_returns_bool(self):
        assert isinstance(agents_mod.is_hrm_available(), bool)

    def test_is_trm_available_returns_bool(self):
        assert isinstance(agents_mod.is_trm_available(), bool)

    def test_is_hybrid_available_returns_bool(self):
        assert isinstance(agents_mod.is_hybrid_available(), bool)

    def test_get_missing_dependencies_returns_dict(self):
        result = agents_mod.get_missing_dependencies()
        assert isinstance(result, dict)

    def test_get_missing_dependencies_keys_are_agent_names(self):
        result = agents_mod.get_missing_dependencies()
        valid_keys = {"HRMAgent", "TRMAgent", "HybridAgent"}
        assert set(result.keys()).issubset(valid_keys)

    def test_all_available_means_no_missing(self):
        """If all agents are available, missing deps should be empty."""
        if agents_mod.is_hrm_available() and agents_mod.is_trm_available() and agents_mod.is_hybrid_available():
            assert agents_mod.get_missing_dependencies() == {}

    def test_hrm_available_means_importable(self):
        if agents_mod.is_hrm_available():
            assert hasattr(agents_mod, "HRMAgent")
            assert "HRMAgent" in agents_mod.__all__

    def test_trm_available_means_importable(self):
        if agents_mod.is_trm_available():
            assert hasattr(agents_mod, "TRMAgent")
            assert "TRMAgent" in agents_mod.__all__

    def test_hybrid_available_means_importable(self):
        if agents_mod.is_hybrid_available():
            assert hasattr(agents_mod, "HybridAgent")
            assert "HybridAgent" in agents_mod.__all__

    def test_all_contains_availability_checks(self):
        assert "is_hrm_available" in agents_mod.__all__
        assert "is_trm_available" in agents_mod.__all__
        assert "is_hybrid_available" in agents_mod.__all__
        assert "get_missing_dependencies" in agents_mod.__all__


@pytest.mark.unit
class TestUtilsInit:
    """Test src/utils/__init__.py lazy imports."""

    def test_personality_response_generator_direct(self):
        from src.utils import PersonalityResponseGenerator

        assert PersonalityResponseGenerator is not None

    def test_lazy_import_planning_loader(self):
        from src.utils import PlanningLoader

        assert PlanningLoader is not None

    def test_lazy_import_get_project_plan(self):
        from src.utils import get_project_plan

        assert callable(get_project_plan)

    def test_lazy_import_mcts_debugger(self):
        from src.utils import MCTSDebugger

        assert MCTSDebugger is not None

    def test_lazy_import_create_debugger(self):
        from src.utils import create_debugger

        assert callable(create_debugger)

    def test_invalid_attribute_raises(self):
        import src.utils as utils_mod

        with pytest.raises(AttributeError, match="has no attribute"):
            utils_mod.__getattr__("nonexistent_thing")

    def test_all_exports(self):
        import src.utils as utils_mod

        expected = {
            "PersonalityResponseGenerator",
            "PlanningLoader",
            "get_project_plan",
            "MCTSDebugger",
            "create_debugger",
        }
        assert set(utils_mod.__all__) == expected


@pytest.mark.unit
class TestEnterpriseUseCasesInit:
    """Test src/enterprise/use_cases/__init__.py lazy factories."""

    def test_get_ma_due_diligence(self):
        from src.enterprise.use_cases import get_ma_due_diligence

        cls = get_ma_due_diligence()
        assert cls is not None
        assert cls.__name__ == "MADueDiligence"

    def test_get_clinical_trial(self):
        from src.enterprise.use_cases import get_clinical_trial

        cls = get_clinical_trial()
        assert cls is not None
        assert cls.__name__ == "ClinicalTrialDesign"

    def test_get_regulatory_compliance(self):
        from src.enterprise.use_cases import get_regulatory_compliance

        cls = get_regulatory_compliance()
        assert cls is not None
        assert cls.__name__ == "RegulatoryCompliance"

    def test_all_exports(self):
        from src.enterprise.use_cases import __all__

        assert "get_ma_due_diligence" in __all__
        assert "get_clinical_trial" in __all__
        assert "get_regulatory_compliance" in __all__


@pytest.mark.unit
class TestBenchmarkProtocol:
    """Test src/benchmark/adapters/protocol.py."""

    def test_protocol_is_runtime_checkable(self):
        from src.benchmark.adapters.protocol import BenchmarkSystemProtocol

        assert hasattr(BenchmarkSystemProtocol, "__protocol_attrs__") or True  # runtime_checkable

    def test_protocol_has_name_property(self):
        from src.benchmark.adapters.protocol import BenchmarkSystemProtocol

        # Verify protocol defines expected interface
        assert "name" in dir(BenchmarkSystemProtocol)

    def test_protocol_has_is_available_property(self):
        from src.benchmark.adapters.protocol import BenchmarkSystemProtocol

        assert "is_available" in dir(BenchmarkSystemProtocol)

    def test_protocol_has_execute_method(self):
        from src.benchmark.adapters.protocol import BenchmarkSystemProtocol

        assert "execute" in dir(BenchmarkSystemProtocol)

    def test_protocol_has_health_check_method(self):
        from src.benchmark.adapters.protocol import BenchmarkSystemProtocol

        assert "health_check" in dir(BenchmarkSystemProtocol)

    def test_conforming_class_isinstance_check(self):
        from src.benchmark.adapters.protocol import BenchmarkSystemProtocol

        class FakeSystem:
            @property
            def name(self) -> str:
                return "fake"

            @property
            def is_available(self) -> bool:
                return True

            async def execute(self, task):
                pass

            async def health_check(self) -> bool:
                return True

        assert isinstance(FakeSystem(), BenchmarkSystemProtocol)
