"""
Extended tests for agents module initialization - Edge Cases.

These tests focus on edge cases for module imports and availability
checking to achieve 70%+ code coverage.
"""

from __future__ import annotations

import pytest

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
]

# Check torch availability
try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# get_missing_dependencies Edge Cases
# =============================================================================


class TestGetMissingDependencies:
    """Extended tests for get_missing_dependencies function."""

    def test_missing_deps_keys_are_strings(self):
        """Test that all keys in missing dependencies are strings."""
        from src.agents import get_missing_dependencies

        missing = get_missing_dependencies()

        for key in missing:
            assert isinstance(key, str)
            assert key in ["HRMAgent", "TRMAgent", "HybridAgent"]

    def test_missing_deps_values_contain_instructions(self):
        """Test that values contain installation instructions."""
        from src.agents import get_missing_dependencies

        missing = get_missing_dependencies()

        for value in missing.values():
            assert isinstance(value, str)
            assert len(value) > 0
            # Should contain some installation guidance
            assert "install" in value.lower() or "pip" in value.lower() or "check" in value.lower()

    def test_missing_deps_pytorch_message(self):
        """Test that PyTorch agents have correct message format."""
        from src.agents import get_missing_dependencies, is_hrm_available

        missing = get_missing_dependencies()

        if not is_hrm_available():
            assert "HRMAgent" in missing
            assert "PyTorch" in missing["HRMAgent"]
            assert "pip install" in missing["HRMAgent"]


# =============================================================================
# Module __all__ Tests
# =============================================================================


class TestModuleAllExports:
    """Tests for __all__ export consistency."""

    def test_all_is_list(self):
        """Test that __all__ is a list."""
        from src import agents

        assert isinstance(agents.__all__, list)

    def test_all_contains_only_strings(self):
        """Test that __all__ contains only strings."""
        from src import agents

        for name in agents.__all__:
            assert isinstance(name, str)

    def test_all_items_are_importable(self):
        """Test that all items in __all__ are actually importable."""
        from src import agents

        for name in agents.__all__:
            # Each name should be a valid attribute
            assert hasattr(agents, name), f"{name} in __all__ but not importable"

    def test_availability_functions_always_in_all(self):
        """Test that availability functions are always in __all__."""
        from src import agents

        required = ["is_hrm_available", "is_trm_available", "is_hybrid_available", "get_missing_dependencies"]

        for name in required:
            assert name in agents.__all__

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_agent_classes_in_all_when_available(self):
        """Test that agent classes are in __all__ when PyTorch is available."""
        from src import agents

        if agents.is_hrm_available():
            assert "HRMAgent" in agents.__all__
            assert "create_hrm_agent" in agents.__all__

        if agents.is_trm_available():
            assert "TRMAgent" in agents.__all__
            assert "create_trm_agent" in agents.__all__


# =============================================================================
# Availability Function Edge Cases
# =============================================================================


class TestAvailabilityFunctions:
    """Extended tests for availability checking functions."""

    def test_availability_functions_are_pure(self):
        """Test that availability functions return same value on repeated calls."""
        from src.agents import is_hrm_available, is_hybrid_available, is_trm_available

        # Call multiple times
        hrm1, hrm2, hrm3 = is_hrm_available(), is_hrm_available(), is_hrm_available()
        assert hrm1 == hrm2 == hrm3

        trm1, trm2, trm3 = is_trm_available(), is_trm_available(), is_trm_available()
        assert trm1 == trm2 == trm3

        hybrid1, hybrid2 = is_hybrid_available(), is_hybrid_available()
        assert hybrid1 == hybrid2

    def test_availability_matches_missing_deps(self):
        """Test that availability functions match get_missing_dependencies."""
        from src.agents import (
            get_missing_dependencies,
            is_hrm_available,
            is_hybrid_available,
            is_trm_available,
        )

        missing = get_missing_dependencies()

        # If available, should not be in missing
        if is_hrm_available():
            assert "HRMAgent" not in missing
        else:
            assert "HRMAgent" in missing

        if is_trm_available():
            assert "TRMAgent" not in missing
        else:
            assert "TRMAgent" in missing

        if is_hybrid_available():
            assert "HybridAgent" not in missing
        else:
            assert "HybridAgent" in missing


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module import behavior."""

    def test_import_does_not_raise(self):
        """Test that importing the module doesn't raise exceptions."""
        # This should not raise even without PyTorch
        from src import agents  # noqa: F401

    def test_availability_checks_importable_directly(self):
        """Test direct import of availability functions."""
        from src.agents import get_missing_dependencies, is_hrm_available

        # Should work regardless of PyTorch
        assert callable(is_hrm_available)
        assert callable(get_missing_dependencies)

    def test_logger_available(self):
        """Test that module has a logger."""
        from src import agents

        # Module should have created a logger
        assert hasattr(agents, "_logger") or hasattr(agents, "logging")


# =============================================================================
# Docstring Tests
# =============================================================================


class TestDocumentation:
    """Tests for module documentation."""

    def test_module_docstring_comprehensive(self):
        """Test that module docstring covers key information."""
        from src import agents

        doc = agents.__doc__

        # Should mention key components
        assert "HRM" in doc or "Hierarchical" in doc
        assert "TRM" in doc or "Task" in doc
        assert "Hybrid" in doc

        # Should mention PyTorch
        assert "PyTorch" in doc or "torch" in doc.lower()

        # Should mention installation
        assert "pip" in doc or "install" in doc.lower()

    def test_get_missing_dependencies_docstring(self):
        """Test get_missing_dependencies has example in docstring."""
        from src.agents import get_missing_dependencies

        doc = get_missing_dependencies.__doc__

        assert doc is not None
        assert "Returns:" in doc or "return" in doc.lower()
        assert "dict" in doc.lower()

    def test_availability_functions_documented(self):
        """Test all availability functions have docstrings."""
        from src.agents import is_hrm_available, is_hybrid_available, is_trm_available

        assert is_hrm_available.__doc__ is not None
        assert is_trm_available.__doc__ is not None
        assert is_hybrid_available.__doc__ is not None

        # Each should mention what it checks
        assert "HRM" in is_hrm_available.__doc__ or "available" in is_hrm_available.__doc__.lower()
        assert "TRM" in is_trm_available.__doc__ or "available" in is_trm_available.__doc__.lower()
        assert "Hybrid" in is_hybrid_available.__doc__ or "available" in is_hybrid_available.__doc__.lower()
