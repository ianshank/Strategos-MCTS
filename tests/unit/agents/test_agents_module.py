"""
Unit tests for agents module initialization and exports.

Tests:
- Availability check functions
- get_missing_dependencies helper
- Graceful handling of optional dependencies
- Module-level exports
"""

import pytest

# Check torch availability at module level (don't raise on import)
try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestAgentsModuleExports:
    """Tests for agents module exports and availability checks."""

    def test_availability_functions_exist(self):
        """Test that availability check functions are always importable."""
        from src.agents import (
            get_missing_dependencies,
            is_hrm_available,
            is_hybrid_available,
            is_trm_available,
        )

        # These functions should always be available
        assert callable(is_hrm_available)
        assert callable(is_trm_available)
        assert callable(is_hybrid_available)
        assert callable(get_missing_dependencies)

    def test_availability_functions_return_bool(self):
        """Test that availability functions return boolean values."""
        from src.agents import is_hrm_available, is_hybrid_available, is_trm_available

        # All should return booleans
        assert isinstance(is_hrm_available(), bool)
        assert isinstance(is_trm_available(), bool)
        assert isinstance(is_hybrid_available(), bool)

    def test_get_missing_dependencies_returns_dict(self):
        """Test that get_missing_dependencies returns a dictionary."""
        from src.agents import get_missing_dependencies

        missing = get_missing_dependencies()

        # Should always return a dict
        assert isinstance(missing, dict)

        # All keys should be strings (agent names)
        for key in missing:
            assert isinstance(key, str)

        # All values should be strings (instructions)
        for value in missing.values():
            assert isinstance(value, str)

    def test_missing_dependencies_matches_availability(self):
        """Test that missing dependencies align with availability checks."""
        from src.agents import (
            get_missing_dependencies,
            is_hrm_available,
            is_hybrid_available,
            is_trm_available,
        )

        missing = get_missing_dependencies()

        # HRMAgent should be in missing iff is_hrm_available is False
        if is_hrm_available():
            assert "HRMAgent" not in missing
        else:
            assert "HRMAgent" in missing

        # TRMAgent should be in missing iff is_trm_available is False
        if is_trm_available():
            assert "TRMAgent" not in missing
        else:
            assert "TRMAgent" in missing

        # HybridAgent should be in missing iff is_hybrid_available is False
        if is_hybrid_available():
            assert "HybridAgent" not in missing
        else:
            assert "HybridAgent" in missing

    def test_module_all_contains_availability_functions(self):
        """Test that __all__ always contains availability functions."""
        from src import agents

        # These should always be in __all__
        assert "is_hrm_available" in agents.__all__
        assert "is_trm_available" in agents.__all__
        assert "is_hybrid_available" in agents.__all__
        assert "get_missing_dependencies" in agents.__all__

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_hrm_available_with_torch(self):
        """Test HRM is available when PyTorch is installed."""
        from src.agents import HRMAgent, is_hrm_available

        assert is_hrm_available() is True
        assert HRMAgent is not None

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_trm_available_with_torch(self):
        """Test TRM is available when PyTorch is installed."""
        from src.agents import TRMAgent, is_trm_available

        assert is_trm_available() is True
        assert TRMAgent is not None


class TestAgentsModuleDocumentation:
    """Tests for agents module documentation."""

    def test_module_has_docstring(self):
        """Test that module has a docstring."""
        from src import agents

        assert agents.__doc__ is not None
        assert len(agents.__doc__) > 100  # Should have meaningful documentation

    def test_docstring_mentions_pytorch(self):
        """Test that docstring mentions PyTorch dependency."""
        from src import agents

        assert "PyTorch" in agents.__doc__ or "torch" in agents.__doc__.lower()

    def test_availability_functions_have_docstrings(self):
        """Test that availability functions are documented."""
        from src.agents import (
            get_missing_dependencies,
            is_hrm_available,
            is_hybrid_available,
            is_trm_available,
        )

        assert is_hrm_available.__doc__ is not None
        assert is_trm_available.__doc__ is not None
        assert is_hybrid_available.__doc__ is not None
        assert get_missing_dependencies.__doc__ is not None
