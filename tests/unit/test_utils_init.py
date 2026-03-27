"""
Unit tests for src/utils/__init__.py

Covers: lazy imports for PlanningLoader, get_project_plan,
MCTSDebugger, create_debugger, and AttributeError for unknown attrs.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestUtilsInit:
    """Tests for src/utils/__init__.py lazy imports."""

    def test_personality_response_generator_available(self):
        """PersonalityResponseGenerator should be directly importable."""
        from src.utils import PersonalityResponseGenerator

        assert PersonalityResponseGenerator is not None

    def test_lazy_import_planning_loader(self):
        """PlanningLoader should be lazily importable."""
        from src.utils import PlanningLoader

        assert PlanningLoader is not None

    def test_lazy_import_get_project_plan(self):
        """get_project_plan should be lazily importable."""
        from src.utils import get_project_plan

        assert callable(get_project_plan)

    def test_lazy_import_mcts_debugger(self):
        """MCTSDebugger should be lazily importable."""
        from src.utils import MCTSDebugger

        assert MCTSDebugger is not None

    def test_lazy_import_create_debugger(self):
        """create_debugger should be lazily importable."""
        from src.utils import create_debugger

        assert callable(create_debugger)

    def test_unknown_attribute_raises(self):
        """Accessing unknown attribute should raise AttributeError."""
        with pytest.raises(AttributeError, match="has no attribute"):
            from src import utils

            _ = utils.nonexistent_thing

    def test_all_exports(self):
        """__all__ should contain expected names."""
        import src.utils as utils

        assert "PersonalityResponseGenerator" in utils.__all__
        assert "PlanningLoader" in utils.__all__
        assert "get_project_plan" in utils.__all__
        assert "MCTSDebugger" in utils.__all__
        assert "create_debugger" in utils.__all__
