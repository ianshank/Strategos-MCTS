"""
Unit tests for src/benchmark/__main__.py

Covers: main() call, KeyboardInterrupt handling.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class TestBenchmarkMain:
    """Tests for the __main__.py entry point."""

    @patch("src.benchmark.cli.main")
    def test_main_called(self, mock_main):
        """__main__ should call cli.main()."""
        mock_main.return_value = None

        # Import and execute the module code
        import importlib
        import src.benchmark.__main__ as mod

        importlib.reload(mod)
        mock_main.assert_called()

    @patch("src.benchmark.cli.main", side_effect=KeyboardInterrupt)
    def test_keyboard_interrupt_exits_130(self, mock_main):
        """KeyboardInterrupt should cause sys.exit(130)."""
        import importlib

        with pytest.raises(SystemExit) as exc_info:
            import src.benchmark.__main__ as mod

            importlib.reload(mod)

        assert exc_info.value.code == 130

    @patch("src.benchmark.cli.main")
    def test_main_no_exception(self, mock_main):
        """Normal execution should not raise."""
        mock_main.return_value = None

        import importlib
        import src.benchmark.__main__ as mod

        # Should not raise
        importlib.reload(mod)
