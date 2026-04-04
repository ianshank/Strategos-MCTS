"""
Unit tests for LLM-Guided MCTS Code Executor.

Tests CodeExecutionResult, CodeExecutor validation, execution, and safety.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("src.framework.mcts.llm_guided.executor")

from src.framework.mcts.llm_guided.executor import (
    ALLOWED_IMPORTS,
    SAFE_BUILTINS,
    CodeExecutionResult,
    CodeExecutor,
    ImportChecker,
    _create_safe_import,
    create_executor_from_config,
)


@pytest.mark.unit
class TestCodeExecutionResult:
    """Tests for CodeExecutionResult dataclass."""

    def test_creation_minimal(self) -> None:
        """Can be created with required fields."""
        result = CodeExecutionResult(
            passed=True,
            num_tests_passed=2,
            num_tests_total=2,
            stdout="ok",
            stderr="",
            errors=[],
            execution_time_ms=10.0,
        )
        assert result.passed is True
        assert result.num_tests_passed == 2
        assert result.timed_out is False
        assert result.syntax_error is False
        assert result.test_results == []

    def test_to_dict(self) -> None:
        """to_dict includes all fields."""
        result = CodeExecutionResult(
            passed=False,
            num_tests_passed=0,
            num_tests_total=1,
            stdout="",
            stderr="err",
            errors=["fail"],
            execution_time_ms=5.0,
            timed_out=True,
        )
        d = result.to_dict()
        assert d["passed"] is False
        assert d["timed_out"] is True
        assert d["errors"] == ["fail"]
        assert "test_results" in d

    def test_repr_passed(self) -> None:
        """repr shows PASSED for passing results."""
        result = CodeExecutionResult(
            passed=True,
            num_tests_passed=3,
            num_tests_total=3,
            stdout="",
            stderr="",
            errors=[],
            execution_time_ms=1.0,
        )
        assert "PASSED" in repr(result)
        assert "3/3" in repr(result)

    def test_repr_failed(self) -> None:
        """repr shows FAILED for failing results."""
        result = CodeExecutionResult(
            passed=False,
            num_tests_passed=1,
            num_tests_total=3,
            stdout="",
            stderr="",
            errors=["x"],
            execution_time_ms=1.0,
        )
        assert "FAILED" in repr(result)
        assert "1/3" in repr(result)


@pytest.mark.unit
class TestImportChecker:
    """Tests for ImportChecker AST visitor."""

    def test_allowed_import(self) -> None:
        """Allowed imports produce no errors."""
        import ast

        code = "import math\nimport re\nimport collections"
        tree = ast.parse(code)
        checker = ImportChecker()
        checker.visit(tree)
        assert checker.disallowed_imports == []

    def test_disallowed_import(self) -> None:
        """Disallowed imports are detected."""
        import ast

        code = "import os\nimport subprocess"
        tree = ast.parse(code)
        checker = ImportChecker()
        checker.visit(tree)
        assert "os" in checker.disallowed_imports
        assert "subprocess" in checker.disallowed_imports

    def test_disallowed_from_import(self) -> None:
        """Disallowed 'from X import Y' detected."""
        import ast

        code = "from os.path import join"
        tree = ast.parse(code)
        checker = ImportChecker()
        checker.visit(tree)
        assert "os.path" in checker.disallowed_imports

    def test_allowed_from_import(self) -> None:
        """Allowed 'from X import Y' passes."""
        import ast

        code = "from collections import defaultdict"
        tree = ast.parse(code)
        checker = ImportChecker()
        checker.visit(tree)
        assert checker.disallowed_imports == []


@pytest.mark.unit
class TestSafeImport:
    """Tests for _create_safe_import."""

    def test_allowed_module(self) -> None:
        """Allowed modules can be imported."""
        safe_import = _create_safe_import({"math"})
        mod = safe_import("math")
        assert hasattr(mod, "sqrt")

    def test_disallowed_module(self) -> None:
        """Disallowed modules raise ImportError."""
        safe_import = _create_safe_import({"math"})
        with pytest.raises(ImportError, match="not allowed"):
            safe_import("os")

    def test_submodule_allowed_by_base(self) -> None:
        """Submodule is allowed if base module is in allowed set."""
        safe_import = _create_safe_import({"collections"})
        mod = safe_import("collections.abc")
        assert mod is not None


@pytest.mark.unit
class TestCodeExecutorInit:
    """Tests for CodeExecutor initialization."""

    def test_default_init(self) -> None:
        """Default initialization sets expected values."""
        executor = CodeExecutor()
        assert executor.timeout_seconds == 5.0
        assert executor.max_memory_mb == 256
        assert executor.allow_network is False
        assert executor.use_subprocess is False

    def test_custom_init(self) -> None:
        """Custom parameters are set correctly."""
        executor = CodeExecutor(
            timeout_seconds=10.0,
            max_memory_mb=512,
            allow_network=True,
            use_subprocess=True,
        )
        assert executor.timeout_seconds == 10.0
        assert executor.max_memory_mb == 512
        assert executor.allow_network is True
        assert executor.use_subprocess is True


@pytest.mark.unit
class TestCodeExecutorValidation:
    """Tests for CodeExecutor.validate_code."""

    def test_valid_code(self) -> None:
        """Valid Python code passes validation."""
        executor = CodeExecutor()
        is_valid, errors = executor.validate_code("x = 1 + 2\nprint(x)")
        assert is_valid is True
        assert errors == []

    def test_syntax_error(self) -> None:
        """Syntax errors are caught."""
        executor = CodeExecutor()
        is_valid, errors = executor.validate_code("def f(\n")
        assert is_valid is False
        assert any("Syntax error" in e for e in errors)

    def test_disallowed_import(self) -> None:
        """Disallowed imports detected in validation."""
        executor = CodeExecutor()
        is_valid, errors = executor.validate_code("import os\nx = os.getcwd()")
        assert is_valid is False
        assert any("Disallowed imports" in e for e in errors)

    def test_dangerous_patterns(self) -> None:
        """Dangerous patterns are detected."""
        executor = CodeExecutor()

        for pattern_code in [
            "os.system('rm -rf /')",
            "eval('bad')",
            "exec('bad')",
            "__import__('os')",
            "open('file.txt')",
        ]:
            is_valid, errors = executor.validate_code(pattern_code)
            assert is_valid is False, f"Should reject: {pattern_code}"

    def test_allowed_imports_pass(self) -> None:
        """Code with only allowed imports passes."""
        executor = CodeExecutor()
        code = "import math\nimport re\nresult = math.sqrt(4)"
        is_valid, errors = executor.validate_code(code)
        assert is_valid is True


@pytest.mark.unit
class TestCodeExecutorExecution:
    """Tests for CodeExecutor.execute."""

    def test_simple_execution(self) -> None:
        """Simple code executes successfully."""
        executor = CodeExecutor()
        result = executor.execute("x = 42")
        assert result.errors == []

    def test_execution_with_passing_tests(self) -> None:
        """Tests that pass produce a passing result."""
        executor = CodeExecutor()
        code = "def add(a, b): return a + b"
        tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
        result = executor.execute(code, test_cases=tests)
        assert result.passed is True
        assert result.num_tests_passed == 2
        assert result.num_tests_total == 2

    def test_execution_with_failing_test(self) -> None:
        """Failing test produces a failed result."""
        executor = CodeExecutor()
        code = "def add(a, b): return a - b"  # intentional bug
        tests = ["assert add(1, 2) == 3"]
        result = executor.execute(code, test_cases=tests)
        assert result.passed is False
        assert result.num_tests_passed == 0

    def test_execution_syntax_error(self) -> None:
        """Code with syntax error returns syntax_error flag."""
        executor = CodeExecutor()
        result = executor.execute("def f(\n")
        assert result.passed is False
        assert result.syntax_error is True

    def test_execution_runtime_error(self) -> None:
        """Runtime errors are captured."""
        executor = CodeExecutor()
        result = executor.execute("x = 1 / 0")
        assert result.passed is False
        assert any("ZeroDivisionError" in e for e in result.errors)

    def test_execution_captures_stdout(self) -> None:
        """Print output is captured."""
        executor = CodeExecutor()
        result = executor.execute("print('hello world')")
        assert "hello world" in result.stdout

    def test_execution_no_tests_not_passed(self) -> None:
        """No tests means passed=False (need at least one test to pass)."""
        executor = CodeExecutor()
        result = executor.execute("x = 1")
        assert result.passed is False
        assert result.num_tests_total == 0

    def test_execution_disallowed_import_rejected(self) -> None:
        """Code importing disallowed modules is rejected before execution."""
        executor = CodeExecutor()
        result = executor.execute("import os\nos.getcwd()")
        assert result.passed is False
        assert result.syntax_error is True  # validation failure sets this

    def test_execution_with_math(self) -> None:
        """Code using allowed imports works."""
        executor = CodeExecutor()
        code = "import math\nresult = math.sqrt(16)"
        tests = ["assert result == 4.0"]
        result = executor.execute(code, test_cases=tests)
        assert result.passed is True

    def test_execution_time_recorded(self) -> None:
        """Execution time is recorded."""
        executor = CodeExecutor()
        result = executor.execute("x = sum(range(100))")
        assert result.execution_time_ms >= 0


@pytest.mark.unit
class TestCodeExecutorRunWithInputs:
    """Tests for CodeExecutor.run_with_inputs."""

    def test_run_with_inputs_passing(self) -> None:
        """run_with_inputs with correct outputs passes."""
        executor = CodeExecutor()
        code = "def solution(x): return x * 2"
        result = executor.run_with_inputs(
            code,
            inputs=[2, 3, 5],
            expected_outputs=[4, 6, 10],
        )
        assert result.passed is True
        assert result.num_tests_passed == 3

    def test_run_with_inputs_failing(self) -> None:
        """run_with_inputs with incorrect outputs fails."""
        executor = CodeExecutor()
        code = "def solution(x): return x + 1"
        result = executor.run_with_inputs(
            code,
            inputs=[2],
            expected_outputs=[4],
        )
        assert result.passed is False

    def test_run_with_tuple_inputs(self) -> None:
        """run_with_inputs handles tuple inputs (multiple args)."""
        executor = CodeExecutor()
        code = "def solution(a, b): return a + b"
        result = executor.run_with_inputs(
            code,
            inputs=[(1, 2), (3, 4)],
            expected_outputs=[3, 7],
        )
        assert result.passed is True
        assert result.num_tests_passed == 2

    def test_run_with_custom_function_name(self) -> None:
        """run_with_inputs uses custom function name."""
        executor = CodeExecutor()
        code = "def my_func(x): return x ** 2"
        result = executor.run_with_inputs(
            code,
            inputs=[3],
            expected_outputs=[9],
            function_name="my_func",
        )
        assert result.passed is True


@pytest.mark.unit
class TestCreateExecutorFromConfig:
    """Tests for create_executor_from_config factory."""

    def test_from_config_with_attributes(self) -> None:
        """Config attributes are used for executor."""
        config = MagicMock()
        config.execution_timeout_seconds = 10.0
        config.max_memory_mb = 512
        config.allow_network = True

        executor = create_executor_from_config(config)
        assert executor.timeout_seconds == 10.0
        assert executor.max_memory_mb == 512
        assert executor.allow_network is True

    def test_from_config_defaults(self) -> None:
        """Missing config attributes fall back to defaults."""
        config = object()  # no attributes
        executor = create_executor_from_config(config)
        assert executor.timeout_seconds == 5.0
        assert executor.max_memory_mb == 256
        assert executor.allow_network is False


@pytest.mark.unit
class TestSafeBuiltinsAndAllowedImports:
    """Tests for SAFE_BUILTINS and ALLOWED_IMPORTS constants."""

    def test_safe_builtins_has_common_builtins(self) -> None:
        """SAFE_BUILTINS contains essential built-in functions."""
        for name in ["abs", "len", "int", "str", "list", "dict", "range", "print"]:
            assert name in SAFE_BUILTINS

    def test_safe_builtins_has_exceptions(self) -> None:
        """SAFE_BUILTINS contains common exception types."""
        for name in ["ValueError", "TypeError", "KeyError", "RuntimeError"]:
            assert name in SAFE_BUILTINS

    def test_allowed_imports_has_safe_modules(self) -> None:
        """ALLOWED_IMPORTS contains expected safe modules."""
        for mod in ["math", "re", "collections", "itertools", "json"]:
            assert mod in ALLOWED_IMPORTS

    def test_allowed_imports_excludes_dangerous(self) -> None:
        """ALLOWED_IMPORTS does not include dangerous modules."""
        for mod in ["os", "sys", "subprocess", "shutil", "socket"]:
            assert mod not in ALLOWED_IMPORTS
