"""Tests for Code Executor."""

import pytest

pytest.importorskip("numpy", reason="numpy required for MCTS framework")

from src.framework.mcts.llm_guided.executor import (
    ALLOWED_IMPORTS,
    CodeExecutionResult,
    CodeExecutor,
    ImportChecker,
    create_executor_from_config,
)


class TestCodeExecutionResult:
    """Tests for CodeExecutionResult."""

    def test_creation(self):
        """Test basic creation."""
        result = CodeExecutionResult(
            passed=True,
            num_tests_passed=3,
            num_tests_total=3,
            stdout="",
            stderr="",
            errors=[],
            execution_time_ms=100.0,
        )

        assert result.passed is True
        assert result.num_tests_passed == 3
        assert result.num_tests_total == 3
        assert result.timed_out is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = CodeExecutionResult(
            passed=False,
            num_tests_passed=1,
            num_tests_total=3,
            stdout="output",
            stderr="error",
            errors=["AssertionError"],
            execution_time_ms=50.0,
            timed_out=True,
        )

        d = result.to_dict()
        assert d["passed"] is False
        assert d["num_tests_passed"] == 1
        assert d["stdout"] == "output"
        assert d["timed_out"] is True

    def test_repr(self):
        """Test string representation."""
        result = CodeExecutionResult(
            passed=True,
            num_tests_passed=5,
            num_tests_total=5,
            stdout="",
            stderr="",
            errors=[],
            execution_time_ms=10.0,
        )

        repr_str = repr(result)
        assert "PASSED" in repr_str
        assert "5/5" in repr_str


class TestImportChecker:
    """Tests for ImportChecker AST visitor."""

    def test_allowed_imports(self):
        """Test that allowed imports pass."""
        import ast

        code = """
import math
import re
from collections import defaultdict
from typing import List, Dict
"""
        tree = ast.parse(code)
        checker = ImportChecker()
        checker.visit(tree)

        assert checker.disallowed_imports == []

    def test_disallowed_imports(self):
        """Test that disallowed imports are detected."""
        import ast

        code = """
import os
import subprocess
from socket import socket
"""
        tree = ast.parse(code)
        checker = ImportChecker()
        checker.visit(tree)

        assert "os" in checker.disallowed_imports
        assert "subprocess" in checker.disallowed_imports
        assert "socket" in checker.disallowed_imports


class TestCodeExecutor:
    """Tests for CodeExecutor."""

    def test_initialization(self):
        """Test executor initialization."""
        executor = CodeExecutor(
            timeout_seconds=10.0,
            max_memory_mb=512,
            allow_network=False,
        )

        assert executor.timeout_seconds == 10.0
        assert executor.max_memory_mb == 512
        assert executor.allow_network is False

    def test_validate_code_valid(self):
        """Test validation of valid code."""
        executor = CodeExecutor()

        code = """
def add(a, b):
    return a + b

result = add(1, 2)
"""
        is_valid, errors = executor.validate_code(code)
        assert is_valid is True
        assert errors == []

    def test_validate_code_syntax_error(self):
        """Test validation catches syntax errors."""
        executor = CodeExecutor()

        code = """
def foo(:
    pass
"""
        is_valid, errors = executor.validate_code(code)
        assert is_valid is False
        assert any("Syntax error" in e for e in errors)

    def test_validate_code_disallowed_imports(self):
        """Test validation catches disallowed imports."""
        executor = CodeExecutor()

        code = """
import os
os.system("rm -rf /")
"""
        is_valid, errors = executor.validate_code(code)
        assert is_valid is False
        assert any("Disallowed imports" in e for e in errors)

    def test_validate_code_dangerous_patterns(self):
        """Test validation catches dangerous patterns."""
        executor = CodeExecutor()

        code = """
eval("__import__('os').system('ls')")
"""
        is_valid, errors = executor.validate_code(code)
        assert is_valid is False
        assert any("dangerous" in e.lower() for e in errors)

    def test_execute_simple_code(self):
        """Test executing simple code."""
        executor = CodeExecutor()

        code = """
def add(a, b):
    return a + b
"""
        test_cases = ["assert add(1, 2) == 3"]

        result = executor.execute(code, test_cases)

        assert result.passed is True
        assert result.num_tests_passed == 1
        assert result.num_tests_total == 1
        assert result.errors == []

    def test_execute_multiple_tests(self):
        """Test executing code with multiple tests."""
        executor = CodeExecutor()

        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
        test_cases = [
            "assert factorial(0) == 1",
            "assert factorial(1) == 1",
            "assert factorial(5) == 120",
        ]

        result = executor.execute(code, test_cases)

        assert result.passed is True
        assert result.num_tests_passed == 3
        assert result.num_tests_total == 3

    def test_execute_failing_test(self):
        """Test executing code that fails tests."""
        executor = CodeExecutor()

        code = """
def add(a, b):
    return a - b  # Wrong!
"""
        test_cases = ["assert add(1, 2) == 3"]

        result = executor.execute(code, test_cases)

        assert result.passed is False
        assert result.num_tests_passed == 0
        assert result.num_tests_total == 1
        assert len(result.errors) > 0

    def test_execute_runtime_error(self):
        """Test executing code with runtime error."""
        executor = CodeExecutor()

        code = """
def divide(a, b):
    return a / b
"""
        test_cases = ["assert divide(1, 0) == 0"]

        result = executor.execute(code, test_cases)

        assert result.passed is False
        assert "ZeroDivisionError" in str(result.errors)

    def test_execute_syntax_error(self):
        """Test executing code with syntax error."""
        executor = CodeExecutor()

        code = """
def foo(:
    pass
"""
        result = executor.execute(code, [])

        assert result.passed is False
        assert result.syntax_error is True

    def test_execute_with_allowed_imports(self):
        """Test that allowed imports work."""
        executor = CodeExecutor()

        code = """
import math

def circle_area(r):
    return math.pi * r * r
"""
        test_cases = ["assert abs(circle_area(1) - 3.14159) < 0.001"]

        result = executor.execute(code, test_cases)

        assert result.passed is True

    def test_execute_captures_stdout(self):
        """Test that stdout is captured."""
        executor = CodeExecutor()

        code = """
def hello():
    print("Hello, World!")
    return True
"""
        test_cases = ["assert hello() == True"]

        result = executor.execute(code, test_cases)

        assert result.passed is True
        assert "Hello, World!" in result.stdout

    def test_run_with_inputs(self):
        """Test run_with_inputs helper."""
        executor = CodeExecutor()

        code = """
def solution(x):
    return x * 2
"""
        result = executor.run_with_inputs(
            code=code,
            inputs=[1, 2, 3],
            expected_outputs=[2, 4, 6],
            function_name="solution",
        )

        assert result.passed is True
        assert result.num_tests_passed == 3

    def test_run_with_tuple_inputs(self):
        """Test run_with_inputs with tuple inputs."""
        executor = CodeExecutor()

        code = """
def solution(a, b):
    return a + b
"""
        result = executor.run_with_inputs(
            code=code,
            inputs=[(1, 2), (3, 4)],
            expected_outputs=[3, 7],
            function_name="solution",
        )

        assert result.passed is True


class TestCreateExecutorFromConfig:
    """Tests for create_executor_from_config factory."""

    def test_create_from_config(self):
        """Test creating executor from config object."""
        from src.framework.mcts.llm_guided.config import LLMGuidedMCTSConfig

        config = LLMGuidedMCTSConfig(
            execution_timeout_seconds=15.0,
            max_memory_mb=512,
            allow_network=False,
        )

        executor = create_executor_from_config(config)

        assert executor.timeout_seconds == 15.0
        assert executor.max_memory_mb == 512
        assert executor.allow_network is False


class TestAllowedImports:
    """Tests for ALLOWED_IMPORTS set."""

    def test_essential_imports_allowed(self):
        """Test that essential imports are in the allowed set."""
        essential = ["math", "string", "re", "collections", "itertools", "typing"]
        for module in essential:
            assert module in ALLOWED_IMPORTS, f"{module} should be allowed"

    def test_dangerous_imports_not_allowed(self):
        """Test that dangerous imports are not in the allowed set."""
        dangerous = ["os", "sys", "subprocess", "socket", "shutil"]
        for module in dangerous:
            assert module not in ALLOWED_IMPORTS, f"{module} should not be allowed"
