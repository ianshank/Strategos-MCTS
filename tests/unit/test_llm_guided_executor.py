"""
Unit tests for LLM-Guided MCTS Code Executor.

Tests code validation, sandboxed execution, and test case handling.
"""

from __future__ import annotations

import pytest

# Skip all tests if dependencies not available
pytest.importorskip("src.framework.mcts.llm_guided.executor")

from src.framework.mcts.llm_guided.executor import (
    CodeExecutionResult,
    CodeExecutor,
)


class TestCodeExecutionResult:
    """Tests for CodeExecutionResult dataclass."""

    def test_create_result(self) -> None:
        """Result can be created with required fields."""
        result = CodeExecutionResult(
            passed=True,
            num_tests_passed=3,
            num_tests_total=3,
            stdout="output",
            stderr="",
            errors=[],
            execution_time_ms=50.0,
        )
        assert result.passed is True
        assert result.num_tests_passed == 3
        assert result.execution_time_ms == 50.0

    def test_result_with_errors(self) -> None:
        """Result can contain errors."""
        result = CodeExecutionResult(
            passed=False,
            num_tests_passed=0,
            num_tests_total=1,
            stdout="",
            stderr="Error occurred",
            errors=["AssertionError: Expected 1, got 2"],
            execution_time_ms=10.0,
        )
        assert result.passed is False
        assert len(result.errors) == 1

    def test_result_flags(self) -> None:
        """Result flags are set correctly."""
        result = CodeExecutionResult(
            passed=False,
            num_tests_passed=0,
            num_tests_total=0,
            stdout="",
            stderr="",
            errors=[],
            execution_time_ms=0.0,
            syntax_error=True,
        )
        assert result.syntax_error is True
        assert result.timed_out is False

    def test_to_dict(self) -> None:
        """Result can be serialized to dict."""
        result = CodeExecutionResult(
            passed=True,
            num_tests_passed=2,
            num_tests_total=2,
            stdout="out",
            stderr="err",
            errors=[],
            execution_time_ms=100.0,
            timed_out=False,
            syntax_error=False,
        )
        d = result.to_dict()

        assert d["passed"] is True
        assert d["num_tests_passed"] == 2
        assert d["stdout"] == "out"
        assert "timed_out" in d
        assert "syntax_error" in d


class TestCodeValidation:
    """Tests for code validation."""

    def test_valid_code(self) -> None:
        """Valid Python code passes validation."""
        executor = CodeExecutor()
        code = """
def add(a, b):
    return a + b

result = add(1, 2)
"""
        is_valid, errors = executor.validate_code(code)
        assert is_valid is True
        assert len(errors) == 0

    def test_syntax_error_detection(self) -> None:
        """Syntax errors are detected."""
        executor = CodeExecutor()
        code = "def foo(\n  invalid syntax"

        is_valid, errors = executor.validate_code(code)

        assert is_valid is False
        assert any("syntax" in e.lower() for e in errors)

    def test_disallowed_import_os(self) -> None:
        """Import of 'os' module is blocked."""
        executor = CodeExecutor()
        code = "import os\nprint(os.getcwd())"

        is_valid, errors = executor.validate_code(code)

        assert is_valid is False
        assert any("os" in e for e in errors)

    def test_disallowed_import_subprocess(self) -> None:
        """Import of 'subprocess' is blocked."""
        executor = CodeExecutor()
        code = "import subprocess\nsubprocess.run(['ls'])"

        is_valid, errors = executor.validate_code(code)

        assert is_valid is False
        assert any("subprocess" in e for e in errors)

    def test_allowed_import_math(self) -> None:
        """Import of 'math' module is allowed."""
        executor = CodeExecutor()
        code = "import math\nprint(math.sqrt(4))"

        is_valid, errors = executor.validate_code(code)

        assert is_valid is True

    def test_allowed_import_collections(self) -> None:
        """Import of 'collections' is allowed."""
        executor = CodeExecutor()
        code = "from collections import Counter\nc = Counter([1,1,2])"

        is_valid, errors = executor.validate_code(code)

        assert is_valid is True

    @pytest.mark.parametrize(
        "dangerous_code",
        [
            "os.system('ls')",
            "eval(user_input)",
            "exec(code_string)",
            "__import__('os')",
        ],
    )
    def test_dangerous_patterns_detected(self, dangerous_code: str) -> None:
        """Dangerous patterns are caught."""
        executor = CodeExecutor()

        is_valid, errors = executor.validate_code(dangerous_code)

        assert is_valid is False
        assert len(errors) > 0


class TestCodeExecution:
    """Tests for code execution."""

    def test_simple_code_execution(self) -> None:
        """Simple code executes without errors."""
        executor = CodeExecutor()
        code = "x = 1 + 1"

        result = executor.execute(code)

        # With no test cases, passed=False but errors should be empty
        assert result.errors == []
        assert result.syntax_error is False

    def test_code_with_output(self) -> None:
        """Code output is captured."""
        executor = CodeExecutor()
        code = "print('Hello, World!')"

        result = executor.execute(code)

        assert result.errors == []
        assert "Hello" in result.stdout

    def test_test_case_execution_pass(self) -> None:
        """Passing test cases are executed."""
        executor = CodeExecutor()
        code = "def add(a, b): return a + b"
        tests = [
            "assert add(1, 1) == 2",
            "assert add(0, 0) == 0",
            "assert add(-1, 1) == 0",
        ]

        result = executor.execute(code, tests)

        assert result.passed is True
        assert result.num_tests_passed == 3
        assert result.num_tests_total == 3

    def test_test_case_execution_fail(self) -> None:
        """Failing test cases are detected."""
        executor = CodeExecutor()
        code = "def add(a, b): return a * b"  # Wrong!
        tests = ["assert add(2, 3) == 5"]

        result = executor.execute(code, tests)

        assert result.passed is False
        assert result.num_tests_passed == 0
        assert result.num_tests_total == 1

    def test_partial_test_pass(self) -> None:
        """Partial test passes are counted."""
        executor = CodeExecutor()
        code = """
def is_even(n):
    return n % 2 == 0
"""
        tests = [
            "assert is_even(2) == True",
            "assert is_even(3) == False",
            "assert is_even(0) == True",
            "assert is_even(1) == True",  # This will fail
        ]

        result = executor.execute(code, tests)

        assert result.passed is False
        assert result.num_tests_passed == 3
        assert result.num_tests_total == 4

    def test_syntax_error_execution(self) -> None:
        """Syntax errors prevent execution."""
        executor = CodeExecutor()
        code = "def foo(\n  bad"

        result = executor.execute(code)

        assert result.syntax_error is True
        assert result.passed is False

    def test_runtime_error_caught(self) -> None:
        """Runtime errors are caught."""
        executor = CodeExecutor()
        code = "x = 1 / 0"

        result = executor.execute(code)

        assert result.passed is False
        assert any("ZeroDivision" in e for e in result.errors)

    def test_name_error_caught(self) -> None:
        """Name errors are caught."""
        executor = CodeExecutor()
        code = "print(undefined_variable)"

        result = executor.execute(code)

        assert result.passed is False
        assert any("NameError" in e for e in result.errors)

    def test_execution_time_measured(self) -> None:
        """Execution time is measured."""
        executor = CodeExecutor()
        # Use a simple loop instead of time.sleep (time not in ALLOWED_IMPORTS)
        code = """
total = 0
for i in range(100000):
    total += i
"""

        result = executor.execute(code)

        # Execution time should be measured and positive
        assert result.execution_time_ms > 0


class TestTimeoutHandling:
    """Tests for timeout handling."""

    def test_timeout_detection(self) -> None:
        """Code that exceeds timeout is detected."""
        executor = CodeExecutor(timeout_seconds=0.1)
        # Use a busy loop instead of time.sleep (time not in ALLOWED_IMPORTS)
        code = """
# Busy loop that will exceed timeout
i = 0
while i < 10**9:
    i += 1
"""

        result = executor.execute(code)

        assert result.timed_out is True
        assert result.passed is False
        assert any("timed out" in e for e in result.errors)

    def test_infinite_loop_timeout(self) -> None:
        """Infinite loops are stopped by timeout."""
        executor = CodeExecutor(timeout_seconds=0.1)
        code = "while True: pass"

        result = executor.execute(code)

        assert result.timed_out is True
        assert result.passed is False


class TestExecutorConfiguration:
    """Tests for executor configuration."""

    def test_default_timeout(self) -> None:
        """Default timeout is set."""
        executor = CodeExecutor()
        assert executor.timeout_seconds == 5.0

    def test_custom_timeout(self) -> None:
        """Custom timeout can be set."""
        executor = CodeExecutor(timeout_seconds=10.0)
        assert executor.timeout_seconds == 10.0

    def test_default_memory_limit(self) -> None:
        """Default memory limit is set."""
        executor = CodeExecutor()
        assert executor.max_memory_mb == 256

    def test_custom_memory_limit(self) -> None:
        """Custom memory limit can be set."""
        executor = CodeExecutor(max_memory_mb=512)
        assert executor.max_memory_mb == 512

    def test_use_subprocess_config(self) -> None:
        """use_subprocess can be configured."""
        executor = CodeExecutor(use_subprocess=True)
        assert executor.use_subprocess is True


class TestComplexCodeExecution:
    """Tests for more complex code scenarios."""

    def test_recursive_function(self) -> None:
        """Recursive functions execute correctly."""
        executor = CodeExecutor()
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
        tests = [
            "assert factorial(0) == 1",
            "assert factorial(1) == 1",
            "assert factorial(5) == 120",
        ]

        result = executor.execute(code, tests)

        assert result.passed is True
        assert result.num_tests_passed == 3

    def test_class_definition_not_supported(self) -> None:
        """Class definitions are not supported in the sandbox for security."""
        executor = CodeExecutor()
        # Class definitions require __build_class__ which is not in safe builtins
        code = """
class MyCounter:
    def __init__(self):
        self.count = 0
"""

        result = executor.execute(code)

        # Sandbox blocks class definitions for security reasons
        assert result.passed is False
        assert any("__build_class__" in e or "NameError" in e for e in result.errors)

    def test_list_comprehension(self) -> None:
        """List comprehensions work."""
        executor = CodeExecutor()
        code = "def double_evens(nums): return [x*2 for x in nums if x % 2 == 0]"
        tests = [
            "assert double_evens([1,2,3,4]) == [4, 8]",
            "assert double_evens([]) == []",
            "assert double_evens([1,3,5]) == []",
        ]

        result = executor.execute(code, tests)

        assert result.passed is True

    def test_generator_function(self) -> None:
        """Generator functions work."""
        executor = CodeExecutor()
        code = """
def count_up(n):
    for i in range(n):
        yield i
"""
        tests = [
            "assert list(count_up(3)) == [0, 1, 2]",
            "assert list(count_up(0)) == []",
        ]

        result = executor.execute(code, tests)

        assert result.passed is True


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_code(self) -> None:
        """Empty code runs without errors."""
        executor = CodeExecutor()

        result = executor.execute("")

        # Empty code has no errors but no tests to pass
        assert result.errors == []
        assert result.syntax_error is False

    def test_whitespace_only_code(self) -> None:
        """Whitespace-only code runs without errors."""
        executor = CodeExecutor()

        result = executor.execute("   \n\t\n   ")

        assert result.errors == []
        assert result.syntax_error is False

    def test_comment_only_code(self) -> None:
        """Comment-only code runs without errors."""
        executor = CodeExecutor()
        code = """
# This is a comment
# Another comment
"""

        result = executor.execute(code)

        assert result.errors == []
        assert result.syntax_error is False

    def test_unicode_in_code(self) -> None:
        """Unicode characters are handled."""
        executor = CodeExecutor()
        code = """
def greet(name):
    return f"Hello, {name}! 👋"
"""
        tests = [
            'assert greet("World") == "Hello, World! 👋"',
        ]

        result = executor.execute(code, tests)

        assert result.passed is True

    def test_multiline_string(self) -> None:
        """Multiline strings are handled."""
        executor = CodeExecutor()
        code = '''
text = """
Line 1
Line 2
Line 3
"""
'''
        tests = ["assert 'Line 2' in text"]

        result = executor.execute(code, tests)

        assert result.passed is True


class TestRunWithInputs:
    """Tests for run_with_inputs convenience method."""

    def test_run_with_single_inputs(self) -> None:
        """run_with_inputs handles single-argument functions."""
        executor = CodeExecutor()
        code = "def solution(x): return x * 2"

        result = executor.run_with_inputs(
            code,
            inputs=[1, 2, 3],
            expected_outputs=[2, 4, 6],
        )

        assert result.passed is True
        assert result.num_tests_passed == 3

    def test_run_with_tuple_inputs(self) -> None:
        """run_with_inputs handles multi-argument functions."""
        executor = CodeExecutor()
        code = "def solution(a, b): return a + b"

        result = executor.run_with_inputs(
            code,
            inputs=[(1, 2), (3, 4), (0, 0)],
            expected_outputs=[3, 7, 0],
        )

        assert result.passed is True
        assert result.num_tests_passed == 3

    def test_run_with_custom_function_name(self) -> None:
        """run_with_inputs uses custom function names."""
        executor = CodeExecutor()
        code = "def my_func(x): return x ** 2"

        result = executor.run_with_inputs(
            code,
            inputs=[2, 3, 4],
            expected_outputs=[4, 9, 16],
            function_name="my_func",
        )

        assert result.passed is True


class TestResultRepr:
    """Tests for result string representation."""

    def test_passed_result_repr(self) -> None:
        """Passed result has correct repr."""
        result = CodeExecutionResult(
            passed=True,
            num_tests_passed=3,
            num_tests_total=3,
            stdout="",
            stderr="",
            errors=[],
            execution_time_ms=10.0,
        )

        repr_str = repr(result)

        assert "PASSED" in repr_str
        assert "3/3" in repr_str

    def test_failed_result_repr(self) -> None:
        """Failed result has correct repr."""
        result = CodeExecutionResult(
            passed=False,
            num_tests_passed=1,
            num_tests_total=3,
            stdout="",
            stderr="",
            errors=["Test failed"],
            execution_time_ms=10.0,
        )

        repr_str = repr(result)

        assert "FAILED" in repr_str
        assert "1/3" in repr_str
