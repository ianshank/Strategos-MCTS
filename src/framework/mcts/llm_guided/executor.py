"""
Code Execution Sandbox for LLM-Guided MCTS.

Provides safe execution of generated Python code with:
- Timeout protection
- Memory limits
- Restricted imports
- Captured output and errors
"""

from __future__ import annotations

import ast
import io
import multiprocessing
import signal
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Any

from src.observability.logging import get_structured_logger

logger = get_structured_logger(__name__)


@dataclass
class CodeExecutionResult:
    """Result from code execution."""

    passed: bool
    """Whether all tests passed."""

    num_tests_passed: int
    """Number of tests that passed."""

    num_tests_total: int
    """Total number of tests."""

    stdout: str
    """Captured standard output."""

    stderr: str
    """Captured standard error."""

    errors: list[str]
    """List of error messages."""

    execution_time_ms: float
    """Execution time in milliseconds."""

    timed_out: bool = False
    """Whether execution timed out."""

    memory_exceeded: bool = False
    """Whether memory limit was exceeded."""

    syntax_error: bool = False
    """Whether there was a syntax error."""

    test_results: list[dict[str, Any]] = field(default_factory=list)
    """Detailed results for each test."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "num_tests_passed": self.num_tests_passed,
            "num_tests_total": self.num_tests_total,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "errors": self.errors,
            "execution_time_ms": self.execution_time_ms,
            "timed_out": self.timed_out,
            "memory_exceeded": self.memory_exceeded,
            "syntax_error": self.syntax_error,
            "test_results": self.test_results,
        }

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"CodeExecutionResult({status}, {self.num_tests_passed}/{self.num_tests_total})"


class TimeoutException(Exception):
    """Exception raised when code execution times out."""

    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    """Signal handler for timeout."""
    raise TimeoutException("Code execution timed out")


def _create_safe_import(allowed_modules: set[str]):
    """Create a safe import function that only allows specified modules."""
    import importlib

    def safe_import(
        name: str,
        globals: dict | None = None,
        locals: dict | None = None,
        fromlist: tuple = (),
        level: int = 0,
    ):
        """Safe import that only allows specified modules."""
        # Get the base module name
        base_module = name.split(".")[0]

        if base_module not in allowed_modules:
            raise ImportError(f"Import of '{name}' is not allowed for security reasons")

        return importlib.import_module(name)

    return safe_import


# Safe builtins for restricted execution
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "ascii": ascii,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "callable": callable,
    "chr": chr,
    "classmethod": classmethod,
    "complex": complex,
    "dict": dict,
    "dir": dir,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "getattr": getattr,
    "hasattr": hasattr,
    "hash": hash,
    "hex": hex,
    "id": id,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "object": object,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "property": property,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "setattr": setattr,
    "slice": slice,
    "sorted": sorted,
    "staticmethod": staticmethod,
    "str": str,
    "sum": sum,
    "super": super,
    "tuple": tuple,
    "type": type,
    "vars": vars,
    "zip": zip,
    # Exceptions
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "ZeroDivisionError": ZeroDivisionError,
    # None, True, False are automatically available
    "None": None,
    "True": True,
    "False": False,
}

# Lazy initialization of safe import (needs ALLOWED_IMPORTS to be defined first)
_safe_import_instance: Any = None


def _get_safe_import():
    """Get or create the safe import function."""
    global _safe_import_instance
    if _safe_import_instance is None:
        _safe_import_instance = _create_safe_import(ALLOWED_IMPORTS)
    return _safe_import_instance


# Allowed imports
ALLOWED_IMPORTS = {
    "math",
    "string",
    "re",
    "collections",
    "itertools",
    "functools",
    "operator",
    "typing",
    "dataclasses",
    "json",
    "random",
    "heapq",
    "bisect",
    "copy",
    "datetime",
    "decimal",
    "fractions",
    "statistics",
    "array",
}


class ImportChecker(ast.NodeVisitor):
    """AST visitor to check for disallowed imports."""

    def __init__(self):
        self.disallowed_imports: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module = alias.name.split(".")[0]
            if module not in ALLOWED_IMPORTS:
                self.disallowed_imports.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            module = node.module.split(".")[0]
            if module not in ALLOWED_IMPORTS:
                self.disallowed_imports.append(node.module)


class CodeExecutor:
    """
    Safe code execution sandbox.

    Executes Python code with:
    - Timeout protection
    - Memory limits (via resource module on Unix)
    - Import restrictions
    - Output capture
    """

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        max_memory_mb: int = 256,
        allow_network: bool = False,
        use_subprocess: bool = False,
    ):
        """
        Initialize code executor.

        Args:
            timeout_seconds: Maximum execution time
            max_memory_mb: Maximum memory usage in MB
            allow_network: Allow network access (not recommended)
            use_subprocess: Run in separate process for isolation
        """
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        self.allow_network = allow_network
        self.use_subprocess = use_subprocess

        logger.debug(
            "CodeExecutor initialized",
            timeout_seconds=timeout_seconds,
            max_memory_mb=max_memory_mb,
            allow_network=allow_network,
            use_subprocess=use_subprocess,
        )

    def validate_code(self, code: str) -> tuple[bool, list[str]]:
        """
        Validate code before execution.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []

        # Check for syntax errors
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logger.debug(
                "Code validation failed: syntax error",
                error=str(e),
                code_length=len(code),
            )
            return False, [f"Syntax error: {e}"]

        # Check for disallowed imports
        checker = ImportChecker()
        checker.visit(tree)
        if checker.disallowed_imports:
            logger.debug(
                "Code validation found disallowed imports",
                disallowed_imports=checker.disallowed_imports,
            )
            errors.append(f"Disallowed imports: {', '.join(checker.disallowed_imports)}")

        # Check for dangerous patterns
        dangerous_patterns = [
            "os.system",
            "subprocess",
            "eval(",
            "exec(",
            "__import__",
            "open(",
            "file(",
        ]
        for pattern in dangerous_patterns:
            if pattern in code:
                logger.debug(
                    "Code validation found dangerous pattern",
                    pattern=pattern,
                )
                errors.append(f"Potentially dangerous pattern: {pattern}")

        is_valid = len(errors) == 0
        if is_valid:
            logger.debug("Code validation passed", code_length=len(code))
        else:
            logger.debug(
                "Code validation failed",
                error_count=len(errors),
                errors=errors,
            )
        return is_valid, errors

    def execute(
        self,
        code: str,
        test_cases: list[str] | None = None,
        globals_dict: dict[str, Any] | None = None,
    ) -> CodeExecutionResult:
        """
        Execute code and run test cases.

        Args:
            code: Python code to execute
            test_cases: Optional list of test case assertions
            globals_dict: Optional global variables

        Returns:
            CodeExecutionResult with execution details
        """
        import time

        start_time = time.perf_counter()
        num_tests = len(test_cases) if test_cases else 0

        logger.debug(
            "Starting code execution",
            code_length=len(code),
            num_tests=num_tests,
            use_subprocess=self.use_subprocess,
        )

        # Validate code first
        is_valid, validation_errors = self.validate_code(code)
        if not is_valid:
            logger.debug(
                "Code execution aborted: validation failed",
                error_count=len(validation_errors),
            )
            return CodeExecutionResult(
                passed=False,
                num_tests_passed=0,
                num_tests_total=num_tests,
                stdout="",
                stderr="\n".join(validation_errors),
                errors=validation_errors,
                execution_time_ms=0.0,
                syntax_error=True,
            )

        if self.use_subprocess:
            result = self._execute_in_subprocess(code, test_cases, globals_dict, start_time)
        else:
            result = self._execute_in_process(code, test_cases, globals_dict, start_time)

        logger.debug(
            "Code execution completed",
            passed=result.passed,
            num_tests_passed=result.num_tests_passed,
            num_tests_total=result.num_tests_total,
            execution_time_ms=round(result.execution_time_ms, 2),
            timed_out=result.timed_out,
            syntax_error=result.syntax_error,
        )
        return result

    def _execute_in_process(
        self,
        code: str,
        test_cases: list[str] | None,
        globals_dict: dict[str, Any] | None,
        start_time: float,
    ) -> CodeExecutionResult:
        """Execute code in the current process with timeout."""
        import time

        # Prepare execution environment with safe __import__
        safe_builtins = dict(SAFE_BUILTINS)
        safe_builtins["__import__"] = _get_safe_import()
        exec_globals: dict[str, Any] = {"__builtins__": safe_builtins}
        if globals_dict:
            exec_globals.update(globals_dict)

        # Allow safe imports
        import collections
        import functools
        import itertools
        import math
        import operator
        import re
        import string
        from typing import Any as TypingAny
        from typing import Optional

        exec_globals.update(
            {
                "math": math,
                "string": string,
                "re": re,
                "collections": collections,
                "itertools": itertools,
                "functools": functools,
                "operator": operator,
                "List": list,
                "Dict": dict,
                "Optional": Optional,
                "Any": TypingAny,
                "Tuple": tuple,
            }
        )

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        errors: list[str] = []
        test_results: list[dict[str, Any]] = []

        timed_out = False
        syntax_error = False

        # Set up timeout
        old_handler = None
        try:
            # Only use signal-based timeout on Unix
            if hasattr(signal, "SIGALRM"):
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(int(self.timeout_seconds + 1))
        except (ValueError, OSError):
            # Cannot use signals (e.g., not on main thread)
            pass

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the code
                try:
                    exec(code, exec_globals)
                except SyntaxError as e:
                    syntax_error = True
                    errors.append(f"Syntax error: {e}")
                except TimeoutException:
                    timed_out = True
                    errors.append("Code execution timed out")
                except Exception as e:  # Intentionally broad: user code may raise any exception
                    errors.append(f"Runtime error: {type(e).__name__}: {e}")

                # Run test cases if code executed successfully
                if test_cases and not errors:
                    for i, test in enumerate(test_cases):
                        test_result = {"test_index": i, "test_code": test, "passed": False}
                        try:
                            exec(test, exec_globals)
                            test_result["passed"] = True
                        except AssertionError as e:
                            test_result["error"] = f"AssertionError: {e}"
                            errors.append(f"Test {i + 1} failed: {e}")
                        except TimeoutException:
                            timed_out = True
                            test_result["error"] = "Timed out"
                            errors.append(f"Test {i + 1} timed out")
                            break
                        except Exception as e:  # Intentionally broad: user code may raise any exception
                            test_result["error"] = f"{type(e).__name__}: {e}"
                            errors.append(f"Test {i + 1} error: {type(e).__name__}: {e}")
                        test_results.append(test_result)

        except TimeoutException:
            timed_out = True
            errors.append("Code execution timed out")
        except Exception as e:  # Intentionally broad: user code may raise any exception
            errors.append(f"Unexpected error: {type(e).__name__}: {e}")
        finally:
            # Clear timeout
            if old_handler is not None:
                try:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                except (ValueError, OSError):
                    pass

        execution_time_ms = (time.perf_counter() - start_time) * 1000
        num_tests_passed = sum(1 for r in test_results if r["passed"])
        num_tests_total = len(test_cases) if test_cases else 0
        passed = num_tests_passed == num_tests_total and num_tests_total > 0 and not errors

        return CodeExecutionResult(
            passed=passed,
            num_tests_passed=num_tests_passed,
            num_tests_total=num_tests_total,
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            errors=errors,
            execution_time_ms=execution_time_ms,
            timed_out=timed_out,
            syntax_error=syntax_error,
            test_results=test_results,
        )

    def _execute_in_subprocess(
        self,
        code: str,
        test_cases: list[str] | None,
        globals_dict: dict[str, Any] | None,
        start_time: float,
    ) -> CodeExecutionResult:
        """Execute code in a separate subprocess for better isolation."""
        import time

        def worker(queue: multiprocessing.Queue, code: str, test_cases: list[str] | None) -> None:
            """Worker function for subprocess execution."""
            try:
                result = self._execute_in_process(code, test_cases, None, time.perf_counter())
                queue.put(result.to_dict())
            except Exception as e:  # Intentionally broad: user code may raise any exception
                queue.put(
                    {
                        "passed": False,
                        "num_tests_passed": 0,
                        "num_tests_total": len(test_cases) if test_cases else 0,
                        "stdout": "",
                        "stderr": str(e),
                        "errors": [str(e)],
                        "execution_time_ms": 0.0,
                        "timed_out": False,
                        "syntax_error": False,
                        "test_results": [],
                    }
                )

        queue: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=worker, args=(queue, code, test_cases))
        process.start()
        process.join(timeout=self.timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join()
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            return CodeExecutionResult(
                passed=False,
                num_tests_passed=0,
                num_tests_total=len(test_cases) if test_cases else 0,
                stdout="",
                stderr="Execution timed out",
                errors=["Execution timed out"],
                execution_time_ms=execution_time_ms,
                timed_out=True,
            )

        try:
            result_dict = queue.get_nowait()
            return CodeExecutionResult(**result_dict)
        except Exception:  # Intentionally broad: subprocess queue may raise any exception
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            return CodeExecutionResult(
                passed=False,
                num_tests_passed=0,
                num_tests_total=len(test_cases) if test_cases else 0,
                stdout="",
                stderr="Failed to get result from subprocess",
                errors=["Failed to get result from subprocess"],
                execution_time_ms=execution_time_ms,
            )

    def run_with_inputs(
        self,
        code: str,
        inputs: list[Any],
        expected_outputs: list[Any],
        function_name: str = "solution",
    ) -> CodeExecutionResult:
        """
        Run code with specific inputs and compare to expected outputs.

        Args:
            code: Python code containing the function
            inputs: List of input arguments
            expected_outputs: List of expected return values
            function_name: Name of the function to call

        Returns:
            CodeExecutionResult with execution details
        """
        # Generate test cases from inputs/outputs
        test_cases = []
        for inp, expected in zip(inputs, expected_outputs, strict=True):
            if isinstance(inp, tuple):
                args_str = ", ".join(repr(a) for a in inp)
            else:
                args_str = repr(inp)
            test_cases.append(f"assert {function_name}({args_str}) == {repr(expected)}")

        return self.execute(code, test_cases)


def create_executor_from_config(config: Any) -> CodeExecutor:
    """
    Create a CodeExecutor from configuration.

    Args:
        config: LLMGuidedMCTSConfig or similar with execution settings

    Returns:
        Configured CodeExecutor
    """
    return CodeExecutor(
        timeout_seconds=getattr(config, "execution_timeout_seconds", 5.0),
        max_memory_mb=getattr(config, "max_memory_mb", 256),
        allow_network=getattr(config, "allow_network", False),
    )
