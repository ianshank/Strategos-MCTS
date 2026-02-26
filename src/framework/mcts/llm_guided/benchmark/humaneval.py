"""
HumanEval Benchmark Loader.

Provides utilities for loading and parsing HumanEval problems.
HumanEval is a benchmark of 164 hand-written Python programming problems.

Reference: https://github.com/openai/human-eval
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.observability.logging import get_structured_logger

logger = get_structured_logger(__name__)


@dataclass
class HumanEvalProblem:
    """A single HumanEval problem."""

    task_id: str
    """Unique task identifier (e.g., 'HumanEval/0')."""

    prompt: str
    """Function signature and docstring."""

    canonical_solution: str
    """Reference solution."""

    entry_point: str
    """Function name to call."""

    test: str
    """Test code that checks the solution."""

    # Derived fields
    function_name: str = ""
    """Extracted function name."""

    docstring: str = ""
    """Extracted docstring."""

    test_cases: list[str] = field(default_factory=list)
    """Individual test assertions."""

    difficulty: str = "unknown"
    """Estimated difficulty: easy, medium, hard."""

    tags: list[str] = field(default_factory=list)
    """Problem tags/categories."""

    def __post_init__(self):
        """Extract derived fields."""
        self.function_name = self.entry_point
        self._extract_docstring()
        self._extract_test_cases()
        self._estimate_difficulty()

    def _extract_docstring(self) -> None:
        """Extract docstring from prompt."""
        # Look for triple-quoted docstring
        patterns = [
            r'"""(.*?)"""',
            r"'''(.*?)'''",
        ]
        for pattern in patterns:
            match = re.search(pattern, self.prompt, re.DOTALL)
            if match:
                self.docstring = match.group(1).strip()
                break

    def _extract_test_cases(self) -> None:
        """Extract individual test assertions from test code."""
        # Find assert statements
        assertions = re.findall(r"assert\s+.+", self.test)
        self.test_cases = assertions

    def _estimate_difficulty(self) -> None:
        """Estimate problem difficulty based on heuristics."""
        # Simple heuristics based on solution length and complexity
        sol_lines = len(self.canonical_solution.strip().split("\n"))
        num_tests = len(self.test_cases)

        if sol_lines <= 5 and num_tests <= 3:
            self.difficulty = "easy"
        elif sol_lines <= 15 or num_tests <= 6:
            self.difficulty = "medium"
        else:
            self.difficulty = "hard"

    def get_problem_description(self) -> str:
        """Get formatted problem description for LLM."""
        return f"""## Problem: {self.task_id}

{self.prompt}

### Requirements:
- Implement the function `{self.entry_point}`
- The function should pass all test cases
- Follow Python best practices

### Test Examples:
{chr(10).join(self.test_cases[:3]) if self.test_cases else "See test code for examples"}
"""

    def get_test_harness(self) -> str:
        """Get test harness code for execution."""
        return f"""
{self.test}

check({self.entry_point})
"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "canonical_solution": self.canonical_solution,
            "entry_point": self.entry_point,
            "test": self.test,
            "function_name": self.function_name,
            "docstring": self.docstring,
            "test_cases": self.test_cases,
            "difficulty": self.difficulty,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumanEvalProblem:
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            prompt=data["prompt"],
            canonical_solution=data["canonical_solution"],
            entry_point=data["entry_point"],
            test=data["test"],
        )


class HumanEvalBenchmark:
    """
    HumanEval Benchmark manager.

    Loads and provides access to HumanEval problems.
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        problems: list[HumanEvalProblem] | None = None,
    ):
        """
        Initialize benchmark.

        Args:
            data_path: Path to HumanEval JSONL file
            problems: Pre-loaded problems (for testing)
        """
        self._data_path = Path(data_path) if data_path else None
        self._problems: dict[str, HumanEvalProblem] = {}

        if problems is not None:
            for p in problems:
                self._problems[p.task_id] = p
        elif self._data_path is not None:
            self._load_problems()
        else:
            # Try to load from default location or create sample
            self._create_sample_problems()

        logger.info(
            "Initialized HumanEvalBenchmark",
            num_problems=len(self._problems),
            data_path=str(self._data_path) if self._data_path else "in-memory",
        )

    def _load_problems(self) -> None:
        """Load problems from JSONL file."""
        assert self._data_path is not None, "Data path must be set to load problems"

        if not self._data_path.exists():
            logger.warning(
                "HumanEval data not found, creating sample problems",
                data_path=str(self._data_path),
            )
            self._create_sample_problems()
            return

        with open(self._data_path) as f:
            for line in f:
                data = json.loads(line)
                problem = HumanEvalProblem.from_dict(data)
                self._problems[problem.task_id] = problem

    def _create_sample_problems(self) -> None:
        """Create sample problems for testing."""
        # Sample problems that demonstrate different complexity levels
        sample_problems = [
            HumanEvalProblem(
                task_id="HumanEval/0",
                prompt='''def has_close_elements(numbers: list[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
''',
                canonical_solution="""    for i, num1 in enumerate(numbers):
        for j, num2 in enumerate(numbers):
            if i != j and abs(num1 - num2) < threshold:
                return True
    return False
""",
                entry_point="has_close_elements",
                test="""
def check(candidate):
    assert candidate([1.0, 2.0, 3.0], 0.5) == False
    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
""",
            ),
            HumanEvalProblem(
                task_id="HumanEval/1",
                prompt='''def separate_paren_groups(paren_string: str) -> list[str]:
    """Input to this function is a string containing multiple groups of nested parentheses.
    Your goal is to separate those groups into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other.
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
''',
                canonical_solution="""    result = []
    current_string = []
    current_depth = 0
    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)
            if current_depth == 0:
                result.append(''.join(current_string))
                current_string = []
    return result
""",
                entry_point="separate_paren_groups",
                test="""
def check(candidate):
    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']
    assert candidate('(()(()))') == ['(()(()))']
    assert candidate('( ) ( )') == ['()', '()']
""",
            ),
            HumanEvalProblem(
                task_id="HumanEval/2",
                prompt='''def truncate_number(number: float) -> float:
    """Given a positive floating point number, it can be decomposed into an integer part
    (largest integer smaller than given number) and decimals (leftover part always smaller than 1).
    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """
''',
                canonical_solution="""    return number % 1.0
""",
                entry_point="truncate_number",
                test="""
def check(candidate):
    assert candidate(3.5) == 0.5
    assert abs(candidate(1.33) - 0.33) < 1e-6
    assert abs(candidate(123.456) - 0.456) < 1e-6
""",
            ),
            HumanEvalProblem(
                task_id="HumanEval/3",
                prompt='''def below_zero(operations: list[int]) -> bool:
    """You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance falls below zero, and
    at that point the function should return True. Otherwise it should return False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """
''',
                canonical_solution="""    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False
""",
                entry_point="below_zero",
                test="""
def check(candidate):
    assert candidate([]) == False
    assert candidate([1, 2, -3, 1, 2, -3]) == False
    assert candidate([1, 2, -4, 5, 6]) == True
    assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False
    assert candidate([1, -1, 2, -2, 5, -5, 4, -5]) == True
""",
            ),
            HumanEvalProblem(
                task_id="HumanEval/4",
                prompt='''def mean_absolute_deviation(numbers: list[float]) -> float:
    """For a given list of input numbers, calculate Mean Absolute Deviation around the mean.
    Mean Absolute Deviation is the average absolute difference between each element and the mean.
    MAD = average |x - mean|
    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    1.0
    """
''',
                canonical_solution="""    mean = sum(numbers) / len(numbers)
    return sum(abs(x - mean) for x in numbers) / len(numbers)
""",
                entry_point="mean_absolute_deviation",
                test="""
def check(candidate):
    assert abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6
    assert abs(candidate([1.0, 2.0, 3.0, 4.0, 5.0]) - 1.2) < 1e-6
""",
            ),
        ]

        for problem in sample_problems:
            self._problems[problem.task_id] = problem

    def __len__(self) -> int:
        """Number of problems."""
        return len(self._problems)

    def __iter__(self) -> Iterator[HumanEvalProblem]:
        """Iterate over problems."""
        return iter(self._problems.values())

    def __getitem__(self, task_id: str) -> HumanEvalProblem:
        """Get problem by task ID."""
        return self._problems[task_id]

    def get_problem(self, task_id: str) -> HumanEvalProblem | None:
        """Get problem by task ID, returning None if not found."""
        return self._problems.get(task_id)

    def get_problems_by_difficulty(self, difficulty: str) -> list[HumanEvalProblem]:
        """Get problems filtered by difficulty."""
        return [p for p in self._problems.values() if p.difficulty == difficulty]

    def get_problem_ids(self) -> list[str]:
        """Get all problem IDs."""
        return list(self._problems.keys())

    def sample(self, n: int, seed: int | None = None) -> list[HumanEvalProblem]:
        """Sample n random problems."""
        import random

        if seed is not None:
            random.seed(seed)

        problems = list(self._problems.values())
        return random.sample(problems, min(n, len(problems)))

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Convert all problems to dictionary."""
        return {task_id: p.to_dict() for task_id, p in self._problems.items()}


def load_humaneval_problems(
    data_path: str | Path | None = None,
) -> list[HumanEvalProblem]:
    """
    Load HumanEval problems from file.

    Args:
        data_path: Path to HumanEval JSONL file (optional)

    Returns:
        List of HumanEvalProblem instances
    """
    benchmark = HumanEvalBenchmark(data_path=data_path)
    return list(benchmark)
