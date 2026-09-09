"""local_distillation_contract AC-5: no llm_guided / Unsloth / Hub / HRM-TRM wrappers."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_FORBIDDEN_MODULES = (
    "unsloth",
    "huggingface_hub",
    "src.framework.mcts.llm_guided",
    "src.agents.hrm_agent",
    "src.agents.trm_agent",
)
_FORBIDDEN_NAMES = {"HRMAgent", "TRMAgent"}


def _imported_modules(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _imported_symbols(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[-1])
    return names


def test_package_sources_have_no_forbidden_imports() -> None:
    root = Path(__file__).resolve().parents[4]
    pkg = root / "scripts" / "local_distillation"
    hits: list[str] = []
    for path in sorted(pkg.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        modules = _imported_modules(tree)
        symbols = _imported_symbols(tree)
        for module in _FORBIDDEN_MODULES:
            if module in modules or any(item == module or item.startswith(module + ".") for item in modules):
                hits.append(f"{path.name} imports {module}")
        for symbol in _FORBIDDEN_NAMES:
            if symbol in symbols:
                hits.append(f"{path.name} imports {symbol}")
    assert hits == []
