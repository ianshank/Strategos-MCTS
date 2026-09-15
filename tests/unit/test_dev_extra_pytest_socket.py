"""Pin: pytest-socket is declared in the ``[dev]`` extra.

Spec: ``hygiene_test_triage`` AC-4. This test only asserts the extra exists so the
plugin is installable. Wiring ``--disable-socket`` for ``tests/unit/`` collection
is the rest of that AC and is not claimed here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_pyproject() -> dict:
    try:
        import tomllib as toml_reader
    except ModuleNotFoundError:  # pragma: no cover - only on Python 3.10
        import tomli as toml_reader  # type: ignore[import-not-found]

    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return toml_reader.load(fh)


def test_pytest_socket_is_declared_in_dev_extra() -> None:
    extras = _load_pyproject()["project"]["optional-dependencies"]["dev"]
    assert any(req.startswith("pytest-socket") for req in extras), (
        "hygiene_test_triage AC-4 requires pytest-socket in the [dev] extra so unit "
        "tests can fail closed on unexpected sockets"
    )
