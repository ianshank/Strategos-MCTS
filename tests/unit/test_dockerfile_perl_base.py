"""Production image must upgrade perl-base (CI Trivy CRITICAL gate).

CI Pipeline docker-build (run 34705825904) failed on three fixable CRITICAL
findings in ``perl-base`` 5.40.1-6 on ``python:3.11-slim`` (Debian 13). The
fixes ship in 5.40.1-6+deb13u1. Dropping Gradio does not unred that job.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[2]
PERL_CVES = ("CVE-2026-13221", "CVE-2026-42496", "CVE-2026-8376")
TOP_LEVEL_DOCKERFILE_INSTRUCTION = re.compile(r"^[A-Z][A-Z0-9_]*(?:\s|$)", re.IGNORECASE)


def _is_instruction(line: str, name: str | None = None) -> bool:
    if not TOP_LEVEL_DOCKERFILE_INSTRUCTION.match(line):
        return False
    if name is None:
        return True
    return line.split(maxsplit=1)[0].lower() == name.lower()


def _production_stage(dockerfile: str) -> str:
    lines = dockerfile.splitlines()
    start = next(
        (idx for idx, line in enumerate(lines) if _is_instruction(line, "FROM") and " as production" in line.lower()),
        None,
    )
    assert start is not None, "Dockerfile has no production stage"

    end = next((idx for idx, line in enumerate(lines[start + 1 :], start + 1) if _is_instruction(line, "FROM")), len(lines))
    return "\n".join(lines[start:end])


def _run_instructions(stage: str) -> list[str]:
    instructions: list[str] = []
    current: list[str] = []

    for line in stage.splitlines():
        if _is_instruction(line, "RUN"):
            if current:
                instructions.append("\n".join(current))
            current = [line]
            continue
        if current:
            if _is_instruction(line):
                instructions.append("\n".join(current))
                current = []
            else:
                current.append(line)

    if current:
        instructions.append("\n".join(current))

    return instructions


def test_production_stage_installs_perl_base() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    production = _production_stage(dockerfile)
    run_instruction = next((run for run in _run_instructions(production) if "apt-get update" in run), None)

    assert run_instruction is not None, "production stage must run apt-get update before runtime package install"
    assert "perl-base" in run_instruction, (
        "production RUN must install/upgrade perl-base after apt-get update "
        "so the image is at least 5.40.1-6+deb13u1"
    )


def test_dockerfile_parser_is_case_insensitive() -> None:
    stage = _production_stage(
        """FROM python:3.11-slim As builder
RUN echo builder
from python:3.11-slim aS production
run apt-get update && apt-get install -y perl-base
Label stage=production
"""
    )

    assert stage.splitlines()[0] == "from python:3.11-slim aS production"
    assert _run_instructions(stage) == ["run apt-get update && apt-get install -y perl-base"]



def test_run_instruction_parser_stops_at_next_top_level_instruction() -> None:
    instructions = _run_instructions(
        """FROM python:3.11-slim AS production
RUN echo preflight
RUN apt-get update && apt-get install -y --no-install-recommends \\
    perl-base
ARG BUILD_DATE=2026-09-16
"""
    )

    assert instructions == [
        "RUN echo preflight",
        "RUN apt-get update && apt-get install -y --no-install-recommends \\\n    perl-base",
    ]



def test_trivyignore_does_not_accept_perl_base_cves() -> None:
    ignore = (REPO_ROOT / ".trivyignore").read_text(encoding="utf-8")
    uncommented = [line.strip() for line in ignore.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    for cve in PERL_CVES:
        assert not any(cve in line for line in uncommented), f"{cve} must not be ignored; upgrade perl-base instead"
