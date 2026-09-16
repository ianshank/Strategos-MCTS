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
DOCKERFILE_INSTRUCTIONS = {
    "ADD",
    "ARG",
    "CMD",
    "COPY",
    "ENTRYPOINT",
    "ENV",
    "EXPOSE",
    "FROM",
    "HEALTHCHECK",
    "LABEL",
    "MAINTAINER",
    "ONBUILD",
    "RUN",
    "SHELL",
    "STOPSIGNAL",
    "USER",
    "VOLUME",
    "WORKDIR",
}
PRODUCTION_STAGE_ALIAS = re.compile(r"\bAS\s+production\b", re.IGNORECASE)
PERL_APT_INSTALL = re.compile(
    r"(?:^|[(&;|])\s*apt-get\b[^;&|)]*\b(?:install|upgrade)\b[^;&|)]*\bperl-base\b",
    re.IGNORECASE,
)
DOCKERFILE_CONTINUATION = re.compile(r"\\\s*\n\s*")


def _is_instruction(line: str, name: str | None = None) -> bool:
    tokens = line.lstrip().split(maxsplit=1)
    if not tokens:
        return False

    instruction = tokens[0].upper()
    if instruction not in DOCKERFILE_INSTRUCTIONS:
        return False
    if name is None:
        return True
    return instruction == name.upper()


def _production_stage(dockerfile: str) -> str:
    lines = dockerfile.splitlines()
    start = next(
        (idx for idx, line in enumerate(lines) if _is_instruction(line, "FROM") and PRODUCTION_STAGE_ALIAS.search(line)),
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


def _installs_or_upgrades_perl_base(run_instruction: str) -> bool:
    normalized = " ".join(DOCKERFILE_CONTINUATION.sub("", run_instruction).split())
    return bool(PERL_APT_INSTALL.search(normalized))


def test_production_stage_installs_perl_base() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    production = _production_stage(dockerfile)
    apt_get_runs = [run for run in _run_instructions(production) if "apt-get update" in run.lower()]

    assert apt_get_runs, "production stage must run apt-get update before runtime package install"
    assert any(_installs_or_upgrades_perl_base(run) for run in apt_get_runs), (
        "production RUN must install/upgrade perl-base after apt-get update "
        "so the image is at least 5.40.1-6+deb13u1"
    )


def test_dockerfile_parser_is_case_insensitive() -> None:
    stage = _production_stage(
        """  FROM python:3.11-slim As builder
RUN echo builder
  from python:3.11-slim   aS   production
  run apt-get update && apt-get install -y perl-base
  Label stage=production
"""
    )

    assert stage.splitlines()[0] == "  from python:3.11-slim   aS   production"
    assert _run_instructions(stage) == ["  run apt-get update && apt-get install -y perl-base"]



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



def test_perl_base_policy_matches_install_run_without_cve_strings() -> None:
    stage = _production_stage(
        """FROM python:3.11-slim AS production
RUN echo preflight
RUN apt-get update && apt-get upgrade -y perl-base curl
"""
    )

    assert any(_installs_or_upgrades_perl_base(run) for run in _run_instructions(stage))



def test_perl_base_install_only_upgrade_counts_as_valid_fix() -> None:
    stage = _production_stage(
        """FROM python:3.11-slim AS production
RUN apt-get update && apt-get install --only-upgrade -y perl-base
"""
    )

    assert any(_installs_or_upgrades_perl_base(run) for run in _run_instructions(stage))



def test_perl_base_grouped_shell_command_counts_as_install() -> None:
    stage = _production_stage(
        """FROM python:3.11-slim AS production
RUN (apt-get update && apt-get install -y perl-base)
"""
    )

    assert any(_installs_or_upgrades_perl_base(run) for run in _run_instructions(stage))



def test_perl_base_grouped_multiline_command_counts_as_install() -> None:
    stage = _production_stage(
        """FROM python:3.11-slim AS production
RUN (apt-get update && \\
    apt-get install -y perl-\\
    base)
"""
    )

    assert any(_installs_or_upgrades_perl_base(run) for run in _run_instructions(stage))



def test_perl_base_must_be_in_production_apt_get_run() -> None:
    stage = _production_stage(
        """FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y perl-base
FROM python:3.11-slim AS production
RUN echo preflight
RUN apt-get update && apt-get install -y curl
"""
    )

    assert not any(_installs_or_upgrades_perl_base(run) for run in _run_instructions(stage))



def test_perl_base_mentions_outside_install_do_not_count() -> None:
    stage = _production_stage(
        """FROM python:3.11-slim AS production
RUN apt-get update && apt-get install -y curl && echo perl-base && apt-get remove -y perl-base
"""
    )

    assert not any(_installs_or_upgrades_perl_base(run) for run in _run_instructions(stage))



def test_trivyignore_does_not_accept_perl_base_cves() -> None:
    ignore = (REPO_ROOT / ".trivyignore").read_text(encoding="utf-8")
    uncommented = [line.strip() for line in ignore.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    for cve in PERL_CVES:
        assert not any(cve in line for line in uncommented), f"{cve} must not be ignored; upgrade perl-base instead"
