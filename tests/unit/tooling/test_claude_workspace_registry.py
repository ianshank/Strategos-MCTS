"""Registry integrity for the `.claude/` workspace: skills, agents and hooks.

``src/tools/context_docs.py`` already validates each skill and agent *document* — its
frontmatter schema, the repo paths it cites, and the pinned value claims it quotes. What no
check covered is the **registry** those documents form, and the failure mode there is silent
in the opposite direction: an artefact that exists but is not wired, or two artefacts that
claim the same address.

Three concrete defects this pins, none of which any existing gate would have caught:

* **An unregistered hook.** A file under ``.claude/hooks/`` that ``.claude/settings.json``
  never names is inert. It reviews clean, its own unit tests pass, and it enforces nothing —
  the most expensive shape of dead code because it reads as a control.
* **A dangling registration.** A ``settings.json`` command pointing at a hook that was
  renamed or deleted fails at the moment a tool call is intercepted, in an environment where
  hook errors are deliberately non-fatal, so the gate simply stops firing.
* **A colliding name.** Skills and agents are addressed by their frontmatter ``name``. Two
  artefacts sharing one makes dispatch order-dependent on the filesystem.

Pure filesystem and JSON, no imports from ``src``: the same tree always yields the same
verdict, which is the property that lets this run as a gate rather than as advice.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
CLAUDE_DIR = REPO_ROOT / ".claude"
SETTINGS = CLAUDE_DIR / "settings.json"
HOOKS_DIR = CLAUDE_DIR / "hooks"
SKILLS_DIR = CLAUDE_DIR / "skills"
AGENTS_DIR = CLAUDE_DIR / "agents"

#: The hook events this repository wires. Listed so a typo'd event key — which the runtime
#: would simply never dispatch — fails here rather than silently disabling a gate.
KNOWN_HOOK_EVENTS = frozenset({"PreToolUse", "PostToolUse", "SessionStart", "Stop", "SubagentStop"})

#: A frontmatter ``description`` shorter than this cannot carry the "use when …" triggers a
#: dispatcher routes on. The bound is deliberately loose; it catches an empty or placeholder
#: value, not a terse-but-real one.
MIN_DESCRIPTION_CHARS = 40


def _settings() -> dict:
    return json.loads(SETTINGS.read_text(encoding="utf-8"))


def _registered_hook_commands() -> list[str]:
    commands: list[str] = []
    for event, entries in _settings().get("hooks", {}).items():
        assert event in KNOWN_HOOK_EVENTS, f"settings.json wires unknown hook event {event!r}; it would never fire"
        for entry in entries:
            for hook in entry.get("hooks", []):
                commands.append(hook.get("command", ""))
    return commands


def _hook_files() -> list[Path]:
    return sorted(p for p in HOOKS_DIR.glob("*.py") if not p.name.startswith("_"))


def _frontmatter(path: Path) -> dict[str, str]:
    """Parse the leading ``---`` block. Deliberately not YAML — no dependency, and the
    frontmatter here is flat ``key: value`` with occasional folded scalars."""
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, f"{path.relative_to(REPO_ROOT)} has no frontmatter block"
    fields: dict[str, str] = {}
    key: str | None = None
    for line in match.group(1).splitlines():
        header = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if header:
            key, value = header.group(1), header.group(2).strip()
            fields[key] = value.lstrip(">|-").strip()
        elif key is not None and line.strip():
            fields[key] = f"{fields[key]} {line.strip()}".strip()
    return fields


def _skill_docs() -> list[Path]:
    return sorted(SKILLS_DIR.glob("*/SKILL.md"))


def _agent_docs() -> list[Path]:
    return sorted(AGENTS_DIR.glob("*.md"))


# ----------------------------------------------------------------------- hook wiring


def test_settings_json_is_parseable() -> None:
    """A malformed settings file disables every gate at once, silently."""
    assert isinstance(_settings(), dict)


def test_every_hook_file_is_registered() -> None:
    """An unwired hook enforces nothing while reading as a control."""
    commands = _registered_hook_commands()
    unwired = [p.name for p in _hook_files() if not any(p.name in command for command in commands)]
    assert not unwired, (
        f"hook file(s) present but not registered in {SETTINGS.relative_to(REPO_ROOT)}: {unwired}. "
        "An unregistered hook never fires — either wire it or delete it."
    )


def test_every_registration_points_at_a_file_that_exists() -> None:
    """A dangling command fails only when a tool call is intercepted, where it is non-fatal."""
    missing: list[str] = []
    for command in _registered_hook_commands():
        for referenced in re.findall(r"\.claude/hooks/([\w.-]+\.py)", command):
            if not (HOOKS_DIR / referenced).is_file():
                missing.append(referenced)
    assert not missing, f"settings.json registers hook(s) with no file on disk: {sorted(set(missing))}"


def test_hook_commands_use_the_project_dir_variable() -> None:
    """A relative or absolute path makes the hook depend on the invoking cwd or machine."""
    offenders = [c for c in _registered_hook_commands() if ".claude/hooks/" in c and "CLAUDE_PROJECT_DIR" not in c]
    assert not offenders, f"hook command(s) not rooted at ${{CLAUDE_PROJECT_DIR}}: {offenders}"


def test_every_hook_has_its_own_test_module() -> None:
    """A gate with no tests is a gate nobody can safely change.

    Enforced by naming convention (``test_<hook stem>.py``) so adding a hook forces the
    question at the moment it is added rather than at the next audit.
    """
    here = Path(__file__).parent
    missing = [p.name for p in _hook_files() if not (here / f"test_{p.stem}.py").is_file()]
    assert not missing, f"hook(s) with no tests/unit/tooling/test_<name>.py: {missing}"


# ------------------------------------------------------------------ skill/agent registry


@pytest.mark.parametrize("path", _skill_docs(), ids=lambda p: p.parent.name)
def test_a_skill_name_matches_its_directory(path: Path) -> None:
    """The directory is the address; a mismatched ``name`` makes the skill unfindable."""
    assert _frontmatter(path).get("name") == path.parent.name


@pytest.mark.parametrize("path", _agent_docs(), ids=lambda p: p.stem)
def test_an_agent_name_matches_its_filename(path: Path) -> None:
    assert _frontmatter(path).get("name") == path.stem


@pytest.mark.parametrize("path", [*_skill_docs(), *_agent_docs()], ids=lambda p: p.stem or p.parent.name)
def test_every_artefact_carries_a_routable_description(path: Path) -> None:
    """Dispatch is by description. An empty one means the artefact is never selected."""
    description = _frontmatter(path).get("description", "")
    assert len(description) >= MIN_DESCRIPTION_CHARS, (
        f"{path.relative_to(REPO_ROOT)} has a {len(description)}-char description; too short to route on. "
        "Say what it does and when to use it."
    )


def test_no_two_artefacts_share_a_name() -> None:
    """Skills and agents share one namespace; a collision makes dispatch order-dependent."""
    seen: dict[str, str] = {}
    collisions: list[str] = []
    for path in (*_skill_docs(), *_agent_docs()):
        name = _frontmatter(path).get("name", "")
        rel = path.relative_to(REPO_ROOT).as_posix()
        if name in seen:
            collisions.append(f"{name!r}: {seen[name]} and {rel}")
        seen[name] = rel
    assert not collisions, "duplicate skill/agent name(s): " + "; ".join(collisions)


def test_every_agent_declares_its_tool_surface() -> None:
    """An agent without ``tools`` inherits everything, including write access it should not have."""
    missing = [p.stem for p in _agent_docs() if not _frontmatter(p).get("tools")]
    assert not missing, f"agent(s) with no `tools:` declaration: {missing}"


def test_the_registry_is_not_empty() -> None:
    """Guards the checks above against a glob that silently stops matching."""
    assert _skill_docs() and _agent_docs() and _hook_files()
