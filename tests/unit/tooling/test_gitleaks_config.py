"""Structural invariants for `.gitleaks.toml`.

The gitleaks binary is not in the `[dev]` extra, so no test here can run a scan. These assert
the properties of the *configuration* that decide whether a scan is worth running at all —
which is where this repository has already been burned once.

`docs/reviews/2026-07-31-charter-alignment-audit.md` F-20: the first version of this config
allowlisted a documentation file **by path** after reading two of its three flagged lines. The
third was a second copy of the exact live W&B key the config exists to catch. The lesson is
written into the config's own header and is enforced here rather than left as prose: an
allowlist entry for human-authored content must name the literal value, so allowlisting one
placeholder can never silently cover a real secret that lands in the same file later.

The second failure mode is subtler and is what this file was added for. Before 2026-09-04 the
documented local command (`make secrets`, step 8 of the `quality-gate` skill) exited 1 on 17
placeholder findings. A gate that always fails teaches the reader to ignore it — worse than no
gate, because it looks like coverage on the checklist. CI did not catch that, because
`gitleaks-action` scans a push's *commit range* while the local command scans the whole working
tree: the two were never checking the same thing.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / ".gitleaks.toml"

#: Credential prefixes that appear in this repository's fixtures. An allowlist entry equal to
#: (or shorter than) one of these would exempt every key of that provider while reading as a
#: single fixture — the one way a short entry is genuinely dangerous.
KNOWN_CREDENTIAL_PREFIXES = ("sk-", "sk-ant-", "sk_live_", "br-", "pc-", "ghp_", "AKIA")

#: Derived, not invented: one character longer than the longest prefix above is the shortest
#: entry that cannot be a bare prefix. Recomputes if a prefix is added.
MIN_ALLOWLIST_REGEX_CHARS = max(len(prefix) for prefix in KNOWN_CREDENTIAL_PREFIXES) + 1

#: Path patterns may only exempt *generated* content, whose bytes are already scanned at their
#: source. Anything a human writes must be exempted by value instead — F-20.
JUSTIFIED_PATH_EXEMPTION_MARKERS = ("__pycache__", "py[co]", "secrets", "baseline")

#: Regex metacharacters that would make an allowlist entry match far more than one value.
DANGEROUSLY_BROAD = (".*", ".+", "[\\s\\S]", "(?s)")


def _config() -> dict:
    try:
        import tomllib as toml_reader
    except ModuleNotFoundError:  # pragma: no cover - only on Python 3.10
        import tomli as toml_reader  # type: ignore[import-not-found]

    with CONFIG.open("rb") as handle:
        return toml_reader.load(handle)


def test_the_config_parses() -> None:
    """A malformed config makes gitleaks fall back or fail, and neither is visible in a diff."""
    assert _config()


def test_it_extends_the_builtin_ruleset_rather_than_replacing_it() -> None:
    """`useDefault = false` would silently reduce the scan to whatever rules we wrote ourselves."""
    assert _config()["extend"]["useDefault"] is True


def test_no_allowlist_regex_is_a_bare_credential_prefix() -> None:
    """`sk-` as an entry would exempt every OpenAI key while reading as one fixture.

    The bound is derived from :data:`KNOWN_CREDENTIAL_PREFIXES` rather than chosen, so it
    stays correct if a provider is added.
    """
    regexes = _config()["allowlist"]["regexes"]
    too_short = [r for r in regexes if len(r) < MIN_ALLOWLIST_REGEX_CHARS]
    assert not too_short, (
        f"allowlist entries under {MIN_ALLOWLIST_REGEX_CHARS} chars can be bare credential prefixes "
        f"and would exempt real secrets: {too_short}"
    )
    exact_prefixes = [r for r in regexes if r in KNOWN_CREDENTIAL_PREFIXES]
    assert not exact_prefixes, f"allowlist entries that are exactly a credential prefix: {exact_prefixes}"


def test_no_allowlist_regex_is_a_wildcard() -> None:
    """`.*` anywhere in an entry turns the allowlist into an off switch."""
    broad = [r for r in _config()["allowlist"]["regexes"] if any(token in r for token in DANGEROUSLY_BROAD)]
    assert not broad, f"allowlist entries matching arbitrary text: {broad}"


def test_every_allowlist_regex_compiles() -> None:
    """An invalid pattern is silently inert in some gitleaks versions."""
    for pattern in _config()["allowlist"]["regexes"]:
        re.compile(pattern)


def test_path_exemptions_cover_only_generated_content() -> None:
    """The F-20 rule, enforced rather than described.

    Path-based exemption is safe exactly when the file's bytes are generated from something
    already being scanned (bytecode, a fingerprint baseline). For anything a human writes,
    exempting the path means the next secret committed to that file is invisible.
    """
    unjustified = [
        pattern
        for pattern in _config()["allowlist"].get("paths", [])
        # Compare against the unescaped text: `\.secrets\.baseline` and `.secrets.baseline`
        # name the same file, and the marker list should not have to know about escaping.
        if not any(marker in pattern.replace("\\", "") for marker in JUSTIFIED_PATH_EXEMPTION_MARKERS)
    ]
    assert not unjustified, (
        "path-based allowlist entries for human-authored content re-create audit finding F-20 — a "
        f"real key hid in a file exempted this way. Allowlist the literal value instead: {unjustified}"
    )


def test_the_allowlist_has_no_duplicate_entries() -> None:
    """Duplicates suggest the list was appended to without reading, which is how F-20 happened."""
    regexes = _config()["allowlist"]["regexes"]
    duplicates = sorted({r for r in regexes if regexes.count(r) > 1})
    assert not duplicates, f"duplicate allowlist entries: {duplicates}"


def test_the_scan_is_wired_into_ci_and_the_makefile() -> None:
    """An unwired scanner is a config file, not a control."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "gitleaks" in workflow, "no CI job runs gitleaks"

    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "gitleaks detect" in makefile, "`make secrets` does not offer the repo-wide scan"
    assert "--config .gitleaks.toml" in makefile, "the Makefile scan ignores this config, so it uses different rules"


def test_the_scan_is_not_invoked_through_the_swallowing_shell_pattern() -> None:
    """`command -v gitleaks && gitleaks detect ... || echo "not installed"` is a silent pass.

    In `A && B || C`, a *found leak* (B exits 1) takes the `||` branch: the target printed
    "gitleaks not installed locally" and exited 0. Both `make secrets` and step 8 of the
    `quality-gate` skill shipped that shape — the skill's own step 7 warns about the pitfall
    two lines above where it committed it. A scanner that reports "absent" when it means
    "found something" is worse than no scanner, so the shape is banned mechanically.
    """
    pattern = re.compile(r"command -v gitleaks[^\n]*&&[^\n]*\|\|", re.MULTILINE)
    for path in (
        REPO_ROOT / "Makefile",
        REPO_ROOT / ".claude" / "skills" / "quality-gate" / "SKILL.md",
    ):
        text = path.read_text(encoding="utf-8")
        offenders = [line.strip() for line in text.splitlines() if pattern.search(line) and "NOT `command" not in line]
        assert not offenders, (
            f"{path.relative_to(REPO_ROOT)} guards gitleaks with `A && B || C`, so a real leak "
            f"reports as 'not installed' and exits 0. Use if/else:\n  " + "\n  ".join(offenders)
        )


def test_the_config_records_why_each_exemption_is_safe() -> None:
    """An allowlist without reasons becomes an append-only dumping ground.

    Not a proxy metric: the ratio matters because every entry here is a decision that a
    specific string is *not* a secret, and that decision is only reviewable if the argument
    for it is written next to it.
    """
    text = CONFIG.read_text(encoding="utf-8")
    comment_lines = sum(1 for line in text.splitlines() if line.strip().startswith("#"))
    entry_lines = sum(1 for line in text.splitlines() if line.strip().startswith("'''"))
    assert comment_lines >= entry_lines, (
        f"{entry_lines} allowlist entries but only {comment_lines} comment lines — every exemption "
        "needs the argument for why that value is not a secret."
    )
