---
id: security_secret_scan_hardening
goal: Close the two structural gaps the charter audit found in secret detection with a real, CI-enforced, repo-wide gitleaks scan
module: security/
status: draft
---

# Goal

`docs/reviews/2026-07-31-charter-alignment-audit.md` finding F-17 found a real, committed API key
in `docs/API_CONFIGURATION_GUIDE.md`, invisible to both of the repository's existing guards at
once: the CI secret-scan step (`.github/workflows/ci.yml` "Spec Validation & Secret Scan" job) runs
`git grep -nE "sk-[A-Za-z0-9]{20,}" -- src/ kubernetes/`, which excludes `docs/` entirely and
matches only OpenAI-shaped `sk-` keys; and `detect-secrets` is configured in
`.pre-commit-config.yaml` but has no corresponding job in `.github/workflows/*.yml`, so it never
runs as part of the branch-protection-enforced pipeline. A guard that exists on paper and cannot
fire is not a guard.

This adds a third, independent layer: a repo-wide, pattern-agnostic `gitleaks` scan wired into CI
as a blocking job, alongside — not instead of — the existing fast `git grep` check.

# Acceptance Criteria

- AC-1: `.gitleaks.toml` exists at the repository root, extends gitleaks' built-in ruleset rather
  than reimplementing it, and allowlists only the specific test fixtures and documentation
  placeholders already verified as non-secrets (named literally, not by broad path or pattern
  exclusion).
- AC-2: A `secret-scan-gitleaks` job exists in `.github/workflows/ci.yml`, scans the full
  repository working tree (not scoped to `src/` and `kubernetes/` as the existing check is), and is
  included in the `summary` job's `needs` list and its critical-failure check — so, unlike the gap
  `specs/hygiene_ci_mechanical.SPEC.md` already found in two other jobs, this new job cannot fail
  silently from day one.
- AC-3: The existing `git grep` secret-scan step is unchanged and untouched; the two layers are
  complementary (fast and dependency-free vs. broad-coverage and pattern-aware), not a replacement
  of one by the other.
- AC-4: `CHARTER.md`'s audit disposition for F-17 and the `CHANGELOG.md` entry are updated to
  reference this spec instead of describing the work as unscoped future work.
- AC-5: The configuration's syntax is validated locally (TOML and the surrounding workflow YAML
  both parse); the scan's actual behavior against the live repository is verified by the first CI
  run, not locally — this environment has no `gitleaks` binary installed, and that limitation is
  stated rather than an unverified claim of a clean run being made.

# Constraints

- No behavior change to the existing `git grep` secret-scan step or to `src/**`.
- History scanning is out of scope: the job scans the commit range introduced by each push/PR, not
  the repository's full prior history — re-flagging already-known, already-remediated historical
  secrets (e.g. the F-17 key, now redacted and pending rotation) would be noise, not signal, and a
  full-history sweep is a separate, larger decision belonging to its own spec.
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest
  --cov-fail-under=85 / secret grep).
- CHANGELOG `[Unreleased]` entry.

# Out of Scope

- Scanning git history for previously-committed secrets.
- Removing or narrowing the existing `git grep` check.
- The other CI-honesty gaps `specs/hygiene_ci_mechanical.SPEC.md` already owns (summary job missing
  `chess-tests`/`integration-test`, unused pytest markers, the e2e workflow's `|| true`, mypy's
  strictness claim) — this spec touches none of that job's scope.
