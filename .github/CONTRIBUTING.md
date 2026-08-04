# Contributing to Strategos-MCTS

Thanks for your interest in improving Strategos-MCTS (distributed on PyPI as
`langgraph-multi-agent-mcts`). This guide covers environment setup, the quality gate every change must
pass, and the spec-driven development (SDD) workflow this repository uses.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Table of Contents

- [Development environment](#development-environment)
- [The quality gate](#the-quality-gate)
- [Spec-driven development](#spec-driven-development)
- [Commit and PR conventions](#commit-and-pr-conventions)
- [Reporting bugs and requesting features](#reporting-bugs-and-requesting-features)
- [Security issues](#security-issues)

## Development environment

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev,neural,api]"   # dev tooling + PyTorch + FastAPI — the CI test job's exact set
cp .env.example .env                 # then add OPENAI_API_KEY or ANTHROPIC_API_KEY
```

`ruff` and `mypy` are pinned to a validated minor in the `[dev]` extra so local and CI runs use identical
tool versions. Bump these pins deliberately and re-run the full gate — lint/type behavior shifts across
releases.

## The quality gate

Every change must pass the full local gate before it is pushed. Run the `quality-gate` skill, or the
commands it wraps:

```bash
black . --check --line-length 120
ruff check .
mypy src/
pytest tests/unit/ --cov=src --cov-fail-under=85    # 85% branch-coverage gate (CI enforces this)
```

- **Coverage:** CI enforces an **85% branch-coverage gate**. The current baseline is tracked in
  [`docs/STATUS.md`](../docs/STATUS.md) (the single source of truth for test/coverage status).
- **No hardcoded values:** all configuration flows through Pydantic Settings (`src/config/settings.py`).
  Never hardcode API keys, model names, or magic numbers.
- **Secrets:** CI runs two independent scans — a fast `src/`/`kubernetes/`-scoped grep, and a
  repo-wide, pattern-agnostic `gitleaks` scan (`.gitleaks.toml`). Both must pass; see the
  `quality-gate` skill for how to run each locally.
- **Async-first:** new I/O paths must be `async`; async tests use `@pytest.mark.asyncio` + `await`.
- If you touch anything under `.claude/`, also run `validate-context-docs` (the `validate-context` skill).
- If you touch `specs/`, run `harness validate-spec specs/*.SPEC.md`.

## Spec-driven development

Substantive work under `src/**` is specified before it is implemented:

1. `/spec-new <id> <module>` scaffolds a `draft` spec under `specs/<id>.SPEC.md` (schema v2).
2. The `spec-review` agent gates `draft → approved`; a human flips the status.
3. `/spec-implement <id>` requires an `approved` spec and cuts/switches to a `spec/<id>` branch from
   `origin/main`.

**CI traceability:** a PR whose diff touches `src/**` needs either a `spec/<id>` branch with an approved
spec on the base branch, **or** a `No-Spec: <reason>` trailer on the commit. Approved specs now exist, so
the `spec/<id>` branch is the **default** channel for `src/**` work and the `No-Spec:` trailer is the
written exception — it must state a real reason, not simply assert one. Documentation, governance, and
tooling changes (like this file) do not require a spec. See [`CHARTER.md`](../CHARTER.md) §3 NG-4 for the
boundary this enforces and §7 for how exceptions are budgeted.

## Commit and PR conventions

- Write clear, imperative commit messages; group related changes.
- Open pull requests as **drafts** until CI is green, then mark ready for review.
- Fill in the [pull request template](PULL_REQUEST_TEMPLATE.md), including spec linkage (or a `No-Spec:`
  reason) and the checklist.
- Update [`CHANGELOG.md`](../CHANGELOG.md) under `[Unreleased]` (Keep a Changelog format) for any
  user-facing change.

## Reporting bugs and requesting features

Use the issue templates: **Bug report** or **Feature request**. For questions and help, see
[SUPPORT.md](SUPPORT.md).

## Security issues

Do **not** open a public issue for a vulnerability. Follow the process in [SECURITY.md](SECURITY.md).
