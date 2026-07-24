# Security Policy

## Supported versions

Strategos-MCTS is pre-1.0 and under active development. Security fixes are applied to the latest `0.1.x`
line only.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Preferred channel: use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository (the **Security → Report a vulnerability** tab). This keeps the report private until a
fix is available.

If private reporting is unavailable, email **ianshank@gmail.com** with the details below.

Please include:

- A description of the vulnerability and its impact.
- Steps to reproduce (proof-of-concept where possible).
- Affected version/commit and configuration (LLM provider, deployment mode).

You can expect an initial acknowledgement within a few business days. Please give us a reasonable window to
investigate and release a fix before any public disclosure.

## Handling of secrets and sensitive data

This project takes credential hygiene seriously and enforces it in CI:

- Secrets are never committed. Configuration flows through environment variables / Pydantic Settings; see
  [`docs/SECRETS_MANAGEMENT.md`](../docs/SECRETS_MANAGEMENT.md) for the External Secrets Operator setup and
  rotation guidance.
- CI runs secret scanning (`detect-secrets`), static security analysis (`bandit`), and dependency
  vulnerability auditing (`pip-audit`), plus a hardcoded-key grep in the local quality gate.

If you discover committed secrets, treat it as a vulnerability and report it via the channel above.
