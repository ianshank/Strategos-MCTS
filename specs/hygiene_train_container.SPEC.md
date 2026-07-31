---
id: hygiene_train_container
goal: Make the training container install from the root dependency source so it can import src/
module: Dockerfile.train
status: draft
---

# Goal

Dockerfile.train installs only training/requirements.txt, whose pins conflict with the root
requirements on every shared package and omit pydantic-settings/httpx/anthropic — the
container cannot import src/.

# Acceptance Criteria

- AC-1: The container installs from the root requirements/pyproject (single pin source); docker build succeeds and python -c 'import src' passes inside the image.
- AC-2: The docker-deployment.yml training-demo container run stays green.
- AC-3: Annotated tag pre-training-defork exists.

# Constraints

- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
