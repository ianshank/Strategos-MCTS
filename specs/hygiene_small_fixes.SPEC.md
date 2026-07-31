---
id: hygiene_small_fixes
goal: Wire the harness security hooks, fix tz-naive datetimes, unblock the M5 artifact path, add the shared deprecation helper
module: src/framework/harness/
status: draft
---

# Goal

Bundle of small verified fixes: the secret_scan/payload_size/required_keys hooks exist but are
never registered; 8 call sites use deprecated naive datetime.utcnow; .gitignore's benchmarks/
rule blocks the approved m5_policy_lift deliverable; verify_setup imports a nonexistent S3
symbol; and later phases need one shared DeprecationWarning helper.

# Acceptance Criteria

- AC-1: HarnessFactory.create_hook_chain() registers the three builtin hooks from new HarnessSettings fields HOOK_SECRET_SCAN / HOOK_PAYLOAD_SIZE_LIMIT / HOOK_REQUIRED_KEYS (env HARNESS_HOOK_*), default ON, with unit tests for default contents, each toggle, and the settings-driven size limit.
- AC-2: Surviving datetime.utcnow sites use src.utils.time_utils.utc_now; ruff DTZ rules are enabled and green; sites scheduled for deletion are listed as skipped in the PR body.
- AC-3: .gitignore uses benchmarks/* plus !benchmarks/results/ (and !data/training_with_assembly.json); git check-ignore confirms benchmarks/results/m5_policy_lift.json is trackable.
- AC-4: scripts/verification/verify_setup.py no longer imports src.storage.s3_client (check dropped; module is scheduled for deletion).
- AC-5: src/utils/deprecation.py provides warn_deprecated(old, new, stacklevel) with a parametrized test asserting exactly one DeprecationWarning with the correct stacklevel.

# Constraints

- /security-review runs on this PR (secret-scan semantics change what the harness accepts).
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.
