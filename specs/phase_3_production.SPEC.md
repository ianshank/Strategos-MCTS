---
goal: Make the staging deployment ship-safe (secret hygiene, JWT auth selection, soak)
phase: "3"
milestone: M4
status: active
---

# Goal

Close M4 deployment readiness: remove plaintext secrets from version control, make JWT vs API-key auth
runtime-selectable without breaking existing clients, and validate a staging deployment under soak.

# Acceptance Criteria

- `git grep` finds no plaintext key material in `kubernetes/`; manifests reference an external secret
  store (External Secrets Operator by default); `docs/SECRETS_MANAGEMENT.md` documents rotation.
- A settings-driven `AUTH_MODE` selects the already-implemented `JWTAuthenticator` or the existing
  `APIKeyAuthenticator`; JWT issue/verify is covered by tests; API-key clients keep working with no config
  change under the default mode; `get_authenticator()`'s public return contract is unchanged.
- JWT algorithm, expiry, and secret are sourced from settings; PyJWT is an optional extra.
- Staging health checks stay green for 24h; the smoke suite passes in staging; rollback is documented.

# Constraints

- Backward compatible: the default `AUTH_MODE` preserves current API-key behavior exactly.
- No secret material committed to the repo; values resolve env -> settings.
- No hardcoded values; full local gate green before push; CI secret-scan passes.
