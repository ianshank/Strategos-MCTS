"""Tests for the fail-loud deployment posture validator.

Spec: ``specs/evidence_claim_ledger.SPEC.md`` AC-9.

``CHARTER.md`` promises the service fails loud rather than serving mock model output. Before this
change that promise was only a *default*: ``ALLOW_MOCK_LLM_FALLBACK`` defaults to ``False``, but a
single environment variable could flip it on in production, and every container manifest passes the
environment through untouched. These tests assert the promise is now structural for the environments
where mock output would be a correctness incident — and, just as importantly, that introducing the
new field changed no existing behaviour.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.constants import (
    DEFAULT_DEPLOYMENT_ENV,
    DEPLOYMENT_ENV_DEVELOPMENT,
    DEPLOYMENT_ENV_PRODUCTION,
    DEPLOYMENT_ENV_STAGING,
    DEPLOYMENT_ENV_TEST,
    DEPLOYMENT_ENVS,
    FAIL_LOUD_ENFORCED_ENVS,
)
from src.config.settings import Settings


def _settings(**overrides: object) -> Settings:
    """Construct settings from explicit values only, ignoring any ambient .env or environment.

    Without this the test outcome would depend on the developer's shell, which is precisely the class
    of accident this validator exists to prevent.
    """
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


# --------------------------------------------------------------- backwards compatibility


@pytest.mark.unit
def test_default_deployment_env_is_permissive() -> None:
    """The new field must not change the behaviour of any deployment that never sets it."""
    assert _settings().DEPLOYMENT_ENV == DEFAULT_DEPLOYMENT_ENV
    assert DEFAULT_DEPLOYMENT_ENV not in FAIL_LOUD_ENFORCED_ENVS


@pytest.mark.unit
def test_mock_fallback_still_defaults_off() -> None:
    """The pre-existing default is unchanged; enforcement is additive, not a re-default."""
    assert _settings().ALLOW_MOCK_LLM_FALLBACK is False


@pytest.mark.unit
def test_existing_dev_workflow_of_enabling_the_fallback_still_works() -> None:
    """Local development and the test suite rely on this combination. It must keep working."""
    assert _settings(ALLOW_MOCK_LLM_FALLBACK=True).ALLOW_MOCK_LLM_FALLBACK is True


@pytest.mark.unit
@pytest.mark.parametrize("environment", [DEPLOYMENT_ENV_DEVELOPMENT, DEPLOYMENT_ENV_TEST])
def test_fallback_is_allowed_in_non_enforced_environments(environment: str) -> None:
    settings = _settings(DEPLOYMENT_ENV=environment, ALLOW_MOCK_LLM_FALLBACK=True)
    assert settings.ALLOW_MOCK_LLM_FALLBACK is True


# --------------------------------------------------------------- the refusal


@pytest.mark.unit
@pytest.mark.parametrize("environment", FAIL_LOUD_ENFORCED_ENVS)
def test_fallback_is_refused_in_enforced_environments(environment: str) -> None:
    """The core property: production plus mock fallback is not a configuration, it is an error."""
    with pytest.raises(ValidationError, match="ALLOW_MOCK_LLM_FALLBACK=true is refused"):
        _settings(DEPLOYMENT_ENV=environment, ALLOW_MOCK_LLM_FALLBACK=True)


@pytest.mark.unit
def test_production_without_the_fallback_is_accepted() -> None:
    """Declaring production must not itself be an error, or nobody will declare it."""
    settings = _settings(DEPLOYMENT_ENV=DEPLOYMENT_ENV_PRODUCTION)
    assert settings.ALLOW_MOCK_LLM_FALLBACK is False


@pytest.mark.unit
def test_staging_is_enforced_as_well_as_production() -> None:
    """Staging is what production is validated against; mock output there invalidates the rehearsal."""
    assert DEPLOYMENT_ENV_STAGING in FAIL_LOUD_ENFORCED_ENVS
    with pytest.raises(ValidationError):
        _settings(DEPLOYMENT_ENV=DEPLOYMENT_ENV_STAGING, ALLOW_MOCK_LLM_FALLBACK=True)


@pytest.mark.unit
def test_refusal_message_names_the_remedy() -> None:
    """An error an operator cannot act on gets worked around, usually by deleting the check."""
    with pytest.raises(ValidationError) as caught:
        _settings(DEPLOYMENT_ENV=DEPLOYMENT_ENV_PRODUCTION, ALLOW_MOCK_LLM_FALLBACK=True)
    message = str(caught.value)
    assert "Remedy" in message
    assert DEPLOYMENT_ENV_DEVELOPMENT in message
    assert "correctness incident" in message


# --------------------------------------------------------------- input handling


@pytest.mark.unit
@pytest.mark.parametrize("spelling", ["PRODUCTION", "Production", "  production  "])
def test_enforcement_is_case_and_whitespace_insensitive(spelling: str) -> None:
    """`ENV=Production` must not be a bypass. Operators type what reads naturally."""
    with pytest.raises(ValidationError, match="refused"):
        _settings(DEPLOYMENT_ENV=spelling, ALLOW_MOCK_LLM_FALLBACK=True)


@pytest.mark.unit
@pytest.mark.parametrize("bogus", ["prod", "prd", "live", "staging-2", ""])
def test_unrecognised_deployment_env_is_rejected(bogus: str) -> None:
    """A typo must be an error, not a silent downgrade to "not production".

    Accepting unknown values as non-production would make ``ENV=prod`` a one-character bypass of the
    entire check, which is the failure mode this test exists to close.
    """
    with pytest.raises(ValidationError, match="is not recognised"):
        _settings(DEPLOYMENT_ENV=bogus)


@pytest.mark.unit
def test_unrecognised_env_error_lists_the_legal_values() -> None:
    with pytest.raises(ValidationError) as caught:
        _settings(DEPLOYMENT_ENV="prod")
    message = str(caught.value)
    for environment in DEPLOYMENT_ENVS:
        assert environment in message


@pytest.mark.unit
@pytest.mark.parametrize("environment", DEPLOYMENT_ENVS)
def test_every_declared_environment_is_constructible(environment: str) -> None:
    """The vocabulary and the validator must agree, or a documented value is unusable."""
    assert environment == _settings(DEPLOYMENT_ENV=environment).DEPLOYMENT_ENV


@pytest.mark.unit
def test_enforced_environments_are_a_subset_of_the_vocabulary() -> None:
    """Guards against a future edit that enforces a value no operator can legally set."""
    assert set(FAIL_LOUD_ENFORCED_ENVS) <= set(DEPLOYMENT_ENVS)


@pytest.mark.unit
def test_deployment_env_is_readable_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators configure containers with env vars, so the field must bind to one."""
    monkeypatch.setenv("DEPLOYMENT_ENV", DEPLOYMENT_ENV_PRODUCTION)
    monkeypatch.setenv("ALLOW_MOCK_LLM_FALLBACK", "true")
    with pytest.raises(ValidationError, match="refused"):
        Settings(_env_file=None)  # type: ignore[call-arg]
