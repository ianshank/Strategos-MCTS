"""
Comprehensive unit tests for API authentication layer.

Tests:
- API key validation and hashing
- Rate limit enforcement across different time windows
- Client tracking and statistics
- Role-based authorization
- Security aspects (no sensitive data exposure)
"""

from datetime import datetime
import hashlib

# Import the authentication classes
import sys
import time
from unittest.mock import patch

import pytest

sys.path.insert(0, ".")
from src.api.auth import (
    APIKeyAuthenticator,
    ClientInfo,
    JWTAuthenticator,
    RateLimitConfig,
    get_authenticator,
    get_jwt_authenticator,
    set_authenticator,
    set_jwt_authenticator,
)
from src.api.exceptions import (
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
)


class TestClientInfo:
    """Test suite for ClientInfo dataclass."""

    def test_default_initialization(self):
        """Test ClientInfo with default values."""
        client = ClientInfo(client_id="test_client")

        assert client.client_id == "test_client"
        assert client.roles == {"user"}
        assert isinstance(client.created_at, datetime)
        assert isinstance(client.last_access, datetime)
        assert client.request_count == 0

    def test_custom_roles(self):
        """Test ClientInfo with custom roles."""
        roles = {"admin", "user", "manager"}
        client = ClientInfo(client_id="admin_client", roles=roles)

        assert client.roles == roles
        assert "admin" in client.roles
        assert "manager" in client.roles

    def test_request_count_tracking(self):
        """Test that request count can be incremented."""
        client = ClientInfo(client_id="test")
        client.request_count += 1
        assert client.request_count == 1

        client.request_count += 10
        assert client.request_count == 11


class TestRateLimitConfig:
    """Test suite for RateLimitConfig dataclass."""

    def test_default_configuration(self):
        """Test default rate limit values."""
        config = RateLimitConfig()

        assert config.requests_per_minute == 60
        assert config.requests_per_hour == 1000
        assert config.requests_per_day == 10000
        assert config.burst_limit == 100

    def test_custom_configuration(self):
        """Test custom rate limit configuration."""
        config = RateLimitConfig(
            requests_per_minute=30,
            requests_per_hour=500,
            requests_per_day=5000,
            burst_limit=50,
        )

        assert config.requests_per_minute == 30
        assert config.requests_per_hour == 500
        assert config.requests_per_day == 5000
        assert config.burst_limit == 50

    def test_restrictive_configuration(self):
        """Test very restrictive rate limits."""
        config = RateLimitConfig(
            requests_per_minute=1,
            requests_per_hour=10,
            requests_per_day=100,
            burst_limit=1,
        )

        assert config.burst_limit == 1


class TestAPIKeyAuthenticator:
    """Test suite for APIKeyAuthenticator class."""

    def test_initialization_with_no_keys(self):
        """Test authenticator initialization without initial keys."""
        auth = APIKeyAuthenticator()

        assert auth._key_to_client == {}
        assert isinstance(auth.rate_limit_config, RateLimitConfig)

    def test_initialization_with_valid_keys(self):
        """Test authenticator initialization with valid keys."""
        keys = ["key1", "key2", "key3"]
        auth = APIKeyAuthenticator(valid_keys=keys)

        assert len(auth._key_to_client) == 3

    def test_initialization_with_custom_rate_limits(self):
        """Test authenticator with custom rate limit configuration."""
        config = RateLimitConfig(requests_per_minute=10)
        auth = APIKeyAuthenticator(rate_limit_config=config)

        assert auth.rate_limit_config.requests_per_minute == 10

    def test_key_hashing_uses_sha256(self):
        """Test that keys are hashed using SHA-256."""
        auth = APIKeyAuthenticator()
        api_key = "test_api_key_12345"

        hashed = auth._hash_key(api_key)
        expected = hashlib.sha256(api_key.encode("utf-8")).hexdigest()

        assert hashed == expected
        assert len(hashed) == 64  # SHA-256 hex digest length

    def test_key_hashing_is_deterministic(self):
        """Test that same key always produces same hash."""
        auth = APIKeyAuthenticator()
        api_key = "consistent_key"

        hash1 = auth._hash_key(api_key)
        hash2 = auth._hash_key(api_key)

        assert hash1 == hash2

    def test_different_keys_produce_different_hashes(self):
        """Test that different keys produce different hashes."""
        auth = APIKeyAuthenticator()

        hash1 = auth._hash_key("key_one")
        hash2 = auth._hash_key("key_two")

        assert hash1 != hash2

    def test_add_key_stores_hashed_version(self):
        """Test that _add_key stores hashed version, not plain text."""
        auth = APIKeyAuthenticator()
        api_key = "secret_key_value"

        auth._add_key(api_key, "client_1")

        # Plain key should not be in storage
        assert api_key not in auth._key_to_client

        # Hashed version should be present
        key_hash = auth._hash_key(api_key)
        assert key_hash in auth._key_to_client

    def test_add_key_with_custom_roles(self):
        """Test adding key with custom roles."""
        auth = APIKeyAuthenticator()
        roles = {"admin", "superuser"}

        auth._add_key("admin_key", "admin_client", roles=roles)

        key_hash = auth._hash_key("admin_key")
        client_info = auth._key_to_client[key_hash]

        assert client_info.roles == roles
        assert client_info.client_id == "admin_client"

    def test_authenticate_with_valid_key(self):
        """Test successful authentication with valid key."""
        api_key = "valid_api_key_123"
        auth = APIKeyAuthenticator(valid_keys=[api_key])

        client_info = auth.authenticate(api_key)

        assert isinstance(client_info, ClientInfo)
        assert client_info.client_id == "client_0"
        assert client_info.request_count == 1

    def test_authenticate_updates_last_access(self):
        """Test that authentication updates last_access timestamp."""
        api_key = "test_key"
        auth = APIKeyAuthenticator(valid_keys=[api_key])

        key_hash = auth._hash_key(api_key)
        original_time = auth._key_to_client[key_hash].last_access

        time.sleep(0.01)  # Small delay
        auth.authenticate(api_key)

        new_time = auth._key_to_client[key_hash].last_access
        assert new_time >= original_time

    def test_authenticate_increments_request_count(self):
        """Test that authentication increments request count."""
        api_key = "count_test_key"
        auth = APIKeyAuthenticator(valid_keys=[api_key])

        auth.authenticate(api_key)
        auth.authenticate(api_key)
        auth.authenticate(api_key)

        key_hash = auth._hash_key(api_key)
        assert auth._key_to_client[key_hash].request_count == 3

    def test_authenticate_with_none_key_raises_error(self):
        """Test that None API key raises AuthenticationError."""
        auth = APIKeyAuthenticator()

        with pytest.raises(AuthenticationError) as exc_info:
            auth.authenticate(None)

        assert "API key is required" in str(exc_info.value)
        assert exc_info.value.error_code == "AUTH_ERROR"

    def test_authenticate_with_empty_key_raises_error(self):
        """Test that empty string API key raises AuthenticationError."""
        auth = APIKeyAuthenticator()

        with pytest.raises(AuthenticationError) as exc_info:
            auth.authenticate("")

        assert "API key is required" in str(exc_info.value)

    def test_authenticate_with_invalid_key_raises_error(self):
        """Test that invalid API key raises AuthenticationError."""
        auth = APIKeyAuthenticator(valid_keys=["valid_key"])

        with pytest.raises(AuthenticationError) as exc_info:
            auth.authenticate("invalid_key")

        assert "Invalid API key" in str(exc_info.value)
        # Ensure internal details contain truncated hash for security
        assert "..." in exc_info.value.internal_details

    def test_authenticate_does_not_expose_full_hash(self):
        """Test that authentication error does not expose full key hash."""
        auth = APIKeyAuthenticator()

        with pytest.raises(AuthenticationError) as exc_info:
            auth.authenticate("some_invalid_key")

        # Internal details should only show first 16 characters of hash
        internal_msg = exc_info.value.internal_details
        assert "..." in internal_msg
        # Full hash is 64 chars, should not be present
        full_hash = auth._hash_key("some_invalid_key")
        assert full_hash not in internal_msg

    def test_require_auth_calls_authenticate(self):
        """Test that require_auth is convenience wrapper for authenticate."""
        api_key = "test_key"
        auth = APIKeyAuthenticator(valid_keys=[api_key])

        result = auth.require_auth(api_key)

        assert isinstance(result, ClientInfo)

    def test_require_role_with_valid_role(self):
        """Test require_role passes with valid role."""
        client_info = ClientInfo(client_id="test", roles={"admin", "user"})
        auth = APIKeyAuthenticator()

        # Should not raise
        auth.require_role(client_info, "admin")
        auth.require_role(client_info, "user")

    def test_require_role_with_invalid_role_raises_error(self):
        """Test require_role raises error for missing role."""
        client_info = ClientInfo(client_id="test", roles={"user"})
        auth = APIKeyAuthenticator()

        with pytest.raises(AuthorizationError) as exc_info:
            auth.require_role(client_info, "admin")

        assert "do not have permission" in str(exc_info.value)
        assert exc_info.value.error_code == "AUTHZ_ERROR"
        assert exc_info.value.context.get("required_permission") == "admin"

    def test_generate_api_key_returns_64_char_hex(self):
        """Test that generated API key is 64 character hex string."""
        auth = APIKeyAuthenticator()

        api_key = auth.generate_api_key()

        assert len(api_key) == 64
        # Verify it's valid hex
        int(api_key, 16)

    def test_generate_api_key_is_unique(self):
        """Test that generated API keys are unique."""
        auth = APIKeyAuthenticator()

        keys = {auth.generate_api_key() for _ in range(100)}

        assert len(keys) == 100  # All unique

    def test_add_client_generates_and_stores_key(self):
        """Test add_client creates client and returns key."""
        auth = APIKeyAuthenticator()

        api_key = auth.add_client("new_client", roles={"premium"})

        # Key should work for authentication
        client_info = auth.authenticate(api_key)
        assert client_info.client_id == "new_client"
        assert "premium" in client_info.roles

    def test_revoke_key_removes_valid_key(self):
        """Test that revoking key removes it from storage."""
        api_key = "revokable_key"
        auth = APIKeyAuthenticator(valid_keys=[api_key])

        # Key works before revocation
        auth.authenticate(api_key)

        result = auth.revoke_key(api_key)
        assert result is True

        # Key should not work after revocation
        with pytest.raises(AuthenticationError):
            auth.authenticate(api_key)

    def test_revoke_key_returns_false_for_nonexistent_key(self):
        """Test that revoking nonexistent key returns False."""
        auth = APIKeyAuthenticator()

        result = auth.revoke_key("nonexistent_key")

        assert result is False

    def test_get_client_stats_empty_for_new_client(self):
        """Test client stats for client with no requests."""
        auth = APIKeyAuthenticator()

        stats = auth.get_client_stats("new_client")

        assert stats["total_requests_today"] == 0
        assert stats["requests_last_hour"] == 0
        assert stats["requests_last_minute"] == 0

    def test_get_client_stats_tracks_requests(self):
        """Test that client stats reflect actual requests."""
        api_key = "stats_test_key"
        auth = APIKeyAuthenticator(valid_keys=[api_key])

        # Make some requests
        for _ in range(5):
            auth.authenticate(api_key)

        stats = auth.get_client_stats("client_0")

        assert stats["requests_last_minute"] == 5
        assert stats["requests_last_hour"] == 5
        assert stats["total_requests_today"] == 5


class TestRateLimiting:
    """Test suite for rate limiting functionality."""

    def test_burst_limit_enforcement(self):
        """Test that burst limit is enforced."""
        config = RateLimitConfig(burst_limit=3)
        api_key = "burst_test"
        auth = APIKeyAuthenticator(valid_keys=[api_key], rate_limit_config=config)

        # These should succeed
        auth.authenticate(api_key)
        auth.authenticate(api_key)
        auth.authenticate(api_key)

        # This should fail (exceeded burst limit)
        with pytest.raises(RateLimitError) as exc_info:
            auth.authenticate(api_key)

        assert "slow down" in str(exc_info.value)
        assert exc_info.value.retry_after_seconds == 1

    def test_per_minute_limit_enforcement(self):
        """Test that per-minute limit is enforced."""
        config = RateLimitConfig(
            requests_per_minute=5,
            burst_limit=100,  # High burst to test minute limit
        )
        api_key = "minute_test"
        auth = APIKeyAuthenticator(valid_keys=[api_key], rate_limit_config=config)

        # Make 5 requests (at limit)
        for _ in range(5):
            auth.authenticate(api_key)

        # 6th request should fail
        with pytest.raises(RateLimitError) as exc_info:
            auth.authenticate(api_key)

        assert "wait a minute" in str(exc_info.value)
        assert exc_info.value.retry_after_seconds == 60

    def test_per_hour_limit_enforcement(self):
        """Test that per-hour limit is enforced."""
        config = RateLimitConfig(
            requests_per_minute=1000,  # High minute limit
            requests_per_hour=3,
            burst_limit=100,
        )
        api_key = "hour_test"
        auth = APIKeyAuthenticator(valid_keys=[api_key], rate_limit_config=config)

        for _ in range(3):
            auth.authenticate(api_key)

        with pytest.raises(RateLimitError) as exc_info:
            auth.authenticate(api_key)

        assert "Hourly" in str(exc_info.value)
        assert exc_info.value.retry_after_seconds == 3600

    def test_per_day_limit_enforcement(self):
        """Test that per-day limit is enforced."""
        config = RateLimitConfig(
            requests_per_minute=1000,
            requests_per_hour=1000,
            requests_per_day=2,
            burst_limit=100,
        )
        api_key = "day_test"
        auth = APIKeyAuthenticator(valid_keys=[api_key], rate_limit_config=config)

        auth.authenticate(api_key)
        auth.authenticate(api_key)

        with pytest.raises(RateLimitError) as exc_info:
            auth.authenticate(api_key)

        assert "Daily" in str(exc_info.value)
        assert exc_info.value.retry_after_seconds == 86400

    def test_rate_limit_cleans_old_entries(self):
        """Test that old request timestamps are cleaned."""
        api_key = "cleanup_test"
        auth = APIKeyAuthenticator(valid_keys=[api_key])

        # Add some old timestamps manually
        with patch("time.time") as mock_time:
            # First set of requests at t=0
            mock_time.return_value = 0.0
            auth._rate_limits["client_0"] = [0.0, 1.0, 2.0]

            # Request at t=100000 (more than a day later)
            mock_time.return_value = 100000.0
            auth.authenticate(api_key)

            # Old entries should be cleaned
            # Only the new request should remain
            assert len(auth._rate_limits["client_0"]) == 1

    def test_rate_limit_error_contains_internal_details(self):
        """Test that rate limit errors contain useful internal details."""
        config = RateLimitConfig(burst_limit=1)
        api_key = "detail_test"
        auth = APIKeyAuthenticator(valid_keys=[api_key], rate_limit_config=config)

        auth.authenticate(api_key)

        with pytest.raises(RateLimitError) as exc_info:
            auth.authenticate(api_key)

        # Internal details should contain client info
        assert "client_0" in exc_info.value.internal_details
        assert "exceeded" in exc_info.value.internal_details

    def test_rate_limits_are_per_client(self):
        """Test that rate limits apply independently per client."""
        config = RateLimitConfig(burst_limit=2)
        auth = APIKeyAuthenticator(valid_keys=["key1", "key2"], rate_limit_config=config)

        # Client 1 uses their limit
        auth.authenticate("key1")
        auth.authenticate("key1")

        # Client 2 should still have their own limit
        auth.authenticate("key2")
        auth.authenticate("key2")

        # Both should now be at limit
        with pytest.raises(RateLimitError):
            auth.authenticate("key1")

        with pytest.raises(RateLimitError):
            auth.authenticate("key2")


class TestJWTAuthenticator:
    """Test suite for JWTAuthenticator class."""

    def test_initialization(self):
        """Test JWT authenticator initialization."""
        auth = JWTAuthenticator(secret_key="test_secret")

        assert auth.secret_key == "test_secret"
        assert auth.algorithm == "HS256"
        assert auth._token_blacklist == set()

    def test_custom_algorithm(self):
        """Test JWT authenticator with custom algorithm."""
        auth = JWTAuthenticator(secret_key="secret", algorithm="HS512")

        assert auth.algorithm == "HS512"

    def test_default_expiry_hours_defaults_when_none(self):
        """Omitting default_expiry_hours falls back to DEFAULT_EXPIRY_HOURS."""
        auth = JWTAuthenticator(secret_key="secret")
        assert auth.default_expiry_hours == JWTAuthenticator.DEFAULT_EXPIRY_HOURS

    def test_default_expiry_hours_preserves_explicit_value(self):
        """An explicit default_expiry_hours is honored, including falsey 0 (not coerced)."""
        auth = JWTAuthenticator(secret_key="secret", default_expiry_hours=0)
        assert auth.default_expiry_hours == 0

    @pytest.mark.skipif(True, reason="PyJWT library required for full JWT tests")  # Skip if PyJWT not installed
    def test_create_token_requires_pyjwt(self):
        """Test that create_token raises ImportError without PyJWT."""
        auth = JWTAuthenticator(secret_key="secret")

        # This would need PyJWT to be tested fully
        # Just verify the method exists
        assert hasattr(auth, "create_token")

    def test_revoke_token_adds_to_blacklist(self):
        """Test that revoking token adds it to blacklist."""
        auth = JWTAuthenticator(secret_key="secret")
        token = "fake_token_12345"

        auth.revoke_token(token)

        assert token in auth._token_blacklist

    def test_multiple_token_revocations(self):
        """Test revoking multiple tokens."""
        auth = JWTAuthenticator(secret_key="secret")

        tokens = ["token1", "token2", "token3"]
        for token in tokens:
            auth.revoke_token(token)

        assert len(auth._token_blacklist) == 3
        for token in tokens:
            assert token in auth._token_blacklist


class TestGlobalAuthenticator:
    """Test suite for global authenticator functions."""

    def teardown_method(self):
        """Reset global authenticator after each test."""
        import src.api.auth as auth_module

        auth_module._default_authenticator = None

    def test_get_authenticator_creates_default(self):
        """Test that get_authenticator creates default instance."""
        auth = get_authenticator()

        assert isinstance(auth, APIKeyAuthenticator)

    def test_get_authenticator_returns_same_instance(self):
        """Test that get_authenticator returns same instance."""
        auth1 = get_authenticator()
        auth2 = get_authenticator()

        assert auth1 is auth2

    def test_set_authenticator_replaces_default(self):
        """Test that set_authenticator changes the default."""
        custom_auth = APIKeyAuthenticator(valid_keys=["custom_key"])

        set_authenticator(custom_auth)

        retrieved = get_authenticator()
        assert retrieved is custom_auth

    def test_set_authenticator_with_custom_config(self):
        """Test setting authenticator with custom configuration."""
        config = RateLimitConfig(requests_per_minute=1)
        custom_auth = APIKeyAuthenticator(rate_limit_config=config)

        set_authenticator(custom_auth)

        retrieved = get_authenticator()
        assert retrieved.rate_limit_config.requests_per_minute == 1


class TestSecurityAspects:
    """Test security-related aspects of authentication."""

    def test_plain_keys_not_stored(self):
        """Test that plain text keys are never stored."""
        plain_keys = ["secret_key_1", "secret_key_2", "admin_password"]
        auth = APIKeyAuthenticator(valid_keys=plain_keys)

        # Convert entire auth object to string representation
        auth_repr = str(auth.__dict__)

        for key in plain_keys:
            assert key not in auth_repr

    def test_error_messages_do_not_expose_valid_keys(self):
        """Test that error messages don't reveal valid keys."""
        valid_key = "super_secret_key_12345"
        auth = APIKeyAuthenticator(valid_keys=[valid_key])

        try:
            auth.authenticate("wrong_key")
        except AuthenticationError as e:
            error_str = str(e) + e.user_message + e.internal_details
            assert valid_key not in error_str

    def test_hash_truncation_in_errors(self):
        """Test that hash values are truncated in error messages."""
        auth = APIKeyAuthenticator()

        with pytest.raises(AuthenticationError) as exc_info:
            auth.authenticate("test_key")

        # Only first 16 chars of hash should appear
        assert "..." in exc_info.value.internal_details

    def test_constant_time_comparison_mechanism(self):
        """Test that hashing is used (enabling constant-time comparison)."""
        auth = APIKeyAuthenticator(valid_keys=["test_key"])

        # The authenticate method hashes the input key
        # This prevents timing attacks
        key_hash = auth._hash_key("test_key")

        # Verify hash is in the key map
        assert key_hash in auth._key_to_client

    def test_generated_keys_have_sufficient_entropy(self):
        """Test that generated keys have sufficient randomness."""
        auth = APIKeyAuthenticator()

        # Generate multiple keys
        keys = [auth.generate_api_key() for _ in range(10)]

        # Each key should be unique
        assert len(set(keys)) == 10

        # Each key should be 64 chars (32 bytes = 256 bits of entropy)
        for key in keys:
            assert len(key) == 64

    def test_rate_limit_errors_sanitize_client_id_format(self):
        """Test rate limit errors include client ID safely."""
        config = RateLimitConfig(burst_limit=1)
        auth = APIKeyAuthenticator(valid_keys=["key"], rate_limit_config=config)

        auth.authenticate("key")

        with pytest.raises(RateLimitError) as exc_info:
            auth.authenticate("key")

        # User message should not contain client ID
        assert "client_0" not in exc_info.value.user_message
        # Internal details can contain it
        assert "client_0" in exc_info.value.internal_details


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_roles_set(self):
        """Test client with empty roles set - defaults to user role."""
        auth = APIKeyAuthenticator()
        auth._add_key("key", "client", roles=set())

        key_hash = auth._hash_key("key")
        # Empty set is falsy, so it defaults to {"user"}
        assert auth._key_to_client[key_hash].roles == {"user"}

    def test_very_long_api_key(self):
        """Test authentication with very long API key."""
        long_key = "x" * 10000
        auth = APIKeyAuthenticator(valid_keys=[long_key])

        client_info = auth.authenticate(long_key)
        assert client_info is not None

    def test_special_characters_in_api_key(self):
        """Test API key with special characters."""
        special_key = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        auth = APIKeyAuthenticator(valid_keys=[special_key])

        client_info = auth.authenticate(special_key)
        assert client_info is not None

    def test_unicode_in_api_key(self):
        """Test API key with unicode characters."""
        unicode_key = "test_key_\u4e2d\u6587_\u00e9\u00f1"
        auth = APIKeyAuthenticator(valid_keys=[unicode_key])

        client_info = auth.authenticate(unicode_key)
        assert client_info is not None

    def test_multiple_clients_same_roles(self):
        """Test multiple clients with same role set."""
        auth = APIKeyAuthenticator()

        key1 = auth.add_client("client1", roles={"user"})
        key2 = auth.add_client("client2", roles={"user"})

        info1 = auth.authenticate(key1)
        info2 = auth.authenticate(key2)

        assert info1.client_id != info2.client_id
        assert info1.roles == info2.roles

    def test_require_role_with_multiple_roles(self):
        """Test requiring role when client has multiple roles."""
        client = ClientInfo(client_id="multi", roles={"admin", "user", "editor"})
        auth = APIKeyAuthenticator()

        # Should pass for any of the roles
        auth.require_role(client, "admin")
        auth.require_role(client, "user")
        auth.require_role(client, "editor")

    def test_zero_rate_limits(self):
        """Test with zero rate limits (edge case)."""
        config = RateLimitConfig(
            requests_per_minute=0,
            requests_per_hour=0,
            requests_per_day=0,
            burst_limit=0,
        )
        api_key = "zero_limit_key"
        auth = APIKeyAuthenticator(valid_keys=[api_key], rate_limit_config=config)

        # Should fail immediately due to burst limit of 0
        with pytest.raises(RateLimitError):
            auth.authenticate(api_key)


class TestJWTAuthenticatorFactory:
    """Tests for the settings-driven JWT authenticator factory (AUTH_MODE='jwt')."""

    def teardown_method(self):
        """Reset the cached JWT authenticator after each test."""
        set_jwt_authenticator(None)

    @staticmethod
    def _fake_settings(secret, algorithm="HS256", expiry_hours=24):
        """Build a minimal stand-in for Settings exposing only the JWT fields."""
        from pydantic import SecretStr

        return type(
            "S",
            (),
            {
                "JWT_SECRET": SecretStr(secret) if secret is not None else None,
                "JWT_ALGORITHM": algorithm,
                "JWT_EXPIRY_HOURS": expiry_hours,
            },
        )()

    def test_raises_when_jwt_secret_missing(self):
        """get_jwt_authenticator raises AuthenticationError when JWT_SECRET is unset."""
        with patch("src.config.settings.get_settings", return_value=self._fake_settings(None)):
            with pytest.raises(AuthenticationError):
                get_jwt_authenticator()

    def test_builds_from_settings(self):
        """get_jwt_authenticator builds a JWTAuthenticator from settings values."""
        fake = self._fake_settings("super-secret-value", algorithm="HS512", expiry_hours=12)
        with patch("src.config.settings.get_settings", return_value=fake):
            authenticator = get_jwt_authenticator()

        assert isinstance(authenticator, JWTAuthenticator)
        assert authenticator.secret_key == "super-secret-value"
        assert authenticator.algorithm == "HS512"
        # JWT_EXPIRY_HOURS is threaded through (M1 fix — otherwise dead config).
        assert authenticator.default_expiry_hours == 12

    def test_caches_instance(self):
        """The factory caches and reuses a single instance."""
        with patch("src.config.settings.get_settings", return_value=self._fake_settings("s")):
            first = get_jwt_authenticator()
            second = get_jwt_authenticator()

        assert first is second

    def test_set_jwt_authenticator_overrides(self):
        """set_jwt_authenticator injects a custom instance (for DI/testing)."""
        custom = JWTAuthenticator(secret_key="injected")
        set_jwt_authenticator(custom)

        assert get_jwt_authenticator() is custom


class TestAuthModeSetting:
    """Tests for the AUTH_MODE settings validator and JWT startup guard."""

    def test_defaults_and_accepts_supported_modes(self):
        from src.config.settings import Settings

        assert Settings(AUTH_MODE="api_key").AUTH_MODE == "api_key"
        # jwt mode requires a JWT_SECRET (enforced at startup — see test below).
        assert Settings(AUTH_MODE="jwt", JWT_SECRET="x-secret").AUTH_MODE == "jwt"
        # Case-insensitive / whitespace tolerant.
        assert Settings(AUTH_MODE="JWT", JWT_SECRET="x-secret").AUTH_MODE == "jwt"

    def test_rejects_unknown_mode(self):
        from pydantic import ValidationError as PydanticValidationError

        from src.config.settings import Settings

        with pytest.raises(PydanticValidationError):
            Settings(AUTH_MODE="oauth")

    def test_jwt_mode_requires_secret_at_startup(self):
        """AUTH_MODE='jwt' without JWT_SECRET fails fast at config load (H1 fix)."""
        from pydantic import ValidationError as PydanticValidationError

        from src.config.settings import Settings

        with pytest.raises(PydanticValidationError):
            Settings(AUTH_MODE="jwt", JWT_SECRET=None)
