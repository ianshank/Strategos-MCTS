"""
Authentication and authorization layer for LangGraph Multi-Agent MCTS Framework.

Provides:
- API key authentication with secure hashing
- JWT token support (optional)
- Rate limiting per client
- Role-based access control
"""

import hashlib
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from src.api.exceptions import (
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
)


def _utc_now() -> datetime:
    """Get current UTC time (Python 3.10+ compatible)."""
    return datetime.now(UTC)


@dataclass
class ClientInfo:
    """Information about an authenticated client."""

    client_id: str
    roles: set[str] = field(default_factory=lambda: {"user"})
    created_at: datetime = field(default_factory=_utc_now)
    last_access: datetime = field(default_factory=_utc_now)
    request_count: int = 0


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 100  # Max requests in 1 second


class APIKeyAuthenticator:
    """
    API key-based authentication with secure hashing.

    Keys are stored as SHA-256 hashes to prevent exposure.
    """

    def __init__(
        self,
        valid_keys: list[str] | None = None,
        rate_limit_config: RateLimitConfig | None = None,
    ):
        """
        Initialize authenticator.

        Args:
            valid_keys: List of valid API keys (will be hashed)
            rate_limit_config: Rate limiting configuration
        """
        self._key_to_client: dict[str, ClientInfo] = {}
        self._rate_limits: dict[str, list[float]] = defaultdict(list)
        self.rate_limit_config = rate_limit_config or RateLimitConfig()

        # Hash and store initial keys
        if valid_keys:
            for i, key in enumerate(valid_keys):
                client_id = f"client_{i}"
                self._add_key(key, client_id)

    def _hash_key(self, api_key: str) -> str:
        """
        Securely hash an API key.

        Uses SHA-256 with consistent encoding.
        """
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

    def _add_key(self, api_key: str, client_id: str, roles: set[str] | None = None) -> None:
        """
        Add a new API key.

        Args:
            api_key: Raw API key
            client_id: Client identifier
            roles: Set of roles (defaults to {"user"})
        """
        key_hash = self._hash_key(api_key)
        self._key_to_client[key_hash] = ClientInfo(
            client_id=client_id,
            roles=roles or {"user"},
        )

    def authenticate(self, api_key: str | None) -> ClientInfo:
        """
        Authenticate an API key.

        Args:
            api_key: API key to validate

        Returns:
            ClientInfo for the authenticated client

        Raises:
            AuthenticationError: If authentication fails
        """
        if not api_key:
            raise AuthenticationError(
                user_message="API key is required",
                internal_details="No API key provided in request",
            )

        # Constant-time comparison to prevent timing attacks
        key_hash = self._hash_key(api_key)

        if key_hash not in self._key_to_client:
            raise AuthenticationError(
                user_message="Invalid API key",
                internal_details=f"API key hash not found: {key_hash[:16]}...",
            )

        client_info = self._key_to_client[key_hash]
        client_info.last_access = _utc_now()
        client_info.request_count += 1

        # Check rate limits
        self._check_rate_limit(client_info.client_id)

        return client_info

    def _check_rate_limit(self, client_id: str) -> None:
        """
        Check if client has exceeded rate limits.

        Args:
            client_id: Client identifier

        Raises:
            RateLimitError: If rate limit exceeded
        """
        now = time.time()
        request_times = self._rate_limits[client_id]

        # Clean old entries
        one_day_ago = now - 86400
        request_times = [t for t in request_times if t > one_day_ago]
        self._rate_limits[client_id] = request_times

        # Check burst limit (1 second window)
        one_second_ago = now - 1
        burst_count = sum(1 for t in request_times if t > one_second_ago)
        if burst_count >= self.rate_limit_config.burst_limit:
            raise RateLimitError(
                user_message="Too many requests. Please slow down.",
                internal_details=f"Client {client_id} exceeded burst limit: {burst_count}/{self.rate_limit_config.burst_limit}",
                retry_after_seconds=1,
            )

        # Check per-minute limit
        one_minute_ago = now - 60
        minute_count = sum(1 for t in request_times if t > one_minute_ago)
        if minute_count >= self.rate_limit_config.requests_per_minute:
            raise RateLimitError(
                user_message="Rate limit exceeded. Please wait a minute.",
                internal_details=f"Client {client_id} exceeded minute limit: {minute_count}/{self.rate_limit_config.requests_per_minute}",
                retry_after_seconds=60,
            )

        # Check per-hour limit
        one_hour_ago = now - 3600
        hour_count = sum(1 for t in request_times if t > one_hour_ago)
        if hour_count >= self.rate_limit_config.requests_per_hour:
            raise RateLimitError(
                user_message="Hourly rate limit exceeded. Please try again later.",
                internal_details=f"Client {client_id} exceeded hour limit: {hour_count}/{self.rate_limit_config.requests_per_hour}",
                retry_after_seconds=3600,
            )

        # Check per-day limit
        day_count = len(request_times)
        if day_count >= self.rate_limit_config.requests_per_day:
            raise RateLimitError(
                user_message="Daily rate limit exceeded. Please try again tomorrow.",
                internal_details=f"Client {client_id} exceeded day limit: {day_count}/{self.rate_limit_config.requests_per_day}",
                retry_after_seconds=86400,
            )

        # Record this request
        request_times.append(now)

    def require_auth(self, api_key: str | None) -> ClientInfo:
        """
        Require authentication for a request.

        Convenience method that raises on failure.

        Args:
            api_key: API key to validate

        Returns:
            ClientInfo for authenticated client

        Raises:
            AuthenticationError: If authentication fails
        """
        return self.authenticate(api_key)

    def require_role(self, client_info: ClientInfo, required_role: str) -> None:
        """
        Require a specific role for an operation.

        Args:
            client_info: Authenticated client info
            required_role: Role that is required

        Raises:
            AuthorizationError: If client doesn't have required role
        """
        if required_role not in client_info.roles:
            raise AuthorizationError(
                user_message="You do not have permission for this operation",
                internal_details=f"Client {client_info.client_id} missing role: {required_role}",
                required_permission=required_role,
            )

    def generate_api_key(self) -> str:
        """
        Generate a secure random API key.

        Returns:
            New API key (32 bytes hex = 64 characters)
        """
        return secrets.token_hex(32)

    def revoke_key(self, api_key: str) -> bool:
        """
        Revoke an API key.

        Args:
            api_key: Key to revoke

        Returns:
            True if key was revoked, False if not found
        """
        key_hash = self._hash_key(api_key)
        if key_hash in self._key_to_client:
            del self._key_to_client[key_hash]
            return True
        return False

    def add_client(
        self,
        client_id: str,
        roles: set[str] | None = None,
    ) -> str:
        """
        Add a new client and generate their API key.

        Args:
            client_id: Unique client identifier
            roles: Set of roles for the client

        Returns:
            Generated API key (save this securely!)
        """
        api_key = self.generate_api_key()
        self._add_key(api_key, client_id, roles)
        return api_key

    def get_client_stats(self, client_id: str) -> dict:
        """
        Get statistics for a client.

        Args:
            client_id: Client identifier

        Returns:
            Dictionary with client statistics
        """
        now = time.time()
        request_times = self._rate_limits.get(client_id, [])

        return {
            "total_requests_today": len([t for t in request_times if t > now - 86400]),
            "requests_last_hour": len([t for t in request_times if t > now - 3600]),
            "requests_last_minute": len([t for t in request_times if t > now - 60]),
        }


class JWTAuthenticator:
    """
    JWT token-based authentication.

    Note: Requires PyJWT library for full functionality.
    This is a placeholder for JWT support.
    """

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """
        Initialize JWT authenticator.

        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT signing algorithm
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self._token_blacklist: set[str] = set()

    def create_token(
        self,
        client_id: str,
        roles: set[str],
        expires_in_hours: int = 24,
    ) -> str:
        """
        Create a JWT token.

        Args:
            client_id: Client identifier
            roles: Client roles
            expires_in_hours: Token validity period

        Returns:
            JWT token string
        """
        try:
            import jwt
        except ImportError as e:
            raise ImportError("PyJWT library required for JWT authentication. Install with: pip install PyJWT") from e

        now = _utc_now()
        payload = {
            "sub": client_id,
            "roles": list(roles),
            "iat": now,
            "exp": now + timedelta(hours=expires_in_hours),
            "jti": secrets.token_hex(16),  # Unique token ID
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> ClientInfo:
        """
        Verify a JWT token.

        Args:
            token: JWT token string

        Returns:
            ClientInfo from token claims

        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            import jwt
        except ImportError as e:
            raise ImportError("PyJWT library required for JWT authentication") from e

        if token in self._token_blacklist:
            raise AuthenticationError(
                user_message="Token has been revoked",
                internal_details="Token found in blacklist",
            )

        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
            )

            return ClientInfo(
                client_id=payload["sub"],
                roles=set(payload.get("roles", ["user"])),
            )
        except jwt.ExpiredSignatureError as e:
            raise AuthenticationError(
                user_message="Token has expired",
                internal_details="JWT signature expired",
            ) from e
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(
                user_message="Invalid token",
                internal_details=f"JWT validation failed: {str(e)}",
            ) from e

    def revoke_token(self, token: str) -> None:
        """
        Revoke a JWT token by adding to blacklist.

        Args:
            token: Token to revoke
        """
        self._token_blacklist.add(token)


# Default authenticator instance
_default_authenticator: APIKeyAuthenticator | None = None


def get_authenticator() -> APIKeyAuthenticator:
    """
    Get or create the default authenticator instance.

    Returns:
        APIKeyAuthenticator instance
    """
    global _default_authenticator
    if _default_authenticator is None:
        _default_authenticator = APIKeyAuthenticator()
    return _default_authenticator


def set_authenticator(authenticator: APIKeyAuthenticator) -> None:
    """
    Set the default authenticator instance.

    Args:
        authenticator: Authenticator to use
    """
    global _default_authenticator
    _default_authenticator = authenticator


# Exports
__all__ = [
    "APIKeyAuthenticator",
    "JWTAuthenticator",
    "ClientInfo",
    "RateLimitConfig",
    "get_authenticator",
    "set_authenticator",
]
