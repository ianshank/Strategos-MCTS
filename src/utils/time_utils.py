"""Shared time utilities."""

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Get current UTC time (Python 3.10+ compatible)."""
    return datetime.now(UTC)
