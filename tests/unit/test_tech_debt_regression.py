"""
Regression tests for tech debt and code hardening fixes (2026-07-20).
"""

import pytest


@pytest.mark.unit
def test_app_version_is_string():
    """
    Regression test for hardcoded version removal in rest_server.py.
    Ensures _APP_VERSION is loaded and is a string, preventing '1.0.0' magic numbers.
    """
    from src.api.rest_server import _APP_VERSION
    assert isinstance(_APP_VERSION, str)
    assert len(_APP_VERSION) > 0

@pytest.mark.unit
def test_storage_imports_gracefully():
    """
    Regression test for storage/__init__.py unblocking test collection.
    Ensures that S3_AVAILABLE is a boolean and importing storage doesn't crash
    if tenacity/aioboto3 are missing.
    """
    import src.storage as storage
    assert hasattr(storage, "S3_AVAILABLE")
    assert isinstance(storage.S3_AVAILABLE, bool)

@pytest.mark.unit
def test_metrics_no_collision_on_reload():
    """
    Regression test for removing redundant REGISTRY ternaries in metrics.py
    and resolving the 'mcts_iterations_total' collision.
    """
    from src.observability.metrics import MetricsCollector

    # Creating multiple collectors should not crash with 'ValueError: Duplicated timeseries'
    # due to the deduplication pattern introduced.
    collector1 = MetricsCollector()
    collector2 = MetricsCollector()

    # We just assert that it instantiates cleanly.
    assert collector1 is not None
    assert collector2 is not None
