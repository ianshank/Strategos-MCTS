"""Tests for the lazy chess-domain registration (optional ``chess`` extra).

Two families:
- Behavior when python-chess is absent (registration no-ops, KeyError mentions the
  optional domain) — runs in the main CI job, which does not install the extra.
- Behavior when python-chess is installed (spec fields, lazy DomainRegistry.get path,
  initial legal moves) — importorskip-guarded, exercised for real by the chess CI job.
"""

from __future__ import annotations

import importlib.util

import pytest

from src.framework.domain_registry import METRIC_WIN_RATE, DomainRegistry
from src.games.chess.registration import CHESS_DOMAIN, chess_available, register_chess_domain

pytestmark = [pytest.mark.unit]

_CHESS_INSTALLED = importlib.util.find_spec("chess") is not None


@pytest.fixture()
def unregistered_chess():
    """Remove chess from the registry (if present) and restore afterward."""
    saved = DomainRegistry._registry.pop(CHESS_DOMAIN, None)
    yield
    if saved is not None:
        DomainRegistry._registry[CHESS_DOMAIN] = saved
    else:
        DomainRegistry._registry.pop(CHESS_DOMAIN, None)


def test_chess_available_matches_import_reality():
    assert chess_available() is _CHESS_INSTALLED


@pytest.mark.skipif(_CHESS_INSTALLED, reason="requires python-chess to be ABSENT")
class TestWithoutPythonChess:
    def test_registration_is_a_noop(self, unregistered_chess):
        assert register_chess_domain() is False
        assert CHESS_DOMAIN not in DomainRegistry.list_domains()

    def test_get_raises_keyerror_mentioning_optional_domain(self, unregistered_chess):
        with pytest.raises(KeyError, match="optional.*chess"):
            DomainRegistry.get(CHESS_DOMAIN)

    def test_unknown_domain_error_lists_chess_as_optional(self, unregistered_chess):
        with pytest.raises(KeyError, match="optional.*chess"):
            DomainRegistry.get("no_such_domain")


class TestWithPythonChess:
    @pytest.fixture(autouse=True)
    def _require_chess(self):
        pytest.importorskip("chess", reason="python-chess required (install the 'chess' extra)")

    def test_registration_succeeds_with_expected_spec(self):
        assert register_chess_domain() is True
        spec = DomainRegistry.get(CHESS_DOMAIN)
        assert spec.action_space_size == 4672  # 73 move planes * 64 squares
        assert spec.single_agent is False
        assert spec.metric == METRIC_WIN_RATE

    def test_registration_is_idempotent(self):
        assert register_chess_domain() is True
        assert register_chess_domain() is True
        assert DomainRegistry.list_domains().count(CHESS_DOMAIN) == 1

    def test_lazy_get_registers_on_miss(self, unregistered_chess):
        assert CHESS_DOMAIN not in DomainRegistry.list_domains()
        spec = DomainRegistry.get(CHESS_DOMAIN)  # triggers the lazy loader
        assert spec.name == CHESS_DOMAIN
        assert CHESS_DOMAIN in DomainRegistry.list_domains()

    def test_initial_state_has_twenty_legal_moves(self):
        register_chess_domain()
        state = DomainRegistry.get_initial_state(CHESS_DOMAIN)
        actions = state.get_legal_actions()
        assert len(actions) == 20  # 16 pawn moves + 4 knight moves
        assert all(isinstance(action, str) for action in actions)  # UCI strings are hashable
        assert state.is_terminal() is False
