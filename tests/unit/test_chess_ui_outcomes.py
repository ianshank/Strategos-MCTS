"""
Game-over outcome paths in ``src/games/chess/ui.py``.

``GameSession.record_game_result`` previously credited the human with every
checkmate, because it matched the substring ``"checkmate"`` — which appears in
both produced strings ("You win by checkmate!" and "AI wins by checkmate!").
The helper itself is now tested directly, but nothing exercised the *handlers*
that build those strings and call it, so the attribution could silently regress
again at the call site while the helper's own tests stayed green.

These tests drive the two handlers end to end from real positions and assert on
the scorecard, which is what a player actually sees. They also cover the
stalemate and draw arms, which no test reached before.

Positions are minimal and verified legal; each is annotated with the outcome it
produces so a future reader need not replay it mentally.

Contributes to ui_test_coverage AC-6 (the module must be measured by a gate): these
tests carry ``src/games/chess/ui.py`` from 86.41% to 91.48% branch coverage, the
margin that lets ``.coveragerc.ui`` gate at the repo's 85% rather than below it. The
win-attribution behaviour they pin has no AC of its own — the fix shipped without
one, which is part of why the call sites went untested.
"""

from __future__ import annotations

import pytest

chess = pytest.importorskip("chess", reason="python-chess not installed")

from src.games.chess import ui as ui_mod  # noqa: E402

pytestmark = pytest.mark.unit


# Verified legal; the listed move ends the game in the named way.
PLAYER_MATES = ("6k1/5ppp/8/8/8/8/8/R3K3 w Q - 0 1", "a1a8")
PLAYER_STALEMATES = ("k7/8/8/8/8/8/8/K1Q5 w - - 0 1", "c1c7")
PLAYER_DRAWS_INSUFFICIENT = ("k7/8/8/4b3/8/2B5/8/K7 w - - 0 1", "c3e5")
AI_MATES = ("r3k3/8/8/8/8/8/5PPP/6K1 b q - 0 1", "a8a1")
AI_STALEMATES = ("7k/8/8/8/8/8/1q6/7K b - - 0 1", "b2f2")
ALREADY_STALEMATED = "7k/8/8/8/8/8/5q2/7K w - - 1 2"


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch):
    """Give each test a private session so the module global cannot leak between them."""
    fresh = ui_mod.GameSession()
    monkeypatch.setattr(ui_mod, "_session", fresh)
    return fresh


def _play(session, position: tuple[str, str], player_color: str = "white"):
    """Set the session to ``position`` and apply the player's move."""
    fen, move = position
    session.reset(player_color)
    session.fen = fen
    return ui_mod.apply_player_move(move)


class TestPlayerMoveEndsGame:
    """The player's own move can end the game three different ways."""

    def test_checkmate_credits_the_player(self, session) -> None:
        """A mate delivered by the human is a win for the human's colour."""
        _play(session, PLAYER_MATES)

        assert session.game_over is True
        assert session.result == "You win by checkmate!"
        assert session.scorecard.white_wins == 1
        assert session.scorecard.black_wins == 0

    def test_checkmate_as_black_credits_black(self, session) -> None:
        """Attribution follows the player's colour, not a fixed side."""
        _play(session, PLAYER_MATES, player_color="black")

        assert session.scorecard.black_wins == 1
        assert session.scorecard.white_wins == 0

    def test_stalemate_is_a_draw(self, session) -> None:
        """Stalemate must not be recorded as a win for whoever moved last."""
        _play(session, PLAYER_STALEMATES)

        assert session.result == "Draw by stalemate"
        assert session.scorecard.draws == 1
        assert session.scorecard.white_wins == 0

    def test_insufficient_material_is_a_draw(self, session) -> None:
        """Game-over that is neither mate nor stalemate falls through to Draw."""
        _play(session, PLAYER_DRAWS_INSUFFICIENT)

        assert session.result == "Draw"
        assert session.scorecard.draws == 1


class TestAiMoveEndsGame:
    """The regression's real home: an AI mate must never be credited to the human."""

    def _ai_plays(self, monkeypatch, session, position, player_color="white"):
        """Point the session at ``position`` and force the AI to play its move."""
        fen, move = position
        session.reset(player_color)
        session.fen = fen
        monkeypatch.setattr(ui_mod, "get_ai_move", lambda _fen: (move, {}))
        return ui_mod.make_ai_move_sync()

    def test_ai_checkmate_credits_the_ai(self, monkeypatch, session) -> None:
        """The exact bug: an AI mate used to be recorded as a player win."""
        self._ai_plays(monkeypatch, session, AI_MATES)

        assert session.result == "AI wins by checkmate!"
        assert session.scorecard.black_wins == 1, "AI plays black here; the win is black's"
        assert session.scorecard.white_wins == 0

    def test_ai_checkmate_against_a_black_player_credits_white(self, monkeypatch, session) -> None:
        """With the player as black the AI is white, so the win flips sides."""
        self._ai_plays(monkeypatch, session, AI_MATES, player_color="black")

        assert session.scorecard.white_wins == 1
        assert session.scorecard.black_wins == 0

    def test_ai_stalemate_is_a_draw(self, monkeypatch, session) -> None:
        """A stalemate reached by the AI's move is a draw, not an AI win."""
        self._ai_plays(monkeypatch, session, AI_STALEMATES)

        assert session.result == "Draw by stalemate"
        assert session.scorecard.draws == 1

    def test_game_already_over_on_entry_is_recorded_once(self, monkeypatch, session) -> None:
        """Entering with a finished board short-circuits before any AI move is asked for."""
        session.reset("white")
        session.fen = ALREADY_STALEMATED

        def _must_not_be_called(_fen):
            raise AssertionError("get_ai_move must not be consulted on a finished board")

        monkeypatch.setattr(ui_mod, "get_ai_move", _must_not_be_called)

        ui_mod.make_ai_move_sync()

        assert session.game_over is True
        assert session.result == "Draw"
        assert session.scorecard.total_games == 1


class TestContinuousLearningLifecycle:
    """Start/stop of the learning session, whose stop path is owned by the session object."""

    @pytest.fixture(autouse=True)
    def _isolate_learning_global(self, monkeypatch: pytest.MonkeyPatch):
        """Never let a test start or observe a real background learning thread."""
        monkeypatch.setattr(ui_mod, "_learning_session", None)

    def test_stop_without_a_session_is_reported_not_raised(self) -> None:
        """Stopping when nothing runs must be a message, not an exception."""
        message, _status = ui_mod.stop_continuous_learning()

        assert "No learning session running" in message

    def test_stop_delegates_to_the_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stopping is cooperative: the session clears its own is_running flag."""
        stopped: list[bool] = []

        class _Running:
            is_running = True

            def stop(self) -> None:
                stopped.append(True)

        monkeypatch.setattr(ui_mod, "_learning_session", _Running())
        monkeypatch.setattr(ui_mod, "render_learning_status", lambda: "")

        message, _status = ui_mod.stop_continuous_learning()

        assert stopped == [True], "stop() on the session is the only stop mechanism"
        assert "Stopping" in message

    def test_start_refuses_to_double_start(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A second start must not replace a live session or spawn a second thread."""

        class _Running:
            is_running = True

        running = _Running()
        monkeypatch.setattr(ui_mod, "_learning_session", running)
        monkeypatch.setattr(ui_mod, "render_learning_status", lambda: "")

        message, _status = ui_mod.start_continuous_learning(10, 50)

        assert "already running" in message.lower()
        assert ui_mod._learning_session is running, "the live session must survive untouched"
