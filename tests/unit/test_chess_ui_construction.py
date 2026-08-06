"""
Construction and scoring tests for the chess Gradio UI.

Two defects motivate this file, and both survived on ``main`` behind 66 passing
tests in ``tests/unit/test_chess_ui.py``:

1. ``create_chess_ui()`` raised ``TypeError`` at ``ui.py:1125`` — a hard crash
   before a socket opened — because no test in the repository ever built the
   Blocks graph. Every existing test exercised pure helper functions.
2. ``record_game_result("AI wins by checkmate!")`` credited the win to the human,
   because a ``"checkmate"`` substring test shadowed the ``"AI wins"`` branch.
"""

from __future__ import annotations

import pytest

pytest.importorskip("chess", reason="requires the [chess] extra")
pytest.importorskip("gradio", reason="requires the [ui] extra")

from src.games.chess.continuous_learning import GameResult  # noqa: E402
from src.games.chess.ui import GameSession, create_chess_ui  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.ui]


class TestUIConstruction:
    """The test whose absence let a launch-blocking TypeError reach main."""

    def test_create_chess_ui_builds_a_blocks_graph(self) -> None:
        import gradio as gr

        demo = create_chess_ui()

        assert isinstance(demo, gr.Blocks)

    def test_building_twice_is_safe(self) -> None:
        """The launcher and tests may each build; neither may poison the other."""
        assert create_chess_ui() is not create_chess_ui()


class TestGameResultAttribution:
    """A win must be credited to whoever actually won."""

    @pytest.mark.parametrize(
        ("player_color", "result_text", "expected"),
        [
            ("white", "AI wins by checkmate!", GameResult.BLACK_WIN),
            ("black", "AI wins by checkmate!", GameResult.WHITE_WIN),
            ("white", "You win by checkmate!", GameResult.WHITE_WIN),
            ("black", "You win by checkmate!", GameResult.BLACK_WIN),
            ("white", "Draw by stalemate", GameResult.DRAW),
            ("white", "Draw", GameResult.DRAW),
        ],
    )
    def test_result_is_attributed_to_the_actual_winner(
        self, player_color: str, result_text: str, expected: GameResult
    ) -> None:
        session = GameSession(player_color=player_color)

        session.record_game_result(result_text)

        assert session.scorecard.total_games == 1
        if expected is GameResult.WHITE_WIN:
            assert session.scorecard.white_wins == 1
            assert session.scorecard.black_wins == 0
        elif expected is GameResult.BLACK_WIN:
            assert session.scorecard.black_wins == 1
            assert session.scorecard.white_wins == 0
        else:
            assert session.scorecard.white_wins == 0
            assert session.scorecard.black_wins == 0

    def test_ai_checkmate_is_not_credited_to_the_human(self) -> None:
        """The exact regression: 'checkmate' must not imply a player win."""
        session = GameSession(player_color="white")

        session.record_game_result("AI wins by checkmate!")

        assert session.scorecard.white_wins == 0, "AI's checkmate was credited to the human player"
        assert session.scorecard.black_wins == 1

    def test_attribution_is_case_insensitive(self) -> None:
        session = GameSession(player_color="white")

        session.record_game_result("ai WINS by Checkmate!")

        assert session.scorecard.black_wins == 1
