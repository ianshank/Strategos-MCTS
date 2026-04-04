"""
Unit tests for chess verification type definitions.

Tests enums, dataclasses, properties, and serialization.
"""

from __future__ import annotations

import pytest

from src.games.chess.verification.types import (
    BatchVerificationResult,
    EnsembleConsistencyResult,
    GameResult,
    GameVerificationResult,
    MoveSequenceResult,
    MoveType,
    MoveValidationResult,
    PositionVerificationResult,
    VerificationIssue,
    VerificationSeverity,
)

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
]


# =============================================================================
# Enum Tests
# =============================================================================


class TestMoveType:
    """Tests for MoveType enum."""

    def test_all_move_types_defined(self) -> None:
        """Test all expected move types exist."""
        expected = [
            "NORMAL",
            "CAPTURE",
            "CASTLE_KINGSIDE",
            "CASTLE_QUEENSIDE",
            "EN_PASSANT",
            "PROMOTION",
            "PROMOTION_CAPTURE",
            "CHECK",
            "CHECKMATE",
        ]
        for name in expected:
            assert hasattr(MoveType, name)

    def test_move_type_values(self) -> None:
        """Test MoveType enum values are lowercase strings."""
        assert MoveType.NORMAL.value == "normal"
        assert MoveType.CASTLE_KINGSIDE.value == "castle_kingside"
        assert MoveType.PROMOTION_CAPTURE.value == "promotion_capture"

    def test_move_type_count(self) -> None:
        """Test expected number of move types."""
        assert len(MoveType) == 9


class TestGameResult:
    """Tests for GameResult enum."""

    def test_all_results_defined(self) -> None:
        """Test all expected game results exist."""
        expected = [
            "WHITE_WINS",
            "BLACK_WINS",
            "DRAW_STALEMATE",
            "DRAW_INSUFFICIENT_MATERIAL",
            "DRAW_FIFTY_MOVES",
            "DRAW_THREEFOLD_REPETITION",
            "DRAW_AGREEMENT",
            "IN_PROGRESS",
        ]
        for name in expected:
            assert hasattr(GameResult, name)

    def test_game_result_values(self) -> None:
        """Test GameResult enum values."""
        assert GameResult.WHITE_WINS.value == "white_wins"
        assert GameResult.IN_PROGRESS.value == "in_progress"

    def test_game_result_count(self) -> None:
        """Test expected number of game results."""
        assert len(GameResult) == 8


class TestVerificationSeverity:
    """Tests for VerificationSeverity enum."""

    def test_all_severities_defined(self) -> None:
        """Test all severity levels exist."""
        assert VerificationSeverity.INFO.value == "info"
        assert VerificationSeverity.WARNING.value == "warning"
        assert VerificationSeverity.ERROR.value == "error"
        assert VerificationSeverity.CRITICAL.value == "critical"

    def test_severity_count(self) -> None:
        """Test expected number of severity levels."""
        assert len(VerificationSeverity) == 4


# =============================================================================
# VerificationIssue Tests
# =============================================================================


class TestVerificationIssue:
    """Tests for VerificationIssue dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic issue creation."""
        issue = VerificationIssue(
            code="INVALID_MOVE",
            message="Move is not legal",
            severity=VerificationSeverity.ERROR,
        )
        assert issue.code == "INVALID_MOVE"
        assert issue.message == "Move is not legal"
        assert issue.severity == VerificationSeverity.ERROR

    def test_default_fields(self) -> None:
        """Test default field values."""
        issue = VerificationIssue(code="TEST", message="test", severity=VerificationSeverity.INFO)
        assert issue.context == {}
        assert issue.move_number is None
        assert issue.fen is None

    def test_with_optional_fields(self) -> None:
        """Test creation with optional fields."""
        issue = VerificationIssue(
            code="BAD_POSITION",
            message="Invalid position",
            severity=VerificationSeverity.CRITICAL,
            context={"detail": "missing king"},
            move_number=15,
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        )
        assert issue.context["detail"] == "missing king"
        assert issue.move_number == 15
        assert issue.fen is not None

    def test_str_representation_with_move(self) -> None:
        """Test string representation includes move number."""
        issue = VerificationIssue(
            code="ILLEGAL", message="Illegal move", severity=VerificationSeverity.ERROR, move_number=10
        )
        result = str(issue)
        assert "[ERROR]" in result
        assert "ILLEGAL" in result
        assert "(move 10)" in result

    def test_str_representation_without_move(self) -> None:
        """Test string representation without move number."""
        issue = VerificationIssue(code="WARN", message="Warning", severity=VerificationSeverity.WARNING)
        result = str(issue)
        assert "[WARNING]" in result
        assert "WARN" in result
        assert "move" not in result


# =============================================================================
# MoveValidationResult Tests
# =============================================================================


class TestMoveValidationResult:
    """Tests for MoveValidationResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic result creation."""
        result = MoveValidationResult(
            is_valid=True,
            move_uci="e2e4",
            move_type=MoveType.NORMAL,
        )
        assert result.is_valid is True
        assert result.move_uci == "e2e4"
        assert result.move_type == MoveType.NORMAL

    def test_default_fields(self) -> None:
        """Test default field values."""
        result = MoveValidationResult(is_valid=True, move_uci="e2e4", move_type=MoveType.NORMAL)
        assert result.encoded_index is None
        assert result.issues == []
        assert result.extra_info == {}
        assert result.from_square is None
        assert result.to_square is None
        assert result.piece_moved is None
        assert result.piece_captured is None
        assert result.promotion_piece is None
        assert result.is_check is False
        assert result.is_checkmate is False
        assert result.is_legal_in_position is True

    def test_has_errors_with_error_issue(self) -> None:
        """Test has_errors is True when there is an error issue."""
        result = MoveValidationResult(
            is_valid=False,
            move_uci="e2e5",
            move_type=MoveType.NORMAL,
            issues=[
                VerificationIssue(code="ERR", message="Error", severity=VerificationSeverity.ERROR),
            ],
        )
        assert result.has_errors is True

    def test_has_errors_with_critical_issue(self) -> None:
        """Test has_errors is True for critical issue."""
        result = MoveValidationResult(
            is_valid=False,
            move_uci="e2e5",
            move_type=MoveType.NORMAL,
            issues=[
                VerificationIssue(code="CRIT", message="Critical", severity=VerificationSeverity.CRITICAL),
            ],
        )
        assert result.has_errors is True

    def test_has_errors_with_only_warnings(self) -> None:
        """Test has_errors is False when only warnings exist."""
        result = MoveValidationResult(
            is_valid=True,
            move_uci="e2e4",
            move_type=MoveType.NORMAL,
            issues=[
                VerificationIssue(code="WARN", message="Warning", severity=VerificationSeverity.WARNING),
            ],
        )
        assert result.has_errors is False

    def test_has_errors_no_issues(self) -> None:
        """Test has_errors is False when no issues."""
        result = MoveValidationResult(is_valid=True, move_uci="e2e4", move_type=MoveType.NORMAL)
        assert result.has_errors is False

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        result = MoveValidationResult(
            is_valid=True,
            move_uci="e2e4",
            move_type=MoveType.NORMAL,
            from_square="e2",
            to_square="e4",
            piece_moved="P",
            is_check=False,
            is_checkmate=False,
            issues=[
                VerificationIssue(code="INFO", message="Info", severity=VerificationSeverity.INFO),
            ],
        )
        d = result.to_dict()

        assert d["is_valid"] is True
        assert d["move_uci"] == "e2e4"
        assert d["move_type"] == "normal"
        assert d["from_square"] == "e2"
        assert d["to_square"] == "e4"
        assert d["piece_moved"] == "P"
        assert d["is_check"] is False
        assert d["is_checkmate"] is False
        assert len(d["issues"]) == 1
        assert d["issues"][0]["code"] == "INFO"
        assert d["issues"][0]["severity"] == "info"

    def test_to_dict_empty_issues(self) -> None:
        """Test to_dict with no issues."""
        result = MoveValidationResult(is_valid=True, move_uci="d2d4", move_type=MoveType.NORMAL)
        d = result.to_dict()
        assert d["issues"] == []

    def test_move_details(self) -> None:
        """Test move detail fields."""
        result = MoveValidationResult(
            is_valid=True,
            move_uci="e7e8q",
            move_type=MoveType.PROMOTION,
            from_square="e7",
            to_square="e8",
            piece_moved="P",
            promotion_piece="Q",
            is_check=True,
        )
        assert result.from_square == "e7"
        assert result.to_square == "e8"
        assert result.promotion_piece == "Q"
        assert result.is_check is True


# =============================================================================
# PositionVerificationResult Tests
# =============================================================================


class TestPositionVerificationResult:
    """Tests for PositionVerificationResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        result = PositionVerificationResult(
            is_valid=True,
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        )
        assert result.is_valid is True
        assert "rnbqkbnr" in result.fen

    def test_default_fields(self) -> None:
        """Test default field values."""
        result = PositionVerificationResult(is_valid=True, fen="test")
        assert result.issues == []
        assert result.extra_info == {}
        assert result.is_terminal is False
        assert result.game_result is None
        assert result.legal_moves_count == 0
        assert result.material_balance == 0
        assert result.game_phase is None
        assert result.has_valid_king_positions is True
        assert result.has_valid_pawn_positions is True
        assert result.has_valid_castling_rights is True
        assert result.has_valid_en_passant is True

    def test_has_errors_property(self) -> None:
        """Test has_errors property."""
        result = PositionVerificationResult(
            is_valid=False,
            fen="test",
            issues=[
                VerificationIssue(code="ERR", message="Error", severity=VerificationSeverity.ERROR),
            ],
        )
        assert result.has_errors is True

    def test_has_errors_no_errors(self) -> None:
        """Test has_errors with no errors."""
        result = PositionVerificationResult(is_valid=True, fen="test")
        assert result.has_errors is False


# =============================================================================
# MoveSequenceResult Tests
# =============================================================================


class TestMoveSequenceResult:
    """Tests for MoveSequenceResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        result = MoveSequenceResult(
            is_valid=True,
            initial_fen="start",
            moves=["e2e4", "e7e5"],
        )
        assert result.is_valid is True
        assert result.moves == ["e2e4", "e7e5"]

    def test_default_fields(self) -> None:
        """Test default field values."""
        result = MoveSequenceResult(is_valid=True, initial_fen="start", moves=[])
        assert result.final_fen is None
        assert result.issues == []
        assert result.move_results == []
        assert result.total_moves == 0
        assert result.valid_moves == 0
        assert result.captures == 0
        assert result.checks == 0
        assert result.castles == 0
        assert result.promotions == 0
        assert result.validation_time_ms == 0.0

    def test_error_rate_no_moves(self) -> None:
        """Test error_rate with no moves returns 0."""
        result = MoveSequenceResult(is_valid=True, initial_fen="start", moves=[], total_moves=0)
        assert result.error_rate == 0.0

    def test_error_rate_all_valid(self) -> None:
        """Test error_rate when all moves are valid."""
        result = MoveSequenceResult(
            is_valid=True,
            initial_fen="start",
            moves=["e2e4"],
            total_moves=10,
            valid_moves=10,
        )
        assert result.error_rate == 0.0

    def test_error_rate_some_invalid(self) -> None:
        """Test error_rate with some invalid moves."""
        result = MoveSequenceResult(
            is_valid=False,
            initial_fen="start",
            moves=["e2e4"],
            total_moves=10,
            valid_moves=7,
        )
        assert result.error_rate == pytest.approx(0.3)

    def test_error_rate_all_invalid(self) -> None:
        """Test error_rate when all moves are invalid."""
        result = MoveSequenceResult(
            is_valid=False,
            initial_fen="start",
            moves=[],
            total_moves=5,
            valid_moves=0,
        )
        assert result.error_rate == 1.0

    def test_has_errors_property(self) -> None:
        """Test has_errors property."""
        result = MoveSequenceResult(
            is_valid=False,
            initial_fen="start",
            moves=[],
            issues=[
                VerificationIssue(code="CRIT", message="Critical", severity=VerificationSeverity.CRITICAL),
            ],
        )
        assert result.has_errors is True


# =============================================================================
# GameVerificationResult Tests
# =============================================================================


class TestGameVerificationResult:
    """Tests for GameVerificationResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        result = GameVerificationResult(
            is_valid=True,
            game_id="game-001",
            moves=["e2e4", "e7e5"],
            result=GameResult.IN_PROGRESS,
        )
        assert result.is_valid is True
        assert result.game_id == "game-001"
        assert result.result == GameResult.IN_PROGRESS

    def test_default_fields(self) -> None:
        """Test default field values."""
        result = GameVerificationResult(is_valid=True, game_id="g1", moves=[], result=GameResult.IN_PROGRESS)
        assert result.issues == []
        assert result.move_sequence_result is None
        assert "rnbqkbnr" in result.initial_fen  # STARTING_FEN
        assert result.final_fen is None
        assert result.total_moves == 0
        assert result.total_plies == 0
        assert result.expected_result is None
        assert result.result_matches_expected is True
        assert result.verification_time_ms == 0.0

    def test_has_errors_property(self) -> None:
        """Test has_errors property."""
        result = GameVerificationResult(
            is_valid=False,
            game_id="g1",
            moves=[],
            result=GameResult.WHITE_WINS,
            issues=[
                VerificationIssue(code="ERR", message="Error", severity=VerificationSeverity.ERROR),
            ],
        )
        assert result.has_errors is True

    def test_has_errors_no_errors(self) -> None:
        """Test has_errors with only warnings."""
        result = GameVerificationResult(
            is_valid=True,
            game_id="g1",
            moves=[],
            result=GameResult.DRAW_AGREEMENT,
            issues=[
                VerificationIssue(code="W", message="Warn", severity=VerificationSeverity.WARNING),
            ],
        )
        assert result.has_errors is False

    def test_summary_valid_game(self) -> None:
        """Test summary for a valid game."""
        result = GameVerificationResult(
            is_valid=True,
            game_id="game-123",
            moves=["e2e4"],
            result=GameResult.WHITE_WINS,
            total_moves=40,
        )
        summary = result.summary()
        assert "game-123" in summary
        assert "VALID" in summary
        assert "40 moves" in summary
        assert "0 errors" in summary
        assert "0 warnings" in summary

    def test_summary_invalid_game(self) -> None:
        """Test summary for an invalid game."""
        result = GameVerificationResult(
            is_valid=False,
            game_id="game-456",
            moves=[],
            result=GameResult.BLACK_WINS,
            total_moves=10,
            issues=[
                VerificationIssue(code="E1", message="e1", severity=VerificationSeverity.ERROR),
                VerificationIssue(code="W1", message="w1", severity=VerificationSeverity.WARNING),
                VerificationIssue(code="W2", message="w2", severity=VerificationSeverity.WARNING),
            ],
        )
        summary = result.summary()
        assert "INVALID" in summary
        assert "1 errors" in summary
        assert "2 warnings" in summary


# =============================================================================
# EnsembleConsistencyResult Tests
# =============================================================================


class TestEnsembleConsistencyResult:
    """Tests for EnsembleConsistencyResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        result = EnsembleConsistencyResult(
            is_consistent=True,
            state_fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        )
        assert result.is_consistent is True

    def test_default_fields(self) -> None:
        """Test default field values."""
        result = EnsembleConsistencyResult(is_consistent=True, state_fen="test")
        assert result.issues == []
        assert result.agreement_rate == 0.0
        assert result.move_variance == {}
        assert result.agent_divergences == {}
        assert result.routing_consistency == 0.0
        assert result.agent_moves == {}
        assert result.agent_confidences == {}
        assert result.agent_values == {}
        assert result.ensemble_move is None
        assert result.ensemble_confidence == 0.0
        assert result.primary_agent is None
        assert result.check_time_ms == 0.0

    def test_all_agents_agree_empty(self) -> None:
        """Test all_agents_agree with no agents."""
        result = EnsembleConsistencyResult(is_consistent=True, state_fen="test")
        assert result.all_agents_agree is True

    def test_all_agents_agree_same_move(self) -> None:
        """Test all_agents_agree when all choose same move."""
        result = EnsembleConsistencyResult(
            is_consistent=True,
            state_fen="test",
            agent_moves={"hrm": "e2e4", "trm": "e2e4", "mcts": "e2e4"},
        )
        assert result.all_agents_agree is True

    def test_all_agents_agree_different_moves(self) -> None:
        """Test all_agents_agree when agents disagree."""
        result = EnsembleConsistencyResult(
            is_consistent=False,
            state_fen="test",
            agent_moves={"hrm": "e2e4", "trm": "d2d4", "mcts": "e2e4"},
        )
        assert result.all_agents_agree is False

    def test_get_disagreeing_agents_no_ensemble_move(self) -> None:
        """Test get_disagreeing_agents when no ensemble move."""
        result = EnsembleConsistencyResult(is_consistent=True, state_fen="test")
        assert result.get_disagreeing_agents() == []

    def test_get_disagreeing_agents(self) -> None:
        """Test get_disagreeing_agents identifies dissenters."""
        result = EnsembleConsistencyResult(
            is_consistent=False,
            state_fen="test",
            agent_moves={"hrm": "e2e4", "trm": "d2d4", "mcts": "e2e4"},
            ensemble_move="e2e4",
        )
        disagreeing = result.get_disagreeing_agents()
        assert disagreeing == ["trm"]

    def test_get_disagreeing_agents_all_agree(self) -> None:
        """Test get_disagreeing_agents when all agree."""
        result = EnsembleConsistencyResult(
            is_consistent=True,
            state_fen="test",
            agent_moves={"hrm": "e2e4", "trm": "e2e4"},
            ensemble_move="e2e4",
        )
        assert result.get_disagreeing_agents() == []

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        result = EnsembleConsistencyResult(
            is_consistent=True,
            state_fen="test-fen",
            agreement_rate=0.95,
            agent_moves={"hrm": "e2e4", "trm": "e2e4"},
            agent_confidences={"hrm": 0.9, "trm": 0.8},
            ensemble_move="e2e4",
            primary_agent="hrm",
        )
        d = result.to_dict()

        assert d["is_consistent"] is True
        assert d["state_fen"] == "test-fen"
        assert d["agreement_rate"] == 0.95
        assert d["agent_moves"] == {"hrm": "e2e4", "trm": "e2e4"}
        assert d["ensemble_move"] == "e2e4"
        assert d["primary_agent"] == "hrm"
        assert d["all_agents_agree"] is True


# =============================================================================
# BatchVerificationResult Tests
# =============================================================================


class TestBatchVerificationResult:
    """Tests for BatchVerificationResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        result = BatchVerificationResult(total_items=10, valid_items=8, invalid_items=2, results=[])
        assert result.total_items == 10
        assert result.valid_items == 8
        assert result.invalid_items == 2

    def test_default_fields(self) -> None:
        """Test default field values."""
        result = BatchVerificationResult(total_items=0, valid_items=0, invalid_items=0, results=[])
        assert result.issues == []
        assert result.total_time_ms == 0.0
        assert result.avg_time_per_item_ms == 0.0

    def test_success_rate_zero_items(self) -> None:
        """Test success_rate with zero items."""
        result = BatchVerificationResult(total_items=0, valid_items=0, invalid_items=0, results=[])
        assert result.success_rate == 0.0

    def test_success_rate_all_valid(self) -> None:
        """Test success_rate when all items valid."""
        result = BatchVerificationResult(total_items=5, valid_items=5, invalid_items=0, results=[])
        assert result.success_rate == 1.0

    def test_success_rate_partial(self) -> None:
        """Test success_rate with some invalid."""
        result = BatchVerificationResult(total_items=10, valid_items=7, invalid_items=3, results=[])
        assert result.success_rate == pytest.approx(0.7)

    def test_summary(self) -> None:
        """Test summary generation."""
        result = BatchVerificationResult(
            total_items=10,
            valid_items=8,
            invalid_items=2,
            results=[],
            total_time_ms=150.5,
        )
        summary = result.summary()
        assert "8/10" in summary
        assert "80.0%" in summary
        assert "150.5ms" in summary

    def test_summary_zero_items(self) -> None:
        """Test summary with zero items."""
        result = BatchVerificationResult(total_items=0, valid_items=0, invalid_items=0, results=[])
        summary = result.summary()
        assert "0/0" in summary
        assert "0.0%" in summary
