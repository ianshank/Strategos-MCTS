"""
Unit tests for chess verification modules.

Tests cover:
- types.py: MoveType, GameResult, VerificationSeverity, VerificationIssue,
  MoveValidationResult, PositionVerificationResult, MoveSequenceResult,
  GameVerificationResult, EnsembleConsistencyResult, BatchVerificationResult
- protocols.py: Protocol structural conformance
- move_validator.py: MoveValidator, MoveValidatorConfig, create_move_validator
- ensemble_checker.py: EnsembleConsistencyChecker, EnsembleCheckerConfig, create_ensemble_checker
- game_verifier.py: ChessGameVerifier, GameVerifierConfig, create_game_verifier
- factory.py: ChessVerificationFactory, VerificationBuilder, create_verification_factory
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

chess = pytest.importorskip("chess", reason="python-chess not installed")

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

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMoveType:
    """Test MoveType enum."""

    def test_values(self) -> None:
        assert MoveType.NORMAL.value == "normal"
        assert MoveType.CAPTURE.value == "capture"
        assert MoveType.CASTLE_KINGSIDE.value == "castle_kingside"
        assert MoveType.CASTLE_QUEENSIDE.value == "castle_queenside"
        assert MoveType.EN_PASSANT.value == "en_passant"
        assert MoveType.PROMOTION.value == "promotion"
        assert MoveType.PROMOTION_CAPTURE.value == "promotion_capture"
        assert MoveType.CHECK.value == "check"
        assert MoveType.CHECKMATE.value == "checkmate"


@pytest.mark.unit
class TestGameResult:
    """Test GameResult enum."""

    def test_values(self) -> None:
        assert GameResult.WHITE_WINS.value == "white_wins"
        assert GameResult.BLACK_WINS.value == "black_wins"
        assert GameResult.DRAW_STALEMATE.value == "draw_stalemate"
        assert GameResult.IN_PROGRESS.value == "in_progress"


@pytest.mark.unit
class TestVerificationSeverity:
    """Test VerificationSeverity enum."""

    def test_values(self) -> None:
        assert VerificationSeverity.INFO.value == "info"
        assert VerificationSeverity.WARNING.value == "warning"
        assert VerificationSeverity.ERROR.value == "error"
        assert VerificationSeverity.CRITICAL.value == "critical"


# ---------------------------------------------------------------------------
# VerificationIssue
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVerificationIssue:
    """Test VerificationIssue dataclass."""

    def test_creation(self) -> None:
        issue = VerificationIssue(
            code="TEST_CODE",
            message="Test message",
            severity=VerificationSeverity.ERROR,
        )
        assert issue.code == "TEST_CODE"
        assert issue.message == "Test message"
        assert issue.severity == VerificationSeverity.ERROR
        assert issue.context == {}
        assert issue.move_number is None
        assert issue.fen is None

    def test_str_with_move_number(self) -> None:
        issue = VerificationIssue(
            code="ILLEGAL_MOVE",
            message="Move is illegal",
            severity=VerificationSeverity.ERROR,
            move_number=5,
        )
        s = str(issue)
        assert "[ERROR]" in s
        assert "ILLEGAL_MOVE" in s
        assert "(move 5)" in s

    def test_str_without_move_number(self) -> None:
        issue = VerificationIssue(
            code="BAD_FEN",
            message="Invalid FEN",
            severity=VerificationSeverity.CRITICAL,
        )
        s = str(issue)
        assert "[CRITICAL]" in s
        assert "BAD_FEN" in s
        assert "(move" not in s


# ---------------------------------------------------------------------------
# MoveValidationResult
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMoveValidationResult:
    """Test MoveValidationResult dataclass."""

    def test_defaults(self) -> None:
        r = MoveValidationResult(
            is_valid=True,
            move_uci="e2e4",
            move_type=MoveType.NORMAL,
        )
        assert r.is_valid is True
        assert r.encoded_index is None
        assert r.issues == []
        assert r.is_check is False
        assert r.is_checkmate is False
        assert r.is_legal_in_position is True

    def test_has_errors_false(self) -> None:
        r = MoveValidationResult(is_valid=True, move_uci="e2e4", move_type=MoveType.NORMAL)
        assert r.has_errors is False

    def test_has_errors_true(self) -> None:
        r = MoveValidationResult(
            is_valid=False,
            move_uci="zzzz",
            move_type=MoveType.NORMAL,
            issues=[
                VerificationIssue(code="ERR", message="bad", severity=VerificationSeverity.ERROR),
            ],
        )
        assert r.has_errors is True

    def test_has_errors_warning_only(self) -> None:
        r = MoveValidationResult(
            is_valid=True,
            move_uci="e2e4",
            move_type=MoveType.NORMAL,
            issues=[
                VerificationIssue(code="WARN", message="warning", severity=VerificationSeverity.WARNING),
            ],
        )
        assert r.has_errors is False

    def test_to_dict(self) -> None:
        r = MoveValidationResult(
            is_valid=True,
            move_uci="e2e4",
            move_type=MoveType.NORMAL,
            from_square="e2",
            to_square="e4",
            piece_moved="P",
        )
        d = r.to_dict()
        assert d["is_valid"] is True
        assert d["move_uci"] == "e2e4"
        assert d["move_type"] == "normal"
        assert d["from_square"] == "e2"


# ---------------------------------------------------------------------------
# PositionVerificationResult
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPositionVerificationResult:
    """Test PositionVerificationResult dataclass."""

    def test_defaults(self) -> None:
        r = PositionVerificationResult(
            is_valid=True,
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        )
        assert r.is_terminal is False
        assert r.game_result is None
        assert r.has_valid_king_positions is True

    def test_has_errors(self) -> None:
        r = PositionVerificationResult(
            is_valid=False,
            fen="bad",
            issues=[
                VerificationIssue(code="ERR", message="bad", severity=VerificationSeverity.CRITICAL),
            ],
        )
        assert r.has_errors is True


# ---------------------------------------------------------------------------
# MoveSequenceResult
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMoveSequenceResult:
    """Test MoveSequenceResult dataclass."""

    def test_error_rate_zero_moves(self) -> None:
        r = MoveSequenceResult(is_valid=True, initial_fen="x", moves=[], total_moves=0)
        assert r.error_rate == 0.0

    def test_error_rate(self) -> None:
        r = MoveSequenceResult(is_valid=False, initial_fen="x", moves=["a", "b"], total_moves=4, valid_moves=3)
        assert abs(r.error_rate - 0.25) < 0.001

    def test_has_errors(self) -> None:
        r = MoveSequenceResult(
            is_valid=False,
            initial_fen="x",
            moves=[],
            issues=[
                VerificationIssue(code="ERR", message="bad", severity=VerificationSeverity.ERROR),
            ],
        )
        assert r.has_errors is True


# ---------------------------------------------------------------------------
# GameVerificationResult
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGameVerificationResult:
    """Test GameVerificationResult dataclass."""

    def test_summary(self) -> None:
        r = GameVerificationResult(
            is_valid=True,
            game_id="abc",
            moves=["e2e4", "e7e5"],
            result=GameResult.IN_PROGRESS,
            total_moves=2,
        )
        s = r.summary()
        assert "abc" in s
        assert "VALID" in s
        assert "2 moves" in s

    def test_summary_invalid(self) -> None:
        r = GameVerificationResult(
            is_valid=False,
            game_id="xyz",
            moves=[],
            result=GameResult.IN_PROGRESS,
            issues=[
                VerificationIssue(code="ERR", message="bad", severity=VerificationSeverity.ERROR),
            ],
        )
        s = r.summary()
        assert "INVALID" in s
        assert "1 errors" in s

    def test_has_errors(self) -> None:
        r = GameVerificationResult(
            is_valid=False,
            game_id="test",
            moves=[],
            result=GameResult.IN_PROGRESS,
            issues=[
                VerificationIssue(code="ERR", message="bad", severity=VerificationSeverity.ERROR),
            ],
        )
        assert r.has_errors is True


# ---------------------------------------------------------------------------
# EnsembleConsistencyResult
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEnsembleConsistencyResult:
    """Test EnsembleConsistencyResult dataclass."""

    def test_all_agents_agree_empty(self) -> None:
        r = EnsembleConsistencyResult(is_consistent=True, state_fen="x")
        assert r.all_agents_agree is True

    def test_all_agents_agree_true(self) -> None:
        r = EnsembleConsistencyResult(
            is_consistent=True,
            state_fen="x",
            agent_moves={"hrm": "e2e4", "trm": "e2e4", "mcts": "e2e4"},
        )
        assert r.all_agents_agree is True

    def test_all_agents_agree_false(self) -> None:
        r = EnsembleConsistencyResult(
            is_consistent=False,
            state_fen="x",
            agent_moves={"hrm": "e2e4", "trm": "d2d4"},
        )
        assert r.all_agents_agree is False

    def test_get_disagreeing_agents(self) -> None:
        r = EnsembleConsistencyResult(
            is_consistent=False,
            state_fen="x",
            agent_moves={"hrm": "e2e4", "trm": "d2d4", "mcts": "e2e4"},
            ensemble_move="e2e4",
        )
        disagreeing = r.get_disagreeing_agents()
        assert disagreeing == ["trm"]

    def test_get_disagreeing_agents_no_ensemble_move(self) -> None:
        r = EnsembleConsistencyResult(is_consistent=True, state_fen="x")
        assert r.get_disagreeing_agents() == []

    def test_to_dict(self) -> None:
        r = EnsembleConsistencyResult(
            is_consistent=True,
            state_fen="x",
            agreement_rate=1.0,
            agent_moves={"hrm": "e2e4"},
            ensemble_move="e2e4",
        )
        d = r.to_dict()
        assert d["is_consistent"] is True
        assert d["agreement_rate"] == 1.0
        assert d["all_agents_agree"] is True


# ---------------------------------------------------------------------------
# BatchVerificationResult
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBatchVerificationResult:
    """Test BatchVerificationResult dataclass."""

    def test_success_rate_zero(self) -> None:
        r = BatchVerificationResult(total_items=0, valid_items=0, invalid_items=0, results=[])
        assert r.success_rate == 0.0

    def test_success_rate(self) -> None:
        r = BatchVerificationResult(total_items=10, valid_items=8, invalid_items=2, results=[])
        assert abs(r.success_rate - 0.8) < 0.001

    def test_summary(self) -> None:
        r = BatchVerificationResult(
            total_items=10, valid_items=8, invalid_items=2, results=[], total_time_ms=150.0
        )
        s = r.summary()
        assert "8/10" in s
        assert "80.0%" in s


# ---------------------------------------------------------------------------
# MoveValidator tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMoveValidatorConfig:
    """Test MoveValidatorConfig."""

    def test_defaults(self) -> None:
        from src.games.chess.verification.move_validator import MoveValidatorConfig

        config = MoveValidatorConfig()
        assert config.validate_encoding is True
        assert config.validate_legality is True
        assert config.validate_san_format is False
        assert config.log_validations is False


@pytest.mark.unit
class TestMoveValidator:
    """Test MoveValidator with real chess library."""

    def _make_validator(self, validate_encoding: bool = False) -> MoveValidator:  # noqa: F821
        from src.games.chess.verification.move_validator import MoveValidator, MoveValidatorConfig

        config = MoveValidatorConfig(validate_encoding=validate_encoding, validate_legality=True)
        with patch("src.games.chess.verification.move_validator.get_structured_logger"):
            return MoveValidator(config=config)

    def test_init_defaults(self) -> None:
        with patch("src.games.chess.verification.move_validator.get_structured_logger"):
            from src.games.chess.verification.move_validator import MoveValidator

            v = MoveValidator()
            assert v.config is not None
            assert v.encoder is not None

    def test_validate_legal_move(self) -> None:
        from src.games.chess.state import ChessGameState

        validator = self._make_validator()
        state = ChessGameState.initial()
        result = validator.validate_move(state, "e2e4")
        assert result.is_valid is True
        assert result.move_uci == "e2e4"
        assert result.move_type == MoveType.NORMAL
        assert result.from_square == "e2"
        assert result.to_square == "e4"
        assert result.piece_moved == "P"
        assert result.is_legal_in_position is True

    def test_validate_illegal_move(self) -> None:
        from src.games.chess.state import ChessGameState

        validator = self._make_validator()
        state = ChessGameState.initial()
        result = validator.validate_move(state, "e2e5")
        assert result.is_valid is False
        assert any(i.code == "ILLEGAL_MOVE" for i in result.issues)

    def test_validate_invalid_uci(self) -> None:
        from src.games.chess.state import ChessGameState

        validator = self._make_validator()
        state = ChessGameState.initial()
        result = validator.validate_move(state, "zzzz")
        assert result.is_valid is False
        assert any(i.code == "INVALID_UCI_FORMAT" for i in result.issues)

    def test_validate_capture_move(self) -> None:
        from src.games.chess.state import ChessGameState

        # Italian Game position with capture possible
        ChessGameState.from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
        validator = self._make_validator()
        # Nxe4 is not legal here, but d7d5 is, or check a known capture
        # Let's use a position where a capture is clearly legal
        ChessGameState.from_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
        # In this position, no direct captures. Use a simpler test:
        state3 = ChessGameState.from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
        result = validator.validate_move(state3, "e4d5")
        assert result.is_valid is True
        assert result.move_type == MoveType.CAPTURE

    def test_validate_castling_kingside(self) -> None:
        from src.games.chess.state import ChessGameState

        # Position where white can castle kingside
        state = ChessGameState.from_fen("rnbqkbnr/pppppppp/8/8/8/5NP1/PPPPPPBP/RNBQK2R w KQkq - 0 1")
        validator = self._make_validator()
        result = validator.validate_castling(state, kingside=True)
        assert result.is_valid is True
        assert result.move_type == MoveType.CASTLE_KINGSIDE
        assert result.move_uci == "e1g1"

    def test_validate_castling_no_rights(self) -> None:
        from src.games.chess.state import ChessGameState

        # Position with no castling rights
        state = ChessGameState.from_fen("rnbqkbnr/pppppppp/8/8/8/5NP1/PPPPPPBP/RNBQK2R w - - 0 1")
        validator = self._make_validator()
        result = validator.validate_castling(state, kingside=True)
        assert result.is_valid is False
        assert any(i.code == "NO_CASTLING_RIGHTS" for i in result.issues)

    def test_validate_promotion(self) -> None:
        from src.games.chess.state import ChessGameState

        # White pawn on e7 about to promote
        state = ChessGameState.from_fen("8/4P3/8/8/8/8/8/4K2k w - - 0 1")
        validator = self._make_validator()
        result = validator.validate_promotion(state, "e7e8q")
        assert result.is_valid is True
        assert result.move_type == MoveType.PROMOTION
        assert result.promotion_piece == "queen"

    def test_validate_promotion_no_piece(self) -> None:
        from src.games.chess.state import ChessGameState

        # e7 has no pawn (test NOT_PAWN)
        state = ChessGameState.from_fen("8/8/8/8/8/8/8/4K2k w - - 0 1")
        validator = self._make_validator()
        result = validator.validate_promotion(state, "e7e8q")
        assert result.is_valid is False

    def test_validate_en_passant_no_ep_square(self) -> None:
        from src.games.chess.state import ChessGameState

        state = ChessGameState.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        validator = self._make_validator()
        result = validator.validate_en_passant(state, "e5d6")
        assert result.is_valid is False

    def test_validate_en_passant_valid(self) -> None:
        from src.games.chess.state import ChessGameState

        # White pawn on e5, black just played d7d5 (en passant square d6)
        state = ChessGameState.from_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
        validator = self._make_validator()
        result = validator.validate_en_passant(state, "e5d6")
        assert result.is_valid is True
        assert result.move_type == MoveType.EN_PASSANT

    def test_validate_all_legal_moves(self) -> None:
        from src.games.chess.state import ChessGameState

        state = ChessGameState.initial()
        validator = self._make_validator()
        results = validator.validate_all_legal_moves(state)
        # Starting position has 20 legal moves
        assert len(results) == 20
        assert all(r.is_valid for r in results)


@pytest.mark.unit
class TestCreateMoveValidator:
    """Test create_move_validator factory."""

    def test_create_default(self) -> None:
        from src.games.chess.verification.move_validator import MoveValidator, create_move_validator

        with patch("src.games.chess.verification.move_validator.get_structured_logger"):
            v = create_move_validator()
            assert isinstance(v, MoveValidator)


# ---------------------------------------------------------------------------
# EnsembleConsistencyChecker tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEnsembleCheckerConfig:
    """Test EnsembleCheckerConfig."""

    def test_defaults_fallback(self) -> None:
        from src.games.chess.verification.ensemble_checker import EnsembleCheckerConfig

        with patch("src.config.settings.get_settings", side_effect=RuntimeError):
            config = EnsembleCheckerConfig()
            assert config.agreement_threshold == 0.6
            assert config.confidence_divergence_threshold == 0.3
            assert config.routing_threshold == 0.5


@pytest.mark.unit
class TestEnsembleConsistencyChecker:
    """Test EnsembleConsistencyChecker."""

    def _make_checker(self, agent=None) -> EnsembleConsistencyChecker:  # noqa: F821
        from src.games.chess.verification.ensemble_checker import (
            EnsembleCheckerConfig,
            EnsembleConsistencyChecker,
        )

        with patch(
            "src.config.settings.get_settings",
            side_effect=RuntimeError,
        ):
            config = EnsembleCheckerConfig(log_checks=False)

        with patch("src.games.chess.verification.ensemble_checker.get_structured_logger"):
            return EnsembleConsistencyChecker(ensemble_agent=agent, config=config)

    def test_init(self) -> None:
        checker = self._make_checker()
        assert checker.ensemble_agent is None
        assert checker.config is not None

    def test_set_ensemble_agent(self) -> None:
        checker = self._make_checker()
        mock_agent = MagicMock()
        checker.ensemble_agent = mock_agent
        assert checker.ensemble_agent is mock_agent

    def test_get_divergence_threshold(self) -> None:
        checker = self._make_checker()
        assert checker.get_divergence_threshold() == 0.3

    @pytest.mark.asyncio
    async def test_check_position_no_agent(self) -> None:
        from src.games.chess.state import ChessGameState

        checker = self._make_checker(agent=None)
        state = ChessGameState.initial()
        result = await checker.check_position_consistency(state)
        assert result.is_consistent is False
        assert any(i.code == "NO_ENSEMBLE_AGENT" for i in result.issues)

    def test_calculate_agreement_rate_empty(self) -> None:
        checker = self._make_checker()
        assert checker._calculate_agreement_rate({}) == 0.0

    def test_calculate_agreement_rate_single(self) -> None:
        checker = self._make_checker()
        assert checker._calculate_agreement_rate({"hrm": "e2e4"}) == 1.0

    def test_calculate_agreement_rate_all_agree(self) -> None:
        checker = self._make_checker()
        rate = checker._calculate_agreement_rate({"hrm": "e2e4", "trm": "e2e4", "mcts": "e2e4"})
        assert rate == 1.0

    def test_calculate_agreement_rate_none_agree(self) -> None:
        checker = self._make_checker()
        rate = checker._calculate_agreement_rate({"hrm": "e2e4", "trm": "d2d4", "mcts": "g1f3"})
        assert rate == 0.0

    def test_calculate_agreement_rate_partial(self) -> None:
        checker = self._make_checker()
        rate = checker._calculate_agreement_rate({"hrm": "e2e4", "trm": "e2e4", "mcts": "d2d4"})
        # 3 pairs: (hrm,trm)=agree, (hrm,mcts)=disagree, (trm,mcts)=disagree
        assert abs(rate - 1.0 / 3.0) < 0.001

    def test_calculate_divergences(self) -> None:
        checker = self._make_checker()
        divergences = checker._calculate_divergences(
            {"hrm": "e2e4", "trm": "d2d4"},
            {"hrm": 0.9, "trm": 0.8},
            "e2e4",
        )
        assert divergences["hrm"] == 0.0  # matches ensemble
        assert divergences["trm"] == 0.8  # confident + wrong

    def test_check_routing_consistency(self) -> None:
        from src.games.chess.config import AgentType, GamePhase
        from src.games.chess.state import ChessGameState

        checker = self._make_checker()
        state = ChessGameState.initial()

        # Opening position with HRM (expected) should give high score
        with patch.object(state, "get_game_phase", return_value=GamePhase.OPENING):
            score = checker._check_routing_consistency(state, AgentType.HRM)
            assert score == 1.0  # match

    def test_get_expected_agent_for_phase(self) -> None:
        from src.games.chess.config import AgentType, GamePhase

        checker = self._make_checker()
        assert checker._get_expected_agent_for_phase(GamePhase.OPENING) == AgentType.HRM
        assert checker._get_expected_agent_for_phase(GamePhase.MIDDLEGAME) == AgentType.MCTS
        assert checker._get_expected_agent_for_phase(GamePhase.ENDGAME) == AgentType.TRM


@pytest.mark.unit
class TestCreateEnsembleChecker:
    """Test create_ensemble_checker factory."""

    def test_create(self) -> None:
        from src.games.chess.verification.ensemble_checker import (
            EnsembleConsistencyChecker,
            create_ensemble_checker,
        )

        with patch("src.games.chess.verification.ensemble_checker.get_structured_logger"):
            with patch(
                "src.config.settings.get_settings",
                side_effect=RuntimeError,
            ):
                checker = create_ensemble_checker()
                assert isinstance(checker, EnsembleConsistencyChecker)


# ---------------------------------------------------------------------------
# GameVerifier tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGameVerifierConfig:
    """Test GameVerifierConfig."""

    def test_defaults(self) -> None:
        from src.games.chess.verification.game_verifier import GameVerifierConfig

        config = GameVerifierConfig()
        assert config.verify_encoding is True
        assert config.verify_terminal_state is True
        assert config.verify_result is True
        assert config.stop_on_first_error is False
        assert config.max_moves == 500
        assert config.max_repetitions == 10


@pytest.mark.unit
class TestChessGameVerifier:
    """Test ChessGameVerifier."""

    def _make_verifier(self, **kwargs) -> ChessGameVerifier:  # noqa: F821
        from src.games.chess.verification.game_verifier import ChessGameVerifier, GameVerifierConfig
        from src.games.chess.verification.move_validator import MoveValidator, MoveValidatorConfig

        mv_config = MoveValidatorConfig(validate_encoding=False, validate_legality=True)
        gv_config = GameVerifierConfig(log_verifications=False, **kwargs)

        with patch("src.games.chess.verification.move_validator.get_structured_logger"), \
             patch("src.games.chess.verification.game_verifier.get_structured_logger"):
            mv = MoveValidator(config=mv_config)
            return ChessGameVerifier(move_validator=mv, config=gv_config)

    def test_init(self) -> None:
        verifier = self._make_verifier()
        assert verifier.config is not None
        assert verifier.move_validator is not None

    def test_verify_position_valid(self) -> None:
        from src.games.chess.constants import STARTING_FEN

        verifier = self._make_verifier()
        result = verifier.verify_position(STARTING_FEN)
        assert result.is_valid is True
        assert result.has_valid_king_positions is True
        assert result.has_valid_pawn_positions is True
        assert result.legal_moves_count == 20

    def test_verify_position_invalid_fen(self) -> None:
        verifier = self._make_verifier()
        result = verifier.verify_position("not a valid fen")
        assert result.is_valid is False
        assert any(i.code == "INVALID_FEN" for i in result.issues)

    def test_verify_move_sequence_valid(self) -> None:
        from src.games.chess.constants import STARTING_FEN

        verifier = self._make_verifier()
        result = verifier.verify_move_sequence(STARTING_FEN, ["e2e4", "e7e5", "g1f3"])
        assert result.is_valid is True
        assert result.total_moves == 3
        assert result.valid_moves == 3

    def test_verify_move_sequence_invalid_move(self) -> None:
        from src.games.chess.constants import STARTING_FEN

        verifier = self._make_verifier()
        result = verifier.verify_move_sequence(STARTING_FEN, ["e2e4", "e2e5"])
        assert result.is_valid is False
        assert result.valid_moves < result.total_moves

    def test_verify_move_sequence_invalid_initial_fen(self) -> None:
        verifier = self._make_verifier()
        result = verifier.verify_move_sequence("bad fen", ["e2e4"])
        assert result.is_valid is False
        assert any(i.code == "INVALID_INITIAL_FEN" for i in result.issues)

    @pytest.mark.asyncio
    async def test_verify_full_game_valid(self) -> None:
        verifier = self._make_verifier()
        result = await verifier.verify_full_game(["e2e4", "e7e5", "g1f3", "b8c6"])
        assert result.is_valid is True
        assert result.total_moves == 4
        assert result.result == GameResult.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_verify_full_game_with_expected_outcome_mismatch(self) -> None:
        verifier = self._make_verifier()
        result = await verifier.verify_full_game(
            ["e2e4", "e7e5"],
            expected_outcome=GameResult.WHITE_WINS,
        )
        assert result.is_valid is False
        assert result.result_matches_expected is False
        assert any(i.code == "RESULT_MISMATCH" for i in result.issues)

    @pytest.mark.asyncio
    async def test_verify_full_game_invalid_move(self) -> None:
        verifier = self._make_verifier()
        result = await verifier.verify_full_game(["e2e4", "e2e5"])
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_verify_game_playthrough(self) -> None:
        from src.games.chess.state import ChessGameState

        verifier = self._make_verifier()
        state = ChessGameState.initial()
        result = await verifier.verify_game_playthrough(state, max_moves=5)
        assert result.is_valid is True
        assert result.total_moves <= 5

    def test_determine_game_result_none_fen(self) -> None:
        verifier = self._make_verifier()
        result = verifier._determine_game_result(None)
        assert result == GameResult.IN_PROGRESS

    def test_determine_game_result_invalid_fen(self) -> None:
        verifier = self._make_verifier()
        result = verifier._determine_game_result("bad fen string")
        assert result == GameResult.IN_PROGRESS

    def test_calculate_material_balance_starting(self) -> None:
        import chess

        verifier = self._make_verifier()
        board = chess.Board()
        balance = verifier._calculate_material_balance(board)
        assert balance == 0  # Equal material at start

    def test_verify_position_checks_pawns_on_back_rank(self) -> None:
        import chess

        verifier = self._make_verifier()
        # Position with pawn on rank 1 (invalid)
        board = chess.Board("8/8/8/8/8/8/8/P3K2k w - - 0 1")
        checks = verifier._validate_position_checks(board)
        assert checks["pawn_positions"][0] is False


@pytest.mark.unit
class TestCreateGameVerifier:
    """Test create_game_verifier factory."""

    def test_create(self) -> None:
        from src.games.chess.verification.game_verifier import ChessGameVerifier, create_game_verifier

        with patch("src.games.chess.verification.move_validator.get_structured_logger"), \
             patch("src.games.chess.verification.game_verifier.get_structured_logger"):
            v = create_game_verifier()
            assert isinstance(v, ChessGameVerifier)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestChessVerificationFactory:
    """Test ChessVerificationFactory."""

    def _make_factory(self) -> ChessVerificationFactory:  # noqa: F821
        from src.games.chess.verification.factory import ChessVerificationFactory

        mock_settings = MagicMock()
        mock_settings.LOG_LEVEL = MagicMock(value="INFO")

        with patch("src.games.chess.verification.factory.get_structured_logger"):
            return ChessVerificationFactory(settings=mock_settings)

    def test_init(self) -> None:
        factory = self._make_factory()
        assert factory.settings is not None
        assert factory.chess_config is None

    def test_create_move_validator(self) -> None:
        from src.games.chess.verification.move_validator import MoveValidator

        factory = self._make_factory()
        with patch("src.games.chess.verification.move_validator.get_structured_logger"):
            v = factory.create_move_validator()
            assert isinstance(v, MoveValidator)

    def test_create_move_validator_reuses_encoder(self) -> None:
        factory = self._make_factory()
        with patch("src.games.chess.verification.move_validator.get_structured_logger"):
            v1 = factory.create_move_validator(reuse_encoder=True)
            v2 = factory.create_move_validator(reuse_encoder=True)
            assert v1.encoder is v2.encoder

    def test_create_game_verifier(self) -> None:
        from src.games.chess.verification.game_verifier import ChessGameVerifier

        factory = self._make_factory()
        with patch("src.games.chess.verification.move_validator.get_structured_logger"), \
             patch("src.games.chess.verification.game_verifier.get_structured_logger"):
            v = factory.create_game_verifier()
            assert isinstance(v, ChessGameVerifier)

    def test_create_ensemble_checker(self) -> None:
        from src.games.chess.verification.ensemble_checker import EnsembleConsistencyChecker

        factory = self._make_factory()
        with patch("src.games.chess.verification.ensemble_checker.get_structured_logger"), \
             patch(
                 "src.config.settings.get_settings",
                 side_effect=RuntimeError,
             ):
            c = factory.create_ensemble_checker()
            assert isinstance(c, EnsembleConsistencyChecker)

    def test_create_all_verifiers(self) -> None:
        factory = self._make_factory()
        with patch("src.games.chess.verification.move_validator.get_structured_logger"), \
             patch("src.games.chess.verification.game_verifier.get_structured_logger"), \
             patch("src.games.chess.verification.ensemble_checker.get_structured_logger"), \
             patch(
                 "src.config.settings.get_settings",
                 side_effect=RuntimeError,
             ):
            result = factory.create_all_verifiers()
            assert "move_validator" in result
            assert "game_verifier" in result
            assert "ensemble_checker" in result


@pytest.mark.unit
class TestVerificationBuilder:
    """Test VerificationBuilder fluent API."""

    def test_build_move_validator(self) -> None:
        from src.games.chess.verification.factory import VerificationBuilder
        from src.games.chess.verification.move_validator import MoveValidator

        with patch("src.games.chess.verification.move_validator.get_structured_logger"):
            v = VerificationBuilder().with_encoding_validation(True).with_legality_validation(True).build_move_validator()
            assert isinstance(v, MoveValidator)
            assert v.config.validate_encoding is True

    def test_build_game_verifier(self) -> None:
        from src.games.chess.verification.factory import VerificationBuilder
        from src.games.chess.verification.game_verifier import ChessGameVerifier

        with patch("src.games.chess.verification.move_validator.get_structured_logger"), \
             patch("src.games.chess.verification.game_verifier.get_structured_logger"):
            v = (
                VerificationBuilder()
                .with_stop_on_first_error(True)
                .with_logging(True)
                .build_game_verifier()
            )
            assert isinstance(v, ChessGameVerifier)
            assert v.config.stop_on_first_error is True

    def test_build_ensemble_checker(self) -> None:
        from src.games.chess.verification.ensemble_checker import EnsembleConsistencyChecker
        from src.games.chess.verification.factory import VerificationBuilder

        with patch("src.games.chess.verification.ensemble_checker.get_structured_logger"), \
             patch(
                 "src.config.settings.get_settings",
                 side_effect=RuntimeError,
             ):
            c = (
                VerificationBuilder()
                .with_agreement_threshold(0.8)
                .with_divergence_threshold(0.2)
                .build_ensemble_checker()
            )
            assert isinstance(c, EnsembleConsistencyChecker)
            assert c.config.agreement_threshold == 0.8

    def test_fluent_chaining(self) -> None:
        from src.games.chess.verification.factory import VerificationBuilder

        builder = (
            VerificationBuilder()
            .with_encoding_validation(False)
            .with_legality_validation(False)
            .with_terminal_verification(False)
            .with_result_verification(False)
            .with_stop_on_first_error(True)
            .with_logging(True)
            .with_agreement_threshold(0.9)
            .with_divergence_threshold(0.1)
        )
        assert builder._validate_encoding is False
        assert builder._validate_legality is False
        assert builder._verify_terminal is False
        assert builder._verify_result is False
        assert builder._stop_on_first_error is True
        assert builder._log_validations is True
        assert builder._agreement_threshold == 0.9
        assert builder._divergence_threshold == 0.1

    def test_with_settings(self) -> None:
        from src.games.chess.verification.factory import VerificationBuilder

        mock_settings = MagicMock()
        builder = VerificationBuilder().with_settings(mock_settings)
        assert builder._settings is mock_settings

    def test_with_chess_config(self) -> None:
        from src.games.chess.verification.factory import VerificationBuilder

        mock_config = MagicMock()
        builder = VerificationBuilder().with_chess_config(mock_config)
        assert builder._chess_config is mock_config


@pytest.mark.unit
class TestCreateVerificationFactory:
    """Test create_verification_factory function."""

    def test_create(self) -> None:
        from src.games.chess.verification.factory import ChessVerificationFactory, create_verification_factory

        mock_settings = MagicMock()
        with patch("src.games.chess.verification.factory.get_structured_logger"):
            f = create_verification_factory(settings=mock_settings)
            assert isinstance(f, ChessVerificationFactory)


# ---------------------------------------------------------------------------
# Protocol structural conformance tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProtocolConformance:
    """Test that implementations conform to protocols."""

    def test_move_validator_has_protocol_methods(self) -> None:
        from src.games.chess.verification.move_validator import MoveValidator

        assert hasattr(MoveValidator, "validate_move")
        assert hasattr(MoveValidator, "validate_castling")
        assert hasattr(MoveValidator, "validate_en_passant")
        assert hasattr(MoveValidator, "validate_promotion")
        assert hasattr(MoveValidator, "validate_encoding_roundtrip")

    def test_game_verifier_has_protocol_methods(self) -> None:
        from src.games.chess.verification.game_verifier import ChessGameVerifier

        assert hasattr(ChessGameVerifier, "verify_full_game")
        assert hasattr(ChessGameVerifier, "verify_position")
        assert hasattr(ChessGameVerifier, "verify_move_sequence")
        assert hasattr(ChessGameVerifier, "verify_game_playthrough")

    def test_ensemble_checker_has_protocol_methods(self) -> None:
        from src.games.chess.verification.ensemble_checker import EnsembleConsistencyChecker

        assert hasattr(EnsembleConsistencyChecker, "check_position_consistency")
        assert hasattr(EnsembleConsistencyChecker, "check_sequence_consistency")
        assert hasattr(EnsembleConsistencyChecker, "check_game_consistency")
        assert hasattr(EnsembleConsistencyChecker, "get_divergence_threshold")
