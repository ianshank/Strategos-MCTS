"""
Unit tests for EnsembleConsistencyChecker.

Tests configuration, consistency checking, agreement calculation,
divergence calculation, and routing consistency.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.games.chess.config import AgentType, GamePhase
from src.games.chess.constants import (
    DEFAULT_AGREEMENT_THRESHOLD,
    DEFAULT_CONFIDENCE,
    DEFAULT_CONFIDENCE_DIVERGENCE_THRESHOLD,
    DEFAULT_ROUTING_THRESHOLD,
    DEFAULT_VALUE_DIVERGENCE_THRESHOLD,
)
from src.games.chess.verification.ensemble_checker import (
    EnsembleCheckerConfig,
    EnsembleConsistencyChecker,
    create_ensemble_checker,
)
from src.games.chess.verification.types import VerificationSeverity

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_state() -> MagicMock:
    """Create a mock ChessGameState."""
    state = MagicMock()
    state.fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    state.get_game_phase.return_value = GamePhase.OPENING
    return state


@pytest.fixture
def config() -> EnsembleCheckerConfig:
    """Create a test configuration with explicit values.

    We patch get_settings to avoid needing real API keys during tests.
    """
    with patch(
        "src.config.settings.get_settings",
        side_effect=RuntimeError("no settings in test"),
    ):
        return EnsembleCheckerConfig(
            agreement_threshold=0.6,
            confidence_divergence_threshold=0.3,
            value_divergence_threshold=0.2,
            routing_threshold=0.5,
            default_confidence=0.5,
            log_checks=False,
        )


@pytest.fixture
def checker(config: EnsembleCheckerConfig) -> EnsembleConsistencyChecker:
    """Create a checker without an ensemble agent."""
    return EnsembleConsistencyChecker(config=config)


@pytest.fixture
def mock_ensemble_response() -> MagicMock:
    """Create a mock ensemble response with all agents agreeing."""
    response = MagicMock()
    response.best_move = "e2e4"
    response.confidence = 0.9

    # Agent responses
    hrm_response = MagicMock()
    hrm_response.move = "e2e4"
    hrm_response.confidence = 0.9
    hrm_response.value_estimate = 0.5

    trm_response = MagicMock()
    trm_response.move = "e2e4"
    trm_response.confidence = 0.8
    trm_response.value_estimate = 0.45

    mcts_response = MagicMock()
    mcts_response.move = "e2e4"
    mcts_response.confidence = 0.85
    mcts_response.value_estimate = 0.52

    response.agent_responses = {
        "hrm": hrm_response,
        "trm": trm_response,
        "mcts": mcts_response,
    }

    response.move_probabilities = {"e2e4": 0.6, "d2d4": 0.3, "c2c4": 0.1}
    response.routing_decision = MagicMock()
    response.routing_decision.primary_agent = AgentType.HRM

    return response


@pytest.fixture
def mock_disagreeing_response() -> MagicMock:
    """Create a mock ensemble response with agents disagreeing."""
    response = MagicMock()
    response.best_move = "e2e4"
    response.confidence = 0.6

    hrm_response = MagicMock()
    hrm_response.move = "e2e4"
    hrm_response.confidence = 0.9
    hrm_response.value_estimate = 0.5

    trm_response = MagicMock()
    trm_response.move = "d2d4"
    trm_response.confidence = 0.7
    trm_response.value_estimate = 0.45

    mcts_response = MagicMock()
    mcts_response.move = "c2c4"
    mcts_response.confidence = 0.6
    mcts_response.value_estimate = 0.4

    response.agent_responses = {
        "hrm": hrm_response,
        "trm": trm_response,
        "mcts": mcts_response,
    }

    response.move_probabilities = {"e2e4": 0.4, "d2d4": 0.35, "c2c4": 0.25}
    response.routing_decision = MagicMock()
    response.routing_decision.primary_agent = AgentType.HRM

    return response


# =============================================================================
# EnsembleCheckerConfig Tests
# =============================================================================


class TestEnsembleCheckerConfig:
    """Tests for EnsembleCheckerConfig dataclass."""

    def _make_config(self, **kwargs: object) -> EnsembleCheckerConfig:
        """Helper to create config with get_settings mocked."""
        with patch(
            "src.config.settings.get_settings",
            side_effect=RuntimeError("no settings in test"),
        ):
            return EnsembleCheckerConfig(**kwargs)  # type: ignore[arg-type]

    def test_default_config_loads_from_constants(self) -> None:
        """Test default config loads values from constants when settings unavailable."""
        config = self._make_config()

        # Should have values from defaults (either settings or constants)
        assert config.agreement_threshold is not None
        assert config.confidence_divergence_threshold is not None
        assert config.routing_threshold is not None
        assert config.value_divergence_threshold is not None
        assert config.default_confidence is not None

    def test_explicit_values_preserved(self) -> None:
        """Test explicitly set values are not overridden."""
        config = self._make_config(
            agreement_threshold=0.9,
            confidence_divergence_threshold=0.1,
            routing_threshold=0.8,
            value_divergence_threshold=0.05,
            default_confidence=0.7,
        )
        assert config.agreement_threshold == 0.9
        assert config.confidence_divergence_threshold == 0.1
        assert config.routing_threshold == 0.8
        assert config.value_divergence_threshold == 0.05
        assert config.default_confidence == 0.7

    def test_default_expected_agents(self) -> None:
        """Test default expected agents per phase."""
        config = self._make_config()
        assert config.opening_expected_agent == AgentType.HRM
        assert config.middlegame_expected_agent == AgentType.MCTS
        assert config.endgame_expected_agent == AgentType.TRM

    def test_analysis_options_default(self) -> None:
        """Test default analysis options."""
        config = self._make_config()
        assert config.compute_move_variance is True
        assert config.compute_divergences is True

    def test_log_checks_default(self) -> None:
        """Test log_checks default."""
        config = self._make_config()
        assert config.log_checks is True

    def test_fallback_when_settings_unavailable(self) -> None:
        """Test fallback to constants when settings raise errors."""
        config = self._make_config()

        assert config.agreement_threshold == DEFAULT_AGREEMENT_THRESHOLD
        assert config.confidence_divergence_threshold == DEFAULT_CONFIDENCE_DIVERGENCE_THRESHOLD
        assert config.routing_threshold == DEFAULT_ROUTING_THRESHOLD
        assert config.value_divergence_threshold == DEFAULT_VALUE_DIVERGENCE_THRESHOLD
        assert config.default_confidence == DEFAULT_CONFIDENCE


# =============================================================================
# EnsembleConsistencyChecker Init Tests
# =============================================================================


class TestEnsembleConsistencyCheckerInit:
    """Tests for EnsembleConsistencyChecker initialization."""

    def test_init_without_agent(self, config: EnsembleCheckerConfig) -> None:
        """Test initialization without ensemble agent."""
        checker = EnsembleConsistencyChecker(config=config)
        assert checker.ensemble_agent is None
        assert checker.config is config

    def test_init_with_agent(self, config: EnsembleCheckerConfig) -> None:
        """Test initialization with ensemble agent."""
        mock_agent = MagicMock()
        checker = EnsembleConsistencyChecker(ensemble_agent=mock_agent, config=config)
        assert checker.ensemble_agent is mock_agent

    def test_init_default_config(self) -> None:
        """Test initialization creates default config."""
        with patch(
            "src.config.settings.get_settings",
            side_effect=RuntimeError("no settings in test"),
        ):
            checker = EnsembleConsistencyChecker()
        assert checker.config is not None
        assert isinstance(checker.config, EnsembleCheckerConfig)

    def test_set_ensemble_agent(self, checker: EnsembleConsistencyChecker) -> None:
        """Test setting ensemble agent after init."""
        mock_agent = MagicMock()
        checker.ensemble_agent = mock_agent
        assert checker.ensemble_agent is mock_agent

    def test_get_divergence_threshold(self, checker: EnsembleConsistencyChecker) -> None:
        """Test get_divergence_threshold returns config value."""
        threshold = checker.get_divergence_threshold()
        assert threshold == 0.3

    def test_get_divergence_threshold_none_config(self) -> None:
        """Test get_divergence_threshold returns default when config value is None."""
        with patch(
            "src.config.settings.get_settings",
            side_effect=RuntimeError("no settings in test"),
        ):
            config = EnsembleCheckerConfig(confidence_divergence_threshold=None)
        # __post_init__ will set it, but let's force None
        config.confidence_divergence_threshold = None
        checker = EnsembleConsistencyChecker(config=config)
        threshold = checker.get_divergence_threshold()
        assert threshold == 0.3  # Default fallback


# =============================================================================
# check_position_consistency Tests
# =============================================================================


class TestCheckPositionConsistency:
    """Tests for check_position_consistency method."""

    @pytest.mark.asyncio
    async def test_no_ensemble_agent_returns_error(
        self, checker: EnsembleConsistencyChecker, mock_state: MagicMock
    ) -> None:
        """Test returns error when no ensemble agent is set."""
        result = await checker.check_position_consistency(mock_state)

        assert result.is_consistent is False
        assert len(result.issues) == 1
        assert result.issues[0].code == "NO_ENSEMBLE_AGENT"
        assert result.issues[0].severity == VerificationSeverity.ERROR

    @pytest.mark.asyncio
    async def test_all_agents_agree(
        self,
        config: EnsembleCheckerConfig,
        mock_state: MagicMock,
        mock_ensemble_response: MagicMock,
    ) -> None:
        """Test consistent result when all agents agree."""
        mock_agent = AsyncMock()
        mock_agent.get_best_move.return_value = mock_ensemble_response

        checker = EnsembleConsistencyChecker(ensemble_agent=mock_agent, config=config)

        with patch("src.games.chess.verification.ensemble_checker.get_routing_scores") as mock_scores:
            mock_scores.return_value = {
                "match": 1.0,
                "middlegame_fallback": 0.7,
                "phase_appropriate": 0.8,
                "phase_mismatch": 0.4,
                "default": 0.5,
            }
            result = await checker.check_position_consistency(mock_state)

        assert result.is_consistent is True
        assert result.agreement_rate == 1.0
        assert result.ensemble_move == "e2e4"
        assert result.agent_moves == {"hrm": "e2e4", "trm": "e2e4", "mcts": "e2e4"}

    @pytest.mark.asyncio
    async def test_agents_disagree(
        self,
        config: EnsembleCheckerConfig,
        mock_state: MagicMock,
        mock_disagreeing_response: MagicMock,
    ) -> None:
        """Test inconsistent result when agents disagree."""
        mock_agent = AsyncMock()
        mock_agent.get_best_move.return_value = mock_disagreeing_response

        checker = EnsembleConsistencyChecker(ensemble_agent=mock_agent, config=config)

        with patch("src.games.chess.verification.ensemble_checker.get_routing_scores") as mock_scores:
            mock_scores.return_value = {
                "match": 1.0,
                "middlegame_fallback": 0.7,
                "phase_appropriate": 0.8,
                "phase_mismatch": 0.4,
                "default": 0.5,
            }
            result = await checker.check_position_consistency(mock_state)

        # All 3 agents choose different moves: 0 agreeing pairs / 3 total pairs = 0.0
        assert result.agreement_rate == 0.0
        assert result.is_consistent is False

    @pytest.mark.asyncio
    async def test_ensemble_execution_failure(self, config: EnsembleCheckerConfig, mock_state: MagicMock) -> None:
        """Test handles ensemble execution failure."""
        mock_agent = AsyncMock()
        mock_agent.get_best_move.side_effect = RuntimeError("execution failed")

        checker = EnsembleConsistencyChecker(ensemble_agent=mock_agent, config=config)
        result = await checker.check_position_consistency(mock_state)

        assert result.is_consistent is False
        assert any(i.code == "ENSEMBLE_EXECUTION_FAILED" for i in result.issues)

    @pytest.mark.asyncio
    async def test_unexpected_exception_reraises(self, config: EnsembleCheckerConfig, mock_state: MagicMock) -> None:
        """Test unexpected exceptions are re-raised."""
        mock_agent = AsyncMock()
        mock_agent.get_best_move.side_effect = KeyboardInterrupt()

        checker = EnsembleConsistencyChecker(ensemble_agent=mock_agent, config=config)

        with pytest.raises(KeyboardInterrupt):
            await checker.check_position_consistency(mock_state)

    @pytest.mark.asyncio
    async def test_low_agreement_warning(
        self,
        mock_state: MagicMock,
        mock_disagreeing_response: MagicMock,
    ) -> None:
        """Test low agreement produces warning issue."""
        with patch(
            "src.config.settings.get_settings",
            side_effect=RuntimeError("no settings in test"),
        ):
            config = EnsembleCheckerConfig(
                agreement_threshold=0.6,
                confidence_divergence_threshold=0.3,
                routing_threshold=0.5,
                default_confidence=0.5,
                log_checks=False,
            )
        mock_agent = AsyncMock()
        mock_agent.get_best_move.return_value = mock_disagreeing_response

        checker = EnsembleConsistencyChecker(ensemble_agent=mock_agent, config=config)

        with patch("src.games.chess.verification.ensemble_checker.get_routing_scores") as mock_scores:
            mock_scores.return_value = {
                "match": 1.0,
                "middlegame_fallback": 0.7,
                "phase_appropriate": 0.8,
                "phase_mismatch": 0.4,
                "default": 0.5,
            }
            result = await checker.check_position_consistency(mock_state)

        low_agreement_issues = [i for i in result.issues if i.code == "LOW_AGREEMENT"]
        assert len(low_agreement_issues) >= 1

    @pytest.mark.asyncio
    async def test_high_divergence_warning(
        self,
        mock_state: MagicMock,
        mock_disagreeing_response: MagicMock,
    ) -> None:
        """Test high divergence produces warning issue."""
        with patch(
            "src.config.settings.get_settings",
            side_effect=RuntimeError("no settings in test"),
        ):
            config = EnsembleCheckerConfig(
                agreement_threshold=0.6,
                confidence_divergence_threshold=0.3,
                routing_threshold=0.5,
                default_confidence=0.5,
                log_checks=False,
            )
        mock_agent = AsyncMock()
        mock_agent.get_best_move.return_value = mock_disagreeing_response

        checker = EnsembleConsistencyChecker(ensemble_agent=mock_agent, config=config)

        with patch("src.games.chess.verification.ensemble_checker.get_routing_scores") as mock_scores:
            mock_scores.return_value = {
                "match": 1.0,
                "middlegame_fallback": 0.7,
                "phase_appropriate": 0.8,
                "phase_mismatch": 0.4,
                "default": 0.5,
            }
            result = await checker.check_position_consistency(mock_state)

        high_div_issues = [i for i in result.issues if i.code == "HIGH_DIVERGENCE"]
        # trm and mcts disagree with ensemble (e2e4), so they have divergence = their confidence
        # trm: 0.7 > 0.3 threshold, mcts: 0.6 > 0.3 threshold
        assert len(high_div_issues) >= 1

    @pytest.mark.asyncio
    async def test_timing_recorded(
        self,
        config: EnsembleCheckerConfig,
        mock_state: MagicMock,
        mock_ensemble_response: MagicMock,
    ) -> None:
        """Test check_time_ms is recorded."""
        mock_agent = AsyncMock()
        mock_agent.get_best_move.return_value = mock_ensemble_response

        checker = EnsembleConsistencyChecker(ensemble_agent=mock_agent, config=config)

        with patch("src.games.chess.verification.ensemble_checker.get_routing_scores") as mock_scores:
            mock_scores.return_value = {
                "match": 1.0,
                "middlegame_fallback": 0.7,
                "phase_appropriate": 0.8,
                "phase_mismatch": 0.4,
                "default": 0.5,
            }
            result = await checker.check_position_consistency(mock_state)

        assert result.check_time_ms >= 0


# =============================================================================
# check_sequence_consistency Tests
# =============================================================================


class TestCheckSequenceConsistency:
    """Tests for check_sequence_consistency method."""

    @pytest.mark.asyncio
    async def test_empty_sequence(self, checker: EnsembleConsistencyChecker) -> None:
        """Test empty sequence returns empty results."""
        results = await checker.check_sequence_consistency([])
        assert results == []

    @pytest.mark.asyncio
    async def test_multiple_states(
        self,
        config: EnsembleCheckerConfig,
        mock_ensemble_response: MagicMock,
    ) -> None:
        """Test checking multiple states."""
        mock_agent = AsyncMock()
        mock_agent.get_best_move.return_value = mock_ensemble_response

        checker = EnsembleConsistencyChecker(ensemble_agent=mock_agent, config=config)

        states = [MagicMock() for _ in range(3)]
        for s in states:
            s.fen = "test_fen"
            s.get_game_phase.return_value = GamePhase.OPENING

        with patch("src.games.chess.verification.ensemble_checker.get_routing_scores") as mock_scores:
            mock_scores.return_value = {
                "match": 1.0,
                "middlegame_fallback": 0.7,
                "phase_appropriate": 0.8,
                "phase_mismatch": 0.4,
                "default": 0.5,
            }
            results = await checker.check_sequence_consistency(states)

        assert len(results) == 3


# =============================================================================
# _calculate_agreement_rate Tests
# =============================================================================


class TestCalculateAgreementRate:
    """Tests for _calculate_agreement_rate internal method."""

    def test_empty_moves(self, checker: EnsembleConsistencyChecker) -> None:
        """Test agreement rate for empty moves."""
        rate = checker._calculate_agreement_rate({})
        assert rate == 0.0

    def test_single_agent(self, checker: EnsembleConsistencyChecker) -> None:
        """Test agreement rate with single agent."""
        rate = checker._calculate_agreement_rate({"hrm": "e2e4"})
        assert rate == 1.0

    def test_all_agree(self, checker: EnsembleConsistencyChecker) -> None:
        """Test agreement rate when all agree."""
        rate = checker._calculate_agreement_rate({"hrm": "e2e4", "trm": "e2e4", "mcts": "e2e4"})
        assert rate == 1.0

    def test_none_agree(self, checker: EnsembleConsistencyChecker) -> None:
        """Test agreement rate when none agree."""
        rate = checker._calculate_agreement_rate({"hrm": "e2e4", "trm": "d2d4", "mcts": "c2c4"})
        assert rate == 0.0

    def test_two_of_three_agree(self, checker: EnsembleConsistencyChecker) -> None:
        """Test agreement rate when 2 of 3 agree."""
        rate = checker._calculate_agreement_rate({"hrm": "e2e4", "trm": "e2e4", "mcts": "d2d4"})
        # 1 agreeing pair out of 3 total pairs = 1/3
        assert rate == pytest.approx(1.0 / 3.0)


# =============================================================================
# _calculate_divergences Tests
# =============================================================================


class TestCalculateDivergences:
    """Tests for _calculate_divergences internal method."""

    def test_all_agree_zero_divergence(self, checker: EnsembleConsistencyChecker) -> None:
        """Test zero divergence when all agents match ensemble."""
        divs = checker._calculate_divergences(
            agent_moves={"hrm": "e2e4", "trm": "e2e4"},
            agent_confidences={"hrm": 0.9, "trm": 0.8},
            ensemble_move="e2e4",
        )
        assert divs["hrm"] == 0.0
        assert divs["trm"] == 0.0

    def test_disagreeing_agent_divergence(self, checker: EnsembleConsistencyChecker) -> None:
        """Test divergence for disagreeing agent equals its confidence."""
        divs = checker._calculate_divergences(
            agent_moves={"hrm": "e2e4", "trm": "d2d4"},
            agent_confidences={"hrm": 0.9, "trm": 0.7},
            ensemble_move="e2e4",
        )
        assert divs["hrm"] == 0.0
        assert divs["trm"] == 0.7  # confidence of disagreeing agent

    def test_default_confidence_used(self, checker: EnsembleConsistencyChecker) -> None:
        """Test default confidence used when agent confidence missing."""
        divs = checker._calculate_divergences(
            agent_moves={"hrm": "d2d4"},
            agent_confidences={},  # No confidences
            ensemble_move="e2e4",
        )
        assert divs["hrm"] == 0.5  # default_confidence from config


# =============================================================================
# _calculate_move_variance Tests
# =============================================================================


class TestCalculateMoveVariance:
    """Tests for _calculate_move_variance internal method."""

    def test_single_move_variance(self, checker: EnsembleConsistencyChecker) -> None:
        """Test variance calculation with single selected move."""
        variance = checker._calculate_move_variance(
            agent_moves={"hrm": "e2e4"},
            move_probabilities={"e2e4": 0.6, "d2d4": 0.3, "c2c4": 0.1},
        )
        assert "e2e4" in variance
        # Variance = |0.6 - 1/3| = |0.6 - 0.333...| = ~0.267
        assert variance["e2e4"] == pytest.approx(abs(0.6 - 1.0 / 3.0))

    def test_multiple_moves_variance(self, checker: EnsembleConsistencyChecker) -> None:
        """Test variance with multiple different moves."""
        variance = checker._calculate_move_variance(
            agent_moves={"hrm": "e2e4", "trm": "d2d4"},
            move_probabilities={"e2e4": 0.5, "d2d4": 0.3, "c2c4": 0.2},
        )
        assert "e2e4" in variance
        assert "d2d4" in variance

    def test_empty_probabilities(self, checker: EnsembleConsistencyChecker) -> None:
        """Test variance with empty move probabilities."""
        variance = checker._calculate_move_variance(
            agent_moves={"hrm": "e2e4"},
            move_probabilities={},
        )
        assert "e2e4" in variance
        assert variance["e2e4"] == 0.0  # prob=0, expected=0


# =============================================================================
# _check_routing_consistency Tests
# =============================================================================


class TestCheckRoutingConsistency:
    """Tests for _check_routing_consistency internal method."""

    def test_opening_hrm_match(self, checker: EnsembleConsistencyChecker, mock_state: MagicMock) -> None:
        """Test perfect score for HRM in opening."""
        mock_state.get_game_phase.return_value = GamePhase.OPENING

        with patch("src.games.chess.verification.ensemble_checker.get_routing_scores") as mock_scores:
            mock_scores.return_value = {
                "match": 1.0,
                "middlegame_fallback": 0.7,
                "phase_appropriate": 0.8,
                "phase_mismatch": 0.4,
                "default": 0.5,
            }
            score = checker._check_routing_consistency(mock_state, AgentType.HRM)

        assert score == 1.0

    def test_middlegame_any_agent(self, checker: EnsembleConsistencyChecker) -> None:
        """Test middlegame gives fallback score for non-MCTS agents."""
        state = MagicMock()
        state.get_game_phase.return_value = GamePhase.MIDDLEGAME

        with patch("src.games.chess.verification.ensemble_checker.get_routing_scores") as mock_scores:
            mock_scores.return_value = {
                "match": 1.0,
                "middlegame_fallback": 0.7,
                "phase_appropriate": 0.8,
                "phase_mismatch": 0.4,
                "default": 0.5,
            }
            score = checker._check_routing_consistency(state, AgentType.HRM)

        assert score == 0.7  # middlegame_fallback

    def test_middlegame_mcts_match(self, checker: EnsembleConsistencyChecker) -> None:
        """Test middlegame perfect score for MCTS."""
        state = MagicMock()
        state.get_game_phase.return_value = GamePhase.MIDDLEGAME

        with patch("src.games.chess.verification.ensemble_checker.get_routing_scores") as mock_scores:
            mock_scores.return_value = {
                "match": 1.0,
                "middlegame_fallback": 0.7,
                "phase_appropriate": 0.8,
                "phase_mismatch": 0.4,
                "default": 0.5,
            }
            score = checker._check_routing_consistency(state, AgentType.MCTS)

        assert score == 1.0

    def test_endgame_trm_match(self, checker: EnsembleConsistencyChecker) -> None:
        """Test endgame perfect score for TRM."""
        state = MagicMock()
        state.get_game_phase.return_value = GamePhase.ENDGAME

        with patch("src.games.chess.verification.ensemble_checker.get_routing_scores") as mock_scores:
            mock_scores.return_value = {
                "match": 1.0,
                "middlegame_fallback": 0.7,
                "phase_appropriate": 0.8,
                "phase_mismatch": 0.4,
                "default": 0.5,
            }
            score = checker._check_routing_consistency(state, AgentType.TRM)

        assert score == 1.0

    def test_endgame_mcts_appropriate(self, checker: EnsembleConsistencyChecker) -> None:
        """Test endgame partial credit for MCTS."""
        state = MagicMock()
        state.get_game_phase.return_value = GamePhase.ENDGAME

        with patch("src.games.chess.verification.ensemble_checker.get_routing_scores") as mock_scores:
            mock_scores.return_value = {
                "match": 1.0,
                "middlegame_fallback": 0.7,
                "phase_appropriate": 0.8,
                "phase_mismatch": 0.4,
                "default": 0.5,
            }
            score = checker._check_routing_consistency(state, AgentType.MCTS)

        assert score == 0.8  # phase_appropriate

    def test_opening_trm_mismatch(self, checker: EnsembleConsistencyChecker) -> None:
        """Test opening low score for TRM (mismatch)."""
        state = MagicMock()
        state.get_game_phase.return_value = GamePhase.OPENING

        with patch("src.games.chess.verification.ensemble_checker.get_routing_scores") as mock_scores:
            mock_scores.return_value = {
                "match": 1.0,
                "middlegame_fallback": 0.7,
                "phase_appropriate": 0.8,
                "phase_mismatch": 0.4,
                "default": 0.5,
            }
            score = checker._check_routing_consistency(state, AgentType.TRM)

        assert score == 0.4  # phase_mismatch

    def test_endgame_hrm_mismatch(self, checker: EnsembleConsistencyChecker) -> None:
        """Test endgame low score for HRM (mismatch)."""
        state = MagicMock()
        state.get_game_phase.return_value = GamePhase.ENDGAME

        with patch("src.games.chess.verification.ensemble_checker.get_routing_scores") as mock_scores:
            mock_scores.return_value = {
                "match": 1.0,
                "middlegame_fallback": 0.7,
                "phase_appropriate": 0.8,
                "phase_mismatch": 0.4,
                "default": 0.5,
            }
            score = checker._check_routing_consistency(state, AgentType.HRM)

        assert score == 0.4  # phase_mismatch


# =============================================================================
# _get_expected_agent_for_phase Tests
# =============================================================================


class TestGetExpectedAgentForPhase:
    """Tests for _get_expected_agent_for_phase internal method."""

    def test_opening(self, checker: EnsembleConsistencyChecker) -> None:
        """Test opening phase returns HRM."""
        assert checker._get_expected_agent_for_phase(GamePhase.OPENING) == AgentType.HRM

    def test_middlegame(self, checker: EnsembleConsistencyChecker) -> None:
        """Test middlegame phase returns MCTS."""
        assert checker._get_expected_agent_for_phase(GamePhase.MIDDLEGAME) == AgentType.MCTS

    def test_endgame(self, checker: EnsembleConsistencyChecker) -> None:
        """Test endgame phase returns TRM."""
        assert checker._get_expected_agent_for_phase(GamePhase.ENDGAME) == AgentType.TRM

    def test_custom_phase_agents(self) -> None:
        """Test custom phase agent configuration."""
        with patch(
            "src.config.settings.get_settings",
            side_effect=RuntimeError("no settings in test"),
        ):
            config = EnsembleCheckerConfig(
                opening_expected_agent=AgentType.MCTS,
                middlegame_expected_agent=AgentType.HRM,
                endgame_expected_agent=AgentType.MCTS,
            )
        checker = EnsembleConsistencyChecker(config=config)

        assert checker._get_expected_agent_for_phase(GamePhase.OPENING) == AgentType.MCTS
        assert checker._get_expected_agent_for_phase(GamePhase.MIDDLEGAME) == AgentType.HRM
        assert checker._get_expected_agent_for_phase(GamePhase.ENDGAME) == AgentType.MCTS


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateEnsembleChecker:
    """Tests for create_ensemble_checker factory function."""

    def _mock_settings(self):
        """Context manager to mock get_settings."""
        return patch(
            "src.config.settings.get_settings",
            side_effect=RuntimeError("no settings in test"),
        )

    def test_create_without_args(self) -> None:
        """Test factory creates checker without arguments."""
        with self._mock_settings():
            checker = create_ensemble_checker()
        assert isinstance(checker, EnsembleConsistencyChecker)
        assert checker.ensemble_agent is None

    def test_create_with_agent(self) -> None:
        """Test factory creates checker with agent."""
        mock_agent = MagicMock()
        with self._mock_settings():
            checker = create_ensemble_checker(ensemble_agent=mock_agent)
        assert checker.ensemble_agent is mock_agent

    def test_create_with_config(self) -> None:
        """Test factory creates checker with custom config."""
        with self._mock_settings():
            config = EnsembleCheckerConfig(agreement_threshold=0.99)
        checker = create_ensemble_checker(config=config)
        assert checker.config.agreement_threshold == 0.99

    def test_create_with_both(self) -> None:
        """Test factory creates checker with both agent and config."""
        mock_agent = MagicMock()
        with self._mock_settings():
            config = EnsembleCheckerConfig(routing_threshold=0.9)
        checker = create_ensemble_checker(ensemble_agent=mock_agent, config=config)
        assert checker.ensemble_agent is mock_agent
        assert checker.config.routing_threshold == 0.9
