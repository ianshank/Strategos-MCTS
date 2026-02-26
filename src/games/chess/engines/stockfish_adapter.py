"""
Stockfish Engine Adapter.

Provides an adapter for integrating Stockfish chess engine
for analysis and evaluation.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.games.chess.ensemble_agent import ChessEnsembleAgent

from src.games.chess.constants import get_stockfish_executables
from src.games.chess.state import ChessGameState
from src.observability.logging import StructuredLogger, get_structured_logger


@dataclass
class StockfishConfig:
    """Configuration for Stockfish adapter.

    All parameters are configurable - no hardcoded values.
    """

    # Path configuration
    stockfish_path: str | None = None

    # Engine settings
    elo_limit: int | None = None
    hash_size_mb: int = 128
    threads: int = 1

    # Analysis settings
    default_depth: int = 20
    comparison_depth: int = 15  # Lower depth for faster agent comparison
    time_limit_ms: int = 1000
    multipv: int = 1

    # Evaluation settings
    skill_level: int | None = None

    def __post_init__(self) -> None:
        """Apply environment overrides."""
        if self.stockfish_path is None:
            self.stockfish_path = os.getenv("STOCKFISH_PATH")


@dataclass
class StockfishAnalysis:
    """Result of Stockfish analysis."""

    best_move: str | None
    evaluation_cp: int  # Centipawns
    evaluation_mate: int | None  # Mate in N (positive = white, negative = black)
    depth: int
    nodes: int
    time_ms: float
    pv: list[str]  # Principal variation
    extra_info: dict[str, Any] = field(default_factory=dict)

    @property
    def evaluation_score(self) -> float:
        """Get normalized evaluation score (-1 to 1)."""
        if self.evaluation_mate is not None:
            return 1.0 if self.evaluation_mate > 0 else -1.0

        # Sigmoid-like transformation of centipawn score
        import math

        cp = self.evaluation_cp / 100.0
        return 2.0 / (1.0 + math.exp(-0.4 * cp)) - 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "best_move": self.best_move,
            "evaluation_cp": self.evaluation_cp,
            "evaluation_mate": self.evaluation_mate,
            "depth": self.depth,
            "nodes": self.nodes,
            "time_ms": self.time_ms,
            "pv": self.pv,
            "evaluation_score": round(self.evaluation_score, 4),
        }


@dataclass
class EvaluationResult:
    """Result of evaluation against Stockfish."""

    total_games: int
    agent_wins: int
    stockfish_wins: int
    draws: int
    avg_move_diff_cp: float
    agreement_rate: float
    games: list[dict[str, Any]] = field(default_factory=list)

    @property
    def agent_win_rate(self) -> float:
        """Calculate agent win rate."""
        if self.total_games == 0:
            return 0.0
        return self.agent_wins / self.total_games


class StockfishAdapter:
    """Adapter for Stockfish chess engine integration.

    Provides methods for position analysis, move evaluation,
    and agent vs engine comparison.

    Example:
        >>> adapter = StockfishAdapter()
        >>> analysis = await adapter.analyze_position(state, depth=20)
        >>> print(f"Best move: {analysis.best_move}")
    """

    def __init__(
        self,
        config: StockfishConfig | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        """Initialize the Stockfish adapter.

        Args:
            config: Stockfish configuration
            logger: Optional logger instance
        """
        self._config = config or StockfishConfig()
        self._logger = logger or get_structured_logger("chess.engines.stockfish")
        self._engine: Any | None = None
        self._available: bool | None = None

    @property
    def config(self) -> StockfishConfig:
        """Get the configuration."""
        return self._config

    @property
    def is_available(self) -> bool:
        """Check if Stockfish is available."""
        if self._available is not None:
            return self._available

        self._available = self._find_stockfish() is not None
        return self._available

    def _find_stockfish(self) -> str | None:
        """Find Stockfish executable.

        Returns:
            Path to Stockfish executable or None if not found
        """
        # Check configured path
        if self._config.stockfish_path and os.path.isfile(self._config.stockfish_path):
            return self._config.stockfish_path

        # Check common paths using centralized constants
        for name in get_stockfish_executables():
            path = shutil.which(name)
            if path:
                return path

        return None

    async def _ensure_engine(self) -> bool:
        """Ensure engine is initialized.

        Returns:
            True if engine is ready, False otherwise
        """
        if self._engine is not None:
            return True

        stockfish_path = self._find_stockfish()
        if stockfish_path is None:
            self._logger.warning("Stockfish not found")
            return False

        try:
            import chess.engine

            self._engine = await chess.engine.SimpleEngine.popen_uci(stockfish_path)

            # Configure engine
            if self._config.hash_size_mb:
                await self._engine.configure({"Hash": self._config.hash_size_mb})
            if self._config.threads:
                await self._engine.configure({"Threads": self._config.threads})
            if self._config.skill_level is not None:
                await self._engine.configure({"Skill Level": self._config.skill_level})

            self._logger.info(
                "Stockfish initialized",
                path=stockfish_path,
                hash_mb=self._config.hash_size_mb,
                threads=self._config.threads,
            )

            return True

        except Exception as e:
            self._logger.error(f"Failed to initialize Stockfish: {e}")
            return False

    async def analyze_position(
        self,
        state: ChessGameState,
        depth: int | None = None,
        time_limit_ms: int | None = None,
    ) -> StockfishAnalysis:
        """Analyze a chess position.

        Args:
            state: Chess position to analyze
            depth: Analysis depth (uses config default if None)
            time_limit_ms: Time limit in milliseconds

        Returns:
            StockfishAnalysis with analysis results

        Raises:
            RuntimeError: If Stockfish is not available
        """
        import time

        if not await self._ensure_engine():
            raise RuntimeError("Stockfish is not available")

        import chess
        import chess.engine

        depth = depth or self._config.default_depth
        time_limit_ms = time_limit_ms or self._config.time_limit_ms

        board = chess.Board(state.fen)
        limit = chess.engine.Limit(
            depth=depth,
            time=time_limit_ms / 1000.0,
        )

        start_time = time.perf_counter()
        assert self._engine is not None, "Engine not initialized. Call initialize() first."
        info = await self._engine.analyse(board, limit, multipv=self._config.multipv)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Parse results
        if isinstance(info, list):
            info = info[0] if info else {}

        score = info.get("score")
        pv = info.get("pv", [])

        # Extract evaluation
        evaluation_cp = 0
        evaluation_mate = None

        if score:
            if score.is_mate():
                evaluation_mate = score.white().mate()
                evaluation_cp = 10000 if evaluation_mate > 0 else -10000
            else:
                evaluation_cp = score.white().score(mate_score=10000) or 0

        return StockfishAnalysis(
            best_move=pv[0].uci() if pv else None,
            evaluation_cp=evaluation_cp,
            evaluation_mate=evaluation_mate,
            depth=info.get("depth", depth),
            nodes=info.get("nodes", 0),
            time_ms=elapsed_ms,
            pv=[m.uci() for m in pv],
        )

    async def get_best_move(
        self,
        state: ChessGameState,
        time_limit_ms: int | None = None,
    ) -> tuple[str, float]:
        """Get best move for a position.

        Args:
            state: Chess position
            time_limit_ms: Time limit in milliseconds

        Returns:
            Tuple of (best_move_uci, evaluation_score)
        """
        analysis = await self.analyze_position(
            state,
            time_limit_ms=time_limit_ms,
        )

        return (
            analysis.best_move or state.get_legal_actions()[0],
            analysis.evaluation_score,
        )

    async def evaluate_vs_agent(
        self,
        agent: ChessEnsembleAgent,
        num_games: int = 10,
        max_moves_per_game: int = 100,
    ) -> EvaluationResult:
        """Evaluate agent performance against Stockfish.

        Args:
            agent: Chess agent to evaluate
            num_games: Number of games to play
            max_moves_per_game: Maximum moves per game

        Returns:
            EvaluationResult with comparison statistics
        """
        if not await self._ensure_engine():
            raise RuntimeError("Stockfish is not available")

        agent_wins = 0
        stockfish_wins = 0
        draws = 0
        total_move_diff = 0.0
        move_count = 0
        agreements = 0
        games: list[dict[str, Any]] = []

        for game_num in range(num_games):
            state = ChessGameState.initial()
            game_moves: list[str] = []
            agent_plays_white = game_num % 2 == 0

            for _move_num in range(max_moves_per_game):
                if state.is_terminal():
                    break

                agent_turn = (state.current_player == 1) == agent_plays_white

                if agent_turn:
                    # Agent's turn
                    response = await agent.get_best_move(state, temperature=0.0)
                    move = response.best_move

                    # Compare with Stockfish (using comparison_depth for faster evaluation)
                    sf_analysis = await self.analyze_position(state, depth=self._config.comparison_depth)
                    if sf_analysis.best_move:
                        total_move_diff += abs(response.value_estimate - sf_analysis.evaluation_score)
                        move_count += 1
                        if move == sf_analysis.best_move:
                            agreements += 1
                else:
                    # Stockfish's turn
                    sf_analysis = await self.analyze_position(state)
                    move = sf_analysis.best_move or state.get_legal_actions()[0]

                game_moves.append(move)
                state = state.apply_action(move)

            # Determine game result
            result_str = "incomplete"
            if state.is_terminal():
                reward = state.get_reward(1 if agent_plays_white else -1)
                if reward > 0:
                    agent_wins += 1
                    result_str = "agent_win"
                elif reward < 0:
                    stockfish_wins += 1
                    result_str = "stockfish_win"
                else:
                    draws += 1
                    result_str = "draw"

            games.append(
                {
                    "game_num": game_num,
                    "agent_color": "white" if agent_plays_white else "black",
                    "moves": game_moves,
                    "result": result_str,
                }
            )

        return EvaluationResult(
            total_games=num_games,
            agent_wins=agent_wins,
            stockfish_wins=stockfish_wins,
            draws=draws,
            avg_move_diff_cp=total_move_diff / max(move_count, 1) * 100,
            agreement_rate=agreements / max(move_count, 1),
            games=games,
        )

    async def close(self) -> None:
        """Close the engine."""
        if self._engine:
            await self._engine.quit()
            self._engine = None

    async def __aenter__(self) -> StockfishAdapter:
        """Async context manager entry."""
        await self._ensure_engine()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


def create_stockfish_adapter(
    config: StockfishConfig | None = None,
) -> StockfishAdapter:
    """Factory function to create a StockfishAdapter.

    Args:
        config: Optional Stockfish configuration

    Returns:
        Configured StockfishAdapter instance
    """
    return StockfishAdapter(config=config)
