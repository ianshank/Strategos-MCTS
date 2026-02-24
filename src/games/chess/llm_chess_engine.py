"""
LLM-powered Chess Engine using Multi-Agent MCTS Framework.

Orchestrates all framework agents (HRM, TRM, MCTS, Meta-Controller) to
analyze chess positions and select moves through LLM-based reasoning.

Works with optional python-chess: when available, provides full move
validation and legal move generation; without it, relies on FEN parsing
and LLM-based move analysis.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from src.adapters.llm.base import LLMClient
from src.framework.agents.base import (
    AgentContext,
    AgentResult,
    AsyncAgentBase,
    ParallelAgent,
    SequentialAgent,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional python-chess import
# ---------------------------------------------------------------------------

try:
    import chess as _chess

    CHESS_AVAILABLE = True
except ImportError:
    _chess = None  # type: ignore[assignment]
    CHESS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configurable constants (no hardcoded magic numbers)
# ---------------------------------------------------------------------------

DEFAULT_CHESS_TEMPERATURE = 0.3
DEFAULT_CHESS_MAX_TOKENS = 1000
DEFAULT_MCTS_DEPTH = 8
DEFAULT_TOP_MOVES = 5
DEFAULT_CONSENSUS_TOP_K = 3

# Game phase thresholds
PHASE_OPENING_THRESHOLD = 10  # move number
PHASE_ENDGAME_MATERIAL = 26  # total material below this

# Routing weights per phase  {agent: weight}
OPENING_WEIGHTS = {"hrm": 0.5, "trm": 0.2, "mcts": 0.3}
MIDDLEGAME_WEIGHTS = {"hrm": 0.2, "trm": 0.3, "mcts": 0.5}
ENDGAME_WEIGHTS = {"hrm": 0.1, "trm": 0.5, "mcts": 0.4}

# Material piece values for phase detection
PIECE_VALUES = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}

# UCI move pattern
UCI_MOVE_PATTERN = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")

# ---------------------------------------------------------------------------
# Chess-specific prompts
# ---------------------------------------------------------------------------

POSITION_DESCRIPTION_TEMPLATE = """\
## Chess Position (FEN)
{fen}

## Board
{board_ascii}

## Position Info
- Side to move: {side_to_move}
- Move number: {move_number}
- Game phase: {game_phase}
- Material balance: {material_balance}
{legal_moves_section}\
"""

HRM_CHESS_PROMPT = """\
You are a chess Hierarchical Reasoning Module (HRM). Analyze this position by
decomposing it into strategic sub-problems, solving each, then synthesizing
a move recommendation.

{position_description}

## Instructions
1. Decompose the position into 2-4 strategic themes (e.g., pawn structure,
   king safety, piece activity, center control)
2. Analyze each theme independently
3. Synthesize into a concrete move recommendation in UCI format (e.g., e2e4)

## Response Format
### Sub-problems
1. <theme>
   **Analysis:** <analysis>
   **Implication:** <what this means for move choice>

### Synthesis
**Recommended move:** <UCI move>
**Reasoning:** <brief explanation>
"""

TRM_CHESS_PROMPT = """\
You are a chess Task Refinement Module (TRM). Evaluate candidate moves for
this position through iterative refinement.

{position_description}

## Instructions
1. Suggest an initial candidate move
2. Critically evaluate it - what are the weaknesses?
3. Consider alternatives and produce a refined recommendation

## Response Format
### Initial Candidate
**Move:** <UCI move>
**Reasoning:** <why this move>

### Critical Review
- Weakness: <issue with initial move>
- Alternative: <better option and why>

### Refined Recommendation
**Move:** <UCI move>
**Score:** <0.0-1.0 confidence>
**Reasoning:** <final justification>
"""

MCTS_CHESS_STRATEGIES: dict[str, str] = {
    "tactical": (
        "Analyze this chess position for TACTICAL opportunities. "
        "Look for captures, checks, pins, forks, skewers, discovered attacks, "
        "and forcing sequences.\n\n"
        "{position_description}\n\n"
        "Recommend the best tactical move in UCI format (e.g., e2e4). "
        "If no tactics exist, suggest the most forcing move.\n"
        "**Move:** <UCI>\n**Score:** <0.0-1.0>\n**Reasoning:** <explanation>"
    ),
    "positional": (
        "Analyze this chess position for POSITIONAL factors. "
        "Consider piece placement, pawn structure, outposts, weak squares, "
        "file control, and piece coordination.\n\n"
        "{position_description}\n\n"
        "Recommend the best positional move in UCI format (e.g., e2e4).\n"
        "**Move:** <UCI>\n**Score:** <0.0-1.0>\n**Reasoning:** <explanation>"
    ),
    "prophylactic": (
        "Analyze this chess position with PROPHYLACTIC thinking. "
        "Consider what your opponent wants to do and how to prevent it. "
        "Think about defensive needs and improving piece placement.\n\n"
        "{position_description}\n\n"
        "Recommend the best prophylactic move in UCI format (e.g., e2e4).\n"
        "**Move:** <UCI>\n**Score:** <0.0-1.0>\n**Reasoning:** <explanation>"
    ),
    "endgame": (
        "Analyze this chess position with ENDGAME principles. "
        "Consider king activity, passed pawns, opposition, zugzwang, "
        "pawn promotion paths, and piece exchanges.\n\n"
        "{position_description}\n\n"
        "Recommend the best endgame-oriented move in UCI format (e.g., e2e4).\n"
        "**Move:** <UCI>\n**Score:** <0.0-1.0>\n**Reasoning:** <explanation>"
    ),
}

CONSENSUS_CHESS_PROMPT = """\
You are synthesizing chess analysis from multiple agents. Each agent analyzed
the same position using different approaches and recommended moves.

{position_description}

## Agent Recommendations
{agent_summaries}

## Instructions
Evaluate all recommendations and select the single best move.
Consider the reasoning quality, tactical soundness, and positional merit.

**Best Move:** <UCI move>
**Confidence:** <0.0-1.0>
**Reasoning:** <why this move is best, integrating insights from all agents>
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RoutingDecision:
    """Result of meta-controller routing decision (LLM chess variant)."""

    primary_agent: str  # "hrm", "trm", or "mcts"
    agent_weights: dict[str, float]
    confidence: float
    game_phase: str
    reasoning: str


@dataclass
class ChessMoveResult:
    """Result from a single agent's move analysis."""

    move: str  # UCI format
    score: float
    reasoning: str
    agent_name: str
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChessAnalysis:
    """Full position analysis combining all agents."""

    best_move: str
    candidate_moves: list[ChessMoveResult]
    routing_decision: RoutingDecision
    agent_results: dict[str, ChessMoveResult]
    consensus_move: str | None = None
    consensus_reasoning: str | None = None
    total_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM Chess Meta-Controller
# ---------------------------------------------------------------------------


class LLMChessMetaController:
    """Rule-based routing between LLM agents for chess positions.

    Routes to HRM for strategic/opening positions, TRM for endgames
    requiring precision, and MCTS for complex middlegame positions.
    """

    def __init__(
        self,
        opening_weights: dict[str, float] | None = None,
        middlegame_weights: dict[str, float] | None = None,
        endgame_weights: dict[str, float] | None = None,
        opening_threshold: int = PHASE_OPENING_THRESHOLD,
        endgame_material: int = PHASE_ENDGAME_MATERIAL,
    ):
        self._opening_weights = opening_weights or OPENING_WEIGHTS.copy()
        self._middlegame_weights = middlegame_weights or MIDDLEGAME_WEIGHTS.copy()
        self._endgame_weights = endgame_weights or ENDGAME_WEIGHTS.copy()
        self._opening_threshold = opening_threshold
        self._endgame_material = endgame_material

    def route(self, fen: str) -> RoutingDecision:
        """Decide which agent to use based on position features.

        Args:
            fen: FEN string of the current position.

        Returns:
            RoutingDecision with agent selection and weights.
        """
        move_number = self._get_move_number(fen)
        material = self._get_total_material(fen)
        phase = self._classify_phase(move_number, material)

        if phase == "opening":
            weights = self._opening_weights.copy()
        elif phase == "endgame":
            weights = self._endgame_weights.copy()
        else:
            weights = self._middlegame_weights.copy()

        # Adjust for tactical features
        if self._has_check(fen):
            weights["mcts"] = weights.get("mcts", 0.0) * 1.5
            weights["hrm"] = weights.get("hrm", 0.0) * 0.7

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        primary = max(weights, key=lambda k: weights[k])
        sorted_w = sorted(weights.values(), reverse=True)
        confidence = (sorted_w[0] - sorted_w[1] + 0.3) if len(sorted_w) > 1 else 1.0
        confidence = min(1.0, confidence)

        reasoning = self._build_reasoning(phase, primary)

        return RoutingDecision(
            primary_agent=primary,
            agent_weights=weights,
            confidence=confidence,
            game_phase=phase,
            reasoning=reasoning,
        )

    # -- helpers --

    @staticmethod
    def _get_move_number(fen: str) -> int:
        parts = fen.split()
        if len(parts) >= 6:
            try:
                return int(parts[5])
            except ValueError:
                pass
        return 1

    @staticmethod
    def _get_total_material(fen: str) -> int:
        board_part = fen.split()[0] if fen else ""
        total = 0
        for ch in board_part:
            upper = ch.upper()
            if upper in PIECE_VALUES:
                total += PIECE_VALUES[upper]
        return total

    @staticmethod
    def _has_check(fen: str) -> bool:
        if CHESS_AVAILABLE:
            try:
                board = _chess.Board(fen)
                return board.is_check()
            except (ValueError, AttributeError):
                pass
        return False

    def _classify_phase(self, move_number: int, material: int) -> str:
        if move_number <= self._opening_threshold:
            return "opening"
        if material < self._endgame_material:
            return "endgame"
        return "middlegame"

    @staticmethod
    def _build_reasoning(phase: str, primary: str) -> str:
        reasons = {
            ("opening", "hrm"): "Opening phase — strategic planning via HRM",
            ("opening", "mcts"): "Opening exploration via MCTS",
            ("middlegame", "mcts"): "Complex middlegame — deep search via MCTS",
            ("middlegame", "trm"): "Middlegame refinement via TRM",
            ("endgame", "trm"): "Endgame precision via TRM iterative refinement",
            ("endgame", "mcts"): "Endgame search via MCTS",
        }
        return reasons.get(
            (phase, primary),
            f"{phase.capitalize()} phase — routing to {primary.upper()}",
        )


# ---------------------------------------------------------------------------
# Position description helpers
# ---------------------------------------------------------------------------


def fen_to_board_ascii(fen: str) -> str:
    """Convert FEN to ASCII board representation."""
    if CHESS_AVAILABLE:
        try:
            return str(_chess.Board(fen))
        except (ValueError, AttributeError):
            pass
    # Fallback: simple rendering
    board_part = fen.split()[0] if fen else ""
    rows = board_part.split("/")
    lines = []
    for rank_idx, row in enumerate(rows):
        line = ""
        for ch in row:
            if ch.isdigit():
                line += ". " * int(ch)
            else:
                line += ch + " "
        lines.append(f"  {8 - rank_idx} {line.strip()}")
    lines.append("    a b c d e f g h")
    return "\n".join(lines)


def get_legal_moves_list(fen: str) -> list[str] | None:
    """Return legal moves in UCI format, or None if python-chess unavailable."""
    if not CHESS_AVAILABLE:
        return None
    try:
        board = _chess.Board(fen)
        return [m.uci() for m in board.legal_moves]
    except (ValueError, AttributeError):
        return None


def describe_position(fen: str) -> str:
    """Build a natural-language description of the position for LLM prompts."""
    parts = fen.split()
    side = "White" if (len(parts) > 1 and parts[1] == "w") else "Black"
    move_num = int(parts[5]) if len(parts) >= 6 else 1

    material = LLMChessMetaController._get_total_material(fen)
    if move_num <= PHASE_OPENING_THRESHOLD:
        phase = "Opening"
    elif material < PHASE_ENDGAME_MATERIAL:
        phase = "Endgame"
    else:
        phase = "Middlegame"

    legal = get_legal_moves_list(fen)
    legal_section = ""
    if legal is not None:
        display = ", ".join(legal[:20])
        if len(legal) > 20:
            display += f"  ... ({len(legal)} total)"
        legal_section = f"- Legal moves: {display}\n"

    # Material balance description
    board_part = parts[0] if parts else ""
    white_mat = sum(PIECE_VALUES.get(c, 0) for c in board_part if c.isupper())
    black_mat = sum(PIECE_VALUES.get(c.upper(), 0) for c in board_part if c.islower())
    balance = white_mat - black_mat
    if balance > 0:
        balance_str = f"White +{balance}"
    elif balance < 0:
        balance_str = f"Black +{abs(balance)}"
    else:
        balance_str = "Equal"

    return POSITION_DESCRIPTION_TEMPLATE.format(
        fen=fen,
        board_ascii=fen_to_board_ascii(fen),
        side_to_move=side,
        move_number=move_num,
        game_phase=phase,
        material_balance=balance_str,
        legal_moves_section=legal_section,
    )


def extract_uci_move(text: str) -> str | None:
    """Extract a UCI move from LLM response text."""
    # Try explicit **Move:** or Move: patterns first
    move_patterns = [
        r"\*\*(?:Recommended\s+)?[Mm]ove:\*\*\s*([a-h][1-8][a-h][1-8][qrbn]?)",
        r"[Mm]ove:\s*([a-h][1-8][a-h][1-8][qrbn]?)",
        r"\*\*Best Move:\*\*\s*([a-h][1-8][a-h][1-8][qrbn]?)",
    ]
    for pattern in move_patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1)

    # Fallback: find any UCI-like move in the text
    for word in text.split():
        cleaned = word.strip(".,;:!?*`'\"()[]")
        if UCI_MOVE_PATTERN.match(cleaned):
            return cleaned
    return None


def extract_score(text: str) -> float:
    """Extract a numeric score from LLM response text."""
    score_patterns = [
        r"\*\*Score:\*\*\s*([\d.]+)",
        r"Score:\s*([\d.]+)",
        r"\*\*Confidence:\*\*\s*([\d.]+)",
        r"Confidence:\s*([\d.]+)",
    ]
    for pattern in score_patterns:
        m = re.search(pattern, text)
        if m:
            try:
                val = float(m.group(1))
                return min(max(val, 0.0), 1.0)
            except ValueError:
                continue
    return 0.5  # default


# ---------------------------------------------------------------------------
# LLM Chess Agent (wraps HRM/TRM with chess prompts)
# ---------------------------------------------------------------------------


class LLMChessHRMAgent(AsyncAgentBase):
    """Chess-specialised HRM agent for hierarchical position analysis."""

    def __init__(
        self,
        model_adapter: LLMClient,
        name: str = "Chess_HRM",
        temperature: float = DEFAULT_CHESS_TEMPERATURE,
        max_tokens: int = DEFAULT_CHESS_MAX_TOKENS,
        **config: Any,
    ):
        super().__init__(model_adapter, name=name, **config)
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def _process_impl(self, context: AgentContext) -> AgentResult:
        pos_desc = describe_position(context.query)
        prompt = HRM_CHESS_PROMPT.format(position_description=pos_desc)

        response = await self.generate_llm_response(
            prompt=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        text = response.text
        move = extract_uci_move(text) or "e2e4"
        score = extract_score(text)

        return AgentResult(
            response=text,
            confidence=score,
            metadata={
                "move": move,
                "score": score,
                "strategy": "hierarchical_decomposition",
                "agent": "chess_hrm",
            },
            token_usage=response.usage,
        )


class LLMChessTRMAgent(AsyncAgentBase):
    """Chess-specialised TRM agent for iterative move refinement."""

    def __init__(
        self,
        model_adapter: LLMClient,
        name: str = "Chess_TRM",
        temperature: float = DEFAULT_CHESS_TEMPERATURE,
        max_tokens: int = DEFAULT_CHESS_MAX_TOKENS,
        **config: Any,
    ):
        super().__init__(model_adapter, name=name, **config)
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def _process_impl(self, context: AgentContext) -> AgentResult:
        pos_desc = describe_position(context.query)
        prompt = TRM_CHESS_PROMPT.format(position_description=pos_desc)

        response = await self.generate_llm_response(
            prompt=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        text = response.text
        move = extract_uci_move(text) or "e2e4"
        score = extract_score(text)

        return AgentResult(
            response=text,
            confidence=score,
            metadata={
                "move": move,
                "score": score,
                "strategy": "iterative_refinement",
                "agent": "chess_trm",
            },
            token_usage=response.usage,
        )


class LLMChessMCTSAgent(AsyncAgentBase):
    """Chess-specialised MCTS agent exploring multiple chess strategies."""

    def __init__(
        self,
        model_adapter: LLMClient,
        name: str = "Chess_MCTS",
        temperature: float = DEFAULT_CHESS_TEMPERATURE,
        max_tokens: int = DEFAULT_CHESS_MAX_TOKENS,
        strategies: list[str] | None = None,
        **config: Any,
    ):
        super().__init__(model_adapter, name=name, **config)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._strategies = strategies or list(MCTS_CHESS_STRATEGIES.keys())

    async def _process_impl(self, context: AgentContext) -> AgentResult:
        pos_desc = describe_position(context.query)

        # Run all strategies in parallel
        tasks = []
        for strat_name in self._strategies:
            template = MCTS_CHESS_STRATEGIES.get(strat_name, MCTS_CHESS_STRATEGIES["tactical"])
            prompt = template.format(position_description=pos_desc)
            tasks.append(self._run_strategy(strat_name, prompt))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results
        strategy_results: list[dict[str, Any]] = []
        for res in results:
            if isinstance(res, dict):
                strategy_results.append(res)

        if not strategy_results:
            return AgentResult(
                response="No strategies produced results",
                confidence=0.0,
                metadata={"move": "e2e4", "score": 0.0, "agent": "chess_mcts"},
            )

        # Pick best scoring strategy result
        best = max(strategy_results, key=lambda r: r.get("score", 0.0))

        return AgentResult(
            response=best.get("text", ""),
            confidence=best.get("score", 0.5),
            metadata={
                "move": best.get("move", "e2e4"),
                "score": best.get("score", 0.5),
                "strategy": best.get("strategy", "tactical"),
                "all_strategies": strategy_results,
                "agent": "chess_mcts",
            },
            token_usage=best.get("usage", {}),
        )

    async def _run_strategy(self, strategy_name: str, prompt: str) -> dict[str, Any]:
        """Run a single MCTS strategy and return parsed result."""
        response = await self.generate_llm_response(
            prompt=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        text = response.text
        move = extract_uci_move(text) or "e2e4"
        score = extract_score(text)
        return {
            "strategy": strategy_name,
            "move": move,
            "score": score,
            "text": text,
            "usage": response.usage,
        }


# ---------------------------------------------------------------------------
# LLM Chess Engine (orchestrator)
# ---------------------------------------------------------------------------


class LLMChessEngine:
    """Orchestrates all LLM agents for chess position analysis.

    Uses:
    - LLMChessMetaController for routing
    - LLMChessHRMAgent for strategic decomposition
    - LLMChessTRMAgent for move refinement
    - LLMChessMCTSAgent for multi-strategy exploration
    - ParallelAgent for concurrent execution
    - SequentialAgent for analysis→refinement pipelines
    - Consensus synthesis for combining agent outputs
    """

    def __init__(
        self,
        model_adapter: LLMClient,
        *,
        mcts_depth: int = DEFAULT_MCTS_DEPTH,
        temperature: float = DEFAULT_CHESS_TEMPERATURE,
        max_tokens: int = DEFAULT_CHESS_MAX_TOKENS,
        consensus_top_k: int = DEFAULT_CONSENSUS_TOP_K,
        strategies: list[str] | None = None,
    ):
        self._adapter = model_adapter
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._mcts_depth = mcts_depth
        self._consensus_top_k = consensus_top_k
        self._move_count = 0

        # Meta-controller
        self.meta_controller = LLMChessMetaController()

        # Agents
        self.hrm_agent = LLMChessHRMAgent(
            model_adapter, temperature=temperature, max_tokens=max_tokens,
        )
        self.trm_agent = LLMChessTRMAgent(
            model_adapter, temperature=temperature, max_tokens=max_tokens,
        )
        self.mcts_agent = LLMChessMCTSAgent(
            model_adapter, temperature=temperature, max_tokens=max_tokens,
            strategies=strategies,
        )

        # Composite agents
        self.parallel_agent = ParallelAgent(
            model_adapter,
            name="ChessParallel",
            sub_agents=[self.hrm_agent, self.trm_agent],
        )
        self.sequential_agent = SequentialAgent(
            model_adapter,
            name="ChessSequential",
            sub_agents=[self.hrm_agent, self.trm_agent],
        )

        logger.info(
            "LLMChessEngine initialised: depth=%d, temperature=%.2f, strategies=%s",
            mcts_depth, temperature, strategies or list(MCTS_CHESS_STRATEGIES.keys()),
        )

    async def analyze_position(self, fen: str) -> ChessAnalysis:
        """Analyse a chess position using all agents.

        Args:
            fen: FEN string of the position to analyse.

        Returns:
            ChessAnalysis with best move, agent results, and consensus.
        """
        start = time.perf_counter()

        # 1. Meta-controller routing
        routing = self.meta_controller.route(fen)
        logger.info(
            "Routing decision: primary=%s, phase=%s, confidence=%.2f",
            routing.primary_agent, routing.game_phase, routing.confidence,
        )

        # 2. Run all agents in parallel (HRM, TRM, MCTS)
        hrm_task = self.hrm_agent.process(query=fen)
        trm_task = self.trm_agent.process(query=fen)
        mcts_task = self.mcts_agent.process(query=fen)

        hrm_raw, trm_raw, mcts_raw = await asyncio.gather(
            hrm_task, trm_task, mcts_task, return_exceptions=True,
        )

        # 3. Parse agent results
        agent_results: dict[str, ChessMoveResult] = {}
        all_candidates: list[ChessMoveResult] = []

        for name, raw in [("hrm", hrm_raw), ("trm", trm_raw), ("mcts", mcts_raw)]:
            if isinstance(raw, Exception):
                logger.warning("Agent %s failed: %s", name, raw)
                continue
            if not isinstance(raw, dict):
                continue
            meta = raw.get("metadata", {})
            move = meta.get("move", "e2e4")
            score = meta.get("score", 0.5)
            result = ChessMoveResult(
                move=move,
                score=score * routing.agent_weights.get(name, 0.33),
                reasoning=raw.get("response", "")[:200],
                agent_name=name,
                confidence=meta.get("confidence", score),
                metadata=meta,
            )
            agent_results[name] = result
            all_candidates.append(result)

        # 4. Sort candidates by weighted score
        all_candidates.sort(key=lambda c: c.score, reverse=True)

        # 5. Consensus synthesis
        consensus_move: str | None = None
        consensus_reasoning: str | None = None
        if len(agent_results) >= 2:
            consensus_move, consensus_reasoning = await self._synthesize_consensus(
                fen, agent_results,
            )

        # 6. Determine best move
        if consensus_move:
            best_move = consensus_move
        elif all_candidates:
            best_move = all_candidates[0].move
        else:
            best_move = "e2e4"

        elapsed = (time.perf_counter() - start) * 1000
        self._move_count += 1

        return ChessAnalysis(
            best_move=best_move,
            candidate_moves=all_candidates,
            routing_decision=routing,
            agent_results=agent_results,
            consensus_move=consensus_move,
            consensus_reasoning=consensus_reasoning,
            total_time_ms=elapsed,
            metadata={
                "move_count": self._move_count,
                "agents_used": list(agent_results.keys()),
                "mcts_depth": self._mcts_depth,
            },
        )

    async def get_best_move(self, fen: str) -> str:
        """Get the best move for a position (convenience method).

        Args:
            fen: FEN string.

        Returns:
            Best move in UCI format.
        """
        analysis = await self.analyze_position(fen)
        return analysis.best_move

    async def _synthesize_consensus(
        self,
        fen: str,
        agent_results: dict[str, ChessMoveResult],
    ) -> tuple[str | None, str | None]:
        """Synthesize a consensus move from multiple agent results."""
        pos_desc = describe_position(fen)

        summaries = []
        for name, result in agent_results.items():
            summaries.append(
                f"- **{name.upper()}**: Move={result.move}, "
                f"Score={result.confidence:.2f}\n"
                f"  Reasoning: {result.reasoning[:150]}"
            )
        agent_summaries = "\n".join(summaries)

        prompt = CONSENSUS_CHESS_PROMPT.format(
            position_description=pos_desc,
            agent_summaries=agent_summaries,
        )

        try:
            response = await self.hrm_agent.generate_llm_response(
                prompt=prompt,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            text = response.text
            move = extract_uci_move(text)
            return move, text
        except Exception as e:
            logger.warning("Consensus synthesis failed: %s", e)
            return None, None

    @property
    def stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "move_count": self._move_count,
            "hrm_stats": self.hrm_agent.stats,
            "trm_stats": self.trm_agent.stats,
            "mcts_stats": self.mcts_agent.stats,
        }
