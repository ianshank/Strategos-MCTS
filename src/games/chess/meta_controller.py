"""
Chess Meta-Controller Module.

Implements intelligent routing between HRM, TRM, and MCTS agents
based on chess-specific features like game phase, position complexity,
and tactical vs strategic nature of the position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    import chess

from src.games.chess.config import AgentType, ChessEnsembleConfig, GamePhase
from src.games.chess.state import ChessGameState


@dataclass
class RoutingDecision:
    """Result of meta-controller routing decision."""

    primary_agent: AgentType
    agent_weights: dict[str, float]
    confidence: float
    features: dict[str, Any]
    reasoning: str


@dataclass
class ChessPositionFeatures:
    """Features extracted from chess position for routing decisions."""

    # Game phase features
    game_phase: GamePhase
    move_number: int
    is_opening: bool
    is_middlegame: bool
    is_endgame: bool

    # Material features
    total_material: int
    material_balance: int
    has_queens: bool
    is_material_imbalanced: bool

    # Tactical features
    is_check: bool
    num_legal_moves: int
    has_captures: bool
    has_promotions: bool
    is_forcing: bool

    # Positional features
    center_control: float
    king_safety: float
    pawn_structure_complexity: float

    # Time features (if applicable)
    time_pressure: bool

    def to_tensor(self) -> torch.Tensor:
        """Convert features to tensor for neural routing."""
        return torch.tensor(
            [
                float(self.is_opening),
                float(self.is_middlegame),
                float(self.is_endgame),
                self.move_number / 100.0,
                self.total_material / 78.0,  # Max material = 78
                self.material_balance / 39.0,  # Normalize to [-1, 1]
                float(self.has_queens),
                float(self.is_material_imbalanced),
                float(self.is_check),
                self.num_legal_moves / 50.0,
                float(self.has_captures),
                float(self.has_promotions),
                float(self.is_forcing),
                self.center_control,
                self.king_safety,
                self.pawn_structure_complexity,
                float(self.time_pressure),
            ],
            dtype=torch.float32,
        )


class ChessFeatureExtractor:
    """Extracts routing-relevant features from chess positions."""

    def __init__(self) -> None:
        """Initialize the feature extractor."""
        self._piece_values = {
            1: 1,  # Pawn
            2: 3,  # Knight
            3: 3,  # Bishop
            4: 5,  # Rook
            5: 9,  # Queen
            6: 0,  # King (not counted in material)
        }

    def extract(
        self,
        state: ChessGameState,
        time_pressure: bool = False,
    ) -> ChessPositionFeatures:
        """Extract features from a chess position.

        Args:
            state: Chess game state
            time_pressure: Whether the player is in time trouble

        Returns:
            ChessPositionFeatures for routing decisions
        """
        import chess

        board = state.board
        phase = state.get_game_phase()

        # Calculate material
        total_material = 0
        white_material = 0
        black_material = 0
        has_queens = False

        for piece_type in range(1, 7):
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            value = self._piece_values[piece_type]
            white_material += white_count * value
            black_material += black_count * value
            total_material += (white_count + black_count) * value
            if piece_type == 5 and (white_count > 0 or black_count > 0):
                has_queens = True

        material_balance = white_material - black_material

        # Analyze moves
        legal_moves = list(board.legal_moves)
        num_legal_moves = len(legal_moves)
        has_captures = any(board.is_capture(m) for m in legal_moves)
        has_promotions = any(m.promotion is not None for m in legal_moves)
        is_forcing = state.is_check() or (has_captures and num_legal_moves < 10)

        # Center control (d4, d5, e4, e5)
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        center_control = self._calculate_center_control(board, center_squares)

        # King safety (simplified)
        king_safety = self._calculate_king_safety(board)

        # Pawn structure complexity
        pawn_complexity = self._calculate_pawn_complexity(board)

        return ChessPositionFeatures(
            game_phase=phase,
            move_number=state.move_number,
            is_opening=phase == GamePhase.OPENING,
            is_middlegame=phase == GamePhase.MIDDLEGAME,
            is_endgame=phase == GamePhase.ENDGAME,
            total_material=total_material,
            material_balance=material_balance,
            has_queens=has_queens,
            is_material_imbalanced=abs(material_balance) >= 3,
            is_check=state.is_check(),
            num_legal_moves=num_legal_moves,
            has_captures=has_captures,
            has_promotions=has_promotions,
            is_forcing=is_forcing,
            center_control=center_control,
            king_safety=king_safety,
            pawn_structure_complexity=pawn_complexity,
            time_pressure=time_pressure,
        )

    def _calculate_center_control(
        self,
        board: chess.Board,
        center_squares: list[int],
    ) -> float:
        """Calculate center control score.

        Args:
            board: Chess board
            center_squares: List of center square indices

        Returns:
            Normalized center control score [0, 1]
        """
        import chess

        score = 0.0
        max_score = len(center_squares) * 2  # Max 2 points per square

        for square in center_squares:
            piece = board.piece_at(square)
            if piece is not None and piece.color == board.turn:
                score += 1

            # Count attackers
            white_attackers = len(board.attackers(chess.WHITE, square))
            black_attackers = len(board.attackers(chess.BLACK, square))

            if board.turn == chess.WHITE:
                score += min(1, white_attackers - black_attackers) * 0.5
            else:
                score += min(1, black_attackers - white_attackers) * 0.5

        return max(0, min(1, score / max_score))

    def _calculate_king_safety(self, board: chess.Board) -> float:
        """Calculate king safety score.

        Args:
            board: Chess board

        Returns:
            Normalized king safety score [0, 1]
        """
        import chess

        # Simplified king safety based on pawn shield and attackers
        color = board.turn
        king_square = board.king(color)
        if king_square is None:
            return 0.5

        safety = 1.0

        # Check pawn shield
        pawn_shield_squares = self._get_pawn_shield_squares(king_square, color)
        shield_pawns = sum(1 for sq in pawn_shield_squares if board.piece_at(sq) == chess.Piece(chess.PAWN, color))
        safety *= 0.5 + (shield_pawns / len(pawn_shield_squares)) * 0.5

        # Penalize for attackers near king
        king_zone = list(board.attacks(king_square))
        attackers = sum(len(board.attackers(not color, sq)) for sq in king_zone)
        safety *= max(0.3, 1.0 - attackers * 0.1)

        return safety

    def _get_pawn_shield_squares(self, king_square: int, color: bool) -> list[int]:
        """Get pawn shield squares for a king.

        Args:
            king_square: King's square
            color: King's color (True = White)

        Returns:
            List of pawn shield square indices
        """

        file = king_square % 8
        rank = king_square // 8
        direction = 1 if color else -1
        shield_rank = rank + direction

        if not (0 <= shield_rank <= 7):
            return []

        squares = []
        for f in range(max(0, file - 1), min(8, file + 2)):
            squares.append(shield_rank * 8 + f)

        return squares

    def _calculate_pawn_complexity(self, board: chess.Board) -> float:
        """Calculate pawn structure complexity.

        Args:
            board: Chess board

        Returns:
            Normalized complexity score [0, 1]
        """
        import chess

        complexity = 0.0

        # Count pawn islands, doubled pawns, passed pawns
        for color in [chess.WHITE, chess.BLACK]:
            pawns = board.pieces(chess.PAWN, color)
            if not pawns:
                continue

            files_with_pawns = {sq % 8 for sq in pawns}

            # Pawn islands (groups of adjacent files with pawns)
            islands = 0
            prev_file = -2
            for f in sorted(files_with_pawns):
                if f > prev_file + 1:
                    islands += 1
                prev_file = f
            complexity += islands * 0.1

            # Doubled pawns
            for f in files_with_pawns:
                pawns_on_file = sum(1 for sq in pawns if sq % 8 == f)
                if pawns_on_file > 1:
                    complexity += 0.1

        return min(1.0, complexity)


class NeuralRouter(nn.Module):
    """Neural network for learning routing decisions."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_agents: int,
        num_layers: int,
    ) -> None:
        """Initialize the neural router.

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            num_agents: Number of agents to route between
            num_layers: Number of hidden layers
        """
        super().__init__()

        layers = []
        current_dim = input_dim

        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                ]
            )
            current_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_agents)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input features (batch, input_dim)

        Returns:
            (agent_logits, confidence) tensors
        """
        encoded = self.encoder(x)
        agent_logits = self.classifier(encoded)
        confidence = torch.sigmoid(self.confidence_head(encoded))
        return agent_logits, confidence


class ChessMetaController:
    """Meta-controller for routing between HRM, TRM, and MCTS agents.

    Uses a combination of rule-based heuristics and learned routing
    to select the best agent for each position.
    """

    AGENT_NAMES: list[str] = ["hrm", "trm", "mcts"]

    def __init__(
        self,
        config: ChessEnsembleConfig,
        device: str = "cpu",
    ) -> None:
        """Initialize the meta-controller.

        Args:
            config: Ensemble configuration
            device: Device to use for neural routing
        """
        self.config = config
        self.device = device
        self.feature_extractor = ChessFeatureExtractor()

        # Initialize neural router if using learned routing
        self.neural_router: NeuralRouter | None = None
        if config.use_learned_routing:
            self.neural_router = NeuralRouter(
                input_dim=17,  # Number of features
                hidden_dim=config.routing_hidden_dim,
                num_agents=len(self.AGENT_NAMES),
                num_layers=config.routing_num_layers,
            ).to(device)

    def route(
        self,
        state: ChessGameState,
        time_pressure: bool = False,
    ) -> RoutingDecision:
        """Decide which agent to use for the given position.

        Args:
            state: Current chess position
            time_pressure: Whether player is in time trouble

        Returns:
            RoutingDecision with agent selection and weights
        """
        features = self.feature_extractor.extract(state, time_pressure)

        if self.config.use_learned_routing and self.neural_router is not None:
            return self._neural_route(features)
        else:
            return self._heuristic_route(features)

    def _heuristic_route(self, features: ChessPositionFeatures) -> RoutingDecision:
        """Make routing decision using rule-based heuristics.

        Args:
            features: Position features

        Returns:
            Routing decision
        """
        # Get phase-specific weights
        phase_weights = self.config.get_phase_weights(features.game_phase)

        # Adjust weights based on position characteristics
        weights = phase_weights.copy()

        # Tactical positions favor MCTS
        if features.is_forcing or features.is_check:
            weights["mcts"] *= 1.5
            weights["hrm"] *= 0.7

        # Complex endgames favor TRM for precise calculation
        if features.is_endgame and not features.has_queens:
            weights["trm"] *= 1.3

        # Opening with book moves favors HRM for strategic planning
        if features.is_opening and features.move_number < 15:
            weights["hrm"] *= 1.3

        # Time pressure favors faster agents (TRM, then MCTS)
        if features.time_pressure:
            weights["trm"] *= 1.5
            weights["hrm"] *= 0.5

        # Normalize weights
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # Select primary agent
        primary = max(weights.items(), key=lambda x: x[1])
        primary_agent = AgentType(primary[0])

        # Calculate confidence based on how dominant the primary agent is
        sorted_weights = sorted(weights.values(), reverse=True)
        confidence = sorted_weights[0] - sorted_weights[1] if len(sorted_weights) > 1 else 1.0
        confidence = min(1.0, confidence + 0.3)  # Boost confidence

        # Generate reasoning
        reasoning = self._generate_reasoning(features, primary_agent)

        return RoutingDecision(
            primary_agent=primary_agent,
            agent_weights=weights,
            confidence=confidence,
            features={
                "phase": features.game_phase.value,
                "material": features.total_material,
                "is_forcing": features.is_forcing,
                "num_moves": features.num_legal_moves,
            },
            reasoning=reasoning,
        )

    def _neural_route(self, features: ChessPositionFeatures) -> RoutingDecision:
        """Make routing decision using neural network.

        Args:
            features: Position features

        Returns:
            Routing decision
        """
        if self.neural_router is None:
            return self._heuristic_route(features)

        # Convert features to tensor
        feature_tensor = features.to_tensor().unsqueeze(0).to(self.device)

        # Get predictions
        with torch.no_grad():
            agent_logits, confidence = self.neural_router(feature_tensor)
            agent_probs = F.softmax(agent_logits, dim=-1).squeeze()
            confidence = confidence.item()

        # Build weights dictionary
        weights = {name: float(agent_probs[i]) for i, name in enumerate(self.AGENT_NAMES)}

        # Select primary agent
        primary_idx = torch.argmax(agent_probs).item()
        primary_agent = AgentType(self.AGENT_NAMES[primary_idx])

        # Generate reasoning
        reasoning = self._generate_reasoning(features, primary_agent)

        return RoutingDecision(
            primary_agent=primary_agent,
            agent_weights=weights,
            confidence=confidence,
            features={
                "phase": features.game_phase.value,
                "material": features.total_material,
                "is_forcing": features.is_forcing,
                "num_moves": features.num_legal_moves,
            },
            reasoning=reasoning,
        )

    def _generate_reasoning(
        self,
        features: ChessPositionFeatures,
        primary_agent: AgentType,
    ) -> str:
        """Generate human-readable reasoning for routing decision.

        Args:
            features: Position features
            primary_agent: Selected primary agent

        Returns:
            Reasoning string
        """
        reasons = []

        if primary_agent == AgentType.HRM:
            if features.is_opening:
                reasons.append("Opening phase requires strategic planning")
            if not features.is_forcing:
                reasons.append("Position is quiet, favoring strategic analysis")
        elif primary_agent == AgentType.TRM:
            if features.is_endgame:
                reasons.append("Endgame requires precise calculation")
            if features.time_pressure:
                reasons.append("Time pressure favors fast refinement")
        else:  # MCTS
            if features.is_forcing:
                reasons.append("Tactical position requires deep search")
            if features.is_middlegame:
                reasons.append("Complex middlegame benefits from tree search")

        return "; ".join(reasons) if reasons else f"Default routing to {primary_agent.value}"

    def update(
        self,
        features: ChessPositionFeatures,
        best_agent: AgentType,
    ) -> None:
        """Update the neural router based on outcome.

        Args:
            features: Position features
            best_agent: Agent that performed best
        """
        # This would be called during training to update routing policy
        # Implementation depends on training setup
        pass

    def save(self, path: str) -> None:
        """Save the neural router weights.

        Args:
            path: Path to save weights
        """
        if self.neural_router is not None:
            torch.save(self.neural_router.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the neural router weights.

        Args:
            path: Path to load weights from
        """
        if self.neural_router is not None:
            self.neural_router.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
