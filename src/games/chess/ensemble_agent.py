"""
Chess Ensemble Agent Module.

Combines HRM (Hierarchical Reasoning Model), TRM (Tiny Recursive Model),
and Neural MCTS into an ensemble agent with intelligent routing via
meta-controller.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.agents.hrm_agent import HRMAgent
    from src.agents.trm_agent import TRMAgent
    from src.framework.mcts.neural_mcts import NeuralMCTS
    from src.models.policy_value_net import PolicyValueNetwork

from src.games.chess.config import AgentType, ChessConfig
from src.games.chess.meta_controller import ChessMetaController, RoutingDecision
from src.games.chess.state import ChessGameState


@dataclass
class AgentResponse:
    """Response from an individual agent."""

    agent_type: AgentType
    move: str
    confidence: float
    value_estimate: float
    move_probabilities: dict[str, float]
    thinking_time_ms: float
    extra_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleResponse:
    """Response from the ensemble agent."""

    best_move: str
    move_probabilities: dict[str, float]
    value_estimate: float
    confidence: float
    routing_decision: RoutingDecision
    agent_responses: dict[str, AgentResponse]
    ensemble_method: str
    thinking_time_ms: float


class ChessStateEncoder(nn.Module):
    """Encodes chess state for HRM and TRM agents."""

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize the state encoder.

        Args:
            input_channels: Number of input planes
            output_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode board state.

        Args:
            x: Board tensor (batch, channels, 8, 8)

        Returns:
            Encoded state (batch, output_dim)
        """
        conv_out = self.conv_encoder(x)
        flat = conv_out.view(conv_out.size(0), -1)
        return self.fc(flat)


class ChessEnsembleAgent:
    """Ensemble agent combining HRM, TRM, and MCTS for chess.

    This agent intelligently routes between different reasoning models
    based on the position characteristics, and combines their outputs
    using configurable ensemble methods.
    """

    def __init__(
        self,
        config: ChessConfig,
        policy_value_net: PolicyValueNetwork | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the ensemble agent.

        Args:
            config: Chess configuration
            policy_value_net: Pre-trained policy-value network (optional)
            device: Device to use (defaults to config.device)
        """
        self.config = config
        self.device = device or config.device

        # Initialize meta-controller
        self.meta_controller = ChessMetaController(config.ensemble, self.device)

        # Initialize state encoder for HRM/TRM
        self.state_encoder = ChessStateEncoder(
            input_channels=config.input_channels,
            output_dim=config.hrm.h_dim,
            hidden_dim=config.neural_net.num_channels,
        ).to(self.device)

        # Initialize components (lazy loading)
        self._policy_value_net = policy_value_net
        self._hrm_agent: HRMAgent | None = None
        self._trm_agent: TRMAgent | None = None
        self._mcts: NeuralMCTS | None = None

        # Action encoder
        from src.games.chess.action_space import ChessActionEncoder

        self.action_encoder = ChessActionEncoder(config.action_space)

    @property
    def policy_value_net(self) -> PolicyValueNetwork:
        """Lazy-load policy-value network."""
        if self._policy_value_net is None:
            from src.models.policy_value_net import create_policy_value_network

            system_config = self.config.to_system_config()
            self._policy_value_net = create_policy_value_network(
                system_config.neural_net,
                board_size=self.config.board.board_size,
                device=self.device,
            )
        return self._policy_value_net

    @property
    def hrm_agent(self) -> HRMAgent:
        """Lazy-load HRM agent."""
        if self._hrm_agent is None:
            from src.agents.hrm_agent import create_hrm_agent

            system_config = self.config.to_system_config()
            self._hrm_agent = create_hrm_agent(system_config.hrm, self.device)
        return self._hrm_agent

    @property
    def trm_agent(self) -> TRMAgent:
        """Lazy-load TRM agent."""
        if self._trm_agent is None:
            from src.agents.trm_agent import create_trm_agent

            system_config = self.config.to_system_config()
            self._trm_agent = create_trm_agent(
                system_config.trm,
                output_dim=self.config.action_size,
                device=self.device,
            )
        return self._trm_agent

    @property
    def mcts(self) -> NeuralMCTS:
        """Lazy-load Neural MCTS."""
        if self._mcts is None:
            from src.framework.mcts.neural_mcts import NeuralMCTS

            system_config = self.config.to_system_config()
            self._mcts = NeuralMCTS(
                self.policy_value_net,
                system_config.mcts,
                self.device,
            )
        return self._mcts

    async def get_best_move(
        self,
        state: ChessGameState,
        temperature: float = 0.0,
        time_limit_ms: int | None = None,
        use_ensemble: bool = True,
    ) -> EnsembleResponse:
        """Get the best move for the given position.

        Args:
            state: Current chess position
            temperature: Temperature for move selection (0 = deterministic)
            time_limit_ms: Optional time limit in milliseconds
            use_ensemble: Whether to use full ensemble or just routed agent

        Returns:
            EnsembleResponse with best move and analysis
        """
        import time

        start_time = time.time()

        # Get routing decision
        time_pressure = time_limit_ms is not None and time_limit_ms < 1000
        routing = self.meta_controller.route(state, time_pressure)

        # Collect agent responses
        agent_responses: dict[str, AgentResponse] = {}

        if use_ensemble:
            # Run all agents and combine
            responses = await self._run_all_agents(state, temperature)
            agent_responses = responses
            best_move, move_probs, value = self._combine_responses(
                responses,
                routing.agent_weights,
            )
            ensemble_method = self.config.ensemble.combination_method
        else:
            # Run only the primary agent
            response = await self._run_agent(
                routing.primary_agent,
                state,
                temperature,
            )
            agent_responses[routing.primary_agent.value] = response
            best_move = response.move
            move_probs = response.move_probabilities
            value = response.value_estimate
            ensemble_method = "single_agent"

        # Calculate confidence
        confidence = max(move_probs.values()) if move_probs else 0.5

        thinking_time_ms = (time.time() - start_time) * 1000

        return EnsembleResponse(
            best_move=best_move,
            move_probabilities=move_probs,
            value_estimate=value,
            confidence=confidence,
            routing_decision=routing,
            agent_responses=agent_responses,
            ensemble_method=ensemble_method,
            thinking_time_ms=thinking_time_ms,
        )

    async def _run_all_agents(
        self,
        state: ChessGameState,
        temperature: float,
    ) -> dict[str, AgentResponse]:
        """Run all agents in parallel.

        Args:
            state: Chess position
            temperature: Move selection temperature

        Returns:
            Dictionary of agent responses
        """
        # Run agents concurrently
        hrm_task = asyncio.create_task(self._run_agent(AgentType.HRM, state, temperature))
        trm_task = asyncio.create_task(self._run_agent(AgentType.TRM, state, temperature))
        mcts_task = asyncio.create_task(self._run_agent(AgentType.MCTS, state, temperature))

        hrm_response, trm_response, mcts_response = await asyncio.gather(hrm_task, trm_task, mcts_task)

        return {
            "hrm": hrm_response,
            "trm": trm_response,
            "mcts": mcts_response,
        }

    async def _run_agent(
        self,
        agent_type: AgentType,
        state: ChessGameState,
        temperature: float,
    ) -> AgentResponse:
        """Run a specific agent on the position.

        Args:
            agent_type: Which agent to run
            state: Chess position
            temperature: Move selection temperature

        Returns:
            Agent response
        """
        import time

        start_time = time.time()

        if agent_type == AgentType.HRM:
            response = await self._run_hrm(state, temperature)
        elif agent_type == AgentType.TRM:
            response = await self._run_trm(state, temperature)
        else:  # MCTS
            response = await self._run_mcts(state, temperature)

        response.thinking_time_ms = (time.time() - start_time) * 1000
        return response

    async def _run_hrm(
        self,
        state: ChessGameState,
        temperature: float,
    ) -> AgentResponse:
        """Run HRM agent on the position.

        Args:
            state: Chess position
            temperature: Move selection temperature

        Returns:
            Agent response
        """
        # Encode state for HRM
        board_tensor = state.to_tensor().unsqueeze(0).to(self.device)
        encoded_state = self.state_encoder(board_tensor)

        # Add sequence dimension for HRM
        hrm_input = encoded_state.unsqueeze(1)  # (1, 1, h_dim)

        # Run HRM
        with torch.no_grad():
            hrm_output = self.hrm_agent(hrm_input, max_steps=self.config.hrm.max_outer_steps)

        # Get policy from HRM output through policy network
        # HRM provides strategic guidance, we use it to weight the policy
        policy_logits, value = self.policy_value_net(board_tensor)
        policy_logits = policy_logits.squeeze()

        # Modulate policy with HRM confidence
        hrm_confidence = hrm_output.convergence_path[-1] if hrm_output.convergence_path else 0.5

        # Get move probabilities
        # legal_moves = state.get_legal_actions()  # Unused
        from_black = state.current_player == -1

        move_probs = self.action_encoder.filter_policy_to_legal(
            policy_logits.cpu().numpy(),
            state.board,
            from_black,
            temperature=max(0.1, temperature),
        )

        # Select best move
        best_move = max(move_probs.items(), key=lambda x: x[1])[0]

        return AgentResponse(
            agent_type=AgentType.HRM,
            move=best_move,
            confidence=hrm_confidence,
            value_estimate=float(value.item()),
            move_probabilities=move_probs,
            thinking_time_ms=0,  # Set by caller
            extra_info={
                "halt_step": hrm_output.halt_step,
                "ponder_cost": hrm_output.total_ponder_cost,
            },
        )

    async def _run_trm(
        self,
        state: ChessGameState,
        temperature: float,
    ) -> AgentResponse:
        """Run TRM agent on the position.

        Args:
            state: Chess position
            temperature: Move selection temperature

        Returns:
            Agent response
        """
        # Encode state for TRM
        board_tensor = state.to_tensor().unsqueeze(0).to(self.device)
        encoded_state = self.state_encoder(board_tensor)

        # Resize for TRM input dimension if needed
        if encoded_state.shape[-1] != self.config.trm.latent_dim:
            # Use a projection
            projection = nn.Linear(
                encoded_state.shape[-1],
                self.config.trm.latent_dim,
            ).to(self.device)
            encoded_state = projection(encoded_state)

        # Run TRM for recursive refinement
        with torch.no_grad():
            trm_output = self.trm_agent(
                encoded_state,
                num_recursions=self.config.trm.num_recursions,
                check_convergence=True,
            )

        # TRM output is action logits
        action_logits = trm_output.final_prediction.squeeze()

        # Get move probabilities
        from_black = state.current_player == -1
        move_probs = self.action_encoder.filter_policy_to_legal(
            action_logits.cpu().numpy(),
            state.board,
            from_black,
            temperature=max(0.1, temperature),
        )

        # Get value estimate from policy network
        _, value = self.policy_value_net(board_tensor)

        # Select best move
        best_move = max(move_probs.items(), key=lambda x: x[1])[0]

        # Calculate confidence based on convergence
        confidence = 0.8 if trm_output.converged else 0.5

        return AgentResponse(
            agent_type=AgentType.TRM,
            move=best_move,
            confidence=confidence,
            value_estimate=float(value.item()),
            move_probabilities=move_probs,
            thinking_time_ms=0,
            extra_info={
                "converged": trm_output.converged,
                "convergence_step": trm_output.convergence_step,
                "recursion_depth": trm_output.recursion_depth,
            },
        )

    async def _run_mcts(
        self,
        state: ChessGameState,
        temperature: float,
    ) -> AgentResponse:
        """Run Neural MCTS on the position.

        Args:
            state: Chess position
            temperature: Move selection temperature

        Returns:
            Agent response
        """
        # Run MCTS search
        num_simulations = self.config.mcts.num_simulations
        add_noise = temperature > 0

        action_probs, root = await self.mcts.search(
            state,
            num_simulations=num_simulations,
            temperature=temperature if temperature > 0 else 0.1,
            add_root_noise=add_noise,
        )

        # Get value estimate from root
        value_estimate = root.value if root else 0.0

        # Select best move
        if temperature == 0:
            # Deterministic - pick highest probability
            best_move = max(action_probs.items(), key=lambda x: x[1])[0]
        else:
            # Stochastic - sample from distribution
            moves = list(action_probs.keys())
            probs = np.array(list(action_probs.values()))
            probs = probs / probs.sum()
            best_move = np.random.choice(moves, p=probs)

        # Calculate confidence from visit counts
        if action_probs:
            sorted_probs = sorted(action_probs.values(), reverse=True)
            confidence = sorted_probs[0]
            if len(sorted_probs) > 1:
                confidence = sorted_probs[0] - sorted_probs[1] + 0.5
        else:
            confidence = 0.5

        return AgentResponse(
            agent_type=AgentType.MCTS,
            move=best_move,
            confidence=min(1.0, confidence),
            value_estimate=value_estimate,
            move_probabilities=action_probs,
            thinking_time_ms=0,
            extra_info={
                "num_simulations": num_simulations,
                "root_visits": root.visit_count if root else 0,
            },
        )

    def _combine_responses(
        self,
        responses: dict[str, AgentResponse],
        weights: dict[str, float],
    ) -> tuple[str, dict[str, float], float]:
        """Combine agent responses using ensemble method.

        Args:
            responses: Agent responses
            weights: Agent weights from routing

        Returns:
            (best_move, combined_probabilities, combined_value)
        """
        method = self.config.ensemble.combination_method

        if method == "weighted_vote":
            return self._weighted_vote_combination(responses, weights)
        elif method == "max_confidence":
            return self._max_confidence_combination(responses)
        elif method == "bayesian":
            return self._bayesian_combination(responses, weights)
        else:
            return self._weighted_vote_combination(responses, weights)

    def _weighted_vote_combination(
        self,
        responses: dict[str, AgentResponse],
        weights: dict[str, float],
    ) -> tuple[str, dict[str, float], float]:
        """Combine using weighted voting.

        Args:
            responses: Agent responses
            weights: Agent weights

        Returns:
            (best_move, combined_probabilities, combined_value)
        """
        # Collect all moves
        all_moves: set[str] = set()
        for response in responses.values():
            all_moves.update(response.move_probabilities.keys())

        # Weighted average of probabilities
        combined_probs: dict[str, float] = {}
        combined_value = 0.0

        for move in all_moves:
            prob = 0.0
            for agent_name, response in responses.items():
                weight = weights.get(agent_name, 0.0)
                move_prob = response.move_probabilities.get(move, 0.0)
                prob += weight * move_prob
            combined_probs[move] = prob

        # Weighted average of values
        for agent_name, response in responses.items():
            weight = weights.get(agent_name, 0.0)
            combined_value += weight * response.value_estimate

        # Normalize probabilities
        total_prob = sum(combined_probs.values())
        if total_prob > 0:
            combined_probs = {k: v / total_prob for k, v in combined_probs.items()}

        # Select best move
        best_move = max(combined_probs.items(), key=lambda x: x[1])[0]

        return best_move, combined_probs, combined_value

    def _max_confidence_combination(
        self,
        responses: dict[str, AgentResponse],
    ) -> tuple[str, dict[str, float], float]:
        """Combine by selecting most confident agent.

        Args:
            responses: Agent responses

        Returns:
            (best_move, combined_probabilities, combined_value)
        """
        # Find most confident agent
        best_agent = max(responses.values(), key=lambda r: r.confidence)

        return (
            best_agent.move,
            best_agent.move_probabilities,
            best_agent.value_estimate,
        )

    def _bayesian_combination(
        self,
        responses: dict[str, AgentResponse],
        weights: dict[str, float],
    ) -> tuple[str, dict[str, float], float]:
        """Combine using Bayesian model averaging.

        Args:
            responses: Agent responses
            weights: Prior weights (treated as prior probabilities)

        Returns:
            (best_move, combined_probabilities, combined_value)
        """
        # Weight agents by prior * confidence
        posterior_weights = {}
        for agent_name, response in responses.items():
            prior = weights.get(agent_name, 1.0 / len(responses))
            likelihood = response.confidence
            posterior_weights[agent_name] = prior * likelihood

        # Normalize posterior
        total = sum(posterior_weights.values())
        if total > 0:
            posterior_weights = {k: v / total for k, v in posterior_weights.items()}

        # Use weighted combination with posterior weights
        return self._weighted_vote_combination(responses, posterior_weights)

    def save(self, path: str) -> None:
        """Save ensemble agent state.

        Args:
            path: Directory to save to
        """
        import os

        os.makedirs(path, exist_ok=True)

        # Save policy-value network
        torch.save(
            self.policy_value_net.state_dict(),
            os.path.join(path, "policy_value_net.pt"),
        )

        # Save state encoder
        torch.save(
            self.state_encoder.state_dict(),
            os.path.join(path, "state_encoder.pt"),
        )

        # Save meta-controller
        self.meta_controller.save(os.path.join(path, "meta_controller.pt"))

        # Save config
        self.config.save(os.path.join(path, "config.json"))

    @classmethod
    def load(cls, path: str, device: str | None = None) -> ChessEnsembleAgent:
        """Load ensemble agent from saved state.

        Args:
            path: Directory to load from
            device: Device to load to

        Returns:
            Loaded ensemble agent
        """
        import os

        # Load config
        config = ChessConfig.load(os.path.join(path, "config.json"))
        if device:
            config.device = device

        # Create agent
        agent = cls(config)

        # Load policy-value network
        agent.policy_value_net.load_state_dict(
            torch.load(
                os.path.join(path, "policy_value_net.pt"),
                map_location=agent.device,
                weights_only=True,
            )
        )

        # Load state encoder
        agent.state_encoder.load_state_dict(
            torch.load(
                os.path.join(path, "state_encoder.pt"),
                map_location=agent.device,
                weights_only=True,
            )
        )

        # Load meta-controller
        agent.meta_controller.load(os.path.join(path, "meta_controller.pt"))

        return agent
