"""
Enterprise Use Case Protocol and Base Class.

Defines the contract for all enterprise use cases, enabling
dynamic configuration and seamless MCTS integration.

Design Principles:
- Protocol-based interfaces for flexibility
- Dependency injection for testability
- No hardcoded values - all configuration via settings
- Full async support for I/O operations
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from src.framework.mcts.core import MCTSState
    from src.framework.mcts.policies import RolloutPolicy


# Type variables for domain-specific state and action
DomainStateT = TypeVar("DomainStateT", bound="BaseDomainState")
DomainActionT = TypeVar("DomainActionT")


# Custom exceptions for better error handling
class EnterpriseUseCaseError(Exception):
    """Base exception for enterprise use case errors."""

    pass


class MCTSSearchError(EnterpriseUseCaseError):
    """Error during MCTS search operations."""

    pass


class AgentProcessingError(EnterpriseUseCaseError):
    """Error during agent processing."""

    def __init__(self, agent_name: str, original_error: Exception) -> None:
        self.agent_name = agent_name
        self.original_error = original_error
        super().__init__(f"Agent '{agent_name}' failed: {original_error}")


class StateValidationError(EnterpriseUseCaseError):
    """Error during state validation."""

    pass


@dataclass
class BaseDomainState:
    """
    Base class for domain-specific states.

    All enterprise use cases must define a state that inherits from this class.
    The state captures the current context, progress, and accumulated knowledge
    within the domain.

    Attributes:
        state_id: Unique identifier for this state
        domain: Domain name (e.g., 'ma_due_diligence', 'clinical_trial')
        features: Dictionary of domain-specific features for MCTS
        metadata: Additional metadata for tracking and debugging
    """

    state_id: str
    domain: str
    features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_mcts_state(self) -> MCTSState:
        """
        Convert to MCTS-compatible state.

        Returns:
            MCTSState instance for use in MCTS search
        """
        from src.framework.mcts.core import MCTSState

        return MCTSState(
            state_id=self.state_id,
            features={
                "domain": self.domain,
                **self.features,
            },
        )

    def to_hash_key(self) -> str:
        """
        Generate a hashable key for caching.

        Returns:
            String key for use in caches
        """
        import hashlib
        import json

        # Create deterministic hash from features
        feature_str = json.dumps(self.features, sort_keys=True, default=str)
        return hashlib.sha256(f"{self.state_id}:{feature_str}".encode()).hexdigest()[:16]

    def copy(self) -> BaseDomainState:
        """Create a deep copy of the state."""
        import copy

        return copy.deepcopy(self)


class DomainAgentProtocol(Protocol):
    """
    Protocol for domain-specific agents.

    Domain agents are specialized AI agents that operate within a specific
    enterprise domain. They must implement async processing and confidence
    scoring to integrate with the multi-agent orchestration framework.
    """

    async def process(
        self,
        query: str,
        domain_state: BaseDomainState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Process query within domain context.

        Args:
            query: User query or task description
            domain_state: Current domain state
            context: Additional context (RAG results, MCTS action, etc.)

        Returns:
            Dictionary containing:
                - response: Agent's response text
                - findings: Structured findings
                - confidence: Confidence score (0-1)
                - metadata: Additional metadata
        """
        ...

    def get_confidence(self) -> float:
        """
        Return confidence score for last result.

        Returns:
            Confidence score between 0 and 1
        """
        ...

    @property
    def name(self) -> str:
        """Return agent name for logging and tracking."""
        ...


class RewardFunctionProtocol(Protocol):
    """
    Protocol for MCTS reward functions.

    Reward functions evaluate the quality of actions taken in the domain,
    guiding MCTS exploration toward high-value paths.
    """

    def evaluate(
        self,
        state: BaseDomainState,
        action: str,
        context: dict[str, Any],
    ) -> float:
        """
        Evaluate action in state, return reward in [0, 1].

        Args:
            state: Current domain state
            action: Action being evaluated
            context: Additional context

        Returns:
            Reward value between 0 (worst) and 1 (best)
        """
        ...

    def get_components(
        self,
        state: BaseDomainState,
        action: str,
        context: dict[str, Any],
    ) -> dict[str, float]:
        """
        Get individual reward components for debugging.

        Args:
            state: Current domain state
            action: Action being evaluated
            context: Additional context

        Returns:
            Dictionary mapping component names to their values
        """
        ...


class UseCaseProtocol(Protocol[DomainStateT]):
    """
    Protocol defining enterprise use case interface.

    All enterprise use cases must implement this protocol to ensure
    consistent behavior across the framework.
    """

    @property
    def name(self) -> str:
        """Unique use case identifier."""
        ...

    @property
    def domain(self) -> str:
        """Domain category (e.g., 'finance', 'healthcare', 'legal')."""
        ...

    def get_initial_state(self, query: str, context: dict[str, Any]) -> DomainStateT:
        """
        Create initial domain state from query.

        Args:
            query: User query or task description
            context: Initial context including RAG results

        Returns:
            Initialized domain state
        """
        ...

    def get_available_actions(self, state: DomainStateT) -> list[str]:
        """
        Return available actions for MCTS expansion.

        Args:
            state: Current domain state

        Returns:
            List of available action strings
        """
        ...

    def apply_action(self, state: DomainStateT, action: str) -> DomainStateT:
        """
        State transition function for MCTS.

        Args:
            state: Current domain state
            action: Action to apply

        Returns:
            New domain state after action
        """
        ...

    def get_reward_function(self) -> RewardFunctionProtocol:
        """
        Return reward function for MCTS rollouts.

        Returns:
            Reward function instance
        """
        ...

    def get_domain_agents(self) -> dict[str, DomainAgentProtocol]:
        """
        Return domain-specific agents.

        Returns:
            Dictionary mapping agent names to agent instances
        """
        ...

    def get_rollout_policy(self) -> RolloutPolicy:
        """
        Return MCTS rollout policy for this use case.

        Returns:
            Configured rollout policy
        """
        ...

    async def process(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        use_mcts: bool = True,
    ) -> dict[str, Any]:
        """
        Main entry point for processing queries.

        Args:
            query: User query
            context: Optional context
            use_mcts: Whether to use MCTS for exploration

        Returns:
            Processing results
        """
        ...


class BaseUseCase(Generic[DomainStateT]):
    """
    Base implementation for enterprise use cases.

    Provides common functionality and enforces patterns.
    Subclasses must implement the abstract methods.

    Example:
        >>> class MADueDiligence(BaseUseCase[MAState]):
        ...     @property
        ...     def name(self) -> str:
        ...         return "ma_due_diligence"
        ...
        ...     def get_initial_state(self, query, context):
        ...         return MAState(...)
    """

    def __init__(
        self,
        config: Any,
        llm_client: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the use case.

        Args:
            config: Use case configuration (subclass-specific)
            llm_client: Optional LLM client for agent operations
            logger: Optional logger instance
        """
        self._config = config
        self._llm_client = llm_client
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._agents: dict[str, DomainAgentProtocol] = {}
        self._reward_function: RewardFunctionProtocol | None = None
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique use case identifier."""
        pass

    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain category."""
        pass

    @property
    def config(self) -> Any:
        """Return the configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if use case is initialized."""
        return self._initialized

    def initialize(self) -> None:
        """
        Initialize the use case (lazy initialization).

        Override in subclasses to perform setup like loading models.
        """
        if self._initialized:
            return

        self._logger.info(f"Initializing use case: {self.name}")
        self._setup_agents()
        self._setup_reward_function()
        self._initialized = True

    def _setup_agents(self) -> None:
        """Set up domain agents. Override in subclasses."""
        pass

    def _setup_reward_function(self) -> None:
        """Set up reward function. Override in subclasses."""
        pass

    @abstractmethod
    def get_initial_state(self, query: str, context: dict[str, Any]) -> DomainStateT:
        """Create initial domain state from query."""
        pass

    @abstractmethod
    def get_available_actions(self, state: DomainStateT) -> list[str]:
        """Return available actions for MCTS expansion."""
        pass

    @abstractmethod
    def apply_action(self, state: DomainStateT, action: str) -> DomainStateT:
        """State transition function for MCTS."""
        pass

    def get_reward_function(self) -> RewardFunctionProtocol:
        """Return reward function for MCTS rollouts."""
        if self._reward_function is None:
            raise RuntimeError("Reward function not initialized. Call initialize() first.")
        return self._reward_function

    def get_domain_agents(self) -> dict[str, DomainAgentProtocol]:
        """Return domain-specific agents."""
        if not self._initialized:
            self.initialize()
        return self._agents

    def get_rollout_policy(self) -> RolloutPolicy:
        """Return MCTS rollout policy for this use case."""
        from src.framework.mcts.policies import HybridRolloutPolicy

        # Get rollout policy config from settings
        rollout_config = self._config.rollout_policy
        depth_divisor = rollout_config.depth_bonus_divisor
        max_bonus = rollout_config.max_depth_bonus

        def heuristic_fn(mcts_state: MCTSState) -> float:
            """Default heuristic based on state features."""
            features = mcts_state.features
            base = 0.5
            action_count = int(features.get("action_count", 0))
            depth_bonus = min(action_count / depth_divisor, max_bonus)
            return float(min(base + depth_bonus, 1.0))

        return HybridRolloutPolicy(
            heuristic_fn=heuristic_fn,
            heuristic_weight=rollout_config.heuristic_weight,
            random_weight=rollout_config.random_weight,
        )

    async def process(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        use_mcts: bool = True,
    ) -> dict[str, Any]:
        """
        Main entry point for processing queries.

        Args:
            query: User query
            context: Optional context
            use_mcts: Whether to use MCTS for exploration

        Returns:
            Dictionary containing:
                - result: Main processing result
                - domain_state: Final domain state
                - agent_results: Results from domain agents
                - mcts_stats: MCTS statistics (if used)
                - confidence: Overall confidence score
        """
        if not self._initialized:
            self.initialize()

        context = context or {}
        self._logger.info(
            f"Processing query in {self.name}: {query[:100]}...",
            extra={"domain": self.domain, "use_mcts": use_mcts},
        )

        # Initialize state
        state = self.get_initial_state(query, context)

        # Run MCTS if enabled
        mcts_result = {}
        if use_mcts and self._config.enabled:
            mcts_result = await self._run_mcts(state, context)

        # Process with domain agents
        agent_results = await self._process_with_agents(query, state, context, mcts_result)

        # Synthesize results
        final_result = self._synthesize_results(agent_results, mcts_result)

        return {
            "result": final_result.get("response", ""),
            "domain_state": state.__dict__ if hasattr(state, "__dict__") else {},
            "agent_results": agent_results,
            "mcts_stats": mcts_result.get("stats", {}),
            "confidence": final_result.get("confidence", 0.0),
            "use_case": self.name,
            "domain": self.domain,
        }

    async def _run_mcts(
        self,
        state: DomainStateT,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run MCTS exploration for the use case.

        Override for custom MCTS behavior.
        """
        try:
            from src.framework.mcts.core import MCTSEngine, MCTSNode

            engine = MCTSEngine(
                seed=self._config.mcts_seed,
                exploration_weight=self._config.mcts_exploration_weight,
            )

            # Create root node for MCTS search tree
            # Note: The root node is created here to validate state conversion works correctly.
            # Full MCTS tree search (selection, expansion, simulation, backpropagation)
            # will be implemented in a future iteration to find optimal action sequences.
            mcts_state = state.to_mcts_state()
            MCTSNode(state=mcts_state, rng=engine.rng)  # Validates state is MCTS-compatible

            # Placeholder return until full MCTS integration
            return {
                "best_action": None,
                "stats": {
                    "iterations": self._config.max_mcts_iterations,
                    "explored_nodes": 0,
                },
            }
        except ImportError as e:
            self._logger.warning(f"MCTS module not available: {e}")
            return {"best_action": None, "stats": {}}
        except (ValueError, TypeError, AttributeError) as e:
            # Catch specific errors related to MCTS state/node creation
            self._logger.warning(f"MCTS search failed due to configuration error: {e}")
            raise MCTSSearchError(f"MCTS configuration error: {e}") from e
        except MCTSSearchError:
            raise  # Re-raise our custom exceptions
        except RuntimeError as e:
            # Catch runtime errors from MCTS engine
            self._logger.warning(f"MCTS runtime error: {e}")
            return {"best_action": None, "stats": {}}

    async def _process_with_agents(
        self,
        query: str,
        state: DomainStateT,
        context: dict[str, Any],
        mcts_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Process query with domain agents."""
        agent_results = {}

        # Add MCTS action to context
        enriched_context = {
            **context,
            "mcts_action": mcts_result.get("best_action"),
        }

        for agent_name, agent in self._agents.items():
            try:
                result = await agent.process(query, state, enriched_context)
                agent_results[agent_name] = result
            except (ValueError, TypeError, KeyError) as e:
                # Catch specific errors from agent processing
                self._logger.error(f"Agent {agent_name} failed with data error: {e}")
                agent_results[agent_name] = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "confidence": 0.0,
                }
            except (ConnectionError, TimeoutError, OSError) as e:
                # Catch network/IO related errors
                self._logger.error(f"Agent {agent_name} failed with IO error: {e}")
                agent_results[agent_name] = {
                    "error": str(e),
                    "error_type": "IOError",
                    "confidence": 0.0,
                }
            except AgentProcessingError:
                raise  # Re-raise our custom exceptions
            except RuntimeError as e:
                # Catch general runtime errors
                self._logger.error(f"Agent {agent_name} runtime error: {e}")
                agent_results[agent_name] = {
                    "error": str(e),
                    "error_type": "RuntimeError",
                    "confidence": 0.0,
                }

        return agent_results

    def _synthesize_results(
        self,
        agent_results: dict[str, Any],
        mcts_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Synthesize results from agents and MCTS."""
        # Aggregate agent responses
        responses = []
        confidences = []

        for _agent_name, result in agent_results.items():
            if "error" not in result:
                responses.append(result.get("response", ""))
                confidences.append(result.get("confidence", 0.5))

        # Compute aggregate confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Combine responses (simple concatenation for now)
        combined_response = "\n\n".join(filter(None, responses))

        return {
            "response": combined_response,
            "confidence": avg_confidence,
            "mcts_action": mcts_result.get("best_action"),
            "agent_count": len(responses),
        }
