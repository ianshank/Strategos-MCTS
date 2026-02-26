"""
Factory patterns for creating framework components.

This module provides factory classes for creating various components
in a consistent, testable, and modular way. Factories enable:
- Dependency injection
- Component configuration management
- Easy mocking for testing
- Loose coupling between components

Best Practices 2025:
- Use Protocol classes for type safety
- Support async initialization
- Enable configuration from multiple sources
- Provide sensible defaults
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from src.adapters.llm.base import LLMClient
from src.config.settings import Settings, get_settings


class ComponentFactory(Protocol):
    """Protocol for component factories."""

    def create(self, **kwargs: Any) -> Any:
        """Create a component instance with the given configuration."""
        ...


class LLMClientFactory:
    """
    Factory for creating LLM client instances.

    Supports multiple providers with unified configuration.
    Enables easy testing via dependency injection.

    Example:
        >>> factory = LLMClientFactory()
        >>> client = factory.create(provider="openai", model="gpt-4")
        >>> # Or use settings-based creation
        >>> client = factory.create_from_settings()
    """

    def __init__(self, settings: Settings | None = None):
        """
        Initialize factory with optional settings.

        Args:
            settings: Settings instance (uses defaults if not provided)
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)

    def create(
        self,
        provider: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> LLMClient:
        """
        Create an LLM client with specified configuration.

        Args:
            provider: LLM provider name (openai, anthropic, lmstudio)
            model: Model identifier
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            **kwargs: Additional provider-specific arguments

        Returns:
            Configured LLM client instance

        Raises:
            ValueError: If provider is not supported
        """
        from src.adapters.llm import create_client

        provider = provider or self.settings.LLM_PROVIDER
        model = model or self._get_default_model(provider)
        timeout = timeout if timeout is not None else self.settings.HTTP_TIMEOUT_SECONDS
        max_retries = max_retries if max_retries is not None else self.settings.HTTP_MAX_RETRIES

        self.logger.info(f"Creating LLM client: provider={provider}, model={model}")

        return create_client(
            provider=provider,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

    def create_from_settings(self) -> LLMClient:
        """
        Create LLM client using current settings.

        Returns:
            Configured LLM client based on environment settings
        """
        return self.create()

    def _get_default_model(self, provider: str) -> str:
        """Get default model for a provider."""
        defaults = {
            "openai": "gpt-4-turbo-preview",
            "anthropic": "claude-3-sonnet-20240229",
            "lmstudio": "local-model",
        }
        return defaults.get(provider, "gpt-4-turbo-preview")


class AgentFactory:
    """
    Factory for creating agent instances.

    Manages agent lifecycle and dependencies.

    Example:
        >>> factory = AgentFactory(llm_client=client, logger=logger)
        >>> hrm_agent = factory.create_hrm_agent(max_depth=5)
        >>> trm_agent = factory.create_trm_agent(max_iterations=3)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        logger: logging.Logger | None = None,
        settings: Settings | None = None,
    ):
        """
        Initialize agent factory.

        Args:
            llm_client: LLM client for agent communication
            logger: Optional logger instance
            settings: Optional settings instance
        """
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
        self.settings = settings or get_settings()

    def create_hrm_agent(
        self,
        h_dim: int | None = None,
        l_dim: int | None = None,
        num_h_layers: int | None = None,
        num_l_layers: int | None = None,
        max_outer_steps: int | None = None,
        halt_threshold: float | None = None,
        device: str | None = None,
        **config: Any,
    ) -> Any:
        """
        Create a Hierarchical Reasoning Model (HRM) agent.

        All parameters are configurable - no hardcoded values.

        Args:
            h_dim: High-level planning dimension (default from settings)
            l_dim: Low-level execution dimension (default from settings)
            num_h_layers: Number of high-level layers
            num_l_layers: Number of low-level layers
            max_outer_steps: Maximum planning steps
            halt_threshold: Confidence threshold for halting
            device: Device to place model on (cpu/cuda)
            **config: Additional agent-specific configuration

        Returns:
            Configured HRM agent instance
        """
        from src.agents.hrm_agent import create_hrm_agent
        from src.training.system_config import HRMConfig

        # Build config from parameters, using settings as defaults
        hrm_config = HRMConfig(
            h_dim=h_dim if h_dim is not None else self.settings.HRM_H_DIM,
            l_dim=l_dim if l_dim is not None else self.settings.HRM_L_DIM,
            num_h_layers=num_h_layers if num_h_layers is not None else self.settings.HRM_NUM_H_LAYERS,
            num_l_layers=num_l_layers if num_l_layers is not None else self.settings.HRM_NUM_L_LAYERS,
            max_outer_steps=max_outer_steps if max_outer_steps is not None else self.settings.HRM_MAX_OUTER_STEPS,
            halt_threshold=halt_threshold if halt_threshold is not None else self.settings.HRM_HALT_THRESHOLD,
            **{k: v for k, v in config.items() if k in HRMConfig.__dataclass_fields__},
        )

        # Determine device
        agent_device = device if device is not None else "cpu"

        self.logger.info(
            "Creating HRM agent: h_dim=%d, l_dim=%d, device=%s",
            hrm_config.h_dim,
            hrm_config.l_dim,
            agent_device,
        )

        return create_hrm_agent(config=hrm_config, device=agent_device)

    def create_trm_agent(
        self,
        latent_dim: int | None = None,
        hidden_dim: int | None = None,
        num_recursions: int | None = None,
        convergence_threshold: float | None = None,
        deep_supervision: bool | None = None,
        output_dim: int | None = None,
        device: str | None = None,
        **config: Any,
    ) -> Any:
        """
        Create a Task Refinement Model (TRM) agent.

        All parameters are configurable - no hardcoded values.

        Args:
            latent_dim: Latent state dimension
            hidden_dim: Hidden layer dimension
            num_recursions: Maximum recursion depth
            convergence_threshold: L2 distance threshold for convergence
            deep_supervision: Enable supervision at all recursion levels
            output_dim: Output dimension (defaults to latent_dim)
            device: Device to place model on (cpu/cuda)
            **config: Additional agent-specific configuration

        Returns:
            Configured TRM agent instance
        """
        from src.agents.trm_agent import create_trm_agent
        from src.training.system_config import TRMConfig

        # Build config from parameters, using settings as defaults
        trm_config = TRMConfig(
            latent_dim=latent_dim if latent_dim is not None else self.settings.TRM_LATENT_DIM,
            hidden_dim=hidden_dim if hidden_dim is not None else self.settings.TRM_HIDDEN_DIM,
            num_recursions=num_recursions if num_recursions is not None else self.settings.TRM_NUM_RECURSIONS,
            convergence_threshold=convergence_threshold
            if convergence_threshold is not None
            else self.settings.TRM_CONVERGENCE_THRESHOLD,
            deep_supervision=deep_supervision if deep_supervision is not None else True,
            **{k: v for k, v in config.items() if k in TRMConfig.__dataclass_fields__},
        )

        # Determine device
        agent_device = device if device is not None else "cpu"

        self.logger.info(
            "Creating TRM agent: latent_dim=%d, num_recursions=%d, device=%s",
            trm_config.latent_dim,
            trm_config.num_recursions,
            agent_device,
        )

        return create_trm_agent(config=trm_config, output_dim=output_dim, device=agent_device)


class MCTSEngineFactory:
    """
    Factory for creating MCTS engine instances.

    Supports different MCTS implementations and configurations.

    Example:
        >>> factory = MCTSEngineFactory()
        >>> engine = factory.create(
        ...     seed=42,
        ...     exploration_weight=1.414,
        ...     config_preset="balanced"
        ... )
    """

    def __init__(self, settings: Settings | None = None):
        """
        Initialize MCTS engine factory.

        Args:
            settings: Optional settings instance
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)

    def create(
        self,
        seed: int | None = None,
        exploration_weight: float | None = None,
        config_preset: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create an MCTS engine instance.

        Args:
            seed: Random seed for determinism
            exploration_weight: UCB1 exploration constant
            config_preset: Configuration preset name (fast, balanced, thorough)
            **kwargs: Additional engine configuration

        Returns:
            Configured MCTS engine instance
        """
        from src.framework.mcts.core import MCTSEngine

        # Default seed to 42 if not provided
        seed = seed if seed is not None else (self.settings.SEED if self.settings.SEED is not None else 42)
        exploration_weight = exploration_weight if exploration_weight is not None else self.settings.MCTS_C

        if config_preset:
            config = self._get_preset_config(config_preset)
            kwargs.update(config)

        self.logger.info(
            f"Creating MCTS engine: seed={seed}, exploration_weight={exploration_weight}, preset={config_preset}"
        )

        return MCTSEngine(
            seed=seed,
            exploration_weight=exploration_weight,
            **kwargs,
        )

    def _get_preset_config(self, preset: str) -> dict[str, Any]:
        """Get configuration for a preset."""
        from src.framework.mcts.config import BALANCED_CONFIG, FAST_CONFIG, THOROUGH_CONFIG

        presets = {
            "fast": FAST_CONFIG.__dict__,
            "balanced": BALANCED_CONFIG.__dict__,
            "thorough": THOROUGH_CONFIG.__dict__,
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Valid presets: {list(presets.keys())}")

        return presets[preset]


class MetaControllerFactory:
    """
    Factory for creating neural meta-controller instances.

    Supports different meta-controller implementations:
    - RNN (GRU-based sequential routing)
    - BERT (DeBERTa-based semantic routing)
    - Hybrid (combines multiple approaches)
    - Assembly (assembly theory-based routing)

    Example:
        >>> factory = MetaControllerFactory()
        >>> controller = factory.create(controller_type="rnn")
    """

    def __init__(self, settings: Settings | None = None):
        """
        Initialize meta-controller factory.

        Args:
            settings: Optional settings instance
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)

    def create(
        self,
        controller_type: str | None = None,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        num_layers: int | None = None,
        num_agents: int | None = None,
        dropout: float | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create a meta-controller instance.

        Args:
            controller_type: Type of controller (rnn, bert, hybrid, assembly)
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            num_agents: Number of agents to route between
            dropout: Dropout rate
            device: Device to place model on (cpu/cuda)
            **kwargs: Additional controller-specific arguments

        Returns:
            Configured meta-controller instance
        """
        controller_type = controller_type or self.settings.META_CONTROLLER_TYPE
        input_dim = input_dim if input_dim is not None else self.settings.META_CONTROLLER_INPUT_DIM
        hidden_dim = hidden_dim if hidden_dim is not None else self.settings.META_CONTROLLER_HIDDEN_DIM
        num_layers = num_layers if num_layers is not None else self.settings.META_CONTROLLER_NUM_LAYERS
        num_agents = num_agents if num_agents is not None else self.settings.META_CONTROLLER_NUM_AGENTS
        dropout = dropout if dropout is not None else self.settings.META_CONTROLLER_DROPOUT
        device = device or "cpu"

        self.logger.info(
            f"Creating meta-controller: type={controller_type}, "
            f"input_dim={input_dim}, hidden_dim={hidden_dim}, device={device}"
        )

        if controller_type == "rnn":
            return self._create_rnn_controller(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                device=device,
                **kwargs,
            )
        elif controller_type == "bert":
            return self._create_bert_controller(
                dropout=dropout,
                device=device,
                **kwargs,
            )
        elif controller_type == "hybrid":
            return self._create_hybrid_controller(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                device=device,
                **kwargs,
            )
        elif controller_type == "assembly":
            return self._create_assembly_router(**kwargs)
        else:
            raise ValueError(f"Unknown controller type: {controller_type}. Valid types: rnn, bert, hybrid, assembly")

    def _create_rnn_controller(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        device: str,
        **kwargs: Any,
    ) -> Any:
        """Create RNN-based meta-controller."""
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        controller = RNNMetaController(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            device=device,
            **kwargs,
        )
        return controller

    def _create_bert_controller(
        self,
        dropout: float,
        device: str,
        **kwargs: Any,
    ) -> Any:
        """Create BERT-based meta-controller."""
        from src.agents.meta_controller.bert_controller import BERTMetaController

        controller = BERTMetaController(
            lora_dropout=dropout,
            device=device,
            **kwargs,
        )
        return controller

    def _create_hybrid_controller(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        device: str,
        **kwargs: Any,
    ) -> Any:
        """Create hybrid meta-controller combining neural and assembly routing."""
        from src.agents.meta_controller.hybrid_controller import HybridMetaController
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        # Create underlying neural controller
        neural_controller = RNNMetaController(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            device=device,
        )

        controller = HybridMetaController(
            neural_controller=neural_controller,
            **kwargs,
        )
        return controller

    def _create_assembly_router(
        self,
        **kwargs: Any,
    ) -> Any:
        """Create assembly theory-based router."""
        from src.agents.meta_controller.assembly_router import AssemblyRouter

        router = AssemblyRouter(**kwargs)
        return router


class HybridAgentFactory:
    """
    Factory for creating hybrid LLM-Neural agents.

    Combines neural networks with LLM reasoning for cost-effective decision making.

    Example:
        >>> factory = HybridAgentFactory(llm_client=client)
        >>> agent = factory.create(mode="adaptive")
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        settings: Settings | None = None,
    ):
        """
        Initialize hybrid agent factory.

        Args:
            llm_client: Optional LLM client (created if not provided)
            settings: Optional settings instance
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)
        self._llm_client = llm_client

    @property
    def llm_client(self) -> LLMClient:
        """Get or create LLM client."""
        if self._llm_client is None:
            llm_factory = LLMClientFactory(settings=self.settings)
            self._llm_client = llm_factory.create()
        return self._llm_client

    def create(
        self,
        policy_net: Any | None = None,
        value_net: Any | None = None,
        mode: str | None = None,
        policy_confidence_threshold: float | None = None,
        value_confidence_threshold: float | None = None,
        neural_cost_per_call: float | None = None,
        llm_cost_per_1k_tokens: float | None = None,
        track_costs: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Create a hybrid LLM-Neural agent.

        Args:
            policy_net: Optional policy network
            value_net: Optional value network
            mode: Operating mode (auto, neural_only, llm_only, adaptive)
            policy_confidence_threshold: Threshold for policy network confidence
            value_confidence_threshold: Threshold for value network confidence
            neural_cost_per_call: Cost per neural network inference (USD)
            llm_cost_per_1k_tokens: Cost per 1000 LLM tokens (USD)
            track_costs: Whether to track costs
            **kwargs: Additional configuration

        Returns:
            Configured hybrid agent instance
        """
        from typing import Literal, cast

        from src.agents.hybrid_agent import HybridAgent, HybridConfig

        # Cast mode to expected Literal type
        mode_value = mode or self.settings.HYBRID_MODE
        valid_modes = ("auto", "neural_only", "llm_only", "adaptive")
        if mode_value not in valid_modes:
            mode_value = "auto"  # Default to auto if invalid
        typed_mode = cast(Literal["auto", "neural_only", "llm_only", "adaptive"], mode_value)

        config = HybridConfig(
            mode=typed_mode,
            policy_confidence_threshold=policy_confidence_threshold or self.settings.HYBRID_POLICY_CONFIDENCE_THRESHOLD,
            value_confidence_threshold=value_confidence_threshold or self.settings.HYBRID_VALUE_CONFIDENCE_THRESHOLD,
            neural_cost_per_call=neural_cost_per_call or self.settings.HYBRID_NEURAL_COST_PER_CALL,
            llm_cost_per_1k_tokens=llm_cost_per_1k_tokens or self.settings.HYBRID_LLM_COST_PER_1K_TOKENS,
            track_costs=track_costs,
        )

        self.logger.info(
            f"Creating hybrid agent: mode={config.mode}, policy_threshold={config.policy_confidence_threshold}"
        )

        return HybridAgent(
            policy_net=policy_net,
            value_net=value_net,
            llm_client=self.llm_client,
            config=config,
            **kwargs,
        )


class FrameworkFactory:
    """
    Master factory for creating complete framework instances.

    Coordinates all component factories to build the entire system.

    Example:
        >>> factory = FrameworkFactory()
        >>> framework = factory.create_framework(
        ...     llm_provider="openai",
        ...     mcts_enabled=True,
        ...     mcts_seed=42
        ... )
    """

    def __init__(self, settings: Settings | None = None):
        """
        Initialize framework factory.

        Args:
            settings: Optional settings instance
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)
        self.llm_factory = LLMClientFactory(settings=self.settings)
        self.mcts_factory = MCTSEngineFactory(settings=self.settings)
        self.meta_controller_factory = MetaControllerFactory(settings=self.settings)

    def create_framework(
        self,
        llm_provider: str | None = None,
        mcts_enabled: bool | None = None,
        mcts_seed: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create a complete framework instance.

        Args:
            llm_provider: LLM provider to use
            mcts_enabled: Whether to enable MCTS
            mcts_seed: Random seed for MCTS
            **kwargs: Additional framework configuration

        Returns:
            Configured framework instance
        """
        mcts_enabled = mcts_enabled if mcts_enabled is not None else self.settings.MCTS_ENABLED

        # Create LLM client
        llm_client = self.llm_factory.create(provider=llm_provider)

        # Create agent factory
        agent_factory = AgentFactory(llm_client=llm_client, settings=self.settings)

        # Create MCTS engine if enabled
        mcts_engine = None
        if mcts_enabled:
            mcts_engine = self.mcts_factory.create(seed=mcts_seed)

        # Create meta-controller if requested
        meta_controller = None
        if kwargs.pop("meta_controller_enabled", True):
            meta_controller = self.meta_controller_factory.create()

        self.logger.info(
            f"Creating framework: provider={llm_provider}, "
            f"mcts_enabled={mcts_enabled}, "
            f"seed={mcts_seed}, "
            f"additional_config={list(kwargs.keys())}"
        )

        # Return configured framework components
        return {
            "llm_client": llm_client,
            "agent_factory": agent_factory,
            "mcts_engine": mcts_engine,
            "meta_controller": meta_controller,
            "settings": self.settings,
            "additional_config": kwargs,
        }

    def create_meta_controller(self, **kwargs: Any) -> Any:
        """
        Create a meta-controller using the framework's factory.

        Args:
            **kwargs: Meta-controller configuration

        Returns:
            Configured meta-controller instance
        """
        return self.meta_controller_factory.create(**kwargs)

    def create_hybrid_agent(
        self,
        policy_net: Any | None = None,
        value_net: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create a hybrid agent using the framework's LLM client.

        Args:
            policy_net: Optional policy network
            value_net: Optional value network
            **kwargs: Additional configuration

        Returns:
            Configured hybrid agent instance
        """
        llm_client = self.llm_factory.create()
        hybrid_factory = HybridAgentFactory(
            llm_client=llm_client,
            settings=self.settings,
        )
        return hybrid_factory.create(
            policy_net=policy_net,
            value_net=value_net,
            **kwargs,
        )


# Convenience function for quick framework creation
def create_framework(**kwargs: Any) -> Any:
    """
    Convenience function to create a framework instance.

    Args:
        **kwargs: Framework configuration arguments

    Returns:
        Configured framework instance

    Example:
        >>> framework = create_framework(llm_provider="openai", mcts_enabled=True)
    """
    factory = FrameworkFactory()
    return factory.create_framework(**kwargs)
