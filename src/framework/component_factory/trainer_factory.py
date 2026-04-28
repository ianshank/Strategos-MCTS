"""Internal submodule (split from component_factory.py)."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, cast

from src.config.settings import Settings, get_settings
from src.observability.logging import StructuredLogger, get_structured_logger

from .configs import TrainerConfig

if TYPE_CHECKING:
    import torch.optim as optim

    from src.agents.hrm_agent import HRMAgent, HRMLoss
    from src.agents.trm_agent import TRMAgent, TRMLoss
    from src.framework.mcts.neural_mcts import NeuralMCTS
    from src.training.agent_trainer import HRMTrainer, SelfPlayEvaluator, TRMTrainer
    from src.training.replay_buffer import AugmentedReplayBuffer, PrioritizedReplayBuffer, ReplayBuffer


class TrainerFactory:
    """
    Factory for creating training components.

    Creates:
    - HRMTrainer: Hierarchical Reasoning Model trainer
    - TRMTrainer: Task Refinement Model trainer
    - SelfPlayEvaluator: Model evaluation through self-play
    - ReplayBuffer: Experience replay buffer (uniform, prioritized, augmented)

    Example:
        >>> factory = TrainerFactory(settings=get_settings())
        >>> hrm_trainer = factory.create_hrm_trainer(
        ...     agent=my_hrm_agent,
        ...     optimizer=my_optimizer,
        ...     loss_fn=hrm_loss,
        ... )
        >>> buffer = factory.create_replay_buffer(buffer_type="prioritized")
    """

    # Singleton instance cache for expensive components
    _replay_buffer_instances: dict[str, Any] = {}
    _instance_lock = threading.Lock()

    def __init__(
        self,
        settings: Settings | None = None,
        logger: StructuredLogger | None = None,
        config: TrainerConfig | None = None,
    ) -> None:
        """
        Initialize trainer factory.

        Args:
            settings: Application settings (uses defaults if not provided)
            logger: Optional logger instance
            config: Optional trainer configuration (derived from settings if not provided)
        """
        self._settings = settings or get_settings()
        self._logger = logger or get_structured_logger(__name__)
        self._config = config or TrainerConfig.from_settings(self._settings)

    def create_hrm_trainer(
        self,
        agent: HRMAgent,
        optimizer: optim.Optimizer,
        loss_fn: HRMLoss,
        batch_size: int | None = None,
        num_batches: int | None = None,
        gradient_clip_norm: float | None = None,
        ponder_weight: float | None = None,
        consistency_weight: float | None = None,
        use_mixed_precision: bool | None = None,
        device: str | None = None,
        scaler: Any | None = None,
        **_kwargs: Any,
    ) -> HRMTrainer:
        """
        Create an HRM (Hierarchical Reasoning Model) trainer.

        Args:
            agent: HRM agent to train
            optimizer: PyTorch optimizer
            loss_fn: HRM loss function
            batch_size: Training batch size
            num_batches: Number of batches per epoch
            gradient_clip_norm: Maximum gradient norm for clipping
            ponder_weight: Weight for ponder cost regularization
            consistency_weight: Weight for consistency loss
            use_mixed_precision: Enable mixed precision training
            device: Device for training (cpu/cuda)
            scaler: Optional gradient scaler for mixed precision
            **kwargs: Additional configuration

        Returns:
            Configured HRMTrainer instance

        Example:
            >>> trainer = factory.create_hrm_trainer(
            ...     agent=hrm_agent,
            ...     optimizer=torch.optim.Adam(hrm_agent.parameters(), lr=0.001),
            ...     loss_fn=HRMLoss(),
            ... )
        """
        from src.training.agent_trainer import HRMTrainer, HRMTrainingConfig

        # Build config using factory defaults and overrides
        training_config = HRMTrainingConfig(
            batch_size=batch_size if batch_size is not None else self._config.batch_size,
            num_batches=num_batches if num_batches is not None else self._config.num_batches,
            gradient_clip_norm=(
                gradient_clip_norm if gradient_clip_norm is not None else self._config.gradient_clip_norm
            ),
            ponder_weight=ponder_weight if ponder_weight is not None else self._config.ponder_weight,
            consistency_weight=(
                consistency_weight if consistency_weight is not None else self._config.consistency_weight
            ),
            use_mixed_precision=(
                use_mixed_precision if use_mixed_precision is not None else self._config.use_mixed_precision
            ),
        )

        trainer_device = device if device is not None else self._config.device

        self._logger.info(
            "Creating HRM trainer",
            batch_size=training_config.batch_size,
            device=trainer_device,
            mixed_precision=training_config.use_mixed_precision,
        )

        return HRMTrainer(
            agent=agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=training_config,
            device=trainer_device,
            scaler=scaler,
        )

    def create_trm_trainer(
        self,
        agent: TRMAgent,
        optimizer: optim.Optimizer,
        loss_fn: TRMLoss,
        batch_size: int | None = None,
        num_batches: int | None = None,
        gradient_clip_norm: float | None = None,
        supervision_weight_decay: float | None = None,
        use_mixed_precision: bool | None = None,
        device: str | None = None,
        scaler: Any | None = None,
        **_kwargs: Any,
    ) -> TRMTrainer:
        """
        Create a TRM (Task Refinement Model) trainer.

        Args:
            agent: TRM agent to train
            optimizer: PyTorch optimizer
            loss_fn: TRM loss function
            batch_size: Training batch size
            num_batches: Number of batches per epoch
            gradient_clip_norm: Maximum gradient norm for clipping
            supervision_weight_decay: Decay factor for intermediate supervision weights
            use_mixed_precision: Enable mixed precision training
            device: Device for training (cpu/cuda)
            scaler: Optional gradient scaler for mixed precision
            **kwargs: Additional configuration

        Returns:
            Configured TRMTrainer instance

        Example:
            >>> trainer = factory.create_trm_trainer(
            ...     agent=trm_agent,
            ...     optimizer=torch.optim.Adam(trm_agent.parameters(), lr=0.001),
            ...     loss_fn=TRMLoss(),
            ... )
        """
        from src.training.agent_trainer import TRMTrainer, TRMTrainingConfig

        training_config = TRMTrainingConfig(
            batch_size=batch_size if batch_size is not None else self._config.batch_size,
            num_batches=num_batches if num_batches is not None else self._config.num_batches,
            gradient_clip_norm=(
                gradient_clip_norm if gradient_clip_norm is not None else self._config.gradient_clip_norm
            ),
            supervision_weight_decay=(
                supervision_weight_decay
                if supervision_weight_decay is not None
                else self._config.supervision_weight_decay
            ),
            use_mixed_precision=(
                use_mixed_precision if use_mixed_precision is not None else self._config.use_mixed_precision
            ),
        )

        trainer_device = device if device is not None else self._config.device

        self._logger.info(
            "Creating TRM trainer",
            batch_size=training_config.batch_size,
            device=trainer_device,
            supervision_weight_decay=training_config.supervision_weight_decay,
        )

        return TRMTrainer(
            agent=agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=training_config,
            device=trainer_device,
            scaler=scaler,
        )

    def create_self_play_evaluator(
        self,
        mcts: NeuralMCTS,
        initial_state_fn: Any,
        num_games: int | None = None,
        temperature: float | None = None,
        mcts_iterations: int | None = None,
        win_threshold: float | None = None,
        device: str | None = None,
        **_kwargs: Any,
    ) -> SelfPlayEvaluator:
        """
        Create a self-play evaluator for model comparison.

        Args:
            mcts: Neural MCTS instance
            initial_state_fn: Function to create initial game states
            num_games: Number of evaluation games
            temperature: Temperature for move selection (0 = deterministic)
            mcts_iterations: Number of MCTS iterations per move
            win_threshold: Minimum win rate to consider model better
            device: Device for evaluation (cpu/cuda)
            **kwargs: Additional configuration

        Returns:
            Configured SelfPlayEvaluator instance

        Example:
            >>> evaluator = factory.create_self_play_evaluator(
            ...     mcts=neural_mcts,
            ...     initial_state_fn=create_initial_state,
            ...     num_games=50,
            ... )
            >>> result = await evaluator.evaluate(current_model, best_model)
        """
        from src.training.agent_trainer import EvaluationConfig, SelfPlayEvaluator

        eval_config = EvaluationConfig(
            num_games=num_games if num_games is not None else self._config.num_eval_games,
            temperature=temperature if temperature is not None else self._config.eval_temperature,
            mcts_iterations=mcts_iterations if mcts_iterations is not None else self._config.mcts_iterations,
            win_threshold=win_threshold if win_threshold is not None else self._config.win_threshold,
        )

        eval_device = device if device is not None else self._config.device

        self._logger.info(
            "Creating self-play evaluator",
            num_games=eval_config.num_games,
            mcts_iterations=eval_config.mcts_iterations,
            device=eval_device,
        )

        return SelfPlayEvaluator(
            mcts=mcts,
            initial_state_fn=initial_state_fn,
            config=eval_config,
            device=eval_device,
        )

    def create_replay_buffer(
        self,
        buffer_type: str = "uniform",
        capacity: int | None = None,
        alpha: float | None = None,
        beta_start: float | None = None,
        beta_frames: int | None = None,
        augmentation_fn: Any | None = None,
        use_singleton: bool = True,
        **_kwargs: Any,
    ) -> ReplayBuffer | PrioritizedReplayBuffer | AugmentedReplayBuffer:
        """
        Create an experience replay buffer.

        Supports:
        - uniform: Simple uniform sampling replay buffer
        - prioritized: Prioritized experience replay (PER)
        - augmented: Replay buffer with data augmentation

        Args:
            buffer_type: Type of buffer ("uniform", "prioritized", "augmented")
            capacity: Maximum buffer capacity
            alpha: Priority exponent for PER (0=uniform, 1=full prioritization)
            beta_start: Initial importance sampling weight for PER
            beta_frames: Number of frames to anneal beta to 1.0 for PER
            augmentation_fn: Augmentation function for augmented buffer
            use_singleton: Whether to use cached singleton instance
            **kwargs: Additional configuration

        Returns:
            Configured replay buffer instance

        Example:
            >>> # Create prioritized replay buffer
            >>> buffer = factory.create_replay_buffer(
            ...     buffer_type="prioritized",
            ...     capacity=100000,
            ...     alpha=0.6,
            ... )
        """
        from src.training.replay_buffer import AugmentedReplayBuffer, PrioritizedReplayBuffer, ReplayBuffer

        buffer_capacity = capacity if capacity is not None else self._config.buffer_capacity
        cache_key = f"{buffer_type}_{buffer_capacity}"

        # Check for cached singleton
        if use_singleton and cache_key in self._replay_buffer_instances:
            self._logger.info(
                "Returning cached replay buffer",
                buffer_type=buffer_type,
                capacity=buffer_capacity,
            )
            return cast(
                "ReplayBuffer | PrioritizedReplayBuffer | AugmentedReplayBuffer",
                self._replay_buffer_instances[cache_key],
            )

        self._logger.info(
            "Creating replay buffer",
            buffer_type=buffer_type,
            capacity=buffer_capacity,
        )

        buffer: ReplayBuffer | PrioritizedReplayBuffer | AugmentedReplayBuffer

        if buffer_type == "uniform":
            buffer = ReplayBuffer(capacity=buffer_capacity)
        elif buffer_type == "prioritized":
            buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity,
                alpha=alpha if alpha is not None else self._config.prioritized_alpha,
                beta_start=beta_start if beta_start is not None else self._config.prioritized_beta_start,
                beta_frames=beta_frames if beta_frames is not None else self._config.prioritized_beta_frames,
            )
        elif buffer_type == "augmented":
            buffer = AugmentedReplayBuffer(
                capacity=buffer_capacity,
                augmentation_fn=augmentation_fn,
            )
        else:
            raise ValueError(f"Unknown buffer_type: {buffer_type}. Valid types: uniform, prioritized, augmented")

        # Cache singleton if requested
        if use_singleton:
            with self._instance_lock:
                self._replay_buffer_instances[cache_key] = buffer

        return buffer

    @classmethod
    def clear_singleton_cache(cls) -> None:
        """Clear the singleton instance cache for replay buffers."""
        with cls._instance_lock:
            cls._replay_buffer_instances.clear()
