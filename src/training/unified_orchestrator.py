"""
Unified Training Orchestrator for LangGraph Multi-Agent MCTS with DeepMind-Style Learning.

Coordinates:
- HRM Agent
- TRM Agent
- Neural MCTS
- Policy-Value Network
- Self-play data generation
- Training loops
- Evaluation
- Checkpointing
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from ..agents.hrm_agent import HRMLoss, create_hrm_agent
from ..agents.trm_agent import TRMLoss, create_trm_agent
from ..framework.mcts.neural_mcts import GameState, NeuralMCTS, SelfPlayCollector
from ..models.policy_value_net import (
    AlphaZeroLoss,
    create_policy_value_network,
)
from ..observability.logging import (
    LogContext,
    get_correlation_id,
    get_structured_logger,
    set_correlation_id,
)
from .performance_monitor import PerformanceMonitor, TimingContext
from .replay_buffer import Experience, PrioritizedReplayBuffer, collate_experiences
from .system_config import SystemConfig

# Initialize structured logger for this module
logger = get_structured_logger(__name__)


class UnifiedTrainingOrchestrator:
    """
    Complete training pipeline integrating all framework components.

    This orchestrator manages:
    1. Self-play data generation using MCTS
    2. Neural network training (policy-value)
    3. HRM agent training
    4. TRM agent training
    5. Evaluation and checkpointing
    6. Performance monitoring
    """

    def __init__(
        self,
        config: SystemConfig,
        initial_state_fn: Callable[[], GameState],
        board_size: int = 19,
    ):
        """
        Initialize training orchestrator.

        Args:
            config: System configuration
            initial_state_fn: Function that returns initial game state
            board_size: Board/grid size for spatial games
        """
        self.config = config
        self.initial_state_fn = initial_state_fn
        self.board_size = board_size

        # Setup device
        self.device = config.device
        torch.manual_seed(config.seed)

        # Initialize performance monitor
        self.monitor = PerformanceMonitor(
            window_size=100,
            enable_gpu_monitoring=(self.device != "cpu"),
        )

        # Initialize components
        self._initialize_components()

        # Training state
        self.current_iteration = 0
        self.best_win_rate = 0.0
        self.best_model_path: Path | None = None

        # Setup paths
        self._setup_paths()

        # Setup experiment tracking
        if config.use_wandb:
            self._setup_wandb()

    def _initialize_components(self):
        """Initialize all framework components."""
        logger.info(
            "Initializing training orchestrator components",
            correlation_id=get_correlation_id(),
            device=self.device,
            board_size=self.board_size,
        )

        init_start_time = time.perf_counter()

        # Policy-Value Network
        pv_start = time.perf_counter()
        self.policy_value_net = create_policy_value_network(
            config=self.config.neural_net,
            board_size=self.board_size,
            device=self.device,
        )
        pv_params = self.policy_value_net.get_parameter_count()
        logger.info(
            "Policy-Value Network initialized",
            component="policy_value_net",
            parameter_count=pv_params,
            num_res_blocks=self.config.neural_net.num_res_blocks,
            num_channels=self.config.neural_net.num_channels,
            init_time_ms=round((time.perf_counter() - pv_start) * 1000, 2),
        )

        # HRM Agent
        hrm_start = time.perf_counter()
        self.hrm_agent = create_hrm_agent(self.config.hrm, self.device)
        hrm_params = self.hrm_agent.get_parameter_count()
        logger.info(
            "HRM Agent initialized",
            component="hrm_agent",
            parameter_count=hrm_params,
            h_dim=self.config.hrm.h_dim,
            l_dim=self.config.hrm.l_dim,
            max_outer_steps=self.config.hrm.max_outer_steps,
            init_time_ms=round((time.perf_counter() - hrm_start) * 1000, 2),
        )

        # TRM Agent
        trm_start = time.perf_counter()
        self.trm_agent = create_trm_agent(
            self.config.trm, output_dim=self.config.neural_net.action_size, device=self.device
        )
        trm_params = self.trm_agent.get_parameter_count()
        logger.info(
            "TRM Agent initialized",
            component="trm_agent",
            parameter_count=trm_params,
            latent_dim=self.config.trm.latent_dim,
            num_recursions=self.config.trm.num_recursions,
            init_time_ms=round((time.perf_counter() - trm_start) * 1000, 2),
        )

        # Neural MCTS
        mcts_start = time.perf_counter()
        self.mcts = NeuralMCTS(
            policy_value_network=self.policy_value_net,
            config=self.config.mcts,
            device=self.device,
        )
        logger.info(
            "Neural MCTS initialized",
            component="neural_mcts",
            num_simulations=self.config.mcts.num_simulations,
            c_puct=self.config.mcts.c_puct,
            dirichlet_alpha=self.config.mcts.dirichlet_alpha,
            init_time_ms=round((time.perf_counter() - mcts_start) * 1000, 2),
        )

        # Self-play collector
        self.self_play_collector = SelfPlayCollector(mcts=self.mcts, config=self.config.mcts)
        logger.debug("Self-play collector initialized", component="self_play_collector")

        # Optimizers
        self._setup_optimizers()

        # Loss functions
        self.pv_loss_fn = AlphaZeroLoss(value_loss_weight=1.0)
        self.hrm_loss_fn = HRMLoss(ponder_weight=self.config.hrm.ponder_weight)
        self.trm_loss_fn = TRMLoss(
            task_loss_fn=nn.MSELoss(),
            supervision_weight_decay=self.config.trm.supervision_weight_decay,
        )
        logger.debug(
            "Loss functions initialized",
            hrm_ponder_weight=self.config.hrm.ponder_weight,
            trm_supervision_decay=self.config.trm.supervision_weight_decay,
        )

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.training.buffer_size,
            alpha=0.6,
            beta_start=0.4,
            beta_frames=self.config.training.games_per_iteration * 10,
        )
        logger.info(
            "Replay buffer initialized",
            component="replay_buffer",
            capacity=self.config.training.buffer_size,
            alpha=0.6,
            beta_start=0.4,
        )

        # Mixed precision scaler
        self.scaler = GradScaler() if self.config.use_mixed_precision else None

        # Log total initialization summary
        total_params = pv_params + hrm_params + trm_params
        total_init_time = (time.perf_counter() - init_start_time) * 1000
        logger.info(
            "All components initialized successfully",
            total_parameter_count=total_params,
            total_init_time_ms=round(total_init_time, 2),
            mixed_precision_enabled=self.config.use_mixed_precision,
        )

    def _setup_optimizers(self):
        """Setup optimizers and learning rate schedulers."""
        # Policy-Value optimizer
        self.pv_optimizer = torch.optim.SGD(
            self.policy_value_net.parameters(),
            lr=self.config.training.learning_rate,
            momentum=self.config.training.momentum,
            weight_decay=self.config.training.weight_decay,
        )

        # HRM optimizer
        self.hrm_optimizer = torch.optim.Adam(self.hrm_agent.parameters(), lr=1e-3)

        # TRM optimizer
        self.trm_optimizer = torch.optim.Adam(self.trm_agent.parameters(), lr=1e-3)

        # Learning rate scheduler for policy-value network
        if self.config.training.lr_schedule == "cosine":
            self.pv_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.pv_optimizer, T_max=100)
        elif self.config.training.lr_schedule == "step":
            self.pv_scheduler = torch.optim.lr_scheduler.StepLR(
                self.pv_optimizer,
                step_size=self.config.training.lr_decay_steps,
                gamma=self.config.training.lr_decay_gamma,
            )
        else:
            self.pv_scheduler = None

    def _setup_paths(self):
        """Setup directory paths."""
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = Path(self.config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _setup_wandb(self):
        """Setup Weights & Biases experiment tracking."""
        try:
            import wandb

            run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=self.config.to_dict(),
                name=run_name,
            )
            logger.info(
                "Weights & Biases initialized",
                wandb_project=self.config.wandb_project,
                wandb_entity=self.config.wandb_entity,
                run_name=run_name,
            )
        except ImportError:
            logger.warning(
                "Weights & Biases not installed, experiment tracking disabled",
                recommendation="Install wandb with: pip install wandb",
            )
            self.config.use_wandb = False
        except Exception as e:
            logger.error(
                "Failed to initialize Weights & Biases",
                error=str(e),
                wandb_project=self.config.wandb_project,
            )
            self.config.use_wandb = False

    async def train_iteration(self, iteration: int) -> dict[str, Any]:
        """
        Execute single training iteration.

        Args:
            iteration: Current iteration number

        Returns:
            Dictionary of metrics
        """
        # Set up logging context for this iteration
        import uuid

        iteration_correlation_id = f"iter-{iteration}-{uuid.uuid4().hex[:8]}"
        set_correlation_id(iteration_correlation_id)

        iteration_start_time = time.perf_counter()

        # Log memory/GPU utilization at iteration start
        memory_info = self._get_memory_utilization()

        logger.info(
            "Training iteration started",
            iteration=iteration,
            total_iterations=self.current_iteration,
            device=self.device,
            buffer_size=len(self.replay_buffer) if hasattr(self.replay_buffer, "__len__") else "N/A",
            **memory_info,
        )

        metrics: dict[str, Any] = {}

        with LogContext(iteration=iteration, phase="training_iteration"):
            # Phase 1: Self-play data generation
            logger.info(
                "Phase 1/5: Starting self-play data generation",
                iteration=iteration,
                games_per_iteration=self.config.training.games_per_iteration,
            )
            phase_start = time.perf_counter()
            with TimingContext(self.monitor, "self_play_generation"):
                game_data = await self._generate_self_play_data()
                metrics["games_generated"] = len(game_data)
            logger.info(
                "Phase 1/5: Self-play data generation completed",
                iteration=iteration,
                examples_generated=len(game_data),
                phase_time_ms=round((time.perf_counter() - phase_start) * 1000, 2),
            )

            # Phase 2: Policy-Value network training
            logger.info(
                "Phase 2/5: Starting Policy-Value network training",
                iteration=iteration,
                batch_size=self.config.training.batch_size,
            )
            phase_start = time.perf_counter()
            with TimingContext(self.monitor, "pv_training"):
                pv_metrics = await self._train_policy_value_network()
                metrics.update(pv_metrics)
            logger.info(
                "Phase 2/5: Policy-Value network training completed",
                iteration=iteration,
                policy_loss=pv_metrics.get("policy_loss", 0.0),
                value_loss=pv_metrics.get("value_loss", 0.0),
                phase_time_ms=round((time.perf_counter() - phase_start) * 1000, 2),
            )

            # Phase 3: HRM agent training (optional, if using HRM)
            if hasattr(self, "hrm_agent"):
                logger.info(
                    "Phase 3/5: Starting HRM agent training",
                    iteration=iteration,
                    hrm_train_batches=self.config.training.hrm_train_batches,
                )
                phase_start = time.perf_counter()
                with TimingContext(self.monitor, "hrm_training"):
                    hrm_metrics = await self._train_hrm_agent()
                    metrics.update(hrm_metrics)
                logger.log_agent_execution(
                    agent_name="HRM",
                    duration_ms=round((time.perf_counter() - phase_start) * 1000, 2),
                    confidence=1.0 - hrm_metrics.get("hrm_loss", 0.0),
                    success=True,
                    iteration=iteration,
                    loss=hrm_metrics.get("hrm_loss", 0.0),
                    halt_step=hrm_metrics.get("hrm_halt_step", 0.0),
                )

            # Phase 4: TRM agent training (optional, if using TRM)
            if hasattr(self, "trm_agent"):
                logger.info(
                    "Phase 4/5: Starting TRM agent training",
                    iteration=iteration,
                    trm_train_batches=self.config.training.trm_train_batches,
                )
                phase_start = time.perf_counter()
                with TimingContext(self.monitor, "trm_training"):
                    trm_metrics = await self._train_trm_agent()
                    metrics.update(trm_metrics)
                logger.log_agent_execution(
                    agent_name="TRM",
                    duration_ms=round((time.perf_counter() - phase_start) * 1000, 2),
                    confidence=1.0 - trm_metrics.get("trm_loss", 0.0),
                    success=True,
                    iteration=iteration,
                    loss=trm_metrics.get("trm_loss", 0.0),
                    convergence_step=trm_metrics.get("trm_convergence_step", 0.0),
                )

            # Phase 5: Evaluation
            if iteration % self.config.training.checkpoint_interval == 0:
                logger.info(
                    "Phase 5/5: Starting model evaluation",
                    iteration=iteration,
                    evaluation_games=self.config.training.evaluation_games,
                    checkpoint_interval=self.config.training.checkpoint_interval,
                )
                phase_start = time.perf_counter()
                eval_metrics = await self._evaluate()
                metrics.update(eval_metrics)

                logger.info(
                    "Phase 5/5: Model evaluation completed",
                    iteration=iteration,
                    win_rate=eval_metrics.get("win_rate", 0.0),
                    wins=eval_metrics.get("wins", 0),
                    losses=eval_metrics.get("losses", 0),
                    draws=eval_metrics.get("draws", 0),
                    phase_time_ms=round((time.perf_counter() - phase_start) * 1000, 2),
                )

                # Save checkpoint if improved
                if eval_metrics.get("win_rate", 0) > self.best_win_rate:
                    old_best = self.best_win_rate
                    self.best_win_rate = eval_metrics["win_rate"]
                    self._save_checkpoint(iteration, metrics, is_best=True)
                    logger.info(
                        "New best model saved",
                        iteration=iteration,
                        new_win_rate=self.best_win_rate,
                        previous_win_rate=old_best,
                        improvement=self.best_win_rate - old_best,
                    )
            else:
                logger.debug(
                    "Phase 5/5: Evaluation skipped (not at checkpoint interval)",
                    iteration=iteration,
                    checkpoint_interval=self.config.training.checkpoint_interval,
                    next_evaluation_at=iteration
                    + (self.config.training.checkpoint_interval - iteration % self.config.training.checkpoint_interval),
                )

        # Log metrics
        self._log_metrics(iteration, metrics)

        # Performance check
        self.monitor.alert_if_slow()

        # Log iteration completion with summary
        iteration_time = time.perf_counter() - iteration_start_time
        final_memory_info = self._get_memory_utilization()

        logger.info(
            "Training iteration completed",
            iteration=iteration,
            iteration_time_seconds=round(iteration_time, 2),
            policy_loss=metrics.get("policy_loss", 0.0),
            value_loss=metrics.get("value_loss", 0.0),
            hrm_loss=metrics.get("hrm_loss", 0.0),
            trm_loss=metrics.get("trm_loss", 0.0),
            win_rate=metrics.get("win_rate"),
            best_win_rate=self.best_win_rate,
            **final_memory_info,
        )

        return metrics

    def _get_memory_utilization(self) -> dict[str, Any]:
        """Get current memory and GPU utilization metrics."""
        memory_info = {}

        # CPU memory
        process = psutil.Process()
        memory_info["cpu_memory_mb"] = round(process.memory_info().rss / (1024 * 1024), 2)
        memory_info["cpu_percent"] = process.cpu_percent()

        # GPU memory (if available)
        if self.device != "cpu" and torch.cuda.is_available():
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                memory_info["gpu_memory_allocated_mb"] = round(gpu_memory_allocated, 2)
                memory_info["gpu_memory_reserved_mb"] = round(gpu_memory_reserved, 2)
                memory_info["gpu_memory_total_mb"] = round(gpu_memory_total, 2)
                memory_info["gpu_utilization_percent"] = round((gpu_memory_allocated / gpu_memory_total) * 100, 2)
            except Exception as e:
                logger.debug("Failed to get GPU memory info", error=str(e))

        return memory_info

    async def _generate_self_play_data(self) -> list[Experience]:
        """Generate training data from self-play games."""
        num_games = self.config.training.games_per_iteration

        logger.debug(
            "Starting self-play data generation",
            total_games=num_games,
            temperature_threshold=self.config.mcts.temperature_threshold,
            mcts_simulations=self.config.mcts.num_simulations,
        )

        # In production, this would use parallel actors
        # For simplicity, we'll do sequential self-play
        all_examples = []
        game_start_time = time.perf_counter()
        log_interval = max(1, num_games // self.config.log_interval) if self.config.log_interval > 0 else 5

        for game_idx in range(num_games):
            game_iter_start = time.perf_counter()

            try:
                examples = await self.self_play_collector.play_game(
                    initial_state=self.initial_state_fn(),
                    temperature_threshold=self.config.mcts.temperature_threshold,
                )

                # Convert to Experience objects
                for ex in examples:
                    all_examples.append(Experience(state=ex.state, policy=ex.policy_target, value=ex.value_target))

                game_time = (time.perf_counter() - game_iter_start) * 1000

                # Log progress at regular intervals
                if (game_idx + 1) % log_interval == 0:
                    elapsed = time.perf_counter() - game_start_time
                    avg_game_time = elapsed / (game_idx + 1)
                    remaining_games = num_games - (game_idx + 1)
                    eta_seconds = remaining_games * avg_game_time

                    logger.debug(
                        "Self-play progress",
                        games_completed=game_idx + 1,
                        total_games=num_games,
                        progress_percent=round(((game_idx + 1) / num_games) * 100, 1),
                        examples_collected=len(all_examples),
                        last_game_time_ms=round(game_time, 2),
                        avg_game_time_ms=round(avg_game_time * 1000, 2),
                        eta_seconds=round(eta_seconds, 1),
                    )

            except Exception as e:
                logger.error(
                    "Self-play game failed",
                    game_idx=game_idx,
                    error=str(e),
                    examples_so_far=len(all_examples),
                )
                # Continue with remaining games rather than failing completely
                continue

        # Add to replay buffer
        buffer_size_before = len(self.replay_buffer) if hasattr(self.replay_buffer, "__len__") else 0
        self.replay_buffer.add_batch(all_examples)
        buffer_size_after = len(self.replay_buffer) if hasattr(self.replay_buffer, "__len__") else 0

        total_time = time.perf_counter() - game_start_time
        logger.info(
            "Self-play data generation completed",
            games_completed=num_games,
            examples_generated=len(all_examples),
            total_time_seconds=round(total_time, 2),
            avg_examples_per_game=round(len(all_examples) / max(num_games, 1), 2),
            buffer_size_before=buffer_size_before,
            buffer_size_after=buffer_size_after,
        )

        return all_examples

    async def _train_policy_value_network(self) -> dict[str, float]:
        """Train policy-value network on replay buffer data."""
        if not self.replay_buffer.is_ready(self.config.training.batch_size):
            logger.warning(
                "Replay buffer not ready for training",
                required_size=self.config.training.batch_size,
                current_size=len(self.replay_buffer) if hasattr(self.replay_buffer, "__len__") else "unknown",
            )
            return {"policy_loss": 0.0, "value_loss": 0.0}

        self.policy_value_net.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_combined_loss = 0.0
        num_batches = 10  # Train for 10 batches per iteration
        batch_times = []
        gradient_norms = []

        logger.debug(
            "Starting Policy-Value network training",
            num_batches=num_batches,
            batch_size=self.config.training.batch_size,
            learning_rate=self.pv_optimizer.param_groups[0]["lr"],
            mixed_precision=self.config.use_mixed_precision,
        )

        training_start = time.perf_counter()

        for batch_idx in range(num_batches):
            batch_start = time.perf_counter()

            # Sample batch
            experiences, indices, weights = self.replay_buffer.sample(self.config.training.batch_size)
            states, policies, values = collate_experiences(experiences)

            states = states.to(self.device)
            policies = policies.to(self.device)
            values = values.to(self.device)
            weights = torch.from_numpy(weights).to(self.device)

            # Forward pass
            if self.config.use_mixed_precision and self.scaler:
                with autocast():
                    policy_logits, value_pred = self.policy_value_net(states)
                    loss, loss_dict = self.pv_loss_fn(policy_logits, value_pred, policies, values)
                    # Apply importance sampling weights
                    loss = (loss * weights).mean()

                # Backward pass with mixed precision
                self.pv_optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Compute gradient norm before unscaling for logging
                self.scaler.unscale_(self.pv_optimizer)
                grad_norm = self._compute_gradient_norm(self.policy_value_net)
                gradient_norms.append(grad_norm)

                self.scaler.step(self.pv_optimizer)
                self.scaler.update()
            else:
                policy_logits, value_pred = self.policy_value_net(states)
                loss, loss_dict = self.pv_loss_fn(policy_logits, value_pred, policies, values)
                loss = (loss * weights).mean()

                self.pv_optimizer.zero_grad()
                loss.backward()

                # Compute gradient norm for logging
                grad_norm = self._compute_gradient_norm(self.policy_value_net)
                gradient_norms.append(grad_norm)

                self.pv_optimizer.step()

            # Update priorities in replay buffer
            with torch.no_grad():
                td_errors = torch.abs(value_pred.squeeze() - values)
                self.replay_buffer.update_priorities(indices, td_errors.cpu().numpy())

            batch_policy_loss = loss_dict["policy"]
            batch_value_loss = loss_dict["value"]
            batch_total_loss = loss_dict["total"]

            total_policy_loss += batch_policy_loss
            total_value_loss += batch_value_loss
            total_combined_loss += batch_total_loss

            batch_time = (time.perf_counter() - batch_start) * 1000
            batch_times.append(batch_time)

            # Log losses
            self.monitor.log_loss(batch_policy_loss, batch_value_loss, batch_total_loss)

            # Log detailed batch info at debug level
            logger.debug(
                "Policy-Value network batch completed",
                batch=batch_idx + 1,
                total_batches=num_batches,
                policy_loss=round(batch_policy_loss, 6),
                value_loss=round(batch_value_loss, 6),
                total_loss=round(batch_total_loss, 6),
                gradient_norm=round(grad_norm, 4),
                batch_time_ms=round(batch_time, 2),
                avg_td_error=round(td_errors.mean().item(), 6),
            )

        # Step learning rate scheduler
        old_lr = self.pv_optimizer.param_groups[0]["lr"]
        if self.pv_scheduler:
            self.pv_scheduler.step()
            new_lr = self.pv_optimizer.param_groups[0]["lr"]
            if new_lr != old_lr:
                logger.debug(
                    "Learning rate updated",
                    old_lr=old_lr,
                    new_lr=new_lr,
                    schedule=self.config.training.lr_schedule,
                )

        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_combined_loss = total_combined_loss / num_batches
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_gradient_norm = sum(gradient_norms) / len(gradient_norms)
        total_training_time = time.perf_counter() - training_start

        logger.info(
            "Policy-Value network training completed",
            avg_policy_loss=round(avg_policy_loss, 6),
            avg_value_loss=round(avg_value_loss, 6),
            avg_combined_loss=round(avg_combined_loss, 6),
            avg_gradient_norm=round(avg_gradient_norm, 4),
            max_gradient_norm=round(max(gradient_norms), 4),
            num_batches=num_batches,
            avg_batch_time_ms=round(avg_batch_time, 2),
            total_training_time_ms=round(total_training_time * 1000, 2),
            current_lr=self.pv_optimizer.param_groups[0]["lr"],
        )

        return {"policy_loss": avg_policy_loss, "value_loss": avg_value_loss}

    def _compute_gradient_norm(self, model: nn.Module) -> float:
        """Compute the total gradient norm for a model."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return float(total_norm**0.5)

    async def _train_hrm_agent(self) -> dict[str, float]:
        """
        Train HRM agent with proper loss computation.

        Uses:
        - Adaptive Computation Time loss
        - Ponder cost regularization
        - Convergence consistency loss
        """
        from .agent_trainer import (
            HRMTrainer,
            HRMTrainingConfig,
            create_data_loader_from_buffer,
        )

        logger.debug(
            "Initializing HRM agent training",
            batch_size=self.config.training.batch_size,
            num_batches=self.config.training.hrm_train_batches,
            ponder_weight=self.config.hrm.ponder_weight,
            gradient_clip_norm=self.config.training.gradient_clip_norm,
        )

        training_start = time.perf_counter()

        # Create training config from system config
        hrm_train_config = HRMTrainingConfig(
            batch_size=self.config.training.batch_size,
            num_batches=self.config.training.hrm_train_batches,
            gradient_clip_norm=self.config.training.gradient_clip_norm,
            ponder_weight=self.config.hrm.ponder_weight,
            use_mixed_precision=self.config.use_mixed_precision,
        )

        # Create trainer
        trainer = HRMTrainer(
            agent=self.hrm_agent,
            optimizer=self.hrm_optimizer,
            loss_fn=self.hrm_loss_fn,
            config=hrm_train_config,
            device=self.device,
            scaler=self.scaler,
        )

        # Create data loader from replay buffer
        try:
            data_loader = create_data_loader_from_buffer(
                replay_buffer=self.replay_buffer,
                batch_size=hrm_train_config.batch_size,
                input_dim=self.config.hrm.h_dim,
                output_dim=self.config.hrm.h_dim,
                device=self.device,
            )
        except Exception as e:
            logger.error(
                "Failed to create data loader for HRM training",
                error=str(e),
                batch_size=hrm_train_config.batch_size,
            )
            return {"hrm_loss": 0.0, "hrm_halt_step": 0.0, "hrm_ponder_cost": 0.0, "hrm_gradient_norm": 0.0}

        # Train for epoch
        try:
            metrics = await trainer.train_epoch(data_loader)
        except Exception as e:
            logger.error(
                "HRM agent training epoch failed",
                error=str(e),
            )
            return {"hrm_loss": 0.0, "hrm_halt_step": 0.0, "hrm_ponder_cost": 0.0, "hrm_gradient_norm": 0.0}

        # Extract key metrics for logging
        result = {
            "hrm_loss": metrics.get("loss", 0.0),
            "hrm_halt_step": metrics.get("hrm_halt_step", 0.0),
            "hrm_ponder_cost": metrics.get("hrm_ponder_cost", 0.0),
            "hrm_gradient_norm": metrics.get("gradient_norm", 0.0),
        }

        training_time = time.perf_counter() - training_start

        logger.info(
            "HRM agent training completed",
            loss=round(result["hrm_loss"], 6),
            halt_step=round(result["hrm_halt_step"], 2),
            ponder_cost=round(result["hrm_ponder_cost"], 6),
            gradient_norm=round(result["hrm_gradient_norm"], 4),
            num_batches=self.config.training.hrm_train_batches,
            training_time_ms=round(training_time * 1000, 2),
            h_dim=self.config.hrm.h_dim,
            l_dim=self.config.hrm.l_dim,
        )

        return result

    async def _train_trm_agent(self) -> dict[str, float]:
        """
        Train TRM agent with deep supervision.

        Uses:
        - Supervision at all recursion levels
        - Convergence monitoring
        - Residual norm tracking
        """
        from .agent_trainer import (
            TRMTrainer,
            TRMTrainingConfig,
            create_data_loader_from_buffer,
        )

        logger.debug(
            "Initializing TRM agent training",
            batch_size=self.config.training.batch_size,
            num_batches=self.config.training.trm_train_batches,
            supervision_weight_decay=self.config.trm.supervision_weight_decay,
            gradient_clip_norm=self.config.training.gradient_clip_norm,
        )

        training_start = time.perf_counter()

        # Create training config from system config
        trm_train_config = TRMTrainingConfig(
            batch_size=self.config.training.batch_size,
            num_batches=self.config.training.trm_train_batches,
            gradient_clip_norm=self.config.training.gradient_clip_norm,
            supervision_weight_decay=self.config.trm.supervision_weight_decay,
            use_mixed_precision=self.config.use_mixed_precision,
        )

        # Create trainer
        trainer = TRMTrainer(
            agent=self.trm_agent,
            optimizer=self.trm_optimizer,
            loss_fn=self.trm_loss_fn,
            config=trm_train_config,
            device=self.device,
            scaler=self.scaler,
        )

        # Create data loader from replay buffer
        try:
            data_loader = create_data_loader_from_buffer(
                replay_buffer=self.replay_buffer,
                batch_size=trm_train_config.batch_size,
                input_dim=self.config.trm.latent_dim,
                output_dim=self.config.neural_net.action_size,
                device=self.device,
            )
        except Exception as e:
            logger.error(
                "Failed to create data loader for TRM training",
                error=str(e),
                batch_size=trm_train_config.batch_size,
            )
            return {"trm_loss": 0.0, "trm_convergence_step": 0.0, "trm_final_residual": 0.0, "trm_gradient_norm": 0.0}

        # Train for epoch
        try:
            metrics = await trainer.train_epoch(data_loader)
        except Exception as e:
            logger.error(
                "TRM agent training epoch failed",
                error=str(e),
            )
            return {"trm_loss": 0.0, "trm_convergence_step": 0.0, "trm_final_residual": 0.0, "trm_gradient_norm": 0.0}

        # Extract key metrics for logging
        result = {
            "trm_loss": metrics.get("loss", 0.0),
            "trm_convergence_step": metrics.get("trm_convergence_step", 0.0),
            "trm_final_residual": metrics.get("trm_final_residual", 0.0),
            "trm_gradient_norm": metrics.get("gradient_norm", 0.0),
        }

        training_time = time.perf_counter() - training_start

        logger.info(
            "TRM agent training completed",
            loss=round(result["trm_loss"], 6),
            convergence_step=round(result["trm_convergence_step"], 2),
            final_residual=round(result["trm_final_residual"], 6),
            gradient_norm=round(result["trm_gradient_norm"], 4),
            num_batches=self.config.training.trm_train_batches,
            training_time_ms=round(training_time * 1000, 2),
            latent_dim=self.config.trm.latent_dim,
            num_recursions=self.config.trm.num_recursions,
        )

        return result

    async def _evaluate(self) -> dict[str, float]:
        """
        Evaluate current model against previous best through self-play.

        Uses arena-style evaluation with alternating starting positions.
        """
        from .agent_trainer import EvaluationConfig, SelfPlayEvaluator

        logger.info(
            "Starting model evaluation",
            num_games=self.config.training.evaluation_games,
            temperature=self.config.training.eval_temperature,
            mcts_iterations=self.config.mcts.num_simulations,
            win_threshold=self.config.training.win_threshold,
        )

        eval_start = time.perf_counter()

        # Create evaluation config from system config
        eval_config = EvaluationConfig(
            num_games=self.config.training.evaluation_games,
            temperature=self.config.training.eval_temperature,
            mcts_iterations=self.config.mcts.num_simulations,
            win_threshold=self.config.training.win_threshold,
        )

        # Load previous best model if available
        best_model = None
        if self.best_model_path is not None and self.best_model_path.exists():
            try:
                load_start = time.perf_counter()
                checkpoint = torch.load(
                    self.best_model_path,
                    map_location=self.device,
                    weights_only=True,
                )
                # Create a copy of the current model with best weights
                from copy import deepcopy

                best_model = deepcopy(self.policy_value_net)
                best_model.load_state_dict(checkpoint["policy_value_net"])
                best_model.eval()
                logger.debug(
                    "Loaded best model for evaluation",
                    model_path=str(self.best_model_path),
                    load_time_ms=round((time.perf_counter() - load_start) * 1000, 2),
                    checkpoint_iteration=checkpoint.get("iteration", "unknown"),
                    checkpoint_win_rate=checkpoint.get("best_win_rate", "unknown"),
                )
            except Exception as e:
                logger.warning(
                    "Could not load best model for evaluation",
                    error=str(e),
                    model_path=str(self.best_model_path) if self.best_model_path else None,
                )
                best_model = None
        else:
            logger.debug(
                "No previous best model available for comparison",
                best_model_path=str(self.best_model_path) if self.best_model_path else None,
            )

        # Create evaluator
        evaluator = SelfPlayEvaluator(
            mcts=self.mcts,
            initial_state_fn=self.initial_state_fn,
            config=eval_config,
            device=self.device,
        )

        # Run evaluation
        self.policy_value_net.eval()
        try:
            eval_run_start = time.perf_counter()
            metrics = await evaluator.evaluate(
                current_model=self.policy_value_net,
                best_model=best_model,
            )
            eval_run_time = time.perf_counter() - eval_run_start

            logger.info(
                "Model evaluation completed",
                win_rate=round(metrics.get("win_rate", 0.0), 4),
                wins=metrics.get("wins", 0),
                losses=metrics.get("losses", 0),
                draws=metrics.get("draws", 0),
                total_games=self.config.training.evaluation_games,
                avg_game_time_ms=round((eval_run_time / max(self.config.training.evaluation_games, 1)) * 1000, 2),
                evaluation_time_seconds=round(eval_run_time, 2),
                win_threshold=self.config.training.win_threshold,
                meets_threshold=metrics.get("win_rate", 0.0) >= self.config.training.win_threshold,
                compared_to_best=best_model is not None,
            )

        except Exception as e:
            logger.error(
                "Model evaluation failed",
                error=str(e),
            )
            # Return default metrics on failure
            metrics = {"win_rate": 0.0, "wins": 0, "losses": 0, "draws": 0}
        finally:
            self.policy_value_net.train()

        total_eval_time = time.perf_counter() - eval_start
        logger.debug(
            "Total evaluation phase time",
            total_time_seconds=round(total_eval_time, 2),
        )

        return metrics

    def _save_checkpoint(self, iteration: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        save_start = time.perf_counter()

        checkpoint = {
            "iteration": iteration,
            "policy_value_net": self.policy_value_net.state_dict(),
            "hrm_agent": self.hrm_agent.state_dict(),
            "trm_agent": self.trm_agent.state_dict(),
            "pv_optimizer": self.pv_optimizer.state_dict(),
            "hrm_optimizer": self.hrm_optimizer.state_dict(),
            "trm_optimizer": self.trm_optimizer.state_dict(),
            "config": self.config.to_dict(),
            "metrics": metrics,
            "best_win_rate": self.best_win_rate,
        }

        # Save regular checkpoint
        path = self.checkpoint_dir / f"checkpoint_iter_{iteration}.pt"
        try:
            torch.save(checkpoint, path)
            checkpoint_size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(
                "Checkpoint saved",
                checkpoint_path=str(path),
                iteration=iteration,
                checkpoint_size_mb=round(checkpoint_size_mb, 2),
                save_time_ms=round((time.perf_counter() - save_start) * 1000, 2),
                is_best=is_best,
            )
        except Exception as e:
            logger.error(
                "Failed to save checkpoint",
                error=str(e),
                checkpoint_path=str(path),
                iteration=iteration,
            )
            return

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            try:
                torch.save(checkpoint, best_path)
                self.best_model_path = best_path
                logger.info(
                    "Best model checkpoint saved",
                    best_model_path=str(best_path),
                    iteration=iteration,
                    win_rate=self.best_win_rate,
                )
            except Exception as e:
                logger.error(
                    "Failed to save best model checkpoint",
                    error=str(e),
                    best_model_path=str(best_path),
                )

    def _log_metrics(self, iteration: int, metrics: dict):
        """Log metrics to console and tracking systems."""
        # Log metrics summary at INFO level with structured data
        logger.info(
            "Iteration metrics summary",
            iteration=iteration,
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        )

        # Log to wandb
        if self.config.use_wandb:
            try:
                import wandb

                wandb_metrics = self.monitor.export_to_wandb(iteration)
                wandb_metrics.update(metrics)
                wandb.log(wandb_metrics, step=iteration)
                logger.debug(
                    "Metrics logged to Weights & Biases",
                    iteration=iteration,
                    num_metrics=len(wandb_metrics),
                )
            except Exception as e:
                logger.warning(
                    "Failed to log metrics to Weights & Biases",
                    error=str(e),
                    iteration=iteration,
                )

    async def train(self, num_iterations: int):
        """
        Run complete training loop.

        Args:
            num_iterations: Number of training iterations
        """
        import uuid

        training_session_id = f"train-{uuid.uuid4().hex[:12]}"
        set_correlation_id(training_session_id)

        # Get initial system state
        initial_memory = self._get_memory_utilization()

        logger.info(
            "Training session started",
            session_id=training_session_id,
            total_iterations=num_iterations,
            device=self.device,
            mixed_precision=self.config.use_mixed_precision,
            batch_size=self.config.training.batch_size,
            games_per_iteration=self.config.training.games_per_iteration,
            checkpoint_interval=self.config.training.checkpoint_interval,
            learning_rate=self.config.training.learning_rate,
            seed=self.config.seed,
            **initial_memory,
        )

        start_time = time.time()
        completed_iterations = 0
        final_status = "completed"

        for iteration in range(1, num_iterations + 1):
            self.current_iteration = iteration

            try:
                _ = await self.train_iteration(iteration)
                completed_iterations = iteration

                # Check early stopping
                if self._should_early_stop(iteration):
                    logger.warning(
                        "Early stopping triggered",
                        iteration=iteration,
                        best_win_rate=self.best_win_rate,
                        patience=self.config.training.patience,
                    )
                    final_status = "early_stopped"
                    break

            except KeyboardInterrupt:
                logger.warning(
                    "Training interrupted by user",
                    iteration=iteration,
                    completed_iterations=completed_iterations,
                )
                final_status = "interrupted"
                break
            except Exception as e:
                logger.exception(
                    "Training iteration failed with error",
                    iteration=iteration,
                    error=str(e),
                )
                final_status = "error"
                break

        elapsed = time.time() - start_time
        final_memory = self._get_memory_utilization()

        logger.info(
            "Training session completed",
            session_id=training_session_id,
            status=final_status,
            completed_iterations=completed_iterations,
            total_iterations=num_iterations,
            elapsed_hours=round(elapsed / 3600, 2),
            elapsed_seconds=round(elapsed, 2),
            best_win_rate=round(self.best_win_rate, 4),
            best_model_path=str(self.best_model_path) if self.best_model_path else None,
            avg_iteration_time_seconds=round(elapsed / max(completed_iterations, 1), 2),
            **final_memory,
        )

        # Log final performance summary
        logger.debug("Generating final performance summary")
        self.monitor.print_summary()

    def _should_early_stop(self, iteration: int) -> bool:
        """
        Check early stopping criteria based on win rate improvement.

        Uses patience-based early stopping: stop if no improvement
        for `patience` consecutive evaluations.
        """
        # Only check at evaluation intervals
        if iteration % self.config.training.checkpoint_interval != 0:
            return False

        # Initialize tracking if not exists
        if not hasattr(self, "_best_seen_win_rate"):
            self._best_seen_win_rate = 0.0
            self._iterations_without_improvement = 0

        # Check if current best is an improvement
        current_win_rate = self.best_win_rate
        min_delta = self.config.training.min_delta

        if current_win_rate > self._best_seen_win_rate + min_delta:
            # Improvement found
            previous_best = self._best_seen_win_rate
            self._best_seen_win_rate = current_win_rate
            self._iterations_without_improvement = 0

            logger.debug(
                "Win rate improvement detected",
                iteration=iteration,
                current_win_rate=round(current_win_rate, 4),
                previous_best=round(previous_best, 4),
                improvement=round(current_win_rate - previous_best, 4),
                min_delta=min_delta,
            )
            return False

        # No improvement
        self._iterations_without_improvement += 1

        logger.debug(
            "No win rate improvement",
            iteration=iteration,
            current_win_rate=round(current_win_rate, 4),
            best_seen_win_rate=round(self._best_seen_win_rate, 4),
            iterations_without_improvement=self._iterations_without_improvement,
            patience=self.config.training.patience,
        )

        # Check patience
        if self._iterations_without_improvement >= self.config.training.patience:
            logger.info(
                "Early stopping criteria met",
                iteration=iteration,
                iterations_without_improvement=self._iterations_without_improvement,
                patience=self.config.training.patience,
                best_win_rate_seen=round(self._best_seen_win_rate, 4),
                min_delta=min_delta,
            )
            return True

        return False

    def load_checkpoint(self, path: str):
        """Load checkpoint from file."""
        logger.info(
            "Loading checkpoint",
            checkpoint_path=path,
            device=self.device,
        )

        load_start = time.perf_counter()

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except Exception as e:
            logger.error(
                "Failed to load checkpoint file",
                checkpoint_path=path,
                error=str(e),
            )
            raise

        try:
            self.policy_value_net.load_state_dict(checkpoint["policy_value_net"])
            self.hrm_agent.load_state_dict(checkpoint["hrm_agent"])
            self.trm_agent.load_state_dict(checkpoint["trm_agent"])

            self.pv_optimizer.load_state_dict(checkpoint["pv_optimizer"])
            self.hrm_optimizer.load_state_dict(checkpoint["hrm_optimizer"])
            self.trm_optimizer.load_state_dict(checkpoint["trm_optimizer"])

            self.current_iteration = checkpoint["iteration"]
            self.best_win_rate = checkpoint.get("best_win_rate", 0.0)

            load_time = time.perf_counter() - load_start

            # Get checkpoint file size
            checkpoint_path = Path(path)
            checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024) if checkpoint_path.exists() else 0

            logger.info(
                "Checkpoint loaded successfully",
                checkpoint_path=path,
                iteration=self.current_iteration,
                best_win_rate=round(self.best_win_rate, 4),
                checkpoint_size_mb=round(checkpoint_size_mb, 2),
                load_time_ms=round(load_time * 1000, 2),
                checkpoint_metrics=checkpoint.get("metrics", {}),
            )

        except KeyError as e:
            logger.error(
                "Checkpoint is missing required keys",
                checkpoint_path=path,
                missing_key=str(e),
                available_keys=list(checkpoint.keys()),
            )
            raise
        except Exception as e:
            logger.error(
                "Failed to restore model state from checkpoint",
                checkpoint_path=path,
                error=str(e),
            )
            raise
