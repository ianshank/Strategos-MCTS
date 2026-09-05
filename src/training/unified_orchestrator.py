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

from collections.abc import Callable
from pathlib import Path
import time
from typing import Any

import torch
from torch.amp import GradScaler
import torch.nn as nn

from src.utils.seeding import set_all_seeds

from ..agents.hrm_agent import HRMLoss, create_hrm_agent
from ..agents.trm_agent import TRMLoss, create_trm_agent
from ..framework.mcts.neural_mcts import GameState, NeuralMCTS, SelfPlayCollector
from ..models.policy_value_net import AlphaZeroLoss, create_policy_value_network
from ..observability.logging import LogContext, get_correlation_id, get_structured_logger, set_correlation_id
from .orchestrator_components.checkpoint_mixin import CheckpointMixin
from .orchestrator_components.metrics_mixin import MetricsMixin
from .orchestrator_components.selfplay_mixin import SelfPlayMixin
from .orchestrator_components.training_mixin import TrainingMixin
from .orchestrator_components.utils import _handle_training_failure, _strict_training_errors
from .performance_monitor import PerformanceMonitor, TimingContext
from .replay_buffer import PrioritizedReplayBuffer
from .system_config import SystemConfig

__all__ = [
    "UnifiedTrainingOrchestrator",
    "_handle_training_failure",
    "_strict_training_errors",
]

logger = get_structured_logger(__name__)


class UnifiedTrainingOrchestrator(MetricsMixin, CheckpointMixin, SelfPlayMixin, TrainingMixin):
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

    def __init__(self, config: SystemConfig, initial_state_fn: Callable[[], GameState], board_size: int = 19):
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
        self.device = config.device
        set_all_seeds(config.seed, rank=getattr(config, "rank", 0))
        self.monitor = PerformanceMonitor(window_size=100, enable_gpu_monitoring=self.device != "cpu")
        self._initialize_components()
        self.current_iteration = 0
        self.best_win_rate = 0.0
        self.best_model_path: Path | None = None
        self._setup_paths()
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
        pv_start = time.perf_counter()
        self.policy_value_net = create_policy_value_network(
            config=self.config.neural_net, board_size=self.board_size, device=self.device
        )
        if getattr(self.config, "distributed", False):
            from src.utils import distributed

            self.policy_value_net = distributed.wrap_ddp(self.policy_value_net, self.device)
        from src.utils import distributed

        model_for_params = distributed.unwrap_model(self.policy_value_net)
        pv_params = model_for_params.get_parameter_count()
        logger.info(
            "Policy-Value Network initialized",
            component="policy_value_net",
            parameter_count=pv_params,
            num_res_blocks=self.config.neural_net.num_res_blocks,
            num_channels=self.config.neural_net.num_channels,
            init_time_ms=round((time.perf_counter() - pv_start) * 1000, 2),
        )
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
        mcts_start = time.perf_counter()
        self.mcts = NeuralMCTS(
            policy_value_network=self.policy_value_net,
            config=self.config.mcts,
            device=self.device,
            seed=self.config.seed,
        )
        logger.info(
            "Neural MCTS initialized",
            component="neural_mcts",
            num_simulations=self.config.mcts.num_simulations,
            c_puct=self.config.mcts.c_puct,
            dirichlet_alpha=self.config.mcts.dirichlet_alpha,
            init_time_ms=round((time.perf_counter() - mcts_start) * 1000, 2),
        )
        self.self_play_collector = SelfPlayCollector(
            mcts=self.mcts, config=self.config.mcts, action_space_size=self.config.neural_net.action_size
        )
        logger.debug("Self-play collector initialized", component="self_play_collector")
        self._setup_optimizers()
        self.pv_loss_fn = AlphaZeroLoss(value_loss_weight=1.0)
        self.hrm_loss_fn = HRMLoss(ponder_weight=self.config.hrm.ponder_weight)
        self.trm_loss_fn = TRMLoss(
            task_loss_fn=nn.MSELoss(), supervision_weight_decay=self.config.trm.supervision_weight_decay
        )
        logger.debug(
            "Loss functions initialized",
            hrm_ponder_weight=self.config.hrm.ponder_weight,
            trm_supervision_decay=self.config.trm.supervision_weight_decay,
        )
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
        self.scaler = (
            GradScaler("cuda" if "cuda" in str(self.device) else "cpu", enabled=self.config.use_mixed_precision)
            if self.config.use_mixed_precision
            else None
        )
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
        self.pv_optimizer = torch.optim.SGD(
            self.policy_value_net.parameters(),
            lr=self.config.training.learning_rate,
            momentum=self.config.training.momentum,
            weight_decay=self.config.training.weight_decay,
        )
        self.hrm_optimizer = torch.optim.Adam(self.hrm_agent.parameters(), lr=0.001)
        self.trm_optimizer = torch.optim.Adam(self.trm_agent.parameters(), lr=0.001)
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
        from src.utils import distributed

        if getattr(self.config, "distributed", False) and (not distributed.is_main_process()):
            self.config.use_wandb = False
            return
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
            logger.error("Failed to initialize Weights & Biases", error=str(e), wandb_project=self.config.wandb_project)
            self.config.use_wandb = False

    async def train_iteration(self, iteration: int) -> dict[str, Any]:
        """
        Execute single training iteration.

        Args:
            iteration: Current iteration number

        Returns:
            Dictionary of metrics
        """
        import uuid

        iteration_correlation_id = f"iter-{iteration}-{uuid.uuid4().hex[:8]}"
        set_correlation_id(iteration_correlation_id)
        iteration_start_time = time.perf_counter()
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
                hrm_degraded = hrm_metrics.get("degraded", False)
                logger.log_agent_execution(
                    agent_name="HRM",
                    duration_ms=round((time.perf_counter() - phase_start) * 1000, 2),
                    confidence=0.0 if hrm_degraded else 1.0 - hrm_metrics.get("hrm_loss", 0.0),
                    success=not hrm_degraded,
                    iteration=iteration,
                    loss=hrm_metrics.get("hrm_loss", 0.0),
                    halt_step=hrm_metrics.get("hrm_halt_step", 0.0),
                )
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
                trm_degraded = trm_metrics.get("degraded", False)
                logger.log_agent_execution(
                    agent_name="TRM",
                    duration_ms=round((time.perf_counter() - phase_start) * 1000, 2),
                    confidence=0.0 if trm_degraded else 1.0 - trm_metrics.get("trm_loss", 0.0),
                    success=not trm_degraded,
                    iteration=iteration,
                    loss=trm_metrics.get("trm_loss", 0.0),
                    convergence_step=trm_metrics.get("trm_convergence_step", 0.0),
                )
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
        self._log_metrics(iteration, metrics)
        self.monitor.alert_if_slow()
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

    async def train(self, num_iterations: int):
        """
        Run complete training loop.

        Args:
            num_iterations: Number of training iterations
        """
        import uuid

        training_session_id = f"train-{uuid.uuid4().hex[:12]}"
        set_correlation_id(training_session_id)
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
                    "Training interrupted by user", iteration=iteration, completed_iterations=completed_iterations
                )
                final_status = "interrupted"
                break
            except Exception as e:
                logger.exception("Training iteration failed with error", iteration=iteration, error=str(e))
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
        logger.debug("Generating final performance summary")
        self.monitor.print_summary()

    def _should_early_stop(self, iteration: int) -> bool:
        """
        Check early stopping criteria based on win rate improvement.

        Uses patience-based early stopping: stop if no improvement
        for `patience` consecutive evaluations.
        """
        if iteration % self.config.training.checkpoint_interval != 0:
            return False
        if not hasattr(self, "_best_seen_win_rate"):
            self._best_seen_win_rate = 0.0
            self._iterations_without_improvement = 0
        current_win_rate = self.best_win_rate
        min_delta = self.config.training.min_delta
        if current_win_rate > self._best_seen_win_rate + min_delta:
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
        self._iterations_without_improvement += 1
        logger.debug(
            "No win rate improvement",
            iteration=iteration,
            current_win_rate=round(current_win_rate, 4),
            best_seen_win_rate=round(self._best_seen_win_rate, 4),
            iterations_without_improvement=self._iterations_without_improvement,
            patience=self.config.training.patience,
        )
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
