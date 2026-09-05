# mypy: disable-error-code="attr-defined,misc,assignment"
import time
from typing import Any

import torch
from torch.amp import autocast
import torch.nn as nn

from src.observability.logging import get_structured_logger
from src.training.replay_buffer import collate_experiences

from .utils import _handle_training_failure

logger = get_structured_logger(__name__)


class TrainingMixin:

    async def _train_policy_value_network(self) -> dict[str, float]:
        """Train policy-value network on replay buffer data."""
        if not self.replay_buffer.is_ready(self.config.training.batch_size):
            logger.warning(
                "Replay buffer not ready for training",
                event="training_step_skipped_buffer_not_ready",
                required_size=self.config.training.batch_size,
                current_size=len(self.replay_buffer) if hasattr(self.replay_buffer, "__len__") else "unknown",
            )
            return {"policy_loss": 0.0, "value_loss": 0.0}
        self.policy_value_net.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_combined_loss = 0.0
        num_batches = 10
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
            experiences, indices, weights = self.replay_buffer.sample(self.config.training.batch_size)
            states, policies, values = collate_experiences(experiences)
            states = states.to(self.device)
            policies = policies.to(self.device)
            values = values.to(self.device)
            weights = torch.from_numpy(weights).to(self.device)
            if self.config.use_mixed_precision and self.scaler:
                with autocast(
                    device_type="cuda" if "cuda" in str(self.device) else "cpu", enabled=self.config.use_mixed_precision
                ):
                    policy_logits, value_pred = self.policy_value_net(states)
                    loss, loss_dict = self.pv_loss_fn(policy_logits, value_pred, policies, values)
                    loss = (loss * weights).mean()
                self.pv_optimizer.zero_grad()
                self.scaler.scale(loss).backward()
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
                grad_norm = self._compute_gradient_norm(self.policy_value_net)
                gradient_norms.append(grad_norm)
                self.pv_optimizer.step()
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
            self.monitor.log_loss(batch_policy_loss, batch_value_loss, batch_total_loss)
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
        old_lr = self.pv_optimizer.param_groups[0]["lr"]
        if self.pv_scheduler:
            self.pv_scheduler.step()
            new_lr = self.pv_optimizer.param_groups[0]["lr"]
            if new_lr != old_lr:
                logger.debug(
                    "Learning rate updated", old_lr=old_lr, new_lr=new_lr, schedule=self.config.training.lr_schedule
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

    async def _train_hrm_agent(self) -> dict[str, Any]:
        """
        Train HRM agent with proper loss computation.

        Uses:
        - Adaptive Computation Time loss
        - Ponder cost regularization
        - Convergence consistency loss
        """
        from ..agent_trainer import HRMTrainer, HRMTrainingConfig, create_data_loader_from_buffer

        logger.debug(
            "Initializing HRM agent training",
            batch_size=self.config.training.batch_size,
            num_batches=self.config.training.hrm_train_batches,
            ponder_weight=self.config.hrm.ponder_weight,
            gradient_clip_norm=self.config.training.gradient_clip_norm,
        )
        training_start = time.perf_counter()
        hrm_train_config = HRMTrainingConfig(
            batch_size=self.config.training.batch_size,
            num_batches=self.config.training.hrm_train_batches,
            gradient_clip_norm=self.config.training.gradient_clip_norm,
            ponder_weight=self.config.hrm.ponder_weight,
            use_mixed_precision=self.config.use_mixed_precision,
        )
        trainer = HRMTrainer(
            agent=self.hrm_agent,
            optimizer=self.hrm_optimizer,
            loss_fn=self.hrm_loss_fn,
            config=hrm_train_config,
            device=self.device,
            scaler=self.scaler,
        )
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
                "Failed to create data loader for HRM training", error=str(e), batch_size=hrm_train_config.batch_size
            )
            return _handle_training_failure(
                stage="hrm_data_loader",
                reason=str(e),
                zero_metrics={"hrm_loss": 0.0, "hrm_halt_step": 0.0, "hrm_ponder_cost": 0.0, "hrm_gradient_norm": 0.0},
            )
        try:
            metrics = await trainer.train_epoch(data_loader)
        except Exception as e:
            logger.error("HRM agent training epoch failed", error=str(e))
            return _handle_training_failure(
                stage="hrm_train_epoch",
                reason=str(e),
                zero_metrics={"hrm_loss": 0.0, "hrm_halt_step": 0.0, "hrm_ponder_cost": 0.0, "hrm_gradient_norm": 0.0},
            )

        def _to_flt(v: Any) -> float:
            return float(v.item() if hasattr(v, "item") else v or 0.0)

        result = {
            "hrm_loss": _to_flt(metrics.get("loss", 0.0)),
            "hrm_halt_step": _to_flt(metrics.get("hrm_halt_step", 0.0)),
            "hrm_ponder_cost": _to_flt(metrics.get("hrm_ponder_cost", 0.0)),
            "hrm_gradient_norm": _to_flt(metrics.get("gradient_norm", 0.0)),
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

    async def _train_trm_agent(self) -> dict[str, Any]:
        """
        Train TRM agent with deep supervision.

        Uses:
        - Supervision at all recursion levels
        - Convergence monitoring
        - Residual norm tracking
        """
        from ..agent_trainer import TRMTrainer, TRMTrainingConfig, create_data_loader_from_buffer

        logger.debug(
            "Initializing TRM agent training",
            batch_size=self.config.training.batch_size,
            num_batches=self.config.training.trm_train_batches,
            supervision_weight_decay=self.config.trm.supervision_weight_decay,
            gradient_clip_norm=self.config.training.gradient_clip_norm,
        )
        training_start = time.perf_counter()
        trm_train_config = TRMTrainingConfig(
            batch_size=self.config.training.batch_size,
            num_batches=self.config.training.trm_train_batches,
            gradient_clip_norm=self.config.training.gradient_clip_norm,
            supervision_weight_decay=self.config.trm.supervision_weight_decay,
            use_mixed_precision=self.config.use_mixed_precision,
        )
        trainer = TRMTrainer(
            agent=self.trm_agent,
            optimizer=self.trm_optimizer,
            loss_fn=self.trm_loss_fn,
            config=trm_train_config,
            device=self.device,
            scaler=self.scaler,
        )
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
                "Failed to create data loader for TRM training", error=str(e), batch_size=trm_train_config.batch_size
            )
            return _handle_training_failure(
                stage="trm_data_loader",
                reason=str(e),
                zero_metrics={
                    "trm_loss": 0.0,
                    "trm_convergence_step": 0.0,
                    "trm_final_residual": 0.0,
                    "trm_gradient_norm": 0.0,
                },
            )
        try:
            metrics = await trainer.train_epoch(data_loader)
        except Exception as e:
            logger.error("TRM agent training epoch failed", error=str(e))
            return _handle_training_failure(
                stage="trm_train_epoch",
                reason=str(e),
                zero_metrics={
                    "trm_loss": 0.0,
                    "trm_convergence_step": 0.0,
                    "trm_final_residual": 0.0,
                    "trm_gradient_norm": 0.0,
                },
            )

        def _to_flt(v: Any) -> float:
            return float(v.item() if hasattr(v, "item") else v or 0.0)

        result = {
            "trm_loss": _to_flt(metrics.get("loss", 0.0)),
            "trm_convergence_step": _to_flt(metrics.get("trm_convergence_step", 0.0)),
            "trm_final_residual": _to_flt(metrics.get("trm_final_residual", 0.0)),
            "trm_gradient_norm": _to_flt(metrics.get("gradient_norm", 0.0)),
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
        from ..agent_trainer import EvaluationConfig, SelfPlayEvaluator

        logger.info(
            "Starting model evaluation",
            num_games=self.config.training.evaluation_games,
            temperature=self.config.training.eval_temperature,
            mcts_iterations=self.config.mcts.num_simulations,
            win_threshold=self.config.training.win_threshold,
        )
        eval_start = time.perf_counter()
        eval_config = EvaluationConfig(
            num_games=self.config.training.evaluation_games,
            temperature=self.config.training.eval_temperature,
            mcts_iterations=self.config.mcts.num_simulations,
            win_threshold=self.config.training.win_threshold,
        )
        best_model = None
        if self.best_model_path is not None and self.best_model_path.exists():
            try:
                load_start = time.perf_counter()
                checkpoint = torch.load(self.best_model_path, map_location=self.device, weights_only=True)
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
        evaluator = SelfPlayEvaluator(
            mcts=self.mcts, initial_state_fn=self.initial_state_fn, config=eval_config, device=self.device
        )
        self.policy_value_net.eval()
        try:
            eval_run_start = time.perf_counter()
            metrics = await evaluator.evaluate(current_model=self.policy_value_net, best_model=best_model)
            eval_run_time = time.perf_counter() - eval_run_start
            logger.info(
                "Model evaluation completed",
                win_rate=round(metrics.get("win_rate", 0.0), 4),
                wins=metrics.get("wins", 0),
                losses=metrics.get("losses", 0),
                draws=metrics.get("draws", 0),
                total_games=self.config.training.evaluation_games,
                avg_game_time_ms=round(eval_run_time / max(self.config.training.evaluation_games, 1) * 1000, 2),
                evaluation_time_seconds=round(eval_run_time, 2),
                win_threshold=self.config.training.win_threshold,
                meets_threshold=metrics.get("win_rate", 0.0) >= self.config.training.win_threshold,
                compared_to_best=best_model is not None,
            )
        except Exception as e:
            logger.error("Model evaluation failed", error=str(e))
            metrics = {"win_rate": 0.0, "wins": 0, "losses": 0, "draws": 0}
        finally:
            self.policy_value_net.train()
        total_eval_time = time.perf_counter() - eval_start
        logger.debug("Total evaluation phase time", total_time_seconds=round(total_eval_time, 2))
        return metrics
