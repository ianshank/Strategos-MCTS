"""
Agent Training Framework for Multi-Agent MCTS

Implements specialized training loops for:
- HRM (Hierarchical Reasoning Model)
- TRM (Task Refinement Model)
- MCTS (Neural components)

Supports LoRA/Alora parameter-efficient fine-tuning.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import yaml

try:
    from transformers import (
        AutoModel,
        DebertaV2Tokenizer,
        get_linear_schedule_with_warmup,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, get_peft_model

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    epoch: int
    step: int
    loss: float
    accuracy: float
    learning_rate: float
    additional_metrics: dict[str, float]


class BaseAgentTrainer(ABC):
    """Abstract base class for agent trainers."""

    def __init__(self, config: dict[str, Any], device: str | None = None):
        """
        Initialize base trainer.

        Args:
            config: Training configuration
            device: Device to use (auto-detected if None)
        """
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.tokenizer = None

        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")
        self.training_history = []

        logger.info(f"Initialized {self.__class__.__name__} on device: {self.device}")

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build the model architecture."""
        pass

    @abstractmethod
    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """Compute training loss for a batch."""
        pass

    @abstractmethod
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate model on validation data."""
        pass

    def setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"]["weight_decay"]),
        )

        total_steps = self.config["training"]["epochs"] * 1000  # Approximate
        warmup_steps = int(total_steps * self.config["training"]["warmup_ratio"])

        self.scheduler = (
            get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
            if HAS_TRANSFORMERS
            else CosineAnnealingWarmRestarts(self.optimizer, T_0=100)
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Average epoch loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        accumulation_steps = self.config["training"]["gradient_accumulation_steps"]

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = self._move_to_device(batch)

            # Forward pass
            loss = self.compute_loss(batch)
            loss = loss / accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["gradient_clip_norm"])

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

            total_loss += loss.item() * accumulation_steps
            num_batches += 1

            # Logging
            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}, Loss: {loss.item() * accumulation_steps:.4f}"
                )

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _move_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move batch tensors to device."""
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                moved_batch[key] = [v.to(self.device) for v in value]
            else:
                moved_batch[key] = value
        return moved_batch

    def save_checkpoint(self, path: str, additional_info: dict | None = None) -> None:
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
            additional_info: Additional metadata to save
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "training_history": self.training_history,
            "config": self.config,
        }

        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["current_epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint["best_metric"]
        self.training_history = checkpoint.get("training_history", [])

        logger.info(f"Loaded checkpoint from {path}, epoch {self.current_epoch}")


class HRMTrainer(BaseAgentTrainer):
    """Trainer for Hierarchical Reasoning Model."""

    def __init__(self, config: dict[str, Any], device: str | None = None):
        super().__init__(config, device)
        self.hrm_config = config["agents"]["hrm"]
        self.build_model()
        self.setup_optimizer()

    def build_model(self) -> nn.Module:
        """Build HRM model with LoRA adapters."""
        if not HAS_TRANSFORMERS:
            logger.warning("Transformers not available, using simplified model")
            self.model = self._build_simple_hrm()
            return self.model

        # Load base model
        base_model = AutoModel.from_pretrained(self.hrm_config["model_name"])
        # Use explicit slow tokenizer for DeBERTa to avoid tiktoken conversion issues
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.hrm_config["model_name"])

        # Apply LoRA if available
        if HAS_PEFT:
            lora_config = LoraConfig(
                r=self.hrm_config["lora_rank"],
                lora_alpha=self.config["training"]["lora"]["alpha"],
                lora_dropout=self.config["training"]["lora"]["dropout"],
                target_modules=self.config["training"]["lora"]["target_modules"],
            )
            base_model = get_peft_model(base_model, lora_config)
            logger.info(f"Applied LoRA with rank {self.hrm_config['lora_rank']}")

        # Add decomposition head
        # Hub snapshots may store fp16 weights (e.g. deberta-v3-base safetensors) and
        # transformers >= 5 keeps that dtype by default; force fp32 so the backbone
        # matches the fp32 heads and CPU inference (no Half matmul support) works.
        self.model = HRMModel(
            base_model=base_model,
            hidden_size=self.hrm_config["hidden_size"],
            num_labels=self.hrm_config["num_labels"],
            max_depth=self.hrm_config["max_decomposition_depth"],
        ).to(device=self.device, dtype=torch.float32)

        return self.model

    def _build_simple_hrm(self) -> nn.Module:
        """Build simplified HRM model without transformers."""
        return SimpleHRMModel(
            vocab_size=30000,
            embedding_dim=256,
            hidden_size=self.hrm_config["hidden_size"],
            num_labels=self.hrm_config["num_labels"],
        ).to(self.device)

    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """
        Compute hierarchical decomposition loss.

        Args:
            batch: Batch of training data

        Returns:
            Loss tensor
        """
        # Tokenize input if needed
        if self.tokenizer and "input_ids" not in batch:
            inputs = self.tokenizer(
                batch["input_text"], padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
        else:
            input_ids = batch.get("input_ids")
            attention_mask = batch.get("attention_mask")

        labels = batch["labels"]

        # Forward pass
        if hasattr(self.model, "decomposition_head"):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
        else:
            logits = self.model(input_ids)

        # FIX #18: With CLS pooling, logits shape is [batch_size, num_labels]
        # Use first label from each sample (labels shape: [batch_size, label_length])
        if labels.dim() == 2:
            # Take first label from each sample
            labels = labels[:, 0]  # [batch_size]

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels, ignore_index=-100)

        # Add depth regularization
        if "depth" in batch:
            # FIX #19: Convert depth list to tensor before calling .float()
            target_depth = torch.tensor(batch["depth"], dtype=torch.float32).to(self.device)
            predicted_depth = self._predict_depth(logits)
            depth_loss = F.mse_loss(predicted_depth.float(), target_depth)
            loss = loss + 0.1 * depth_loss

        return loss

    def _predict_depth(self, logits: torch.Tensor) -> torch.Tensor:
        """Predict decomposition depth from logits."""
        # Count number of END tokens (label=2)
        predictions = logits.argmax(dim=-1)
        depth = (predictions == 2).sum(dim=-1)
        return depth

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Evaluate HRM on validation data.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        depth_errors = []

        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_to_device(batch)
                loss = self.compute_loss(batch)
                total_loss += loss.item()

                # Calculate accuracy
                if self.tokenizer and "input_ids" not in batch:
                    inputs = self.tokenizer(
                        batch["input_text"], padding=True, truncation=True, max_length=512, return_tensors="pt"
                    )
                    input_ids = inputs["input_ids"].to(self.device)
                else:
                    input_ids = batch.get("input_ids")

                if hasattr(self.model, "decomposition_head"):
                    outputs = self.model(input_ids=input_ids)
                    predictions = outputs["logits"].argmax(dim=-1)
                else:
                    predictions = self.model(input_ids).argmax(dim=-1)

                labels = batch["labels"]
                mask = labels != -100
                correct += ((predictions == labels) & mask).sum().item()
                total += mask.sum().item()

                # Depth error
                if "depth" in batch:
                    pred_depth = self._predict_depth(self.model(input_ids).get("logits", self.model(input_ids)))
                    depth_errors.extend((pred_depth - batch["depth"]).abs().cpu().tolist())

        metrics = {
            "loss": total_loss / len(dataloader),
            "accuracy": correct / total if total > 0 else 0.0,
            "avg_depth_error": np.mean(depth_errors) if depth_errors else 0.0,
        }

        self.model.train()
        return metrics


class TRMTrainer(BaseAgentTrainer):
    """Trainer for Task Refinement Model."""

    def __init__(self, config: dict[str, Any], device: str | None = None):
        super().__init__(config, device)
        self.trm_config = config["agents"]["trm"]
        # Ensure numeric values are floats (YAML may parse them as strings)
        self.trm_config["convergence_threshold"] = float(self.trm_config["convergence_threshold"])
        self.build_model()
        self.setup_optimizer()

    def build_model(self) -> nn.Module:
        """Build TRM model with refinement-specific heads."""
        if not HAS_TRANSFORMERS:
            logger.warning("Transformers not available, using simplified model")
            self.model = self._build_simple_trm()
            return self.model

        # Load base model (can share with HRM)
        base_model = AutoModel.from_pretrained(self.trm_config["model_name"])
        # Use explicit slow tokenizer for DeBERTa to avoid tiktoken conversion issues
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.trm_config["model_name"])

        # Apply LoRA
        if HAS_PEFT:
            lora_config = LoraConfig(
                r=self.trm_config["lora_rank"],
                lora_alpha=self.config["training"]["lora"]["alpha"],
                lora_dropout=self.config["training"]["lora"]["dropout"],
                target_modules=self.config["training"]["lora"]["target_modules"],
            )
            base_model = get_peft_model(base_model, lora_config)

        # Add refinement head
        # Same fp32 normalization as HRM: fp16 hub snapshots must not leak Half
        # params into a model that runs (and trains) in fp32, especially on CPU.
        self.model = TRMModel(
            base_model=base_model,
            hidden_size=self.trm_config["hidden_size"],
            max_iterations=self.trm_config["max_refinement_iterations"],
            convergence_threshold=self.trm_config["convergence_threshold"],
        ).to(device=self.device, dtype=torch.float32)

        return self.model

    def _build_simple_trm(self) -> nn.Module:
        """Build simplified TRM model."""
        return SimpleTRMModel(
            vocab_size=30000,
            embedding_dim=256,
            hidden_size=self.trm_config["hidden_size"],
            max_iterations=self.trm_config["max_refinement_iterations"],
        ).to(self.device)

    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """
        Compute task refinement loss.

        Args:
            batch: Batch of training data

        Returns:
            Loss tensor
        """
        # Tokenize
        if self.tokenizer and "input_ids" not in batch:
            inputs = self.tokenizer(
                batch["initial_task"], padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(self.device)
        else:
            input_ids = batch.get("input_ids")

        # FIX #20: Trim target to max_iterations to match model output
        target_scores = batch["improvement_scores"][:, : self.trm_config["max_refinement_iterations"]]

        # Forward pass
        if hasattr(self.model, "refinement_head"):
            outputs = self.model(input_ids=input_ids)
            predicted_scores = outputs["improvement_predictions"]
        else:
            predicted_scores = self.model(input_ids)

        # MSE loss for improvement scores
        loss = F.mse_loss(predicted_scores, target_scores)

        # Add convergence penalty
        # Penalize not reaching high scores quickly
        convergence_loss = self._compute_convergence_penalty(predicted_scores, target_scores)
        loss = loss + 0.1 * convergence_loss

        return loss

    def _compute_convergence_penalty(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute penalty for slow convergence."""
        # Higher penalty for later steps that don't reach target
        step_weights = torch.arange(1, predicted.shape[-1] + 1).float().to(self.device)
        weighted_diff = (target - predicted) * step_weights
        return F.relu(weighted_diff).mean()

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Evaluate TRM on validation data.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        avg_iterations = []
        convergence_rates = []

        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_to_device(batch)
                loss = self.compute_loss(batch)
                total_loss += loss.item()

                # Calculate average iterations to convergence
                target_scores = batch["improvement_scores"]
                converged_mask = target_scores >= self.trm_config["convergence_threshold"]

                # Find first convergence iteration for each sample
                # If no convergence, use max_iterations (sequence length)
                max_iterations = target_scores.size(-1)
                has_converged = converged_mask.any(dim=-1)
                # argmax returns 0 for non-converging, but we need first True index
                first_convergence = converged_mask.int().argmax(dim=-1)
                # Set non-converging samples to max_iterations
                iterations = torch.where(
                    has_converged, first_convergence, max_iterations * torch.ones_like(first_convergence)
                )
                avg_iterations.extend(iterations.cpu().tolist())

                # Convergence rate
                final_scores = target_scores[:, -1]
                converged = (final_scores >= self.trm_config["convergence_threshold"]).float()
                convergence_rates.append(converged.mean().item())

        metrics = {
            "loss": total_loss / len(dataloader),
            "avg_iterations": np.mean(avg_iterations),
            "convergence_rate": np.mean(convergence_rates),
        }

        self.model.train()
        return metrics


class MCTSTrainer(BaseAgentTrainer):
    """Trainer for MCTS neural components (value and policy networks)."""

    def __init__(self, config: dict[str, Any], device: str | None = None):
        super().__init__(config, device)
        self.mcts_config = config["agents"]["mcts"]
        # Ensure numeric values are floats/ints (YAML may parse them as strings)
        self.mcts_config["value_network"]["learning_rate"] = float(self.mcts_config["value_network"]["learning_rate"])
        self.mcts_config["policy_network"]["learning_rate"] = float(self.mcts_config["policy_network"]["learning_rate"])
        self.mcts_config["self_play"]["buffer_size"] = int(self.mcts_config["self_play"]["buffer_size"])
        self.mcts_config["self_play"]["games_per_iteration"] = int(self.mcts_config["self_play"]["games_per_iteration"])
        self.mcts_config["self_play"]["update_frequency"] = int(self.mcts_config["self_play"]["update_frequency"])
        self.mcts_config["simulations"] = int(self.mcts_config["simulations"])
        self.mcts_config["exploration_constant"] = float(self.mcts_config["exploration_constant"])
        self.mcts_config["discount_factor"] = float(self.mcts_config["discount_factor"])
        self.build_model()
        self.setup_optimizer()

        # Experience replay buffer
        self.replay_buffer = []
        self.buffer_size = self.mcts_config["self_play"]["buffer_size"]

    def build_model(self) -> nn.Module:
        """Build MCTS value and policy networks."""
        self.model = MCTSNeuralComponents(
            state_dim=256,  # Will be defined by actual state representation
            action_dim=100,  # Maximum number of actions
            value_hidden_layers=self.mcts_config["value_network"]["hidden_layers"],
            policy_hidden_layers=self.mcts_config["policy_network"]["hidden_layers"],
            value_lr=self.mcts_config["value_network"]["learning_rate"],
            policy_lr=self.mcts_config["policy_network"]["learning_rate"],
        ).to(self.device)

        return self.model

    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """
        Compute combined value and policy loss.

        Args:
            batch: Batch containing states, actions, rewards

        Returns:
            Combined loss tensor
        """
        states = batch["states"]
        target_values = batch["values"]
        target_policies = batch["policies"]

        # Value loss
        predicted_values = self.model.value_network(states)
        value_loss = F.mse_loss(predicted_values.squeeze(), target_values)

        # Policy loss (cross-entropy)
        predicted_policies = self.model.policy_network(states)
        policy_loss = F.cross_entropy(predicted_policies, target_policies)

        # Combined loss
        loss = value_loss + policy_loss

        return loss

    def generate_self_play_data(self, num_games: int = 100) -> list[dict[str, Any]]:
        """
        Generate training data through self-play.

        Args:
            num_games: Number of self-play games

        Returns:
            List of experience tuples
        """
        experiences = []

        for game_idx in range(num_games):
            # Simulate MCTS game
            game_experiences = self._play_game()
            experiences.extend(game_experiences)

            if game_idx % 10 == 0:
                logger.info(f"Self-play game {game_idx}/{num_games}")

        # Add to replay buffer
        self.replay_buffer.extend(experiences)
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer = self.replay_buffer[-self.buffer_size :]

        return experiences

    def _play_game(self) -> list[dict[str, Any]]:
        """Simulate a single MCTS self-play game."""
        experiences = []

        # Simplified game simulation
        state_dim = 256
        action_dim = 100

        for _step in range(20):  # Max 20 steps per game
            # Current state (random for simulation)
            state = torch.randn(state_dim)

            # Get MCTS policy (simplified)
            policy = F.softmax(torch.randn(action_dim), dim=0)
            action = policy.argmax().item()

            # Simulate reward
            reward = np.random.randn()

            # Value target (temporal difference)
            value = reward + self.mcts_config["discount_factor"] * np.random.randn()

            experiences.append({"state": state, "action": action, "policy": policy, "value": value, "reward": reward})

        return experiences

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Evaluate MCTS networks.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        value_errors = []
        policy_accuracies = []

        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_to_device(batch)

                states = batch["states"]
                target_values = batch["values"]
                target_policies = batch["policies"]

                # Value error
                predicted_values = self.model.value_network(states)
                value_mae = (predicted_values.squeeze() - target_values).abs().mean()
                value_errors.append(value_mae.item())

                # Policy accuracy
                predicted_policies = self.model.policy_network(states)
                pred_actions = predicted_policies.argmax(dim=-1)
                target_actions = target_policies.argmax(dim=-1)
                accuracy = (pred_actions == target_actions).float().mean()
                policy_accuracies.append(accuracy.item())

        metrics = {
            "value_mae": np.mean(value_errors),
            "policy_accuracy": np.mean(policy_accuracies),
            "buffer_size": len(self.replay_buffer),
        }

        self.model.train()
        return metrics


class AgentTrainingOrchestrator:
    """Orchestrate training across all agent types."""

    def __init__(self, config_path: str = "training/config.yaml"):
        """
        Initialize training orchestrator.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize trainers
        self.hrm_trainer = None
        self.trm_trainer = None
        self.mcts_trainer = None

        self.checkpoint_dir = Path("training/models/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized AgentTrainingOrchestrator on {self.device}")

    def initialize_trainers(self) -> None:
        """Initialize all agent trainers."""
        logger.info("Initializing agent trainers...")

        self.hrm_trainer = HRMTrainer(self.config, self.device)
        self.trm_trainer = TRMTrainer(self.config, self.device)
        self.mcts_trainer = MCTSTrainer(self.config, self.device)

        logger.info("All trainers initialized")

    def train_phase(
        self,
        phase_name: str,
        hrm_dataloader: DataLoader | None = None,
        trm_dataloader: DataLoader | None = None,
        val_dataloaders: dict[str, DataLoader] | None = None,
    ) -> dict[str, Any]:
        """
        Run a training phase.

        Args:
            phase_name: Name of the training phase
            hrm_dataloader: HRM training data
            trm_dataloader: TRM training data
            val_dataloaders: Validation data loaders

        Returns:
            Training results
        """
        logger.info(f"Starting training phase: {phase_name}")

        results = {"phase": phase_name, "hrm_metrics": [], "trm_metrics": [], "mcts_metrics": []}

        num_epochs = self.config["training"]["epochs"]

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Train HRM
            if self.hrm_trainer and hrm_dataloader:
                hrm_loss = self.hrm_trainer.train_epoch(hrm_dataloader)
                self.hrm_trainer.current_epoch = epoch
                results["hrm_metrics"].append({"epoch": epoch, "loss": hrm_loss})

            # Train TRM
            if self.trm_trainer and trm_dataloader:
                trm_loss = self.trm_trainer.train_epoch(trm_dataloader)
                self.trm_trainer.current_epoch = epoch
                results["trm_metrics"].append({"epoch": epoch, "loss": trm_loss})

            # Train MCTS (self-play)
            if self.mcts_trainer:
                self.mcts_trainer.generate_self_play_data(
                    num_games=self.config["agents"]["mcts"]["self_play"]["games_per_iteration"]
                )

            # Validation
            if val_dataloaders:
                val_results = self._run_validation(val_dataloaders)
                results[f"epoch_{epoch}_validation"] = val_results

            # Save checkpoints
            if (epoch + 1) % self.config["training"]["checkpointing"]["save_steps"] == 0:
                self._save_all_checkpoints(epoch)

        return results

    def _run_validation(self, dataloaders: dict[str, DataLoader]) -> dict[str, Any]:
        """Run validation for all agents."""
        val_results = {}

        if self.hrm_trainer and "hrm" in dataloaders:
            val_results["hrm"] = self.hrm_trainer.evaluate(dataloaders["hrm"])

        if self.trm_trainer and "trm" in dataloaders:
            val_results["trm"] = self.trm_trainer.evaluate(dataloaders["trm"])

        if self.mcts_trainer and "mcts" in dataloaders:
            val_results["mcts"] = self.mcts_trainer.evaluate(dataloaders["mcts"])

        return val_results

    def _save_all_checkpoints(self, epoch: int) -> None:
        """Save checkpoints for all agents."""
        if self.hrm_trainer:
            path = self.checkpoint_dir / f"hrm_epoch_{epoch}.pt"
            self.hrm_trainer.save_checkpoint(str(path))

        if self.trm_trainer:
            path = self.checkpoint_dir / f"trm_epoch_{epoch}.pt"
            self.trm_trainer.save_checkpoint(str(path))

        if self.mcts_trainer:
            path = self.checkpoint_dir / f"mcts_epoch_{epoch}.pt"
            self.mcts_trainer.save_checkpoint(str(path))

    def load_all_checkpoints(self, epoch: int) -> None:
        """Load checkpoints for all agents."""
        if self.hrm_trainer:
            path = self.checkpoint_dir / f"hrm_epoch_{epoch}.pt"
            if path.exists():
                self.hrm_trainer.load_checkpoint(str(path))

        if self.trm_trainer:
            path = self.checkpoint_dir / f"trm_epoch_{epoch}.pt"
            if path.exists():
                self.trm_trainer.load_checkpoint(str(path))

        if self.mcts_trainer:
            path = self.checkpoint_dir / f"mcts_epoch_{epoch}.pt"
            if path.exists():
                self.mcts_trainer.load_checkpoint(str(path))


# Model Architectures


class HRMModel(nn.Module):
    """Hierarchical Reasoning Model with decomposition head."""

    def __init__(self, base_model: nn.Module, hidden_size: int, num_labels: int, max_depth: int):
        super().__init__()
        self.base_model = base_model
        self.decomposition_head = nn.Linear(hidden_size, num_labels)
        self.depth_predictor = nn.Linear(hidden_size, max_depth)
        self.max_depth = max_depth

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        # Get sequence output
        if hasattr(outputs, "last_hidden_state"):
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = outputs[0]

        # FIX #18: Use CLS token for decomposition (pooling) to avoid label alignment issues
        # This matches the depth_predictor approach and simplifies training for demo mode
        cls_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]

        # Decomposition logits from CLS token
        logits = self.decomposition_head(cls_output)  # [batch_size, num_labels]

        # Depth prediction (from CLS token)
        depth_logits = self.depth_predictor(cls_output)

        return {"logits": logits, "depth_logits": depth_logits}


class SimpleHRMModel(nn.Module):
    """Simplified HRM model without transformer backbone."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_labels: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        logits = self.classifier(lstm_out)
        return logits


class TRMModel(nn.Module):
    """Task Refinement Model with improvement prediction."""

    def __init__(self, base_model: nn.Module, hidden_size: int, max_iterations: int, convergence_threshold: float):
        super().__init__()
        self.base_model = base_model
        self.refinement_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, max_iterations)
        )
        self.convergence_threshold = convergence_threshold

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.base_model(input_ids=input_ids)

        if hasattr(outputs, "last_hidden_state"):
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = outputs[0]

        # Predict improvement scores
        cls_output = sequence_output[:, 0, :]
        improvement_predictions = torch.sigmoid(self.refinement_head(cls_output))

        return {"improvement_predictions": improvement_predictions}


class SimpleTRMModel(nn.Module):
    """Simplified TRM model without transformer backbone."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, max_iterations: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.refinement_head = nn.Linear(hidden_size, max_iterations)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        _, (h_n, _) = self.lstm(embedded)
        scores = torch.sigmoid(self.refinement_head(h_n.squeeze(0)))
        return scores


class MCTSNeuralComponents(nn.Module):
    """Neural network components for MCTS."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        value_hidden_layers: list[int],
        policy_hidden_layers: list[int],
        value_lr: float,
        policy_lr: float,
    ):
        super().__init__()

        # Value network
        value_layers = []
        prev_dim = state_dim
        for hidden_dim in value_hidden_layers:
            value_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim
        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_network = nn.Sequential(*value_layers)

        # Policy network
        policy_layers = []
        prev_dim = state_dim
        for hidden_dim in policy_hidden_layers:
            policy_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim
        policy_layers.append(nn.Linear(prev_dim, action_dim))
        self.policy_network = nn.Sequential(*policy_layers)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.value_network(state)
        policy = F.softmax(self.policy_network(state), dim=-1)
        return value, policy


if __name__ == "__main__":
    # Test the agent training framework
    logging.basicConfig(level=logging.INFO)

    logger.info("Testing Agent Training Framework")

    # Load config
    config_path = "training/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Test HRM trainer
    hrm_trainer = HRMTrainer(config)
    logger.info(f"HRM model parameters: {sum(p.numel() for p in hrm_trainer.model.parameters())}")

    # Test TRM trainer
    trm_trainer = TRMTrainer(config)
    logger.info(f"TRM model parameters: {sum(p.numel() for p in trm_trainer.model.parameters())}")

    # Test MCTS trainer
    mcts_trainer = MCTSTrainer(config)
    logger.info(f"MCTS model parameters: {sum(p.numel() for p in mcts_trainer.model.parameters())}")

    # Test self-play
    experiences = mcts_trainer.generate_self_play_data(num_games=5)
    logger.info(f"Generated {len(experiences)} self-play experiences")

    # Test orchestrator
    orchestrator = AgentTrainingOrchestrator(config_path)
    orchestrator.initialize_trainers()

    logger.info("Agent Training Framework test complete")
