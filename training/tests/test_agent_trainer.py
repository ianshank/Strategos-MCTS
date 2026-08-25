"""Unit tests for agent training framework."""

from pathlib import Path
import sys

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.agent_trainer import (
    AgentTrainingOrchestrator,
    HRMTrainer,
    MCTSNeuralComponents,
    MCTSTrainer,
    SimpleHRMModel,
    SimpleTRMModel,
    TRMTrainer,
)


@pytest.fixture
def training_config():
    """Create training configuration."""
    return {
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "epochs": 2,
            "warmup_ratio": 0.1,
            "gradient_accumulation_steps": 1,
            "gradient_clip_norm": 1.0,
            "fp16": False,
            "lora": {"rank": 8, "alpha": 16, "dropout": 0.1, "target_modules": ["query", "key"]},
        },
        "agents": {
            "hrm": {
                "model_name": "bert-base-uncased",
                "max_decomposition_depth": 5,
                "lora_rank": 8,
                "hidden_size": 256,
                "num_labels": 3,
            },
            "trm": {
                "model_name": "bert-base-uncased",
                "max_refinement_iterations": 3,
                "convergence_threshold": 0.95,
                "lora_rank": 8,
                "hidden_size": 256,
            },
            "mcts": {
                "simulations": 50,
                "exploration_constant": 1.414,
                "discount_factor": 0.99,
                "value_network": {"hidden_layers": [64, 32], "learning_rate": 1e-3},
                "policy_network": {"hidden_layers": [64, 32], "learning_rate": 1e-3},
                "self_play": {"games_per_iteration": 10, "buffer_size": 1000, "update_frequency": 5},
            },
        },
    }


class TestHRMTrainer:
    """Tests for HRM trainer."""

    def test_trainer_initialization(self, training_config):
        """Test HRM trainer initializes correctly."""
        trainer = HRMTrainer(training_config)

        assert trainer is not None
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.device in ["cpu", "cuda"]

    def test_simple_hrm_model(self, training_config):
        """Test simple HRM model architecture."""
        model = SimpleHRMModel(vocab_size=100, embedding_dim=32, hidden_size=64, num_labels=3)

        # Test forward pass
        input_ids = torch.randint(0, 100, (2, 10))
        output = model(input_ids)

        assert output.shape == (2, 10, 3)

    def test_compute_loss(self, training_config):
        """Test loss computation."""
        trainer = HRMTrainer(training_config)

        # Create mock batch
        batch = {"input_text": ["Test task 1", "Test task 2"], "labels": torch.randint(0, 3, (2, 10))}

        loss = trainer.compute_loss(batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_checkpoint_save_load(self, training_config, tmp_path):
        """Test checkpoint saving and loading."""
        trainer = HRMTrainer(training_config)

        # Save checkpoint
        checkpoint_path = tmp_path / "hrm_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

        # Load checkpoint
        new_trainer = HRMTrainer(training_config)
        new_trainer.load_checkpoint(str(checkpoint_path))

        assert new_trainer.current_epoch == trainer.current_epoch
        assert new_trainer.global_step == trainer.global_step


class TestTRMTrainer:
    """Tests for TRM trainer."""

    def test_trainer_initialization(self, training_config):
        """Test TRM trainer initializes correctly."""
        trainer = TRMTrainer(training_config)

        assert trainer is not None
        assert trainer.model is not None

    def test_simple_trm_model(self, training_config):
        """Test simple TRM model architecture."""
        model = SimpleTRMModel(vocab_size=100, embedding_dim=32, hidden_size=64, max_iterations=3)

        input_ids = torch.randint(0, 100, (2, 10))
        output = model(input_ids)

        assert output.shape == (2, 3)
        assert (output >= 0).all() and (output <= 1).all()  # Sigmoid output

    def test_convergence_penalty(self, training_config):
        """Test convergence penalty computation."""
        trainer = TRMTrainer(training_config)

        predicted = torch.tensor([[0.5, 0.7, 0.8]])
        target = torch.tensor([[0.6, 0.8, 0.95]])

        penalty = trainer._compute_convergence_penalty(predicted, target)

        assert isinstance(penalty, torch.Tensor)
        assert penalty.item() >= 0


class TestMCTSTrainer:
    """Tests for MCTS trainer."""

    def test_trainer_initialization(self, training_config):
        """Test MCTS trainer initializes correctly."""
        trainer = MCTSTrainer(training_config)

        assert trainer is not None
        assert trainer.model is not None
        assert isinstance(trainer.replay_buffer, list)

    def test_mcts_neural_components(self):
        """Test MCTS neural network components."""
        model = MCTSNeuralComponents(
            state_dim=64,
            action_dim=10,
            value_hidden_layers=[32, 16],
            policy_hidden_layers=[32],
            value_lr=1e-3,
            policy_lr=1e-3,
        )

        state = torch.randn(4, 64)
        value, policy = model(state)

        assert value.shape == (4, 1)
        assert policy.shape == (4, 10)
        assert torch.allclose(policy.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_self_play_generation(self, training_config):
        """Test self-play data generation."""
        trainer = MCTSTrainer(training_config)

        experiences = trainer.generate_self_play_data(num_games=5)

        assert len(experiences) > 0
        assert "state" in experiences[0]
        assert "action" in experiences[0]
        assert "policy" in experiences[0]
        assert "value" in experiences[0]

    def test_replay_buffer_size(self, training_config):
        """Test replay buffer respects max size."""
        trainer = MCTSTrainer(training_config)

        # Generate more data than buffer size
        for _ in range(20):
            trainer.generate_self_play_data(num_games=10)

        assert len(trainer.replay_buffer) <= trainer.buffer_size


class TestAgentTrainingOrchestrator:
    """Tests for training orchestrator."""

    @pytest.fixture
    def temp_config_file(self, training_config, tmp_path):
        """Create temporary config file."""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(training_config, f)
        return str(config_path)

    def test_orchestrator_initialization(self, temp_config_file):
        """Test orchestrator initializes correctly."""
        orchestrator = AgentTrainingOrchestrator(temp_config_file)

        assert orchestrator is not None
        assert orchestrator.device in ["cpu", "cuda"]

    def test_trainer_initialization(self, temp_config_file):
        """Test all trainers are initialized."""
        orchestrator = AgentTrainingOrchestrator(temp_config_file)
        orchestrator.initialize_trainers()

        assert orchestrator.hrm_trainer is not None
        assert orchestrator.trm_trainer is not None
        assert orchestrator.mcts_trainer is not None

    def test_checkpoint_directory_creation(self, temp_config_file):
        """Test checkpoint directory is created."""
        orchestrator = AgentTrainingOrchestrator(temp_config_file)

        assert orchestrator.checkpoint_dir.exists()


class TestModelArchitectures:
    """Tests for model architectures."""

    def test_hrm_model_parameters(self):
        """Test HRM model has trainable parameters."""
        model = SimpleHRMModel(100, 32, 64, 3)

        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert num_params > 0
        assert trainable_params == num_params  # All should be trainable

    def test_trm_model_parameters(self):
        """Test TRM model has trainable parameters."""
        model = SimpleTRMModel(100, 32, 64, 3)

        num_params = sum(p.numel() for p in model.parameters())

        assert num_params > 0

    def test_mcts_components_separation(self):
        """Test MCTS value and policy networks are separate."""
        model = MCTSNeuralComponents(64, 10, [32], [32], 1e-3, 1e-3)

        value_params = list(model.value_network.parameters())
        policy_params = list(model.policy_network.parameters())

        assert len(value_params) > 0
        assert len(policy_params) > 0

        # Check they don't share parameters
        for vp in value_params:
            for pp in policy_params:
                assert vp is not pp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
