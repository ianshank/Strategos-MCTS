"""
Extended unit tests for LLM-Guided MCTS training modules.

Covers uncovered code paths in:
- trainer.py: training loop, _compute_loss, _evaluate, _train_epoch, early stopping
- networks.py: CombinedNetwork, config validation, pooling modes (cls, max)
- metrics.py: compute_kl_divergence, compute_policy_entropy, EvaluationMetrics, edge cases
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None  # type: ignore

try:
    import torch
    import torch.nn as nn  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not _TORCH_AVAILABLE or not _NUMPY_AVAILABLE,
        reason="PyTorch and numpy required",
    ),
]


# ===========================================================================
# networks.py - Config validation
# ===========================================================================


class TestCodeEncoderConfigValidation:
    def test_hidden_dim_not_divisible_by_heads(self):
        from src.framework.mcts.llm_guided.training.networks import CodeEncoderConfig

        config = CodeEncoderConfig(hidden_dim=100, num_heads=8)
        with pytest.raises(ValueError, match="hidden_dim must be divisible"):
            config.validate()

    def test_num_layers_zero(self):
        from src.framework.mcts.llm_guided.training.networks import CodeEncoderConfig

        config = CodeEncoderConfig(num_layers=0)
        with pytest.raises(ValueError, match="num_layers must be >= 1"):
            config.validate()

    def test_dropout_out_of_range(self):
        from src.framework.mcts.llm_guided.training.networks import CodeEncoderConfig

        config = CodeEncoderConfig(dropout=1.5)
        with pytest.raises(ValueError, match="dropout must be in"):
            config.validate()

    def test_invalid_pooling(self):
        from src.framework.mcts.llm_guided.training.networks import CodeEncoderConfig

        config = CodeEncoderConfig(pooling="invalid")
        with pytest.raises(ValueError, match="pooling must be"):
            config.validate()


class TestPolicyNetworkConfigValidation:
    def test_invalid_max_actions(self):
        from src.framework.mcts.llm_guided.training.networks import PolicyNetworkConfig

        config = PolicyNetworkConfig(max_actions=0)
        with pytest.raises(ValueError, match="max_actions must be >= 1"):
            config.validate()

    def test_invalid_hidden_dim(self):
        from src.framework.mcts.llm_guided.training.networks import PolicyNetworkConfig

        config = PolicyNetworkConfig(hidden_dim=0)
        with pytest.raises(ValueError, match="hidden_dim must be >= 1"):
            config.validate()

    def test_invalid_num_layers(self):
        from src.framework.mcts.llm_guided.training.networks import PolicyNetworkConfig

        config = PolicyNetworkConfig(num_layers=0)
        with pytest.raises(ValueError, match="num_layers must be >= 1"):
            config.validate()

    def test_invalid_dropout(self):
        from src.framework.mcts.llm_guided.training.networks import PolicyNetworkConfig

        config = PolicyNetworkConfig(dropout=-0.1)
        with pytest.raises(ValueError, match="dropout must be in"):
            config.validate()

    def test_invalid_temperature(self):
        from src.framework.mcts.llm_guided.training.networks import PolicyNetworkConfig

        config = PolicyNetworkConfig(temperature=0)
        with pytest.raises(ValueError, match="temperature must be > 0"):
            config.validate()


class TestValueNetworkConfigValidation:
    def test_invalid_hidden_dim(self):
        from src.framework.mcts.llm_guided.training.networks import ValueNetworkConfig

        config = ValueNetworkConfig(hidden_dim=0)
        with pytest.raises(ValueError, match="hidden_dim must be >= 1"):
            config.validate()

    def test_invalid_num_layers(self):
        from src.framework.mcts.llm_guided.training.networks import ValueNetworkConfig

        config = ValueNetworkConfig(num_layers=0)
        with pytest.raises(ValueError, match="num_layers must be >= 1"):
            config.validate()

    def test_invalid_dropout(self):
        from src.framework.mcts.llm_guided.training.networks import ValueNetworkConfig

        config = ValueNetworkConfig(dropout=2.0)
        with pytest.raises(ValueError, match="dropout must be in"):
            config.validate()

    def test_invalid_value_range(self):
        from src.framework.mcts.llm_guided.training.networks import ValueNetworkConfig

        config = ValueNetworkConfig(min_value=1.0, max_value=-1.0)
        with pytest.raises(ValueError, match="min_value must be less than max_value"):
            config.validate()


# ===========================================================================
# networks.py - Pooling modes
# ===========================================================================


class TestCodeEncoderPooling:
    def _make_encoder(self, pooling: str):
        from src.framework.mcts.llm_guided.training.networks import CodeEncoder, CodeEncoderConfig

        config = CodeEncoderConfig(
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            max_seq_length=64,
            pooling=pooling,
        )
        return CodeEncoder(config)

    def test_mean_pooling_with_mask(self):
        encoder = self._make_encoder("mean")
        input_ids = torch.randint(0, 1000, (2, 16))
        mask = torch.ones(2, 16)
        mask[0, 8:] = 0  # mask second half for first batch
        output = encoder(input_ids, mask)
        assert output.shape == (2, 32)

    def test_mean_pooling_without_mask(self):
        encoder = self._make_encoder("mean")
        input_ids = torch.randint(0, 1000, (2, 16))
        output = encoder(input_ids, attention_mask=None)
        assert output.shape == (2, 32)

    def test_cls_pooling(self):
        encoder = self._make_encoder("cls")
        input_ids = torch.randint(0, 1000, (2, 16))
        mask = torch.ones(2, 16)
        output = encoder(input_ids, mask)
        assert output.shape == (2, 32)

    def test_max_pooling_with_mask(self):
        encoder = self._make_encoder("max")
        input_ids = torch.randint(0, 1000, (2, 16))
        mask = torch.ones(2, 16)
        mask[0, 8:] = 0
        output = encoder(input_ids, mask)
        assert output.shape == (2, 32)

    def test_max_pooling_without_mask(self):
        encoder = self._make_encoder("max")
        input_ids = torch.randint(0, 1000, (2, 16))
        output = encoder(input_ids, attention_mask=None)
        assert output.shape == (2, 32)


# ===========================================================================
# networks.py - CombinedNetwork
# ===========================================================================


class TestCombinedNetwork:
    @pytest.fixture
    def combined_network(self):
        from src.framework.mcts.llm_guided.training.networks import (
            CodeEncoderConfig,
            CombinedNetwork,
        )

        encoder_config = CodeEncoderConfig(
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
        )
        return CombinedNetwork(
            encoder_config=encoder_config,
            max_actions=5,
            hidden_dim=32,
        )

    def test_forward(self, combined_network):
        batch_size, seq_len = 2, 16
        code_tokens = torch.randint(0, 1000, (batch_size, seq_len))
        code_mask = torch.ones(batch_size, seq_len)
        problem_tokens = torch.randint(0, 1000, (batch_size, seq_len))
        problem_mask = torch.ones(batch_size, seq_len)
        action_mask = torch.ones(batch_size, 5)

        log_probs, values = combined_network(code_tokens, code_mask, problem_tokens, problem_mask, action_mask)

        assert log_probs.shape == (batch_size, 5)
        assert values.shape == (batch_size,)
        assert (log_probs <= 0).all()
        probs = torch.exp(log_probs)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    def test_forward_without_action_mask(self, combined_network):
        batch_size, seq_len = 2, 16
        code_tokens = torch.randint(0, 1000, (batch_size, seq_len))
        code_mask = torch.ones(batch_size, seq_len)
        problem_tokens = torch.randint(0, 1000, (batch_size, seq_len))
        problem_mask = torch.ones(batch_size, seq_len)

        log_probs, values = combined_network(code_tokens, code_mask, problem_tokens, problem_mask)
        assert log_probs.shape == (batch_size, 5)
        assert values.shape == (batch_size,)

    def test_value_bounds(self, combined_network):
        batch_size, seq_len = 4, 16
        code_tokens = torch.randint(0, 1000, (batch_size, seq_len))
        code_mask = torch.ones(batch_size, seq_len)
        problem_tokens = torch.randint(0, 1000, (batch_size, seq_len))
        problem_mask = torch.ones(batch_size, seq_len)

        _, values = combined_network(code_tokens, code_mask, problem_tokens, problem_mask)
        assert (values >= -1.0).all()
        assert (values <= 1.0).all()

    def test_custom_value_bounds(self):
        from src.framework.mcts.llm_guided.training.networks import (
            CodeEncoderConfig,
            CombinedNetwork,
        )

        encoder_config = CodeEncoderConfig(hidden_dim=32, num_layers=1, num_heads=4)
        net = CombinedNetwork(
            encoder_config=encoder_config,
            max_actions=3,
            hidden_dim=32,
            min_value=0.0,
            max_value=1.0,
        )
        code_tokens = torch.randint(0, 1000, (2, 8))
        mask = torch.ones(2, 8)
        _, values = net(code_tokens, mask, code_tokens, mask)
        assert (values >= 0.0).all()
        assert (values <= 1.0).all()

    def test_default_config(self):
        from src.framework.mcts.llm_guided.training.networks import CombinedNetwork

        net = CombinedNetwork()
        assert net._max_actions == 10
        assert net._hidden_dim == 256


# ===========================================================================
# networks.py - PositionalEncoding
# ===========================================================================


class TestPositionalEncoding:
    def test_forward(self):
        from src.framework.mcts.llm_guided.training.networks import PositionalEncoding

        pe = PositionalEncoding(d_model=32, max_len=100, dropout=0.0)
        x = torch.randn(2, 10, 32)
        out = pe(x)
        assert out.shape == (2, 10, 32)
        # With 0 dropout, positional encoding should just add pe values
        assert not torch.allclose(out, x)


# ===========================================================================
# networks.py - Policy network with action masking
# ===========================================================================


class TestPolicyNetworkMasking:
    def test_action_mask_zeros_out_actions(self):
        from src.framework.mcts.llm_guided.training.networks import (
            CodeEncoderConfig,
            PolicyNetwork,
            PolicyNetworkConfig,
        )

        encoder_config = CodeEncoderConfig(hidden_dim=32, num_layers=1, num_heads=4)
        config = PolicyNetworkConfig(encoder_config=encoder_config, max_actions=5, hidden_dim=32)
        net = PolicyNetwork(config)

        code = torch.randint(0, 1000, (1, 8))
        mask = torch.ones(1, 8)
        action_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])

        log_probs = net(code, mask, code, mask, action_mask)
        probs = torch.exp(log_probs)

        # Masked actions should have ~0 probability
        assert probs[0, 2].item() < 1e-6
        assert probs[0, 3].item() < 1e-6
        assert probs[0, 4].item() < 1e-6


# ===========================================================================
# metrics.py - EvaluationMetrics
# ===========================================================================


class TestEvaluationMetrics:
    def test_defaults(self):
        from src.framework.mcts.llm_guided.training.metrics import EvaluationMetrics

        metrics = EvaluationMetrics()
        assert metrics.policy_accuracy == 0.0
        assert metrics.value_mse == 0.0
        assert metrics.total_samples == 0
        assert metrics.accuracy_by_depth == {}
        assert metrics.mse_by_depth == {}
        assert metrics.samples_by_depth == {}

    def test_to_dict(self):
        from src.framework.mcts.llm_guided.training.metrics import EvaluationMetrics

        metrics = EvaluationMetrics(
            policy_accuracy=0.9,
            value_mse=0.05,
            accuracy_by_depth={0: 0.95, 1: 0.85},
            predicted_value_mean=0.5,
            predicted_value_std=0.1,
            target_value_mean=0.55,
            target_value_std=0.12,
            total_samples=100,
            samples_by_depth={0: 60, 1: 40},
        )
        d = metrics.to_dict()
        assert d["policy_accuracy"] == 0.9
        assert d["value_mse"] == 0.05
        assert d["accuracy_by_depth"] == {0: 0.95, 1: 0.85}
        assert d["total_samples"] == 100


# ===========================================================================
# metrics.py - TrainingMetrics
# ===========================================================================


class TestTrainingMetricsExtended:
    def test_to_dict_contains_all_fields(self):
        from src.framework.mcts.llm_guided.training.metrics import TrainingMetrics

        m = TrainingMetrics(
            policy_loss=0.1,
            value_loss=0.2,
            total_loss=0.3,
            policy_accuracy=0.8,
            policy_top3_accuracy=0.95,
            policy_kl_divergence=0.05,
            policy_entropy=1.2,
            value_mse=0.04,
            value_mae=0.15,
            value_correlation=0.9,
            learning_rate=0.001,
            gradient_norm=0.5,
            num_samples=100,
            epoch=3,
            step=500,
        )
        d = m.to_dict()
        assert len(d) == 15
        assert d["policy_loss"] == 0.1
        assert d["step"] == 500


# ===========================================================================
# metrics.py - MetricsAccumulator edge cases
# ===========================================================================


class TestMetricsAccumulatorExtended:
    def test_compute_empty(self):
        from src.framework.mcts.llm_guided.training.metrics import MetricsAccumulator

        acc = MetricsAccumulator()
        metrics = acc.compute()
        assert metrics.num_samples == 0
        assert metrics.policy_loss == 0.0

    def test_compute_with_torch_tensors(self):
        from src.framework.mcts.llm_guided.training.metrics import MetricsAccumulator

        acc = MetricsAccumulator()
        acc.update(
            policy_loss=0.5,
            value_loss=0.3,
            total_loss=0.8,
            policy_correct=3,
            policy_top3_correct=4,
            value_predictions=torch.tensor([0.5, 0.6]),
            value_targets=torch.tensor([0.6, 0.7]),
            batch_size=5,
        )
        metrics = acc.compute(learning_rate=0.001, epoch=1, step=10)
        assert metrics.num_samples == 5
        assert metrics.learning_rate == 0.001

    def test_correlation_zero_std(self):
        """Correlation is 0 when predictions have zero std."""
        from src.framework.mcts.llm_guided.training.metrics import MetricsAccumulator

        acc = MetricsAccumulator()
        acc.update(
            policy_loss=0.1,
            value_loss=0.1,
            total_loss=0.2,
            policy_correct=2,
            policy_top3_correct=2,
            value_predictions=np.array([0.5, 0.5, 0.5]),
            value_targets=np.array([0.1, 0.5, 0.9]),
            batch_size=3,
        )
        metrics = acc.compute()
        assert metrics.value_correlation == 0.0

    def test_reset(self):
        from src.framework.mcts.llm_guided.training.metrics import MetricsAccumulator

        acc = MetricsAccumulator()
        acc.update(
            policy_loss=1.0,
            value_loss=1.0,
            total_loss=2.0,
            policy_correct=5,
            policy_top3_correct=5,
            value_predictions=np.array([1.0]),
            value_targets=np.array([1.0]),
            batch_size=5,
        )
        acc.reset()
        metrics = acc.compute()
        assert metrics.num_samples == 0


# ===========================================================================
# metrics.py - compute_kl_divergence
# ===========================================================================


class TestComputeKLDivergence:
    def test_kl_divergence_same_distribution(self):
        from src.framework.mcts.llm_guided.training.metrics import compute_kl_divergence

        probs = torch.tensor([[0.5, 0.3, 0.2]])
        log_probs = torch.log(probs)
        kl = compute_kl_divergence(log_probs, probs)
        assert kl == pytest.approx(0.0, abs=1e-5)

    def test_kl_divergence_different_distributions(self):
        from src.framework.mcts.llm_guided.training.metrics import compute_kl_divergence

        log_probs = torch.log(torch.tensor([[0.7, 0.2, 0.1]]))
        target_probs = torch.tensor([[0.3, 0.5, 0.2]])
        kl = compute_kl_divergence(log_probs, target_probs)
        assert kl > 0

    def test_kl_divergence_with_action_mask(self):
        from src.framework.mcts.llm_guided.training.metrics import compute_kl_divergence

        log_probs = torch.log(torch.tensor([[0.6, 0.4, 0.0001]]))
        target_probs = torch.tensor([[0.5, 0.5, 0.0]])
        action_mask = torch.tensor([[1.0, 1.0, 0.0]])
        kl = compute_kl_divergence(log_probs, target_probs, action_mask)
        assert isinstance(kl, float)

    def test_kl_divergence_without_mask(self):
        from src.framework.mcts.llm_guided.training.metrics import compute_kl_divergence

        log_probs = torch.log(torch.tensor([[0.5, 0.5]]))
        target_probs = torch.tensor([[0.5, 0.5]])
        kl = compute_kl_divergence(log_probs, target_probs, action_mask=None)
        assert kl == pytest.approx(0.0, abs=1e-5)


# ===========================================================================
# metrics.py - compute_policy_entropy
# ===========================================================================


class TestComputePolicyEntropy:
    def test_entropy_uniform(self):
        from src.framework.mcts.llm_guided.training.metrics import compute_policy_entropy

        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        log_probs = torch.log(probs)
        entropy = compute_policy_entropy(log_probs)
        import math

        expected = math.log(4)
        assert entropy == pytest.approx(expected, abs=1e-4)

    def test_entropy_deterministic(self):
        from src.framework.mcts.llm_guided.training.metrics import compute_policy_entropy

        probs = torch.tensor([[1.0, 0.0, 0.0]])
        log_probs = torch.log(probs.clamp(min=1e-8))
        entropy = compute_policy_entropy(log_probs)
        assert entropy == pytest.approx(0.0, abs=1e-4)

    def test_entropy_with_action_mask(self):
        from src.framework.mcts.llm_guided.training.metrics import compute_policy_entropy

        probs = torch.tensor([[0.5, 0.5, 0.001]])
        log_probs = torch.log(probs)
        action_mask = torch.tensor([[1.0, 1.0, 0.0]])
        entropy = compute_policy_entropy(log_probs, action_mask)
        assert isinstance(entropy, float)
        assert entropy >= 0.0

    def test_entropy_without_mask(self):
        from src.framework.mcts.llm_guided.training.metrics import compute_policy_entropy

        probs = torch.tensor([[0.5, 0.5]])
        log_probs = torch.log(probs)
        entropy = compute_policy_entropy(log_probs, action_mask=None)
        assert entropy > 0


# ===========================================================================
# metrics.py - compute_policy_accuracy edge cases
# ===========================================================================


class TestComputePolicyAccuracyExtended:
    def test_top3_accuracy(self):
        from src.framework.mcts.llm_guided.training.metrics import compute_policy_accuracy

        log_probs = torch.log(torch.tensor([[0.1, 0.3, 0.4, 0.2]]))
        targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        # Target action is 0, top-3 predictions are [2, 1, 3], action 0 not in top-3
        accuracy = compute_policy_accuracy(log_probs, targets, k=3)
        assert accuracy == pytest.approx(0.0)

    def test_accuracy_with_mask_all_invalid(self):
        from src.framework.mcts.llm_guided.training.metrics import compute_policy_accuracy

        log_probs = torch.log(torch.tensor([[0.5, 0.5]]))
        targets = torch.tensor([[1.0, 0.0]])
        action_mask = torch.tensor([[0.0, 0.0]])  # no valid actions
        accuracy = compute_policy_accuracy(log_probs, targets, action_mask=action_mask, k=1)
        assert accuracy == 0.0

    def test_accuracy_with_valid_mask(self):
        from src.framework.mcts.llm_guided.training.metrics import compute_policy_accuracy

        log_probs = torch.log(torch.tensor([[0.7, 0.3]]))
        targets = torch.tensor([[1.0, 0.0]])
        action_mask = torch.tensor([[1.0, 1.0]])
        accuracy = compute_policy_accuracy(log_probs, targets, action_mask=action_mask, k=1)
        assert accuracy == pytest.approx(1.0)


# ===========================================================================
# trainer.py - DistillationTrainerConfig validation edge cases
# ===========================================================================


class TestDistillationTrainerConfigExtended:
    def test_validate_negative_max_grad_norm(self):
        from src.framework.mcts.llm_guided.training.trainer import DistillationTrainerConfig

        config = DistillationTrainerConfig(max_grad_norm=-1.0)
        with pytest.raises(ValueError, match="max_grad_norm"):
            config.validate()

    def test_validate_negative_value_loss_weight(self):
        from src.framework.mcts.llm_guided.training.trainer import DistillationTrainerConfig

        config = DistillationTrainerConfig(value_loss_weight=-0.5)
        with pytest.raises(ValueError, match="loss weights"):
            config.validate()

    def test_validate_valid_metrics(self):
        """All valid metric names should pass."""
        from src.framework.mcts.llm_guided.training.trainer import DistillationTrainerConfig

        for metric in [
            "policy_loss",
            "value_loss",
            "total_loss",
            "policy_accuracy",
            "policy_top3_accuracy",
            "policy_kl_divergence",
            "policy_entropy",
            "value_mse",
            "value_mae",
            "value_correlation",
        ]:
            config = DistillationTrainerConfig(early_stopping_metric=metric)
            config.validate()  # Should not raise


# ===========================================================================
# trainer.py - Training loop with mock batch
# ===========================================================================


@dataclass
class _FakeBatch:
    """Minimal batch for testing the training loop."""

    code_tokens: Any
    code_attention_mask: Any
    problem_tokens: Any
    problem_attention_mask: Any
    llm_policy: Any
    mcts_policy: Any
    action_mask: Any
    llm_value: Any
    outcome: Any
    q_value: Any
    episode_ids: list
    depths: Any
    visits: Any

    def to(self, device):
        return self


class _FakeDataset:
    """Minimal dataset for DataLoader."""

    def __init__(self, size: int = 4):
        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return idx


def _make_fake_batch(batch_size=2, max_actions=5):
    return _FakeBatch(
        code_tokens=torch.randint(0, 1000, (batch_size, 8)),
        code_attention_mask=torch.ones(batch_size, 8),
        problem_tokens=torch.randint(0, 1000, (batch_size, 8)),
        problem_attention_mask=torch.ones(batch_size, 8),
        llm_policy=torch.softmax(torch.randn(batch_size, max_actions), dim=-1),
        mcts_policy=torch.softmax(torch.randn(batch_size, max_actions), dim=-1),
        action_mask=torch.ones(batch_size, max_actions),
        llm_value=torch.rand(batch_size),
        outcome=torch.rand(batch_size),
        q_value=torch.rand(batch_size),
        episode_ids=[f"ep_{i}" for i in range(batch_size)],
        depths=torch.zeros(batch_size),
        visits=torch.ones(batch_size) * 10,
    )


class _FakeLoader:
    """Fake DataLoader that yields a fixed batch."""

    def __init__(self, batch, num_batches=2):
        self._batch = batch
        self._num_batches = num_batches
        self.dataset = _FakeDataset(num_batches * 2)

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        for _ in range(self._num_batches):
            yield self._batch


class TestDistillationTrainerTrainLoop:
    def _make_small_networks(self):
        from src.framework.mcts.llm_guided.training.networks import (
            CodeEncoderConfig,
            PolicyNetwork,
            PolicyNetworkConfig,
            ValueNetwork,
            ValueNetworkConfig,
        )

        encoder_config = CodeEncoderConfig(hidden_dim=32, num_layers=1, num_heads=4, max_seq_length=64)
        policy = PolicyNetwork(PolicyNetworkConfig(encoder_config=encoder_config, max_actions=5, hidden_dim=32))
        value = ValueNetwork(ValueNetworkConfig(encoder_config=encoder_config, hidden_dim=32))
        return policy, value

    def test_train_one_epoch(self, tmp_path):
        from src.framework.mcts.llm_guided.training.trainer import (
            DistillationTrainer,
            DistillationTrainerConfig,
        )

        policy, value = self._make_small_networks()
        config = DistillationTrainerConfig(
            num_epochs=1,
            device="cpu",
            checkpoint_dir=str(tmp_path),
            log_every_steps=1,
            warmup_steps=0,
        )
        trainer = DistillationTrainer(
            policy_network=policy,
            value_network=value,
            config=config,
        )

        batch = _make_fake_batch()
        loader = _FakeLoader(batch, num_batches=2)

        metrics = trainer.train(loader)

        assert metrics.num_samples > 0
        assert metrics.total_loss > 0
        assert trainer._global_step == 2

    def test_train_with_validation(self, tmp_path):
        from src.framework.mcts.llm_guided.training.trainer import (
            DistillationTrainer,
            DistillationTrainerConfig,
        )

        policy, value = self._make_small_networks()
        config = DistillationTrainerConfig(
            num_epochs=1,
            device="cpu",
            checkpoint_dir=str(tmp_path),
            warmup_steps=0,
            eval_every_epochs=1,
        )
        trainer = DistillationTrainer(
            policy_network=policy,
            value_network=value,
            config=config,
        )

        batch = _make_fake_batch()
        train_loader = _FakeLoader(batch, num_batches=2)
        val_loader = _FakeLoader(batch, num_batches=1)

        metrics = trainer.train(train_loader, val_loader=val_loader)
        assert metrics.num_samples > 0

    def test_train_early_stopping(self, tmp_path):
        from src.framework.mcts.llm_guided.training.trainer import (
            DistillationTrainer,
            DistillationTrainerConfig,
        )

        policy, value = self._make_small_networks()
        config = DistillationTrainerConfig(
            num_epochs=100,
            device="cpu",
            checkpoint_dir=str(tmp_path),
            warmup_steps=0,
            early_stopping_patience=2,
            early_stopping_metric="total_loss",
        )
        trainer = DistillationTrainer(
            policy_network=policy,
            value_network=value,
            config=config,
        )
        # Set best_metric very low so all epochs are worse -> triggers early stopping
        trainer._best_metric = -999.0

        batch = _make_fake_batch()
        loader = _FakeLoader(batch, num_batches=1)

        trainer.train(loader)

        # Should have stopped after patience (2 epochs) + however many to detect
        assert trainer._current_epoch < 99

    def test_train_callbacks_called(self, tmp_path):
        from src.framework.mcts.llm_guided.training.trainer import (
            DistillationTrainer,
            DistillationTrainerConfig,
            TrainingCallback,
        )

        policy, value = self._make_small_networks()
        config = DistillationTrainerConfig(
            num_epochs=1,
            device="cpu",
            checkpoint_dir=str(tmp_path),
            warmup_steps=0,
        )
        mock_cb = MagicMock(spec=TrainingCallback)
        trainer = DistillationTrainer(
            policy_network=policy,
            value_network=value,
            config=config,
            callbacks=[mock_cb],
        )

        batch = _make_fake_batch()
        loader = _FakeLoader(batch, num_batches=2)
        trainer.train(loader)

        mock_cb.on_epoch_start.assert_called()
        mock_cb.on_epoch_end.assert_called()
        mock_cb.on_batch_end.assert_called()
        mock_cb.on_training_end.assert_called()

    def test_train_policy_only(self, tmp_path):
        """Train with only policy network (no value network)."""
        from src.framework.mcts.llm_guided.training.trainer import (
            DistillationTrainer,
            DistillationTrainerConfig,
        )

        policy, _ = self._make_small_networks()
        config = DistillationTrainerConfig(
            num_epochs=1,
            device="cpu",
            checkpoint_dir=str(tmp_path),
            warmup_steps=0,
        )
        trainer = DistillationTrainer(policy_network=policy, config=config)

        batch = _make_fake_batch()
        loader = _FakeLoader(batch, num_batches=1)
        metrics = trainer.train(loader)
        assert metrics.num_samples > 0

    def test_train_value_only(self, tmp_path):
        """Train with only value network (no policy network)."""
        from src.framework.mcts.llm_guided.training.trainer import (
            DistillationTrainer,
            DistillationTrainerConfig,
        )

        _, value = self._make_small_networks()
        config = DistillationTrainerConfig(
            num_epochs=1,
            device="cpu",
            checkpoint_dir=str(tmp_path),
            warmup_steps=0,
        )
        trainer = DistillationTrainer(value_network=value, config=config)

        batch = _make_fake_batch()
        loader = _FakeLoader(batch, num_batches=1)
        metrics = trainer.train(loader)
        assert metrics.num_samples > 0

    def test_compute_loss_use_llm_policy(self, tmp_path):
        """Test _compute_loss with use_mcts_policy=False."""
        from src.framework.mcts.llm_guided.training.trainer import (
            DistillationTrainer,
            DistillationTrainerConfig,
        )

        policy, value = self._make_small_networks()
        config = DistillationTrainerConfig(
            device="cpu",
            checkpoint_dir=str(tmp_path),
            use_mcts_policy=False,
            use_outcome_value=False,
        )
        trainer = DistillationTrainer(
            policy_network=policy,
            value_network=value,
            config=config,
        )

        batch = _make_fake_batch()
        loss, batch_metrics = trainer._compute_loss(batch)
        assert loss.item() >= 0
        assert batch_metrics["policy_loss"] >= 0


# ===========================================================================
# trainer.py - _evaluate method
# ===========================================================================


class TestDistillationTrainerEvaluate:
    def test_evaluate(self, tmp_path):
        from src.framework.mcts.llm_guided.training.networks import (
            CodeEncoderConfig,
            PolicyNetwork,
            PolicyNetworkConfig,
            ValueNetwork,
            ValueNetworkConfig,
        )
        from src.framework.mcts.llm_guided.training.trainer import (
            DistillationTrainer,
            DistillationTrainerConfig,
        )

        encoder_config = CodeEncoderConfig(hidden_dim=32, num_layers=1, num_heads=4, max_seq_length=64)
        policy = PolicyNetwork(PolicyNetworkConfig(encoder_config=encoder_config, max_actions=5, hidden_dim=32))
        value = ValueNetwork(ValueNetworkConfig(encoder_config=encoder_config, hidden_dim=32))
        config = DistillationTrainerConfig(
            device="cpu",
            checkpoint_dir=str(tmp_path),
        )
        trainer = DistillationTrainer(
            policy_network=policy,
            value_network=value,
            config=config,
        )

        batch = _make_fake_batch()
        val_loader = _FakeLoader(batch, num_batches=2)
        metrics = trainer._evaluate(val_loader)

        assert metrics.num_samples > 0
        assert metrics.total_loss >= 0


# ===========================================================================
# trainer.py - create_trainer factory
# ===========================================================================


class TestCreateTrainerExtended:
    def test_create_with_networks(self):
        from src.framework.mcts.llm_guided.training.networks import (
            CodeEncoderConfig,
            PolicyNetwork,
            PolicyNetworkConfig,
            ValueNetwork,
            ValueNetworkConfig,
        )
        from src.framework.mcts.llm_guided.training.trainer import create_trainer

        encoder_config = CodeEncoderConfig(hidden_dim=32, num_layers=1, num_heads=4)
        policy = PolicyNetwork(PolicyNetworkConfig(encoder_config=encoder_config, max_actions=5, hidden_dim=32))
        value = ValueNetwork(ValueNetworkConfig(encoder_config=encoder_config, hidden_dim=32))

        trainer = create_trainer(
            policy_network=policy,
            value_network=value,
            learning_rate=0.01,
            num_epochs=5,
            device="cpu",
        )
        assert trainer._config.learning_rate == 0.01
        assert trainer._policy_network is not None
        assert trainer._value_network is not None
