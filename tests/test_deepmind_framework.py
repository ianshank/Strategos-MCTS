"""
Comprehensive Test Suite for LangGraph Multi-Agent MCTS with DeepMind-Style Learning.

Tests all core components:
- HRM Agent
- TRM Agent
- Neural MCTS
- Policy-Value Networks
- Training Infrastructure
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch required")
nn = torch.nn

from src.agents.hrm_agent import HRMAgent, HRMLoss, create_hrm_agent
from src.agents.trm_agent import TRMAgent, create_trm_agent
from src.framework.mcts.neural_mcts import (
    GameState,
    NeuralMCTS,
    NeuralMCTSNode,
)
from src.models.policy_value_net import (
    AlphaZeroLoss,
    PolicyValueNetwork,
    create_policy_value_network,
)
from src.training.performance_monitor import PerformanceMonitor
from src.training.replay_buffer import (
    Experience,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from src.training.system_config import (
    HRMConfig,
    NeuralNetworkConfig,
    SystemConfig,
    TRMConfig,
    get_small_config,
)


# Fixtures
@pytest.fixture
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def small_config():
    """Get small configuration for fast tests."""
    return get_small_config()


@pytest.fixture
def hrm_config():
    """Get HRM configuration."""
    return HRMConfig(
        h_dim=64,  # Small for testing
        l_dim=32,
        num_h_layers=2,
        num_l_layers=2,
        max_outer_steps=5,
    )


@pytest.fixture
def trm_config():
    """Get TRM configuration."""
    return TRMConfig(
        latent_dim=64,  # Small for testing
        num_recursions=8,
        hidden_dim=128,
    )


@pytest.fixture
def neural_net_config():
    """Get neural network configuration."""
    return NeuralNetworkConfig(
        num_res_blocks=3,  # Small for testing
        num_channels=32,
        input_channels=3,
        action_size=9,  # 3x3 tic-tac-toe
    )


# Test HRM Agent
class TestHRMAgent:
    """Test suite for Hierarchical Reasoning Model."""

    def test_hrm_initialization(self, hrm_config, device):
        """Test HRM agent initialization."""
        agent = create_hrm_agent(hrm_config, device)

        assert isinstance(agent, HRMAgent)
        assert agent.config == hrm_config
        assert agent.get_parameter_count() > 0

    def test_hrm_forward_pass(self, hrm_config, device):
        """Test HRM forward pass."""
        agent = create_hrm_agent(hrm_config, device)
        agent.eval()

        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, hrm_config.h_dim).to(device)

        with torch.no_grad():
            output = agent(x, max_steps=3)

        assert output.final_state.shape == (batch_size, seq_len, hrm_config.h_dim)
        assert isinstance(output.halt_step, int)
        assert output.halt_step <= 3
        assert len(output.convergence_path) == output.halt_step

    @pytest.mark.asyncio
    async def test_hrm_decomposition(self, hrm_config, device):
        """Test HRM problem decomposition."""
        agent = create_hrm_agent(hrm_config, device)
        agent.eval()

        query = "Solve this problem"
        state = torch.randn(1, 4, hrm_config.h_dim).to(device)

        # Run async function
        subproblems = await agent.decompose_problem(query, state)

        assert isinstance(subproblems, list)
        assert all(hasattr(sp, "level") for sp in subproblems)
        assert all(hasattr(sp, "confidence") for sp in subproblems)

    def test_hrm_loss(self, hrm_config, device):
        """Test HRM loss computation."""
        agent = create_hrm_agent(hrm_config, device)
        loss_fn = HRMLoss()

        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, hrm_config.h_dim).to(device)

        output = agent(x, max_steps=3)

        # Create dummy predictions and targets
        predictions = torch.randn(batch_size, 10).to(device)
        targets = torch.randint(0, 10, (batch_size,)).to(device)
        task_loss_fn = nn.CrossEntropyLoss()

        total_loss, loss_dict = loss_fn(output, predictions, targets, task_loss_fn)

        assert isinstance(total_loss, torch.Tensor)
        assert "total" in loss_dict
        assert "task" in loss_dict
        assert "ponder" in loss_dict


# Test TRM Agent
class TestTRMAgent:
    """Test suite for Tiny Recursive Model."""

    def test_trm_initialization(self, trm_config, device):
        """Test TRM agent initialization."""
        agent = create_trm_agent(trm_config, output_dim=10, device=device)

        assert isinstance(agent, TRMAgent)
        assert agent.config == trm_config
        assert agent.get_parameter_count() > 0

    def test_trm_forward_pass(self, trm_config, device):
        """Test TRM forward pass."""
        agent = create_trm_agent(trm_config, output_dim=10, device=device)
        agent.eval()

        batch_size = 2
        x = torch.randn(batch_size, trm_config.latent_dim).to(device)

        with torch.no_grad():
            output = agent(x, num_recursions=5)

        assert output.final_prediction.shape == (batch_size, 10)
        assert len(output.intermediate_predictions) <= 5
        assert isinstance(output.converged, bool)
        assert output.convergence_step <= 5

    def test_trm_convergence(self, trm_config, device):
        """Test TRM convergence detection."""
        # Use very high threshold to force convergence despite random initialization
        trm_config.convergence_threshold = 100.0
        trm_config.min_recursions = 2

        agent = create_trm_agent(trm_config, output_dim=10, device=device)
        agent.eval()

        x = torch.randn(1, trm_config.latent_dim).to(device)

        with torch.no_grad():
            output = agent(x, num_recursions=10, check_convergence=True)

        # Should converge before max recursions
        assert output.convergence_step < 10 or output.converged

    @pytest.mark.asyncio
    async def test_trm_refine_solution(self, trm_config, device):
        """Test TRM solution refinement."""
        agent = create_trm_agent(trm_config, output_dim=10, device=device)
        agent.eval()

        initial = torch.randn(1, trm_config.latent_dim).to(device)

        refined, info = await agent.refine_solution(initial, num_recursions=5)

        assert refined.shape == (1, 10)
        assert "converged" in info
        assert "convergence_step" in info


# Test Neural MCTS
class TestNeuralMCTS:
    """Test suite for Neural-Guided MCTS."""

    class DummyGameState(GameState):
        """Dummy game state for testing."""

        def __init__(self, value=0):
            self.value = value
            self.terminal = False

        def get_legal_actions(self):
            return ["a", "b", "c"]

        def apply_action(self, action):
            new_state = TestNeuralMCTS.DummyGameState(self.value + 1)
            if self.value >= 2:
                new_state.terminal = True
            return new_state

        def is_terminal(self):
            return self.terminal

        def get_reward(self, player=1):
            return 1.0 if self.terminal else 0.0

        def to_tensor(self):
            return torch.randn(3, 3, 3)  # Dummy tensor

        def get_hash(self):
            return f"state_{self.value}"

    class DummyPolicyValueNet(nn.Module):
        """Dummy policy-value network for testing."""

        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(27, 16)

        def forward(self, x):
            batch_size = x.size(0)
            # Return dummy policy and value
            policy = torch.randn(batch_size, 3)
            value = torch.randn(batch_size, 1)
            return policy, value

    def test_mcts_node(self):
        """Test MCTS node creation and statistics."""
        state = self.DummyGameState()
        node = NeuralMCTSNode(state, prior=0.5)

        assert node.visit_count == 0
        assert node.value == 0.0
        assert not node.is_expanded

        # Update node
        node.update(1.0)
        assert node.visit_count == 1
        assert node.value == 1.0

    def test_mcts_expansion(self):
        """Test node expansion."""
        state = self.DummyGameState()
        node = NeuralMCTSNode(state)

        policy_probs = np.array([0.5, 0.3, 0.2])
        valid_actions = ["a", "b", "c"]

        node.expand(policy_probs, valid_actions)

        assert node.is_expanded
        assert len(node.children) == 3
        assert "a" in node.children
        assert node.children["a"].prior == 0.5

    def test_mcts_puct_selection(self):
        """Test PUCT child selection."""
        state = self.DummyGameState()
        node = NeuralMCTSNode(state)
        node.visit_count = 10

        # Add children with different statistics
        for i, action in enumerate(["a", "b", "c"]):
            child_state = state.apply_action(action)
            child = NeuralMCTSNode(child_state, parent=node, action=action, prior=0.33)
            child.visit_count = i + 1
            child.value_sum = (i + 1) * 0.5
            node.children[action] = child

        # Select best child
        action, child = node.select_child(c_puct=1.0)

        assert action in ["a", "b", "c"]
        assert child is not None

    @pytest.mark.asyncio
    async def test_neural_mcts_search(self, device):
        """Test MCTS search."""
        from src.training.system_config import MCTSConfig

        config = MCTSConfig(num_simulations=10, c_puct=1.0)

        net = self.DummyPolicyValueNet().to(device)
        mcts = NeuralMCTS(net, config, device)

        initial_state = self.DummyGameState()

        action_probs, root = await mcts.search(initial_state, num_simulations=10, temperature=1.0)

        assert isinstance(action_probs, dict)
        assert len(action_probs) > 0
        assert root.visit_count > 0


# Test Policy-Value Network
class TestPolicyValueNetwork:
    """Test suite for Policy-Value Networks."""

    def test_network_initialization(self, neural_net_config, device):
        """Test network initialization."""
        net = create_policy_value_network(neural_net_config, board_size=3, device=device)

        assert isinstance(net, PolicyValueNetwork)
        assert net.get_parameter_count() > 0

    def test_network_forward_pass(self, neural_net_config, device):
        """Test forward pass."""
        net = create_policy_value_network(neural_net_config, board_size=3, device=device)
        net.eval()

        batch_size = 2
        x = torch.randn(batch_size, neural_net_config.input_channels, 3, 3).to(device)

        with torch.no_grad():
            policy_logits, value = net(x)

        assert policy_logits.shape == (batch_size, neural_net_config.action_size)
        assert value.shape == (batch_size, 1)
        assert value.min() >= -1 and value.max() <= 1  # Tanh bounded

    def test_alphazero_loss(self, neural_net_config, device):
        """Test AlphaZero loss computation."""
        loss_fn = AlphaZeroLoss()

        batch_size = 4
        action_size = neural_net_config.action_size

        policy_logits = torch.randn(batch_size, action_size).to(device)
        value = torch.randn(batch_size, 1).to(device)

        # Create targets
        target_policy = torch.softmax(torch.randn(batch_size, action_size), dim=1).to(device)
        target_value = torch.randn(batch_size).to(device)

        loss, loss_dict = loss_fn(policy_logits, value, target_policy, target_value)

        assert isinstance(loss, torch.Tensor)
        assert "policy" in loss_dict
        assert "value" in loss_dict


# Test Training Infrastructure
class TestTrainingInfrastructure:
    """Test suite for training infrastructure."""

    def test_replay_buffer(self):
        """Test replay buffer."""
        buffer = ReplayBuffer(capacity=100)

        # Add experiences
        for _ in range(50):
            exp = Experience(
                state=torch.randn(3, 3, 3),
                policy=np.random.random(9),
                value=np.random.random(),
            )
            buffer.add(exp)

        assert len(buffer) == 50

        # Sample batch
        batch = buffer.sample(batch_size=10)
        assert len(batch) == 10

    def test_prioritized_replay_buffer(self):
        """Test prioritized replay buffer."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        # Add experiences
        for idx in range(50):
            exp = Experience(
                state=torch.randn(3, 3, 3),
                policy=np.random.random(9),
                value=np.random.random(),
            )
            buffer.add(exp, priority=float(idx))

        assert len(buffer) == 50

        # Sample batch
        experiences, indices, weights = buffer.sample(batch_size=10)
        assert len(experiences) == 10
        assert len(indices) == 10
        assert len(weights) == 10

    def test_performance_monitor(self):
        """Test performance monitoring."""
        monitor = PerformanceMonitor(window_size=10)

        # Log some timings
        for idx in range(20):
            monitor.log_timing("hrm_decomposition", float(idx))

        stats = monitor.get_stats("hrm_decomposition_time")

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

    def test_system_config(self):
        """Test system configuration."""
        config = SystemConfig()

        # Test serialization
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "hrm" in config_dict
        assert "trm" in config_dict

        # Test deserialization
        config2 = SystemConfig.from_dict(config_dict)
        assert config2.hrm.h_dim == config.hrm.h_dim
        assert config2.trm.latent_dim == config.trm.latent_dim


# Integration Tests
class TestIntegration:
    """Integration tests for complete pipeline."""

    def test_end_to_end_small_config(self, small_config, device):
        """Test end-to-end pipeline with small configuration."""
        # Create all components
        hrm_agent = create_hrm_agent(small_config.hrm, device)
        trm_agent = create_trm_agent(small_config.trm, device=device)
        policy_value_net = create_policy_value_network(small_config.neural_net, board_size=3, device=device)

        # Test forward passes
        x = torch.randn(1, 4, small_config.hrm.h_dim).to(device)

        with torch.no_grad():
            hrm_output = hrm_agent(x)
            assert hrm_output.final_state.shape[0] == 1

            trm_input = torch.randn(1, small_config.trm.latent_dim).to(device)
            trm_output = trm_agent(trm_input)
            assert trm_output.final_prediction is not None

            state_input = torch.randn(1, small_config.neural_net.input_channels, 3, 3).to(device)
            policy, value = policy_value_net(state_input)
            assert policy.shape[1] == small_config.neural_net.action_size


# Mark slow tests
@pytest.mark.slow
class TestSlowIntegration:
    """Slow integration tests (skipped in CI)."""

    def test_training_iteration(self, small_config, device):
        """Test single training iteration (slow)."""
        # This would test a full training iteration
        # Skipped in fast test runs
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
