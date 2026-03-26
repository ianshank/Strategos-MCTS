"""
Tests for experience replay buffer module.

Tests uniform replay buffer, prioritized replay buffer,
augmented replay buffer, and collation utilities.
"""

import numpy as np
import pytest
import torch

from src.training.replay_buffer import (
    AugmentedReplayBuffer,
    BoardGameAugmentation,
    Experience,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    collate_experiences,
)


def _make_experience(value: float = 0.5, state_dim: int = 4, policy_size: int = 8) -> Experience:
    """Create a test experience with given parameters."""
    return Experience(
        state=torch.randn(state_dim),
        policy=np.random.dirichlet(np.ones(policy_size)),
        value=value,
        metadata={"test": True},
    )


@pytest.mark.unit
class TestReplayBuffer:
    """Tests for uniform sampling ReplayBuffer."""

    def test_init(self):
        """Test buffer initializes with correct capacity."""
        capacity = 100
        buf = ReplayBuffer(capacity=capacity)
        assert buf.capacity == capacity
        assert len(buf) == 0

    def test_add_single(self):
        """Test adding a single experience."""
        buf = ReplayBuffer(capacity=10)
        exp = _make_experience()
        buf.add(exp)
        assert len(buf) == 1

    def test_add_batch(self):
        """Test adding multiple experiences at once."""
        buf = ReplayBuffer(capacity=100)
        batch = [_make_experience(value=i / 10) for i in range(5)]
        buf.add_batch(batch)
        assert len(buf) == 5

    def test_circular_buffer_evicts_oldest(self):
        """Test that oldest experiences are evicted when buffer is full."""
        capacity = 3
        buf = ReplayBuffer(capacity=capacity)

        for i in range(5):
            buf.add(_make_experience(value=float(i)))

        assert len(buf) == capacity

    def test_sample_returns_correct_size(self):
        """Test sampling returns requested batch size."""
        buf = ReplayBuffer(capacity=100)
        for _ in range(20):
            buf.add(_make_experience())

        sample = buf.sample(batch_size=5)
        assert len(sample) == 5

    def test_sample_when_buffer_smaller_than_batch(self):
        """Test sampling when buffer has fewer items than requested."""
        buf = ReplayBuffer(capacity=100)
        for _ in range(3):
            buf.add(_make_experience())

        sample = buf.sample(batch_size=10)
        assert len(sample) == 3

    def test_sample_returns_experience_objects(self):
        """Test sampled items are Experience instances."""
        buf = ReplayBuffer(capacity=100)
        for _ in range(10):
            buf.add(_make_experience())

        sample = buf.sample(batch_size=3)
        for exp in sample:
            assert isinstance(exp, Experience)
            assert isinstance(exp.state, torch.Tensor)
            assert isinstance(exp.policy, np.ndarray)

    def test_is_ready(self):
        """Test is_ready check."""
        buf = ReplayBuffer(capacity=100)
        assert not buf.is_ready(min_size=5)

        for _ in range(5):
            buf.add(_make_experience())

        assert buf.is_ready(min_size=5)
        assert not buf.is_ready(min_size=10)

    def test_clear(self):
        """Test clearing the buffer."""
        buf = ReplayBuffer(capacity=100)
        for _ in range(10):
            buf.add(_make_experience())

        assert len(buf) == 10
        buf.clear()
        assert len(buf) == 0


@pytest.mark.unit
class TestPrioritizedReplayBuffer:
    """Tests for PrioritizedReplayBuffer."""

    def test_init(self):
        """Test PER buffer initializes correctly."""
        capacity = 100
        buf = PrioritizedReplayBuffer(capacity=capacity, alpha=0.6, beta_start=0.4)
        assert buf.capacity == capacity
        assert buf.alpha == 0.6
        assert buf.beta_start == 0.4
        assert len(buf) == 0

    def test_add_with_default_priority(self):
        """Test adding experience with auto-priority."""
        buf = PrioritizedReplayBuffer(capacity=100)
        buf.add(_make_experience())
        assert len(buf) == 1

    def test_add_with_explicit_priority(self):
        """Test adding experience with explicit priority."""
        buf = PrioritizedReplayBuffer(capacity=100)
        buf.add(_make_experience(), priority=5.0)
        assert len(buf) == 1
        # Priority should be stored as priority^alpha
        assert buf.priorities[0] > 0

    def test_add_batch(self):
        """Test adding batch of experiences."""
        buf = PrioritizedReplayBuffer(capacity=100)
        experiences = [_make_experience() for _ in range(5)]
        buf.add_batch(experiences)
        assert len(buf) == 5

    def test_add_batch_with_priorities(self):
        """Test adding batch with explicit priorities."""
        buf = PrioritizedReplayBuffer(capacity=100)
        experiences = [_make_experience() for _ in range(3)]
        priorities = [1.0, 2.0, 3.0]
        buf.add_batch(experiences, priorities=priorities)
        assert len(buf) == 3

    def test_circular_eviction(self):
        """Test buffer wraps around when full."""
        capacity = 5
        buf = PrioritizedReplayBuffer(capacity=capacity)
        for i in range(10):
            buf.add(_make_experience(value=float(i)))

        assert len(buf) == capacity

    def test_sample_returns_tuple(self):
        """Test sample returns (experiences, indices, weights)."""
        buf = PrioritizedReplayBuffer(capacity=100)
        for _ in range(20):
            buf.add(_make_experience())

        experiences, indices, weights = buf.sample(batch_size=5)
        assert len(experiences) == 5
        assert len(indices) == 5
        assert len(weights) == 5

    def test_sample_weights_normalized(self):
        """Test importance sampling weights are normalized to max=1."""
        buf = PrioritizedReplayBuffer(capacity=100)
        for i in range(20):
            buf.add(_make_experience(), priority=float(i + 1))

        _, _, weights = buf.sample(batch_size=5)
        assert np.max(weights) == pytest.approx(1.0)
        assert np.all(weights > 0)
        assert np.all(weights <= 1.0)

    def test_update_priorities(self):
        """Test updating priorities for sampled experiences."""
        buf = PrioritizedReplayBuffer(capacity=100)
        for _ in range(10):
            buf.add(_make_experience())

        _, indices, _ = buf.sample(batch_size=3)
        new_priorities = np.array([10.0, 20.0, 30.0])
        buf.update_priorities(indices, new_priorities)

        # Priorities should be updated
        for idx, new_p in zip(indices, new_priorities):
            expected = (new_p + 1e-6) ** buf.alpha
            assert buf.priorities[idx] == pytest.approx(expected, rel=1e-5)

    def test_beta_annealing(self):
        """Test beta anneals from beta_start toward 1.0."""
        buf = PrioritizedReplayBuffer(capacity=100, beta_start=0.4, beta_frames=100)
        beta_initial = buf._get_beta()
        assert beta_initial == pytest.approx(0.4, abs=0.01)

        # Simulate frames
        buf.frame = 50
        beta_mid = buf._get_beta()
        assert beta_mid > beta_initial

        buf.frame = 200
        beta_final = buf._get_beta()
        assert beta_final == 1.0

    def test_is_ready(self):
        """Test is_ready check for PER buffer."""
        buf = PrioritizedReplayBuffer(capacity=100)
        assert not buf.is_ready(min_size=5)

        for _ in range(5):
            buf.add(_make_experience())
        assert buf.is_ready(min_size=5)


@pytest.mark.unit
class TestAugmentedReplayBuffer:
    """Tests for AugmentedReplayBuffer."""

    def test_sample_without_augmentation(self):
        """Test sampling without augmentation returns original experiences."""
        buf = AugmentedReplayBuffer(capacity=100)
        for _ in range(10):
            buf.add(_make_experience())

        sample = buf.sample(batch_size=3, apply_augmentation=False)
        assert len(sample) == 3

    def test_sample_with_augmentation_fn(self):
        """Test sampling applies augmentation function."""
        called = {"count": 0}

        def augment(state, policy):
            called["count"] += 1
            return state * 2, policy * 0.5

        buf = AugmentedReplayBuffer(capacity=100, augmentation_fn=augment)
        for _ in range(10):
            buf.add(_make_experience())

        sample = buf.sample(batch_size=3, apply_augmentation=True)
        assert len(sample) == 3
        assert called["count"] == 3

    def test_sample_no_augment_fn_returns_original(self):
        """Test sampling when no augmentation fn set returns originals."""
        buf = AugmentedReplayBuffer(capacity=100, augmentation_fn=None)
        for _ in range(10):
            buf.add(_make_experience())

        sample = buf.sample(batch_size=3, apply_augmentation=True)
        assert len(sample) == 3


@pytest.mark.unit
class TestBoardGameAugmentation:
    """Tests for board game augmentation utilities."""

    @pytest.fixture
    def board_state(self):
        """Create a test board state [channels, height, width]."""
        return torch.randn(2, 8, 8)

    @pytest.fixture
    def board_policy(self):
        """Create a test policy for 8x8 board with pass action."""
        board_size = 8
        policy = np.random.dirichlet(np.ones(board_size * board_size + 1))
        return policy

    def test_rotate_90_state_shape(self, board_state, board_policy):
        """Test 90-degree rotation preserves shape."""
        rotated_state, rotated_policy = BoardGameAugmentation.rotate_90(
            board_state, board_policy, board_size=8
        )
        assert rotated_state.shape == board_state.shape
        assert len(rotated_policy) == len(board_policy)

    def test_flip_horizontal_state_shape(self, board_state, board_policy):
        """Test horizontal flip preserves shape."""
        flipped_state, flipped_policy = BoardGameAugmentation.flip_horizontal(
            board_state, board_policy, board_size=8
        )
        assert flipped_state.shape == board_state.shape
        assert len(flipped_policy) == len(board_policy)

    def test_random_symmetry_preserves_shape(self, board_state, board_policy):
        """Test random symmetry always preserves shape."""
        for _ in range(10):
            aug_state, aug_policy = BoardGameAugmentation.random_symmetry(
                board_state, board_policy, board_size=8
            )
            assert aug_state.shape == board_state.shape

    def test_rotate_90_non_board_policy_unchanged(self, board_state):
        """Test rotation with non-board-sized policy returns unchanged."""
        policy = np.array([0.5, 0.3, 0.2])
        _, rotated_policy = BoardGameAugmentation.rotate_90(board_state, policy, board_size=8)
        np.testing.assert_array_equal(rotated_policy, policy)


@pytest.mark.unit
class TestCollateExperiences:
    """Tests for experience collation utility."""

    def test_collate_basic(self):
        """Test basic collation of experiences into tensors."""
        policy_size = 8
        experiences = [_make_experience(value=float(i), policy_size=policy_size) for i in range(4)]

        states, policies, values = collate_experiences(experiences)

        assert states.shape == (4, 4)
        assert policies.shape == (4, policy_size)
        assert values.shape == (4,)

    def test_collate_values_correct(self):
        """Test collated values match input."""
        experiences = [_make_experience(value=v) for v in [0.0, 0.5, 1.0]]
        _, _, values = collate_experiences(experiences)
        assert values[0].item() == pytest.approx(0.0)
        assert values[1].item() == pytest.approx(0.5)
        assert values[2].item() == pytest.approx(1.0)

    def test_collate_pads_variable_policies(self):
        """Test collation pads policies to max size."""
        exp1 = Experience(
            state=torch.randn(4),
            policy=np.array([0.5, 0.5]),
            value=0.5,
        )
        exp2 = Experience(
            state=torch.randn(4),
            policy=np.array([0.3, 0.3, 0.4]),
            value=0.7,
        )

        _, policies, _ = collate_experiences([exp1, exp2])
        assert policies.shape == (2, 3)  # Padded to max size=3
        assert policies[0, 2].item() == pytest.approx(0.0)  # Padded with zero
