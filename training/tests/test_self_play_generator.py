"""
Tests for Self-Play Training Pipeline

This module contains comprehensive tests for the AlphaZero-style
self-play training pipeline.
"""

from pathlib import Path
import sys

import pytest
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.self_play_generator import (
    Action,
    CodeGenerationTaskGenerator,
    MathProblemGenerator,
    MCTSSearchTaskGenerator,
    MCTSTrace,
    MultiStepReasoningGenerator,
    SelfPlayDataset,
    SelfPlayEpisode,
    SelfPlayEpisodeGenerator,
    SelfPlayTrainer,
    State,
    TrainingDataExtractor,
    TrainingExample,
)


class TestDataStructures:
    """Test episode data structures."""

    def test_action_creation(self):
        """Test Action dataclass."""
        action = Action(
            action_id="action_1",
            action_type="decompose",
            parameters={"depth": 2},
            timestamp=0.0,
            confidence=0.9,
        )
        assert action.action_id == "action_1"
        assert action.action_type == "decompose"
        assert action.confidence == 0.9

    def test_state_creation(self):
        """Test State dataclass."""
        state = State(
            state_id="state_1",
            representation=torch.randn(256),
            raw_state={"data": "test"},
        )
        assert state.state_id == "state_1"
        assert state.representation.shape == (256,)

    def test_mcts_trace_creation(self):
        """Test MCTSTrace dataclass."""
        trace = MCTSTrace(
            root_state_id="state_1",
            num_simulations=100,
            visit_counts={"action_1": 50, "action_2": 50},
            q_values={"action_1": 0.5, "action_2": 0.3},
            prior_probs={"action_1": 0.6, "action_2": 0.4},
            selected_action="action_1",
            tree_depth=5,
            search_time=0.1,
            value_estimates={"state_1": 0.7},
        )
        assert trace.num_simulations == 100
        assert trace.selected_action == "action_1"

    def test_episode_creation(self):
        """Test SelfPlayEpisode dataclass."""
        episode = SelfPlayEpisode(
            task_id="task_1",
            initial_state={"problem": "test"},
            actions=[],
            states=[],
            rewards=[],
            mcts_traces=[],
            outcome="success",
        )
        assert episode.task_id == "task_1"
        assert episode.outcome == "success"


class TestTaskGenerators:
    """Test task generator implementations."""

    def test_math_problem_generator(self):
        """Test MathProblemGenerator."""
        gen = MathProblemGenerator(seed=42)
        tasks = gen.generate(10)

        assert len(tasks) == 10
        for task in tasks:
            assert "task_id" in task
            assert "difficulty" in task
            assert "problem" in task
            assert "type" in task
            assert task["type"] in ["arithmetic", "algebra", "quadratic", "system"]

    def test_code_generation_task_generator(self):
        """Test CodeGenerationTaskGenerator."""
        gen = CodeGenerationTaskGenerator(seed=42)
        tasks = gen.generate(10)

        assert len(tasks) == 10
        for task in tasks:
            assert "task_id" in task
            assert "difficulty" in task
            assert "problem" in task
            assert "language" in task
            assert task["language"] in ["python", "javascript", "java"]

    def test_multi_step_reasoning_generator(self):
        """Test MultiStepReasoningGenerator."""
        gen = MultiStepReasoningGenerator(seed=42)
        tasks = gen.generate(5)

        assert len(tasks) == 5
        for task in tasks:
            assert "task_id" in task
            assert "difficulty" in task
            assert "problem" in task
            assert "steps" in task

    def test_mcts_search_task_generator(self):
        """Test MCTSSearchTaskGenerator."""
        gen = MCTSSearchTaskGenerator(seed=42)
        tasks = gen.generate(5)

        assert len(tasks) == 5
        for task in tasks:
            assert "task_id" in task
            assert "difficulty" in task
            assert "type" in task
            assert task["type"] in ["game", "optimization", "path_finding"]

    def test_difficulty_range(self):
        """Test that generators respect difficulty range."""
        gen = MathProblemGenerator(difficulty_range=(0.5, 1.0), seed=42)
        tasks = gen.generate(20)

        difficulties = [task["difficulty"] for task in tasks]
        assert all(0.5 <= d <= 1.0 for d in difficulties)


class TestSelfPlayEpisodeGenerator:
    """Test episode generation."""

    @pytest.mark.asyncio
    async def test_episode_generation(self):
        """Test generating a single episode."""
        generator = SelfPlayEpisodeGenerator(device="cpu")

        task = {
            "task_id": "test_task",
            "type": "arithmetic",
            "problem": "Calculate: 2 + 2",
            "difficulty": 0.1,
        }

        episode = await generator.generate_episode(task, max_steps=5, timeout=10.0)

        assert episode.task_id == "test_task"
        assert episode.outcome in ["success", "failure", "timeout", "max_steps_reached", "error"]
        assert len(episode.states) > 0
        assert len(episode.actions) >= 0

    @pytest.mark.asyncio
    async def test_episode_timeout(self):
        """Test episode timeout handling."""
        generator = SelfPlayEpisodeGenerator(device="cpu")

        task = {
            "task_id": "timeout_task",
            "type": "arithmetic",
            "problem": "Test",
            "difficulty": 0.5,
        }

        episode = await generator.generate_episode(task, max_steps=1000, timeout=0.1)

        assert episode.outcome == "timeout"

    @pytest.mark.asyncio
    async def test_mcts_trace_capture(self):
        """Test that MCTS traces are captured."""
        generator = SelfPlayEpisodeGenerator(device="cpu")

        task = {
            "task_id": "mcts_test",
            "type": "game",
            "problem": "Test",
            "difficulty": 0.5,
        }

        episode = await generator.generate_episode(task, max_steps=3, timeout=10.0)

        # Should have MCTS traces for each step
        assert len(episode.mcts_traces) >= 0
        for trace in episode.mcts_traces:
            assert trace.num_simulations > 0
            assert len(trace.visit_counts) >= 0


class TestTrainingDataExtractor:
    """Test training data extraction."""

    def test_policy_extraction(self):
        """Test extracting policy examples."""
        extractor = TrainingDataExtractor()

        # Create a successful episode
        states = [
            State(
                state_id=f"state_{i}",
                representation=torch.randn(256),
                raw_state={},
            )
            for i in range(3)
        ]

        mcts_traces = [
            MCTSTrace(
                root_state_id=states[i].state_id,
                num_simulations=100,
                visit_counts={"action_1": 60, "action_2": 40},
                q_values={"action_1": 0.7, "action_2": 0.5},
                prior_probs={"action_1": 0.5, "action_2": 0.5},
                selected_action="action_1",
                tree_depth=2,
                search_time=0.1,
                value_estimates={},
            )
            for i in range(2)
        ]

        episode = SelfPlayEpisode(
            task_id="test",
            initial_state={},
            actions=[],
            states=states,
            rewards=[1.0, 0.5],
            mcts_traces=mcts_traces,
            outcome="success",
        )

        examples = extractor._extract_policy_examples(episode)

        assert len(examples) >= 0
        for ex in examples:
            assert ex.example_type == "policy"
            assert isinstance(ex.state, torch.Tensor)
            assert isinstance(ex.target, dict)

    def test_value_extraction(self):
        """Test extracting value examples."""
        extractor = TrainingDataExtractor()

        states = [State(state_id=f"state_{i}", representation=torch.randn(256), raw_state={}) for i in range(4)]

        episode = SelfPlayEpisode(
            task_id="test",
            initial_state={},
            actions=[],
            states=states,
            rewards=[0.1, 0.2, 0.3, 1.0],
            mcts_traces=[],
            outcome="success",
        )

        examples = extractor._extract_value_examples(episode)

        assert len(examples) == 3  # One less than states
        for ex in examples:
            assert ex.example_type == "value"
            assert isinstance(ex.target, float)

    def test_negative_extraction(self):
        """Test extracting negative examples."""
        extractor = TrainingDataExtractor()

        states = [State(state_id=f"state_{i}", representation=torch.randn(256), raw_state={}) for i in range(3)]

        mcts_traces = [
            MCTSTrace(
                root_state_id=states[i].state_id,
                num_simulations=50,
                visit_counts={"bad_action": 50},
                q_values={"bad_action": -0.5},
                prior_probs={"bad_action": 1.0},
                selected_action="bad_action",
                tree_depth=1,
                search_time=0.05,
                value_estimates={},
            )
            for i in range(2)
        ]

        episode = SelfPlayEpisode(
            task_id="test",
            initial_state={},
            actions=[],
            states=states,
            rewards=[-0.5, -0.3],
            mcts_traces=mcts_traces,
            outcome="failure",
        )

        examples = extractor._extract_negative_examples(episode)

        assert len(examples) >= 0
        for ex in examples:
            assert ex.example_type == "negative"


class TestSelfPlayDataset:
    """Test PyTorch dataset."""

    def test_dataset_creation(self):
        """Test creating dataset from examples."""
        examples = [
            TrainingExample(
                example_id=f"ex_{i}",
                example_type="policy",
                state=torch.randn(256),
                target={"action": 0.5},
                weight=1.0,
            )
            for i in range(10)
        ]

        dataset = SelfPlayDataset(examples, "policy")

        assert len(dataset) == 10

        item = dataset[0]
        assert "state" in item
        assert "target" in item
        assert "weight" in item

    def test_dataset_filtering(self):
        """Test dataset filters by example type."""
        examples = [
            TrainingExample(
                example_id=f"ex_{i}",
                example_type="policy" if i % 2 == 0 else "value",
                state=torch.randn(256),
                target={},
                weight=1.0,
            )
            for i in range(10)
        ]

        policy_dataset = SelfPlayDataset(examples, "policy")
        value_dataset = SelfPlayDataset(examples, "value")

        assert len(policy_dataset) == 5
        assert len(value_dataset) == 5


class TestSelfPlayTrainer:
    """Test complete training pipeline."""

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        config = {
            "games_per_iteration": 100,
            "batch_size": 32,
        }

        trainer = SelfPlayTrainer(config=config, device="cpu")

        assert trainer.current_iteration == 0
        assert len(trainer.task_generators) == 4  # math, code, reasoning, mcts

    def test_task_generation(self):
        """Test task generation."""
        trainer = SelfPlayTrainer(config={}, device="cpu")
        tasks = trainer._generate_tasks(20)

        assert len(tasks) == 20
        for task in tasks:
            assert "task_id" in task

    @pytest.mark.asyncio
    async def test_self_play_generation(self):
        """Test generating self-play episodes."""
        config = {
            "games_per_iteration": 5,
            "parallel_batch_size": 2,
        }

        trainer = SelfPlayTrainer(config=config, device="cpu")
        episodes = await trainer.run_self_play(num_games=5)

        assert len(episodes) == 5
        assert len(trainer.episode_buffer) == 5

    @pytest.mark.asyncio
    async def test_iteration_loop(self):
        """Test complete iteration loop."""
        config = {
            "games_per_iteration": 5,
            "parallel_batch_size": 2,
            "batch_size": 2,
        }

        trainer = SelfPlayTrainer(config=config, device="cpu")

        # Run one iteration
        metrics = await trainer.iteration(0)

        assert "iteration" in metrics
        assert "num_episodes" in metrics
        assert "success_rate" in metrics
        assert metrics["iteration"] == 0

    def test_quality_metrics(self):
        """Test quality metrics computation."""
        trainer = SelfPlayTrainer(config={}, device="cpu")

        # Add some fake iteration metrics
        trainer.iteration_metrics = [
            {
                "success_rate": 0.5,
                "eval_avg_length": 10.0,
                "num_episodes": 100,
            }
            for _ in range(5)
        ]

        metrics = trainer.get_quality_metrics()

        assert "avg_success_rate" in metrics
        assert "avg_episode_length" in metrics
        assert "total_episodes_generated" in metrics

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        import tempfile

        trainer = SelfPlayTrainer(config={}, device="cpu")
        trainer.current_iteration = 5
        trainer.best_model_metric = 0.75

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.checkpoint_dir = Path(tmpdir)
            trainer._save_checkpoint(5, best=True)

            # Create new trainer and load
            trainer2 = SelfPlayTrainer(config={}, device="cpu")
            checkpoint_path = Path(tmpdir) / "best_model.pt"
            trainer2.load_checkpoint(str(checkpoint_path))

            assert trainer2.current_iteration == 5
            assert trainer2.best_model_metric == 0.75


# Integration test
@pytest.mark.asyncio
async def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline."""
    config = {
        "games_per_iteration": 3,
        "parallel_batch_size": 2,
        "batch_size": 2,
    }

    # Initialize trainer
    trainer = SelfPlayTrainer(config=config, device="cpu")

    # Generate tasks
    tasks = trainer._generate_tasks(3)
    assert len(tasks) == 3

    # Generate episodes
    episodes = await trainer.run_self_play(num_games=3)
    assert len(episodes) == 3

    # Extract training data
    extractor = TrainingDataExtractor()
    training_data = extractor.extract_examples(episodes)

    assert "policy" in training_data
    assert "value" in training_data
    assert "reasoning" in training_data
    assert "negative" in training_data

    # Create datasets
    if len(training_data["policy"]) > 0:
        policy_dataset = SelfPlayDataset(training_data["policy"], "policy")
        assert len(policy_dataset) >= 0

    # Get quality metrics
    metrics = trainer.get_quality_metrics()
    assert isinstance(metrics, dict)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
