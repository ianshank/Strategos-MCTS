"""Tests for Training Data Collector."""

import json
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="numpy required for MCTS framework")

from src.framework.mcts.llm_guided.data_collector import (
    EpisodeMetadata,
    TrainingDataCollector,
    TrainingExample,
    merge_collectors,
)
from src.framework.mcts.llm_guided.node import (
    LLMGuidedMCTSNode,
    NodeState,
)


class TestTrainingExample:
    """Tests for TrainingExample."""

    def test_creation(self):
        """Test basic creation."""
        example = TrainingExample(
            state_code="def foo(): return 1",
            state_problem="Return 1",
            state_hash="abc123",
            depth=2,
            llm_action_probs={"a": 0.5, "b": 0.5},
            mcts_action_probs={"a": 0.7, "b": 0.3},
            llm_value_estimate=0.8,
            outcome=1.0,
            episode_id="ep1",
            timestamp=123456.0,
            visits=10,
            q_value=0.6,
        )

        assert example.state_code == "def foo(): return 1"
        assert example.depth == 2
        assert example.llm_action_probs == {"a": 0.5, "b": 0.5}
        assert example.outcome == 1.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        example = TrainingExample(
            state_code="code",
            state_problem="problem",
            state_hash="hash",
            depth=1,
            llm_action_probs={},
            mcts_action_probs={},
            llm_value_estimate=0.5,
            outcome=0.0,
            episode_id="ep",
            timestamp=0.0,
            visits=0,
            q_value=0.0,
        )

        d = example.to_dict()
        assert d["state_code"] == "code"
        assert d["episode_id"] == "ep"

    def test_to_json(self):
        """Test JSON serialization."""
        example = TrainingExample(
            state_code="code",
            state_problem="problem",
            state_hash="hash",
            depth=1,
            llm_action_probs={},
            mcts_action_probs={},
            llm_value_estimate=0.5,
            outcome=0.0,
            episode_id="ep",
            timestamp=0.0,
            visits=0,
            q_value=0.0,
        )

        json_str = example.to_json()
        data = json.loads(json_str)
        assert data["state_code"] == "code"

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "state_code": "code",
            "state_problem": "problem",
            "state_hash": "hash",
            "depth": 1,
            "llm_action_probs": {"a": 1.0},
            "mcts_action_probs": {},
            "llm_value_estimate": 0.5,
            "outcome": 1.0,
            "episode_id": "ep",
            "timestamp": 0.0,
            "visits": 5,
            "q_value": 0.3,
        }

        example = TrainingExample.from_dict(d)
        assert example.state_code == "code"
        assert example.llm_action_probs == {"a": 1.0}
        assert example.visits == 5

    def test_from_json(self):
        """Test creation from JSON."""
        json_str = json.dumps(
            {
                "state_code": "code",
                "state_problem": "problem",
                "state_hash": "hash",
                "depth": 1,
                "llm_action_probs": {},
                "mcts_action_probs": {},
                "llm_value_estimate": 0.5,
                "outcome": 0.0,
                "episode_id": "ep",
                "timestamp": 0.0,
                "visits": 0,
                "q_value": 0.0,
            }
        )

        example = TrainingExample.from_json(json_str)
        assert example.state_code == "code"


class TestEpisodeMetadata:
    """Tests for EpisodeMetadata."""

    def test_creation(self):
        """Test basic creation."""
        meta = EpisodeMetadata(
            episode_id="ep123",
            problem_type="code_generation",
            difficulty="medium",
            start_time=1000.0,
        )

        assert meta.episode_id == "ep123"
        assert meta.problem_type == "code_generation"
        assert meta.difficulty == "medium"
        assert meta.outcome == 0.0
        assert meta.solution_found is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        meta = EpisodeMetadata(
            episode_id="ep123",
            problem_type="test",
            difficulty="easy",
            start_time=1000.0,
            end_time=2000.0,
            outcome=1.0,
            num_iterations=50,
            solution_found=True,
        )

        d = meta.to_dict()
        assert d["episode_id"] == "ep123"
        assert d["outcome"] == 1.0
        assert d["solution_found"] is True


class TestTrainingDataCollector:
    """Tests for TrainingDataCollector."""

    def test_initialization(self):
        """Test collector initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrainingDataCollector(output_dir=tmpdir)

            assert collector.output_dir == Path(tmpdir)
            assert collector.total_episodes == 0
            assert collector.total_examples == 0

    def test_start_episode(self):
        """Test starting a new episode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrainingDataCollector(output_dir=tmpdir)
            collector.start_episode(
                episode_id="ep1",
                problem_type="code_generation",
                difficulty="hard",
            )

            assert collector.episode_metadata is not None
            assert collector.episode_metadata.episode_id == "ep1"
            assert collector.episode_metadata.problem_type == "code_generation"
            assert collector.episode_metadata.difficulty == "hard"
            assert collector.current_episode == []

    def test_record_node(self):
        """Test recording a node."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrainingDataCollector(output_dir=tmpdir)
            collector.start_episode(episode_id="ep1")

            state = NodeState(code="def foo(): pass", problem="test")
            node = LLMGuidedMCTSNode(state=state, episode_id="ep1")
            node.visits = 10
            node.value_sum = 5.0
            node.llm_action_probs = {"a": 0.8}
            node.llm_value_estimate = 0.7

            collector.record_node(node)

            assert len(collector.current_episode) == 1
            example = collector.current_episode[0]
            assert example.state_code == "def foo(): pass"
            assert example.visits == 10
            assert example.llm_action_probs == {"a": 0.8}

    def test_record_node_without_episode(self):
        """Test recording node without starting episode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrainingDataCollector(output_dir=tmpdir)

            state = NodeState(code="def foo(): pass", problem="test")
            node = LLMGuidedMCTSNode(state=state)

            # Should not raise, just log warning
            collector.record_node(node)
            assert len(collector.current_episode) == 0

    def test_record_mcts_policy(self):
        """Test recording MCTS policy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrainingDataCollector(output_dir=tmpdir)
            collector.start_episode(episode_id="ep1")

            state = NodeState(code="def foo(): pass", problem="test")
            parent = LLMGuidedMCTSNode(state=state, episode_id="ep1")

            # Record parent
            collector.record_node(parent)

            # Add children with visits
            child1 = parent.add_child(state=state, action="a")
            child1.visits = 30
            child2 = parent.add_child(state=state, action="b")
            child2.visits = 70

            # Record MCTS policy
            collector.record_mcts_policy(parent)

            # Check that example was updated
            example = collector.current_episode[0]
            assert example.mcts_action_probs == {"a": 0.3, "b": 0.7}

    def test_finalize_episode(self):
        """Test finalizing and saving episode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrainingDataCollector(output_dir=tmpdir)
            collector.start_episode(episode_id="ep1")

            state = NodeState(code="def foo(): return 1", problem="test")
            node = LLMGuidedMCTSNode(state=state, episode_id="ep1")
            collector.record_node(node)

            filepath = collector.finalize_episode(
                outcome=1.0,
                solution_found=True,
                solution_code="def foo(): return 1",
                total_iterations=30,
            )

            # Check file was created
            assert Path(filepath).exists()

            # Check statistics updated
            assert collector.total_episodes == 1
            assert collector.total_examples == 1
            assert collector.successful_episodes == 1

            # Check episode was cleared
            assert collector.current_episode == []
            assert collector.episode_metadata is None

    def test_finalize_episode_without_start(self):
        """Test finalize without starting episode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrainingDataCollector(output_dir=tmpdir)

            with pytest.raises(ValueError, match="No episode started"):
                collector.finalize_episode(outcome=1.0)

    def test_load_episode(self):
        """Test loading a saved episode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrainingDataCollector(output_dir=tmpdir)
            collector.start_episode(episode_id="ep1", problem_type="test")

            state = NodeState(code="def foo(): return 1", problem="test")
            node = LLMGuidedMCTSNode(state=state, episode_id="ep1")
            collector.record_node(node)

            filepath = collector.finalize_episode(outcome=1.0, solution_found=True)

            # Load the episode
            metadata, examples = collector.load_episode(filepath)

            assert metadata.episode_id == "ep1"
            assert metadata.problem_type == "test"
            assert len(examples) == 1
            assert examples[0].state_code == "def foo(): return 1"

    def test_load_all_episodes(self):
        """Test loading all episodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrainingDataCollector(output_dir=tmpdir)

            # Create two episodes
            for i in range(2):
                collector.start_episode(episode_id=f"ep{i}")
                state = NodeState(code=f"code_{i}", problem="test")
                node = LLMGuidedMCTSNode(state=state)
                collector.record_node(node)
                collector.finalize_episode(outcome=float(i))

            episodes = collector.load_all_episodes()
            assert len(episodes) == 2

    def test_get_statistics(self):
        """Test getting collection statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrainingDataCollector(output_dir=tmpdir)

            # Create an episode
            collector.start_episode(episode_id="ep1")
            state = NodeState(code="code", problem="test")
            node = LLMGuidedMCTSNode(state=state)
            collector.record_node(node)
            collector.finalize_episode(outcome=1.0, solution_found=True)

            stats = collector.get_statistics()

            assert stats["total_episodes"] == 1
            assert stats["total_examples"] == 1
            assert stats["successful_episodes"] == 1
            assert stats["success_rate"] == 1.0
            assert stats["num_files"] == 1

    def test_create_train_val_test_split(self):
        """Test creating train/val/test splits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrainingDataCollector(output_dir=tmpdir)

            # Create multiple episodes with multiple examples
            for i in range(10):
                collector.start_episode(episode_id=f"ep{i}")
                for j in range(3):
                    state = NodeState(code=f"code_{i}_{j}", problem="test")
                    node = LLMGuidedMCTSNode(state=state)
                    collector.record_node(node)
                collector.finalize_episode(outcome=1.0)

            # Create splits
            paths = collector.create_train_val_test_split(
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                seed=42,
            )

            assert "train" in paths
            assert "val" in paths
            assert "test" in paths

            # Check files exist
            assert paths["train"].exists()
            assert paths["val"].exists()
            assert paths["test"].exists()

    def test_create_split_invalid_ratios(self):
        """Test split with invalid ratios."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrainingDataCollector(output_dir=tmpdir)

            with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
                collector.create_train_val_test_split(
                    train_ratio=0.5,
                    val_ratio=0.5,
                    test_ratio=0.5,
                )


class TestMergeCollectors:
    """Tests for merge_collectors function."""

    def test_merge_collectors(self):
        """Test merging multiple collectors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first collector
            collector1 = TrainingDataCollector(output_dir=Path(tmpdir) / "c1")
            collector1.start_episode(episode_id="ep1")
            state = NodeState(code="code1", problem="test")
            node = LLMGuidedMCTSNode(state=state)
            collector1.record_node(node)
            collector1.finalize_episode(outcome=1.0, solution_found=True)

            # Create second collector
            collector2 = TrainingDataCollector(output_dir=Path(tmpdir) / "c2")
            collector2.start_episode(episode_id="ep2")
            state = NodeState(code="code2", problem="test")
            node = LLMGuidedMCTSNode(state=state)
            collector2.record_node(node)
            collector2.finalize_episode(outcome=0.0, solution_found=False)

            # Merge
            merged = merge_collectors(
                collector1,
                collector2,
                output_dir=Path(tmpdir) / "merged",
            )

            assert merged.total_episodes == 2
            assert merged.total_examples == 2
            assert merged.successful_episodes == 1
