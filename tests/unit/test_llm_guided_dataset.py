"""Unit tests for src/framework/mcts/llm_guided/training/dataset.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# We need to handle torch being optional
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available"),
]

from src.framework.mcts.llm_guided.training.dataset import (
    MCTSDataset,
    MCTSDatasetConfig,
    RawExample,
    TrainingBatch,
    collate_fn,
    create_dataloaders,
    load_training_data,
)


def _make_raw_example(
    *,
    episode_id: str = "ep_1",
    depth: int = 2,
    visits: int = 10,
    outcome: float = 1.0,
    q_value: float = 0.6,
    llm_value: float = 0.5,
) -> RawExample:
    return RawExample(
        state_code="def solve(): pass",
        state_problem="Find the sum",
        state_hash="abc123",
        depth=depth,
        llm_action_probs={"a": 0.6, "b": 0.3, "c": 0.1},
        mcts_action_probs={"a": 0.4, "b": 0.4, "c": 0.2},
        llm_value_estimate=llm_value,
        outcome=outcome,
        episode_id=episode_id,
        visits=visits,
        q_value=q_value,
    )


def _write_episode_file(filepath: Path, examples: list[dict], metadata: dict | None = None):
    """Write a JSONL episode file."""
    with open(filepath, "w") as f:
        if metadata is not None:
            f.write(json.dumps({"_metadata": metadata}) + "\n")
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def _sample_example_dict(
    *,
    depth: int = 2,
    visits: int = 10,
    outcome: float = 1.0,
    episode_id: str = "ep_1",
) -> dict:
    return {
        "state_code": "def solve(): pass",
        "state_problem": "Find the sum",
        "state_hash": "hash123",
        "depth": depth,
        "llm_action_probs": {"a": 0.6, "b": 0.4},
        "mcts_action_probs": {"a": 0.3, "b": 0.7},
        "llm_value_estimate": 0.5,
        "outcome": outcome,
        "episode_id": episode_id,
        "visits": visits,
        "q_value": 0.6,
    }


# ---------------------------------------------------------------------------
# MCTSDatasetConfig tests
# ---------------------------------------------------------------------------


class TestMCTSDatasetConfig:
    def test_default_values(self):
        cfg = MCTSDatasetConfig()
        assert cfg.data_dir == "./training_data"
        assert cfg.max_code_length == 2048
        assert cfg.max_problem_length == 1024
        assert cfg.max_actions == 10
        assert cfg.tokenizer_name == "gpt2"
        assert cfg.shuffle_actions is False
        assert cfg.min_visits == 1
        assert cfg.exclude_root_nodes is True
        assert cfg.only_successful_episodes is False

    def test_validate_passes(self):
        cfg = MCTSDatasetConfig()
        cfg.validate()  # Should not raise

    def test_validate_max_code_length(self):
        cfg = MCTSDatasetConfig(max_code_length=0)
        with pytest.raises(ValueError, match="max_code_length"):
            cfg.validate()

    def test_validate_max_problem_length(self):
        cfg = MCTSDatasetConfig(max_problem_length=0)
        with pytest.raises(ValueError, match="max_problem_length"):
            cfg.validate()

    def test_validate_max_actions(self):
        cfg = MCTSDatasetConfig(max_actions=0)
        with pytest.raises(ValueError, match="max_actions"):
            cfg.validate()

    def test_validate_min_visits(self):
        cfg = MCTSDatasetConfig(min_visits=-1)
        with pytest.raises(ValueError, match="min_visits"):
            cfg.validate()

    def test_validate_multiple_errors(self):
        cfg = MCTSDatasetConfig(max_code_length=0, max_actions=0)
        with pytest.raises(ValueError) as exc_info:
            cfg.validate()
        msg = str(exc_info.value)
        assert "max_code_length" in msg
        assert "max_actions" in msg


# ---------------------------------------------------------------------------
# RawExample tests
# ---------------------------------------------------------------------------


class TestRawExample:
    def test_fields(self):
        ex = _make_raw_example()
        assert ex.state_code == "def solve(): pass"
        assert ex.state_problem == "Find the sum"
        assert ex.depth == 2
        assert ex.visits == 10
        assert ex.outcome == 1.0
        assert ex.q_value == 0.6
        assert ex.episode_id == "ep_1"
        assert "a" in ex.llm_action_probs
        assert "a" in ex.mcts_action_probs


# ---------------------------------------------------------------------------
# TrainingBatch tests
# ---------------------------------------------------------------------------


class TestTrainingBatch:
    def _make_batch(self):
        bs = 2
        return TrainingBatch(
            code_tokens=torch.zeros(bs, 4, dtype=torch.long),
            code_attention_mask=torch.ones(bs, 4, dtype=torch.long),
            problem_tokens=torch.zeros(bs, 3, dtype=torch.long),
            problem_attention_mask=torch.ones(bs, 3, dtype=torch.long),
            llm_policy=torch.tensor([[0.5, 0.5], [0.3, 0.7]]),
            mcts_policy=torch.tensor([[0.4, 0.6], [0.2, 0.8]]),
            action_mask=torch.ones(bs, 2),
            llm_value=torch.tensor([0.5, 0.6]),
            outcome=torch.tensor([1.0, 0.0]),
            q_value=torch.tensor([0.6, 0.4]),
            episode_ids=["ep_1", "ep_2"],
            depths=torch.tensor([2, 3]),
            visits=torch.tensor([10, 20]),
        )

    def test_to_device(self):
        batch = self._make_batch()
        moved = batch.to("cpu")
        assert moved.code_tokens.device == torch.device("cpu")
        assert moved.episode_ids == ["ep_1", "ep_2"]

    def test_to_preserves_shape(self):
        batch = self._make_batch()
        moved = batch.to("cpu")
        assert moved.code_tokens.shape == (2, 4)
        assert moved.llm_policy.shape == (2, 2)


# ---------------------------------------------------------------------------
# MCTSDataset tests
# ---------------------------------------------------------------------------


class TestMCTSDataset:
    def test_init_with_examples(self):
        examples = [_make_raw_example(), _make_raw_example(episode_id="ep_2")]
        cfg = MCTSDatasetConfig(max_code_length=32, max_problem_length=16, max_actions=5)
        ds = MCTSDataset(config=cfg, examples=examples)
        assert len(ds) == 2

    def test_getitem(self):
        examples = [_make_raw_example()]
        cfg = MCTSDatasetConfig(max_code_length=32, max_problem_length=16, max_actions=5)
        ds = MCTSDataset(config=cfg, examples=examples)
        item = ds[0]
        assert "code_tokens" in item
        assert "code_attention_mask" in item
        assert "problem_tokens" in item
        assert "llm_policy" in item
        assert "mcts_policy" in item
        assert "action_mask" in item
        assert "llm_value" in item
        assert "outcome" in item
        assert "q_value" in item
        assert "episode_id" in item
        assert item["episode_id"] == "ep_1"

    def test_getitem_tensor_shapes(self):
        examples = [_make_raw_example()]
        cfg = MCTSDatasetConfig(max_code_length=32, max_problem_length=16, max_actions=5)
        ds = MCTSDataset(config=cfg, examples=examples)
        item = ds[0]
        assert item["code_tokens"].shape == (32,)
        assert item["code_attention_mask"].shape == (32,)
        assert item["problem_tokens"].shape == (16,)
        assert item["problem_attention_mask"].shape == (16,)
        assert item["llm_policy"].shape == (5,)
        assert item["mcts_policy"].shape == (5,)
        assert item["action_mask"].shape == (5,)

    def test_encode_policies_normalization(self):
        examples = [_make_raw_example()]
        cfg = MCTSDatasetConfig(max_code_length=32, max_problem_length=16, max_actions=5)
        ds = MCTSDataset(config=cfg, examples=examples)
        llm_p, mcts_p, mask = ds._encode_policies(
            {"a": 0.6, "b": 0.4}, {"a": 0.3, "b": 0.7}
        )
        # Normalized sums should be ~1.0 for non-zero actions
        assert abs(sum(llm_p[:2]) - 1.0) < 1e-6
        assert abs(sum(mcts_p[:2]) - 1.0) < 1e-6
        # Mask should have 1s for real actions and 0s for padding
        assert mask[:2] == [1.0, 1.0]
        assert mask[2:] == [0.0, 0.0, 0.0]

    def test_encode_policies_empty(self):
        examples = [_make_raw_example()]
        cfg = MCTSDatasetConfig(max_code_length=32, max_problem_length=16, max_actions=5)
        ds = MCTSDataset(config=cfg, examples=examples)
        llm_p, mcts_p, mask = ds._encode_policies({}, {})
        assert all(v == 0.0 for v in llm_p)
        assert all(v == 0.0 for v in mask)

    def test_encode_policies_more_actions_than_max(self):
        examples = [_make_raw_example()]
        cfg = MCTSDatasetConfig(max_code_length=32, max_problem_length=16, max_actions=2)
        ds = MCTSDataset(config=cfg, examples=examples)
        # 3 actions but max_actions=2
        llm_p, mcts_p, mask = ds._encode_policies(
            {"a": 0.5, "b": 0.3, "c": 0.2},
            {"a": 0.4, "b": 0.4, "c": 0.2},
        )
        assert len(llm_p) == 2
        assert len(mask) == 2

    def test_tokenize_fallback(self):
        """Test the simple character-level fallback tokenizer."""
        examples = [_make_raw_example()]
        cfg = MCTSDatasetConfig(max_code_length=10, max_problem_length=8, max_actions=3)
        ds = MCTSDataset(config=cfg, examples=examples)
        # Force fallback tokenizer
        ds._tokenizer = None
        result = ds._tokenize("hello", 10)
        assert len(result["input_ids"]) == 10
        assert len(result["attention_mask"]) == 10
        # "hello" = 5 chars, so 5 ones + 5 zeros
        assert result["attention_mask"][:5] == [1, 1, 1, 1, 1]
        assert result["attention_mask"][5:] == [0, 0, 0, 0, 0]

    def test_tokenize_fallback_truncation(self):
        examples = [_make_raw_example()]
        cfg = MCTSDatasetConfig(max_code_length=3, max_problem_length=3, max_actions=3)
        ds = MCTSDataset(config=cfg, examples=examples)
        ds._tokenizer = None
        result = ds._tokenize("hello world", 3)
        assert len(result["input_ids"]) == 3
        assert result["attention_mask"] == [1, 1, 1]

    def test_load_examples_missing_dir(self, tmp_path):
        cfg = MCTSDatasetConfig(data_dir=str(tmp_path / "nonexistent"))
        ds = MCTSDataset(config=cfg)
        assert len(ds) == 0

    def test_load_examples_empty_dir(self, tmp_path):
        data_dir = tmp_path / "empty_data"
        data_dir.mkdir()
        cfg = MCTSDatasetConfig(data_dir=str(data_dir))
        ds = MCTSDataset(config=cfg)
        assert len(ds) == 0

    def test_load_episode_file(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        examples = [
            _sample_example_dict(depth=1, visits=5),
            _sample_example_dict(depth=2, visits=10),
        ]
        _write_episode_file(data_dir / "episode_001.jsonl", examples)
        cfg = MCTSDatasetConfig(
            data_dir=str(data_dir), min_visits=1, exclude_root_nodes=False
        )
        ds = MCTSDataset(config=cfg)
        assert len(ds) == 2

    def test_load_episode_file_with_metadata(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        examples = [_sample_example_dict(depth=2, visits=5)]
        _write_episode_file(
            data_dir / "episode_001.jsonl",
            examples,
            metadata={"solution_found": True},
        )
        cfg = MCTSDatasetConfig(
            data_dir=str(data_dir), min_visits=1, exclude_root_nodes=False
        )
        ds = MCTSDataset(config=cfg)
        assert len(ds) == 1

    def test_filter_only_successful_episodes(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        examples = [_sample_example_dict(depth=2, visits=5)]
        _write_episode_file(
            data_dir / "episode_001.jsonl",
            examples,
            metadata={"solution_found": False},
        )
        cfg = MCTSDatasetConfig(
            data_dir=str(data_dir),
            only_successful_episodes=True,
            exclude_root_nodes=False,
        )
        ds = MCTSDataset(config=cfg)
        assert len(ds) == 0

    def test_filter_min_visits(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        examples = [
            _sample_example_dict(depth=1, visits=1),
            _sample_example_dict(depth=2, visits=20),
        ]
        _write_episode_file(data_dir / "episode_001.jsonl", examples)
        cfg = MCTSDatasetConfig(
            data_dir=str(data_dir),
            min_visits=10,
            exclude_root_nodes=False,
        )
        ds = MCTSDataset(config=cfg)
        assert len(ds) == 1

    def test_filter_exclude_root_nodes(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        examples = [
            _sample_example_dict(depth=0, visits=10),
            _sample_example_dict(depth=1, visits=10),
        ]
        _write_episode_file(data_dir / "episode_001.jsonl", examples)
        cfg = MCTSDatasetConfig(
            data_dir=str(data_dir), min_visits=1, exclude_root_nodes=True
        )
        ds = MCTSDataset(config=cfg)
        assert len(ds) == 1

    def test_load_malformed_json(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        filepath = data_dir / "episode_001.jsonl"
        with open(filepath, "w") as f:
            f.write("not valid json\n")
            f.write(json.dumps(_sample_example_dict(depth=2, visits=5)) + "\n")
        cfg = MCTSDatasetConfig(
            data_dir=str(data_dir), min_visits=1, exclude_root_nodes=False
        )
        ds = MCTSDataset(config=cfg)
        assert len(ds) == 1  # Only the valid line

    def test_load_split_files(self, tmp_path):
        data_dir = tmp_path / "data"
        splits_dir = data_dir / "splits"
        splits_dir.mkdir(parents=True)
        examples = [_sample_example_dict(depth=2, visits=5)]
        _write_episode_file(splits_dir / "train.jsonl", examples)
        cfg = MCTSDatasetConfig(
            data_dir=str(data_dir), min_visits=1, exclude_root_nodes=False
        )
        ds = MCTSDataset(config=cfg)
        assert len(ds) == 1

    def test_get_statistics_empty(self):
        cfg = MCTSDatasetConfig(max_code_length=8, max_problem_length=8, max_actions=3)
        ds = MCTSDataset(config=cfg, examples=[])
        stats = ds.get_statistics()
        assert stats["num_examples"] == 0

    def test_get_statistics(self):
        examples = [
            _make_raw_example(depth=1, visits=5, outcome=1.0, q_value=0.8, episode_id="e1"),
            _make_raw_example(depth=3, visits=15, outcome=0.0, q_value=0.2, episode_id="e2"),
        ]
        cfg = MCTSDatasetConfig(max_code_length=8, max_problem_length=8, max_actions=3)
        ds = MCTSDataset(config=cfg, examples=examples)
        stats = ds.get_statistics()
        assert stats["num_examples"] == 2
        assert stats["num_episodes"] == 2
        assert stats["depth_stats"]["min"] == 1
        assert stats["depth_stats"]["max"] == 3
        assert stats["visits_stats"]["min"] == 5
        assert stats["visits_stats"]["max"] == 15
        assert stats["outcome_stats"]["positive_rate"] == 0.5


# ---------------------------------------------------------------------------
# collate_fn tests
# ---------------------------------------------------------------------------


class TestCollateFn:
    def test_collate(self):
        examples = [_make_raw_example(episode_id="e1"), _make_raw_example(episode_id="e2")]
        cfg = MCTSDatasetConfig(max_code_length=8, max_problem_length=8, max_actions=5)
        ds = MCTSDataset(config=cfg, examples=examples)
        batch_dicts = [ds[0], ds[1]]
        batch = collate_fn(batch_dicts)
        assert isinstance(batch, TrainingBatch)
        assert batch.code_tokens.shape[0] == 2
        assert batch.llm_policy.shape[0] == 2
        assert len(batch.episode_ids) == 2


# ---------------------------------------------------------------------------
# load_training_data tests
# ---------------------------------------------------------------------------


class TestLoadTrainingData:
    def test_load_empty(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        examples = load_training_data(data_dir)
        assert examples == []

    def test_load_with_config(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_episode_file(
            data_dir / "episode_001.jsonl",
            [_sample_example_dict(depth=2, visits=5)],
        )
        cfg = MCTSDatasetConfig(min_visits=1, exclude_root_nodes=False)
        examples = load_training_data(data_dir, config=cfg)
        assert len(examples) == 1

    def test_load_without_config(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_episode_file(
            data_dir / "episode_001.jsonl",
            [_sample_example_dict(depth=2, visits=5)],
        )
        examples = load_training_data(data_dir)
        assert len(examples) == 1


# ---------------------------------------------------------------------------
# create_dataloaders tests
# ---------------------------------------------------------------------------


class TestCreateDataloaders:
    def _make_data_dir(self, tmp_path, n_examples=10):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        examples = [
            _sample_example_dict(depth=i + 1, visits=5 + i, episode_id=f"ep_{i}")
            for i in range(n_examples)
        ]
        _write_episode_file(data_dir / "episode_001.jsonl", examples)
        return data_dir

    def test_create_dataloaders(self, tmp_path):
        data_dir = self._make_data_dir(tmp_path, n_examples=10)
        cfg = MCTSDatasetConfig(
            min_visits=1,
            exclude_root_nodes=False,
            max_code_length=8,
            max_problem_length=8,
            max_actions=3,
        )
        loaders = create_dataloaders(
            data_dir, config=cfg, batch_size=2, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        assert "train" in loaders
        assert "val" in loaders
        assert "test" in loaders

    def test_create_dataloaders_invalid_ratios(self, tmp_path):
        data_dir = self._make_data_dir(tmp_path)
        with pytest.raises(ValueError, match="Ratios must sum"):
            create_dataloaders(data_dir, train_ratio=0.5, val_ratio=0.1, test_ratio=0.1)

    def test_create_dataloaders_empty_raises(self, tmp_path):
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        with pytest.raises(ValueError, match="No training examples"):
            create_dataloaders(data_dir)

    def test_create_dataloaders_no_config(self, tmp_path):
        data_dir = self._make_data_dir(tmp_path, n_examples=10)
        # This should use default config; examples with depth>=1 and visits>=1 pass
        loaders = create_dataloaders(
            data_dir, batch_size=2, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        assert "train" in loaders
