"""
Training Data Collector for LLM-Guided MCTS.

Collects (state, action, value) tuples during MCTS execution
for training neural networks in Phase 2.

Provides:
- TrainingExample: Single training example dataclass
- EpisodeMetadata: Episode-level metadata
- TrainingDataCollector: Main collector class
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

from src.observability.logging import get_correlation_id, get_structured_logger

if TYPE_CHECKING:
    from .node import LLMGuidedMCTSNode

logger = get_structured_logger(__name__)


@dataclass
class TrainingExample:
    """
    Single training example for neural network training.

    Contains all fields needed for training policy and value networks.
    """

    # State representation
    state_code: str
    """Code at this state."""

    state_problem: str
    """Problem description."""

    state_hash: str
    """Unique hash of the state."""

    depth: int
    """Depth in the search tree."""

    # Policy targets
    llm_action_probs: dict[str, float]
    """LLM's action probability distribution (teacher labels)."""

    mcts_action_probs: dict[str, float]
    """Improved policy from MCTS visit counts."""

    # Value targets
    llm_value_estimate: float
    """LLM's value estimate for this state."""

    outcome: float
    """Final episode outcome (1.0 = success, -1.0 = failure)."""

    # Metadata
    episode_id: str
    """Episode identifier."""

    timestamp: float
    """When this example was created."""

    visits: int
    """Number of visits to this node."""

    q_value: float
    """Average value from MCTS."""

    # Optional fields
    action: str | None = None
    """Action taken from this state."""

    test_results: dict[str, Any] | None = None
    """Code execution results."""

    errors: list[str] = field(default_factory=list)
    """Errors from code execution."""

    parent_visits: int = 0
    """Number of visits to parent node."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingExample:
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> TrainingExample:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class EpisodeMetadata:
    """Metadata for a complete episode."""

    episode_id: str
    """Unique episode identifier."""

    problem_type: str
    """Type of problem (e.g., 'code_generation', 'bug_fix')."""

    difficulty: str
    """Problem difficulty (e.g., 'easy', 'medium', 'hard')."""

    start_time: float
    """Episode start time."""

    end_time: float | None = None
    """Episode end time."""

    outcome: float = 0.0
    """Final outcome (1.0 = success, -1.0 = failure)."""

    num_iterations: int = 0
    """Number of MCTS iterations."""

    num_examples: int = 0
    """Number of training examples collected."""

    solution_found: bool = False
    """Whether a solution was found."""

    solution_code: str | None = None
    """The solution code if found."""

    total_llm_calls: int = 0
    """Total LLM calls made."""

    total_tokens: int = 0
    """Total tokens used."""

    config_name: str = ""
    """Configuration name used."""

    correlation_id: str = ""
    """Correlation ID for tracing."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())


class TrainingDataCollector:
    """
    Collects (state, action, value) tuples during MCTS execution
    for training neural networks in Phase 2.

    Features:
    - Collects training examples from MCTS nodes
    - Saves to JSONL format for PyTorch DataLoader
    - Computes MCTS-improved policy targets
    - Tracks episode-level statistics
    """

    def __init__(
        self,
        output_dir: str | Path = "./training_data",
        batch_size: int = 100,
        auto_save: bool = True,
    ):
        """
        Initialize data collector.

        Args:
            output_dir: Directory to save training data
            batch_size: Number of examples before auto-saving batch
            auto_save: Automatically save when batch is full
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.auto_save = auto_save

        # Current episode state
        self.current_episode: list[TrainingExample] = []
        self.episode_metadata: EpisodeMetadata | None = None

        # Batch buffer
        self._batch_buffer: list[TrainingExample] = []
        self._batch_count = 0

        # Statistics
        self.total_episodes = 0
        self.total_examples = 0
        self.successful_episodes = 0

        logger.info(
            "Initialized TrainingDataCollector",
            output_dir=str(self.output_dir),
            batch_size=batch_size,
        )

    def start_episode(
        self,
        episode_id: str,
        problem_type: str = "code_generation",
        difficulty: str = "medium",
        config_name: str = "",
    ) -> None:
        """
        Initialize a new episode.

        Args:
            episode_id: Unique episode identifier
            problem_type: Type of problem
            difficulty: Problem difficulty
            config_name: Configuration name
        """
        self.current_episode = []
        self.episode_metadata = EpisodeMetadata(
            episode_id=episode_id,
            problem_type=problem_type,
            difficulty=difficulty,
            start_time=time.time(),
            config_name=config_name,
            correlation_id=get_correlation_id(),
        )

        logger.debug(
            "Started episode",
            episode_id=episode_id,
            problem_type=problem_type,
            difficulty=difficulty,
        )

    def record_node(
        self,
        node: LLMGuidedMCTSNode,
        llm_response: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a node expansion for training.

        Args:
            node: The MCTS node to record
            llm_response: Optional LLM response with action probabilities
        """
        if self.episode_metadata is None:
            logger.warning("Attempted to record node without starting episode")
            return

        # Extract LLM predictions from response or node
        llm_action_probs = llm_response.get("action_probs", {}) if llm_response else node.llm_action_probs
        llm_value = llm_response.get("value", 0.0) if llm_response else node.llm_value_estimate

        example = TrainingExample(
            state_code=node.state.code,
            state_problem=node.state.problem,
            state_hash=node.state.to_hash_key(),
            depth=node.depth,
            llm_action_probs=llm_action_probs,
            mcts_action_probs={},  # Will be filled at end of episode
            llm_value_estimate=llm_value,
            outcome=0.0,  # Will be filled at end of episode
            episode_id=self.episode_metadata.episode_id,
            timestamp=time.time(),
            visits=node.visits,
            q_value=node.q_value,
            action=node.action,
            test_results=node.test_results,
            errors=node.state.errors,
            parent_visits=node.parent.visits if node.parent else 0,
        )

        self.current_episode.append(example)

    def record_mcts_policy(self, node: LLMGuidedMCTSNode) -> None:
        """
        Record MCTS-improved policy for a node.

        The MCTS visit distribution is a better training target
        than raw LLM predictions because it incorporates search.

        Args:
            node: Node with children to compute policy from
        """
        if not node.children:
            return

        # Compute policy from visit counts
        mcts_policy = node.compute_mcts_policy()

        # Find and update the corresponding example
        state_hash = node.state.to_hash_key()
        for example in reversed(self.current_episode):
            if example.state_hash == state_hash:
                example.mcts_action_probs = mcts_policy
                break

    def finalize_episode(
        self,
        outcome: float,
        solution_found: bool = False,
        solution_code: str | None = None,
        total_iterations: int = 0,
        total_llm_calls: int = 0,
        total_tokens: int = 0,
    ) -> str:
        """
        Finalize episode and save training data.

        Args:
            outcome: Final outcome (1.0 = success, -1.0 = failure)
            solution_found: Whether a solution was found
            solution_code: The solution code if found
            total_iterations: Total MCTS iterations
            total_llm_calls: Total LLM calls
            total_tokens: Total tokens used

        Returns:
            Path to saved episode file
        """
        if self.episode_metadata is None:
            raise ValueError("No episode started")

        # Update all examples with final outcome
        for example in self.current_episode:
            example.outcome = outcome

        # Update episode metadata
        self.episode_metadata.end_time = time.time()
        self.episode_metadata.outcome = outcome
        self.episode_metadata.num_iterations = total_iterations
        self.episode_metadata.num_examples = len(self.current_episode)
        self.episode_metadata.solution_found = solution_found
        self.episode_metadata.solution_code = solution_code
        self.episode_metadata.total_llm_calls = total_llm_calls
        self.episode_metadata.total_tokens = total_tokens

        # Save to file
        filepath = self._save_episode()

        # Update statistics
        self.total_episodes += 1
        self.total_examples += len(self.current_episode)
        if solution_found:
            self.successful_episodes += 1

        logger.info(
            "Finalized episode",
            episode_id=self.episode_metadata.episode_id,
            num_examples=len(self.current_episode),
            outcome=outcome,
            solution_found=solution_found,
            filepath=str(filepath),
        )

        # Clear current episode
        self.current_episode = []
        self.episode_metadata = None

        return str(filepath)

    def _save_episode(self) -> Path:
        """Save current episode to JSONL file."""
        if self.episode_metadata is None:
            raise ValueError("No episode metadata")

        episode_id = self.episode_metadata.episode_id

        # Create filename with timestamp for uniqueness
        timestamp = int(time.time())
        filename = f"episode_{episode_id}_{timestamp}.jsonl"
        filepath = self.output_dir / filename

        # Write examples to JSONL
        with open(filepath, "w") as f:
            # Write metadata as first line
            f.write(json.dumps({"_metadata": self.episode_metadata.to_dict()}) + "\n")

            # Write each example
            for example in self.current_episode:
                f.write(example.to_json() + "\n")

        return filepath

    def add_to_batch(self, examples: list[TrainingExample]) -> None:
        """
        Add examples to batch buffer.

        Args:
            examples: Examples to add
        """
        self._batch_buffer.extend(examples)

        if self.auto_save and len(self._batch_buffer) >= self.batch_size:
            self._save_batch()

    def _save_batch(self) -> Path:
        """Save batch buffer to file."""
        self._batch_count += 1
        filename = f"batch_{self._batch_count}_{int(time.time())}.jsonl"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            for example in self._batch_buffer:
                f.write(example.to_json() + "\n")

        logger.info(
            "Saved batch",
            batch_count=self._batch_count,
            num_examples=len(self._batch_buffer),
            filepath=str(filepath),
        )

        self._batch_buffer = []
        return filepath

    def flush(self) -> None:
        """Save any remaining examples in the batch buffer."""
        if self._batch_buffer:
            self._save_batch()

    def get_statistics(self) -> dict[str, Any]:
        """Get collection statistics."""
        # Count files and examples
        files = list(self.output_dir.glob("episode_*.jsonl"))
        total_file_examples = 0

        for file in files:
            with open(file) as f:
                # Subtract 1 for metadata line
                total_file_examples += sum(1 for _ in f) - 1

        return {
            "output_dir": str(self.output_dir),
            "total_episodes": self.total_episodes,
            "total_examples": self.total_examples,
            "successful_episodes": self.successful_episodes,
            "success_rate": (self.successful_episodes / self.total_episodes if self.total_episodes > 0 else 0.0),
            "num_files": len(files),
            "file_examples": total_file_examples,
            "current_episode_size": len(self.current_episode),
            "batch_buffer_size": len(self._batch_buffer),
        }

    def load_episode(self, filepath: str | Path) -> tuple[EpisodeMetadata, list[TrainingExample]]:
        """
        Load an episode from file.

        Args:
            filepath: Path to episode JSONL file

        Returns:
            Tuple of (metadata, examples)
        """
        filepath = Path(filepath)
        metadata = None
        examples = []

        with open(filepath) as f:
            for line in f:
                data = json.loads(line)
                if "_metadata" in data:
                    metadata = EpisodeMetadata(**data["_metadata"])
                else:
                    examples.append(TrainingExample.from_dict(data))

        if metadata is None:
            raise ValueError(f"No metadata found in {filepath}")

        return metadata, examples

    def load_all_episodes(self) -> list[tuple[EpisodeMetadata, list[TrainingExample]]]:
        """
        Load all episodes from output directory.

        Returns:
            List of (metadata, examples) tuples
        """
        episodes = []
        for filepath in sorted(self.output_dir.glob("episode_*.jsonl")):
            try:
                episodes.append(self.load_episode(filepath))
            except Exception as e:
                logger.warning(
                    "Failed to load episode",
                    filepath=str(filepath),
                    error=str(e),
                    error_type=type(e).__name__,
                )

        return episodes

    def create_train_val_test_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> dict[str, Path]:
        """
        Split collected data into train/val/test sets.

        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed for reproducibility

        Returns:
            Dictionary with paths to split files
        """
        import random

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Ratios must sum to 1.0")

        # Load all examples
        all_examples = []
        for _, examples in self.load_all_episodes():
            all_examples.extend(examples)

        # Shuffle
        random.seed(seed)
        random.shuffle(all_examples)

        # Split
        n = len(all_examples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_examples = all_examples[:train_end]
        val_examples = all_examples[train_end:val_end]
        test_examples = all_examples[val_end:]

        # Save splits
        splits_dir = self.output_dir / "splits"
        splits_dir.mkdir(exist_ok=True)

        paths = {}
        for name, examples in [
            ("train", train_examples),
            ("val", val_examples),
            ("test", test_examples),
        ]:
            filepath = splits_dir / f"{name}.jsonl"
            with open(filepath, "w") as f:
                for example in examples:
                    f.write(example.to_json() + "\n")
            paths[name] = filepath

        logger.info(
            "Created train/val/test split",
            train=len(train_examples),
            val=len(val_examples),
            test=len(test_examples),
        )

        return paths


def merge_collectors(*collectors: TrainingDataCollector, output_dir: str | Path) -> TrainingDataCollector:
    """
    Merge multiple collectors into a single one.

    Args:
        *collectors: Collectors to merge
        output_dir: Output directory for merged data

    Returns:
        New collector with merged data
    """
    merged = TrainingDataCollector(output_dir=output_dir)

    for collector in collectors:
        for metadata, examples in collector.load_all_episodes():
            merged.start_episode(
                episode_id=metadata.episode_id,
                problem_type=metadata.problem_type,
                difficulty=metadata.difficulty,
                config_name=metadata.config_name,
            )
            merged.current_episode = examples
            merged.episode_metadata = metadata
            merged._save_episode()
            merged.total_episodes += 1
            merged.total_examples += len(examples)
            if metadata.solution_found:
                merged.successful_episodes += 1
            merged.current_episode = []
            merged.episode_metadata = None

    return merged
