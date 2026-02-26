"""Tests for LLM-Guided MCTS Engine."""

import json
import tempfile
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("numpy", reason="numpy required for MCTS framework")

from src.framework.mcts.llm_guided.config import (
    LLMGuidedMCTSConfig,
    LLMGuidedMCTSPreset,
)
from src.framework.mcts.llm_guided.data_collector import TrainingDataCollector
from src.framework.mcts.llm_guided.engine import (
    LLMGuidedMCTSEngine,
    MCTSSearchResult,
    create_llm_mcts_engine,
)
from src.framework.mcts.llm_guided.node import (
    LLMGuidedMCTSNode,
    NodeState,
    NodeStatus,
    create_root_node,
)


class TestMCTSSearchResult:
    """Tests for MCTSSearchResult."""

    def test_creation(self):
        """Test basic creation."""
        result = MCTSSearchResult(
            solution_found=True,
            best_code="def foo(): return 1",
            best_value=0.9,
            num_iterations=30,
            num_expansions=10,
            num_evaluations=10,
            tree_depth=5,
            tree_size=20,
            execution_time_ms=1000.0,
            llm_calls=20,
            tokens_used=5000,
            episode_id="ep123",
            root_visits=30,
        )

        assert result.solution_found is True
        assert result.best_code == "def foo(): return 1"
        assert result.best_value == 0.9
        assert result.num_iterations == 30

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = MCTSSearchResult(
            solution_found=False,
            best_code="code",
            best_value=0.5,
            num_iterations=10,
            num_expansions=5,
            num_evaluations=5,
            tree_depth=3,
            tree_size=8,
            execution_time_ms=500.0,
            llm_calls=10,
            tokens_used=2000,
            episode_id="ep456",
            root_visits=10,
            action_stats={"a": {"visits": 5}},
        )

        d = result.to_dict()

        assert d["solution_found"] is False
        assert d["best_code"] == "code"
        assert d["action_stats"] == {"a": {"visits": 5}}
        assert d["episode_id"] == "ep456"


class TestLLMGuidedMCTSEngine:
    """Tests for LLMGuidedMCTSEngine."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = AsyncMock()
        client.complete = AsyncMock()
        return client

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LLMGuidedMCTSConfig(
            num_iterations=5,
            max_depth=3,
            collect_training_data=False,
            early_termination_on_solution=True,
        )

    def test_initialization(self, mock_llm_client, config):
        """Test engine initialization."""
        engine = LLMGuidedMCTSEngine(mock_llm_client, config)

        assert engine.config == config
        assert engine._generator is not None
        assert engine._reflector is not None
        assert engine._executor is not None

    def test_initialization_default_config(self, mock_llm_client):
        """Test initialization with default config."""
        engine = LLMGuidedMCTSEngine(mock_llm_client)
        assert engine.config is not None
        assert engine.config.name == "balanced"

    def test_initialization_with_data_collector(self, mock_llm_client, config):
        """Test initialization with data collector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrainingDataCollector(output_dir=tmpdir)
            config.collect_training_data = True

            engine = LLMGuidedMCTSEngine(mock_llm_client, config, collector)

            assert engine._data_collector == collector

    def test_reset_seed(self, mock_llm_client, config):
        """Test resetting random seed."""
        engine = LLMGuidedMCTSEngine(mock_llm_client, config)

        engine.reset_seed(123)

        assert engine.config.seed == 123

    def test_select_single_node(self, mock_llm_client, config):
        """Test selection with single node."""
        engine = LLMGuidedMCTSEngine(mock_llm_client, config)

        root = create_root_node(problem="test", episode_id="ep1")

        selected = engine._select(root)

        assert selected == root

    def test_select_with_children(self, mock_llm_client, config):
        """Test selection with children."""
        engine = LLMGuidedMCTSEngine(mock_llm_client, config)

        root = create_root_node(problem="test", episode_id="ep1")
        root.visits = 10

        child1 = root.add_child(
            state=NodeState(code="code1", problem="test"),
            action="a",
        )
        child1.visits = 5
        child1.value_sum = 2.0

        child2 = root.add_child(
            state=NodeState(code="code2", problem="test"),
            action="b",
        )
        child2.visits = 1
        child2.value_sum = 0.5

        # Child2 should be selected due to higher UCB1 (less explored)
        selected = engine._select(root)
        assert selected == child2

    @pytest.mark.asyncio
    async def test_expand(self, mock_llm_client, config):
        """Test node expansion."""
        mock_llm_client.complete.return_value = json.dumps(
            {
                "variants": [
                    {"code": "def foo(): return 1", "confidence": 0.8},
                    {"code": "def foo(): return int(1)", "confidence": 0.2},
                ]
            }
        )

        engine = LLMGuidedMCTSEngine(mock_llm_client, config)
        node = create_root_node(problem="Return 1", episode_id="ep1")

        children = await engine._expand(node, "ep1")

        assert len(children) == 2
        assert children[0].state.code == "def foo(): return 1"
        assert children[1].state.code == "def foo(): return int(1)"
        assert node.status == NodeStatus.EXPANDED

    @pytest.mark.asyncio
    async def test_expand_terminal_node(self, mock_llm_client, config):
        """Test expansion of terminal node."""
        engine = LLMGuidedMCTSEngine(mock_llm_client, config)
        node = create_root_node(problem="test", episode_id="ep1")
        node.status = NodeStatus.TERMINAL_SUCCESS

        children = await engine._expand(node, "ep1")

        assert children == []

    @pytest.mark.asyncio
    async def test_evaluate_passing_code(self, mock_llm_client, config):
        """Test evaluation of passing code."""
        engine = LLMGuidedMCTSEngine(mock_llm_client, config)

        node = LLMGuidedMCTSNode(
            state=NodeState(
                code="def foo(): return 1",
                problem="Return 1",
                test_cases=["assert foo() == 1"],
            )
        )

        reward, is_solution = await engine._evaluate(node)

        assert is_solution is True
        assert reward == 1.0
        assert node.status == NodeStatus.TERMINAL_SUCCESS

    @pytest.mark.asyncio
    async def test_evaluate_failing_code(self, mock_llm_client, config):
        """Test evaluation of failing code."""
        mock_llm_client.complete.return_value = json.dumps(
            {
                "value": 0.3,
                "reflection": "Code returns wrong value",
                "is_solution": False,
            }
        )

        engine = LLMGuidedMCTSEngine(mock_llm_client, config)

        node = LLMGuidedMCTSNode(
            state=NodeState(
                code="def foo(): return 0",
                problem="Return 1",
                test_cases=["assert foo() == 1"],
            )
        )

        reward, is_solution = await engine._evaluate(node)

        assert is_solution is False
        assert reward < 0  # value 0.3 scaled to [-1, 1]
        assert node.llm_value_estimate == 0.3

    def test_backpropagate(self, mock_llm_client, config):
        """Test reward backpropagation."""
        engine = LLMGuidedMCTSEngine(mock_llm_client, config)

        root = create_root_node(problem="test", episode_id="ep1")
        child = root.add_child(
            state=NodeState(code="code", problem="test"),
            action="a",
        )

        engine._backpropagate(child, 0.5)

        assert child.visits == 1
        assert child.value_sum == 0.5
        assert root.visits == 1
        assert root.value_sum == 0.5

    def test_find_best_node(self, mock_llm_client, config):
        """Test finding best node in tree."""
        engine = LLMGuidedMCTSEngine(mock_llm_client, config)

        root = create_root_node(problem="test", episode_id="ep1")
        root.visits = 10

        child1 = root.add_child(
            state=NodeState(code="code1", problem="test"),
            action="a",
        )
        child1.visits = 5
        child1.value_sum = 2.0  # q = 0.4

        child2 = root.add_child(
            state=NodeState(code="code2", problem="test"),
            action="b",
        )
        child2.visits = 3
        child2.value_sum = 2.4  # q = 0.8

        best = engine._find_best_node(root)

        # child2 should be best due to higher q-value
        assert best == child2

    def test_compute_tree_depth(self, mock_llm_client, config):
        """Test tree depth computation."""
        engine = LLMGuidedMCTSEngine(mock_llm_client, config)

        root = create_root_node(problem="test", episode_id="ep1")
        child1 = root.add_child(
            state=NodeState(code="code1", problem="test"),
            action="a",
        )
        _grandchild = child1.add_child(
            state=NodeState(code="code2", problem="test"),
            action="b",
        )

        depth = engine._compute_tree_depth(root)

        assert depth == 2

    def test_count_nodes(self, mock_llm_client, config):
        """Test node counting."""
        engine = LLMGuidedMCTSEngine(mock_llm_client, config)

        root = create_root_node(problem="test", episode_id="ep1")
        root.add_child(
            state=NodeState(code="code1", problem="test"),
            action="a",
        )
        root.add_child(
            state=NodeState(code="code2", problem="test"),
            action="b",
        )

        count = engine._count_nodes(root)

        assert count == 3

    @pytest.mark.asyncio
    async def test_search_finds_solution(self, mock_llm_client, config):
        """Test that search can find a solution."""
        # Mock generator to return correct code
        # The generator returns code that passes tests, so we should find a solution
        # Use return_value for simplicity since the code passes tests on first try
        mock_llm_client.complete.return_value = json.dumps(
            {
                "variants": [
                    {"code": "def foo(): return 1", "confidence": 0.9},
                ]
            }
        )

        # Configure for faster test with early termination
        config.num_iterations = 5
        config.early_termination_on_solution = True

        engine = LLMGuidedMCTSEngine(mock_llm_client, config)

        result = await engine.search(
            problem="Return 1",
            test_cases=["assert foo() == 1"],
        )

        assert result.solution_found is True
        assert "return 1" in result.best_code
        assert result.num_iterations >= 1

    @pytest.mark.asyncio
    async def test_search_no_solution(self, mock_llm_client, config):
        """Test search when no solution found."""
        # Mock generator to return wrong code
        mock_llm_client.complete.side_effect = [
            json.dumps(
                {
                    "variants": [
                        {"code": "def foo(): return 0", "confidence": 0.5},
                    ]
                }
            ),
            json.dumps(
                {
                    "value": 0.2,
                    "reflection": "Wrong output",
                    "is_solution": False,
                }
            ),
        ] * 10  # Repeat for multiple iterations

        config.num_iterations = 2
        engine = LLMGuidedMCTSEngine(mock_llm_client, config)

        result = await engine.search(
            problem="Return 1",
            test_cases=["assert foo() == 1"],
        )

        assert result.solution_found is False
        assert result.num_iterations <= config.num_iterations

    def test_get_statistics(self, mock_llm_client, config):
        """Test getting engine statistics."""
        engine = LLMGuidedMCTSEngine(mock_llm_client, config)

        stats = engine.get_statistics()

        assert "total_searches" in stats
        assert "total_solutions" in stats
        assert "success_rate" in stats
        assert "generator_stats" in stats
        assert "reflector_stats" in stats
        assert "config" in stats


class TestCreateLLMMCTSEngine:
    """Tests for create_llm_mcts_engine factory."""

    def test_create_with_preset(self):
        """Test creating engine with preset."""
        mock_client = AsyncMock()

        engine = create_llm_mcts_engine(
            mock_client,
            preset=LLMGuidedMCTSPreset.FAST,
        )

        assert engine.config.name == "fast"
        assert engine.config.num_iterations == 10

    def test_create_with_overrides(self):
        """Test creating engine with overrides."""
        mock_client = AsyncMock()

        engine = create_llm_mcts_engine(
            mock_client,
            preset=LLMGuidedMCTSPreset.BALANCED,
            num_iterations=50,
            name="custom",
        )

        assert engine.config.name == "custom"
        assert engine.config.num_iterations == 50


class TestLangGraphIntegration:
    """Tests for LangGraph integration."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = AsyncMock()
        client.complete = AsyncMock()
        return client

    def test_build_langgraph(self, mock_llm_client):
        """Test building LangGraph state machine."""
        config = LLMGuidedMCTSConfig(num_iterations=5)
        engine = LLMGuidedMCTSEngine(mock_llm_client, config)

        try:
            graph = engine.build_langgraph()
            assert graph is not None
        except ImportError:
            pytest.skip("LangGraph not installed")

    @pytest.mark.asyncio
    async def test_search_with_langgraph(self, mock_llm_client):
        """Test search using LangGraph orchestration."""
        mock_llm_client.complete.side_effect = [
            json.dumps(
                {
                    "variants": [
                        {"code": "def foo(): return 1", "confidence": 0.9},
                    ]
                }
            ),
        ]

        config = LLMGuidedMCTSConfig(num_iterations=3)
        engine = LLMGuidedMCTSEngine(mock_llm_client, config)

        try:
            result = await engine.search_with_langgraph(
                problem="Return 1",
                test_cases=["assert foo() == 1"],
            )
            assert result is not None
        except ImportError:
            pytest.skip("LangGraph not installed")
