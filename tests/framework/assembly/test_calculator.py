"""
Tests for Assembly Index Calculator (Story 1.1).
"""

import pytest

nx = pytest.importorskip("networkx", reason="networkx required for assembly tests")

from src.framework.assembly.calculator import AssemblyIndexCalculator


class TestAssemblyIndexCalculator:
    """Test suite for AssemblyIndexCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return AssemblyIndexCalculator(cache_enabled=True)

    def test_empty_string(self, calculator):
        """Test with empty string."""
        assembly_index, copy_number = calculator.calculate("")
        assert assembly_index == 0
        assert copy_number == 0.0

    def test_simple_query(self, calculator):
        """Test with simple query."""
        query = "What is 2+2?"
        assembly_index, copy_number = calculator.calculate(query)

        assert assembly_index >= 1
        assert assembly_index <= 5  # Simple query should have low index
        assert copy_number >= 1.0

    def test_complex_query(self, calculator):
        """Test with complex query."""
        query = "Design a microservices architecture for a scalable e-commerce platform with real-time inventory management and distributed caching"
        assembly_index, copy_number = calculator.calculate(query)

        assert assembly_index >= 5  # Complex query should have higher index
        assert copy_number >= 1.0

    def test_single_node_graph(self, calculator):
        """Test with single-node graph."""
        graph = nx.DiGraph()
        graph.add_node(0)

        assembly_index, copy_number = calculator.calculate(graph)

        assert assembly_index == 1
        assert copy_number == 1.0

    def test_linear_graph(self, calculator):
        """Test with linear dependency graph."""
        graph = nx.DiGraph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 3)])

        assembly_index, copy_number = calculator.calculate(graph)

        assert assembly_index == 3  # Path length - 1
        assert copy_number >= 1.0

    def test_cyclic_graph(self, calculator):
        """Test with cyclic graph (should handle gracefully)."""
        graph = nx.DiGraph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Cycle

        # Should not raise error
        assembly_index, copy_number = calculator.calculate(graph)

        assert assembly_index >= 1
        assert copy_number >= 1.0

    def test_cache_functionality(self, calculator):
        """Test caching works."""
        query = "Test query for caching"

        # First call
        result1 = calculator.calculate(query)

        # Second call (should hit cache)
        result2 = calculator.calculate(query)

        assert result1 == result2

        # Check cache stats
        stats = calculator.get_cache_stats()
        assert stats["size"] > 0

    def test_invalid_input(self, calculator):
        """Test with invalid input type."""
        with pytest.raises(ValueError):
            calculator.calculate(123)  # Invalid type

    def test_monotonicity(self, calculator):
        """Test that assembly index increases with complexity."""
        simple = "hello"
        medium = "hello world how are you"
        complex_query = "hello world how are you doing today with all your complex dependencies and relationships"

        idx_simple, _ = calculator.calculate(simple)
        idx_medium, _ = calculator.calculate(medium)
        idx_complex, _ = calculator.calculate(complex_query)

        # Generally, more complex queries should have higher indices
        # (This is a statistical test, not strict)
        assert idx_complex >= idx_simple

    def test_copy_number_calculation(self, calculator):
        """Test copy number reflects reuse."""
        # Graph with repeated patterns (same degree distributions)
        graph = nx.DiGraph()
        # Create multiple nodes with same pattern
        for i in range(3):
            base = i * 3
            graph.add_edges_from(
                [
                    (base, base + 1),
                    (base, base + 2),
                ]
            )

        assembly_index, copy_number = calculator.calculate(graph)

        # Should detect pattern reuse
        assert copy_number >= 1.0

    def test_clear_cache(self, calculator):
        """Test cache clearing."""
        calculator.calculate("test query")
        assert calculator.get_cache_stats()["size"] > 0

        calculator.clear_cache()
        assert calculator.get_cache_stats()["size"] == 0


@pytest.mark.parametrize(
    "query,expected_min,expected_max",
    [
        ("simple", 1, 3),
        ("more complex query with dependencies", 3, 8),
        ("design microservices api database cache queue async", 5, 12),
    ],
)
def test_assembly_index_ranges(query, expected_min, expected_max):
    """Test assembly indices fall in expected ranges."""
    calculator = AssemblyIndexCalculator()
    assembly_index, _ = calculator.calculate(query)

    assert expected_min <= assembly_index <= expected_max
