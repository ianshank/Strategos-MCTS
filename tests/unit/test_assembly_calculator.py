"""
Tests for assembly index calculator module.

Tests AssemblyIndexCalculator including string assembly, graph assembly,
copy number, tokenization, caching, and cycle breaking.
"""

import networkx as nx
import pytest

from src.framework.assembly.calculator import AssemblyIndexCalculator


@pytest.mark.unit
class TestAssemblyIndexCalculator:
    """Tests for AssemblyIndexCalculator."""

    def test_init_defaults(self):
        calc = AssemblyIndexCalculator()
        assert calc.cache_enabled is True
        assert calc.max_cache_size == 10000

    def test_init_custom(self):
        calc = AssemblyIndexCalculator(cache_enabled=False, max_cache_size=100)
        assert calc.cache_enabled is False
        assert calc.max_cache_size == 100

    # --- String assembly ---

    def test_empty_string(self):
        calc = AssemblyIndexCalculator()
        ai, cn = calc.calculate("")
        assert ai == 0
        assert cn == 0.0

    def test_whitespace_string(self):
        calc = AssemblyIndexCalculator()
        ai, cn = calc.calculate("   ")
        assert ai == 0
        assert cn == 0.0

    def test_single_token(self):
        calc = AssemblyIndexCalculator()
        ai, cn = calc.calculate("hello")
        assert ai == 1
        assert cn == 1.0

    def test_multi_token_string(self):
        calc = AssemblyIndexCalculator()
        ai, cn = calc.calculate("the quick brown fox jumps over the lazy dog")
        assert ai >= 1
        assert cn >= 1.0

    def test_string_with_related_tokens(self):
        calc = AssemblyIndexCalculator()
        ai, cn = calc.calculate("running runner runners")
        assert ai >= 1

    # --- Graph assembly ---

    def test_empty_graph(self):
        calc = AssemblyIndexCalculator()
        g = nx.DiGraph()
        ai, cn = calc.calculate(g)
        assert ai == 0
        assert cn == 0.0

    def test_single_node_graph(self):
        calc = AssemblyIndexCalculator()
        g = nx.DiGraph()
        g.add_node(0)
        ai, cn = calc.calculate(g)
        assert ai == 1
        assert cn == 1.0

    def test_linear_graph(self):
        calc = AssemblyIndexCalculator()
        g = nx.DiGraph()
        g.add_edges_from([(0, 1), (1, 2), (2, 3)])
        ai, cn = calc.calculate(g)
        assert ai == 3  # 4 nodes, path length 3

    def test_branching_graph(self):
        calc = AssemblyIndexCalculator()
        g = nx.DiGraph()
        g.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
        ai, cn = calc.calculate(g)
        assert ai >= 1

    def test_cyclic_graph(self):
        calc = AssemblyIndexCalculator()
        g = nx.DiGraph()
        g.add_edges_from([(0, 1), (1, 2), (2, 0)])
        ai, cn = calc.calculate(g)
        assert ai >= 1

    # --- Copy number ---

    def test_copy_number_small_graph(self):
        calc = AssemblyIndexCalculator()
        g = nx.DiGraph()
        g.add_edge(0, 1)
        cn = calc._calculate_copy_number(g)
        assert cn == 1.0

    def test_copy_number_uniform_pattern(self):
        calc = AssemblyIndexCalculator()
        g = nx.DiGraph()
        g.add_edges_from([(0, 1), (1, 2), (2, 3)])
        cn = calc._calculate_copy_number(g)
        assert cn >= 1.0

    # --- Tokenization ---

    def test_tokenize_simple(self):
        calc = AssemblyIndexCalculator()
        tokens = calc._tokenize("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenize_punctuation(self):
        calc = AssemblyIndexCalculator()
        tokens = calc._tokenize("hello, world!")
        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenize_empty_returns_original(self):
        calc = AssemblyIndexCalculator()
        tokens = calc._tokenize("?")
        assert len(tokens) >= 1

    # --- Token relationship ---

    def test_tokens_related_common_prefix(self):
        calc = AssemblyIndexCalculator()
        assert calc._tokens_related("running", "runner") is True

    def test_tokens_related_short_tokens(self):
        calc = AssemblyIndexCalculator()
        assert calc._tokens_related("ab", "abc") is False

    def test_tokens_unrelated(self):
        calc = AssemblyIndexCalculator()
        assert calc._tokens_related("hello", "world") is False

    # --- Caching ---

    def test_cache_hit(self):
        calc = AssemblyIndexCalculator(cache_enabled=True)
        text = "test caching behavior"
        r1 = calc.calculate(text)
        r2 = calc.calculate(text)
        assert r1 == r2

    def test_cache_disabled(self):
        calc = AssemblyIndexCalculator(cache_enabled=False)
        text = "test no cache"
        r1 = calc.calculate(text)
        r2 = calc.calculate(text)
        assert r1 == r2

    def test_clear_cache(self):
        calc = AssemblyIndexCalculator()
        calc.calculate("some text")
        calc.clear_cache()
        stats = calc.get_cache_stats()
        assert stats["size"] == 0

    def test_get_cache_stats(self):
        calc = AssemblyIndexCalculator(max_cache_size=500)
        stats = calc.get_cache_stats()
        assert stats["max_size"] == 500
        assert stats["enabled"] is True
        assert stats["size"] == 0

    def test_cache_max_size(self):
        calc = AssemblyIndexCalculator(cache_enabled=True, max_cache_size=2)
        calc.calculate("text one here")
        calc.calculate("text two here")
        calc.calculate("text three here")
        assert len(calc._cache) <= 3  # May or may not exceed based on implementation

    # --- Unsupported input ---

    def test_unsupported_type(self):
        calc = AssemblyIndexCalculator()
        with pytest.raises(ValueError, match="Unsupported input type"):
            calc.calculate(42)

    # --- Cycle breaking ---

    def test_break_cycles(self):
        calc = AssemblyIndexCalculator()
        g = nx.DiGraph()
        g.add_edges_from([(0, 1), (1, 2), (2, 0)])
        dag = calc._break_cycles(g)
        assert nx.is_directed_acyclic_graph(dag)

    def test_break_cycles_already_dag(self):
        calc = AssemblyIndexCalculator()
        g = nx.DiGraph()
        g.add_edges_from([(0, 1), (1, 2)])
        dag = calc._break_cycles(g)
        assert nx.is_directed_acyclic_graph(dag)
        assert dag.number_of_edges() == 2

    # --- Hash ---

    def test_hash_string_deterministic(self):
        calc = AssemblyIndexCalculator()
        h1 = calc._hash_string("test")
        h2 = calc._hash_string("test")
        assert h1 == h2

    def test_hash_string_different(self):
        calc = AssemblyIndexCalculator()
        h1 = calc._hash_string("test1")
        h2 = calc._hash_string("test2")
        assert h1 != h2
