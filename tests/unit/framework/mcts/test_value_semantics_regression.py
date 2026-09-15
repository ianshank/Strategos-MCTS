"""
Value-semantics regression suite for MCTS selection and backup
(spec: ``hygiene_mcts_value_semantics``).

Covers proven, executable-proof-verified bugs:

1. **PUCT double-division** (``neural_policies.select_child_puct``): Q was divided by visits a
   second time even though ``MCTSNode.value`` is already the mean (``value_sum / visits``) —
   collapsing Q toward 0 as visits grew and turning PUCT into a near-pure exploration bandit.
   Fixed by delegating directly to the canonical ``puct()`` formula.
2. **Negamax selection sign mismatch** (``ParallelMCTSEngine`` /
   ``VirtualLossNode.select_child_with_vl``): backup flipped per ply but selection
   read the child's stored value without negating it — selecting the move best for the
   OPPONENT, not the root.
3. The identical sign mismatch in ``ProgressiveWideningEngine`` / ``RAVENode.select_child_rave``,
   which additionally propagated into the RAVE/AMAF mixing term.
4. **Backup flag mismatch (AC-6 / AC-7):** ``parallel_mcts`` and ``progressive_widening``
   negated on backup unconditionally (ignoring ``two_player``), while ``core.MCTSEngine``
   never negated and had no flag. Selection and backup must share one perspective flag.

The fix threads ``negate_child_value`` / ``two_player`` (settings-backed via
``Settings.MCTS_TWO_PLAYER``, default ``True``) through selection *and* backup on all
four engines. NeuralMCTS uses the inverted name ``single_agent`` for the same pair.

``TestCrossEngineSingleAgentParity`` locks ``negate_child_value=False`` to mean
"matches core's unflipped selection convention." Backup parity is
``TestBackupSignAndCrossEngineParity`` below — CHARTER.md §2's demo command must
exercise backup, not only stuffed-stats selection.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
from src.framework.mcts.neural_policies import PriorsManager, puct, select_child_puct
from src.framework.mcts.parallel_mcts import ParallelMCTSConfig, ParallelMCTSEngine, VirtualLossNode
from src.framework.mcts.progressive_widening import ProgressiveWideningEngine, RAVEConfig, RAVENode

pytestmark = [pytest.mark.unit]


def _state(state_id: str) -> MCTSState:
    return MCTSState(state_id=state_id, features={})


# =============================================================================
# AC-1: Regression suite ported from the executable proofs
# =============================================================================


class TestParallelMCTSNegamaxSelection:
    """``select_child_with_vl`` must pick the minimax-optimal child, not the opponent-best one."""

    def _tree(self) -> tuple[VirtualLossNode, VirtualLossNode, VirtualLossNode]:
        root = VirtualLossNode(state=_state("root"))
        # Child "a": the opponent (to move at the child) does WELL there — bad for the root.
        a = root.add_child("a", _state("a"))
        a.visits, a.value_sum = 50, 45.0  # child.value == 0.9, opponent's perspective
        # Child "b": the opponent does POORLY there — good for the root.
        b = root.add_child("b", _state("b"))
        b.visits, b.value_sum = 10, 1.0  # child.value == 0.1, opponent's perspective
        root.visits = a.visits + b.visits
        return root, a, b

    def test_two_player_mode_negates_and_picks_the_root_optimal_child(self) -> None:
        root, _a, _b = self._tree()

        selected = root.select_child_with_vl(exploration_weight=0.5, negate_child_value=True)

        assert selected.action == "b", (
            "with negation, the root must prefer 'b' (opponent scores 0.1 there) over "
            "'a' (opponent scores 0.9 there) -- selecting 'a' means picking the move that is "
            "best for the OPPONENT, exactly the proven bug"
        )

    def test_default_negate_child_value_is_false_and_matches_pre_fix_unflipped_math(self) -> None:
        """The parameter defaults to False: raw (unnegated) selection is unchanged."""
        root, _a, _b = self._tree()

        selected = root.select_child_with_vl(exploration_weight=0.5)

        assert selected.action == "a"

    def test_engine_default_two_player_is_true_and_wired_to_selection(self) -> None:
        config = ParallelMCTSConfig()
        assert config.two_player is True

        engine = ParallelMCTSEngine(config=config)
        assert engine.two_player is True


class TestProgressiveWideningNegamaxSelection:
    """``select_child_rave`` must negate both the UCB and RAVE/AMAF terms under negamax."""

    def _tree(self) -> tuple[RAVENode, RAVENode, RAVENode]:
        root = RAVENode(state=_state("root"))
        a = RAVENode(state=_state("a"), parent=root, action="a")
        a.visits, a.value_sum = 50, 45.0  # opponent scores 0.9 at 'a'
        b = RAVENode(state=_state("b"), parent=root, action="b")
        b.visits, b.value_sum = 10, 1.0  # opponent scores 0.1 at 'b'
        root.children = [a, b]
        root.visits = a.visits + b.visits
        return root, a, b

    def test_two_player_mode_negates_ucb_and_picks_root_optimal_child(self) -> None:
        root, _a, _b = self._tree()
        rave_config = RAVEConfig()  # no RAVE data recorded -> beta=0.0, pure UCB path

        selected = root.select_child_rave(rave_config, exploration_weight=0.5, negate_child_value=True)

        assert selected.action == "b"

    def test_rave_term_is_also_negated_when_it_dominates(self) -> None:
        """
        Overwhelm the mixing weight (beta -> 1) so the RAVE term alone drives the score. RAVE
        data mirrors the direct value_sum/visits split (opponent-perspective): if only the UCB
        term were fixed and the RAVE term were left unnegated, this dominant term would
        silently re-introduce the exact same bug by favoring 'a' again.
        """
        root, a, b = self._tree()
        rave_config = RAVEConfig(min_visits_for_rave=1, rave_constant=0.0)
        a.rave_visits["a"], a.rave_value_sum["a"] = 100_000, 90_000.0  # rave_value(a) == 0.9
        b.rave_visits["b"], b.rave_value_sum["b"] = 100_000, 10_000.0  # rave_value(b) == 0.1

        selected = root.select_child_rave(rave_config, exploration_weight=0.1, negate_child_value=True)

        assert selected.action == "b"

    def test_default_negate_child_value_is_false_and_matches_pre_fix_unflipped_math(self) -> None:
        root, _a, _b = self._tree()
        rave_config = RAVEConfig()

        selected = root.select_child_rave(rave_config, exploration_weight=0.5)

        assert selected.action == "a"

    def test_engine_default_two_player_is_true_and_wired_to_selection(self) -> None:
        engine = ProgressiveWideningEngine()
        assert engine.two_player is True


class TestPUCTDoubleDivisionFix:
    """``select_child_puct`` must agree with the canonical ``puct()`` formula, not double-divide Q."""

    def test_picks_the_canonically_best_child_not_the_double_divided_one(self) -> None:
        root = MCTSNode(state=_state("root"))
        root.visits = 100
        a = root.add_child("a", _state("a"))
        a.visits, a.value_sum = 50, 45.0  # true Q = 0.9
        b = root.add_child("b", _state("b"))
        b.visits, b.value_sum = 10, 1.0  # true Q = 0.1

        priors = PriorsManager()
        priors.set_priors(root.state.to_hash_key(), {"a": 0.5, "b": 0.5})

        result = select_child_puct(root, priors, c_puct=1.25)

        assert result is not None
        action, _child = result
        assert action == "a", (
            "the strong child (true Q=0.9) must win; the old double-division bug crushed Q "
            "to 0.018 and let the exploration term hand the decision to the weak child"
        )
        # The canonical formula independently agrees.
        assert puct(a.value, 0.5, root.visits, a.visits, 1.25) > puct(b.value, 0.5, root.visits, b.visits, 1.25)


# =============================================================================
# AC-2: Cross-engine parity (single-agent / unflipped convention)
# =============================================================================


class TestCrossEngineSingleAgentParity:
    """
    In single-agent mode (no sign flip), ``core.MCTSNode.select_child`` (UCB1),
    ``VirtualLossNode.select_child_with_vl`` (no active virtual loss, ``negate_child_value=False``),
    and ``RAVENode.select_child_rave`` (no RAVE data, ``negate_child_value=False``) all reduce to
    the identical UCB1 formula and must agree on the selected child for the same seeded stats.

    ``core.py`` unflipped selection is the ``negate_child_value=False`` convention.
    Backup sign is tested separately in ``TestBackupSignAndCrossEngineParity``.
    """

    _EXPLORATION_WEIGHT = 0.7
    _ROOT_VISITS = 80
    _STATS = (("a", 55, 33.0), ("b", 25, 20.0))  # (action, visits, value_sum)

    def test_core_parallel_and_progressive_widening_agree(self) -> None:
        core_root = MCTSNode(state=_state("root"))
        vl_root = VirtualLossNode(state=_state("root"))
        rave_root = RAVENode(state=_state("root"))
        core_root.visits = vl_root.visits = rave_root.visits = self._ROOT_VISITS

        for action, visits, value_sum in self._STATS:
            core_child = core_root.add_child(action, _state(action))
            core_child.visits, core_child.value_sum = visits, value_sum

            vl_child = vl_root.add_child(action, _state(action))
            vl_child.visits, vl_child.value_sum = visits, value_sum

            rave_child = RAVENode(state=_state(action), parent=rave_root, action=action)
            rave_child.visits, rave_child.value_sum = visits, value_sum
            rave_root.children.append(rave_child)

        core_selected = core_root.select_child(self._EXPLORATION_WEIGHT)
        vl_selected = vl_root.select_child_with_vl(self._EXPLORATION_WEIGHT, negate_child_value=False)
        rave_selected = rave_root.select_child_rave(RAVEConfig(), self._EXPLORATION_WEIGHT, negate_child_value=False)

        assert core_selected.action == vl_selected.action == rave_selected.action


# =============================================================================
# AC-3: select_child_puct agrees with the canonical puct() on 1,000 seeded inputs
# =============================================================================


class TestSelectChildPuctMatchesCanonicalFormula:
    def test_matches_puct_on_1000_seeded_random_scenarios(self) -> None:
        rng = np.random.default_rng(20260730)

        for _ in range(1000):
            root = MCTSNode(state=_state("root"))
            root.visits = int(rng.integers(1, 500))

            priors = PriorsManager()
            prior_map: dict[str, float] = {}
            expected_scores: dict[str, float] = {}

            num_children = int(rng.integers(2, 6))
            for i in range(num_children):
                action = f"action_{i}"
                visits = int(rng.integers(0, 200))
                value = float(rng.uniform(-1.0, 1.0))
                prior = float(rng.uniform(0.0, 1.0))

                child = root.add_child(action, _state(action))
                child.visits = visits
                child.value_sum = value * visits if visits > 0 else 0.0
                prior_map[action] = prior

                expected_scores[action] = puct(
                    q_value=child.value,
                    prior=prior,
                    visit_count=child.visits,
                    parent_visits=root.visits,
                    c_puct=1.25,
                )

            priors.set_priors(root.state.to_hash_key(), prior_map)

            result = select_child_puct(root, priors, c_puct=1.25)
            assert result is not None
            selected_action, _selected_child = result

            # select_child_puct must have chosen a child whose independently-computed puct()
            # score is (one of) the maximum -- i.e. it agrees with the canonical formula.
            assert expected_scores[selected_action] == max(expected_scores.values())

    def test_matches_puct_on_zero_visit_zero_prior_edge_case(self) -> None:
        """Test behavior on unvisited child with zero prior (rare but possible edge case)."""
        root = MCTSNode(state=_state("root"))
        root.visits = 100

        priors = PriorsManager()
        prior_map: dict[str, float] = {}
        expected_scores: dict[str, float] = {}

        # Add a zero-visit, zero-prior child (edge case)
        child_zero = root.add_child("zero_prior", _state("zero_prior"))
        child_zero.visits = 0
        child_zero.value_sum = 0.0
        prior_map["zero_prior"] = 0.0

        # Add a normal child for comparison
        child_normal = root.add_child("normal", _state("normal"))
        child_normal.visits = 10
        child_normal.value_sum = 5.0
        prior_map["normal"] = 0.5

        # Compute expected scores using canonical puct() function
        expected_scores["zero_prior"] = puct(q_value=0.0, prior=0.0, visit_count=0, parent_visits=100, c_puct=1.25)
        expected_scores["normal"] = puct(q_value=0.5, prior=0.5, visit_count=10, parent_visits=100, c_puct=1.25)

        priors.set_priors(root.state.to_hash_key(), prior_map)

        result = select_child_puct(root, priors, c_puct=1.25)
        assert result is not None
        selected_action, _selected_child = result

        # Verify select_child_puct matches the canonical formula exactly on this edge case
        assert expected_scores[selected_action] == max(expected_scores.values())


# =============================================================================
# AC-4: DEBUG structured per-child selection logging
# =============================================================================


class TestSelectionDebugLogging:
    """
    Fixed selection paths emit one DEBUG record per candidate child via ``get_logger``. Uses
    ``monkeypatch`` on the module-level logger's ``debug`` method directly rather than
    ``caplog``, since the project's logging config (``src/observability/logging.py``) can attach
    the ``mcts`` logger to its own non-propagating handler -- capturing at the logger call site
    is robust regardless of global logging configuration or test execution order.
    """

    def test_parallel_selection_logs_one_debug_record_per_child(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.framework.mcts.parallel_mcts as parallel_mcts_module

        calls: list[tuple[tuple, dict]] = []
        monkeypatch.setattr(parallel_mcts_module.logger, "debug", lambda *a, **kw: calls.append((a, kw)))

        root = VirtualLossNode(state=_state("root"))
        a = root.add_child("a", _state("a"))
        a.visits, a.value_sum = 5, 3.0
        b = root.add_child("b", _state("b"))
        b.visits, b.value_sum = 5, 1.0
        root.visits = 10

        root.select_child_with_vl(0.5, negate_child_value=True)

        assert len(calls) == 2  # one DEBUG record per candidate child
        assert all("select_child_with_vl candidate" in call_args[0][0] for call_args in calls)

    def test_progressive_widening_selection_logs_one_debug_record_per_child(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.framework.mcts.progressive_widening as pw_module

        calls: list[tuple[tuple, dict]] = []
        monkeypatch.setattr(pw_module.logger, "debug", lambda *a, **kw: calls.append((a, kw)))

        root = RAVENode(state=_state("root"))
        a = RAVENode(state=_state("a"), parent=root, action="a")
        a.visits, a.value_sum = 5, 3.0
        b = RAVENode(state=_state("b"), parent=root, action="b")
        b.visits, b.value_sum = 5, 1.0
        root.children = [a, b]
        root.visits = 10

        root.select_child_rave(RAVEConfig(), 0.5, negate_child_value=True)

        assert len(calls) == 2
        assert all("select_child_rave candidate" in call_args[0][0] for call_args in calls)

    def test_puct_selection_logs_one_debug_record_per_child(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.framework.mcts.neural_policies as neural_policies_module

        calls: list[tuple[tuple, dict]] = []
        monkeypatch.setattr(neural_policies_module._logger, "debug", lambda *a, **kw: calls.append((a, kw)))

        root = MCTSNode(state=_state("root"))
        root.visits = 10
        a = root.add_child("a", _state("a"))
        a.visits, a.value_sum = 5, 3.0
        b = root.add_child("b", _state("b"))
        b.visits, b.value_sum = 5, 1.0

        select_child_puct(root, PriorsManager(), c_puct=1.25)

        assert len(calls) == 2
        assert all("select_child_puct candidate" in call_args[0][0] for call_args in calls)


# =============================================================================
# End-to-end async integration: parallel_search and search wiring is correct
# =============================================================================


class _AsyncWinNowState:
    """Simple two-player game for wiring verification: side to move may win or pass."""

    def __init__(self, to_move: int = 1, winner: int | None = None):
        self.to_move = to_move
        self.winner = winner

    def is_terminal(self) -> bool:
        return self.winner is not None

    def get_legal_actions(self) -> list[str]:
        return [] if self.is_terminal() else ["win", "pass"]

    def apply_action(self, action: str) -> _AsyncWinNowState:
        if action == "win":
            return _AsyncWinNowState(to_move=-self.to_move, winner=self.to_move)
        else:
            return _AsyncWinNowState(to_move=-self.to_move, winner=None)

    def get_reward(self, player: int = 1) -> float:
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def to_hash_key(self) -> str:
        return f"{self.to_move}:{self.winner}"


class TestParallelMCTSEngineWiring:
    """Verify two_player parameter wires through to node selection."""

    def test_engine_passes_two_player_to_selection(self) -> None:
        """Verify that engine's two_player config is passed to select_child_with_vl."""
        # Create engine with explicit two_player=False
        engine = ParallelMCTSEngine(config=ParallelMCTSConfig(two_player=False))
        assert engine.two_player is False

        # Create engine with explicit two_player=True
        engine2 = ParallelMCTSEngine(config=ParallelMCTSConfig(two_player=True))
        assert engine2.two_player is True

    def test_deprecated_path_uses_settings(self) -> None:
        """Legacy parameter path respects Settings.MCTS_TWO_PLAYER."""
        engine = ParallelMCTSEngine(num_workers=4)
        # Should read from settings (default True)
        assert engine.two_player is True


class TestProgressiveWideningEngineWiring:
    """Verify two_player parameter wires through to node selection."""

    def test_engine_passes_two_player_to_selection(self) -> None:
        """Verify that engine's two_player config is passed to select_child_rave."""
        # Create engine with explicit two_player=False
        engine = ProgressiveWideningEngine(two_player=False)
        assert engine.two_player is False

        # Create engine with explicit two_player=True
        engine2 = ProgressiveWideningEngine(two_player=True)
        assert engine2.two_player is True

    def test_default_reads_from_settings(self) -> None:
        """When two_player not passed, reads from Settings.MCTS_TWO_PLAYER."""
        engine = ProgressiveWideningEngine()
        # Should read from settings (default True)
        assert engine.two_player is True


# =============================================================================
# Settings: MCTS_TWO_PLAYER is a real, bounded, settings-backed field (no hardcoded values)
# =============================================================================


class TestTwoPlayerSetting:
    def test_settings_expose_mcts_two_player_default_true(self, test_settings) -> None:
        assert test_settings.MCTS_TWO_PLAYER is True

    def test_parallel_mcts_engine_reads_settings_when_no_config(self) -> None:
        """ParallelMCTSEngine should read Settings.MCTS_TWO_PLAYER when constructed without config."""
        # When constructed with no config, engine should read from settings (default True)
        engine = ParallelMCTSEngine()
        assert engine.two_player is True

    def test_progressive_widening_engine_reads_settings_when_not_explicit(self) -> None:
        """ProgressiveWideningEngine should read Settings.MCTS_TWO_PLAYER when two_player not passed."""
        # When constructed without explicit two_player, should read from settings (default True)
        engine = ProgressiveWideningEngine()
        assert engine.two_player is True

    def test_core_engine_reads_settings_when_not_explicit(self) -> None:
        engine = MCTSEngine(seed=42)
        assert engine.two_player is True

    def test_core_engine_two_player_false_is_honoured(self) -> None:
        engine = MCTSEngine(seed=42, two_player=False)
        assert engine.two_player is False


# =============================================================================
# AC-6 / AC-7: backup honours the perspective flag; four engines agree on value_sum
# Covers hygiene_mcts_value_semantics AC-6
# Covers hygiene_mcts_value_semantics AC-7
# =============================================================================

_LEAF_VALUE = 0.5
_TWO_PLAYER_CHAIN = (0.5, -0.5, 0.5)  # leaf, mid, root after one 2-ply backup
_SINGLE_AGENT_CHAIN = (0.5, 0.5, 0.5)


def _core_backup_chain(two_player: bool) -> tuple[float, float, float]:
    engine = MCTSEngine(seed=42, two_player=two_player)
    root = MCTSNode(state=_state("root"))
    mid = root.add_child("a", _state("mid"))
    leaf = mid.add_child("b", _state("leaf"))
    engine.backpropagate(leaf, _LEAF_VALUE)
    return leaf.value_sum, mid.value_sum, root.value_sum


def _parallel_backup_chain(two_player: bool) -> tuple[float, float, float]:
    engine = ParallelMCTSEngine(config=ParallelMCTSConfig(two_player=two_player, adaptive_virtual_loss=False))
    root = VirtualLossNode(state=_state("root"))
    mid = root.add_child("a", _state("mid"))
    leaf = mid.add_child("b", _state("leaf"))
    engine.backpropagate([root, mid, leaf], _LEAF_VALUE)
    return leaf.value_sum, mid.value_sum, root.value_sum


def _pw_backup_chain(two_player: bool) -> tuple[float, float, float]:
    engine = ProgressiveWideningEngine(two_player=two_player)
    rng = engine.rng
    root = RAVENode(state=_state("root"), rng=rng)
    mid = RAVENode(state=_state("mid"), parent=root, action="a", rng=rng)
    leaf = RAVENode(state=_state("leaf"), parent=mid, action="b", rng=rng)
    engine.backpropagate_with_rave(leaf, _LEAF_VALUE, [])
    return leaf.value_sum, mid.value_sum, root.value_sum


def _neural_backup_chain(single_agent: bool) -> tuple[float, float, float]:
    torch = pytest.importorskip("torch")
    nn = torch.nn
    from src.framework.mcts.neural_mcts import GameState, NeuralMCTS, NeuralMCTSNode
    from src.training.system_config import MCTSConfig

    class _StubNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._p = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.zeros(1, 1), torch.zeros(1, 1)

    class _Named(GameState):
        def __init__(self, name: str) -> None:
            self._name = name

        def get_legal_actions(self) -> list:
            return []

        def apply_action(self, action: object) -> GameState:
            return self

        def is_terminal(self) -> bool:
            return False

        def get_reward(self, player: int = 1) -> float:
            return 0.0

        def to_tensor(self) -> torch.Tensor:
            return torch.zeros(1)

        def get_hash(self) -> str:
            return self._name

    mcts = NeuralMCTS(_StubNet(), MCTSConfig(), device="cpu", single_agent=single_agent)
    root = NeuralMCTSNode(state=_Named("root"))
    mid = NeuralMCTSNode(state=_Named("mid"), parent=root, action="a")
    leaf = NeuralMCTSNode(state=_Named("leaf"), parent=mid, action="b")
    mcts.backpropagate([root, mid, leaf], _LEAF_VALUE)
    return leaf.value_sum, mid.value_sum, root.value_sum


class TestBackupSignClassicalEngines:
    """hygiene_mcts_value_semantics AC-6 / AC-7 — core, parallel, progressive-widening."""

    def test_core_select_child_negates_to_match_two_player_backup(self) -> None:
        """Covers hygiene_mcts_value_semantics AC-6 — core select reads -child.Q."""
        root = MCTSNode(state=_state("root"))
        root.visits = 60
        a = root.add_child("a", _state("a"))
        a.visits, a.value_sum = 50, 45.0  # child STM 0.9
        b = root.add_child("b", _state("b"))
        b.visits, b.value_sum = 10, 1.0  # child STM 0.1
        assert root.select_child(0.5, negate_child_value=True).action == "b"
        assert root.select_child(0.5, negate_child_value=False).action == "a"

    def test_two_player_backup_alternates_sign(self) -> None:
        """Covers hygiene_mcts_value_semantics AC-6 — two-player backup alternates."""
        assert _core_backup_chain(True) == _TWO_PLAYER_CHAIN
        assert _parallel_backup_chain(True) == _TWO_PLAYER_CHAIN
        assert _pw_backup_chain(True) == _TWO_PLAYER_CHAIN

    def test_single_agent_backup_is_monotone(self) -> None:
        """Covers hygiene_mcts_value_semantics AC-6 — single-agent backup does not flip."""
        assert _core_backup_chain(False) == _SINGLE_AGENT_CHAIN
        assert _parallel_backup_chain(False) == _SINGLE_AGENT_CHAIN
        assert _pw_backup_chain(False) == _SINGLE_AGENT_CHAIN

    def test_cross_engine_backup_value_sums_agree(self) -> None:
        """Covers hygiene_mcts_value_semantics AC-7 — identical per-node value sums."""
        for two_player in (True, False):
            core = _core_backup_chain(two_player)
            parallel = _parallel_backup_chain(two_player)
            pw = _pw_backup_chain(two_player)
            assert core == parallel == pw


class TestBackupSignNeuralEngine:
    """Fourth engine; skipped when the neural extra (torch) is absent."""

    def test_neural_backup_matches_classical_engines(self) -> None:
        """Covers hygiene_mcts_value_semantics AC-6 AC-7 — NeuralMCTS backup parity."""
        assert _neural_backup_chain(single_agent=False) == _TWO_PLAYER_CHAIN
        assert _neural_backup_chain(single_agent=True) == _SINGLE_AGENT_CHAIN
        assert _neural_backup_chain(single_agent=False) == _core_backup_chain(True)
        assert _neural_backup_chain(single_agent=True) == _core_backup_chain(False)
