"""Hygienic NeuralMCTS collector: π_train = visit/sum, STM z, eval() during search."""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

import numpy as np
from torch import nn

from scripts.local_distillation.device import place_network
from scripts.local_distillation.policy_targets import play_temperature, visits_to_policy
from scripts.local_distillation.schema import TrajectoryRow, validate_row
from scripts.local_distillation.settings import DistillationSettings
from src.framework.mcts.neural_mcts import GameState, NeuralMCTS
from src.observability.logging import get_structured_logger
from src.training.system_config import MCTSConfig
from src.utils.seeding import new_rng

logger = get_structured_logger(__name__)


class HygienicCollector:
    """Collect training rows without using play τ as the policy target."""

    def __init__(
        self,
        mcts: NeuralMCTS,
        settings: DistillationSettings,
        *,
        action_space_size: int,
    ) -> None:
        if mcts.config.num_simulations < settings.min_simulations:
            raise ValueError(f"num_simulations={mcts.config.num_simulations} < min {settings.min_simulations}")
        assert_neural_mcts_teacher(mcts)
        self.mcts = mcts
        self.settings = settings
        self.action_space_size = action_space_size

    def _visit_vector(self, root: object, state: GameState) -> np.ndarray:
        visits = np.zeros(self.action_space_size, dtype=np.float64)
        children = getattr(root, "children", {})
        for action, child in children.items():
            idx = state.action_to_index(action)
            if 0 <= idx < self.action_space_size:
                visits[idx] = float(child.visit_count)
        return visits

    def _legal_mask(self, state: GameState) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=bool)
        for action in state.get_legal_actions():
            idx = state.action_to_index(action)
            if 0 <= idx < self.action_space_size:
                mask[idx] = True
        return mask

    async def play_game(
        self,
        initial_state: GameState,
        *,
        game_id: str | None = None,
    ) -> list[TrajectoryRow]:
        """Play one game. Search runs under ``eval()``; π_train ignores play τ."""
        game_id = game_id or uuid4().hex
        network = self.mcts.network
        was_training = bool(network.training)
        network.eval()
        rows: list[TrajectoryRow] = []
        state = initial_state
        move_count = 0
        try:
            while not state.is_terminal():
                tau_play = play_temperature(
                    move_count=move_count,
                    threshold=self.settings.temperature_threshold,
                    tau_init=self.settings.temperature_init,
                    tau_final=self.settings.temperature_final,
                )
                # τ=1 here so the unused return dict is visit/sum; play uses the root.
                _, root = await self.mcts.search(state, temperature=1.0, add_root_noise=True)
                visits = self._visit_vector(root, state)
                policy = visits_to_policy(visits, temperature=1.0)
                current_player = int(state.current_player)
                row = TrajectoryRow(
                    game_id=game_id,
                    ply=move_count,
                    state=state.to_tensor(),
                    visit_counts=visits,
                    policy_target=policy,
                    legal_mask=self._legal_mask(state),
                    value_target=0.0,
                    current_player=current_player,
                )
                validate_row(row, action_size=self.action_space_size)
                rows.append(row)

                play_probs = root.get_action_probs(tau_play)
                action = self.mcts.select_action(play_probs, temperature=tau_play)
                state = state.apply_action(action)
                move_count += 1
            _assign_stm_values(rows, terminal=state, single_agent=self.mcts.single_agent)
            for row in rows:
                validate_row(row, action_size=self.action_space_size)
            logger.info(
                "distillation game collected",
                lineage_id=game_id,
                ply_count=len(rows),
            )
        finally:
            network.train(was_training)
        return rows

    async def generate_batch(
        self,
        num_games: int,
        initial_state_fn: Callable[[], GameState],
        rng: np.random.Generator,
    ) -> list[TrajectoryRow]:
        """Play ``num_games`` and clear the network-blind eval cache between games."""
        all_rows: list[TrajectoryRow] = []
        for game_idx in range(num_games):
            game_id = f"{int(rng.integers(0, 2**32)):08x}-{game_idx}"
            rows = await self.play_game(initial_state_fn(), game_id=game_id)
            all_rows.extend(rows)
            self.mcts.clear_cache()
        return all_rows


def _assign_stm_values(rows: list[TrajectoryRow], *, terminal: GameState, single_agent: bool) -> None:
    """z from the mover stored on the row, not an independent player counter."""
    for row in rows:
        if single_agent:
            row.value_target = float(terminal.get_reward())
        else:
            row.value_target = float(terminal.get_reward(player=row.current_player))


def assert_neural_mcts_teacher(mcts: NeuralMCTS) -> None:
    """Refuse non-NeuralMCTS teachers (core/parallel/PW engines are out of contract)."""
    if type(mcts) is not NeuralMCTS:
        raise TypeError(f"teacher must be NeuralMCTS, got {type(mcts)!r}")


def build_mcts(
    network: nn.Module,
    settings: DistillationSettings,
    *,
    device: str,
    seed: int | None,
    single_agent: bool = False,
    rng: np.random.Generator | None = None,
) -> NeuralMCTS:
    """Wire NeuralMCTS with distillation temperature knobs (play only)."""
    config = MCTSConfig(
        num_simulations=settings.default_simulations,
        temperature_init=settings.temperature_init,
        temperature_final=settings.temperature_final,
        temperature_threshold=settings.temperature_threshold,
    )
    owned_rng = rng if rng is not None else new_rng(seed)
    placed = place_network(network, device)
    return NeuralMCTS(
        placed,
        config,
        device=device,
        single_agent=single_agent,
        rng=owned_rng,
        seed=None if rng is not None else seed,
    )
