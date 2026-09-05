from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .host_protocol import OrchestratorHost
else:
    OrchestratorHost = object  # type: ignore[misc,assignment]

import time

from src.observability.logging import get_structured_logger
from src.training.replay_buffer import Experience

logger = get_structured_logger(__name__)


class SelfPlayMixin:

    async def _generate_self_play_data(self: "OrchestratorHost") -> list[Experience]:
        """Generate training data from self-play games."""
        num_games = self.config.training.games_per_iteration
        logger.debug(
            "Starting self-play data generation",
            total_games=num_games,
            temperature_threshold=self.config.mcts.temperature_threshold,
            mcts_simulations=self.config.mcts.num_simulations,
        )
        all_examples = []
        game_start_time = time.perf_counter()
        log_interval = max(1, num_games // self.config.log_interval) if self.config.log_interval > 0 else 5
        for game_idx in range(num_games):
            game_iter_start = time.perf_counter()
            try:
                examples = await self.self_play_collector.play_game(
                    initial_state=self.initial_state_fn(), temperature_threshold=self.config.mcts.temperature_threshold
                )
                for ex in examples:
                    all_examples.append(Experience(state=ex.state, policy=ex.policy_target, value=ex.value_target))
                game_time = (time.perf_counter() - game_iter_start) * 1000
                if (game_idx + 1) % log_interval == 0:
                    elapsed = time.perf_counter() - game_start_time
                    avg_game_time = elapsed / (game_idx + 1)
                    remaining_games = num_games - (game_idx + 1)
                    eta_seconds = remaining_games * avg_game_time
                    logger.debug(
                        "Self-play progress",
                        games_completed=game_idx + 1,
                        total_games=num_games,
                        progress_percent=round((game_idx + 1) / num_games * 100, 1),
                        examples_collected=len(all_examples),
                        last_game_time_ms=round(game_time, 2),
                        avg_game_time_ms=round(avg_game_time * 1000, 2),
                        eta_seconds=round(eta_seconds, 1),
                    )
            except Exception as e:
                logger.error(
                    "Self-play game failed", game_idx=game_idx, error=str(e), examples_so_far=len(all_examples)
                )
                continue
        buffer_size_before = len(self.replay_buffer) if hasattr(self.replay_buffer, "__len__") else 0
        self.replay_buffer.add_batch(all_examples)
        buffer_size_after = len(self.replay_buffer) if hasattr(self.replay_buffer, "__len__") else 0
        total_time = time.perf_counter() - game_start_time
        logger.info(
            "Self-play data generation completed",
            games_completed=num_games,
            examples_generated=len(all_examples),
            total_time_seconds=round(total_time, 2),
            avg_examples_per_game=round(len(all_examples) / max(num_games, 1), 2),
            buffer_size_before=buffer_size_before,
            buffer_size_after=buffer_size_after,
        )
        return all_examples
