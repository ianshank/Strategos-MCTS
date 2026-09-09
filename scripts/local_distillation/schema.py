"""Versioned trajectory rows and lineage-grouped splits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from scripts.local_distillation.settings import DistillationSettings

SCHEMA_VERSION = 1


@dataclass
class TrajectoryRow:
    """One (state, π, z) example with lineage. Unvisited legal actions stay 0 in π."""

    game_id: str
    ply: int
    state: torch.Tensor
    visit_counts: np.ndarray
    policy_target: np.ndarray
    legal_mask: np.ndarray
    value_target: float
    current_player: int
    schema_version: int = SCHEMA_VERSION

    def state_hash(self) -> str:
        """Stable hash of the stored tensor for leakage checks."""
        blob = np.ascontiguousarray(self.state.detach().cpu().numpy()).tobytes()
        return hashlib.sha256(blob).hexdigest()


def validate_row(row: TrajectoryRow, *, action_size: int) -> None:
    """Reject all-zero / NaN policy rows; keep legal zeros in π as normal."""
    if row.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {row.schema_version}")
    if not row.game_id:
        raise ValueError("game_id is required")
    if row.ply < 0:
        raise ValueError("ply must be >= 0")
    if row.current_player not in (1, -1):
        raise ValueError("current_player must be 1 or -1")
    if row.visit_counts.shape != (action_size,) or row.policy_target.shape != (action_size,):
        raise ValueError("visit_counts and policy_target must match action_size")
    if row.legal_mask.shape != (action_size,):
        raise ValueError("legal_mask must match action_size")
    if not np.isfinite(row.policy_target).all() or not np.isfinite(row.visit_counts).all():
        raise ValueError("visit_counts and policy_target must be finite")
    if not np.isfinite(row.value_target):
        raise ValueError("value_target must be finite")
    if float(row.visit_counts.sum()) <= 0 or float(row.policy_target.sum()) <= 0:
        raise ValueError("reject all-zero visit/policy rows")
    if row.policy_target.min() < -1e-12:
        raise ValueError("policy_target must be non-negative")
    # Unvisited legal actions stay 0 in π; that is not a reason to drop the row.


def grouped_split(
    rows: Sequence[TrajectoryRow],
    settings: DistillationSettings,
    rng: np.random.Generator,
) -> tuple[list[TrajectoryRow], list[TrajectoryRow], list[TrajectoryRow]]:
    """Split by ``game_id``. Overlapping state hashes across splits raise."""
    by_game: dict[str, list[TrajectoryRow]] = {}
    for row in rows:
        by_game.setdefault(row.game_id, []).append(row)
    game_ids = np.array(sorted(by_game.keys()))
    rng.shuffle(game_ids)
    n = len(game_ids)
    if n == 0:
        raise ValueError("no games to split")
    n_train = int(n * settings.train_frac)
    n_val = int(n * settings.val_frac)
    if n_train < 1:
        raise ValueError("train split is empty; increase train_frac or the number of games")
    train_ids = set(game_ids[:n_train].tolist())
    val_ids = set(game_ids[n_train : n_train + n_val].tolist())
    test_ids = set(game_ids[n_train + n_val :].tolist())

    def _take(ids: set[str]) -> list[TrajectoryRow]:
        taken: list[TrajectoryRow] = []
        for gid in ids:
            taken.extend(by_game[gid])
        return taken

    train, val, test = _take(train_ids), _take(val_ids), _take(test_ids)
    _assert_no_hash_leakage(train, val, test)
    return train, val, test


def _assert_no_hash_leakage(
    train: Sequence[TrajectoryRow],
    val: Sequence[TrajectoryRow],
    test: Sequence[TrajectoryRow],
) -> None:
    train_h = {row.state_hash() for row in train}
    val_h = {row.state_hash() for row in val}
    test_h = {row.state_hash() for row in test}
    if train_h & val_h or train_h & test_h or val_h & test_h:
        raise ValueError("state hash overlap across grouped splits")
