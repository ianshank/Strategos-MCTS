"""Coarse-dynamics Mixture Density Network (MDN) for uncertainty-aware subgoal selection.

This module provides the S3 dispersion signal in two independent pieces:

* :class:`CoarseTransitionAggregator` — **torch-free** (numpy) reducer that turns an ordered
  sequence of low-level state vectors, over a configurable ``window``, into one fixed-shape
  *coarse-transition* vector of length ``4 * state_dim`` — the concatenation of
  ``[first, last, element-wise mean, (last - first) delta]``.
* :class:`CoarseDynamicsMDN` — a **torch**, diagonal-Gaussian mixture-density head over the coarse
  vector. ``dispersion()`` returns a non-negative scalar per batch element: the trace of the total
  mixture covariance via the law of total variance (``E_k[sigma_k^2] + Var_k[mu_k]``), which is
  ``>= 0`` by construction.

torch is optional (the ``neural`` extra): the module **imports without torch** — the aggregator and
the numpy dispersion reference (:func:`mixture_variance_trace`) remain usable — while constructing
:class:`CoarseDynamicsMDN` without torch raises a clear :class:`RuntimeError` (never a silent no-op).
This module does not touch ``value_network.py`` / ``ValueOutput``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config.constants import (
    DEFAULT_COARSE_WINDOW,
    DEFAULT_MDN_COMPONENTS,
    DEFAULT_MDN_HIDDEN_DIM,
    MAX_COARSE_WINDOW,
    MAX_MDN_COMPONENTS,
    MIN_COARSE_WINDOW,
    MIN_MDN_COMPONENTS,
)
from src.observability.logging import get_logger

# Optional PyTorch import (the ``neural`` extra). Mirrors the guard idiom in
# ``src/framework/mcts/llm_guided/training/networks.py``.
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = get_logger(__name__)

# The coarse-transition vector concatenates 4 per-dim summaries: first, last, mean, delta.
_COARSE_SUMMARY_PARTS: int = 4


class CoarseTransitionAggregator:
    """Reduce a sequence of low-level state vectors into one fixed-shape coarse vector.

    The output is ``concat[first, last, mean, (last - first)]`` over the last ``window`` states,
    with length ``4 * state_dim`` regardless of ``window`` (``window`` selects *which* states are
    summarized, not the output length). Deterministic and torch-free.
    """

    def __init__(self, window: int = DEFAULT_COARSE_WINDOW) -> None:
        if not MIN_COARSE_WINDOW <= window <= MAX_COARSE_WINDOW:
            raise ValueError(f"window must be in [{MIN_COARSE_WINDOW}, {MAX_COARSE_WINDOW}], got {window}")
        self.window = window

    @staticmethod
    def output_dim(state_dim: int) -> int:
        """Length of the coarse-transition vector for a given per-state dimension.

        Raises:
            ValueError: If ``state_dim`` is not a positive integer (a non-positive dimension
                would yield a zero/negative length and mask an invalid state representation).
        """
        if state_dim <= 0:
            raise ValueError(f"state_dim must be a positive integer, got {state_dim}")
        return _COARSE_SUMMARY_PARTS * state_dim

    def aggregate(self, states: Sequence[Sequence[float]]) -> np.ndarray:
        """Aggregate an ordered ``[T, state_dim]`` sequence into a ``[4 * state_dim]`` vector.

        Args:
            states: Non-empty ordered sequence of low-level state vectors (equal length).

        Returns:
            A float32 numpy array of length ``4 * state_dim``.

        Raises:
            ValueError: If ``states`` is empty, not 2-D (ragged/scalar input), or has a
                zero-width state dimension (e.g. ``[[]]``, shape ``[1, 0]``).
        """
        arr = np.asarray(states, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
            raise ValueError("states must be a non-empty [T, state_dim] sequence with state_dim >= 1")
        window = arr[-self.window :]  # last `window` states (all of them if fewer)
        first = window[0]
        last = window[-1]
        mean = window.mean(axis=0)
        delta = last - first
        return np.concatenate([first, last, mean, delta]).astype(np.float32)


def mixture_variance_trace(
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Non-negative predictive dispersion of a diagonal-Gaussian mixture (numpy reference).

    Implements the law of total variance and traces over the state dimension::

        Var[X] = E_k[sigma_k^2] + Var_k[mu_k]      (per dimension, then summed)

    Both terms are non-negative (``variances >= 0``, weights are convex, the between term is a
    weighted sum of squares), so the returned dispersion is ``>= 0``. This is the torch-free
    reference that :meth:`CoarseDynamicsMDN.dispersion` mirrors.

    Args:
        weights: ``[B, K]`` mixing weights (each row sums to 1 over K).
        means: ``[B, K, D]`` component means.
        variances: ``[B, K, D]`` component variances (``>= 0``).

    Returns:
        ``[B]`` non-negative dispersion (trace of the total mixture covariance).
    """
    w = weights[..., None]  # [B, K, 1]
    within = (w * variances).sum(axis=1)  # E_k[sigma^2] -> [B, D]
    mean_of_means = (w * means).sum(axis=1)  # E_k[mu] -> [B, D]
    between = (w * (means - mean_of_means[:, None, :]) ** 2).sum(axis=1)  # Var_k[mu] -> [B, D]
    total_variance = within + between  # [B, D]
    return total_variance.sum(axis=1)  # trace over D -> [B]


@dataclass
class MDNParams:
    """Parameters of a diagonal-Gaussian mixture emitted by :class:`CoarseDynamicsMDN`.

    Tensors are typed ``Any`` so the module stays importable without torch.
    """

    logits: Any  # [B, K] mixing logits (pre-softmax)
    means: Any  # [B, K, D]
    log_vars: Any  # [B, K, D] log-variances (exp -> variances)


class CoarseDynamicsMDN(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Diagonal-Gaussian Mixture Density head over a coarse-transition vector.

    Requires PyTorch (the ``neural`` extra). ``forward`` returns :class:`MDNParams`;
    ``dispersion`` returns a non-negative ``[B, 1]`` variance-trace metric.
    """

    def __init__(
        self,
        input_dim: int,
        num_components: int = DEFAULT_MDN_COMPONENTS,
        hidden_dim: int = DEFAULT_MDN_HIDDEN_DIM,
        output_dim: int | None = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "CoarseDynamicsMDN requires PyTorch (install the 'neural' extra). "
                "The torch-free CoarseTransitionAggregator and mixture_variance_trace remain available."
            )
        if not MIN_MDN_COMPONENTS <= num_components <= MAX_MDN_COMPONENTS:
            raise ValueError(
                f"num_components must be in [{MIN_MDN_COMPONENTS}, {MAX_MDN_COMPONENTS}], got {num_components}"
            )
        resolved_output_dim = output_dim if output_dim is not None else input_dim
        for _name, _dim in (("input_dim", input_dim), ("hidden_dim", hidden_dim), ("output_dim", resolved_output_dim)):
            if _dim <= 0:
                raise ValueError(f"{_name} must be a positive integer, got {_dim}")
        super().__init__()
        self.input_dim = input_dim
        self.num_components = num_components
        self.output_dim = resolved_output_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.logits_head = nn.Linear(hidden_dim, num_components)
        self.means_head = nn.Linear(hidden_dim, num_components * self.output_dim)
        self.log_var_head = nn.Linear(hidden_dim, num_components * self.output_dim)
        logger.debug(
            "CoarseDynamicsMDN initialized",
            extra={"input_dim": input_dim, "num_components": num_components, "output_dim": self.output_dim},
        )

    def forward(self, coarse: Any) -> MDNParams:
        """Map a ``[B, input_dim]`` coarse vector to diagonal-Gaussian mixture parameters."""
        batch = coarse.shape[0]
        features = self.trunk(coarse)
        logits = self.logits_head(features)  # [B, K]
        means = self.means_head(features).view(batch, self.num_components, self.output_dim)
        log_vars = self.log_var_head(features).view(batch, self.num_components, self.output_dim)
        return MDNParams(logits=logits, means=means, log_vars=log_vars)

    def dispersion(self, coarse: Any) -> Any:
        """Return the non-negative ``[B, 1]`` variance-trace dispersion for a coarse batch.

        Mirrors :func:`mixture_variance_trace` in torch (law of total variance).
        """
        params = self.forward(coarse)
        weights = torch.softmax(params.logits, dim=-1)  # [B, K]
        variances = torch.exp(params.log_vars)  # [B, K, D]
        w = weights.unsqueeze(-1)  # [B, K, 1]
        within = (w * variances).sum(dim=1)  # [B, D]
        mean_of_means = (w * params.means).sum(dim=1)  # [B, D]
        between = (w * (params.means - mean_of_means.unsqueeze(1)) ** 2).sum(dim=1)  # [B, D]
        return (within + between).sum(dim=1, keepdim=True)  # [B, 1]


def create_coarse_dynamics_mdn(
    input_dim: int,
    num_components: int = DEFAULT_MDN_COMPONENTS,
    hidden_dim: int = DEFAULT_MDN_HIDDEN_DIM,
    output_dim: int | None = None,
) -> CoarseDynamicsMDN:
    """Factory for :class:`CoarseDynamicsMDN` (raises without torch — see the class docstring)."""
    return CoarseDynamicsMDN(
        input_dim=input_dim,
        num_components=num_components,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )


__all__ = [
    "CoarseDynamicsMDN",
    "CoarseTransitionAggregator",
    "MDNParams",
    "create_coarse_dynamics_mdn",
    "mixture_variance_trace",
]
