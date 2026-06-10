"""
CVaR safety constraint with Lagrangian dual update.

CVaR_{alpha}(C^pi_k) <= epsilon(alpha)

where C^i = sum_t(gamma^t * c_t) is the cumulative discounted cost of
episode i, alpha is the confidence level (tightened by the curriculum),
and epsilon(alpha) is the tolerance (from hyperparams.json).

Lagrange update (gradient ascent on the dual variable):
    lambda_k <- max(0, lambda_k + eta_lambda * (CVaR_hat - epsilon))

This is a clipped first-order update — lambda stays non-negative.

CVaR estimation (sample-based, non-parametric):
    Sort episode costs descending.
    CVaR_alpha = mean of the top (1-alpha) fraction.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np


class CVaRLagrangian:
    """
    Tracks episode-level cumulative costs, estimates CVaR, and updates lambda.

    Parameters
    ----------
    eta_lambda : float
        Lagrangian dual step size.
    buffer_size : int
        Rolling window of episode costs used for CVaR estimation.
    lambda_init : float
        Initial Lagrange multiplier (should be 0 at the start).
    """

    def __init__(
        self,
        eta_lambda: float = 0.01,
        buffer_size: int = 500,
        lambda_init: float = 0.0,
    ) -> None:
        self._eta = eta_lambda
        self._lambda: float = max(0.0, lambda_init)
        self._costs: Deque[float] = deque(maxlen=buffer_size)

    # ------------------------------------------------------------------
    # Episode collection
    # ------------------------------------------------------------------

    def add_episode(self, cumulative_cost: float) -> None:
        """Record the discounted cumulative cost of one completed episode."""
        self._costs.append(float(cumulative_cost))

    def add_episodes(self, costs: List[float]) -> None:
        for c in costs:
            self.add_episode(c)

    # ------------------------------------------------------------------
    # CVaR estimation
    # ------------------------------------------------------------------

    def estimate_cvar(self, alpha: float) -> float:
        """
        Estimate CVaR_{alpha}(C^pi) from the buffered episode costs.

        Returns 0.0 if no episodes have been collected yet.
        """
        if len(self._costs) == 0:
            return 0.0

        sorted_costs = np.sort(list(self._costs))[::-1]  # descending
        n_tail = max(1, int(np.ceil((1.0 - alpha) * len(sorted_costs))))
        return float(np.mean(sorted_costs[:n_tail]))

    # ------------------------------------------------------------------
    # Lagrangian update
    # ------------------------------------------------------------------

    def update_lambda(self, alpha: float, epsilon: float) -> Tuple[float, float]:
        """
        Estimate CVaR and perform one dual gradient step.

        Returns (new_lambda, cvar_hat).
        """
        cvar_hat = self.estimate_cvar(alpha)
        constraint_violation = cvar_hat - epsilon
        self._lambda = max(0.0, self._lambda + self._eta * constraint_violation)
        return self._lambda, cvar_hat

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def lambda_k(self) -> float:
        return self._lambda

    @lambda_k.setter
    def lambda_k(self, value: float) -> None:
        self._lambda = max(0.0, float(value))

    @property
    def n_episodes(self) -> int:
        return len(self._costs)

    def state_dict(self) -> dict:
        return {
            "lambda": self._lambda,
            "costs": list(self._costs),
            "eta_lambda": self._eta,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "CVaRLagrangian":
        obj = cls(eta_lambda=d["eta_lambda"], lambda_init=d["lambda"])
        obj._costs.extend(d.get("costs", []))
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Batch CVaR helper (used in evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_episode_cost(
    costs_per_step: List[float],
    gamma: float = 0.99,
) -> float:
    """
    Compute the discounted cumulative safety cost of one episode.
    C^i = sum_t(gamma^t * c_t)
    """
    total = 0.0
    for t, c in enumerate(costs_per_step):
        total += (gamma ** t) * c
    return total


def cvar_from_episodes(
    episode_costs: List[float],
    alpha: float,
) -> float:
    """
    Compute CVaR_{alpha} from a list of episode-level cumulative costs.
    """
    if not episode_costs:
        return 0.0
    arr = np.array(episode_costs, dtype=np.float64)
    sorted_arr = np.sort(arr)[::-1]
    n_tail = max(1, int(np.ceil((1.0 - alpha) * len(sorted_arr))))
    return float(np.mean(sorted_arr[:n_tail]))
