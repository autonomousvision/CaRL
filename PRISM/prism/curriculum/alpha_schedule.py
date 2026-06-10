"""
Alpha and epsilon curriculum schedules for PRISM.

alpha(n) controls how conservative the CVaR constraint is:
    - Small alpha (0.20)  -> only the worst 20% of episodes matter (lenient)
    - Large alpha (0.95)  -> almost all episodes must satisfy the constraint (strict)

The schedule starts lenient and tightens over training so the agent first
learns to drive at all, then gradually becomes safer.

    alpha(n) = alpha_start + (alpha_end - alpha_start) * min(1, n / N_curriculum)

epsilon(n) = epsilon_curve[alpha(n)]  -- read from hyperparams.json
"""

import numpy as np
from typing import Dict

from prism.utils.hyperparams import get_epsilon


def compute_alpha(n: int, alpha_start: float, alpha_end: float, n_curriculum: int) -> float:
    """Return the current CVaR confidence level at training iteration n."""
    frac = min(1.0, n / max(n_curriculum, 1))
    return alpha_start + (alpha_end - alpha_start) * frac


def compute_epsilon(alpha: float, hp: Dict) -> float:
    """Return the CVaR tolerance epsilon for the given alpha."""
    return get_epsilon(hp, alpha)


class AlphaSchedule:
    """Stateful schedule tracking alpha/epsilon over training iterations."""

    def __init__(
        self,
        hp: Dict,
        alpha_start: float = 0.20,
        alpha_end: float = 0.95,
        n_curriculum: int = 5000,
    ) -> None:
        self._hp = hp
        self._alpha_start = alpha_start
        self._alpha_end = alpha_end
        self._n_curriculum = n_curriculum

    def get(self, n: int):
        """Return (alpha, epsilon) for training iteration n."""
        alpha = compute_alpha(n, self._alpha_start, self._alpha_end, self._n_curriculum)
        epsilon = compute_epsilon(alpha, self._hp)
        return alpha, epsilon

    def alpha_grid(self, n_steps: int = 200):
        """Return the full alpha trajectory for plotting."""
        ns = np.linspace(0, self._n_curriculum * 1.5, n_steps)
        alphas = [
            compute_alpha(int(n), self._alpha_start, self._alpha_end, self._n_curriculum)
            for n in ns
        ]
        return ns, alphas
