"""
Running EMA normaliser for the cumulative style-return vector z_t.

z_t is a 4-dimensional vector accumulating discounted style rewards:
    z_{t+1} = z_t + gamma^t * r_t      (raw, unnormalised)

Before passing z_t to the policy as a state feature, we normalise each
dimension with a running mean/std estimated via EMA.

EMA update (applied once per episode end, or per step):
    mu_i   <- (1 - beta) * mu_i   + beta * z_i
    sigma_i <- sqrt((1 - beta) * sigma_i^2 + beta * (z_i - mu_i)^2)
"""

import numpy as np
from typing import Optional


class ZtNormaliser:
    def __init__(
        self,
        reward_dim: int = 4,
        beta: float = 0.01,
        mu_init: Optional[np.ndarray] = None,
        sigma_init: Optional[np.ndarray] = None,
    ) -> None:
        self._dim = reward_dim
        self._beta = beta
        self._mu = np.zeros(reward_dim, dtype=np.float64) if mu_init is None else np.array(mu_init, dtype=np.float64)
        self._var = (
            np.ones(reward_dim, dtype=np.float64) if sigma_init is None
            else np.array(sigma_init, dtype=np.float64) ** 2
        )

    def update(self, z: np.ndarray) -> None:
        """Update EMA statistics with the current z_t vector."""
        z = np.asarray(z, dtype=np.float64)
        self._mu = (1 - self._beta) * self._mu + self._beta * z
        self._var = (1 - self._beta) * self._var + self._beta * (z - self._mu) ** 2

    def normalise(self, z: np.ndarray) -> np.ndarray:
        """Return (z - mu) / sigma, clipped to [-5, 5] for stability."""
        z = np.asarray(z, dtype=np.float64)
        sigma = np.sqrt(self._var + 1e-8)
        return np.clip((z - self._mu) / sigma, -5.0, 5.0).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mu": self._mu.tolist(), "var": self._var.tolist(), "beta": self._beta}

    @classmethod
    def from_state_dict(cls, d: dict, reward_dim: int = 4) -> "ZtNormaliser":
        obj = cls(reward_dim=reward_dim, beta=d["beta"])
        obj._mu = np.array(d["mu"], dtype=np.float64)
        obj._var = np.array(d["var"], dtype=np.float64)
        return obj

    @classmethod
    def from_hyperparams(cls, hp: dict, beta: float = 0.01) -> "ZtNormaliser":
        """Initialise from z_normalisation block of hyperparams.json."""
        z_norm = hp.get("z_normalisation", {})
        mu = z_norm.get("z_mu", None)
        sigma = z_norm.get("z_sigma", None)
        return cls(reward_dim=4, beta=beta, mu_init=mu, sigma_init=sigma)
