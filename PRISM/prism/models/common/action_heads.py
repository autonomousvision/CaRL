"""
Shared action head modules used across backbone adapters.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


class StochasticActionHead(nn.Module):
    """
    Diagonal Gaussian policy head for continuous action spaces.

    Any backbone produces a deterministic mean vector.  This head adds
    learnable per-dimension log-standard-deviations and wraps the result
    in a Normal distribution, giving PPO the two things it needs:
      - rsample()     for differentiable action sampling during rollouts
      - log_prob()    for the clipped probability ratio during updates

    The log_std parameters are the only learnable weights here; they are
    initialised to init_log_std across all dimensions.

    Args:
        action_dim:   number of action dimensions (e.g. 2 for accel + steer)
        init_log_std: starting value for log(std).  -0.5 → std ≈ 0.61,
                      a reasonable spread for actions normalised to [-1, 1].
    """

    def __init__(self, action_dim: int, init_log_std: float = -0.5) -> None:
        super().__init__()
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

    def forward(
        self,
        mean: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mean:    (batch, action_dim) — backbone's action prediction
            actions: (batch, action_dim) — if provided, evaluate their
                     log_prob under the current distribution (PPO update).
                     If None, draw a fresh sample (rollout collection).
        Returns:
            (actions, log_prob, entropy)
              actions:  (batch, action_dim)
              log_prob: (batch,)  — summed over action dimensions
              entropy:  (batch,)  — summed over action dimensions
        """
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        if actions is None:
            # reparameterised sample so gradients flow back through mean
            actions = dist.rsample()

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return actions, log_prob, entropy
