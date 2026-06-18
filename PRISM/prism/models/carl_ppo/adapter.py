"""
Adapter that wraps CaRL's PPOPolicy into PRISMPolicyBase.

CaRL's PPOPolicy.forward() returns a positional tuple:
    (action, log_prob, entropy, value, ...)

This adapter unpacks that tuple into PolicyOutput so DPMORLTrainer
stays backend-agnostic and never imports CaRL-specific code directly.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from prism.models.base import PolicyOutput, PRISMPolicyBase


class CaRLPPOAdapter(PRISMPolicyBase):
    """
    Wraps CaRL's PPOPolicy to satisfy the PRISMPolicyBase contract.

    Args:
        carl_policy: an instantiated CaRL PPOPolicy (already on the
                     correct device).
    """

    def __init__(self, carl_policy: nn.Module) -> None:
        super().__init__()
        self.policy = carl_policy

    def forward(
        self,
        obs: dict,
        actions: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        if actions is not None:
            raw = self.policy.forward(obs, actions=actions)
            return PolicyOutput(
                action=actions,
                log_prob=raw[1],
                entropy=raw[2],
                value=raw[3],
            )

        raw = self.policy.forward(obs)
        return PolicyOutput(
            action=raw[0],
            log_prob=raw[1],
            entropy=raw[2],
            value=raw[3],
        )
