"""
Base class and return type shared by every PRISM policy adapter.

Every backbone (CaRL PPO, Alpamayo, or any future model) must subclass
PRISMPolicyBase and implement forward() returning PolicyOutput.  This
contract is what DPMORLTrainer depends on — the trainer never imports
any backbone-specific code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn as nn





@dataclass
class PolicyOutput:
    action: torch.Tensor     # (batch, action_dim)
    log_prob: torch.Tensor   # (batch,)  — sum over action dims
    entropy: torch.Tensor    # (batch,)  — sum over action dims
    value: torch.Tensor      # (batch,)


class PRISMPolicyBase(nn.Module, ABC):
    """
    Abstract base for all PRISM policy adapters.

    Subclasses must implement forward().  They may override
    trainable_parameters() to restrict optimisation to a subset of
    weights (e.g. LoRA adapters + heads only, backbone frozen).
    """

    @abstractmethod
    def forward(
        self,
        obs: dict,
        actions: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        """
        Args:
            obs:     observation dict from PRISMEnv
            actions: if provided, evaluate log_prob of these specific actions
                     (PPO update pass). If None, sample from the policy
                     (rollout collection pass).
        Returns:
            PolicyOutput(action, log_prob, entropy, value)
        """

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """
        Parameters to hand to the optimiser.

        Default: all parameters that have requires_grad=True.
        Override in subclasses that freeze a backbone (e.g. Alpamayo
        with LoRA) to return only the adapter + head parameters.
        """
        return (p for p in self.parameters() if p.requires_grad)


class PRISMCriticBase(nn.Module, ABC):
    """
    Abstract base for PRISM critic modules that read from VLA backbone features.

    The critic receives hidden states extracted from specific layers of a frozen
    backbone, along with a style conditioning vector, and returns a scalar
    value estimate V(s).

    hidden_states : (batch, seq_len, backbone_hidden_dim)
        Concatenated hidden states from the selected extraction layers.
        Gradients are detached before reaching here — the critic loss does
        not affect the backbone.

    style_vec : (batch, style_dim)
        Conditioning vector for the critic.  In PRISM this is [w_k || z_t]:
        the policy's preference vector concatenated with the normalised
        cumulative style return.
    """

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        style_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Returns (batch,) scalar value estimates."""

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)
