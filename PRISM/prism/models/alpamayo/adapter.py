"""
Adapter integrating NVIDIA Alpamayo as the PRISM policy backbone.

Built incrementally — one integration problem per step:

    [x] Problem 5 — StochasticActionHead: PPO-compatible log_prob / entropy
    [x] Problem 2 — QFormerCritic: style-conditioned value head on frozen features
    [x] Problem 3 — z_t injection: normalised z_t serialised into instruction string
    [ ] Problem 4 — LoRA fine-tuning: load backbone, freeze weights, adapt layers
    [ ] Problem 1 — Observation encoding: camera images instead of BEV raster

Until Problem 4 is complete the backbone is None, so forward() returns
zero-mean actions and zero values.  All downstream PPO infrastructure
(GAE, clipping, Lagrangian) can be tested end-to-end in this state.

Backbone architecture (Alpamayo / Qwen2.5-VL-7B):
    num_hidden_layers = 32
    hidden_size       = 4096
    num_attention_heads = 32
    Requires output_hidden_states=True to access intermediate layer states.
    Default extraction layers: [20, 26, 31]  (early-mid, mid-late, final).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from prism.models.alpamayo.instruction import AlpamayoInstructionBuilder
from prism.models.base import PolicyOutput, PRISMCriticBase, PRISMPolicyBase
from prism.models.common.action_heads import StochasticActionHead


class AlpamayoAdapter(PRISMPolicyBase):
    """
    PRISMPolicyBase adapter for NVIDIA Alpamayo.

    Args:
        policy_id:       index k of this policy (0 … K-1).  Used in the
                         instruction preamble ("You are driving style k=…").
        action_dim:      dimensionality of the driving action space.
                         Default 2 → (acceleration, steering_rate).
        reward_dim:      dimensionality of the style reward vector.
                         Used to size the w_k buffer.  Default 4.
        init_log_std:    initial log-std for StochasticActionHead.
        critic:          QFormerCritic (or any PRISMCriticBase) instance.
                         If None, value output is always zero (placeholder).
        extract_layers:  backbone layer indices to extract hidden states from.
                         Passed to the backbone forward call once Problem 4
                         is implemented.  Default [20, 26, 31] for 32-layer
                         Alpamayo/Qwen2.5-VL-7B.
    """

    def __init__(
        self,
        policy_id: int = 0,
        action_dim: int = 2,
        reward_dim: int = 4,
        init_log_std: float = -0.5,
        critic: Optional[PRISMCriticBase] = None,
        extract_layers: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.policy_id = policy_id
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self._extract_layers = extract_layers or [20, 26, 31]

        # ── Problem 5 (done): stochastic action head ──────────────────────
        self.action_head = StochasticActionHead(action_dim, init_log_std)

        # ── Problem 2 (done): Q-Former critic ─────────────────────────────
        # Stored as a plain attribute (not nn.Module child via assignment)
        # so it appears in state_dict and parameters() automatically.
        self.critic = critic  # PRISMCriticBase | None

        # Preference vector w_k for this policy — set via set_w_k() before
        # training begins.  Registered as a buffer: moves with .to(device)
        # but is not a trainable parameter.
        self.register_buffer("_w_k", torch.zeros(reward_dim))

        # ── Problem 3 (done): instruction builder for z_t text injection ────
        # Built lazily in set_w_k() once the preference vector is available.
        self._instruction_builder: Optional[AlpamayoInstructionBuilder] = None

        # ── Problem 4 (TODO): Alpamayo backbone loaded with LoRA ──────────
        self._backbone: Optional[nn.Module] = None

        # ── Problem 1 (TODO): observation encoder (camera → visual tokens) ─
        self._obs_encoder: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    # Per-policy state setters  (mirror PRISMEnv.set_lambda / set_utility_fn)
    # ------------------------------------------------------------------

    def set_w_k(self, w_k: torch.Tensor) -> None:
        """
        Set the preference vector for policy k.

        Called once per policy at the start of Stage 2 training.
        w_k is (reward_dim,) — e.g. [0.55, 0.15, 0.15, 0.15] for comfort.

        Also rebuilds the instruction preamble so the static part of the
        language prompt reflects this policy's priority ordering.
        """
        self._w_k.data.copy_(w_k.to(self._w_k.device))
        w_k_np = w_k.cpu().numpy()
        if self._instruction_builder is None:
            self._instruction_builder = AlpamayoInstructionBuilder(
                policy_id=self.policy_id,
                w_k=w_k_np,
            )
        else:
            self._instruction_builder.update(self.policy_id, w_k_np)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: dict,
        actions: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        ref = next(iter(obs.values()))
        batch = ref.shape[0] if isinstance(ref, torch.Tensor) else 1
        device = self.action_head.log_std.device

        # ── Step 1: encode observation ────────────────────────── Problem 1
        # TODO: visual_tokens = self._obs_encoder(obs)

        # ── Step 2: build z_t instruction string ──────────────── Problem 3
        # z_t_normalised comes from obs["value_measurements"] — written by
        # PRISMEnv every step, no changes to the env required.
        instruction: Optional[str] = None
        if self._instruction_builder is not None:
            z_t_val = obs.get("value_measurements")
            if z_t_val is not None:
                if isinstance(z_t_val, torch.Tensor):
                    z_t_np = z_t_val[0].cpu().numpy()
                else:
                    z_t_np = np.asarray(z_t_val[0], dtype=np.float32)
                instruction = self._instruction_builder.build(z_t_np)
        # TODO Problem 4: pass `instruction` to backbone as language input:
        #   outputs = self._backbone(visual_tokens, instruction,
        #                            output_hidden_states=True)

        # ── Step 3: run Alpamayo backbone ─────────────────────── Problem 4
        # TODO (requires output_hidden_states=True):
        #   outputs = self._backbone(visual_tokens, zt_token,
        #                            output_hidden_states=True)
        #   hidden_states = torch.cat(
        #       [outputs.hidden_states[i] for i in self._extract_layers],
        #       dim=1,
        #   )  # (batch, len(extract_layers)*seq_len, 4096)
        #   action_mean = <projection from outputs>
        #
        # Placeholder until backbone is wired in:
        action_mean = torch.zeros(batch, self.action_dim, device=device)
        backbone_hidden_states: Optional[torch.Tensor] = None

        # ── Step 4: stochastic action head ────────────────────── Problem 5
        act, log_prob, entropy = self.action_head(action_mean, actions)

        # ── Step 5: Q-Former critic ───────────────────────────── Problem 2
        if self.critic is not None and backbone_hidden_states is not None:
            z_t = obs.get("value_measurements")
            if z_t is None:
                z_t = torch.zeros(batch, self.reward_dim, device=device)
            w_k = self._w_k.unsqueeze(0).expand(batch, -1)         # (batch, reward_dim)
            style_vec = torch.cat([w_k, z_t], dim=-1)              # (batch, 2*reward_dim)
            value = self.critic(backbone_hidden_states.detach(), style_vec)
        else:
            value = torch.zeros(batch, device=device)

        return PolicyOutput(action=act, log_prob=log_prob, entropy=entropy, value=value)

    # ------------------------------------------------------------------
    # Trainable parameters
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """
        Returns only the parameters that should be updated by the optimiser:
            - StochasticActionHead (log_std)
            - QFormerCritic (query_tokens, style_proj, cross_attn, value_head)
            - TODO Problem 4: LoRA adapter parameters from the frozen backbone

        The backbone weights themselves are never yielded here — they are
        frozen and only the LoRA adapters will be added in Problem 4.
        """
        params = list(self.action_head.parameters())
        if self.critic is not None:
            params += list(self.critic.parameters())
        # TODO Problem 4: params += list(lora_params_from_backbone)
        return iter(params)
