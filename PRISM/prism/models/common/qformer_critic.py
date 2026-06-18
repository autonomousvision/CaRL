"""
Q-Former style critic for VLA backbones.

Implements the cross-attention critic architecture from IG-RFT (arXiv:2602.20715),
adapted for PRISM's multi-objective setting.

Architecture:
    N learnable query tokens cross-attend into frozen VLA backbone hidden states
    extracted from multiple transformer layers.  The queries are conditioned on
    a style vector [w_k || z_t] before cross-attention, so the critic learns to
    extract different value-relevant features for each driving style policy.

    hidden_states  (batch, total_seq_len, backbone_hidden_dim)
          ↓
    cross-attention  [Q: style-conditioned queries,  K/V: backbone features]
          ↓
    aggregated queries  (batch, N_q, query_dim)
          ↓
    flatten → MLP value head  →  V(s)  (batch,)

Gradient isolation:
    hidden_states must be detached from the backbone before being passed here.
    The backbone is updated by the actor's policy gradient only; the critic loss
    flows through the Q-Former and value head exclusively.

Reference: IG-RFT (arXiv:2602.20715), Appendix A.
    N_q=8, query_dim=256, n_heads=8 — all configurable here.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from prism.models.base import PRISMCriticBase


class QFormerCritic(PRISMCriticBase):
    """
    Q-Former critic conditioned on driving style for PRISM.

    Args:
        backbone_hidden_dim: hidden dimension of the VLA backbone whose
            hidden states are passed in as K/V.  For Alpamayo (Qwen2.5-VL-7B)
            this is 4096.
        n_queries: number of learnable query tokens (N_q).  Default 8
            following IG-RFT.  Worth ablating; more queries = richer
            representation but slower training.
        query_dim: embedding dimension of each query token.  Default 256
            following IG-RFT.  Must be divisible by n_heads.
        n_heads: number of attention heads in the cross-attention layer.
            Default 8.
        style_dim: dimension of the style conditioning vector [w_k || z_t].
            For PRISM with reward_dim=4: style_dim = 4 (w_k) + 4 (z_t) = 8.
        value_hidden_dims: hidden layer sizes for the MLP value head.
    """

    def __init__(
        self,
        backbone_hidden_dim: int = 4096,
        n_queries: int = 8,
        query_dim: int = 256,
        n_heads: int = 8,
        style_dim: int = 8,
        value_hidden_dims: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()

        self.n_queries = n_queries
        self.query_dim = query_dim

        # Learnable query tokens — initialised with small truncated normal
        self.query_tokens = nn.Parameter(torch.empty(n_queries, query_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        # Style conditioning: projects [w_k || z_t] to query_dim and adds to queries
        self.style_proj = nn.Linear(style_dim, query_dim)

        # Cross-attention: queries (Q) attend to backbone hidden states (K, V).
        # Setting kdim=vdim=backbone_hidden_dim lets nn.MultiheadAttention handle
        # the K/V projection internally (projects backbone_hidden_dim → query_dim).
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=n_heads,
            kdim=backbone_hidden_dim,
            vdim=backbone_hidden_dim,
            batch_first=True,
        )

        # MLP value head: flattened Q-Former output → scalar V(s)
        in_dim = n_queries * query_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for h in value_hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.value_head = nn.Sequential(*layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        style_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, total_seq_len, backbone_hidden_dim)
                Concatenated hidden states from selected extraction layers.
                Must be detached from the backbone before calling.
            style_vec: (batch, style_dim)
                [w_k || z_t] — preference vector concatenated with the
                normalised cumulative style return.
        Returns:
            (batch,) scalar value estimates.
        """
        batch = hidden_states.shape[0]

        # Expand query tokens to batch and add style conditioning
        queries = self.query_tokens.unsqueeze(0).expand(batch, -1, -1)  # (batch, N_q, d_q)
        style_emb = self.style_proj(style_vec).unsqueeze(1)             # (batch, 1, d_q)
        queries = queries + style_emb                                    # (batch, N_q, d_q)

        # Cross-attention: queries extract value-relevant information from backbone
        attended, _ = self.cross_attn(
            query=queries,
            key=hidden_states,
            value=hidden_states,
        )                                                                # (batch, N_q, d_q)

        # Flatten and project to scalar value
        flat = attended.reshape(batch, self.n_queries * self.query_dim) # (batch, N_q * d_q)
        value = self.value_head(flat).squeeze(-1)                       # (batch,)

        return value
