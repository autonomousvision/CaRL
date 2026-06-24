"""
Instruction string builder for Alpamayo z_t injection.

Encodes PRISM's per-policy state into natural language at every timestep so
the Alpamayo transformer can condition its actions on both the driving style
objective (w_k, static) and the current episode progress (z_t, dynamic).

Design follows π₀.5 (arXiv:2504.16054): proprioceptive / task state is
serialized as text and prepended to the VLM input.  Text injection exploits
Alpamayo's pretrained number and driving-vocabulary representations rather
than requiring a new projector trained from scratch.

The instruction has two parts:
    Preamble  — built once from w_k at policy construction time.
                Describes which driving objectives this policy prioritises.
                Re-built only if w_k changes (never in normal training).

    Dynamic   — rebuilt every step from obs["value_measurements"].
                Contains the EMA-normalised cumulative style returns z_t.
                Normalised values communicate direction and relative magnitude
                without requiring Alpamayo to know the absolute scale.
                Three decimal places used (sufficient to distinguish 0.023
                from 0.031 after EMA normalisation).

Source for z_t: obs["value_measurements"] (shape: (4,)) — the same vector
that PRISMEnv writes every step.  No changes to PRISMEnv are required.
"""

from __future__ import annotations

import numpy as np


_OBJ_NAMES = ["comfort", "progress", "lateral discipline", "spacing"]

_OBJ_DESCS = [
    "smooth ride, minimal jerk and harsh acceleration",
    "forward advancement, maintaining target speed",
    "lane centering, minimal unnecessary lane changes",
    "safe following distance from lead vehicle",
]


class AlpamayoInstructionBuilder:
    """
    Builds Alpamayo's per-step instruction string.

    Args:
        policy_id: index k of this policy (0 … K-1).
        w_k:       preference vector for this policy, shape (reward_dim,).
                   Used to determine priority order for the preamble.
    """

    def __init__(self, policy_id: int, w_k: np.ndarray) -> None:
        self._policy_id = policy_id
        self._preamble = self._build_preamble(policy_id, w_k)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(self, z_t_normalised: np.ndarray, spatial_desc: str = "") -> str:
        """
        Return the full instruction string for one timestep.

        Args:
            z_t_normalised: (reward_dim,) EMA-normalised cumulative style
                            returns from obs["value_measurements"].
            spatial_desc:   compact natural-language spatial state string
                            produced by spatial_description.build_spatial_description().
                            Inserted between the style preamble and the z_t string.
                            Pass "" (default) to omit (backward-compatible).
        """
        zt_string = self._build_dynamic(z_t_normalised)
        if spatial_desc:
            return self._preamble + "\n" + spatial_desc + "\n" + zt_string
        return self._preamble + "\n" + zt_string

    def update(self, policy_id: int, w_k: np.ndarray) -> None:
        """Rebuild the preamble — call if w_k ever changes."""
        self._policy_id = policy_id
        self._preamble = self._build_preamble(policy_id, w_k)

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_preamble(k: int, w_k: np.ndarray) -> str:
        priority = np.argsort(w_k)[::-1]
        names = _OBJ_NAMES
        return (
            f"You are driving style k={k}, which prioritizes "
            f"{names[priority[0]]} above all else, followed by "
            f"{names[priority[1]]}, {names[priority[2]]}, "
            f"and {names[priority[3]]}."
        )

    @staticmethod
    def _build_dynamic(z: np.ndarray) -> str:
        return (
            f"Cumulative style returns this episode (normalised):\n"
            f"- Comfort ({_OBJ_DESCS[0]}): {z[0]:.3f}\n"
            f"- Progress ({_OBJ_DESCS[1]}): {z[1]:.3f}\n"
            f"- Lateral discipline ({_OBJ_DESCS[2]}): {z[2]:.3f}\n"
            f"- Spacing ({_OBJ_DESCS[3]}): {z[3]:.3f}"
        )
