"""
Structured spatial state serializer for Alpamayo instruction injection.

Reads ground-truth spatial quantities already computed by CaRL from the
observation dict and serializes them as a compact natural-language string
(≤60 tokens) to append to Alpamayo's instruction text every RL step.

No new perception — all quantities are written into obs["spatial_state"]
by CaRL's reward builder each step via spatial_state_serializer.from_info().

Spatial state array layout (shape (9,), from spatial_state_serializer.py):
    [0] d_lead    — lead vehicle distance [m]          (50.0 = no lead)
    [1] v_lead    — lead vehicle speed [m/s]
    [2] has_lead  — 1.0 if lead exists, 0.0 otherwise
    [3] v_ego     — ego speed [m/s]
    [4] d_lat     — lateral deviation from lane centre [m] (+ = right)
    [5] v_limit   — speed limit [m/s]
    [6] d_left    — nearest vehicle in left lane [m]   (50.0 = clear)
    [7] d_right   — nearest vehicle in right lane [m]  (50.0 = clear)
    [8] curvature — road curvature ahead [1/m]         (+ = left turn)

Output format (target ≤60 tokens):
    "Spatial state: lead {dist:.1f}m closing {rel_vel:.1f}m/s.
     Left lane: {left}. Right lane: {right}. Speed: {speed:.1f}m/s.
     Lane offset: {offset:+.2f}m. Road: {curvature}."
"""

from __future__ import annotations

import numpy as np

_NO_LEAD_SENTINEL = 40.0    # d_lead above this → treat as no lead vehicle
_CLEAR_THRESHOLD_M = 30.0   # d_left/d_right above this → report as "clear"
_STRAIGHT_THRESHOLD = 0.005  # |curvature| < this → "straight"
_MILD_CURVE_THRESHOLD = 0.02  # else if < this → "mild curve", else "sharp curve"


def build_spatial_description(obs: dict, state=None) -> str:
    """
    Serialize the spatial state from obs into a compact natural-language string.

    Args:
        obs:   observation dict.  Must contain "spatial_state" — the (9,) float32
               array produced by spatial_state_serializer.to_array().  If the key
               is absent or the array is malformed, returns a safe fallback string.
        state: simulation state object (reserved for future use; not read here).

    Returns:
        A single-line string of ≤60 tokens describing the driving scene.
    """
    raw = obs.get("spatial_state")
    if raw is None:
        return "Spatial state: unavailable."

    arr = np.asarray(raw, dtype=np.float32).ravel()
    if arr.shape[0] < 9:
        return "Spatial state: unavailable."

    d_lead, v_lead, has_lead, v_ego, d_lat, v_limit, d_left, d_right, curvature = (
        float(arr[i]) for i in range(9)
    )

    # Lead vehicle — report closing speed (positive = closing gap)
    if has_lead > 0.5 and d_lead < _NO_LEAD_SENTINEL:
        rel_vel = v_ego - v_lead
        lead_str = f"lead {d_lead:.1f}m closing {rel_vel:.1f}m/s"
    else:
        lead_str = "lead none"

    # Adjacent lane occupancy
    left_str = f"{d_left:.1f}m" if d_left < _CLEAR_THRESHOLD_M else "clear"
    right_str = f"{d_right:.1f}m" if d_right < _CLEAR_THRESHOLD_M else "clear"

    # Road curvature (qualitative)
    if abs(curvature) < _STRAIGHT_THRESHOLD:
        curve_str = "straight"
    elif abs(curvature) < _MILD_CURVE_THRESHOLD:
        curve_str = "mild curve"
    else:
        curve_str = "sharp curve"

    return (
        f"Spatial state: {lead_str}. "
        f"Left lane: {left_str}. "
        f"Right lane: {right_str}. "
        f"Speed: {v_ego:.1f}m/s. "
        f"Lane offset: {d_lat:+.2f}m. "
        f"Road: {curve_str}."
    )
