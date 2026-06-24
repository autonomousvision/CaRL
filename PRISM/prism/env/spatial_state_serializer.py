"""
Rule-based text serialization of CaRL's spatial state quantities.

Converts a fixed-layout numpy array of ground-truth spatial measurements
into a compact natural-language string (≤60 tokens) for injection into
Alpamayo's text token pathway alongside the style preamble and z_t string.

No ML, no new perception — all quantities are already computed by CaRL's
reward builder at every step.  This module is a pure transformation.

Layout of the spatial state array (SPATIAL_KEYS, length 9):
    [0] d_lead   — lead vehicle distance [m]          (50.0 = no lead)
    [1] v_lead   — lead vehicle speed [m/s]
    [2] has_lead — 1.0 if lead exists, 0.0 otherwise
    [3] v_ego    — ego speed [m/s]
    [4] d_lat    — lateral deviation from lane centre [m] (+ = right of centre)
    [5] v_limit  — speed limit [m/s]
    [6] d_left   — nearest vehicle in left lane [m]   (50.0 = clear)
    [7] d_right  — nearest vehicle in right lane [m]  (50.0 = clear)
    [8] curvature — road curvature ahead [1/m]        (+ = left turn)
"""

from __future__ import annotations

import numpy as np

# Canonical key order — must match array layout above
SPATIAL_KEYS = [
    "d_lead",
    "v_lead",
    "has_lead",
    "v_ego",
    "d_lat",
    "v_limit",
    "d_left",
    "d_right",
    "curvature",
]

_NO_LEAD_SENTINEL = 40.0   # d_lead above this → treat as no lead
_CLEAR_SENTINEL = 30.0     # d_left/d_right above this → treat as clear
_N = len(SPATIAL_KEYS)


def to_array(
    d_lead: float = 50.0,
    v_lead: float = 0.0,
    has_lead: float = 0.0,
    v_ego: float = 0.0,
    d_lat: float = 0.0,
    v_limit: float = 15.0,
    d_left: float = 50.0,
    d_right: float = 50.0,
    curvature: float = 0.0,
) -> np.ndarray:
    """Pack spatial quantities into the canonical (9,) float32 array."""
    return np.array(
        [d_lead, v_lead, has_lead, v_ego, d_lat, v_limit, d_left, d_right, curvature],
        dtype=np.float32,
    )


def from_info(info: dict) -> np.ndarray:
    """Build spatial array from the info dict populated by PRISMRewardBuilder."""
    return to_array(
        d_lead=float(info.get("d_lead", 50.0)),
        v_lead=float(info.get("v_lead", 0.0)),
        has_lead=float(info.get("has_lead", 0.0)),
        v_ego=float(info.get("v_ego", 0.0)),
        d_lat=float(info.get("d_lat", 0.0)),
        v_limit=float(info.get("v_limit", 15.0)),
        d_left=float(info.get("d_left", 50.0)),
        d_right=float(info.get("d_right", 50.0)),
        curvature=float(info.get("curvature", 0.0)),
    )


def to_text(arr: np.ndarray) -> str:
    """
    Serialize a (9,) spatial state array into a compact natural-language string.

    Targets ≤60 tokens.  Uses condensed phrasing and omits redundant information
    (e.g. does not report lead vehicle speed when there is no lead vehicle).

    Example output:
        "Speed 12.3 m/s, limit 15 m/s. Lead 18 m, closing 2.1 m/s.
         Lane offset -0.15 m. Left clear. Right vehicle 4.5 m. Curve 0.02."
    """
    d_lead, v_lead, has_lead, v_ego, d_lat, v_limit, d_left, d_right, curvature = arr

    parts = []

    # Ego speed and limit
    parts.append(f"Speed {v_ego:.1f} m/s, limit {v_limit:.0f} m/s.")

    # Lead vehicle
    if has_lead > 0.5 and d_lead < _NO_LEAD_SENTINEL:
        closing = v_ego - v_lead
        if abs(closing) < 0.3:
            parts.append(f"Lead {d_lead:.0f} m, matched speed.")
        elif closing > 0:
            parts.append(f"Lead {d_lead:.0f} m, closing {closing:.1f} m/s.")
        else:
            parts.append(f"Lead {d_lead:.0f} m, pulling away {-closing:.1f} m/s.")
    else:
        parts.append("No lead vehicle.")

    # Lane centre offset
    if abs(d_lat) < 0.1:
        parts.append("Centred in lane.")
    elif d_lat > 0:
        parts.append(f"Right of centre {d_lat:.2f} m.")
    else:
        parts.append(f"Left of centre {-d_lat:.2f} m.")

    # Adjacent lane occupancy
    left_clear = d_left >= _CLEAR_SENTINEL
    right_clear = d_right >= _CLEAR_SENTINEL
    if left_clear:
        parts.append("Left clear.")
    else:
        parts.append(f"Left vehicle {d_left:.0f} m.")
    if right_clear:
        parts.append("Right clear.")
    else:
        parts.append(f"Right vehicle {d_right:.0f} m.")

    # Road curvature
    if abs(curvature) < 0.005:
        parts.append("Straight road.")
    elif curvature > 0:
        parts.append(f"Left curve {curvature:.3f}.")
    else:
        parts.append(f"Right curve {-curvature:.3f}.")

    return " ".join(parts)
