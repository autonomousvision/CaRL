"""
PRISM style reward vector (4-dimensional, all in (0, 1]).

Components:
    r_comfort  : jerk-based comfort
    r_progress : speed + acceleration + lane position
    r_lateral  : lateral discipline (deviation + heading)
    r_spacing  : TTC-based safe following distance

All values are in (0, 1].  Floor values (delta_d=0.3, delta_s=0.2) prevent
rewards from reaching zero, as required by the PRISM formulation.

Hyperparameters are always read from hyperparams.json (never hardcoded).
See compute_hyperparams.py for how they are calibrated from IDM rollouts.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


# ------------------------------------------------------------------
# Individual reward components
# ------------------------------------------------------------------

def compute_comfort(j_lon: float, j_lat: float, sigma_j_sq: float) -> float:
    """
    Eq. (comfort) in paper.
    r_comfort = exp(-(j_lon^2 + j_lat^2) / sigma_j_sq)
    """
    jerk_sq = j_lon ** 2 + j_lat ** 2
    return float(math.exp(-jerk_sq / max(sigma_j_sq, 1e-8)))


def compute_progress(
    v_ego: float,
    v_des: float,
    a_ego: float,
    lane_index: int,
    n_lanes: int,
    beta: float,
    gamma_a: float,
) -> float:
    """
    Eq. (progress) in paper.
    r_progress = r_speed * r_accel * (0.5 + 0.5 * r_lane)
    """
    # Speed sub-reward: penalise shortfall from desired speed
    v_des_safe = max(v_des, 0.5)  # avoid division by zero
    shortfall = max(0.0, v_des_safe - v_ego)
    r_speed = math.exp(-shortfall / (max(beta, 1e-6) * v_des_safe))

    # Acceleration sub-reward: reward being in motion.
    # Floor at 0.01 so r_progress never reaches exactly 0 (CLAUDE.md: always in (0,1]).
    r_accel = max(0.01, 1.0 - math.exp(-abs(a_ego) / max(gamma_a, 1e-6)))

    # Lane position sub-reward (leftmost lane = 0, rightmost = N_lanes - 1)
    if n_lanes <= 1:
        r_lane = 0.5  # neutral when there is no lane choice
    else:
        r_lane = float(lane_index) / float(n_lanes - 1)

    return float(r_speed * r_accel * (0.5 + 0.5 * r_lane))


def compute_lateral_discipline(
    d_lat: float,
    delta_psi: float,
    sigma_d: float,
    phi: float,
) -> float:
    """
    Eq. (lateral) in paper.
    r_lateral = r_dev * r_heading
    r_dev     = 0.3 + 0.7 * exp(-(d_lat / sigma_d)^2)
    r_heading = exp(-(|delta_psi| / phi)^2)
    """
    sigma_d_safe = max(sigma_d, 1e-4)
    phi_safe = max(phi, 1e-4)

    r_dev = 0.3 + 0.7 * math.exp(-((d_lat / sigma_d_safe) ** 2))
    r_heading = math.exp(-(((abs(delta_psi)) / phi_safe) ** 2))
    return float(r_dev * r_heading)


def compute_spacing(
    ttc: float,
    tau: float,
) -> float:
    """
    Eq. (spacing) in paper.
    r_spacing = 0.2 + 0.8 * (1 - exp(-(max(0, TTC) / tau)^2))
    """
    tau_safe = max(tau, 0.1)
    ttc_clamped = max(0.0, ttc)
    return float(0.2 + 0.8 * (1.0 - math.exp(-((ttc_clamped / tau_safe) ** 2))))


def compute_ttc(
    d_lead: float,
    v_ego: float,
    v_lead: float,
    has_lead: bool,
) -> float:
    """
    Return TTC in seconds.
    TTC = d_lead / (v_ego - v_lead)  when v_ego > v_lead and has_lead.
    TTC = inf otherwise.
    """
    if not has_lead:
        return float("inf")
    closing_speed = v_ego - v_lead
    if closing_speed <= 0.0:
        return float("inf")
    return float(d_lead / max(closing_speed, 1e-3))


# ------------------------------------------------------------------
# Vector interface
# ------------------------------------------------------------------

def compute_style_rewards(
    j_lon: float,
    j_lat: float,
    v_ego: float,
    v_des: float,
    a_ego: float,
    lane_index: int,
    n_lanes: int,
    d_lat: float,
    delta_psi: float,
    d_lead: float,
    v_lead: float,
    has_lead: bool,
    hp: Dict,
) -> np.ndarray:
    """
    Compute the full 4D style reward vector from the current simulation state.

    Returns np.ndarray of shape (4,) with values in (0, 1]:
        [r_comfort, r_progress, r_lateral, r_spacing]
    """
    scaling = hp["reward_scaling"]

    r_comfort = compute_comfort(
        j_lon=j_lon,
        j_lat=j_lat,
        sigma_j_sq=scaling["sigma_j_sq"],
    )
    r_progress = compute_progress(
        v_ego=v_ego,
        v_des=v_des,
        a_ego=a_ego,
        lane_index=lane_index,
        n_lanes=n_lanes,
        beta=scaling["beta"],
        gamma_a=scaling["gamma_a"],
    )
    r_lateral = compute_lateral_discipline(
        d_lat=d_lat,
        delta_psi=delta_psi,
        sigma_d=hp.get("floor_values", {}).get("delta_d", 0.2),
        phi=scaling["phi"],
    )
    ttc = compute_ttc(d_lead=d_lead, v_ego=v_ego, v_lead=v_lead, has_lead=has_lead)
    r_spacing = compute_spacing(ttc=ttc, tau=scaling["tau"])

    return np.array([r_comfort, r_progress, r_lateral, r_spacing], dtype=np.float32)
