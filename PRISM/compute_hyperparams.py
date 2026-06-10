"""
compute_hyperparams.py
======================
Computes all empirical hyperparameters for the PRISM framework
from IDM warm-up rollouts and stores them to a single JSON file.

During training, this script is called ONLY if the hyperparameter
file is missing. Once computed, the cached values are reused for
all policies and all experiments.

Usage
-----
    # Compute and cache (only run once)
    python compute_hyperparams.py --output_path hyperparams.json \
        --n_warmup_rollouts 200 --nuplan_data_root /path/to/nuplan

    # Check if already computed (used by training script)
    python compute_hyperparams.py --check_only --output_path hyperparams.json

Cached values
-------------
Reward scaling:
    sigma_j_sq   : empirical variance of (j_lon^2 + j_lat^2) -- comfort
    beta         : speed shortfall scaling for progress
    gamma_a      : acceleration scaling for progress
    phi          : heading error scaling for lateral discipline
    tau          : TTC scaling for spacing

CVaR epsilon curve:
    epsilon_curve : dict mapping alpha -> CVaR_alpha(C^IDM)
                    at alphas [0.20, 0.30, ..., 0.95]

Safety cost lead times (T_{j->i}):
    lead_times   : dict mapping indicator_name -> {outcome_name: T_mean}

z_t normalisation statistics:
    z_mu         : list of 4 mean values (one per style objective)
    z_sigma      : list of 4 std values (one per style objective)

All time values are in timesteps (nuPlan runs at 10 Hz).
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# ── nuPlan / PRISM imports (filled in during implementation) ──────────────────
# from nuplan.planning.simulation... import ...
# from prism.env.nuplan_env import PRISMEnv
# from prism.rewards import compute_style_rewards, compute_safety_cost
# from nuplan.planning.simulation.planner.idm_planner import IDMPlanner


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ALPHA_VALUES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                0.80, 0.85, 0.90, 0.95, 0.99]

# Indicator -> list of outcomes it can precede
# Used for weight derivation and lead-time estimation
INDICATOR_OUTCOME_MAP = {
    "ttc":        ["vru_collision", "vehicle_collision", "object_collision"],
    "thw":        ["vru_collision", "vehicle_collision", "object_collision"],
    "speed":      ["vru_collision", "vehicle_collision", "wrong_direction"],
    "blind_spot": ["vru_collision", "vehicle_collision"],
    "red_light":  ["vru_collision", "vehicle_collision", "red_light_violation"],
}

# Outcome weights (Table I in paper)
OUTCOME_WEIGHTS = {
    "vru_collision":        100,
    "wrong_direction":      100,
    "vehicle_collision":    80,
    "red_light_violation":  80,
    "stop_sign_violation":  70,
    "drivable_area":        65,
    "object_collision":     40,
}


# ─────────────────────────────────────────────────────────────────────────────
# Core computation functions
# ─────────────────────────────────────────────────────────────────────────────

def collect_idm_rollouts(n_rollouts: int,
                         nuplan_data_root: str,
                         scenario_filter: str = "val14") -> List[Dict]:
    """
    Collect n_rollouts episodes using nuPlan's IDM planner.
    Returns a list of episode dicts, each containing:
        - j_lon       : np.ndarray (T,)  longitudinal jerk
        - j_lat       : np.ndarray (T,)  lateral jerk
        - v_ego       : np.ndarray (T,)
        - v_des       : np.ndarray (T,)  desired speed (regime-conditioned)
        - a_ego       : np.ndarray (T,)  longitudinal acceleration
        - d_lat       : np.ndarray (T,)  lateral deviation from lane centre
        - delta_psi   : np.ndarray (T,)  heading angle error
        - ttc         : np.ndarray (T,)  time-to-collision
        - style_r     : np.ndarray (T,4) all four style rewards
        - safety_cost : np.ndarray (T,)  per-timestep safety cost c_t
        - cum_cost    : float            episode cumulative C = sum(gamma^t * c_t)
        - indicator_events : Dict[str, List[int]]  timesteps where each
                                                   indicator fired
        - outcome_events   : Dict[str, List[int]]  timesteps where each
                                                   outcome fired
    """
    # TODO: implement using nuPlan simulation API
    # Pseudocode:
    #   env = PRISMEnv(nuplan_data_root, scenario_filter)
    #   planner = IDMPlanner(...)
    #   for _ in range(n_rollouts):
    #       obs = env.reset()
    #       done = False
    #       episode = init_episode_dict()
    #       while not done:
    #           action = planner.plan(obs)
    #           obs, r_vec, cost, done, info = env.step(action)
    #           append_to_episode(episode, info)
    #       episodes.append(finalise_episode(episode))
    raise NotImplementedError("Implement using nuPlan simulation API")


def compute_reward_scaling(episodes: List[Dict]) -> Dict:
    """
    Compute scaling parameters for all four reward functions.

    sigma_j_sq : Var(j_lon^2 + j_lat^2)
    beta       : set so 1-sigma shortfall -> reward = e^{-1}
                 i.e. beta = 1.0 (normalised shortfall is already in [0,1])
    gamma_a    : set so mean |a_ego| -> reward = 1 - e^{-1} ~ 0.63
                 i.e. gamma_a = mean(|a_ego|)
    phi        : set so 1-sigma |delta_psi| -> reward = e^{-1}
                 i.e. phi = std(|delta_psi|)
    tau        : set so mean(TTC) when TTC < 10s -> reward ~0.63
                 i.e. tau = mean(TTC[TTC < 10])
    """
    all_jerk_sq   = np.concatenate([ep["j_lon"]**2 + ep["j_lat"]**2
                                     for ep in episodes])
    all_a_ego     = np.concatenate([np.abs(ep["a_ego"]) for ep in episodes])
    all_delta_psi = np.concatenate([np.abs(ep["delta_psi"]) for ep in episodes])
    all_ttc       = np.concatenate([ep["ttc"][ep["ttc"] < 10]
                                     for ep in episodes])

    sigma_j_sq = float(np.var(all_jerk_sq))
    # Avoid division by zero
    sigma_j_sq = max(sigma_j_sq, 1e-6)

    gamma_a = float(np.mean(all_a_ego))
    gamma_a = max(gamma_a, 1e-6)

    phi = float(np.std(all_delta_psi))
    phi = max(phi, 1e-4)

    tau = float(np.mean(all_ttc)) if len(all_ttc) > 0 else 2.0
    tau = max(tau, 0.1)

    # beta: normalised speed shortfall is already unit-free;
    # set beta = 0.5 so a 50% shortfall -> reward = e^{-1}
    beta = 0.5

    return {
        "sigma_j_sq": sigma_j_sq,
        "beta":       beta,
        "gamma_a":    gamma_a,
        "phi":        phi,
        "tau":        tau,
    }


def compute_cvar_epsilon_curve(episodes: List[Dict],
                               gamma: float = 0.99) -> Dict[str, float]:
    """
    Compute the IDM CVaR curve: alpha -> CVaR_alpha(C^IDM).
    Returns a dict with string keys (for JSON serialisation).
    """
    cum_costs = np.array([ep["cum_cost"] for ep in episodes])
    epsilon_curve = {}
    for alpha in ALPHA_VALUES:
        sorted_costs = np.sort(cum_costs)[::-1]  # descending
        n_tail = max(1, int(np.ceil((1 - alpha) * len(sorted_costs))))
        cvar = float(np.mean(sorted_costs[:n_tail]))
        epsilon_curve[str(alpha)] = cvar
    return epsilon_curve


def compute_lead_times(episodes: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    For each indicator-outcome pair, estimate the mean lead time T_{j->i}:
    the mean number of timesteps between an indicator firing and the
    next occurrence of its associated outcome.

    Returns nested dict: lead_times[indicator][outcome] = T_mean (timesteps)
    """
    lead_times: Dict[str, Dict[str, float]] = {}

    for indicator, outcomes in INDICATOR_OUTCOME_MAP.items():
        lead_times[indicator] = {}
        for outcome in outcomes:
            deltas = []
            for ep in episodes:
                ind_ts  = ep["indicator_events"].get(indicator, [])
                out_ts  = ep["outcome_events"].get(outcome, [])
                for t_ind in ind_ts:
                    # Find the next outcome event after this indicator
                    future = [t_out - t_ind for t_out in out_ts
                              if t_out > t_ind]
                    if future:
                        deltas.append(min(future))
            if deltas:
                lead_times[indicator][outcome] = float(np.mean(deltas))
            else:
                # Fallback: use physical reasoning if no observations
                lead_times[indicator][outcome] = _fallback_lead_time(
                    indicator, outcome)

    return lead_times


def _fallback_lead_time(indicator: str, outcome: str) -> float:
    """
    Physics-based fallback lead times (in timesteps at 10 Hz)
    when insufficient observations are available.
    """
    fallbacks = {
        ("ttc",        "vru_collision"):     15,   # 1.5s threshold
        ("ttc",        "vehicle_collision"): 15,
        ("ttc",        "object_collision"):  15,
        ("thw",        "vru_collision"):     20,   # 2.0s threshold
        ("thw",        "vehicle_collision"): 20,
        ("thw",        "object_collision"):  20,
        ("speed",      "vru_collision"):     30,
        ("speed",      "vehicle_collision"): 30,
        ("speed",      "wrong_direction"):   50,
        ("blind_spot", "vru_collision"):     30,
        ("blind_spot", "vehicle_collision"): 30,
        ("red_light",  "vru_collision"):     40,
        ("red_light",  "vehicle_collision"): 40,
        ("red_light",  "red_light_violation"): 40,
    }
    return float(fallbacks.get((indicator, outcome), 30))


def compute_zt_normalisation(episodes: List[Dict]) -> Dict:
    """
    Compute mean and std of cumulative style return z_T for each
    of the 4 objectives, from IDM warm-up rollouts.
    Used to initialise z_t normalisation before training.
    """
    z_finals = np.array([ep["style_r"].sum(axis=0) for ep in episodes])
    # z_finals shape: (N_rollouts, 4)
    z_mu    = z_finals.mean(axis=0).tolist()
    z_sigma = z_finals.std(axis=0).tolist()
    # Avoid zero std
    z_sigma = [max(s, 1e-6) for s in z_sigma]
    return {"z_mu": z_mu, "z_sigma": z_sigma}


def derive_indicator_weights(lead_times: Dict[str, Dict[str, float]]) -> Dict:
    """
    Compute w_j^lead for each indicator using Eq. (16) from the paper:
        w_j = sum_{i in O_j} W_i / (|O_j| * T_{j->i})
    Also compute the episode-level cap:
        cap_j = mean(W_i for i in O_j)
    """
    weights = {}
    caps    = {}
    for indicator, outcomes in INDICATOR_OUTCOME_MAP.items():
        n = len(outcomes)
        w = 0.0
        for outcome in outcomes:
            W_i   = OUTCOME_WEIGHTS[outcome]
            T_jti = lead_times[indicator][outcome]
            w    += W_i / (n * T_jti)
        weights[indicator] = float(w)
        caps[indicator]    = float(
            np.mean([OUTCOME_WEIGHTS[o] for o in outcomes]))
    return {"indicator_weights": weights, "indicator_caps": caps}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute and cache PRISM hyperparameters from IDM rollouts")
    parser.add_argument("--output_path", type=str,
                        default="hyperparams.json",
                        help="Path to output JSON file")
    parser.add_argument("--n_warmup_rollouts", type=int, default=200,
                        help="Number of IDM warm-up rollouts to collect")
    parser.add_argument("--nuplan_data_root", type=str,
                        default="/data/nuplan",
                        help="Root directory of nuPlan dataset")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (must match training)")
    parser.add_argument("--check_only", action="store_true",
                        help="Exit with code 0 if file exists, 1 if missing")
    args = parser.parse_args()

    # ── Check-only mode ───────────────────────────────────────────────────────
    if args.check_only:
        exists = Path(args.output_path).exists()
        print(f"Hyperparameter file {'found' if exists else 'NOT found'}: "
              f"{args.output_path}")
        exit(0 if exists else 1)

    # ── Skip if already computed ──────────────────────────────────────────────
    if Path(args.output_path).exists():
        print(f"[compute_hyperparams] File already exists at "
              f"{args.output_path}. Skipping computation.")
        print("  Delete the file to force recomputation.")
        return

    print(f"[compute_hyperparams] Collecting {args.n_warmup_rollouts} "
          f"IDM rollouts...")
    episodes = collect_idm_rollouts(
        n_rollouts=args.n_warmup_rollouts,
        nuplan_data_root=args.nuplan_data_root,
    )
    print(f"[compute_hyperparams] Collected {len(episodes)} episodes.")

    print("[compute_hyperparams] Computing reward scaling parameters...")
    scaling = compute_reward_scaling(episodes)

    print("[compute_hyperparams] Computing CVaR epsilon curve...")
    epsilon_curve = compute_cvar_epsilon_curve(episodes, gamma=args.gamma)

    print("[compute_hyperparams] Estimating indicator lead times...")
    lead_times = compute_lead_times(episodes)

    print("[compute_hyperparams] Deriving indicator weights...")
    ind_params = derive_indicator_weights(lead_times)

    print("[compute_hyperparams] Computing z_t normalisation statistics...")
    z_norm = compute_zt_normalisation(episodes)

    # ── Assemble and save ─────────────────────────────────────────────────────
    hyperparams = {
        "metadata": {
            "n_warmup_rollouts": len(episodes),
            "gamma":             args.gamma,
            "nuplan_data_root":  args.nuplan_data_root,
        },
        "reward_scaling":     scaling,
        "epsilon_curve":      epsilon_curve,
        "lead_times":         lead_times,
        "indicator_weights":  ind_params["indicator_weights"],
        "indicator_caps":     ind_params["indicator_caps"],
        "outcome_weights":    OUTCOME_WEIGHTS,
        "z_normalisation":    z_norm,
        "alpha_curriculum": {
            "alpha_start": 0.20,
            "alpha_end":   0.95,
        },
        "regime_detection": {
            "congestion_speed_fraction": 0.5,
            "observation_horizon_m":     50.0,
        },
        "floor_values": {
            "delta_d": 0.3,
            "delta_s": 0.2,
        },
        "safety_thresholds": {
            "ttc_threshold_s":        1.5,
            "thw_threshold_s":        2.0,
            "lane_change_lateral_vel_threshold_ms": 0.3,
        },
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(hyperparams, f, indent=2)

    print(f"[compute_hyperparams] Saved to {output_path}")
    print("\nComputed values:")
    print(f"  sigma_j_sq : {scaling['sigma_j_sq']:.6f}")
    print(f"  beta       : {scaling['beta']:.4f}")
    print(f"  gamma_a    : {scaling['gamma_a']:.4f}")
    print(f"  phi        : {scaling['phi']:.4f}")
    print(f"  tau        : {scaling['tau']:.4f}")
    print(f"  epsilon@0.95: {epsilon_curve['0.95']:.4f}")
    print(f"  indicator weights: {ind_params['indicator_weights']}")


if __name__ == "__main__":
    main()
