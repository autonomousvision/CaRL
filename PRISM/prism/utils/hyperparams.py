"""
Load and validate the cached PRISM hyperparameter file.
"""
import json
from pathlib import Path
from typing import Any, Dict


def load_hyperparams(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"hyperparams.json not found at {path}.\n"
            "Run: python compute_hyperparams.py --output_path hyperparams.json ..."
        )
    with open(p) as f:
        hp = json.load(f)
    _validate(hp)
    return hp


def _validate(hp: Dict[str, Any]) -> None:
    required_top = [
        "reward_scaling", "epsilon_curve", "lead_times",
        "indicator_weights", "indicator_caps", "outcome_weights",
        "z_normalisation", "alpha_curriculum", "safety_thresholds",
    ]
    for key in required_top:
        if key not in hp:
            raise KeyError(f"hyperparams.json is missing required key: '{key}'")

    required_scaling = ["sigma_j_sq", "beta", "gamma_a", "phi", "tau"]
    for key in required_scaling:
        if key not in hp["reward_scaling"]:
            raise KeyError(f"reward_scaling is missing key: '{key}'")


def get_epsilon(hp: Dict[str, Any], alpha: float) -> float:
    """Look up CVaR tolerance for a given alpha.  Keys are strings."""
    curve = hp["epsilon_curve"]
    key = str(round(alpha, 2))
    if key not in curve:
        # nearest key fallback
        keys = sorted(float(k) for k in curve)
        nearest = min(keys, key=lambda k: abs(k - alpha))
        key = str(round(nearest, 2))
    return float(curve[key])
