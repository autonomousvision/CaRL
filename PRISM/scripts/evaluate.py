"""
Evaluate trained PRISM policies on the nuPlan Val14 benchmark.

Usage
-----
    python scripts/evaluate.py \\
        --policy_dir runs/prism_run_001 \\
        --hyperparams hyperparams.json \\
        --config configs/prism_default.yaml \\
        --n_episodes 200 \\
        --output_dir runs/prism_run_001/eval

Outputs (per policy k):
    eval_results_k.json  — episode-level metrics
    pareto_points.npy    — mean z_T per policy (for Pareto front plotting)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_PRISM_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PRISM_ROOT))
sys.path.insert(0, str(_PRISM_ROOT.parent / "nuPlan"))

import numpy as np
import torch
import yaml

from prism.morl.utility_functions import UtilityFunction
from prism.morl.cvar_lagrangian import compute_episode_cost, cvar_from_episodes
from prism.utils.hyperparams import load_hyperparams
from prism.utils.zt_normaliser import ZtNormaliser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prism.evaluate")


def _load_policy(policy_dir: Path, k: int, cfg: dict, device: torch.device):
    """Load the final model checkpoint for policy k."""
    from carl_nuplan.planning.gym.policy.ppo.ppo_config import GlobalConfig
    from carl_nuplan.planning.gym.policy.ppo.ppo_model import PPOPolicy
    import gymnasium as gym

    # Reconstruct a dummy env just for the observation/action spaces
    obs_space = gym.spaces.Dict({
        "bev_semantics": gym.spaces.Box(
            0, 255,
            shape=(cfg.get("bev_semantics_channels", 15),
                   cfg.get("bev_semantics_height", 192),
                   cfg.get("bev_semantics_width", 192)),
            dtype=np.uint8,
        ),
        "measurements": gym.spaces.Box(
            -np.inf, np.inf,
            shape=(cfg.get("obs_num_measurements", 10),),
            dtype=np.float32,
        ),
        "value_measurements": gym.spaces.Box(
            -np.inf, np.inf,
            shape=(cfg.get("num_value_measurements", 4),),
            dtype=np.float32,
        ),
    })
    act_space = gym.spaces.Box(
        -1.0, 1.0, shape=(cfg.get("action_space_dim", 2),), dtype=np.float32
    )

    config = GlobalConfig()
    config.initialize(**{k_: v for k_, v in cfg.items() if hasattr(config, k_)})
    agent = PPOPolicy(obs_space, act_space, config=config).to(device)

    # Find the final model checkpoint
    ckpt_dir = policy_dir / f"policy_{k}"
    ckpts = sorted(ckpt_dir.glob(f"policy_{k}_model_final.pth"))
    if not ckpts:
        ckpts = sorted(ckpt_dir.glob(f"policy_{k}_model_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    ckpt_path = ckpts[-1]
    agent.load_state_dict(torch.load(ckpt_path, map_location=device))
    agent.eval()
    logger.info(f"[Policy {k}] Loaded checkpoint: {ckpt_path}")

    # Load utility function
    util_ckpts = sorted(ckpt_dir.glob(f"policy_{k}_utility_final.pth"))
    if not util_ckpts:
        util_ckpts = sorted(ckpt_dir.glob(f"policy_{k}_utility_*.pth"))
    uf = UtilityFunction(reward_dim=cfg.get("reward_dim", 4))
    if util_ckpts:
        uf.load_state_dict(torch.load(util_ckpts[-1], map_location=device))
    uf.to(device)
    uf.eval()

    return agent, uf


def _run_episodes(env, agent, device, n_episodes: int):
    """Roll out the agent for n_episodes and collect metrics."""
    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False

        episode_rewards = []
        episode_r_vecs = []
        episode_step_costs = []
        episode_zt = None

        lstm_state = None  # CaRL PPO uses LSTM

        while not done:
            next_obs_t = {
                k: torch.from_numpy(np.array(v)).unsqueeze(0).to(device)
                for k, v in obs.items()
            }
            with torch.no_grad():
                action, _, _, _, *_ = agent.forward(
                    next_obs_t,
                    lstm_state=lstm_state,
                    done=torch.zeros(1, device=device),
                )

            act_np = action.squeeze(0).cpu().numpy()
            obs, reward, term, trunc, info = env.step(act_np)
            done = term or trunc

            episode_rewards.append(float(reward))
            episode_step_costs.append(float(info.get("safety_cost", 0.0)))

            if "style_reward" in info:
                pass  # popped in env step; use episode_zt instead
            if done and "episode_zt" in info:
                episode_zt = info["episode_zt"].copy()

        cum_cost = compute_episode_cost(
            episode_step_costs, gamma=env._gamma if hasattr(env, "_gamma") else 0.99
        )
        results.append({
            "episode": ep,
            "total_reward": float(sum(episode_rewards)),
            "n_steps": len(episode_rewards),
            "cumulative_cost": float(cum_cost),
            "episode_zt": episode_zt.tolist() if episode_zt is not None else None,
        })

        if (ep + 1) % 20 == 0:
            avg_r = np.mean([r["total_reward"] for r in results])
            avg_cost = np.mean([r["cumulative_cost"] for r in results])
            logger.info(f"  Episode {ep+1}/{n_episodes}  avg_return={avg_r:.2f}  avg_cost={avg_cost:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PRISM policies")
    parser.add_argument("--policy_dir", required=True, type=str)
    parser.add_argument("--hyperparams", type=str, default="hyperparams.json")
    parser.add_argument("--config", type=str, default="configs/prism_default.yaml")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--n_policies", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    hp = load_hyperparams(args.hyperparams)

    policy_dir = Path(args.policy_dir)
    output_dir = Path(args.output_dir) if args.output_dir else policy_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")  # evaluation typically on CPU
    if cfg.get("cuda") and torch.cuda.is_available():
        device = torch.device(cfg.get("device", "cuda:0"))

    n_policies = args.n_policies or cfg.get("n_policies", 5)

    all_pareto_points = []

    for k in range(n_policies):
        logger.info(f"\n[Evaluate] Policy k={k}")

        try:
            agent, utility_fn = _load_policy(policy_dir, k, cfg, device)
        except FileNotFoundError as e:
            logger.warning(f"  Skipping policy {k}: {e}")
            continue

        zt_normaliser = ZtNormaliser.from_hyperparams(hp)

        # Build eval env
        from scripts.train import _build_env
        env = _build_env(
            cfg=cfg, hp=hp,
            utility_fn=utility_fn,
            lambda_k=0.0,          # no Lagrangian during eval
            zt_normaliser=zt_normaliser,
        )

        results = _run_episodes(env, agent, device, n_episodes=args.n_episodes)
        env.close()

        # Save episode results
        with open(output_dir / f"eval_results_{k}.json", "w") as f:
            json.dump(results, f, indent=2)

        # Pareto point: mean z_T across episodes
        zts = [r["episode_zt"] for r in results if r["episode_zt"] is not None]
        if zts:
            mean_zt = np.mean(zts, axis=0)
            all_pareto_points.append(mean_zt)
            logger.info(f"  Mean z_T: {mean_zt.round(3)}")

        # CVaR at alpha=0.95
        costs = [r["cumulative_cost"] for r in results]
        cvar95 = cvar_from_episodes(costs, alpha=0.95)
        from prism.utils.hyperparams import get_epsilon
        eps95 = get_epsilon(hp, 0.95)
        logger.info(f"  CVaR@0.95={cvar95:.4f}  epsilon@0.95={eps95:.4f}  safe={cvar95 <= eps95}")

    # Save all Pareto points
    if all_pareto_points:
        np.save(output_dir / "pareto_points.npy", np.array(all_pareto_points))
        logger.info(f"\nPareto points saved to {output_dir / 'pareto_points.npy'}")

    logger.info(f"\nEvaluation complete. Results in: {output_dir}")


if __name__ == "__main__":
    main()
