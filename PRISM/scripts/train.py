"""
PRISM training entry point.

Two-stage training:
    Stage 1 — Utility function initialisation (preference-based or DPMORL diversity).
    Stage 2 — Per-policy PPO training with CVaR Lagrangian constraint.

Usage
-----
    python scripts/train.py \\
        --config configs/prism_default.yaml \\
        --hyperparams hyperparams.json \\
        --nuplan_data_root /data/nuplan \\
        --n_policies 5 \\
        --total_timesteps 10000000 \\
        --output_dir runs/prism_run_001

Pre-requisite: hyperparams.json must exist (run compute_hyperparams.py first).
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

# ── Path setup so scripts can import from the PRISM package ──────────────────
_PRISM_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PRISM_ROOT))
sys.path.insert(0, str(_PRISM_ROOT.parent / "nuPlan"))

import numpy as np
import torch
import yaml

from prism.morl.utility_functions import (
    UtilityFunction,
    init_utility_functions_from_preferences,
    train_utility_functions_stage1,
)
from prism.morl.dpmorl_trainer import DPMORLTrainer
from prism.morl.cvar_lagrangian import CVaRLagrangian
from prism.utils.hyperparams import load_hyperparams
from prism.utils.zt_normaliser import ZtNormaliser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("prism.train")


# ─────────────────────────────────────────────────────────────────────────────
# Guard: hyperparams.json must exist
# ─────────────────────────────────────────────────────────────────────────────

def _require_hyperparams(path: str) -> None:
    result = subprocess.run(
        [sys.executable, str(_PRISM_ROOT / "compute_hyperparams.py"),
         "--check_only", "--output_path", path],
        capture_output=True,
    )
    if result.returncode != 0:
        print(
            f"\n[ERROR] hyperparams.json not found at '{path}'.\n"
            "Run compute_hyperparams.py first:\n\n"
            "  python compute_hyperparams.py \\\n"
            "      --output_path hyperparams.json \\\n"
            "      --n_warmup_rollouts 200 \\\n"
            "      --nuplan_data_root /path/to/nuplan\n"
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_env(cfg: dict, hp: dict, utility_fn, lambda_k: float, zt_normaliser):
    """
    Build a PRISMEnv by directly instantiating CaRL components.
    No Hydra / OmegaConf builders needed.

    Requires:
      cfg["cache_path"]  — path to the CaRL scenario cache directory
      cfg["log_names"]   — list of log names to sample (or [] for whole cache)

    Trajectory builder selection:
      carl_ppo  → ActionTrajectoryBuilder  (2D accel + steering)
      alpamayo  → UnicycleTrajectoryBuilder (128D flattened 64-waypoint trajectory)
    """
    from carl_nuplan.planning.gym.environment.scenario_sampler.cache_scenario_sampler import (
        CacheScenarioSampler,
    )
    from carl_nuplan.planning.gym.environment.simulation_builder.default_simulation_builder import (
        DefaultSimulationBuilder,
    )
    from carl_nuplan.planning.gym.environment.observation_builder.default.default_observation_builder import (
        DefaultObservationBuilder,
    )
    from carl_nuplan.planning.gym.environment.observation_builder.default.default_renderer import (
        DefaultRenderer,
    )
    from carl_nuplan.planning.gym.environment.helper.environment_area import (
        RectangleEnvironmentArea,
    )
    from prism.env.nuplan_env import make_prism_env

    cache_path = cfg["cache_path"]
    log_names = cfg.get("log_names", [])
    backend = cfg.get("model_backend", "carl_ppo")

    # ── Trajectory builder — backend-specific ─────────────────────────────────
    if backend == "alpamayo":
        from prism.env.unicycle_trajectory_builder import UnicycleTrajectoryBuilder
        alpamayo_cfg = cfg.get("alpamayo", {})
        trajectory_builder = UnicycleTrajectoryBuilder(
            n_waypoints=alpamayo_cfg.get("n_waypoints", 64),
            dt=0.1,
            accel_bounds=tuple(alpamayo_cfg.get("accel_bounds", [-9.8, 9.8])),
            curvature_bounds=tuple(alpamayo_cfg.get("curvature_bounds", [-0.33, 0.33])),
        )
        action_space_dim = cfg.get("action_space_dim", 128)
    else:
        from carl_nuplan.planning.gym.environment.trajectory_builder.action_trajectory_builder import (
            ActionTrajectoryBuilder,
        )
        trajectory_builder = ActionTrajectoryBuilder()
        action_space_dim = cfg.get("action_space_dim", 2)

    environment_area = RectangleEnvironmentArea(
        front=78.0, back=50.0, left=64.0, right=64.0
    )

    # ── Observation builder — backend-specific ────────────────────────────────
    camera_builder = None  # set below for alpamayo; set_env() called after env creation
    if backend == "alpamayo":
        from prism.observations.camera_observation_builder import CameraObservationBuilder
        obs_cfg = cfg.get("alpamayo_obs", {})
        camera_builder = CameraObservationBuilder(
            camera_channels=obs_cfg.get("camera_channels", ["CAM_F0", "CAM_L0", "CAM_R0"]),
            num_frames=obs_cfg.get("num_frames", 4),
            image_height=obs_cfg.get("image_height", 320),
            image_width=obs_cfg.get("image_width", 576),
            sensor_root_env_var=obs_cfg.get("sensor_root_env_var", "NUPLAN_SENSOR_ROOT"),
            action_space_dim=action_space_dim,
        )
        observation_builder = camera_builder
    else:
        renderer = DefaultRenderer(
            environment_area=environment_area,
            pixel_per_meter=2.0,
            max_vehicle_speed=30.0,
            max_pedestrian_speed=4.0,
            vehicle_scaling=1.0,
            pedestrian_scaling=1.0,
            static_scaling=1.0,
            include_speed_line=False,
            lane_connector_route=False,
        )
        observation_builder = DefaultObservationBuilder(
            environment_area=environment_area,
            renderer=renderer,
            obs_num_measurements=10,
            num_value_measurements=4,
            action_space_dim=action_space_dim,
            inference=False,
        )

    scenario_sampler = CacheScenarioSampler(
        log_names=log_names if log_names else [""],
        cache_path=cache_path,
        format="gz",
        ignore_log_names=(not log_names),
    )
    simulation_builder = DefaultSimulationBuilder(agent_type="tracks")

    env = make_prism_env(
        hp=hp,
        scenario_sampler=scenario_sampler,
        simulation_builder=simulation_builder,
        trajectory_builder=trajectory_builder,
        observation_builder=observation_builder,
        environment_area=environment_area,
        utility_fn=utility_fn.as_callable(),
        lambda_k=lambda_k,
        gamma=cfg.get("gamma", 0.99),
        zt_normaliser=zt_normaliser,
        terminate_on_failure=False,
    )

    # Inject env back-reference so the camera builder can resolve the log file path
    # from the current scenario at each step.
    if camera_builder is not None:
        camera_builder.set_env(env)

    return env


# ─────────────────────────────────────────────────────────────────────────────
# Preference vector helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_preference_vectors(n_policies: int, reward_dim: int) -> list:
    """
    Return the default preference vectors for K policies.

    Mirrors the defaults inside init_utility_functions_from_preferences so
    Stage 2 can retrieve w_k for each policy without re-running Stage 1.
    """
    if n_policies == 5 and reward_dim == 4:
        return [
            [0.55, 0.15, 0.15, 0.15],  # comfort
            [0.15, 0.55, 0.15, 0.15],  # progress
            [0.15, 0.15, 0.55, 0.15],  # lateral discipline
            [0.15, 0.15, 0.15, 0.55],  # spacing
            [0.25, 0.25, 0.25, 0.25],  # balanced
        ]
    base = 1.0 / reward_dim
    vecs = []
    for k in range(n_policies):
        vec = [base] * reward_dim
        vec[k % reward_dim] += 0.4 * (1.0 - base)
        total = sum(vec)
        vecs.append([v / total for v in vec])
    return vecs


# ─────────────────────────────────────────────────────────────────────────────
# Agent factory — returns a PRISMPolicyBase subclass for the chosen backend
# ─────────────────────────────────────────────────────────────────────────────

def _build_agent(
    env,
    cfg: dict,
    device: torch.device,
    policy_id: int = 0,
    backbone_phase: str = "a",
):
    """
    Instantiate the policy adapter for the backend named in cfg["model_backend"].

    backbone_phase — "a" (frozen backbone, default) or "b" (LoRA active).
    Passed through so the adapter knows which mode to operate in.
    Problem 4 will use this to toggle LoRA on/off inside AlpamayoAdapter.

    Returns a PRISMPolicyBase instance on the correct device.  Adding a new
    backbone means adding one branch here and one adapter under prism/models/.
    """
    backend = cfg.get("model_backend", "carl_ppo")

    if backend == "carl_ppo":
        from carl_nuplan.planning.gym.policy.ppo.ppo_config import GlobalConfig
        from carl_nuplan.planning.gym.policy.ppo.ppo_model import PPOPolicy
        from prism.models.carl_ppo.adapter import CaRLPPOAdapter

        config = GlobalConfig()
        config.initialize(**{k: v for k, v in cfg.items() if hasattr(config, k)})
        carl_policy = PPOPolicy(
            env.single_observation_space if hasattr(env, "single_observation_space")
            else env.observation_space,
            env.single_action_space if hasattr(env, "single_action_space")
            else env.action_space,
            config=config,
        ).to(device)
        return CaRLPPOAdapter(carl_policy)

    if backend == "alpamayo":
        from prism.models.alpamayo.adapter import AlpamayoAdapter
        from prism.models.common.qformer_critic import QFormerCritic

        alpamayo_cfg = cfg.get("alpamayo", {})
        reward_dim = cfg.get("reward_dim", 4)

        critic = QFormerCritic(
            backbone_hidden_dim=alpamayo_cfg.get("backbone_hidden_dim", 4096),
            n_queries=alpamayo_cfg.get("critic_n_queries", 8),
            query_dim=alpamayo_cfg.get("critic_query_dim", 256),
            n_heads=alpamayo_cfg.get("critic_n_heads", 8),
            style_dim=alpamayo_cfg.get("critic_style_dim", reward_dim * 2),
            value_hidden_dims=tuple(alpamayo_cfg.get("critic_value_hidden_dims", [256, 128])),
        ).to(device)

        return AlpamayoAdapter(
            policy_id=policy_id,
            action_dim=cfg.get("action_space_dim", 128),
            reward_dim=reward_dim,
            init_log_std=alpamayo_cfg.get("init_log_std", -0.5),
            critic=critic,
            extract_layers=alpamayo_cfg.get("critic_extract_layers", [22, 29, 35]),
            backbone_model_name=alpamayo_cfg.get("backbone_model_name"),
            backbone_phase=backbone_phase,
            alpamayo_cfg=alpamayo_cfg,
            observation_mode=cfg.get("observation_mode", "camera"),
        ).to(device)

    raise ValueError(f"Unknown model_backend: {backend!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1
# ─────────────────────────────────────────────────────────────────────────────

def run_stage1(cfg: dict, hp: dict, device: torch.device):
    """
    Initialise K utility functions (Stage 1).

    Returns (utility_fns, preference_vectors).  preference_vectors is a list
    of K lists (one per policy), used by Stage 2 to set w_k on adapters that
    need explicit style conditioning (e.g. AlpamayoAdapter / QFormerCritic).
    """
    n_policies = cfg.get("n_policies", 5)
    reward_dim = cfg.get("reward_dim", 4)
    mode = cfg.get("stage1", {}).get("mode", "preferences")

    logger.info(f"[Stage 1] mode={mode}  K={n_policies}  reward_dim={reward_dim}")

    preference_vectors = _get_preference_vectors(n_policies, reward_dim)

    if mode == "preferences":
        utility_fns = init_utility_functions_from_preferences(
            n_policies=n_policies,
            reward_dim=reward_dim,
            preference_vectors=preference_vectors,
            device=str(device),
        )
        logger.info("[Stage 1] Utility functions initialised from preference vectors.")
        return utility_fns, preference_vectors

    elif mode == "diversity":
        z_mu = hp.get("z_normalisation", {}).get("z_mu", [0.0] * reward_dim)
        z_sigma = hp.get("z_normalisation", {}).get("z_sigma", [1.0] * reward_dim)

        rng = np.random.default_rng(42)
        z_samples = rng.normal(
            loc=z_mu,
            scale=[max(s, 0.1) for s in z_sigma],
            size=(1000, reward_dim),
        ).astype(np.float32)

        utility_fns = init_utility_functions_from_preferences(
            n_policies=n_policies,
            reward_dim=reward_dim,
            preference_vectors=preference_vectors,
            device=str(device),
        )
        stage1_cfg = cfg.get("stage1", {})
        utility_fns = train_utility_functions_stage1(
            utility_fns=utility_fns,
            z_samples=z_samples,
            n_iters=stage1_cfg.get("n_iters", 2000),
            lr=stage1_cfg.get("lr", 1e-3),
            diversity_weight=stage1_cfg.get("diversity_weight", 1.0),
            device=str(device),
        )
        logger.info("[Stage 1] Utility functions trained with diversity loss.")
        return utility_fns, preference_vectors
    else:
        raise ValueError(f"Unknown stage1 mode: {mode}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2
# ─────────────────────────────────────────────────────────────────────────────

def run_stage2(
    utility_fns,
    cfg: dict,
    hp: dict,
    output_dir: Path,
    device: torch.device,
    preference_vectors: list = None,
    backbone_phase: str = "a",
    phase_a_dir: Path = None,
) -> None:
    """Train K policies sequentially with their respective utility functions."""
    n_policies = len(utility_fns)
    stage2_cfg = cfg.get("stage2", {})
    n_updates = stage2_cfg.get("n_updates", 10000)
    steps_per_update = stage2_cfg.get("steps_per_update", 512)
    lr = stage2_cfg.get("learning_rate", 3e-4)

    for k, utility_fn in enumerate(utility_fns):
        logger.info(f"\n{'='*60}")
        logger.info(f"[Stage 2] Training policy k={k} / {n_policies - 1}  "
                    f"[backbone phase {backbone_phase.upper()}]")
        logger.info(f"{'='*60}")

        policy_dir = output_dir / f"policy_{k}"
        policy_dir.mkdir(parents=True, exist_ok=True)

        zt_normaliser = ZtNormaliser.from_hyperparams(hp, beta=0.01)

        # Build environment for this policy
        env = _build_env(
            cfg=cfg, hp=hp,
            utility_fn=utility_fn,
            lambda_k=0.0,
            zt_normaliser=zt_normaliser,
        )

        # Build agent and set preference vector w_k for this policy
        agent = _build_agent(env, cfg, device, policy_id=k,
                             backbone_phase=backbone_phase)
        if preference_vectors is not None and hasattr(agent, "set_w_k"):
            w_k = torch.tensor(preference_vectors[k], dtype=torch.float32)
            agent.set_w_k(w_k)
            logger.info(f"[Stage 2] Policy {k}: w_k = {preference_vectors[k]}")

        # Phase B: load Phase A agent checkpoint (weights only — Lagrangian
        # starts fresh so the CVaR buffer reflects Phase B experience only)
        if backbone_phase == "b" and phase_a_dir is not None:
            ckpt = phase_a_dir / f"policy_{k}" / f"policy_{k}_model_final.pth"
            if ckpt.exists():
                agent.load_state_dict(
                    torch.load(ckpt, map_location=device), strict=False
                )
                logger.info(f"[Phase B] Loaded Phase A checkpoint: {ckpt}")
            else:
                logger.warning(
                    f"[Phase B] Phase A checkpoint not found at {ckpt} — "
                    "starting Phase B from scratch for policy {k}"
                )

        optimizer = torch.optim.Adam(
            list(agent.trainable_parameters()), lr=lr, eps=1e-5
        )

        # Trainer
        trainer = DPMORLTrainer(
            policy_id=k,
            agent=agent,
            optimizer=optimizer,
            env=env,
            utility_fn=utility_fn,
            hp=hp,
            cfg={**cfg, **stage2_cfg},
            output_dir=policy_dir,
            device=device,
        )

        summary = trainer.train(
            n_updates=n_updates,
            steps_per_update=steps_per_update,
        )

        logger.info(
            f"[Stage 2] Policy {k} done.  "
            f"Final lambda={summary['lambda_history'][-1]:.4f}  "
            f"Final CVaR={summary['cvar_history'][-1]:.4f}"
        )

        # Save summary
        import json
        with open(policy_dir / "training_summary.json", "w") as f:
            json.dump(
                {
                    "policy_id": k,
                    "lambda_history": summary["lambda_history"],
                    "cvar_history": summary["cvar_history"],
                    "episode_zts": [z.tolist() for z in summary["episode_zts"]],
                },
                f,
            )

        env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train PRISM K-policy MORL agent")
    parser.add_argument("--config", type=str, default="configs/prism_default.yaml")
    parser.add_argument("--hyperparams", type=str, default="hyperparams.json")
    parser.add_argument("--nuplan_data_root", type=str, default=None)
    parser.add_argument("--n_policies", type=int, default=None)
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stage1_only", action="store_true",
                        help="Run Stage 1 only and exit")
    parser.add_argument("--skip_stage1", action="store_true",
                        help="Load pre-saved utility functions and go straight to Stage 2")
    parser.add_argument("--utility_fn_dir", type=str, default=None,
                        help="Directory containing saved utility function checkpoints")
    # ── Backbone phase (Alpamayo experiment) ─────────────────────────────────
    parser.add_argument(
        "--model-backend", choices=["carl_ppo", "alpamayo"], default=None,
        help=(
            "Override the model_backend field in the config file. "
            "Useful for quick backend switches without swapping the whole config. "
            "Note: backend-specific hyperparams (alpamayo_obs, alpamayo.*) must "
            "still be present in the config file for the chosen backend."
        ),
    )
    parser.add_argument(
        "--backbone-phase", choices=["a", "b"], default=None,
        help=(
            "Backbone training phase for Alpamayo experiments. "
            "'a' = frozen backbone (only action head + critic train). "
            "'b' = LoRA active (requires --phase-a-dir). "
            "Omit for CaRL or non-phased experiments."
        ),
    )
    parser.add_argument(
        "--phase-a-dir", type=str, default=None,
        help=(
            "Path to a completed Phase A output directory. "
            "Required when --backbone-phase b. "
            "Example: runs/prism_alpamayo_run_001"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override config with CLI args
    if args.model_backend:
        cfg["model_backend"] = args.model_backend
    if args.nuplan_data_root:
        cfg["nuplan_data_root"] = args.nuplan_data_root
    if args.n_policies:
        cfg["n_policies"] = args.n_policies
    if args.total_timesteps:
        n_per_policy = args.total_timesteps // cfg.get("n_policies", 5)
        steps_per_update = cfg.get("stage2", {}).get("steps_per_update", 512)
        cfg.setdefault("stage2", {})["n_updates"] = n_per_policy // steps_per_update
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg["seed"] = args.seed

    # ── Backbone phase (Alpamayo only) ────────────────────────────────────────
    backbone_phase = args.backbone_phase  # "a", "b", or None
    phase_a_dir: Path = None

    if backbone_phase is not None:
        # Apply phase-specific CVaR buffer size from config
        phase_key = f"phase_{backbone_phase}"
        phase_cfg = cfg.get("backbone_phase", {}).get(phase_key, {})
        if "cvar_buffer_size" in phase_cfg:
            cfg["cvar_buffer_size"] = phase_cfg["cvar_buffer_size"]
            logger.info(
                f"[Backbone phase {backbone_phase.upper()}] "
                f"CVaR buffer size set to {cfg['cvar_buffer_size']}"
            )

        if backbone_phase == "b":
            if not args.phase_a_dir:
                print(
                    "\n[ERROR] --phase-a-dir is required when --backbone-phase b.\n"
                    "Point it at the output directory from your Phase A run.\n"
                    "Example: --phase-a-dir runs/prism_alpamayo_run_001\n"
                )
                sys.exit(1)
            phase_a_dir = Path(args.phase_a_dir)
            if not phase_a_dir.exists():
                print(f"\n[ERROR] Phase A directory not found: {phase_a_dir}\n")
                sys.exit(1)
            logger.info(f"[Phase B] Loading Phase A checkpoints from: {phase_a_dir}")

    # ── Guard: hyperparams must exist ─────────────────────────────────────────
    _require_hyperparams(args.hyperparams)

    # ── Load hyperparams ──────────────────────────────────────────────────────
    hp = load_hyperparams(args.hyperparams)

    # ── Seeding ───────────────────────────────────────────────────────────────
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Device ───────────────────────────────────────────────────────────────
    device_str = cfg.get("device", "cpu")
    if cfg.get("cuda", True) and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"Device: {device}")

    output_dir = Path(cfg.get("output_dir", "runs")) / cfg.get("experiment_name", "prism")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    # Phase B skips Stage 1 automatically — utility functions are loaded from
    # the Phase A output directory so they are identical across both phases.
    n_policies = cfg.get("n_policies", 5)
    reward_dim = cfg.get("reward_dim", 4)

    if backbone_phase == "b":
        logger.info("[Stage 1] Skipped for Phase B — loading from Phase A.")
        utility_fn_source = phase_a_dir / "stage1"
        utility_fns = []
        for k in range(n_policies):
            uf = UtilityFunction(reward_dim=reward_dim)
            uf_path = utility_fn_source / f"utility_fn_{k}.pth"
            uf.load_state_dict(torch.load(uf_path, map_location=device))
            uf.to(device)
            utility_fns.append(uf)
        preference_vectors = _get_preference_vectors(n_policies, reward_dim)

    elif args.skip_stage1 and args.utility_fn_dir:
        logger.info("[Stage 1] Skipped. Loading pre-saved utility functions.")
        utility_fns = []
        for k in range(n_policies):
            uf = UtilityFunction(reward_dim=reward_dim)
            uf_path = Path(args.utility_fn_dir) / f"utility_fn_{k}.pth"
            uf.load_state_dict(torch.load(uf_path, map_location=device))
            uf.to(device)
            utility_fns.append(uf)
        preference_vectors = _get_preference_vectors(n_policies, reward_dim)

    else:
        utility_fns, preference_vectors = run_stage1(cfg, hp, device)

        # Save Stage 1 utility functions
        stage1_dir = output_dir / "stage1"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        for k, uf in enumerate(utility_fns):
            torch.save(uf.state_dict(), stage1_dir / f"utility_fn_{k}.pth")
        logger.info(f"[Stage 1] Utility functions saved to {stage1_dir}")

    if args.stage1_only:
        logger.info("[Stage 1] --stage1_only flag set. Exiting after Stage 1.")
        return

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    run_stage2(
        utility_fns=utility_fns,
        cfg=cfg,
        hp=hp,
        output_dir=output_dir,
        device=device,
        preference_vectors=preference_vectors,
        backbone_phase=backbone_phase or "a",
        phase_a_dir=phase_a_dir,
    )

    logger.info(f"\nTraining complete.  Results in: {output_dir}")


if __name__ == "__main__":
    main()
