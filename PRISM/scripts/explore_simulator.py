"""
Simulator exploration and visualisation script for PRISM.

Two modes
---------
--mode db     (DEFAULT) Load scenarios directly from nuPlan .db files.
              No caching step needed. Use this for exploration.

--mode cache  Load from a pre-built CaRL .gz cache.
              Faster per-episode startup. Use this once you have a cache
              (build it with scripts/build_cache.py --max_scenarios 2000).

Usage
-----
    # Explore directly from the mini dataset — no caching needed:
    python scripts/explore_simulator.py \\
        --mode db \\
        --data_root $NUPLAN_DATA_ROOT/nuplan-v1.1/splits/mini \\
        --map_root  $NUPLAN_MAPS_ROOT \\
        --policy    random \\
        --n_episodes 3

    # Or from a pre-built cache:
    python scripts/explore_simulator.py \\
        --mode cache \\
        --cache_path $NUPLAN_EXP_ROOT/mini_cache \\
        --policy random \\
        --n_episodes 3

Outputs
-------
  explore_output/episode_NNN.avi         — BEV video (semantic raster)
  explore_output/episode_NNN_summary.png — per-step reward + z_t breakdown
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_PRISM_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PRISM_ROOT))
sys.path.insert(0, str(_PRISM_ROOT.parent / "nuPlan"))

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("prism.explore")


# ─────────────────────────────────────────────────────────────────────────────
# Policy helpers
# ─────────────────────────────────────────────────────────────────────────────

def _random_policy(obs, action_space):
    return action_space.sample()

def _idle_policy(obs, action_space):
    return np.zeros(action_space.shape, dtype=np.float32)

def _cruise_policy(obs, action_space):
    act = np.zeros(action_space.shape, dtype=np.float32)
    act[0] = 0.3
    return act

POLICIES = {"random": _random_policy, "idle": _idle_policy, "cruise": _cruise_policy}


# ─────────────────────────────────────────────────────────────────────────────
# BEV rendering
# ─────────────────────────────────────────────────────────────────────────────

def _bev_to_rgb(bev: np.ndarray) -> np.ndarray:
    if bev.ndim == 3 and bev.shape[0] > 3:
        class_map = np.argmax(bev.astype(np.float32), axis=0)
    elif bev.ndim == 3 and bev.shape[-1] == 3:
        return bev
    else:
        class_map = bev[0]
    PALETTE = np.array([
        [200, 200, 200], [255, 255, 255], [ 50, 200,  50],
        [  0, 128, 255], [255,  80,  80], [255, 165,   0],
        [128,   0, 128], [255, 255,   0], [200, 100,  50],
        [ 30,  30,  30], [150, 200, 255], [200, 255, 200],
        [255, 200, 200], [180, 180,  80], [ 80, 180, 180],
    ], dtype=np.uint8)
    return PALETTE[class_map % len(PALETTE)]


def _overlay_text(frame: np.ndarray, lines: list) -> np.ndarray:
    try:
        import cv2
        frame = frame.copy()
        y = 15
        for txt, color in lines:
            cv2.putText(frame, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, color, 1, cv2.LINE_AA)
            y += 13
    except ImportError:
        pass
    return frame


def _save_video(frames: list, path: Path, fps: int = 10) -> None:
    if not frames:
        return
    try:
        import cv2
        H, W = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        out = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        logger.info(f"  Video → {path}")
    except Exception as e:
        logger.warning(f"  Could not save video: {e}")
        try:
            from PIL import Image
            fd = path.parent / path.stem
            fd.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(frames):
                Image.fromarray(f).save(fd / f"frame_{i:05d}.png")
            logger.info(f"  Frames → {fd}/")
        except Exception:
            pass


def _plot_summary(zt, rvec, cost, path: Path, ep: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        obj = ["Comfort", "Progress", "Lateral", "Spacing"]
        col = ["#2A9D8F", "#E63946", "#F4A261", "#457B9D"]
        T = len(rvec)
        steps = np.arange(T)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"Episode {ep} — PRISM reward breakdown", fontsize=13)
        ax = axes[0]
        for i, (n, c) in enumerate(zip(obj, col)):
            ax.plot(zt[:T, i], label=n, color=c, linewidth=1.5)
        ax.set_xlabel("Step"); ax.set_ylabel("z_t"); ax.set_title("Cumulative z_t")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax = axes[1]
        bottom = np.zeros(T)
        for i, (n, c) in enumerate(zip(obj, col)):
            ax.bar(steps, rvec[:, i], bottom=bottom, label=n, color=c, width=1.0, alpha=0.85)
            bottom += rvec[:, i]
        ax.set_xlabel("Step"); ax.set_ylabel("Reward")
        ax.set_title("Per-step style rewards"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax = axes[2]
        ax.fill_between(steps, cost, color="#E63946", alpha=0.6, label="c_t")
        ax.set_xlabel("Step"); ax.set_ylabel("c_t")
        ax.set_title("Per-step safety cost"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Summary plot → {path}")
    except Exception as e:
        logger.warning(f"  Could not save summary plot: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder hyperparams
# ─────────────────────────────────────────────────────────────────────────────

_PLACEHOLDER_HP = {
    "reward_scaling": {"sigma_j_sq": 1.0, "beta": 0.5, "gamma_a": 1.0, "phi": 0.3, "tau": 2.0},
    "epsilon_curve": {str(round(a, 2)): 50.0 for a in np.arange(0.20, 1.01, 0.05)},
    "lead_times": {},
    "indicator_weights": {"ttc": 2.0, "thw": 1.5, "speed": 1.0, "blind_spot": 1.5, "red_light": 2.0},
    "indicator_caps": {"ttc": 80.0, "thw": 73.0, "speed": 93.0, "blind_spot": 90.0, "red_light": 80.0},
    "outcome_weights": {"vru_collision": 100, "vehicle_collision": 80, "wrong_direction": 100,
                        "red_light_violation": 80, "drivable_area": 65, "object_collision": 40},
    "z_normalisation": {"z_mu": [0.0]*4, "z_sigma": [1.0]*4},
    "alpha_curriculum": {"alpha_start": 0.20, "alpha_end": 0.95},
    "safety_thresholds": {"ttc_threshold_s": 1.5, "thw_threshold_s": 2.0},
    "regime_detection": {"congestion_speed_fraction": 0.5, "observation_horizon_m": 50.0},
    "floor_values": {"delta_d": 0.2},
}


# ─────────────────────────────────────────────────────────────────────────────
# Scenario samplers
# ─────────────────────────────────────────────────────────────────────────────

class _FixedListSampler:
    """Wraps a list of pre-loaded scenarios. No cache required."""

    def __init__(self, scenarios: list) -> None:
        self._scenarios = scenarios
        self._idx = 0

    def sample(self, seed=None):
        rng = np.random.default_rng(seed)
        return self._scenarios[int(rng.integers(0, len(self._scenarios)))]


def _build_db_sampler(data_root: str, map_root: str, map_version: str,
                      n_scenarios: int = 20):
    """Load n_scenarios directly from .db files — no cache needed."""
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    logger.info(f"Loading up to {n_scenarios} scenarios from {data_root} …")
    builder = NuPlanScenarioBuilder(
        data_root=data_root,
        map_root=map_root,
        sensor_root=data_root,
        db_files=None,
        map_version=map_version,
        include_cameras=False,
        max_workers=1,
        verbose=False,
    )
    scenario_filter = ScenarioFilter(
        scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
        num_scenarios_per_type=None, limit_total_scenarios=n_scenarios,
        timestamp_threshold_s=None, ego_displacement_minimum_m=None,
        ego_start_speed_threshold=None, ego_stop_speed_threshold=None,
        speed_noise_tolerance=None, expand_scenarios=False,
        remove_invalid_goals=True, shuffle=True,
    )
    scenarios = builder.get_scenarios(scenario_filter, Sequential())
    logger.info(f"Loaded {len(scenarios)} scenarios.")
    return _FixedListSampler(scenarios)


def _build_cache_sampler(cache_path: str, log_names: list):
    from carl_nuplan.planning.gym.environment.scenario_sampler.cache_scenario_sampler import CacheScenarioSampler
    return CacheScenarioSampler(
        log_names=log_names if log_names else [""],
        cache_path=cache_path,
        format="gz",
        ignore_log_names=(not log_names),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Env builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_env(scenario_sampler, hp: dict, gamma: float = 0.99,
               agent_type: str = "tracks"):
    from carl_nuplan.planning.gym.environment.simulation_builder.default_simulation_builder import DefaultSimulationBuilder
    from carl_nuplan.planning.gym.environment.trajectory_builder.action_trajectory_builder import ActionTrajectoryBuilder
    from carl_nuplan.planning.gym.environment.observation_builder.default.default_observation_builder import DefaultObservationBuilder
    from carl_nuplan.planning.gym.environment.observation_builder.default.default_renderer import DefaultRenderer
    from carl_nuplan.planning.gym.environment.helper.environment_area import RectangleEnvironmentArea
    from prism.env.nuplan_env import make_prism_env
    from prism.morl.utility_functions import UtilityFunction
    from prism.utils.zt_normaliser import ZtNormaliser

    environment_area = RectangleEnvironmentArea(front=78.0, back=50.0, left=64.0, right=64.0)
    renderer = DefaultRenderer(
        environment_area=environment_area, pixel_per_meter=2.0,
        max_vehicle_speed=30.0, max_pedestrian_speed=4.0,
        vehicle_scaling=1.0, pedestrian_scaling=1.0, static_scaling=1.0,
        include_speed_line=False, lane_connector_route=False,
    )
    observation_builder = DefaultObservationBuilder(
        environment_area=environment_area, renderer=renderer,
        obs_num_measurements=10, num_value_measurements=4,
        action_space_dim=2, inference=False,
    )
    simulation_builder = DefaultSimulationBuilder(agent_type=agent_type)
    trajectory_builder = ActionTrajectoryBuilder()

    utility_fn = UtilityFunction(reward_dim=4)
    utility_fn.eval()
    zt_normaliser = ZtNormaliser.from_hyperparams(hp)

    return make_prism_env(
        hp=hp,
        scenario_sampler=scenario_sampler,
        simulation_builder=simulation_builder,
        trajectory_builder=trajectory_builder,
        observation_builder=observation_builder,
        environment_area=environment_area,
        utility_fn=utility_fn.as_callable(),
        lambda_k=0.0, gamma=gamma,
        zt_normaliser=zt_normaliser,
        terminate_on_failure=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Explore the PRISM nuPlan simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["db", "cache"], default="db",
                        help="'db' = load directly from .db files (no cache needed, default). "
                             "'cache' = load from pre-built .gz cache.")

    # db mode args
    parser.add_argument("--data_root", type=str, default=None,
                        help="[db mode] path to nuplan-v1.1/splits/mini (or trainval)")
    parser.add_argument("--map_root", type=str, default=None,
                        help="[db mode] path to nuPlan maps root (contains nuplan-maps-v1.0.json)")
    parser.add_argument("--map_version", type=str, default="nuplan-maps-v1.0")
    parser.add_argument("--n_load", type=int, default=20,
                        help="[db mode] number of scenarios to load from the database (default: 20)")

    # cache mode args
    parser.add_argument("--cache_path", type=str, default=None,
                        help="[cache mode] path to pre-built CaRL .gz cache directory")
    parser.add_argument("--log_names", nargs="*", default=[],
                        help="[cache mode] log names to sample (empty = whole cache)")

    # shared args
    parser.add_argument("--hyperparams", type=str, default="hyperparams.json")
    parser.add_argument("--config", type=str, default="configs/prism_default.yaml")
    parser.add_argument("--policy", choices=list(POLICIES), default="random")
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--agent_type", choices=["tracks", "idm_agents", "no_tracks"],
                        default="tracks")
    parser.add_argument("--output_dir", type=str, default="explore_output")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--no_video", action="store_true")
    args = parser.parse_args()

    # Env var fallbacks
    if args.data_root is None:
        args.data_root = os.path.join(
            os.environ.get("NUPLAN_DATA_ROOT", ""),
            "nuplan-v1.1/splits/mini",
        )
    if args.map_root is None:
        args.map_root = os.environ.get("NUPLAN_MAPS_ROOT", "")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gamma = 0.99
    try:
        with open(args.config) as f:
            gamma = yaml.safe_load(f).get("gamma", 0.99)
    except Exception:
        pass

    hp_path = Path(args.hyperparams)
    if hp_path.exists():
        from prism.utils.hyperparams import load_hyperparams
        hp = load_hyperparams(str(hp_path))
        logger.info(f"Loaded hyperparams from {hp_path}")
    else:
        hp = _PLACEHOLDER_HP
        logger.warning("hyperparams.json not found — using placeholder values.")

    # ── Build scenario sampler ────────────────────────────────────────────────
    logger.info(f"Mode: {args.mode}")
    try:
        if args.mode == "db":
            if not args.data_root or not Path(args.data_root).exists():
                logger.error(f"--data_root not found: {args.data_root}")
                sys.exit(1)
            sampler = _build_db_sampler(
                data_root=args.data_root,
                map_root=args.map_root,
                map_version=args.map_version,
                n_scenarios=args.n_load,
            )
        else:
            if not args.cache_path or not Path(args.cache_path).exists():
                logger.error(f"--cache_path not found: {args.cache_path}")
                sys.exit(1)
            sampler = _build_cache_sampler(args.cache_path, args.log_names)

        env = _build_env(sampler, hp, gamma=gamma, agent_type=args.agent_type)
    except Exception as e:
        logger.error(f"Failed to build environment: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    logger.info("Environment ready.")
    policy_fn = POLICIES[args.policy]
    action_space = env.action_space

    # ── Episode loop ──────────────────────────────────────────────────────────
    for ep in range(args.n_episodes):
        logger.info(f"\n{'─'*55}")
        logger.info(f"Episode {ep}  [policy={args.policy}  agent={args.agent_type}]")
        logger.info(f"{'─'*55}")

        obs, _ = env.reset()
        done = False
        step = 0
        frames = []
        zt_history = [np.zeros(4)]
        rvec_history = []
        cost_history = []

        while not done and step < args.max_steps:
            action = policy_fn(obs, action_space)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            zt = env._zt.copy() if hasattr(env, "_zt") else np.zeros(4)
            zt_history.append(zt)

            r_vec = np.array([
                float(info.get("comfort_score", 0.5)),
                0.5, 0.5,
                float(info.get("ttc_score", 0.5)),
            ], dtype=np.float32)
            c_t = float(info.get("safety_cost", 0.0))
            regime = info.get("regime", "")

            rvec_history.append(r_vec)
            cost_history.append(c_t)

            logger.info(
                f"  step={step:3d}  R={reward:+.4f}  "
                f"comfort={r_vec[0]:.3f}  spacing={r_vec[3]:.3f}  "
                f"cost={c_t:.3f}  regime={regime}"
            )

            if not args.no_video:
                bev = obs.get("bev_semantics", np.zeros((15, 192, 192), dtype=np.uint8))
                frame = _bev_to_rgb(bev)
                frame = _overlay_text(frame, [
                    (f"Step {step}", (255, 255, 255)),
                    (f"R = {reward:+.3f}", (200, 255, 200)),
                    (f"comfort = {r_vec[0]:.3f}", (180, 255, 180)),
                    (f"spacing = {r_vec[3]:.3f}", (180, 200, 255)),
                    (f"cost    = {c_t:.3f}", (100, 100, 255) if c_t > 0 else (180, 255, 180)),
                    (f"regime  = {regime}", (255, 255, 180)),
                ])
                frames.append(frame)

            if args.interactive:
                input("    [Enter to step]")

            obs = next_obs
            step += 1

        end_reason = "terminated" if term else ("truncated" if trunc else "done")
        logger.info(f"Episode {ep}: {step} steps — {end_reason}")

        if not args.no_video and frames:
            _save_video(frames, output_dir / f"episode_{ep:03d}.avi")

        if rvec_history:
            T = len(rvec_history)
            _plot_summary(
                np.array(zt_history[:T]), np.array(rvec_history), np.array(cost_history),
                output_dir / f"episode_{ep:03d}_summary.png", ep=ep,
            )

    env.close()
    logger.info(f"\nDone.  Outputs → {output_dir}/")
    logger.info("  episode_NNN.avi         — BEV semantic video")
    logger.info("  episode_NNN_summary.png — reward/cost breakdown")


if __name__ == "__main__":
    main()
