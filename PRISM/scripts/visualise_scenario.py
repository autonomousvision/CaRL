"""
Rich nuPlan scenario visualizer for PRISM.

Shows you what a real nuPlan scenario looks like:
  - Top-down map (lanes, crosswalks, intersections)
  - Ego vehicle trajectory (ground-truth human expert)
  - Other agents (vehicles, pedestrians, cyclists) animated over time
  - Traffic light states (red / green / yellow)
  - Speed, heading, and scenario type labels

Two sub-commands
----------------
  browse   Show N random scenarios (one at a time, press Enter for next).
           Great for getting familiar with the dataset.

  replay   Replay one scenario as an animated matplotlib figure or video.
           Shows map + agents moving in real time.

Usage
-----
    # Browse 10 random scenarios from the mini split:
    python scripts/visualise_scenario.py browse \\
        --data_root $NUPLAN_DATA_ROOT/nuplan-v1.1/splits/mini \\
        --map_root  $NUPLAN_MAPS_ROOT \\
        --n 10

    # Replay a scenario as an animated video:
    python scripts/visualise_scenario.py replay \\
        --data_root $NUPLAN_DATA_ROOT/nuplan-v1.1/splits/mini \\
        --map_root  $NUPLAN_MAPS_ROOT \\
        --output    explore_output/replay_000.mp4

    # Interactive (pauses between frames):
    python scripts/visualise_scenario.py replay ... --interactive
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional

_PRISM_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PRISM_ROOT))
sys.path.insert(0, str(_PRISM_ROOT.parent / "nuPlan"))

import matplotlib
matplotlib.use("Agg")   # works headless; switch to "TkAgg" for interactive popup
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("prism.visualise")


# ─────────────────────────────────────────────────────────────────────────────
# nuPlan imports
# ─────────────────────────────────────────────────────────────────────────────

def _load_scenarios(data_root: str, map_root: str, map_version: str,
                    n: int, seed: int = 0) -> list:
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    builder = NuPlanScenarioBuilder(
        data_root=data_root, map_root=map_root, sensor_root=data_root,
        db_files=None, map_version=map_version,
        include_cameras=False, max_workers=1, verbose=False,
    )
    f = ScenarioFilter(
        scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
        num_scenarios_per_type=None, limit_total_scenarios=n,
        timestamp_threshold_s=None, ego_displacement_minimum_m=2.0,
        ego_start_speed_threshold=None, ego_stop_speed_threshold=None,
        speed_noise_tolerance=None, expand_scenarios=False,
        remove_invalid_goals=True, shuffle=True,
    )
    scenarios = builder.get_scenarios(f, Sequential())
    logger.info(f"Loaded {len(scenarios)} scenarios.")
    return scenarios


# ─────────────────────────────────────────────────────────────────────────────
# Map rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _map_query(map_api, cx: float, cy: float, radius: float, layers: list) -> dict:
    """nuplan-devkit 1.1.1 compatible map query: (Point2D, radius, [layers])."""
    from nuplan.common.actor_state.state_representation import Point2D
    return map_api.get_proximal_map_objects(Point2D(cx, cy), radius, layers)


def _render_map(ax, map_api, center_x: float, center_y: float, radius: float = 80.0):
    """Draw map layers in a radius around the ego position."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    LAYER_STYLE = {
        SemanticMapLayer.LANE:            dict(color="#C8C8C8", alpha=0.7, zorder=1),
        SemanticMapLayer.LANE_CONNECTOR:  dict(color="#D0D0D0", alpha=0.5, zorder=1),
        SemanticMapLayer.INTERSECTION:    dict(color="#DCDCDC", alpha=0.5, zorder=1),
        SemanticMapLayer.WALKWAYS:        dict(color="#A8D8A8", alpha=0.6, zorder=2),
        SemanticMapLayer.CROSSWALK:       dict(color="#FFE0A0", alpha=0.7, zorder=2),
        SemanticMapLayer.STOP_LINE:       dict(color="#FF6060", alpha=0.8, zorder=3),
    }

    for layer, style in LAYER_STYLE.items():
        try:
            objs = _map_query(map_api, center_x, center_y, radius, [layer])
            for obj in objs.get(layer, []):
                try:
                    poly = obj.polygon
                    xs, ys = poly.exterior.xy
                    ax.fill(xs, ys, **style)
                except Exception:
                    pass
        except Exception:
            pass

    # Lane centre-lines
    try:
        lanes = _map_query(
            map_api, center_x, center_y, radius, [SemanticMapLayer.LANE]
        ).get(SemanticMapLayer.LANE, [])
        for lane in lanes:
            try:
                path = lane.baseline_path.linestring
                xs, ys = path.xy
                ax.plot(xs, ys, color="#FFFFFF", linewidth=0.6, alpha=0.5, zorder=3)
            except Exception:
                pass
    except Exception:
        pass


def _draw_box(ax, x: float, y: float, heading: float,
              length: float, width: float, color: str, alpha: float = 0.9,
              label: Optional[str] = None, zorder: int = 5):
    """Draw a rotated rectangle representing a vehicle or pedestrian."""
    import matplotlib.transforms as mtransforms

    rect = mpatches.FancyBboxPatch(
        (-length / 2, -width / 2), length, width,
        boxstyle="round,pad=0.05",
        linewidth=0.8, edgecolor="white", facecolor=color, alpha=alpha,
        zorder=zorder,
    )
    t = (matplotlib.transforms.Affine2D()
         .rotate(heading)
         .translate(x, y)
         + ax.transData)
    rect.set_transform(t)
    ax.add_patch(rect)

    # Heading arrow
    dx = np.cos(heading) * length * 0.4
    dy = np.sin(heading) * length * 0.4
    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
                zorder=zorder + 1)

    if label:
        ax.text(x, y + width * 0.7, label, fontsize=5, color="white",
                ha="center", va="bottom", zorder=zorder + 2)


# ─────────────────────────────────────────────────────────────────────────────
# Single scenario snapshot
# ─────────────────────────────────────────────────────────────────────────────

def render_scenario_snapshot(scenario, iteration: int = 0,
                              radius: float = 60.0) -> plt.Figure:
    """
    Render a single timestep of a scenario: map + ego + agents.
    """
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

    ego_state = scenario.get_ego_state_at_iteration(iteration)
    cx, cy = ego_state.center.x, ego_state.center.y
    heading = ego_state.center.heading
    speed = ego_state.dynamic_car_state.speed

    n_iters = scenario.get_number_of_iterations()
    ego_traj = [scenario.get_ego_state_at_iteration(i) for i in range(n_iters)]
    ego_xs = [s.center.x for s in ego_traj]
    ego_ys = [s.center.y for s in ego_traj]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("#1A1A2E")
    fig.patch.set_facecolor("#1A1A2E")

    # Map
    try:
        _render_map(ax, scenario.map_api, cx, cy, radius)
    except Exception as e:
        ax.text(cx, cy, f"Map N/A: {e}", color="white", fontsize=8, ha="center")

    # Expert ego trajectory (full path, faded)
    ax.plot(ego_xs, ego_ys, color="#44BBFF", linewidth=1.2, alpha=0.4,
            linestyle="--", zorder=4, label="Expert trajectory")
    ax.plot(ego_xs[:iteration+1], ego_ys[:iteration+1],
            color="#44BBFF", linewidth=1.8, alpha=0.85, zorder=4)

    # Other agents at this iteration
    try:
        detections = scenario.get_tracked_objects_at_iteration(iteration)
        AGENT_COLORS = {
            TrackedObjectType.VEHICLE:   "#FF6B6B",
            TrackedObjectType.PEDESTRIAN: "#FFD93D",
            TrackedObjectType.BICYCLE:   "#6BCB77",
            TrackedObjectType.GENERIC_OBJECT: "#C77DFF",
        }
        for obj in detections.tracked_objects:
            otype = obj.tracked_object_type
            color = AGENT_COLORS.get(otype, "#AAAAAA")
            ox, oy = obj.center.x, obj.center.y
            oh = obj.center.heading
            try:
                l = obj.box.length
                w = obj.box.width
            except Exception:
                l, w = 4.0, 2.0
            if abs(ox - cx) > radius or abs(oy - cy) > radius:
                continue
            _draw_box(ax, ox, oy, oh, l, w, color, alpha=0.85, zorder=5)
    except Exception:
        pass

    # Ego vehicle
    _draw_box(ax, cx, cy, heading, 4.8, 2.1, "#00D4FF", alpha=1.0,
              label=f"{speed*3.6:.0f}km/h", zorder=7)

    # Formatting
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    # Legend
    legend_elements = [
        mpatches.Patch(color="#00D4FF", label="Ego vehicle"),
        mpatches.Patch(color="#44BBFF", label="Expert trajectory"),
        mpatches.Patch(color="#FF6B6B", label="Vehicles"),
        mpatches.Patch(color="#FFD93D", label="Pedestrians"),
        mpatches.Patch(color="#6BCB77", label="Cyclists"),
        mpatches.Patch(color="#C8C8C8", label="Road / lanes"),
        mpatches.Patch(color="#FFE0A0", label="Crosswalk"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7,
              facecolor="#2A2A3E", labelcolor="white", edgecolor="gray",
              framealpha=0.85)

    title = (f"Scenario: {scenario.scenario_type}  |  "
             f"Map: {scenario.map_api.map_name if hasattr(scenario.map_api, 'map_name') else '?'}  |  "
             f"t={iteration * 0.1:.1f}s / {n_iters * 0.1:.1f}s")
    ax.set_title(title, color="white", fontsize=10, pad=8)
    fig.tight_layout(pad=0.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Browse mode
# ─────────────────────────────────────────────────────────────────────────────

def cmd_browse(args):
    scenarios = _load_scenarios(args.data_root, args.map_root,
                                args.map_version, n=args.n)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  nuPlan Mini Dataset — Scenario Browser")
    print(f"  {len(scenarios)} scenarios loaded")
    print(f"  Snapshots saved to: {output_dir}/")
    print(f"{'='*60}\n")

    for i, scenario in enumerate(scenarios):
        n_iters = scenario.get_number_of_iterations()
        mid = n_iters // 2   # snapshot at middle of scenario

        print(f"[{i+1}/{len(scenarios)}]  type={scenario.scenario_type}  "
              f"log={scenario.log_name}  duration={n_iters*0.1:.1f}s  "
              f"token={scenario.token[:12]}…")

        fig = render_scenario_snapshot(scenario, iteration=mid)
        save_path = output_dir / f"scenario_{i:03d}_{scenario.scenario_type}.png"
        fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="#1A1A2E")
        plt.close(fig)
        print(f"   → {save_path}")

        if args.interactive:
            try:
                input("   Press Enter for next scenario (Ctrl+C to quit) … ")
            except KeyboardInterrupt:
                print("\nBrowse stopped.")
                break

    print(f"\nAll snapshots saved to {output_dir}/")
    print("Open them with:  open " + str(output_dir / "scenario_000_*.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Replay mode
# ─────────────────────────────────────────────────────────────────────────────

def cmd_replay(args):
    scenarios = _load_scenarios(args.data_root, args.map_root,
                                args.map_version, n=5)
    if not scenarios:
        logger.error("No scenarios loaded.")
        sys.exit(1)

    scenario = scenarios[0]
    n_iters = scenario.get_number_of_iterations()
    logger.info(f"Replaying: {scenario.scenario_type}  ({n_iters} frames, "
                f"{n_iters*0.1:.1f}s)  token={scenario.token[:16]}…")

    output_dir = Path(args.output_dir).parent if args.output else Path("explore_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_dir = output_dir / "replay_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []

    step = max(1, n_iters // 60)   # cap at ~60 rendered frames for speed
    iters = list(range(0, n_iters, step))

    for fi, it in enumerate(iters):
        fig = render_scenario_snapshot(scenario, iteration=it, radius=60.0)
        fpath = frame_dir / f"frame_{fi:05d}.png"
        fig.savefig(fpath, dpi=100, bbox_inches="tight", facecolor="#1A1A2E")
        plt.close(fig)
        frame_paths.append(fpath)
        if (fi + 1) % 10 == 0 or fi == len(iters) - 1:
            logger.info(f"  Rendered {fi+1}/{len(iters)} frames …")

        if args.interactive:
            try:
                input(f"  [t={it*0.1:.1f}s]  Enter for next frame … ")
            except KeyboardInterrupt:
                break

    # Assemble into video
    out_path = Path(args.output) if args.output else output_dir / "replay_000.mp4"
    try:
        import cv2
        import re
        frames = [cv2.imread(str(p)) for p in frame_paths]
        H, W = frames[0].shape[:2]
        if str(out_path).endswith(".mp4"):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        else:
            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        out = cv2.VideoWriter(str(out_path), fourcc, 5, (W, H))  # 5 fps
        for f in frames:
            out.write(f)
        out.release()
        logger.info(f"Video saved → {out_path}")
        logger.info(f"Open with:  open {out_path}")
    except Exception as e:
        logger.warning(f"Could not assemble video ({e}). Frames are in {frame_dir}/")
        logger.info(f"Open frames:  open {frame_paths[0]}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rich nuPlan scenario visualizer for PRISM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_root", type=str,
                        default=os.path.join(
                            os.environ.get("NUPLAN_DATA_ROOT", ""),
                            "nuplan-v1.1/splits/mini"))
    parser.add_argument("--map_root", type=str,
                        default=os.environ.get("NUPLAN_MAPS_ROOT", ""))
    parser.add_argument("--map_version", type=str, default="nuplan-maps-v1.0")
    parser.add_argument("--output_dir", type=str, default="explore_output/scenarios")
    parser.add_argument("--output", type=str, default=None,
                        help="[replay] output video path")

    sub = parser.add_subparsers(dest="cmd")

    p_browse = sub.add_parser("browse", help="Browse N random scenarios")
    p_browse.add_argument("--n", type=int, default=10,
                          help="Number of scenarios to browse (default: 10)")
    p_browse.add_argument("--interactive", action="store_true",
                          help="Pause between scenarios (press Enter to continue)")

    p_replay = sub.add_parser("replay", help="Animate one scenario as a video")
    p_replay.add_argument("--interactive", action="store_true",
                          help="Pause between frames")

    args = parser.parse_args()

    if args.cmd == "browse" or args.cmd is None:
        if not hasattr(args, 'n'):
            args.n = 10
        if not hasattr(args, 'interactive'):
            args.interactive = False
        cmd_browse(args)
    elif args.cmd == "replay":
        cmd_replay(args)


if __name__ == "__main__":
    main()
