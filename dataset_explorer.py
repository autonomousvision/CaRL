"""
nuPlan Dataset Explorer
=======================
Standalone tool to understand the nuPlan mini dataset.
No PRISM training code is used. Only nuplan-devkit + standard libraries.

What this does
--------------
1.  DATA SCHEMA   — prints every field available in a nuPlan scenario with
                    units, types, and plain-English explanations.
2.  SCENARIO INFO — shows metadata for N random scenarios.
3.  EGO STATES    — position / speed / acceleration / heading at every timestep.
4.  TRACKED OBJECTS — other vehicles, pedestrians, cyclists at a given timestep.
5.  MAP FEATURES  — lanes, crosswalks, speed limits, traffic lights.
6.  STATISTICS    — speed profile, agent count, scenario-type distribution.
7.  VISUALIZATIONS — saved PNG files:
      overview.png              one snapshot per scenario (map + agents + path)
      ego_speed_<N>.png         ego speed over time for scenario N
      agent_count_<N>.png       how many agents are on screen per timestep
      scenario_types.png        histogram of scenario types in the mini split

Usage
-----
    # Default: 5 scenarios, output to ./nuplan_exploration/
    python dataset_explorer.py

    # Custom:
    python dataset_explorer.py \\
        --data_root  $NUPLAN_DATA_ROOT/nuplan-v1.1/splits/mini \\
        --map_root   $NUPLAN_MAPS_ROOT \\
        --n          10 \\
        --out        ./nuplan_exploration

Environment variable shortcuts (used automatically if --data_root is omitted):
    NUPLAN_DATA_ROOT   parent of nuplan-v1.1/
    NUPLAN_MAPS_ROOT   directory containing nuplan-maps-v1.0.json
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from textwrap import dedent
from typing import List

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── matplotlib setup ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
#  Section 0: DATA SCHEMA EXPLANATION
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_EXPLANATION = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    nuPlan Dataset — Data Schema                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  COORDINATE SYSTEM                                                           ║
║  ─────────────────                                                           ║
║  • Global frame: x = East, y = North  (SI units: metres)                    ║
║  • Heading: radians from East, counter-clockwise positive                   ║
║    (0 = pointing East, π/2 = pointing North, π = West, -π/2 = South)       ║
║                                                                              ║
║  TIME                                                                        ║
║  ────                                                                        ║
║  • Sampling rate: 10 Hz  →  one step = 0.1 seconds                         ║
║  • Typical scenario length: 15 s  =  150 timesteps                          ║
║                                                                              ║
║  EGO STATE  (the autonomous vehicle's own state)                             ║
║  ──────────                                                                  ║
║  center.x / center.y          Global position [m]                           ║
║  center.heading                Direction of travel [rad]                     ║
║  dynamic_car_state.speed       Scalar speed [m/s]   (always ≥ 0)           ║
║  dynamic_car_state.                                                          ║
║    rear_axle_velocity_2d.x     Longitudinal velocity at rear axle [m/s]     ║
║    rear_axle_velocity_2d.y     Lateral velocity at rear axle [m/s]          ║
║    rear_axle_acceleration_2d.x Longitudinal acceleration [m/s²]             ║
║    rear_axle_acceleration_2d.y Lateral acceleration [m/s²]                  ║
║    angular_velocity            Yaw rate [rad/s]                              ║
║    angular_acceleration        Yaw acceleration [rad/s²]                    ║
║  tire_steering_angle           Front wheel steering angle [rad]              ║
║  car_footprint.vehicle_parameters:                                           ║
║    wheel_base                  Distance between axles [m]                    ║
║    width / front_length / rear_length   Vehicle dimensions [m]               ║
║                                                                              ║
║  TRACKED OBJECTS  (other agents in the scene)                                ║
║  ───────────────                                                             ║
║  tracked_object_type     Category:                                           ║
║    VEHICLE               Cars, trucks, buses                                 ║
║    PEDESTRIAN            People on foot                                      ║
║    BICYCLE               Bicycles / cyclists                                 ║
║    TRAFFIC_CONE          Orange traffic cones (static)                       ║
║    BARRIER               Road barriers (static)                              ║
║    GENERIC_OBJECT        Everything else                                     ║
║  center.x / center.y    Global position [m]                                 ║
║  center.heading          Direction the agent faces [rad]                    ║
║  velocity.x / velocity.y Velocity components [m/s]                          ║
║  box.length / box.width  Agent bounding box [m]                              ║
║  track_token             Unique string ID (consistent across timesteps)      ║
║                                                                              ║
║  MAP LAYERS  (static world model)                                            ║
║  ──────────                                                                  ║
║  LANE              Drivable road segment with direction of travel           ║
║    baseline_path   Centreline geometry (list of SE2 poses)                  ║
║    speed_limit_mps Speed limit on this lane [m/s]                           ║
║    left_boundary / right_boundary   Lane edge geometries                    ║
║  LANE_CONNECTOR    Junction segment connecting two lanes                    ║
║  INTERSECTION      Area where lanes cross or merge                          ║
║  CROSSWALK         Pedestrian crossing zone                                  ║
║  STOP_LINE         Stop line before traffic light / sign                    ║
║  WALKWAYS          Pavement / footpaths                                      ║
║                                                                              ║
║  TRAFFIC LIGHTS                                                              ║
║  ──────────────                                                              ║
║  Indexed by lane-connector ID. Status options:                               ║
║    GREEN  — go                                                               ║
║    RED    — stop                                                             ║
║    YELLOW — prepare to stop                                                  ║
║    UNKNOWN — sensor failure or out of range                                  ║
║                                                                              ║
║  ROUTE                                                                       ║
║  ─────                                                                       ║
║  get_route_roadblock_ids()   Ordered list of roadblock IDs the ego should   ║
║                              traverse.  A roadblock is a group of parallel  ║
║                              lanes (like a stretch of a multi-lane road).   ║
║                                                                              ║
║  SCENARIO METADATA                                                           ║
║  ─────────────────                                                           ║
║  scenario_type    What event is happening (e.g. "changing_lane_to_left",    ║
║                   "on_stopline_traffic_light", "following_lane_with_lead")  ║
║  log_name         Which drive-log this scenario came from                    ║
║  token            Unique hex identifier for this 15-second clip              ║
║  map_api.map_name City: us-ma-boston / us-nv-las-vegas-strip /              ║
║                         us-pa-pittsburgh-hazelwood / sg-one-north           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_scenarios(data_root: str, map_root: str, map_version: str,
                    n: int) -> list:
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    print(f"\n[1/7] Loading {n} scenarios from {data_root} …")
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
    flt = ScenarioFilter(
        scenario_types=None, scenario_tokens=None, log_names=None,
        map_names=None, num_scenarios_per_type=None,
        limit_total_scenarios=n,
        timestamp_threshold_s=None, ego_displacement_minimum_m=2.0,
        ego_start_speed_threshold=None, ego_stop_speed_threshold=None,
        speed_noise_tolerance=None, expand_scenarios=False,
        remove_invalid_goals=True, shuffle=True,
    )
    scenarios = builder.get_scenarios(flt, Sequential())
    print(f"    ✓  {len(scenarios)} scenarios loaded.")
    return scenarios


def _map_name(scenario) -> str:
    try:
        return scenario.map_api.map_name
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
#  Section 1: Print scenario metadata
# ─────────────────────────────────────────────────────────────────────────────

def print_scenario_info(scenario, idx: int) -> None:
    n = scenario.get_number_of_iterations()
    duration_s = n * 0.1
    ego0 = scenario.get_ego_state_at_iteration(0)
    ego_end = scenario.get_ego_state_at_iteration(n - 1)

    print(f"\n  ┌── Scenario {idx} ────────────────────────────────────────────")
    print(f"  │  type        : {scenario.scenario_type}")
    print(f"  │  log         : {scenario.log_name}")
    print(f"  │  token       : {scenario.token}")
    print(f"  │  map         : {_map_name(scenario)}")
    print(f"  │  duration    : {n} steps  ({duration_s:.1f} s at 10 Hz)")
    print(f"  │  start pos   : ({ego0.center.x:.1f}, {ego0.center.y:.1f}) m")
    print(f"  │  end pos     : ({ego_end.center.x:.1f}, {ego_end.center.y:.1f}) m")
    dist = np.hypot(ego_end.center.x - ego0.center.x,
                    ego_end.center.y - ego0.center.y)
    print(f"  │  displacement: {dist:.1f} m")
    print(f"  └──────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
#  Section 2: Ego state deep-dive
# ─────────────────────────────────────────────────────────────────────────────

def print_ego_state_detail(scenario, iteration: int = 0) -> None:
    e = scenario.get_ego_state_at_iteration(iteration)
    d = e.dynamic_car_state
    v = e.car_footprint.vehicle_parameters

    print(f"\n  ┌── EGO STATE at iteration {iteration} (t = {iteration*0.1:.1f}s) ────────")
    print(f"  │  POSITION")
    print(f"  │    x (East)        : {e.center.x:+.3f} m")
    print(f"  │    y (North)       : {e.center.y:+.3f} m")
    print(f"  │    heading         : {e.center.heading:+.4f} rad  "
          f"({np.degrees(e.center.heading):+.1f}°)")
    print(f"  │  DYNAMICS")
    print(f"  │    speed           : {d.speed:.3f} m/s  "
          f"({d.speed*3.6:.1f} km/h)")
    print(f"  │    vel longitudinal: {d.rear_axle_velocity_2d.x:+.3f} m/s")
    print(f"  │    vel lateral     : {d.rear_axle_velocity_2d.y:+.3f} m/s")
    print(f"  │    accel long.     : {d.rear_axle_acceleration_2d.x:+.3f} m/s²")
    print(f"  │    accel lat.      : {d.rear_axle_acceleration_2d.y:+.3f} m/s²")
    print(f"  │    angular velocity: {d.angular_velocity:+.4f} rad/s  "
          f"({np.degrees(d.angular_velocity):+.2f}°/s)")
    print(f"  │    steering angle  : {e.tire_steering_angle:+.4f} rad  "
          f"({np.degrees(e.tire_steering_angle):+.2f}°)")
    print(f"  │  VEHICLE GEOMETRY")
    print(f"  │    wheelbase       : {v.wheel_base:.3f} m")
    print(f"  │    width           : {v.width:.3f} m")
    print(f"  │    front length    : {v.front_length:.3f} m")
    print(f"  │    rear length     : {v.rear_length:.3f} m")
    print(f"  └────────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
#  Section 3: Tracked objects
# ─────────────────────────────────────────────────────────────────────────────

def print_tracked_objects(scenario, iteration: int = 0) -> None:
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

    det = scenario.get_tracked_objects_at_iteration(iteration)
    objs = list(det.tracked_objects)

    # Count by type
    counts = {}
    for o in objs:
        t = o.tracked_object_type.name
        counts[t] = counts.get(t, 0) + 1

    ego = scenario.get_ego_state_at_iteration(iteration)

    print(f"\n  ┌── TRACKED OBJECTS at t={iteration*0.1:.1f}s "
          f"({len(objs)} total) ────────────────")
    print(f"  │  Type breakdown: " +
          "  |  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print(f"  │")

    shown = 0
    for o in objs:
        if shown >= 6:
            print(f"  │  … ({len(objs) - shown} more not shown)")
            break
        # distance from ego
        dx = o.center.x - ego.center.x
        dy = o.center.y - ego.center.y
        dist = np.hypot(dx, dy)
        speed = np.hypot(o.velocity.x, o.velocity.y)
        try:
            l, w = o.box.length, o.box.width
        except Exception:
            l, w = 0.0, 0.0

        print(f"  │  [{o.tracked_object_type.name:18s}]  "
              f"token={o.track_token[:8]}…  "
              f"dist={dist:5.1f}m  "
              f"speed={speed:.1f}m/s  "
              f"box={l:.1f}×{w:.1f}m")
        shown += 1

    print(f"  └────────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
#  Section 4: Map features near ego
# ─────────────────────────────────────────────────────────────────────────────

def print_map_features(scenario, iteration: int = 0, radius: float = 50.0) -> None:
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.common.maps.abstract_map import Lane

    ego = scenario.get_ego_state_at_iteration(iteration)
    cx, cy = ego.center.x, ego.center.y
    map_api = scenario.map_api

    print(f"\n  ┌── MAP FEATURES within {radius}m of ego ────────────────────────")

    LAYERS = [
        SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.CROSSWALK, SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.STOP_LINE,
    ]
    for layer in LAYERS:
        try:
            result = _proximal(map_api, cx, cy, radius, [layer])
            items = result.get(layer, [])
            name = layer.name
            if items and name == "LANE":
                limits = set()
                for lane in items:
                    if isinstance(lane, Lane) and lane.speed_limit_mps is not None:
                        limits.add(round(lane.speed_limit_mps * 3.6, 0))
                limit_str = (f"  speed limits: {sorted(limits)} km/h"
                             if limits else "")
                print(f"  │  {name:20s}: {len(items):3d} objects{limit_str}")
            else:
                print(f"  │  {name:20s}: {len(items):3d} objects")
        except Exception as e:
            print(f"  │  {layer.name:20s}: (query failed: {e})")

    # Traffic lights
    try:
        tl_iter = scenario.get_traffic_light_status_at_iteration(iteration)
        tl_list = list(tl_iter)
        from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
        status_count = {}
        for tl in tl_list:
            s = tl.status.name
            status_count[s] = status_count.get(s, 0) + 1
        print(f"  │  {'TRAFFIC_LIGHT':20s}: {len(tl_list):3d} signals  "
              f"{status_count}")
    except Exception:
        pass

    print(f"  │  Route roadblocks: {scenario.get_route_roadblock_ids()[:4]} …")
    print(f"  └────────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
#  Section 5: Build ego trajectory arrays
# ─────────────────────────────────────────────────────────────────────────────

def get_ego_trajectory(scenario) -> dict:
    n = scenario.get_number_of_iterations()
    xs, ys, heads, speeds, accs_lon, accs_lat = [], [], [], [], [], []
    for i in range(n):
        e = scenario.get_ego_state_at_iteration(i)
        d = e.dynamic_car_state
        xs.append(e.center.x)
        ys.append(e.center.y)
        heads.append(e.center.heading)
        speeds.append(d.speed)
        accs_lon.append(d.rear_axle_acceleration_2d.x)
        accs_lat.append(d.rear_axle_acceleration_2d.y)
    return dict(
        x=np.array(xs), y=np.array(ys), heading=np.array(heads),
        speed=np.array(speeds), acc_lon=np.array(accs_lon),
        acc_lat=np.array(accs_lat),
        t=np.arange(n) * 0.1,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_agent_box(ax, x, y, h, length, width, color, alpha=0.85, zorder=5):
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.transforms as mtransforms
    rect = FancyBboxPatch((-length/2, -width/2), length, width,
                          boxstyle="round,pad=0.05",
                          linewidth=0.6, edgecolor="white",
                          facecolor=color, alpha=alpha, zorder=zorder)
    t = (matplotlib.transforms.Affine2D().rotate(h).translate(x, y)
         + ax.transData)
    rect.set_transform(t)
    ax.add_patch(rect)
    ax.annotate("", xy=(x + np.cos(h)*length*0.45, y + np.sin(h)*length*0.45),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="white", lw=0.7),
                zorder=zorder+1)


def _proximal(map_api, cx, cy, radius, layers):
    """Wrapper compatible with nuplan-devkit 1.1.1 public API."""
    from nuplan.common.actor_state.state_representation import Point2D
    return map_api.get_proximal_map_objects(Point2D(cx, cy), radius, layers)


def _render_map_bg(ax, map_api, cx, cy, radius=70):
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    STYLES = {
        SemanticMapLayer.LANE:           ("#3A3A5A", 0.8),
        SemanticMapLayer.LANE_CONNECTOR: ("#3A3A5A", 0.5),
        SemanticMapLayer.INTERSECTION:   ("#44445A", 0.5),
        SemanticMapLayer.CROSSWALK:      ("#8A7A2A", 0.7),
        SemanticMapLayer.WALKWAYS:       ("#2A5A2A", 0.5),
    }
    for layer, (color, alpha) in STYLES.items():
        try:
            for obj in _proximal(map_api, cx, cy, radius, [layer]).get(layer, []):
                try:
                    xs, ys = obj.polygon.exterior.xy
                    ax.fill(xs, ys, color=color, alpha=alpha, zorder=1)
                except Exception:
                    pass
        except Exception:
            pass
    # Lane centrelines
    try:
        for lane in _proximal(map_api, cx, cy, radius, [SemanticMapLayer.LANE]).get(SemanticMapLayer.LANE, []):
            try:
                xs, ys = lane.baseline_path.linestring.xy
                ax.plot(xs, ys, color="#6666AA", lw=0.5, alpha=0.5, zorder=2)
            except Exception:
                pass
    except Exception:
        pass


def _scenario_overview_fig(scenario, idx: int, radius: float = 65.0) -> plt.Figure:
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

    traj = get_ego_trajectory(scenario)
    n = scenario.get_number_of_iterations()
    mid = n // 2

    # Pick 3 snapshot iterations
    snaps = [0, mid, n - 1]

    fig = plt.figure(figsize=(16, 5.5))
    fig.patch.set_facecolor("#12121E")

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.05)

    SNAP_LABELS = ["t = 0 s  (start)", f"t = {mid*0.1:.1f}s  (mid)",
                   f"t = {(n-1)*0.1:.1f}s  (end)"]

    AGENT_COLORS = {
        TrackedObjectType.VEHICLE:        "#FF6B6B",
        TrackedObjectType.PEDESTRIAN:     "#FFD93D",
        TrackedObjectType.BICYCLE:        "#6BCB77",
        TrackedObjectType.TRAFFIC_CONE:   "#FF9F43",
        TrackedObjectType.BARRIER:        "#A29BFE",
        TrackedObjectType.GENERIC_OBJECT: "#AAAAAA",
    }

    for col, (it, snap_label) in enumerate(zip(snaps, SNAP_LABELS)):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("#12121E")

        ego = scenario.get_ego_state_at_iteration(it)
        cx, cy = ego.center.x, ego.center.y

        _render_map_bg(ax, scenario.map_api, cx, cy, radius)

        # Full expert trajectory (faded)
        ax.plot(traj["x"], traj["y"], color="#4488FF", lw=1.0, alpha=0.3,
                linestyle="--", zorder=3)
        # Trajectory up to now
        ax.plot(traj["x"][:it+1], traj["y"][:it+1], color="#44BBFF",
                lw=1.8, alpha=0.85, zorder=4)

        # Agents
        try:
            det = scenario.get_tracked_objects_at_iteration(it)
            for o in det.tracked_objects:
                ox, oy = o.center.x, o.center.y
                if abs(ox - cx) > radius or abs(oy - cy) > radius:
                    continue
                oh = o.center.heading
                color = AGENT_COLORS.get(o.tracked_object_type, "#AAAAAA")
                try:
                    l, w = o.box.length, o.box.width
                except Exception:
                    l, w = 3.5, 1.8
                _draw_agent_box(ax, ox, oy, oh, l, w, color, zorder=5)
        except Exception:
            pass

        # Ego
        _draw_agent_box(ax, cx, cy, ego.center.heading, 4.8, 2.1,
                        "#00D4FF", alpha=1.0, zorder=7)

        ax.set_xlim(cx - radius, cx + radius)
        ax.set_ylim(cy - radius, cy + radius)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(snap_label, color="#AAAAFF", fontsize=9, pad=4)

        if col == 0:
            legend = [
                mpatches.Patch(color="#00D4FF", label="Ego vehicle"),
                mpatches.Patch(color="#44BBFF", label="Expert path"),
                mpatches.Patch(color="#FF6B6B", label="Vehicles"),
                mpatches.Patch(color="#FFD93D", label="Pedestrians"),
                mpatches.Patch(color="#6BCB77", label="Cyclists"),
                mpatches.Patch(color="#3A3A5A", label="Road / lanes"),
                mpatches.Patch(color="#8A7A2A", label="Crosswalk"),
            ]
            ax.legend(handles=legend, loc="upper left", fontsize=6.5,
                      facecolor="#1A1A2E", labelcolor="white",
                      edgecolor="#444466", framealpha=0.9)

    map_name = _map_name(scenario)
    fig.suptitle(
        f"Scenario {idx}:  {scenario.scenario_type}   |   "
        f"Map: {map_name}   |   "
        f"Duration: {n*0.1:.1f}s   |   token: {scenario.token[:14]}…",
        color="white", fontsize=10, y=0.98,
    )
    return fig


def _ego_timeseries_fig(scenario, idx: int) -> plt.Figure:
    traj = get_ego_trajectory(scenario)
    t = traj["t"]
    n = len(t)

    # Agent count at each step (sample every 5 steps for speed)
    agent_counts = []
    sample_steps = list(range(0, n, 5))
    for it in sample_steps:
        try:
            det = scenario.get_tracked_objects_at_iteration(it)
            agent_counts.append(len(list(det.tracked_objects)))
        except Exception:
            agent_counts.append(0)
    agent_t = t[sample_steps]

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.patch.set_facecolor("#12121E")
    for ax in axes:
        ax.set_facecolor("#1A1A2E")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_color("#444466")

    # Speed
    axes[0].plot(t, traj["speed"] * 3.6, color="#44BBFF", lw=1.5)
    axes[0].fill_between(t, 0, traj["speed"] * 3.6, alpha=0.15, color="#44BBFF")
    axes[0].set_ylabel("Speed  [km/h]", color="white")
    axes[0].set_title(f"Ego vehicle time-series  |  Scenario {idx}: "
                      f"{scenario.scenario_type}", color="white", fontsize=10)
    axes[0].axhline(0, color="#444466", lw=0.5)
    axes[0].grid(True, alpha=0.15, color="#444466")

    # Acceleration
    axes[1].plot(t, traj["acc_lon"], color="#FF6B6B", lw=1.2, label="Longitudinal")
    axes[1].plot(t, traj["acc_lat"], color="#FFD93D", lw=1.2, label="Lateral", alpha=0.8)
    axes[1].axhline(0, color="#444466", lw=0.5)
    axes[1].set_ylabel("Acceleration  [m/s²]", color="white")
    axes[1].legend(fontsize=8, facecolor="#1A1A2E", labelcolor="white",
                   edgecolor="#444466")
    axes[1].grid(True, alpha=0.15, color="#444466")

    # Agent count
    axes[2].fill_between(agent_t, agent_counts, alpha=0.4, color="#6BCB77",
                         step="mid")
    axes[2].step(agent_t, agent_counts, color="#6BCB77", lw=1.2, where="mid")
    axes[2].set_ylabel("Nearby agents  [count]", color="white")
    axes[2].set_xlabel("Time  [s]", color="white")
    axes[2].grid(True, alpha=0.15, color="#444466")

    fig.tight_layout(pad=1.2)
    return fig


def _scenario_type_histogram(scenarios: list) -> plt.Figure:
    types = {}
    maps_seen = {}
    for s in scenarios:
        t = s.scenario_type
        types[t] = types.get(t, 0) + 1
        m = _map_name(s)
        maps_seen[m] = maps_seen.get(m, 0) + 1

    # Sort by count
    sorted_types = sorted(types.items(), key=lambda x: -x[1])
    labels = [k for k, _ in sorted_types]
    counts = [v for _, v in sorted_types]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, len(labels)*0.28 + 1)))
    fig.patch.set_facecolor("#12121E")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1A1A2E")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("#444466")

    # Horizontal bar chart of scenario types
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(labels)))
    bars = ax1.barh(labels, counts, color=colors, edgecolor="#444466", height=0.7)
    ax1.set_xlabel("Count", color="white")
    ax1.set_title(f"Scenario types in loaded set  (N={len(scenarios)})",
                  color="white", fontsize=10)
    ax1.tick_params(axis="y", labelsize=7, colors="white")
    ax1.tick_params(axis="x", colors="white")
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 str(count), va="center", fontsize=7, color="white")
    ax1.grid(axis="x", alpha=0.15, color="#444466")

    # Pie chart of maps
    map_labels = list(maps_seen.keys())
    map_counts = [maps_seen[k] for k in map_labels]
    MAP_COLORS = ["#FF6B6B", "#FFD93D", "#6BCB77", "#44BBFF", "#C77DFF"]
    wedges, texts, autotexts = ax2.pie(
        map_counts,
        labels=[m.replace("us-", "").replace("sg-", "").replace("-", "\n")
                for m in map_labels],
        autopct="%1.0f%%",
        colors=MAP_COLORS[:len(map_labels)],
        textprops=dict(color="white", fontsize=9),
    )
    for at in autotexts:
        at.set_color("black")
        at.set_fontsize(8)
    ax2.set_title("Scenarios by city / map", color="white", fontsize=10)

    fig.suptitle("Dataset Distribution", color="white", fontsize=12, y=1.01)
    fig.tight_layout(pad=1.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="nuPlan Dataset Explorer — independent of PRISM training code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
        Examples:
          python dataset_explorer.py
          python dataset_explorer.py --n 10 --out ./my_exploration
          python dataset_explorer.py --data_root /data/nuplan/nuplan-v1.1/splits/mini
        """),
    )
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--map_root", type=str, default=None)
    parser.add_argument("--map_version", type=str, default="nuplan-maps-v1.0")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of scenarios to explore (default: 5)")
    parser.add_argument("--out", type=str, default="nuplan_exploration",
                        help="Output directory for all figures (default: nuplan_exploration/)")
    args = parser.parse_args()

    # Resolve paths from env vars if not provided
    if args.data_root is None:
        data_root_env = os.environ.get("NUPLAN_DATA_ROOT", "")
        args.data_root = os.path.join(data_root_env, "nuplan-v1.1/splits/mini")
    if args.map_root is None:
        args.map_root = os.environ.get("NUPLAN_MAPS_ROOT", "")

    if not Path(args.data_root).exists():
        print(f"\n[ERROR] data_root not found: {args.data_root}")
        print("  Set NUPLAN_DATA_ROOT or pass --data_root")
        sys.exit(1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Section 0: Schema ─────────────────────────────────────────────────────
    print(SCHEMA_EXPLANATION)

    # ── Load scenarios ────────────────────────────────────────────────────────
    scenarios = _load_scenarios(args.data_root, args.map_root,
                                args.map_version, n=args.n)

    # ── Section 1-4: Per-scenario text output ─────────────────────────────────
    print("\n[2/7] Per-scenario summaries\n")
    for i, s in enumerate(scenarios):
        print_scenario_info(s, i)

    print("\n[3/7] Ego state detail for scenario 0\n")
    print_ego_state_detail(scenarios[0], iteration=0)
    print_ego_state_detail(scenarios[0], iteration=scenarios[0].get_number_of_iterations()//2)

    print("\n[4/7] Tracked objects for scenario 0\n")
    print_tracked_objects(scenarios[0], iteration=0)

    print("\n[5/7] Map features for scenario 0\n")
    print_map_features(scenarios[0], iteration=0)

    # ── Section 5: Scenario type distribution ─────────────────────────────────
    print(f"\n[6/7] Generating figures → {out}/\n")

    fig = _scenario_type_histogram(scenarios)
    p = out / "scenario_types.png"
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="#12121E")
    plt.close(fig)
    print(f"  ✓  {p}")

    # ── Section 6: Per-scenario visual + time-series ──────────────────────────
    for i, s in enumerate(scenarios):
        # Overview (map + agents, 3 snapshots)
        fig = _scenario_overview_fig(s, i)
        p = out / f"overview_{i:02d}_{s.scenario_type}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight", facecolor="#12121E")
        plt.close(fig)
        print(f"  ✓  {p}")

        # Time-series (speed, acceleration, agent count)
        fig = _ego_timeseries_fig(s, i)
        p = out / f"timeseries_{i:02d}_{s.scenario_type}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight", facecolor="#12121E")
        plt.close(fig)
        print(f"  ✓  {p}")

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"""
[7/7] Done.

All outputs saved to:  {out}/

Files generated:
  scenario_types.png            — histogram of scenario types + city breakdown
  overview_NN_<type>.png        — map + agents at 3 timepoints per scenario
  timeseries_NN_<type>.png      — speed / acceleration / agent count over time

Open everything at once:
  open {out}/

Understanding what you're looking at:
  overview images   → dark background = outside road, grey = drivable lane,
                      cyan box = ego, red = vehicles, yellow = pedestrians,
                      cyan dashed line = where the human actually drove
  timeseries plots  → top: speed in km/h (should match real urban driving)
                      mid: longitudinal (fore-aft) and lateral acceleration
                      bottom: how many other agents are visible per timestep
""")


if __name__ == "__main__":
    main()
