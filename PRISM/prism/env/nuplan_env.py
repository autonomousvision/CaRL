"""
PRISM nuPlan gymnasium environment.

Architecture:
    PRISMRewardBuilder  (AbstractRewardBuilder)
        • Computes 4D style reward vector r_t  (comfort, progress, lateral, spacing)
        • Computes safety cost c_t (two-tier)
        • Determines episode termination
        • Stores results in info dict; returns scalar=0.0 (replaced by PRISMEnv)

    PRISMEnv  (subclass of EnvironmentWrapper)
        • Maintains cumulative style return z_t
        • Augments observation's value_measurements slot with z_t_normalised
        • Computes DPMORL scalar reward:
              R_t = gamma^{-t} * [f_k(z_{t+1}) - f_k(z_t)]
        • Applies Lagrangian penalty: effective_reward = R_t - lambda_k * c_t
        • Updates z_t normaliser at episode end

Observation space (unchanged from CaRL):
    bev_semantics     : (C, H, W) uint8  — BEV raster
    measurements      : (10,) float32   — ego kinematics
    value_measurements: (4,) float32    — z_t_normalised (repurposed from CaRL)

This repurposing of value_measurements is intentional: the policy sees z_t
at every step so the utility function transformation is well-conditioned.
"""

from __future__ import annotations

import math
import traceback
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

# ── CaRL imports ──────────────────────────────────────────────────────────────
from carl_nuplan.planning.gym.environment.environment_wrapper import EnvironmentWrapper
from carl_nuplan.planning.gym.environment.reward_builder.abstract_reward_builder import AbstractRewardBuilder
from carl_nuplan.planning.gym.environment.helper.environment_cache import (
    environment_cache_manager,
    MapCache,
    DetectionCache,
)
from carl_nuplan.planning.gym.environment.reward_builder.default_reward_builder import (
    _find_current_and_intersecting_lanes,
    _calculate_off_road,
    _calculate_red_light,
)
from carl_nuplan.planning.gym.environment.reward_builder.components.collision import (
    calculate_non_stationary_collisions,
)
from carl_nuplan.planning.gym.environment.simulation_wrapper import SimulationWrapper
from carl_nuplan.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    normalize_angle,
)
from carl_nuplan.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    states_se2_to_array,
)

# ── nuPlan imports ────────────────────────────────────────────────────────────
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import Lane
from shapely.geometry import Point

# ── PRISM imports ─────────────────────────────────────────────────────────────
from prism.env.regime_detector import RegimeDetector, RegimeResult
from prism.env.rewards import compute_style_rewards
from prism.env.safety_cost import SafetyCostBuilder, SafetyCostComponents
from prism.utils.zt_normaliser import ZtNormaliser


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_REWARD_DIM = 4      # (comfort, progress, lateral, spacing)
_DT = 0.1            # nuPlan simulation timestep [s]


# ─────────────────────────────────────────────────────────────────────────────
# PRISMRewardBuilder
# ─────────────────────────────────────────────────────────────────────────────

class PRISMRewardBuilder(AbstractRewardBuilder):
    """
    Computes PRISM style rewards and safety costs from the nuPlan simulation.

    Unlike CaRL's DefaultRewardBuilder, this returns scalar=0.0; the actual
    Lagrangian-penalised scalar reward is assembled by PRISMEnv after z_t is
    updated.  The style reward vector and safety cost are passed via info.
    """

    def __init__(
        self,
        environment_area,
        hp: Dict,
        regime_detector: RegimeDetector,
        terminate_on_collision: bool = True,
        terminate_on_off_road: bool = True,
    ) -> None:
        self._area = environment_area
        self._hp = hp
        self._regime = regime_detector
        self._terminate_collision = terminate_on_collision
        self._terminate_off_road = terminate_on_off_road

        self._safety = SafetyCostBuilder(hp)

        # Episode state
        self._prev_accel_lon: Optional[float] = None
        self._prev_accel_lat: Optional[float] = None
        self._prev_collided_tokens: list = []
        self._expert_rl_infractions: list = []
        self._episode_step: int = 0

    def reset(self) -> None:
        self._safety.reset()
        self._prev_accel_lon = None
        self._prev_accel_lat = None
        self._prev_collided_tokens = []
        self._expert_rl_infractions = []
        self._episode_step = 0

    def build_reward(
        self,
        simulation_wrapper: SimulationWrapper,
        info: Dict[str, Any],
    ) -> Tuple[float, bool, bool]:
        # ── Build map/detection caches ────────────────────────────────────
        map_cache, detection_cache = environment_cache_manager.build_environment_caches(
            planner_input=simulation_wrapper.current_planner_input,
            planner_initialization=simulation_wrapper.planner_initialization,
            environment_area=self._area,
        )
        info["map_cache"] = map_cache
        info["detection_cache"] = detection_cache

        ego_state = simulation_wrapper.current_ego_state
        ego_states = simulation_wrapper.current_planner_input.history.ego_states

        # ── Current lane ─────────────────────────────────────────────────
        current_lane, intersecting_lanes = _find_current_and_intersecting_lanes(
            ego_state, map_cache
        )

        # ── Regime detection ─────────────────────────────────────────────
        regime_result: RegimeResult = self._regime.detect(
            ego_state, map_cache, detection_cache
        )

        # ── Kinematic quantities ─────────────────────────────────────────
        state = ego_state.dynamic_car_state
        v_ego = state.speed
        try:
            a_lon = float(state.rear_axle_acceleration_2d.x)
            a_lat = float(state.rear_axle_acceleration_2d.y)
        except Exception:
            a_lon, a_lat = 0.0, 0.0

        # Jerk = Δa / Δt
        if self._prev_accel_lon is not None:
            j_lon = (a_lon - self._prev_accel_lon) / _DT
            j_lat = (a_lat - self._prev_accel_lat) / _DT
        else:
            j_lon, j_lat = 0.0, 0.0
        self._prev_accel_lon = a_lon
        self._prev_accel_lat = a_lat

        # ── Lateral deviation and heading error ──────────────────────────
        d_lat, delta_psi = self._lateral_quantities(ego_state, current_lane)

        # ── Lane position ────────────────────────────────────────────────
        lane_index, n_lanes = self._lane_position(ego_state, map_cache, current_lane)

        # ── Collision detection ──────────────────────────────────────────
        had_collision, collided_tokens = calculate_non_stationary_collisions(
            ego_state,
            detection_cache.tracked_objects,
            self._prev_collided_tokens,
        )
        self._prev_collided_tokens.extend(collided_tokens)

        had_vru_collision = False
        if had_collision:
            had_vru_collision = self._check_vru_collision(
                ego_state, detection_cache, collided_tokens
            )

        # ── Off-road ─────────────────────────────────────────────────────
        is_off_road, _ = _calculate_off_road(
            ego_state, map_cache, intersecting_lanes, violation_threshold=0.0
        )

        # ── Red light ────────────────────────────────────────────────────
        ran_red_light, self._expert_rl_infractions = _calculate_red_light(
            simulation_wrapper, map_cache, current_lane, self._expert_rl_infractions
        )

        # ── Wrong direction ──────────────────────────────────────────────
        had_wrong_direction = self._check_wrong_direction(ego_state, current_lane)

        # ── Style rewards ─────────────────────────────────────────────────
        r_vec = compute_style_rewards(
            j_lon=j_lon,
            j_lat=j_lat,
            v_ego=v_ego,
            v_des=regime_result.v_des,
            a_ego=a_lon,
            lane_index=lane_index,
            n_lanes=n_lanes,
            d_lat=d_lat,
            delta_psi=delta_psi,
            d_lead=regime_result.d_lead,
            v_lead=regime_result.v_lead,
            has_lead=regime_result.has_lead,
            hp=self._hp,
        )

        # ── Safety cost ───────────────────────────────────────────────────
        cost_comp: SafetyCostComponents = self._safety.compute(
            ego_state=ego_state,
            detection_cache=detection_cache,
            map_cache=map_cache,
            had_collision=had_collision,
            is_off_road=is_off_road,
            ran_red_light=ran_red_light,
            had_vru_collision=had_vru_collision,
            had_wrong_direction=had_wrong_direction,
            v_lead=regime_result.v_lead,
            d_lead=regime_result.d_lead,
            has_lead=regime_result.has_lead,
            v_limit=regime_result.v_limit,
        )

        # ── Termination ───────────────────────────────────────────────────
        termination = (
            (self._terminate_collision and (had_collision or had_vru_collision))
            or (self._terminate_off_road and is_off_road)
        )

        # ── Pack into info for PRISMEnv to consume ────────────────────────
        info["style_reward"] = r_vec                    # np.ndarray (4,)
        info["safety_cost"] = float(cost_comp.total)   # scalar
        info["regime"] = regime_result.regime.name
        info["safety_components"] = cost_comp          # for logging

        # Dummy value_measurements so DefaultObservationBuilder doesn't crash
        # (PRISMEnv will overwrite these with z_t_normalised)
        info["remaining_time"] = 1.0
        info["remaining_progress"] = 1.0
        info["comfort_score"] = float(r_vec[0])
        info["ttc_score"] = float(r_vec[3])

        self._episode_step += 1
        return 0.0, termination, termination  # scalar overwritten by PRISMEnv


    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lateral_quantities(
        ego_state, current_lane
    ) -> Tuple[float, float]:
        """Return (d_lat [m], delta_psi [rad])."""
        if current_lane is None:
            return 0.0, 0.0

        ego_x, ego_y = ego_state.center.x, ego_state.center.y
        ego_heading = ego_state.center.heading

        try:
            ego_pt = Point(ego_x, ego_y)
            d_lat = float(current_lane.baseline_path.linestring.distance(ego_pt))

            lane_path = current_lane.baseline_path.discrete_path
            if len(lane_path) == 0:
                return d_lat, 0.0

            lane_se2 = states_se2_to_array(lane_path)
            dists = np.linalg.norm(
                np.array([[ego_x, ego_y]]) - lane_se2[..., :2], axis=-1
            )
            nearest_heading = lane_path[int(np.argmin(dists))].heading
            delta_psi = float(normalize_angle(ego_heading - nearest_heading))

            return d_lat, delta_psi
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def _lane_position(
        ego_state, map_cache, current_lane
    ) -> Tuple[int, int]:
        """
        Estimate (lane_index, n_lanes) by counting adjacent lanes.
        Returns (0, 1) as a safe fallback (r_lane = 0.5).
        """
        if current_lane is None or not isinstance(current_lane, Lane):
            return 0, 1
        try:
            # Count adjacent lanes in the same roadblock
            rb_id = current_lane.get_roadblock_id()
            all_lanes = [
                l for l in map_cache.lanes.values()
                if l.get_roadblock_id() == rb_id
            ]
            n_lanes = max(len(all_lanes), 1)

            # Sort by lateral position relative to ego
            ego_x, ego_y = ego_state.center.x, ego_state.center.y
            ego_heading = ego_state.center.heading
            sin_h = math.sin(ego_heading)
            cos_h = math.cos(ego_heading)

            def lane_lateral_offset(lane):
                try:
                    cx, cy = lane.baseline_path.linestring.centroid.x, lane.baseline_path.linestring.centroid.y
                    dx, dy = cx - ego_x, cy - ego_y
                    return float(-sin_h * dx + cos_h * dy)
                except Exception:
                    return 0.0

            sorted_lanes = sorted(all_lanes, key=lane_lateral_offset)
            try:
                lane_index = sorted_lanes.index(current_lane)
            except ValueError:
                lane_index = 0

            return lane_index, n_lanes
        except Exception:
            return 0, 1

    @staticmethod
    def _check_vru_collision(ego_state, detection_cache, collided_tokens: list) -> bool:
        """Return True if any collided object is a VRU (pedestrian or cyclist)."""
        VRU_TYPES = {TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE}
        try:
            for obj in detection_cache.tracked_objects:
                if obj.track_token in collided_tokens:
                    if obj.tracked_object_type in VRU_TYPES:
                        return True
        except Exception:
            pass
        return False

    @staticmethod
    def _check_wrong_direction(ego_state, current_lane) -> bool:
        """Return True if ego is driving against the lane direction (heading error > 90°)."""
        if current_lane is None:
            return False
        try:
            lane_path = current_lane.baseline_path.discrete_path
            if len(lane_path) == 0:
                return False
            lane_se2 = states_se2_to_array(lane_path)
            ego_x, ego_y = ego_state.center.x, ego_state.center.y
            dists = np.linalg.norm(
                np.array([[ego_x, ego_y]]) - lane_se2[..., :2], axis=-1
            )
            nearest_heading = lane_path[int(np.argmin(dists))].heading
            delta_psi = abs(normalize_angle(ego_state.center.heading - nearest_heading))
            return delta_psi > math.pi / 2
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# PRISMEnv
# ─────────────────────────────────────────────────────────────────────────────

class PRISMEnv(EnvironmentWrapper):
    """
    PRISM gymnasium environment.

    Wraps CaRL's EnvironmentWrapper, replacing the scalar reward with the
    DPMORL-transformed Lagrangian reward:
        R_t = gamma^{-t} * [f_k(z_{t+1}) - f_k(z_t)] - lambda_k * c_t

    The observation's value_measurements slot (shape 4) is repurposed to
    carry z_t_normalised so the policy is aware of cumulative style returns.

    Parameters
    ----------
    utility_fn : callable
        f_k : np.ndarray (4,) -> float.  The current policy's utility function.
        Pass a lambda or UtilityFunction instance.  Can be updated between
        policies via set_utility_fn().
    lambda_k : float
        Current Lagrange multiplier for the safety constraint.
    gamma : float
        Discount factor (must match training — default 0.99).
    zt_normaliser : ZtNormaliser
        Running EMA normaliser for z_t.
    """

    def __init__(
        self,
        *args,
        utility_fn: Callable[[np.ndarray], float],
        lambda_k: float = 0.0,
        gamma: float = 0.99,
        zt_normaliser: Optional[ZtNormaliser] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._utility_fn = utility_fn
        self._lambda_k = lambda_k
        self._gamma = gamma
        self._zt_normaliser = zt_normaliser or ZtNormaliser(reward_dim=_REWARD_DIM)

        self._zt = np.zeros(_REWARD_DIM, dtype=np.float64)
        self._t = 0

    # ------------------------------------------------------------------
    # Utility function and Lagrange multiplier updates
    # ------------------------------------------------------------------

    def set_utility_fn(self, fn: Callable[[np.ndarray], float]) -> None:
        self._utility_fn = fn

    def set_lambda(self, lam: float) -> None:
        self._lambda_k = max(0.0, float(lam))

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._zt = np.zeros(_REWARD_DIM, dtype=np.float64)
        self._t = 0
        obs["value_measurements"] = self._zt_normaliser.normalise(self._zt)
        return obs, info

    def step(self, action):
        obs, _dummy_reward, termination, truncation, info = super().step(action)

        # Extract style reward and safety cost from info (set by PRISMRewardBuilder)
        r_vec = info.pop("style_reward", np.zeros(_REWARD_DIM, dtype=np.float64))
        c_t = float(info.pop("safety_cost", 0.0))

        # z_{t+1} = z_t + gamma^t * r_t
        zt_next = self._zt + (self._gamma ** self._t) * np.asarray(r_vec, dtype=np.float64)

        # DPMORL scalar reward: R_t = gamma^{-t} * [f(z_{t+1}) - f(z_t)]
        f_next = float(self._utility_fn(zt_next))
        f_curr = float(self._utility_fn(self._zt))
        R_t = (self._gamma ** (-self._t)) * (f_next - f_curr)

        # Lagrangian-penalised effective reward
        effective_reward = float(R_t - self._lambda_k * c_t)

        # Update cumulative state
        self._zt = zt_next
        self._t += 1

        # Update normaliser at episode end
        if termination or truncation:
            self._zt_normaliser.update(self._zt)

        # Augment observation with normalised z_t (4 values)
        obs["value_measurements"] = self._zt_normaliser.normalise(self._zt)

        # Expose episode z_T for CVaR computation
        if termination or truncation:
            info["episode_zt"] = self._zt.copy()
            info["episode_cost"] = float(c_t)

        return obs, effective_reward, termination, truncation, info


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper
# ─────────────────────────────────────────────────────────────────────────────

def make_prism_env(
    hp: Dict,
    scenario_sampler,
    simulation_builder,
    trajectory_builder,
    observation_builder,
    environment_area,
    utility_fn: Callable[[np.ndarray], float],
    lambda_k: float = 0.0,
    gamma: float = 0.99,
    zt_normaliser: Optional[ZtNormaliser] = None,
    terminate_on_failure: bool = False,
) -> PRISMEnv:
    """
    Build a ready-to-use PRISMEnv from the standard CaRL builder objects.
    """
    regime_detector = RegimeDetector(
        congestion_speed_fraction=hp.get("regime_detection", {}).get(
            "congestion_speed_fraction", 0.5
        ),
        observation_horizon_m=hp.get("regime_detection", {}).get(
            "observation_horizon_m", 50.0
        ),
    )
    reward_builder = PRISMRewardBuilder(
        environment_area=environment_area,
        hp=hp,
        regime_detector=regime_detector,
    )
    return PRISMEnv(
        scenario_sampler=scenario_sampler,
        simulation_builder=simulation_builder,
        trajectory_builder=trajectory_builder,
        observation_builder=observation_builder,
        reward_builder=reward_builder,
        terminate_on_failure=terminate_on_failure,
        utility_fn=utility_fn,
        lambda_k=lambda_k,
        gamma=gamma,
        zt_normaliser=zt_normaliser,
    )
