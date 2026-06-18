"""
PRISM two-tier safety cost c_t.

    c_t = c_outcome + sum_j(c_lead_j)

Tier 1 — Outcome events (large, infrequent):
    Fired when an outcome actually occurs (collision, red light, off-road, …).
    Weight W_i comes from OUTCOME_WEIGHTS in compute_hyperparams.py.

Tier 2 — Indicator signals (smaller, continuous warning):
    Fired at every timestep where a safety indicator is active (TTC < 1.5s,
    THW < 2.0s, speed violation, blind spot occupancy, red light ahead).
    Weight w_j and cap cap_j come from hyperparams.json indicator_weights /
    indicator_caps, precomputed from IDM rollouts.

Episode-level cap for persistent indicators:
    sum_t(c_lead_j) <= cap_j  (see Design Decision #3 in CLAUDE.md)
    Prevents cumulative indicator cost from exceeding outcome event cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------
# Blind spot geometry constants (CLAUDE.md conventions)
# ------------------------------------------------------------------
_BLIND_SPOT_ANGLE_INNER = math.radians(45)    # ±45° from heading
_BLIND_SPOT_ANGLE_OUTER = math.radians(135)   # ±135° from heading
_BLIND_SPOT_LONG_M = 10.0                     # longitudinal depth [m]
_BLIND_SPOT_LAT_M = 4.0                       # half-width [m]


@dataclass
class SafetyCostComponents:
    c_outcome: float = 0.0
    c_lead: float = 0.0

    # Individual outcome flags (for logging)
    vru_collision: bool = False
    vehicle_collision: bool = False
    wrong_direction: bool = False
    red_light: bool = False
    off_road: bool = False
    object_collision: bool = False

    # Individual indicator flags (for logging)
    ttc_active: bool = False
    thw_active: bool = False
    speed_active: bool = False
    blind_spot_active: bool = False
    red_light_ahead_active: bool = False

    @property
    def total(self) -> float:
        return self.c_outcome + self.c_lead


class SafetyCostBuilder:
    """
    Stateful (per-episode) builder for the PRISM safety cost.

    Call reset() at the start of each episode, then compute() at each step.
    """

    def __init__(self, hp: Dict) -> None:
        self._hp = hp
        self._ind_weights: Dict[str, float] = hp.get("indicator_weights", {})
        self._ind_caps: Dict[str, float] = hp.get("indicator_caps", {})
        self._out_weights: Dict[str, float] = hp.get("outcome_weights", {})
        self._thresholds: Dict = hp.get("safety_thresholds", {
            "ttc_threshold_s": 1.5,
            "thw_threshold_s": 2.0,
        })
        # Episode accumulators for cap enforcement
        self._acc: Dict[str, float] = {}
        self.reset()

    def reset(self) -> None:
        self._acc = {ind: 0.0 for ind in self._ind_weights}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        ego_state,
        detection_cache,
        map_cache,
        had_collision: bool,
        is_off_road: bool,
        ran_red_light: bool,
        had_vru_collision: bool,
        had_wrong_direction: bool,
        v_lead: float,
        d_lead: float,
        has_lead: bool,
        v_limit: float,
    ) -> SafetyCostComponents:
        comp = SafetyCostComponents()

        # ── Tier 1: outcome events ──────────────────────────────────────
        if had_vru_collision:
            comp.c_outcome += self._out_weights.get("vru_collision", 100)
            comp.vru_collision = True
        elif had_collision:
            comp.c_outcome += self._out_weights.get("vehicle_collision", 80)
            comp.vehicle_collision = True

        if had_wrong_direction:
            comp.c_outcome += self._out_weights.get("wrong_direction", 100)
            comp.wrong_direction = True

        if ran_red_light:
            comp.c_outcome += self._out_weights.get("red_light_violation", 80)
            comp.red_light = True

        if is_off_road:
            comp.c_outcome += self._out_weights.get("drivable_area", 65)
            comp.off_road = True

        # ── Tier 2: indicator signals ──────────────────────────────────
        v_ego = ego_state.dynamic_car_state.speed

        # TTC indicator
        ttc = self._ttc(v_ego, v_lead, d_lead, has_lead)
        if ttc < self._thresholds.get("ttc_threshold_s", 1.5):
            comp.ttc_active = True
            comp.c_lead += self._cap_indicator("ttc")

        # THW indicator
        thw = self._thw(v_ego, d_lead, has_lead)
        if thw < self._thresholds.get("thw_threshold_s", 2.0):
            comp.thw_active = True
            comp.c_lead += self._cap_indicator("thw")

        # Speed violation indicator
        if v_limit is not None and v_limit > 0 and v_ego > v_limit:
            comp.speed_active = True
            comp.c_lead += self._cap_indicator("speed")

        # Blind spot occupancy indicator
        if self._blind_spot_occupied(ego_state, detection_cache):
            comp.blind_spot_active = True
            comp.c_lead += self._cap_indicator("blind_spot")

        # Red light ahead indicator (ego is approaching a red connector)
        if self._red_light_ahead(ego_state, map_cache):
            comp.red_light_ahead_active = True
            comp.c_lead += self._cap_indicator("red_light")

        return comp

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cap_indicator(self, name: str) -> float:
        """Apply weight subject to episode cap; return the capped contribution."""
        if name not in self._ind_weights:
            return 0.0
        cap = self._ind_caps.get(name, float("inf"))
        acc = self._acc.get(name, 0.0)
        remaining = max(0.0, cap - acc)
        contrib = min(self._ind_weights[name], remaining)
        self._acc[name] = acc + contrib
        return contrib

    @staticmethod
    def _ttc(v_ego: float, v_lead: float, d_lead: float, has_lead: bool) -> float:
        if not has_lead:
            return float("inf")
        closing = v_ego - v_lead
        if closing <= 0.0:
            return float("inf")
        return d_lead / max(closing, 1e-3)

    @staticmethod
    def _thw(v_ego: float, d_lead: float, has_lead: bool) -> float:
        if not has_lead or v_ego <= 0.0:
            return float("inf")
        return d_lead / max(v_ego, 1e-3)

    @staticmethod
    def _blind_spot_occupied(ego_state, detection_cache) -> bool:
        """Return True if any tracked object is in the ego blind spot zone."""
        ex, ey = ego_state.center.x, ego_state.center.y
        heading = ego_state.center.heading
        cos_h, sin_h = math.cos(heading), math.sin(heading)

        try:
            for obj in detection_cache.tracked_objects:
                ox, oy = obj.center.x, obj.center.y
                dx, dy = ox - ex, oy - ey
                d_lon = cos_h * dx + sin_h * dy   # signed forward distance
                d_lat = abs(-sin_h * dx + cos_h * dy)

                # Must be behind ego
                if d_lon > 0 or d_lon < -_BLIND_SPOT_LONG_M:
                    continue
                if d_lat > _BLIND_SPOT_LAT_M:
                    continue

                # Angular check in ego frame
                angle = math.atan2(abs(-sin_h * dx + cos_h * dy), abs(d_lon))
                if _BLIND_SPOT_ANGLE_INNER <= angle <= _BLIND_SPOT_ANGLE_OUTER:
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def _red_light_ahead(ego_state, map_cache) -> bool:
        """Return True if ego is within 30m of a red lane connector on its route."""
        try:
            from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
            from shapely.geometry import Point
            ex, ey = ego_state.center.x, ego_state.center.y
            ego_pt = Point(ex, ey)

            route_ids = set(map_cache.route_roadblock_ids)
            for conn_id, conn in map_cache.lane_connectors.items():
                tl = map_cache.traffic_lights.get(conn_id)
                if tl != TrafficLightStatusType.RED:
                    continue
                if conn.get_roadblock_id() not in route_ids:
                    continue
                if conn.polygon.distance(ego_pt) < 30.0:
                    return True
        except Exception:
            pass
        return False
