"""
Driving regime detector.

Detects one of three regimes and returns the desired speed v_des:
    1. Congested     : v_lane_avg < 0.5 * v_limit  →  v_des = v_lane_avg
    2. Car-following : lead vehicle within horizon   →  v_des = v_lead
    3. Free-flow     : otherwise                     →  v_des = v_limit

Check order is congested → car-following → free-flow.
This prevents a slow lead vehicle from triggering car-following when the
broader lane is already congested (Key Design Decision #6 in CLAUDE.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np


class Regime(Enum):
    CONGESTED = auto()
    CAR_FOLLOWING = auto()
    FREE_FLOW = auto()


@dataclass
class RegimeResult:
    regime: Regime
    v_des: float       # desired speed [m/s]
    v_lead: float      # lead vehicle speed [m/s], 0 if no lead
    d_lead: float      # longitudinal distance to lead vehicle [m], inf if no lead
    has_lead: bool     # whether a lead vehicle was detected
    v_lane_avg: float  # average speed of nearby vehicles [m/s]
    v_limit: float     # speed limit of current lane [m/s]


class RegimeDetector:
    """
    Extracts regime and desired speed from the nuPlan simulation state.

    Parameters
    ----------
    congestion_speed_fraction : float
        Threshold fraction of v_limit below which lane is considered congested.
    observation_horizon_m : float
        Longitudinal look-ahead and look-behind distance for lane traffic [m].
    lead_lateral_tolerance_m : float
        Maximum lateral offset for an object to count as in the ego lane [m].
    """

    DEFAULT_SPEED_LIMIT_MPS: float = 13.89  # 50 km/h fallback

    def __init__(
        self,
        congestion_speed_fraction: float = 0.5,
        observation_horizon_m: float = 50.0,
        lead_lateral_tolerance_m: float = 2.5,
    ) -> None:
        self._cong_frac = congestion_speed_fraction
        self._horizon = observation_horizon_m
        self._lat_tol = lead_lateral_tolerance_m

    def detect(
        self,
        ego_state,        # nuPlan EgoState
        map_cache,        # MapCache from CaRL
        detection_cache,  # DetectionCache from CaRL
    ) -> RegimeResult:
        v_ego = ego_state.dynamic_car_state.speed
        v_limit = self._get_speed_limit(ego_state, map_cache)

        v_lead, d_lead, has_lead = self._find_lead(ego_state, detection_cache)
        v_lane_avg = self._lane_avg_speed(ego_state, detection_cache)

        # 1. Congested?
        if v_lane_avg < self._cong_frac * v_limit:
            return RegimeResult(
                regime=Regime.CONGESTED,
                v_des=max(v_lane_avg, 0.3),  # keep positive to avoid division by zero
                v_lead=v_lead,
                d_lead=d_lead,
                has_lead=has_lead,
                v_lane_avg=v_lane_avg,
                v_limit=v_limit,
            )

        # 2. Car-following?
        if has_lead and d_lead < self._horizon:
            return RegimeResult(
                regime=Regime.CAR_FOLLOWING,
                v_des=max(v_lead, 0.0),
                v_lead=v_lead,
                d_lead=d_lead,
                has_lead=has_lead,
                v_lane_avg=v_lane_avg,
                v_limit=v_limit,
            )

        # 3. Free-flow
        return RegimeResult(
            regime=Regime.FREE_FLOW,
            v_des=v_limit,
            v_lead=v_lead,
            d_lead=d_lead,
            has_lead=has_lead,
            v_lane_avg=v_lane_avg,
            v_limit=v_limit,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_speed_limit(self, ego_state, map_cache) -> float:
        """Return speed limit of current lane, or default if unavailable."""
        try:
            from nuplan.common.maps.abstract_map import Lane
            ego_center = ego_state.center
            lanes = map_cache.lanes
            lane_connectors = map_cache.lane_connectors
            all_lanes = {**lanes, **lane_connectors}

            from shapely.geometry import Point
            ego_pt = Point(ego_center.x, ego_center.y)

            best_lane = None
            best_dist = float("inf")
            for lane in all_lanes.values():
                try:
                    dist = lane.polygon.distance(ego_pt)
                    if dist < best_dist:
                        best_dist = dist
                        best_lane = lane
                except Exception:
                    continue

            if best_lane is not None and isinstance(best_lane, Lane):
                if best_lane.speed_limit_mps is not None:
                    return float(best_lane.speed_limit_mps)
        except Exception:
            pass
        return self.DEFAULT_SPEED_LIMIT_MPS

    def _find_lead(
        self, ego_state, detection_cache
    ) -> Tuple[float, float, bool]:
        """Find the closest vehicle directly ahead in the ego lane."""
        cos_h, sin_h = self._ego_basis(ego_state)
        ex, ey = ego_state.center.x, ego_state.center.y

        best_d = float("inf")
        v_lead = 0.0
        has_lead = False

        try:
            for obj in detection_cache.tracked_objects:
                ox, oy = obj.center.x, obj.center.y
                dx, dy = ox - ex, oy - ey
                d_lon = cos_h * dx + sin_h * dy
                d_lat = abs(-sin_h * dx + cos_h * dy)

                if d_lon <= 0.5 or d_lat > self._lat_tol:
                    continue
                if d_lon < best_d:
                    best_d = d_lon
                    try:
                        v_lead = float(obj.velocity.x * cos_h + obj.velocity.y * sin_h)
                    except Exception:
                        v_lead = 0.0
                    has_lead = True
        except Exception:
            pass

        return v_lead, best_d, has_lead

    def _lane_avg_speed(self, ego_state, detection_cache) -> float:
        """Average speed of vehicles within ±horizon ahead/behind in ego frame."""
        cos_h, sin_h = self._ego_basis(ego_state)
        ex, ey = ego_state.center.x, ego_state.center.y

        speeds = []
        try:
            for obj in detection_cache.tracked_objects:
                ox, oy = obj.center.x, obj.center.y
                dx, dy = ox - ex, oy - ey
                d_lon = abs(cos_h * dx + sin_h * dy)
                d_lat = abs(-sin_h * dx + cos_h * dy)
                if d_lon <= self._horizon and d_lat <= self._lat_tol:
                    try:
                        spd = math.sqrt(obj.velocity.x ** 2 + obj.velocity.y ** 2)
                        speeds.append(spd)
                    except Exception:
                        pass
        except Exception:
            pass

        if speeds:
            return float(np.mean(speeds))
        # No surrounding vehicles; treat as free-flow
        return self.DEFAULT_SPEED_LIMIT_MPS

    @staticmethod
    def _ego_basis(ego_state) -> Tuple[float, float]:
        h = ego_state.center.heading
        return math.cos(h), math.sin(h)
