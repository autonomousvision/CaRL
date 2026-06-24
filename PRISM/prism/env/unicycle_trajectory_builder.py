"""
Trajectory builder for Alpamayo's 64-waypoint (acceleration, curvature) output.

Alpamayo predicts a trajectory as 64 waypoints of (acceleration [m/s²], curvature
[1/m]) via flow-matching.  This builder integrates those waypoints using unicycle
kinematics to produce the EgoState sequence that nuPlan's InterpolatedTrajectory
expects.

Action format: float32 array of shape (128,) = 64 × [accel, curvature], flattened
in row-major order.  Bounds match Alpamayo's action_space config:
    accel:     [-9.8, 9.8]  m/s²
    curvature: [-0.33, 0.33] 1/m

Unicycle kinematics (integrated at dt=0.1s):
    v_{t+1}       = max(0, v_t + a_t * dt)
    omega_t       = v_t * kappa_t                    (angular velocity [rad/s])
    heading_{t+1} = heading_t + omega_t * dt
    x_{t+1}       = x_t + v_t * cos(heading_t) * dt
    y_{t+1}       = y_t + v_t * sin(heading_t) * dt
    delta_t       = arctan(kappa_t * L)              (tire steering angle)

where L is the vehicle wheelbase (read from EgoState vehicle parameters).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimeDuration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from carl_nuplan.planning.gym.environment.trajectory_builder.abstract_trajectory_builder import (
    AbstractTrajectoryBuilder,
)


class UnicycleTrajectoryBuilder(AbstractTrajectoryBuilder):
    """
    Implements AbstractTrajectoryBuilder for Alpamayo's 64-waypoint output.

    Args:
        n_waypoints:      number of future waypoints (64 for Alpamayo).
        dt:               simulation timestep [s] (0.1 for nuPlan).
        accel_bounds:     (min, max) clipping for acceleration [m/s²].
        curvature_bounds: (min, max) clipping for curvature [1/m].
    """

    def __init__(
        self,
        n_waypoints: int = 64,
        dt: float = 0.1,
        accel_bounds: tuple = (-9.8, 9.8),
        curvature_bounds: tuple = (-0.33, 0.33),
    ) -> None:
        self._n = n_waypoints
        self._dt = dt
        self._accel_bounds = accel_bounds
        self._curv_bounds = curvature_bounds
        self._action_dim = n_waypoints * 2

    def get_action_space(self) -> spaces.Box:
        low = np.array(
            [self._accel_bounds[0], self._curv_bounds[0]] * self._n,
            dtype=np.float32,
        )
        high = np.array(
            [self._accel_bounds[1], self._curv_bounds[1]] * self._n,
            dtype=np.float32,
        )
        return spaces.Box(low=low, high=high, shape=(self._action_dim,), dtype=np.float32)

    def build_trajectory(
        self,
        action: npt.NDArray[np.float32],
        ego_state: EgoState,
        info: Dict[str, Any],
    ) -> InterpolatedTrajectory:
        """
        Integrate 64 (accel, curvature) waypoints into an InterpolatedTrajectory.

        Args:
            action:    (128,) array — [a_0, k_0, a_1, k_1, ..., a_63, k_63]
            ego_state: current ego state (provides initial pose + vehicle params)
            info:      passed through to environment info dict
        Returns:
            InterpolatedTrajectory with 65 EgoState points (current + 64 future)
        """
        waypoints = action.reshape(self._n, 2)
        accels = np.clip(waypoints[:, 0], *self._accel_bounds)
        curvatures = np.clip(waypoints[:, 1], *self._curv_bounds)

        vehicle_params = ego_state.car_footprint.vehicle_parameters
        wheelbase = float(vehicle_params.wheel_base)

        # Initial state from ego
        x = float(ego_state.rear_axle.x)
        y = float(ego_state.rear_axle.y)
        heading = float(ego_state.rear_axle.heading)
        velocity = float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x)

        time_dot = TimeDuration.from_s(self._dt)
        states = [ego_state]

        for i in range(self._n):
            a = float(accels[i])
            kappa = float(curvatures[i])

            # Angular velocity and next heading
            omega = velocity * kappa
            next_heading = heading + omega * self._dt

            # Next position (integrate at current heading)
            next_x = x + velocity * np.cos(heading) * self._dt
            next_y = y + velocity * np.sin(heading) * self._dt

            # Next velocity (clamp to zero — no reverse)
            next_velocity = max(0.0, velocity + a * self._dt)

            # Steering angle from curvature: delta = arctan(kappa * L)
            steering_angle = float(np.arctan(kappa * wheelbase))

            state = EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(next_x, next_y, next_heading),
                rear_axle_velocity_2d=StateVector2D(next_velocity, 0.0),
                rear_axle_acceleration_2d=StateVector2D(a, 0.0),
                tire_steering_angle=steering_angle,
                time_point=ego_state.time_point + (i + 1) * time_dot,
                vehicle_parameters=vehicle_params,
                is_in_auto_mode=True,
                angular_vel=omega,
                angular_accel=0.0,
            )
            states.append(state)

            x, y, heading, velocity = next_x, next_y, next_heading, next_velocity

        info["last_action"] = action
        return InterpolatedTrajectory(states)
