"""
Camera-based observation builder for the Alpamayo experiment (Problem 1).

Replaces CaRL's BEV raster observation with raw camera images from nuPlan's
sensor blobs on disk.  Three cameras are used by default: front (CAM_F0),
front-left (CAM_L0), and front-right (CAM_R0).  The channel list is
configurable via YAML so it can be changed without code edits.

Output key: obs["camera_images"]
    Shape:  (num_cameras, num_frames, 3, H, W)
    Dtype:  float32
    Range:  [0, 1]

Image file path pattern (per spec):
    {NUPLAN_SENSOR_ROOT}/{log_name}/{camera_channel}/{image_token}.jpg

The sensor root is read from the environment variable named by
sensor_root_env_var (default "NUPLAN_SENSOR_ROOT").  The image token is
retrieved from the nuPlan database image table via the current log file.

Ego-deviation compensation:
    During closed-loop RL the policy ego deviates from the logged trajectory.
    Camera images in the database were captured from the logged ego pose.
    A homographic warp corrects for the rotational component of this deviation
    (heading difference between logged and actual ego).
    H = K @ R_delta @ K_inv  (pure rotation homography)
    Applied with cv2.warpPerspective, borderMode=cv2.BORDER_REPLICATE.

    Limitation: objects closer than ~5 m may be inaccurately compensated
    due to depth-dependent parallax not captured by a pure rotation homography.

Fallback behaviour:
    If sensor blobs are not available, returns zero tensors and logs a warning
    once per episode.  This allows structural testing before blobs are
    downloaded to the training server.

Frame buffer:
    A rolling deque of depth num_frames is maintained per camera channel.
    At episode start (reset()), the first frame is replicated num_frames times
    to fill the buffer immediately — no partial-frame edge case.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import sqlite3
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces

from carl_nuplan.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_center_state_array,
)
from carl_nuplan.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

from prism.env.spatial_state_serializer import from_info, _N as SPATIAL_DIM

logger = logging.getLogger(__name__)


def _quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """Extract yaw (heading) from a quaternion (w, x, y, z)."""
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


class CameraObservationBuilder:
    """
    Reads logged JPEG camera images from nuPlan sensor blobs, applies
    ego-deviation compensation, and returns preprocessed image tensors
    ready for Alpamayo's visual encoder.

    Designed to be instantiated once and called every RL timestep via
    build_observation().  Call reset() at episode boundaries.

    Args:
        camera_channels:      nuPlan camera channel names to include.
        num_frames:           rolling frame buffer depth (default 4, at 10 Hz
                              this covers 0.4 s of temporal context).
        image_height:         output image height in pixels.
        image_width:          output image width in pixels.
        sensor_root_env_var:  name of the environment variable that holds the
                              sensor blob root directory.
    """

    def __init__(
        self,
        camera_channels: Optional[List[str]] = None,
        num_frames: int = 4,
        image_height: int = 320,
        image_width: int = 576,
        sensor_root_env_var: str = "NUPLAN_SENSOR_ROOT",
        action_space_dim: int = 128,
    ) -> None:
        self._channels = camera_channels or ["CAM_F0", "CAM_L0", "CAM_R0"]
        self._num_frames = num_frames
        self._H = image_height
        self._W = image_width
        self._action_dim = action_space_dim

        sensor_root_str = os.environ.get(sensor_root_env_var, "")
        self._sensor_root: Optional[Path] = Path(sensor_root_str) if sensor_root_str else None

        # Per-channel rolling frame buffers
        self._buffers: Dict[str, deque] = {ch: deque(maxlen=num_frames) for ch in self._channels}

        # K matrices loaded once per episode from the nuPlan DB
        self._intrinsics: Dict[str, np.ndarray] = {}

        # Resolved at the first build_observation() call after each reset()
        self._log_file: Optional[str] = None
        self._log_name: Optional[str] = None

        # Back-reference to PRISMEnv — injected via set_env() after construction
        self._env = None

        # Warn-once flags (reset each episode)
        self._missing_root_warned = False
        self._missing_blob_warned = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_env(self, env) -> None:
        """
        Inject a reference to the enclosing PRISMEnv so the builder can
        access self._env._simulation_wrapper.scenario to locate the log file.
        Call from _build_env() in train.py after both objects are constructed.
        """
        self._env = env

    def reset(self) -> None:
        """Called at episode start.  Clears frame buffers and per-episode caches."""
        for ch in self._channels:
            self._buffers[ch].clear()
        self._intrinsics = {}
        self._log_file = None
        self._log_name = None
        self._missing_blob_warned = False

    def get_observation_space(self) -> spaces.Space:
        """Return the gymnasium observation space for the full Alpamayo obs dict."""
        return spaces.Dict({
            "camera_images": spaces.Box(
                0.0, 1.0,
                shape=(len(self._channels), self._num_frames, 3, self._H, self._W),
                dtype=np.float32,
            ),
            "measurements": spaces.Box(
                -math.inf, math.inf, shape=(10,), dtype=np.float32
            ),
            "value_measurements": spaces.Box(
                -math.inf, math.inf, shape=(4,), dtype=np.float32
            ),
            "spatial_state": spaces.Box(
                -math.inf, math.inf, shape=(SPATIAL_DIM,), dtype=np.float32
            ),
        })

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------

    def build_observation(
        self,
        planner_input,
        planner_initialization,
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the full observation dict for the current simulation step.

        Returns:
            camera_images     : (num_cameras, num_frames, 3, H, W) float32 [0,1]
            measurements      : (10,) float32 — ego kinematics
            value_measurements: (4,) float32  — zeros; PRISMEnv overwrites with z_t
            spatial_state     : (9,) float32  — CaRL spatial quantities from info dict
        """
        ego_state = planner_input.history.current_state[0]
        timestamp_us = ego_state.time_point.time_us

        self._resolve_log_file()

        if self._log_file and not self._intrinsics:
            self._load_intrinsics(self._log_file)

        per_channel: List[np.ndarray] = []
        for ch in self._channels:
            frame = self._get_frame(ch, timestamp_us, ego_state)
            self._buffers[ch].append(frame)

            # Cold-start: replicate first frame until buffer is full
            while len(self._buffers[ch]) < self._num_frames:
                self._buffers[ch].appendleft(frame)

            # (num_frames, 3, H, W)
            per_channel.append(np.stack(list(self._buffers[ch]), axis=0))

        # (num_cameras, num_frames, 3, H, W)
        camera_images = np.stack(per_channel, axis=0)

        return {
            "camera_images": torch.from_numpy(camera_images),
            "measurements": self._build_measurements(planner_input, info),
            "value_measurements": np.zeros(4, dtype=np.float32),  # PRISMEnv overwrites
            "spatial_state": from_info(info),
        }

    # ------------------------------------------------------------------
    # Ego kinematics vector (same layout as DefaultObservationBuilder)
    # ------------------------------------------------------------------

    def _build_measurements(
        self, planner_input, info: Dict[str, Any]
    ) -> npt.NDArray[np.float32]:
        ego_state = planner_input.history.current_state[0]
        last_action = info.get("last_action", np.zeros(self._action_dim, dtype=np.float32))
        state_array = ego_state_to_center_state_array(ego_state)
        return np.array([
            float(last_action[0]) if len(last_action) > 0 else 0.0,
            float(last_action[1]) if len(last_action) > 1 else 0.0,
            state_array[StateIndex.VELOCITY_X],
            state_array[StateIndex.VELOCITY_Y],
            state_array[StateIndex.ACCELERATION_X],
            state_array[StateIndex.ACCELERATION_Y],
            state_array[StateIndex.STEERING_ANGLE],
            state_array[StateIndex.STEERING_RATE],
            state_array[StateIndex.ANGULAR_VELOCITY],
            state_array[StateIndex.ANGULAR_ACCELERATION],
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Log file resolution
    # ------------------------------------------------------------------

    def _resolve_log_file(self) -> None:
        if self._log_file is not None:
            return
        if self._env is None:
            return

        sim_wrapper = getattr(self._env, "_simulation_wrapper", None)
        if sim_wrapper is None:
            return

        scenario = getattr(sim_wrapper, "scenario", None)
        if scenario is None:
            return

        log_file = getattr(scenario, "_log_file", None)
        if log_file:
            self._log_file = str(log_file)
            self._log_name = Path(self._log_file).stem

    # ------------------------------------------------------------------
    # Intrinsic loading
    # ------------------------------------------------------------------

    def _load_intrinsics(self, log_file: str) -> None:
        """Load and cache 3×3 K matrices for all channels from the nuPlan DB."""
        try:
            conn = sqlite3.connect(log_file)
            cur = conn.cursor()
            placeholders = ",".join("?" * len(self._channels))
            cur.execute(
                f"SELECT channel, intrinsic FROM camera WHERE channel IN ({placeholders})",
                tuple(self._channels),
            )
            for channel, blob in cur.fetchall():
                intr = pickle.loads(blob)  # CameraIntrinsic: .fx, .fy, .px, .py
                self._intrinsics[channel] = np.array(
                    [[intr.fx, 0.0, intr.px],
                     [0.0, intr.fy, intr.py],
                     [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
            conn.close()
        except Exception as e:
            logger.warning(f"[CameraObservationBuilder] Could not load intrinsics: {e}")

    # ------------------------------------------------------------------
    # Per-frame acquisition
    # ------------------------------------------------------------------

    def _get_frame(self, channel: str, timestamp_us: int, actual_ego_state) -> np.ndarray:
        """Load, compensate, resize, and normalise one frame.  Returns zeros on failure."""
        blank = np.zeros((3, self._H, self._W), dtype=np.float32)

        if self._sensor_root is None:
            if not self._missing_root_warned:
                logger.warning(
                    "[CameraObservationBuilder] NUPLAN_SENSOR_ROOT is not set — "
                    "returning zero frames. Set the env var when sensor blobs are available."
                )
                self._missing_root_warned = True
            return blank

        if self._log_file is None or self._log_name is None:
            return blank

        row = self._query_image(self._log_file, channel, timestamp_us)
        if row is None:
            return blank

        image_token, log_qw, log_qx, log_qy, log_qz = row

        # Construct path per spec: {NUPLAN_SENSOR_ROOT}/{log_name}/{channel}/{token}.jpg
        path = self._sensor_root / self._log_name / channel / f"{image_token}.jpg"

        if not path.exists():
            if not self._missing_blob_warned:
                logger.warning(
                    f"[CameraObservationBuilder] Sensor blob not found: {path}. "
                    "Returning zero frames. Download sensor blobs to enable real images."
                )
                self._missing_blob_warned = True
            return blank

        img = self._load_jpeg(path)
        if img is None:
            return blank

        # Ego-deviation compensation
        logged_heading = _quat_to_yaw(log_qw, log_qx, log_qy, log_qz)
        actual_heading = float(actual_ego_state.rear_axle.heading)
        img = self._compensate_deviation(img, channel, logged_heading, actual_heading)

        # Resize to target resolution
        try:
            import cv2
            img = cv2.resize(img, (self._W, self._H), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return blank

        # Normalise [0, 255] → [0, 1] and convert HWC → CHW
        img = img.astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))

    def _query_image(self, log_file: str, channel: str, timestamp_us: int) -> Optional[tuple]:
        """
        Return (token, qw, qx, qy, qz) for the image nearest to timestamp_us
        for the given camera channel, or None on failure.
        """
        try:
            conn = sqlite3.connect(log_file)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT img.token,
                       ep.qw, ep.qx, ep.qy, ep.qz
                FROM   image     img
                JOIN   camera    cam ON img.camera_token    = cam.token
                JOIN   ego_pose  ep  ON img.ego_pose_token  = ep.token
                WHERE  cam.channel = ?
                ORDER  BY ABS(img.timestamp - ?)
                LIMIT  1
                """,
                (channel, timestamp_us),
            )
            row = cur.fetchone()
            conn.close()
            return row
        except Exception as e:
            logger.debug(f"[CameraObservationBuilder] DB query failed ({channel}): {e}")
            return None

    @staticmethod
    def _load_jpeg(path: Path) -> Optional[np.ndarray]:
        """Load JPEG at path and return an RGB HWC uint8 array, or None on failure."""
        try:
            import cv2
            bgr = cv2.imread(str(path))
            if bgr is None:
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Ego-deviation compensation  H = K @ R_delta @ K_inv
    # ------------------------------------------------------------------

    def _compensate_deviation(
        self,
        image: np.ndarray,
        channel: str,
        logged_heading: float,
        actual_heading: float,
    ) -> np.ndarray:
        """
        Warp image to compensate for the heading difference between the logged
        ego pose (at which the image was captured) and the actual ego pose
        (current simulation state).

        Compensates exactly for pure rotation.  Translation-induced parallax
        is ignored — a good approximation for objects beyond ~10 m.
        """
        K = self._intrinsics.get(channel)
        if K is None:
            return image

        try:
            import cv2
            d = actual_heading - logged_heading
            cos_d, sin_d = np.cos(d), np.sin(d)
            R_delta = np.array(
                [[cos_d, -sin_d, 0.0],
                 [sin_d,  cos_d, 0.0],
                 [0.0,    0.0,   1.0]],
                dtype=np.float64,
            )
            H = K @ R_delta @ np.linalg.inv(K)
            h, w = image.shape[:2]
            return cv2.warpPerspective(
                image, H, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        except Exception:
            return image
