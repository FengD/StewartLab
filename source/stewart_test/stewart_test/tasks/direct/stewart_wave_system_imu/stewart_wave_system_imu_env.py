# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..stewart_test.stewart_test_env import StewartTestEnv
from ..stewart_wave_system.stewart_wave_system_env import StewartWaveSystemEnv, quat_rotate_inverse
from .stewart_wave_system_imu_env_cfg import StewartWaveSystemImuEnvCfg


class StewartWaveSystemImuEnv(StewartWaveSystemEnv):
    """Wave-base task with IMU-only disturbance observations.

    The policy does not receive commanded wave pose/velocity. Instead, it receives
    IMU-like measurements from the Stewart root frame.
    """

    cfg: StewartWaveSystemImuEnvCfg

    def __init__(self, cfg: StewartWaveSystemImuEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._prev_root_lin_vel_w = self.robot.data.root_lin_vel_w.clone()
        self._root_lin_acc_w = torch.zeros_like(self._prev_root_lin_vel_w)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._prev_root_lin_vel_w[env_ids_t] = self.robot.data.root_lin_vel_w[env_ids_t]
        self._root_lin_acc_w[env_ids_t] = 0.0

    def _update_root_imu_acc(self):
        root_lin_vel_w = self.robot.data.root_lin_vel_w
        root_lin_acc_w = (root_lin_vel_w - self._prev_root_lin_vel_w) / max(float(self.step_dt), 1.0e-6)
        accel_clip = float(getattr(self.cfg, "imu_accel_clip", 0.0))
        if accel_clip > 0.0:
            root_lin_acc_w = torch.clamp(root_lin_acc_w, -accel_clip, accel_clip)
        self._root_lin_acc_w = root_lin_acc_w
        self._prev_root_lin_vel_w = root_lin_vel_w.clone()

    def _get_observations(self) -> dict:
        # Get base 31D observation from fixed-base parent (without wave command extras).
        obs_dict = StewartTestEnv._get_observations(self)
        obs = obs_dict["policy"]

        self._update_root_imu_acc()
        root_quat_w = self.robot.data.root_quat_w
        gravity_vec_w = self._gravity_vec_w
        imu_projected_gravity = quat_rotate_inverse(root_quat_w, gravity_vec_w)
        root_ang_vel_w = getattr(self.robot.data, "root_ang_vel_w", self._wave_root_vel_w[:, 3:6])
        imu_gyro = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
        imu_lin_acc = quat_rotate_inverse(root_quat_w, self._root_lin_acc_w)

        obs_dict["policy"] = torch.cat((obs, imu_projected_gravity, imu_gyro, imu_lin_acc), dim=-1)
        return obs_dict
