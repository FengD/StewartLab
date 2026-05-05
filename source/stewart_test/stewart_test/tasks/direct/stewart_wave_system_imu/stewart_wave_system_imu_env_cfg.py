# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ..stewart_wave_system.stewart_wave_system_env_cfg import StewartWaveSystemEnvCfg


@configclass
class StewartWaveSystemImuEnvCfg(StewartWaveSystemEnvCfg):
    # Base obs = 31; append IMU [projected_gravity(3), gyro(3), linear_acc(3)] => 40.
    observation_space = 40
    # Clip simulated linear acceleration to a realistic range for stable learning.
    imu_accel_clip = 12.0
