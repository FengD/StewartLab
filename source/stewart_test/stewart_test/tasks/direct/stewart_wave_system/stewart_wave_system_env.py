# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from ..stewart_test.stewart_test_env import StewartTestEnv
from .stewart_wave_system_env_cfg import StewartWaveSystemEnvCfg


class StewartWaveSystemEnv(StewartTestEnv):
    cfg: StewartWaveSystemEnvCfg

    def __init__(self, cfg: StewartWaveSystemEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._wave_amp = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        self._wave_freq = torch.zeros_like(self._wave_amp)
        self._wave_phase = torch.zeros_like(self._wave_amp)
        self._sample_wave_params(self.robot._ALL_INDICES)
        self._apply_base_wave_motion(self.robot._ALL_INDICES)
        self._reset_ball_above_platform(self.robot._ALL_INDICES)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.wave_base = RigidObject(self.cfg.wave_base_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["ball"] = self.object
        self.scene.rigid_objects["wave_base"] = self.wave_base
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_action(self) -> None:
        if hasattr(self, "_wave_amp"):
            self._apply_base_wave_motion(self.robot._ALL_INDICES)
        super()._apply_action()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        if not hasattr(self, "_wave_amp"):
            super()._reset_idx(env_ids)
            return

        self._sample_wave_params(env_ids)
        super()._reset_idx(env_ids)
        self._apply_base_wave_motion(env_ids)
        self._reset_ball_above_platform(env_ids)
        self._compute_intermediate_values()

    def _sample_wave_params(self, env_ids: Sequence[int] | torch.Tensor):
        count = len(env_ids)
        pos_amp = torch.tensor(self.cfg.wave_pos_amplitude, dtype=torch.float, device=self.device)
        rot_amp = torch.tensor(self.cfg.wave_rot_amplitude, dtype=torch.float, device=self.device)
        max_amp = torch.cat((pos_amp, rot_amp), dim=0)
        self._wave_amp[env_ids] = max_amp * sample_uniform(0.35, 1.0, (count, 6), self.device)
        self._wave_freq[env_ids] = sample_uniform(
            self.cfg.wave_frequency_range[0], self.cfg.wave_frequency_range[1], (count, 6), self.device
        )
        self._wave_phase[env_ids] = sample_uniform(0.0, 6.28318530718, (count, 6), self.device)

    def _apply_base_wave_motion(self, env_ids: Sequence[int] | torch.Tensor):
        time = self.episode_length_buf[env_ids].float().unsqueeze(-1) * self.step_dt
        omega = 6.28318530718 * self._wave_freq[env_ids]
        phase = omega * time + self._wave_phase[env_ids]
        wave = self._wave_amp[env_ids] * torch.sin(phase)
        wave_vel = self._wave_amp[env_ids] * omega * torch.cos(phase)

        root_pos = self.scene.env_origins[env_ids] + wave[:, :3]
        root_pos[:, 2] += self.cfg.system_z_offset
        root_quat = quat_from_rpy(wave[:, 3], wave[:, 4], wave[:, 5])
        root_vel = torch.cat((wave_vel[:, :3], wave_vel[:, 3:]), dim=-1)

        self.robot.write_root_pose_to_sim(torch.cat((root_pos, root_quat), dim=-1), env_ids)
        self.robot.write_root_velocity_to_sim(root_vel, env_ids)

        base_pos = root_pos.clone()
        base_pos[:, 2] += self.cfg.wave_base_z_offset
        self.wave_base.write_root_pose_to_sim(torch.cat((base_pos, root_quat), dim=-1), env_ids)
        self.wave_base.write_root_velocity_to_sim(root_vel, env_ids)

    def _reset_ball_above_platform(self, env_ids: Sequence[int] | torch.Tensor):
        self._compute_intermediate_values()
        object_state = self.object.data.default_root_state[env_ids].clone()
        rand_radius = self.cfg.object_spawn_radius * torch.sqrt(torch.rand(len(env_ids), device=self.device))
        rand_angle = sample_uniform(0.0, 6.28318530718, (len(env_ids),), self.device)
        object_xy = torch.stack((rand_radius * torch.cos(rand_angle), rand_radius * torch.sin(rand_angle)), dim=-1)
        object_height = sample_uniform(
            self.cfg.object_drop_height_range[0],
            self.cfg.object_drop_height_range[1],
            (len(env_ids),),
            self.device,
        )
        object_state[:, 0:2] = self.platform_center_pos[env_ids, 0:2] + object_xy
        object_state[:, 2] = self.platform_center_pos[env_ids, 2] + object_height
        object_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float, device=self.device)
        object_state[:, 7:] = 0.0
        object_state[:, 9] = self.cfg.object_initial_down_velocity
        self.object.write_root_pose_to_sim(object_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids)


def quat_from_rpy(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )
