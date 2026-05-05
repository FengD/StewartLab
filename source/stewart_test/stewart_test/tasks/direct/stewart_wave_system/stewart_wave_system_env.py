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

from ..stewart_test.stewart_test_env import StewartTestEnv, compute_rewards, quat_rotate_inverse
from .stewart_wave_system_env_cfg import StewartWaveSystemEnvCfg


class StewartWaveSystemEnv(StewartTestEnv):
    cfg: StewartWaveSystemEnvCfg

    def __init__(self, cfg: StewartWaveSystemEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._wave_amp = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        self._wave_freq = torch.zeros_like(self._wave_amp)
        self._wave_phase = torch.zeros_like(self._wave_amp)
        # Buffers for current commanded root motion (used for observations / reward shaping).
        self._wave_root_quat_w = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self._wave_root_vel_w = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        self._wave_pose = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
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

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Termination aligned with wave base: avoid false positives from base motion."""
        self._compute_intermediate_values()
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        object_hit_ground = self.object_pos[:, 2] < (
            self.cfg.ground_height + self.cfg.object_termination_radius
        )

        xy_err = torch.linalg.norm(self.object_rel_pos[:, :2], dim=1)
        xy_gate = self.object_rel_pos[:, 2] < self.cfg.object_far_rel_height_gate
        object_far = (xy_err > self.cfg.max_object_xy_dist) & xy_gate

        # Do not penalize mandated root tilt — only Stewart-on-top surplus tilt vs commanded wave pose.
        # Use simulated root orientation as reference to avoid command/sim skew.
        q_rel = quat_mul(quat_conjugate(self.robot.data.root_quat_w), self.platform_quat)
        g_res = quat_rotate_inverse(q_rel, self._gravity_vec_w)
        platform_tilted = torch.linalg.norm(g_res[:, :2], dim=1) > self.cfg.max_platform_tilt

        return object_hit_ground | object_far | platform_tilted, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        if not hasattr(self, "_wave_amp"):
            super()._reset_idx(env_ids)
            return

        self._sample_wave_params(env_ids)
        self._defer_nominal_capture = True
        super()._reset_idx(env_ids)
        self._defer_nominal_capture = False
        self._apply_base_wave_motion(env_ids)
        self._reset_ball_above_platform(env_ids)
        self._compute_intermediate_values()
        self._capture_nominal_disk_offset(env_ids)
        # Expose wave-curriculum state in training logs for easier diagnosis.
        if "log" in self.extras:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            progress = self._get_curriculum_progress_tensor(env_ids_t).squeeze(-1)
            axis_scale = self._get_wave_axis_scale(progress)
            self.extras["log"]["Curriculum/global_progress"] = float(self._get_curriculum_progress())
            self.extras["log"]["Wave/mean_env_progress"] = float(torch.mean(progress).item())
            self.extras["log"]["Wave/mean_amp_xyz"] = float(torch.mean(self._wave_amp[env_ids_t, :3]).item())
            self.extras["log"]["Wave/mean_amp_rpy"] = float(torch.mean(self._wave_amp[env_ids_t, 3:]).item())
            self.extras["log"]["Wave/axis_scale_x"] = float(torch.mean(axis_scale[:, 0]).item())
            self.extras["log"]["Wave/axis_scale_yaw"] = float(torch.mean(axis_scale[:, 5]).item())

    def _sample_wave_params(self, env_ids: Sequence[int] | torch.Tensor):
        count = len(env_ids)
        progress = self._get_curriculum_progress_tensor(env_ids).squeeze(-1)
        pos_start = torch.tensor(self.cfg.wave_pos_amplitude_start, dtype=torch.float, device=self.device)
        pos_end = torch.tensor(self.cfg.wave_pos_amplitude, dtype=torch.float, device=self.device)
        rot_start = torch.tensor(self.cfg.wave_rot_amplitude_start, dtype=torch.float, device=self.device)
        rot_end = torch.tensor(self.cfg.wave_rot_amplitude, dtype=torch.float, device=self.device)
        axis_scale = self._get_wave_axis_scale(progress)
        # Keep non-zero start amplitudes active at low progress, while axis_scale gates only the incremental difficulty.
        start_amp = torch.cat((pos_start, rot_start), dim=0).unsqueeze(0).expand(count, -1)
        end_amp = torch.cat((pos_end, rot_end), dim=0).unsqueeze(0).expand(count, -1)
        max_amp = start_amp + (end_amp - start_amp) * (progress.unsqueeze(-1) * axis_scale)
        freq_low = self._lerp_cfg_tensor(
            self.cfg.wave_frequency_range_start[0], self.cfg.wave_frequency_range[0], progress
        )
        freq_high = self._lerp_cfg_tensor(
            self.cfg.wave_frequency_range_start[1], self.cfg.wave_frequency_range[1], progress
        )
        self._wave_amp[env_ids] = max_amp * sample_uniform(0.35, 1.0, (count, 6), self.device)
        self._wave_freq[env_ids] = freq_low.unsqueeze(-1) + (freq_high - freq_low).unsqueeze(-1) * torch.rand(
            (count, 6), device=self.device
        )
        self._wave_phase[env_ids] = sample_uniform(0.0, 6.28318530718, (count, 6), self.device)

    def _get_wave_axis_scale(self, progress: torch.Tensor) -> torch.Tensor:
        ramp = max(float(self.cfg.wave_axis_ramp_progress), 1.0e-6)
        starts = torch.tensor(self.cfg.wave_axis_start_progress, dtype=torch.float, device=self.device)
        return torch.clamp((progress.unsqueeze(-1) - starts) / ramp, 0.0, 1.0)

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

        self._wave_pose[env_ids] = wave
        self._wave_root_quat_w[env_ids] = root_quat
        self._wave_root_vel_w[env_ids] = root_vel

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

    def _get_observations(self) -> dict:
        """Augment base observations with wave/root motion for feed-forward compensation."""
        obs_dict = super()._get_observations()
        obs = obs_dict["policy"]
        # Include the full commanded wave pose/velocity so the policy can learn feed-forward compensation.
        gravity_vec_w = self._gravity_vec_w
        # Gravity projection should reflect the actual simulated base orientation.
        root_proj_g = quat_rotate_inverse(self.robot.data.root_quat_w, gravity_vec_w)
        obs_dict["policy"] = torch.cat((obs, root_proj_g, self._wave_pose, self._wave_root_vel_w), dim=-1)
        return obs_dict

    def _get_rewards(self) -> torch.Tensor:
        """Use the same reward structure as base task but only penalize residual motion w.r.t. the wave base."""
        # Compute base intermediates.
        self._compute_intermediate_values()

        # Residual angular velocity: platform body ang vel minus root ang vel.
        # Note: root ang vel is what we command in `_apply_base_wave_motion` (world frame).
        platform_ang_vel_res = self.platform_ang_vel - self._wave_root_vel_w[:, 3:6]

        # Residual "flatness": compute gravity projected into the platform frame relative to the root frame.
        # q_rel = q_root^{-1} * q_platform
        q_rel = quat_mul(quat_conjugate(self.robot.data.root_quat_w), self.platform_quat)
        platform_proj_g_res = quat_rotate_inverse(q_rel, self._gravity_vec_w)

        disk_off_b = quat_rotate_inverse(
            self.robot.data.root_quat_w, self.platform_center_pos - self.robot.data.root_pos_w
        )

        rewards = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_center,
            self.cfg.rew_scale_center_velocity,
            self.cfg.rew_scale_height,
            self.cfg.rew_scale_lin_vel,
            self.cfg.rew_scale_ang_vel,
            self.cfg.rew_scale_intercept,
            self.cfg.rew_scale_on_disk,
            self.cfg.rew_scale_world_center_hold,
            self.cfg.rew_scale_platform_flat,
            self.cfg.rew_scale_action,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_action_rate_on_disk,
            self.cfg.rew_scale_slider_vel,
            self.cfg.rew_scale_catch,
            self.cfg.rew_scale_platform_ang_vel,
            self.cfg.rew_scale_platform_lin_residual,
            self.cfg.rew_scale_disk_pose_reg,
            self.cfg.rew_scale_terminated,
            self.object_rel_pos,
            self.object_lin_vel,
            self.object_ang_vel,
            self.platform_center_pos,
            self._nominal_platform_center_w,
            platform_proj_g_res,
            platform_ang_vel_res,
            self.robot.data.root_lin_vel_w,
            disk_off_b,
            self._nominal_disk_offset_root_b,
            self.platform_lin_vel_w,
            self.joint_vel[:, self._slider_dof_idx],
            self._processed_actions,
            self._prev_processed_actions,
            self.reset_terminated,
            self.cfg.object_target_rel_height,
            float(self.cfg.intercept_z_gate),
            float(self.cfg.intercept_time_max),
            float(self.cfg.intercept_xy_sigma),
            float(self.cfg.on_disk_height_band),
            float(self.cfg.on_disk_xy_sigma),
            float(self.cfg.world_center_hold_sigma),
        )
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.sum(torch.stack(list(rewards.values())), dim=0)


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


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for (w, x, y, z)."""
    return torch.cat((q[:, 0:1], -q[:, 1:4]), dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiply for (w, x, y, z)."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=-1)
