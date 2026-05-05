# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .stewart_test_env_cfg import StewartTestEnvCfg


class StewartTestEnv(DirectRLEnv):
    cfg: StewartTestEnvCfg

    def __init__(self, cfg: StewartTestEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._slider_dof_idx = []
        for joint_name in self.cfg.slider_joint_names:
            joint_ids, _ = self.robot.find_joints(joint_name)
            if len(joint_ids) != 1:
                raise RuntimeError(f"Expected one joint matching '{joint_name}', found {len(joint_ids)}.")
            self._slider_dof_idx.append(joint_ids[0])

        platform_body_ids, _ = self.robot.find_bodies(self.cfg.platform_body_name)
        if len(platform_body_ids) != 1:
            raise RuntimeError(
                f"Expected one platform body matching '{self.cfg.platform_body_name}', found {len(platform_body_ids)}."
            )
        self._platform_body_idx = platform_body_ids[0]

        joint_pos_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self._slider_lower_limits = joint_pos_limits[:, self._slider_dof_idx, 0]
        self._slider_upper_limits = joint_pos_limits[:, self._slider_dof_idx, 1]
        self._slider_neutral_pos = self.robot.data.joint_pos[:, self._slider_dof_idx].clone()
        fallback_lower = self._slider_neutral_pos - self.cfg.fallback_slider_limit
        fallback_upper = self._slider_neutral_pos + self.cfg.fallback_slider_limit
        finite_limits = torch.isfinite(self._slider_lower_limits) & torch.isfinite(self._slider_upper_limits)
        self._slider_lower_limits = torch.where(
            finite_limits, self._slider_lower_limits + self.cfg.finite_limit_margin, fallback_lower
        )
        self._slider_upper_limits = torch.where(
            finite_limits, self._slider_upper_limits - self.cfg.finite_limit_margin, fallback_upper
        )

        self._raw_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._prev_processed_actions = torch.zeros_like(self._raw_actions)
        self._gravity_vec_w = torch.tensor((0.0, 0.0, -1.0), dtype=torch.float, device=self.device).repeat(
            self.num_envs, 1
        )
        self._platform_center_offset_b = torch.tensor(
            self.cfg.platform_center_offset, dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self._compute_intermediate_values()

        # Relative platform-center offset (root body frame) captured at reset — used to regularize stable pose.
        self._nominal_disk_offset_root_b = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # World-frame top disk center at reset — encourage holding a fixed inertial target under wave (wave cfg).
        self._nominal_platform_center_w = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self._defer_nominal_capture = False

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "alive",
                "center",
                "center_velocity",
                "height",
                "lin_vel",
                "ang_vel",
                "intercept",
                "on_disk",
                "world_center_hold",
                "platform_flat",
                "action",
                "action_rate",
                "slider_vel",
                "catch",
                "platform_ang_vel",
                "platform_lin_residual",
                "disk_pose_reg",
                "termination",
            ]
        }

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["ellipsoid"] = self.object
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions.clone().clamp(-1.0, 1.0)
        self._prev_processed_actions = self._processed_actions.clone()
        processed = (
            self.cfg.action_smoothing * self._raw_actions + (1.0 - self.cfg.action_smoothing) * self._processed_actions
        )
        # Bandwidth limit: clamp per-step delta to avoid high-frequency jitter.
        max_delta = getattr(self.cfg, "max_action_delta", None)
        if (
            getattr(self.cfg, "adaptive_action_delta", False)
            and getattr(self, "object_rel_pos", None) is not None
            and getattr(self.cfg, "max_action_delta_high", 0.0) > 0.0
            and getattr(self.cfg, "max_action_delta_low", 0.0) > 0.0
        ):
            # Use relative height (w.r.t. platform center) to switch catching vs stabilization bandwidth.
            rel_h = self.object_rel_pos[:, 2] - float(self.cfg.object_target_rel_height)
            is_high = rel_h > float(self.cfg.adaptive_action_delta_switch_height)
            max_delta = torch.where(
                is_high,
                torch.full((self.num_envs,), float(self.cfg.max_action_delta_high), device=self.device),
                torch.full((self.num_envs,), float(self.cfg.max_action_delta_low), device=self.device),
            ).unsqueeze(-1)
        if max_delta is not None:
            delta = processed - self._prev_processed_actions
            if isinstance(max_delta, torch.Tensor):
                delta = torch.clamp(delta, -max_delta, max_delta)
            else:
                if max_delta > 0.0:
                    delta = torch.clamp(delta, -max_delta, max_delta)
            processed = self._prev_processed_actions + delta
        self._processed_actions = processed

    def _apply_action(self) -> None:
        effort = self.cfg.action_scale * self._processed_actions
        self.robot.set_joint_effort_target(effort, joint_ids=self._slider_dof_idx)

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        obs = torch.cat(
            (
                self.joint_pos[:, self._slider_dof_idx],
                self.joint_vel[:, self._slider_dof_idx],
                self.platform_projected_gravity,
                self.platform_ang_vel,
                self.object_rel_pos,
                # Full ball linear velocity (world) for intercept / deployment with vision or LiDAR fusion.
                self.object_lin_vel,
                self._processed_actions,
                self._get_curriculum_progress_tensor(),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        disk_off_b = quat_rotate_inverse(self.robot.data.root_quat_w, self.platform_center_pos - self.robot.data.root_pos_w)
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
            self.platform_projected_gravity,
            self.platform_ang_vel,
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

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Failure when object hits the ground plane (z=0 in world).
        # Using world-z avoids accidental early termination from an inaccurate platform reference offset.
        object_hit_ground = self.object_pos[:, 2] < (self.cfg.ground_height + self.cfg.object_termination_radius)
        object_far = torch.linalg.norm(self.object_rel_pos[:, :2], dim=1) > self.cfg.max_object_xy_dist
        platform_tilted = torch.linalg.norm(self.platform_projected_gravity[:, :2], dim=1) > self.cfg.max_platform_tilt
        return object_hit_ground | object_far | platform_tilted, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        extras = {}
        for key, value in self._episode_sums.items():
            extras[f"Episode_Reward/{key}"] = torch.mean(value[env_ids]) / self.max_episode_length_s
            value[env_ids] = 0.0
        extras["Episode_Termination/failure"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"] = extras

        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._prev_processed_actions[env_ids] = 0.0

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        if self.cfg.reset_robot_joint_state:
            self.joint_pos[env_ids] = joint_pos
            self.joint_vel[env_ids] = joint_vel
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        zero_effort = torch.zeros((len(env_ids), len(self._slider_dof_idx)), device=self.device)
        self.robot.set_joint_effort_target(zero_effort, joint_ids=self._slider_dof_idx, env_ids=env_ids)
        self._compute_intermediate_values()

        object_state = self.object.data.default_root_state[env_ids].clone()
        env_progress = self._get_curriculum_progress_tensor(env_ids).squeeze(-1)
        object_spawn_radius = self._lerp_cfg_tensor(
            self.cfg.object_spawn_radius_start, self.cfg.object_spawn_radius, env_progress
        )
        object_height_low = self._lerp_cfg_tensor(
            self.cfg.object_drop_height_range_start[0], self.cfg.object_drop_height_range[0], env_progress
        )
        object_height_high = self._lerp_cfg_tensor(
            self.cfg.object_drop_height_range_start[1], self.cfg.object_drop_height_range[1], env_progress
        )
        object_initial_down_velocity = self._lerp_cfg_tensor(
            self.cfg.object_initial_down_velocity_start, self.cfg.object_initial_down_velocity, env_progress
        )
        object_spin_velocity = self._lerp_cfg_tensor(
            self.cfg.object_spin_velocity_start, self.cfg.object_spin_velocity_end, env_progress
        )

        rand_radius = object_spawn_radius * torch.sqrt(torch.rand(len(env_ids), device=self.device))
        rand_angle = sample_uniform(0.0, 6.28318530718, (len(env_ids),), self.device)
        object_xy = torch.stack((rand_radius * torch.cos(rand_angle), rand_radius * torch.sin(rand_angle)), dim=-1)
        object_height = object_height_low + (object_height_high - object_height_low) * torch.rand(
            len(env_ids), device=self.device
        )
        object_state[:, 0:2] = self.platform_center_pos[env_ids, 0:2] + object_xy
        object_state[:, 2] = self.platform_center_pos[env_ids, 2] + object_height
        object_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float, device=self.device)
        object_state[:, 7:] = 0.0
        object_state[:, 9] = object_initial_down_velocity
        object_state[:, 10:13] = (2.0 * torch.rand((len(env_ids), 3), device=self.device) - 1.0) * (
            object_spin_velocity.unsqueeze(-1)
        )
        self.object.write_root_pose_to_sim(object_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids)

        self._compute_intermediate_values()
        if not self._defer_nominal_capture:
            self._capture_nominal_disk_offset(env_ids)

    def _capture_nominal_disk_offset(self, env_ids: Sequence[int] | torch.Tensor):
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._compute_intermediate_values()
        root_q = self.robot.data.root_quat_w[env_ids_t]
        root_p = self.robot.data.root_pos_w[env_ids_t]
        diff = self.platform_center_pos[env_ids_t] - root_p
        self._nominal_disk_offset_root_b[env_ids_t] = quat_rotate_inverse(root_q, diff)
        self._nominal_platform_center_w[env_ids_t] = self.platform_center_pos[env_ids_t].clone()

    def _get_curriculum_progress(self) -> float:
        if not getattr(self.cfg, "enable_curriculum", False):
            return 1.0
        duration = max(float(getattr(self.cfg, "curriculum_duration_steps", 1)), 1.0)
        step = float(getattr(self, "common_step_counter", 0))
        return max(0.0, min(step / duration, 1.0))

    def _get_curriculum_progress_tensor(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
        progress = self._get_curriculum_progress()
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, dtype=torch.float, device=self.device)
            count = self.num_envs
        else:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.float, device=self.device)
            count = len(env_ids_t)

        if not getattr(self.cfg, "curriculum_per_env", False) or self.num_envs <= 1:
            return torch.full((count, 1), progress, dtype=torch.float, device=self.device)

        env_fraction = env_ids_t / max(float(self.num_envs - 1), 1.0)
        spread = max(float(getattr(self.cfg, "curriculum_env_progress_spread", 0.0)), 0.0)
        window = spread * (1.0 - progress)
        env_progress = torch.clamp(progress - window + env_fraction * window, 0.0, 1.0)
        return env_progress.unsqueeze(-1)

    @staticmethod
    def _lerp_cfg_value(start: float, end: float, progress: float) -> float:
        return float(start) + (float(end) - float(start)) * progress

    @staticmethod
    def _lerp_cfg_tensor(start: float, end: float, progress: torch.Tensor) -> torch.Tensor:
        return float(start) + (float(end) - float(start)) * progress

    @classmethod
    def _lerp_cfg_pair(
        cls, start: tuple[float, float], end: tuple[float, float], progress: float
    ) -> tuple[float, float]:
        return (
            cls._lerp_cfg_value(start[0], end[0], progress),
            cls._lerp_cfg_value(start[1], end[1], progress),
        )

    @classmethod
    def _lerp_cfg_triple(
        cls, start: tuple[float, float, float], end: tuple[float, float, float], progress: float
    ) -> tuple[float, float, float]:
        return (
            cls._lerp_cfg_value(start[0], end[0], progress),
            cls._lerp_cfg_value(start[1], end[1], progress),
            cls._lerp_cfg_value(start[2], end[2], progress),
        )

    def _compute_intermediate_values(self):
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.platform_pos = self.robot.data.body_pos_w[:, self._platform_body_idx]
        self.platform_quat = self.robot.data.body_quat_w[:, self._platform_body_idx]
        self.platform_ang_vel = self.robot.data.body_ang_vel_w[:, self._platform_body_idx]
        self.platform_projected_gravity = quat_rotate_inverse(self.platform_quat, self._gravity_vec_w)
        self.platform_center_pos = self.platform_pos + quat_rotate(self.platform_quat, self._platform_center_offset_b)
        self.object_pos = self.object.data.root_pos_w
        self.object_rel_pos = self.object_pos - self.platform_center_pos
        self.object_lin_vel = self.object.data.root_lin_vel_w
        self.object_ang_vel = self.object.data.root_ang_vel_w
        self.robot_root_lin_vel_w = self.robot.data.root_lin_vel_w
        self.platform_lin_vel_w = self.robot.data.body_lin_vel_w[:, self._platform_body_idx]


def compute_rewards(
    rew_scale_alive: float,
    rew_scale_center: float,
    rew_scale_center_velocity: float,
    rew_scale_height: float,
    rew_scale_lin_vel: float,
    rew_scale_ang_vel: float,
    rew_scale_intercept: float,
    rew_scale_on_disk: float,
    rew_scale_world_center_hold: float,
    rew_scale_platform_flat: float,
    rew_scale_action: float,
    rew_scale_action_rate: float,
    rew_scale_action_rate_on_disk: float,
    rew_scale_slider_vel: float,
    rew_scale_catch: float,
    rew_scale_platform_ang_vel: float,
    rew_scale_platform_lin_residual: float,
    rew_scale_disk_pose_reg: float,
    rew_scale_terminated: float,
    object_rel_pos: torch.Tensor,
    object_lin_vel: torch.Tensor,
    object_ang_vel: torch.Tensor,
    platform_center_pos_w: torch.Tensor,
    nominal_platform_center_w: torch.Tensor,
    platform_projected_gravity: torch.Tensor,
    platform_ang_vel: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    disk_offset_root_b: torch.Tensor,
    nominal_disk_offset_root_b: torch.Tensor,
    platform_lin_vel_w: torch.Tensor,
    slider_vel: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    object_target_rel_height: float,
    intercept_z_gate: float,
    intercept_time_max: float,
    intercept_xy_sigma: float,
    on_disk_height_band: float,
    on_disk_xy_sigma: float,
    world_center_hold_sigma: float,
):

    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    center_error = torch.linalg.norm(object_rel_pos[:, :2], dim=1)
    height_error = torch.abs(object_rel_pos[:, 2] - object_target_rel_height)
    object_rel_lin_vel = object_lin_vel - platform_lin_vel_w
    object_lin_speed = torch.linalg.norm(object_rel_lin_vel, dim=1)
    object_ang_speed = torch.linalg.norm(object_ang_vel, dim=1)
    platform_flat_error = torch.sum(torch.square(platform_projected_gravity[:, :2]), dim=1)
    near_platform = 1.0 - torch.tanh(torch.clamp(object_rel_pos[:, 2] - object_target_rel_height, min=0.0) / 0.45)
    # 接球区域权重（球越接近平台越大）
    catch_zone = torch.exp(-torch.square(object_rel_pos[:, 2] - object_target_rel_height) / 0.15)

    # 缓冲奖励（鼓励降低球速度）
    soft_catch = catch_zone * (1.0 - torch.tanh(torch.abs(object_rel_lin_vel[:, 2]) / 1.0))
    rew_catch = rew_scale_catch * soft_catch
    radial_velocity = torch.sum(object_rel_pos[:, :2] * object_rel_lin_vel[:, :2], dim=1) / (center_error + 1.0e-6)
    rew_center = rew_scale_center * (0.3 + 0.7 * near_platform) * (1.0 - torch.tanh(center_error / 0.10))
    rew_center_velocity = rew_scale_center_velocity * near_platform * torch.tanh(torch.clamp(-radial_velocity, -2.0, 2.0))
    rew_height = rew_scale_height * near_platform * (1.0 - torch.tanh(height_error / 0.18))
    rew_lin_vel = rew_scale_lin_vel * near_platform * (1.0 - torch.tanh(object_lin_speed / 1.2))
    rew_ang_vel = rew_scale_ang_vel * near_platform * (1.0 - torch.tanh(object_ang_speed / 5.0))

    # Interception shaping: while ball is still high and descending, predict where it will be at target height
    # and reward reducing that predicted xy error (encourages large-range pre-positioning).
    z_above = object_rel_pos[:, 2] - object_target_rel_height
    descending = object_rel_lin_vel[:, 2] < -0.05
    high_enough = z_above > intercept_z_gate
    dz = torch.clamp(z_above, min=0.0)
    vz_down = torch.clamp(-object_rel_lin_vel[:, 2], min=0.2)
    t_hit = torch.clamp(dz / vz_down, min=0.0, max=intercept_time_max)
    pred_xy = object_rel_pos[:, :2] + object_rel_lin_vel[:, :2] * t_hit.unsqueeze(-1)
    pred_xy_err = torch.linalg.norm(pred_xy, dim=1)
    rew_intercept = rew_scale_intercept * (descending & high_enough).float() * torch.exp(
        -torch.square(pred_xy_err) / (2.0 * intercept_xy_sigma**2)
    )
    # "On disk" persistence: smooth gates so short episodes still get learning signal toward staying settled.
    # Height: ~exp(-1) when |rel_height_err| equals `on_disk_height_band` (treat band as characteristic scale).
    on_disk_height_w = torch.exp(-torch.square(height_error / (on_disk_height_band + 1.0e-6)))
    on_disk = on_disk_height_w * torch.exp(-torch.square(center_error) / (2.0 * on_disk_xy_sigma**2))
    rew_on_disk = rew_scale_on_disk * on_disk
    # World XY hold: penalize drifting the top disk center in world frame vs reset snapshot (wave: use sliders to
    # cancel base motion instead of letting the whole mechanism track the disturbance).
    hold_xy_err_sq = torch.sum(
        torch.square(platform_center_pos_w[:, :2] - nominal_platform_center_w[:, :2]),
        dim=1,
    )
    hold_gate = torch.square(near_platform) * on_disk
    hold_norm = 2.0 * world_center_hold_sigma**2 + 1.0e-8
    rew_world_center_hold = rew_scale_world_center_hold * hold_gate * (hold_xy_err_sq / hold_norm)
    # Keep the disk as stable as possible even before contact, but allow more freedom when the ball is far.
    stability_gate = 0.20 + 0.80 * near_platform
    rew_platform_flat = rew_scale_platform_flat * stability_gate * platform_flat_error
    # Gate penalties so the policy can move aggressively during the catching phase (object still high),
    # but becomes smooth/energy-efficient near the platform.
    penalty_gate = 0.05 + 0.95 * near_platform
    rew_action = rew_scale_action * penalty_gate * torch.sum(torch.square(actions), dim=1)
    action_rate_sq = torch.sum(torch.square(actions - prev_actions), dim=1)
    rew_action_rate = (
        rew_scale_action_rate * penalty_gate * action_rate_sq
        + rew_scale_action_rate_on_disk * on_disk * action_rate_sq
    )
    rew_slider_vel = rew_scale_slider_vel * penalty_gate * torch.sum(torch.square(slider_vel), dim=1)
    platform_ang_speed = torch.linalg.norm(platform_ang_vel, dim=1)
    rew_platform_ang_vel = rew_scale_platform_ang_vel * stability_gate * platform_ang_speed

    # Disk linear jitter vs articulation root (world). Gate with near^2 so catching can use large relative motion.
    lin_res_sq = torch.sum(torch.square(platform_lin_vel_w - root_lin_vel_w), dim=1)
    stable_gate = torch.square(near_platform)
    rew_platform_lin_residual = rew_scale_platform_lin_residual * stable_gate * lin_res_sq

    # Keep disk-relative geometry near nominal (root frame), gated when stabilizing the ball.
    pose_err_sq = torch.sum(torch.square(disk_offset_root_b - nominal_disk_offset_root_b), dim=1)
    rew_disk_pose_reg = rew_scale_disk_pose_reg * stable_gate * pose_err_sq

    return {
        "alive": rew_alive,
        "center": rew_center,
        "center_velocity": rew_center_velocity,
        "height": rew_height,
        "lin_vel": rew_lin_vel,
        "ang_vel": rew_ang_vel,
        "intercept": rew_intercept,
        "on_disk": rew_on_disk,
        "world_center_hold": rew_world_center_hold,
        "platform_flat": rew_platform_flat,
        "action": rew_action,
        "action_rate": rew_action_rate,
        "slider_vel": rew_slider_vel,
        "catch": rew_catch,
        "platform_ang_vel": rew_platform_ang_vel,
        "platform_lin_residual": rew_platform_lin_residual,
        "disk_pose_reg": rew_disk_pose_reg,
        "termination": rew_termination,
    }



def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c
