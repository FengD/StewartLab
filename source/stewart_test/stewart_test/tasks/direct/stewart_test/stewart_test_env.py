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
        self._gravity_vec_w = torch.tensor((0.0, 0.0, -1.0), dtype=torch.float, device=self.device).repeat(
            self.num_envs, 1
        )
        self._platform_center_offset_b = torch.tensor(
            self.cfg.platform_center_offset, dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self._compute_intermediate_values()

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "alive",
                "center",
                "center_velocity",
                "height",
                "lin_vel",
                "ang_vel",
                "platform_flat",
                "action",
                "slider_vel",
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
        self._processed_actions = (
            self.cfg.action_smoothing * self._raw_actions + (1.0 - self.cfg.action_smoothing) * self._processed_actions
        )

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
                self._processed_actions,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        rewards = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_center,
            self.cfg.rew_scale_center_velocity,
            self.cfg.rew_scale_height,
            self.cfg.rew_scale_lin_vel,
            self.cfg.rew_scale_ang_vel,
            self.cfg.rew_scale_platform_flat,
            self.cfg.rew_scale_action,
            self.cfg.rew_scale_slider_vel,
            self.cfg.rew_scale_terminated,
            self.object_rel_pos,
            self.object_lin_vel,
            self.object_ang_vel,
            self.platform_projected_gravity,
            self.joint_vel[:, self._slider_dof_idx],
            self._processed_actions,
            self.reset_terminated,
            self.cfg.object_target_rel_height,
        )
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        object_fell = self.object_rel_pos[:, 2] < self.cfg.min_object_rel_height
        object_far = torch.linalg.norm(self.object_rel_pos[:, :2], dim=1) > self.cfg.max_object_xy_dist
        platform_tilted = torch.linalg.norm(self.platform_projected_gravity[:, :2], dim=1) > self.cfg.max_platform_tilt
        return object_fell | object_far | platform_tilted, time_out

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

        self._compute_intermediate_values()

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


def compute_rewards(
    rew_scale_alive: float,
    rew_scale_center: float,
    rew_scale_center_velocity: float,
    rew_scale_height: float,
    rew_scale_lin_vel: float,
    rew_scale_ang_vel: float,
    rew_scale_platform_flat: float,
    rew_scale_action: float,
    rew_scale_slider_vel: float,
    rew_scale_terminated: float,
    object_rel_pos: torch.Tensor,
    object_lin_vel: torch.Tensor,
    object_ang_vel: torch.Tensor,
    platform_projected_gravity: torch.Tensor,
    slider_vel: torch.Tensor,
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    object_target_rel_height: float,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    center_error = torch.linalg.norm(object_rel_pos[:, :2], dim=1)
    height_error = torch.abs(object_rel_pos[:, 2] - object_target_rel_height)
    object_lin_speed = torch.linalg.norm(object_lin_vel, dim=1)
    object_ang_speed = torch.linalg.norm(object_ang_vel, dim=1)
    platform_flat_error = torch.sum(torch.square(platform_projected_gravity[:, :2]), dim=1)
    near_platform = 1.0 - torch.tanh(torch.clamp(object_rel_pos[:, 2] - object_target_rel_height, min=0.0) / 0.45)
    radial_velocity = torch.sum(object_rel_pos[:, :2] * object_lin_vel[:, :2], dim=1) / (center_error + 1.0e-6)
    rew_center = rew_scale_center * (0.3 + 0.7 * near_platform) * (1.0 - torch.tanh(center_error / 0.10))
    rew_center_velocity = rew_scale_center_velocity * near_platform * torch.tanh(torch.clamp(-radial_velocity, -2.0, 2.0))
    rew_height = rew_scale_height * near_platform * (1.0 - torch.tanh(height_error / 0.18))
    rew_lin_vel = rew_scale_lin_vel * near_platform * (1.0 - torch.tanh(object_lin_speed / 1.2))
    rew_ang_vel = rew_scale_ang_vel * near_platform * (1.0 - torch.tanh(object_ang_speed / 5.0))
    rew_platform_flat = rew_scale_platform_flat * platform_flat_error
    rew_action = rew_scale_action * torch.sum(torch.square(actions), dim=1)
    rew_slider_vel = rew_scale_slider_vel * torch.sum(torch.square(slider_vel), dim=1)
    return {
        "alive": rew_alive,
        "center": rew_center,
        "center_velocity": rew_center_velocity,
        "height": rew_height,
        "lin_vel": rew_lin_vel,
        "ang_vel": rew_ang_vel,
        "platform_flat": rew_platform_flat,
        "action": rew_action,
        "slider_vel": rew_slider_vel,
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
