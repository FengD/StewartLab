# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from ..stewart_test.stewart_test_env_cfg import StewartTestEnvCfg


@configclass
class StewartWaveSystemEnvCfg(StewartTestEnvCfg):
    # Add wave/root motion to observations so the policy can do feed-forward compensation.
    # Base obs = 31; append [root_projected_gravity(3), wave_pose(6), wave_velocity(6)] => 46.
    observation_space = 46

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.72, 0.18)),
            # Use realistic restitution (e.g. wooden ball ~0.3-0.4).
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8, restitution=0.35),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    wave_base_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/WaveBase",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.8, 0.08),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.25, 0.32)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.06), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    system_z_offset = 0.14
    wave_base_z_offset = -0.06
    # Start wave training with milder disturbances; increase these again once the policy reliably catches
    # and stabilizes the round ball, or replace them with a curriculum that ramps toward the old values.
    wave_pos_amplitude = (0.04, 0.04, 0.02)
    wave_rot_amplitude = (0.06, 0.06, 0.09)
    wave_frequency_range = (0.25, 0.65)
    wave_pos_amplitude_start = (0.0, 0.0, 0.0)
    wave_rot_amplitude_start = (0.0, 0.0, 0.0)
    wave_frequency_range_start = (0.10, 0.25)
    # Axis curriculum order: x, y, z, roll, pitch, yaw. Each axis ramps after its start progress.
    wave_axis_start_progress = (0.15, 0.25, 0.35, 0.55, 0.70, 0.85)
    wave_axis_ramp_progress = 0.15

    object_spawn_radius = 0.04
    object_drop_height_range = (2.0, 3.0)
    object_initial_down_velocity = -0.20
    object_spawn_radius_start = 0.005
    object_drop_height_range_start = (0.60, 1.00)
    object_initial_down_velocity_start = -0.05
    object_spin_velocity_start = 0.0
    object_spin_velocity_end = 3.0
    max_object_xy_dist = 0.52

    # With a moving wave base, `object_far` on world-frame rel_xy fires falsely while the ball is still high
    # (plate moves sideways; inertia mismatch inflates horizontal rel distance). Gate by vertical separation.
    # Rel z is ball minus platform-center ref; typical fall starts at ~2–3 m → gate off until closer to intercept.
    object_far_rel_height_gate = 0.55

    # Sphere radius for ground-hit termination check.
    object_termination_radius = 0.05

    # For higher drops, we need more time and a more "prepared" posture.
    object_target_rel_height = 0.10

    # Residual tilt budget: overly tight limits terminate healthy rollouts (~100 steps vs ~960 step horizon).
    max_platform_tilt = 0.78

    # Dense survival shaping for long horizons (helps PPO prioritize not dying early).
    rew_scale_alive = 0.45

    # Wave task needs stronger "top plate stabilization" to reject base disturbances after catch.
    # Note: wave reward uses residual (w.r.t. wave base) versions of flatness and angular velocity.
    rew_scale_platform_flat = -0.65
    rew_scale_platform_ang_vel = -0.22
    rew_scale_platform_lin_residual = -0.22
    rew_scale_disk_pose_reg = -0.42

    # After catch: reward keeping the ball centered on disk; penalize world-XY drift of top plate vs reset snapshot
    # (encourages compensating wave motion with sliders instead of letting the whole stack "surf" the disturbance).
    rew_scale_on_disk = 12.0
    on_disk_height_band = 0.14
    on_disk_xy_sigma = 0.10
    rew_scale_world_center_hold = -0.55
    world_center_hold_sigma = 0.058
    # Stronger action-rate penalty only in the on-disk phase to kill high-frequency command chatter.
    rew_scale_action_rate_on_disk = -0.028
