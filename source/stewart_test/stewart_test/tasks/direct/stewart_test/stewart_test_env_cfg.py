# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
import os
from pathlib import Path

import trimesh

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass


PROJECT_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_STEWART_USD_PATH = PROJECT_ROOT / "assets" / "Stewart_full.usd"
STEWART_USD_PATH = os.environ.get("STEWART_USD_PATH", str(DEFAULT_STEWART_USD_PATH))
SLIDER_JOINT_NAMES = ["Slider_13", "Slider_18", "Slider_17", "Slider_14", "Slider_16", "Slider_15"]
# Keep some upward travel margin so the policy can "push up" to stabilize,
# while still leaving room to move down for impact buffering.
SLIDER_INITIAL_EXTENSION = 0.12


@clone
def spawn_ellipsoid(
    prim_path: str,
    cfg: "EllipsoidCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a rigid ellipsoid by scaling sphere mesh vertices before USD creation."""
    del kwargs
    mesh = trimesh.creation.uv_sphere(radius=1.0)
    mesh.vertices *= cfg.semi_axes
    stage = get_current_stage()
    _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation, stage=stage)
    return stage.GetPrimAtPath(prim_path)


@configclass
class EllipsoidCfg(sim_utils.MeshSphereCfg):
    func: Callable = spawn_ellipsoid
    semi_axes: tuple[float, float, float] = (0.03, 0.03, 0.03)


@configclass
class StewartTestEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 16.0
    # - spaces definition
    action_space = 6
    # [6 q, 6 qd, 3 g_proj, 3 w, 3 rel_pos_ball, 3 ball_lin_vel_w, 6 prev_action, 1 curriculum] = 31
    observation_space = 31
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(bounce_threshold_velocity=0.2),
    )

    # robot(s)
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=STEWART_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={joint_name: SLIDER_INITIAL_EXTENSION for joint_name in SLIDER_JOINT_NAMES},
        ),
        actuators={
            "sliders": IdealPDActuatorCfg(
                joint_names_expr=["Slider_13", "Slider_14", "Slider_15", "Slider_16", "Slider_17", "Slider_18"],
                effort_limit_sim=8000.0,
                velocity_limit_sim=2.0,
                stiffness=50000.0,
                damping=5000.0,
            ),
        },
    )

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ellipsoid",
        spawn=EllipsoidCfg(
            semi_axes=(0.03, 0.03, 0.03),
            radius=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.72, 0.18)),
            # Use realistic restitution (e.g. wooden ball ~0.3-0.4). Stability is handled via control/reward shaping.
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6, restitution=0.35),
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
            mass_props=sim_utils.MassPropertiesCfg(mass=4.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 4.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=3.0, replicate_physics=True)

    # custom parameters/scales
    slider_joint_names = SLIDER_JOINT_NAMES
    platform_body_name = "UJ61"
    # NOTE: `platform_center_pos` is used as the reference for object spawn, observations and rewards.
    # This offset should point to the center of the *top surface* of the disk when expressed in the `platform_body_name`
    # body frame (here `UJ61`).
    platform_center_offset = (-0.054, 0.217, 0.2)

    action_scale = 6000.0  # [N] direct slider force command
    # Slightly stronger low-pass to suppress high-frequency jitter.
    action_smoothing = 0.12
    # Hard bandwidth limit: clamp per-step action change (after smoothing).
    # This directly prevents high-frequency small-amplitude oscillations that waste energy.
    max_action_delta = 0.09
    # Optional height-adaptive bandwidth: allow larger changes when the object is still high (catching phase),
    # and tighten the limit near the target height (stabilization phase).
    adaptive_action_delta = True
    max_action_delta_high = 0.18
    max_action_delta_low = 0.06
    adaptive_action_delta_switch_height = 0.25  # [m] above target height
    finite_limit_margin = 0.005
    fallback_slider_limit = 0.08
    reset_robot_joint_state = False

    # Object spawn is sampled on a disk around the platform center. When curriculum is enabled,
    # these values are the final task difficulty.
    object_spawn_radius = 0.03
    object_drop_height_range = (1.0, 1.5)
    object_initial_down_velocity = -0.20
    # Raise the stabilized height so the sliders sit more extended, leaving
    # room to move down for impact buffering.
    object_target_rel_height = 0.08

    # Termination: treat "object hit the ground" as failure.
    # Ground plane is spawned at z = 0.0 (world).
    ground_height = 0.0
    # Approx object radius for termination check (ellipsoid uses max semi-axis).
    object_termination_radius = 0.03

    # - curriculum
    enable_curriculum = True
    curriculum_duration_steps = 80000
    # Distribute parallel envs over a trailing difficulty window. The window narrows to zero near the end,
    # so final training concentrates on the hardest configured task.
    curriculum_per_env = True
    curriculum_env_progress_spread = 0.35
    object_spawn_radius_start = 0.005
    object_drop_height_range_start = (0.45, 0.75)
    object_initial_down_velocity_start = -0.05
    object_spin_velocity_start = 0.0
    object_spin_velocity_end = 1.5

    # - reward scales
    rew_scale_alive = 0.15
    rew_scale_center = 16.0
    rew_scale_center_velocity = 4.0
    rew_scale_height = 1.5
    rew_scale_lin_vel = 2.5
    rew_scale_ang_vel = 1.0
    rew_scale_platform_flat = -0.35
    rew_scale_action = -0.001
    # Penalize rapid command changes to reduce jitter.
    rew_scale_action_rate = -0.03
    rew_scale_slider_vel = -0.0005
    rew_scale_catch = 4.0
    # Predictive shaping while the object is still high and descending.
    rew_scale_intercept = 2.0
    rew_scale_platform_ang_vel = -0.15
    # Penalize top-plate linear jitter relative to articulation root (world frame), gated when ball is near target.
    rew_scale_platform_lin_residual = -0.12
    # Pull disk geometry back toward nominal relative pose to root (root body frame), gated when stabilizing.
    rew_scale_disk_pose_reg = -0.35
    # Fixed-base task keeps the wave-specific post-catch hold terms disabled by default.
    rew_scale_on_disk = 10.0
    rew_scale_world_center_hold = 0.0
    rew_scale_action_rate_on_disk = 0.0
    rew_scale_terminated = -5.0

    # - reward shaping widths / gates
    intercept_z_gate = 0.25
    intercept_time_max = 1.5
    intercept_xy_sigma = 0.14
    on_disk_height_band = 0.14
    on_disk_xy_sigma = 0.10
    world_center_hold_sigma = 0.08

    # - reset states/conditions
    max_object_xy_dist = 0.45
    max_platform_tilt = 0.65
