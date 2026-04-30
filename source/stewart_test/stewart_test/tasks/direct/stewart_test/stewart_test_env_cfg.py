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
SLIDER_INITIAL_EXTENSION = 0.10


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
    semi_axes: tuple[float, float, float] = (0.04, 0.07, 0.05)


@configclass
class StewartTestEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 8.0
    # - spaces definition
    action_space = 6
    observation_space = 27
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
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
                velocity_limit_sim=1.0,
                stiffness=50000.0,
                damping=5000.0,
            ),
        },
    )

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ellipsoid",
        spawn=EllipsoidCfg(
            semi_axes=(0.04, 0.07, 0.05),
            radius=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.72, 0.18)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8, restitution=0.0),
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
    platform_center_offset = (0.0, 0.0, 0.0)

    action_scale = 6000.0  # [N] direct slider force command
    action_smoothing = 0.12
    finite_limit_margin = 0.005
    fallback_slider_limit = 0.08
    reset_robot_joint_state = False

    object_spawn_radius = 0.045
    object_drop_height_range = (2.0, 3.0)
    object_initial_down_velocity = -0.20
    object_target_rel_height = 0.10

    # - reward scales
    rew_scale_alive = 0.2
    rew_scale_center = 8.0
    rew_scale_center_velocity = 1.5
    rew_scale_height = 1.2
    rew_scale_lin_vel = 2.0
    rew_scale_ang_vel = 0.8
    rew_scale_platform_flat = -0.35
    rew_scale_action = -0.01
    rew_scale_slider_vel = -0.01
    rew_scale_terminated = -5.0

    # - reset states/conditions
    max_object_xy_dist = 0.45
    min_object_rel_height = -0.10
    max_platform_tilt = 0.65
