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
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.05,
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
    wave_pos_amplitude = (0.08, 0.08, 0.04)
    wave_rot_amplitude = (0.12, 0.12, 0.18)
    wave_frequency_range = (0.35, 0.90)

    object_spawn_radius = 0.04
    object_drop_height_range = (2.0, 3.0)
    object_initial_down_velocity = -0.20
    max_object_xy_dist = 0.40
