#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to load USD model and view joint information

This script is used for:
1. Load specified USD model file
2. Display model joint information (joint name, type, limits, etc.)
3. Visualize model structure

Usage:
    python scripts/load_usd_model.py --usd_path <path_to_usd_file>
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Add command line arguments
parser = argparse.ArgumentParser(description="Load USD model and view joint information")
parser.add_argument(
    "--usd_path",
    type=str,
    default=None,
    help="Path to USD model file (relative to assets directory or absolute path)",
)
parser.add_argument(
    "--headless",
    action="store_true",
    help="Run in headless mode (no GUI)",
)
# # Add AppLauncher arguments
# AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# If USD path is not specified, use default prompt
if args_cli.usd_path is None:
    print("[WARNING] USD model path not specified, please use --usd_path parameter to specify model path")
    print("[INFO] Example: python scripts/load_usd_model.py --usd_path assets/your_model.usd")
    sys.exit(0)

# Start Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the code executes after simulator starts"""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# import omni.isaac.core.utils.stage as stage_utils
from pxr import Usd, UsdGeom, UsdPhysics


def get_usd_joint_info(usd_path: str):
    """
    Read joint information directly from USD file
    
    Args:
        usd_path: Path to USD file
        
    Returns:
        List of joint information
    """
    from pxr import Usd, UsdPhysics, PhysicsSchemaTools
    
    # Get absolute path
    if not os.path.isabs(usd_path):
        # Check if it's a Nucleus path
        if usd_path.startswith("http://") or usd_path.startswith("https://"):
            pass
        else:
            usd_path = os.path.abspath(usd_path)
    
    print(f"\n{'='*60}")
    print(f"[INFO] Analyzing USD file: {usd_path}")
    print(f"{'='*60}\n")
    
    # Open USD stage
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"[ERROR] Cannot open USD file: {usd_path}")
        return []
    
    joint_info_list = []
    
    # Traverse all Prims
    for prim in stage.Traverse():

        if prim.GetTypeName() == "PhysicsFixedJoint":
            joint = UsdPhysics.FixedJoint(prim)
            attr = prim.GetAttribute("physics:excludeFromArticulation")
            if attr:
                print(prim.GetPath(), attr.Get())

        prim_type = prim.GetTypeName()
        
        # Check if it's a joint type
        if prim_type in ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint", 
                         "PhysicsSphericalJoint", "PhysicsFixedJoint",
                         "RevoluteJoint", "PrismaticJoint", "SphericalJoint", "FixedJoint"]:
            
            joint_info = {
                "path": str(prim.GetPath()),
                "type": prim_type,
                "name": prim.GetPath().name,
            }
            
            # Get joint axis information
            if prim.HasAttribute("axis"):
                joint_info["axis"] = prim.GetAttribute("axis").Get()
            
            # Get joint limits
            if prim.HasAttribute("lower_limit"):
                joint_info["lower_limit"] = prim.GetAttribute("lower_limit").Get()
            if prim.HasAttribute("upper_limit"):
                joint_info["upper_limit"] = prim.GetAttribute("upper_limit").Get()
            
            # Get drive information
            if prim.HasAttribute("drive_type"):
                joint_info["drive_type"] = prim.GetAttribute("drive_type").Get()
            if prim.HasAttribute("target_position"):
                joint_info["target_position"] = prim.GetAttribute("target_position").Get()
            if prim.HasAttribute("target_velocity"):
                joint_info["target_velocity"] = prim.GetAttribute("target_velocity").Get()
            
            # Get rigid body information
            body0 = prim.GetRelationship("physics:body0").GetTargets() if prim.HasRelationship("physics:body0") else []
            body1 = prim.GetRelationship("physics:body1").GetTargets() if prim.HasRelationship("physics:body1") else []
            joint_info["body0"] = str(body0[0]) if body0 else "None"
            joint_info["body1"] = str(body1[0]) if body1 else "None"
            
            joint_info_list.append(joint_info)
    
    return joint_info_list


def print_joint_info_from_articulation(robot: Articulation):
    """
    Print joint information from Articulation object
    
    Args:
        robot: Articulation object
    """
    print(f"\n{'='*60}")
    print("[INFO] Articulation Joint Information")
    print(f"{'='*60}\n")
    
    # Get number of joints
    num_joints = robot.num_joints
    print(f"Total number of joints: {num_joints}")
    
    # Get joint names
    joint_names = robot.joint_names
    print(f"\nJoint name list:")
    for i, name in enumerate(joint_names):
        print(f"  [{i}] {name}")
    
    # Get joint limits
    print(f"\nJoint limits (position):")
    for i, name in enumerate(joint_names):
        try:
            lower = robot.get_joint_position_limit(name, "lower")
            upper = robot.get_joint_position_limit(name, "upper")
            print(f"  [{i}] {name}: [{lower:.4f}, {upper:.4f}]")
        except Exception as e:
            print(f"  [{i}] {name}: Cannot get limits - {e}")
    
    # Get default joint positions
    print(f"\nDefault joint positions:")
    default_pos = robot.data.default_joint_pos
    for i, name in enumerate(joint_names):
        print(f"  [{i}] {name}: {default_pos[0, i].item():.4f}")
    
    # Get link information
    print(f"\nLink information:")
    body_names = robot.body_names
    for i, name in enumerate(body_names):
        print(f"  [{i}] {name}")


def main():
    """Main function"""
    # Process USD path
    usd_path = args_cli.usd_path
    
    # If relative path, try to find in assets directory
    if not os.path.isabs(usd_path) and not usd_path.startswith("http"):
        # First try current directory
        if os.path.exists(usd_path):
            usd_path = os.path.abspath(usd_path)
        # Then try assets directory
        elif os.path.exists(os.path.join("assets", usd_path)):
            usd_path = os.path.abspath(os.path.join("assets", usd_path))
        # Finally try assets under project root
        elif os.path.exists(os.path.join("stewart_test/assets", usd_path)):
            usd_path = os.path.abspath(os.path.join("stewart_test/assets", usd_path))
    
    print(f"\n{'#'*60}")
    print(f"# Stewart Platform Model Loader")
    print(f"{'#'*60}")
    print(f"\n[INFO] USD model path: {usd_path}")
    
    # Check if file exists
    if not usd_path.startswith("http") and not os.path.exists(usd_path):
        print(f"[ERROR] File does not exist: {usd_path}")
        print(f"[INFO] Please ensure the model file is placed in the correct location")
        simulation_app.close()
        return
    
    # Step 1: Read joint information directly from USD file
    print("\n" + "="*60)
    print("Step 1: Read joint information directly from USD file")
    print("="*60)
    
    joint_info_list = get_usd_joint_info(usd_path)
    
    if joint_info_list:
        print(f"\nFound {len(joint_info_list)} joints:\n")
        for i, joint_info in enumerate(joint_info_list):
            print(f"Joint [{i}]:", joint_info)
            print()
    else:
        print("[WARNING] No joint information found in USD file")
    
    # Step 2: Load model using Isaac Lab
    print("\n" + "="*60)
    print("Step 2: Load model using Isaac Lab Articulation")
    print("="*60)
    
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=1/120.0, render_interval=2)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Add ground plane
    # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    
    # # Add light
    # light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    # light_cfg.func("/World/Light", light_cfg)
    
    # Create robot configuration
    robot_cfg = ArticulationCfg(
        prim_path="/World/StewartPlatform",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            # Default actuator configuration, can be modified as needed
            "sliders": ImplicitActuatorCfg(
                joint_names_expr=["Slider_13", "Slider_14", "Slider_15", "Slider_16", "Slider_17", "Slider_18"],
                effort_limit_sim=5000.0,
                velocity_limit_sim=1.5,
                stiffness=200000.0,
                damping=20000.0,
            ),
        },
    )
    
    # Create Articulation
    try:
        robot = Articulation(cfg=robot_cfg)
        
        # Reset simulation
        sim.reset()
        
        # Print joint information
        print_joint_info_from_articulation(robot)
        
        # Step 3: Run simulation to view model
        print("\n" + "="*60)
        print("Step 3: Simulation running...")
        print("="*60)
        print("[INFO] Press Ctrl+C or close window to exit")
        print("[INFO] Use mouse to rotate, zoom and view model")
        
        # Run simulation
        while simulation_app.is_running():
            # Apply some simple joint movements (if there are driven joints)
            # Control logic can be added here
            sim.step()
            
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        print(f"[INFO] Please check if USD file format is correct")
        import traceback
        traceback.print_exc()
    
    


if __name__ == "__main__":
    main()

    # Close simulation
    simulation_app.close()
