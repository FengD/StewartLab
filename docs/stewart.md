############################################################
# Stewart Platform Model Loader
############################################################

[INFO] USD model path: /home/ding/Documents/Stewart-Platform/Stewart_full.usd

============================================================
Step 1: Read joint information directly from USD file
============================================================

============================================================
[INFO] Analyzing USD file: /home/ding/Documents/Stewart-Platform/Stewart_full.usd
============================================================

/World/Stewart/UJ61/J6B1/J6T1/TOP1/J1T1/FixedJoint True
/World/Stewart/UJ61/J6B1/J6T1/TOP1/J2T1/FixedJoint True
/World/Stewart/UJ61/J6B1/J6T1/TOP1/J3T_1/FixedJoint True
/World/Stewart/UJ61/J6B1/J6T1/TOP1/J4T1/FixedJoint True
/World/Stewart/UJ61/J6B1/J6T1/TOP1/J5T_1/FixedJoint True
/World/Stewart/root_joint False

Found 42 joints:

Joint [0]: {'path': '/World/Stewart/joints/Revolute_1', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_1', 'body0': '/World/Stewart/base_link', 'body1': '/World/Stewart/X1bottom1'}

Joint [1]: {'path': '/World/Stewart/joints/Revolute_9', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_9', 'body0': '/World/Stewart/X1bottom1', 'body1': '/World/Stewart/cylinder11'}

Joint [2]: {'path': '/World/Stewart/joints/Slider_13', 'type': 'PhysicsPrismaticJoint', 'name': 'Slider_13', 'body0': '/World/Stewart/cylinder11', 'body1': '/World/Stewart/rod11'}

Joint [3]: {'path': '/World/Stewart/joints/Revolute_19', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_19', 'body0': '/World/Stewart/rod11', 'body1': '/World/Stewart/piston11'}

Joint [4]: {'path': '/World/Stewart/joints/Revolute_26', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_26', 'body0': '/World/Stewart/piston11', 'body1': '/World/Stewart/X1top1'}

Joint [5]: {'path': '/World/Stewart/joints/Revolute_29', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_29', 'body0': '/World/Stewart/X1top1', 'body1': '/World/Stewart/UJ11'}

Joint [6]: {'path': '/World/Stewart/joints/Revolute_2', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_2', 'body0': '/World/Stewart/base_link', 'body1': '/World/Stewart/X6bottom1'}

Joint [7]: {'path': '/World/Stewart/joints/Revolute_7', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_7', 'body0': '/World/Stewart/X6bottom1', 'body1': '/World/Stewart/cylinder61'}

Joint [8]: {'path': '/World/Stewart/joints/Slider_18', 'type': 'PhysicsPrismaticJoint', 'name': 'Slider_18', 'body0': '/World/Stewart/cylinder61', 'body1': '/World/Stewart/rod61'}

Joint [9]: {'path': '/World/Stewart/joints/Revolute_22', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_22', 'body0': '/World/Stewart/rod61', 'body1': '/World/Stewart/piston61'}

Joint [10]: {'path': '/World/Stewart/joints/Revolute_25', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_25', 'body0': '/World/Stewart/piston61', 'body1': '/World/Stewart/X6top1'}

Joint [11]: {'path': '/World/Stewart/joints/Revolute_30', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_30', 'body0': '/World/Stewart/X6top1', 'body1': '/World/Stewart/UJ61'}

Joint [12]: {'path': '/World/Stewart/joints/Revolute_3', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_3', 'body0': '/World/Stewart/base_link', 'body1': '/World/Stewart/X5bottom1'}

Joint [13]: {'path': '/World/Stewart/joints/Revolute_8', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_8', 'body0': '/World/Stewart/X5bottom1', 'body1': '/World/Stewart/cylinder51'}

Joint [14]: {'path': '/World/Stewart/joints/Slider_17', 'type': 'PhysicsPrismaticJoint', 'name': 'Slider_17', 'body0': '/World/Stewart/cylinder51', 'body1': '/World/Stewart/rod51'}

Joint [15]: {'path': '/World/Stewart/joints/Revolute_23', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_23', 'body0': '/World/Stewart/rod51', 'body1': '/World/Stewart/piston51'}

Joint [16]: {'path': '/World/Stewart/joints/Revolute_28', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_28', 'body0': '/World/Stewart/piston51', 'body1': '/World/Stewart/X5top1'}

Joint [17]: {'path': '/World/Stewart/joints/Revolute_31', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_31', 'body0': '/World/Stewart/X5top1', 'body1': '/World/Stewart/UJ51'}

Joint [18]: {'path': '/World/Stewart/joints/Revolute_4', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_4', 'body0': '/World/Stewart/base_link', 'body1': '/World/Stewart/X2bottom1'}

Joint [19]: {'path': '/World/Stewart/joints/Revolute_10', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_10', 'body0': '/World/Stewart/X2bottom1', 'body1': '/World/Stewart/cylinder21'}

Joint [20]: {'path': '/World/Stewart/joints/Slider_14', 'type': 'PhysicsPrismaticJoint', 'name': 'Slider_14', 'body0': '/World/Stewart/cylinder21', 'body1': '/World/Stewart/rod21'}

Joint [21]: {'path': '/World/Stewart/joints/Revolute_20', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_20', 'body0': '/World/Stewart/rod21', 'body1': '/World/Stewart/piston21'}

Joint [22]: {'path': '/World/Stewart/joints/Revolute_35', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_35', 'body0': '/World/Stewart/piston21', 'body1': '/World/Stewart/X2top1'}

Joint [23]: {'path': '/World/Stewart/joints/Revolute_36', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_36', 'body0': '/World/Stewart/X2top1', 'body1': '/World/Stewart/UJ21'}

Joint [24]: {'path': '/World/Stewart/joints/Revolute_5', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_5', 'body0': '/World/Stewart/base_link', 'body1': '/World/Stewart/X4bottom1'}

Joint [25]: {'path': '/World/Stewart/joints/Revolute_12', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_12', 'body0': '/World/Stewart/X4bottom1', 'body1': '/World/Stewart/cylinder41'}

Joint [26]: {'path': '/World/Stewart/joints/Slider_16', 'type': 'PhysicsPrismaticJoint', 'name': 'Slider_16', 'body0': '/World/Stewart/cylinder41', 'body1': '/World/Stewart/rod41'}

Joint [27]: {'path': '/World/Stewart/joints/Revolute_24', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_24', 'body0': '/World/Stewart/rod41', 'body1': '/World/Stewart/piston41'}

Joint [28]: {'path': '/World/Stewart/joints/Revolute_27', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_27', 'body0': '/World/Stewart/piston41', 'body1': '/World/Stewart/X4top1'}

Joint [29]: {'path': '/World/Stewart/joints/Revolute_32', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_32', 'body0': '/World/Stewart/X4top1', 'body1': '/World/Stewart/UJ41'}

Joint [30]: {'path': '/World/Stewart/joints/Revolute_6', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_6', 'body0': '/World/Stewart/base_link', 'body1': '/World/Stewart/X3bottom1'}

Joint [31]: {'path': '/World/Stewart/joints/Revolute_11', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_11', 'body0': '/World/Stewart/X3bottom1', 'body1': '/World/Stewart/cylinder31'}

Joint [32]: {'path': '/World/Stewart/joints/Slider_15', 'type': 'PhysicsPrismaticJoint', 'name': 'Slider_15', 'body0': '/World/Stewart/cylinder31', 'body1': '/World/Stewart/rod31'}

Joint [33]: {'path': '/World/Stewart/joints/Revolute_21', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_21', 'body0': '/World/Stewart/rod31', 'body1': '/World/Stewart/piston31'}

Joint [34]: {'path': '/World/Stewart/joints/Revolute_33', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_33', 'body0': '/World/Stewart/piston31', 'body1': '/World/Stewart/X3top1'}

Joint [35]: {'path': '/World/Stewart/joints/Revolute_34', 'type': 'PhysicsRevoluteJoint', 'name': 'Revolute_34', 'body0': '/World/Stewart/X3top1', 'body1': '/World/Stewart/UJ31'}

Joint [36]: {'path': '/World/Stewart/UJ61/J6B1/J6T1/TOP1/J1T1/FixedJoint', 'type': 'PhysicsFixedJoint', 'name': 'FixedJoint', 'body0': '/World/Stewart/UJ11/J1B_1', 'body1': '/World/Stewart/UJ61/J6B1/J6T1/TOP1/J1T1'}

Joint [37]: {'path': '/World/Stewart/UJ61/J6B1/J6T1/TOP1/J2T1/FixedJoint', 'type': 'PhysicsFixedJoint', 'name': 'FixedJoint', 'body0': '/World/Stewart/UJ21/J2B1', 'body1': '/World/Stewart/UJ61/J6B1/J6T1/TOP1/J2T1'}

Joint [38]: {'path': '/World/Stewart/UJ61/J6B1/J6T1/TOP1/J3T_1/FixedJoint', 'type': 'PhysicsFixedJoint', 'name': 'FixedJoint', 'body0': '/World/Stewart/UJ31/J3B1', 'body1': '/World/Stewart/UJ61/J6B1/J6T1/TOP1/J3T_1'}

Joint [39]: {'path': '/World/Stewart/UJ61/J6B1/J6T1/TOP1/J4T1/FixedJoint', 'type': 'PhysicsFixedJoint', 'name': 'FixedJoint', 'body0': '/World/Stewart/UJ41/J4B1', 'body1': '/World/Stewart/UJ61/J6B1/J6T1/TOP1/J4T1'}

Joint [40]: {'path': '/World/Stewart/UJ61/J6B1/J6T1/TOP1/J5T_1/FixedJoint', 'type': 'PhysicsFixedJoint', 'name': 'FixedJoint', 'body0': '/World/Stewart/UJ51/J5B1', 'body1': '/World/Stewart/UJ61/J6B1/J6T1/TOP1/J5T_1'}

Joint [41]: {'path': '/World/Stewart/root_joint', 'type': 'PhysicsFixedJoint', 'name': 'root_joint', 'body0': 'None', 'body1': '/World/Stewart/base_link'}


============================================================
Step 2: Load model using Isaac Lab Articulation
============================================================

======================================================================================
[INFO][IsaacLab]: Logging to file: /tmp/isaaclab/logs/isaaclab_2026-04-29_22-35-58.log
======================================================================================

22:35:58 [stage.py] WARNING: Isaac Sim < 5.0 does not support thread-local stage contexts. Skipping use_stage().
22:35:58 [simulation_context.py] WARNING: The `enable_external_forces_every_iteration` parameter in the PhysxCfg is set to False. If you are experiencing noisy velocities, consider enabling this flag. You may need to slightly increase the number of velocity iterations (setting it to 1 or 2 rather than 0), together with this flag, to improve the accuracy of velocity updates.
2026-04-29 14:35:58 [3,199ms] [Warning] [omni.physx.plugin] PhysicsUSD: Parse collision - triangle mesh collision (approximation None/MeshSimplification) cannot be a part of a dynamic body, falling back to convexHull approximation: /World/StewartPlatform/Stewart/UJ11/visuals. For dynamic collision please use approximations: convex hull, convex decomposition, box, sphere or SDF approximation.
2026-04-29 14:35:58 [3,200ms] [Warning] [omni.physx.plugin] PhysicsUSD: Parse collision - triangle mesh collision (approximation None/MeshSimplification) cannot be a part of a dynamic body, falling back to convexHull approximation: /World/StewartPlatform/Stewart/UJ11/visuals. For dynamic collision please use approximations: convex hull, convex decomposition, box, sphere or SDF approximation.
2026-04-29 14:35:58 [3,211ms] [Warning] [omni.physx.plugin] PhysicsUSD: Parse collision - triangle mesh collision (approximation None/MeshSimplification) cannot be a part of a dynamic body, falling back to convexHull approximation: /World/StewartPlatform/Stewart/UJ41/visuals. For dynamic collision please use approximations: convex hull, convex decomposition, box, sphere or SDF approximation.
2026-04-29 14:35:58 [3,212ms] [Warning] [omni.physx.plugin] PhysicsUSD: Parse collision - triangle mesh collision (approximation None/MeshSimplification) cannot be a part of a dynamic body, falling back to convexHull approximation: /World/StewartPlatform/Stewart/UJ41/visuals. For dynamic collision please use approximations: convex hull, convex decomposition, box, sphere or SDF approximation.
2026-04-29 14:35:58 [3,213ms] [Warning] [omni.physx.plugin] PhysicsUSD: Parse collision - triangle mesh collision (approximation None/MeshSimplification) cannot be a part of a dynamic body, falling back to convexHull approximation: /World/StewartPlatform/Stewart/UJ41/collisions. For dynamic collision please use approximations: convex hull, convex decomposition, box, sphere or SDF approximation.
22:35:58 [articulation.py] WARNING: Spatial tendons are not supported in Isaac Sim < 5.0: patching spatial-tendon getter and setter to use dummy value
22:35:58 [articulation.py] WARNING: Not all actuators are configured! Total number of actuated joints not equal to number of joints available: 6 != 36.

============================================================
[INFO] Articulation Joint Information
============================================================

Total number of joints: 36

Joint name list:
  [0] Revolute_1
  [1] Revolute_2
  [2] Revolute_3
  [3] Revolute_4
  [4] Revolute_5
  [5] Revolute_6
  [6] Revolute_9
  [7] Revolute_7
  [8] Revolute_8
  [9] Revolute_10
  [10] Revolute_12
  [11] Revolute_11
  [12] Slider_13
  [13] Slider_18
  [14] Slider_17
  [15] Slider_14
  [16] Slider_16
  [17] Slider_15
  [18] Revolute_19
  [19] Revolute_22
  [20] Revolute_23
  [21] Revolute_20
  [22] Revolute_24
  [23] Revolute_21
  [24] Revolute_26
  [25] Revolute_25
  [26] Revolute_28
  [27] Revolute_35
  [28] Revolute_27
  [29] Revolute_33
  [30] Revolute_29
  [31] Revolute_30
  [32] Revolute_31
  [33] Revolute_36
  [34] Revolute_32
  [35] Revolute_34

Joint limits (position):
  [0] Revolute_1: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [1] Revolute_2: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [2] Revolute_3: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [3] Revolute_4: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [4] Revolute_5: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [5] Revolute_6: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [6] Revolute_9: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [7] Revolute_7: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [8] Revolute_8: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [9] Revolute_10: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [10] Revolute_12: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [11] Revolute_11: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [12] Slider_13: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [13] Slider_18: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [14] Slider_17: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [15] Slider_14: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [16] Slider_16: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [17] Slider_15: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [18] Revolute_19: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [19] Revolute_22: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [20] Revolute_23: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [21] Revolute_20: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [22] Revolute_24: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [23] Revolute_21: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [24] Revolute_26: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [25] Revolute_25: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [26] Revolute_28: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [27] Revolute_35: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [28] Revolute_27: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [29] Revolute_33: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [30] Revolute_29: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [31] Revolute_30: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [32] Revolute_31: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [33] Revolute_36: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [34] Revolute_32: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'
  [35] Revolute_34: Cannot get limits - 'Articulation' object has no attribute 'get_joint_position_limit'

Default joint positions:
  [0] Revolute_1: 0.0000
  [1] Revolute_2: 0.0000
  [2] Revolute_3: 0.0000
  [3] Revolute_4: 0.0000
  [4] Revolute_5: 0.0000
  [5] Revolute_6: 0.0000
  [6] Revolute_9: 0.0000
  [7] Revolute_7: 0.0000
  [8] Revolute_8: 0.0000
  [9] Revolute_10: 0.0000
  [10] Revolute_12: 0.0000
  [11] Revolute_11: 0.0000
  [12] Slider_13: 0.0000
  [13] Slider_18: 0.0000
  [14] Slider_17: 0.0000
  [15] Slider_14: 0.0000
  [16] Slider_16: 0.0000
  [17] Slider_15: 0.0000
  [18] Revolute_19: 0.0000
  [19] Revolute_22: 0.0000
  [20] Revolute_23: 0.0000
  [21] Revolute_20: 0.0000
  [22] Revolute_24: 0.0000
  [23] Revolute_21: 0.0000
  [24] Revolute_26: 0.0000
  [25] Revolute_25: 0.0000
  [26] Revolute_28: 0.0000
  [27] Revolute_35: 0.0000
  [28] Revolute_27: 0.0000
  [29] Revolute_33: 0.0000
  [30] Revolute_29: 0.0000
  [31] Revolute_30: 0.0000
  [32] Revolute_31: 0.0000
  [33] Revolute_36: 0.0000
  [34] Revolute_32: 0.0000
  [35] Revolute_34: 0.0000

Link information:
  [0] base_link
  [1] X1bottom1
  [2] X6bottom1
  [3] X5bottom1
  [4] X2bottom1
  [5] X4bottom1
  [6] X3bottom1
  [7] cylinder11
  [8] cylinder61
  [9] cylinder51
  [10] cylinder21
  [11] cylinder41
  [12] cylinder31
  [13] rod11
  [14] rod61
  [15] rod51
  [16] rod21
  [17] rod41
  [18] rod31
  [19] piston11
  [20] piston61
  [21] piston51
  [22] piston21
  [23] piston41
  [24] piston31
  [25] X1top1
  [26] X6top1
  [27] X5top1
  [28] X2top1
  [29] X4top1
  [30] X3top1
  [31] UJ11
  [32] UJ61
  [33] UJ51
  [34] UJ21
  [35] UJ41
  [36] UJ31

============================================================
Step 3: Simulation running...
============================================================
[INFO] Press Ctrl+C or close window to exit
