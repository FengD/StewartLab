# Stewart平台模型加载与使用指南

> 当前项目的强化学习任务说明请优先参考 `docs/STEWART_RL_TASKS.md`。该文档包含 Stewart 平台结构、两个训练环境、观测/动作定义、奖励设计、资产路径和训练命令。本文件主要保留 USD 模型检查脚本 `scripts/load_usd_model.py` 的使用说明。

本指南详细说明如何使用 `load_usd_model.py` 脚本加载和检查Stewart平台USD模型。

---

## 目录

1. [脚本功能概述](#脚本功能概述)
2. [准备工作](#准备工作)
3. [使用方法](#使用方法)
4. [输出信息解读](#输出信息解读)
5. [常见问题](#常见问题)

---

## 脚本功能概述

`load_usd_model.py` 脚本提供以下功能：

1. **直接读取USD文件**：无需启动仿真即可查看模型结构
2. **关节信息提取**：显示所有关节的名称、类型、限制等
3. **Isaac Lab加载测试**：使用Articulation类加载模型并验证
4. **可视化展示**：在Isaac Sim中显示模型

---

## 准备工作

### 1. 放置USD模型

将你的Stewart平台USD模型文件放入 `assets/` 目录：

```
stewart_test/
├── assets/
│   ├── stewart_platform.usd          # 主模型文件
│   └── stewart_platform_materials/   # 材质文件（可选）
```

### 2. 检查模型格式

确保USD模型包含：
- ✅ 刚体（RigidBody）
- ✅ 关节（RevoluteJoint/PrismaticJoint）
- ✅ 物理属性（质量、惯性等）
- ✅ 碰撞体（可选）

---

## 使用方法

### 基本用法

```bash
# 加载模型（使用相对路径）
python scripts/load_usd_model.py --usd_path assets/stewart_platform.usd

# 加载模型（使用绝对路径）
python scripts/load_usd_model.py --usd_path /path/to/your/model.usd

# 无头模式（不显示GUI）
python scripts/load_usd_model.py --usd_path assets/stewart_platform.usd --headless
```

### 命令行参数

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `--usd_path` | USD模型文件路径 | None | ✅ |
| `--headless` | 无头模式运行 | False | ❌ |

### 示例

```bash
# 示例1: 加载本地模型
python scripts/load_usd_model.py --usd_path assets/my_stewart.usd

# 示例2: 加载Nucleus上的模型
python scripts/load_usd_model.py --usd_path "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Robots/UniversalRobots/UR10/ur10.usd"

# 示例3: 无头模式（仅打印信息）
python scripts/load_usd_model.py --usd_path assets/stewart.usd --headless
```

---

## 输出信息解读

### 步骤1: USD文件直接读取

脚本首先直接解析USD文件，显示关节信息：

```
============================================================
步骤1: 从USD文件直接读取关节信息
============================================================

找到 6 个关节:

关节 [0]:
  名称: actuator_1
  路径: /World/Robot/actuator_1
  类型: PhysicsPrismaticJoint
  轴: (0.0, 0.0, 1.0)
  限制: [-0.2000, 0.2000]
  连接体0: /World/Robot/base
  连接体1: /World/Robot/upper_leg_1

关节 [1]:
  名称: actuator_2
  路径: /World/Robot/actuator_2
  类型: PhysicsPrismaticJoint
  轴: (0.0, 0.0, 1.0)
  限制: [-0.2000, 0.2000]
  连接体0: /World/Robot/base
  连接体1: /World/Robot/upper_leg_2
...
```

**关键信息说明**：

- **名称**: 关节在USD中的名称，用于在代码中引用
- **类型**: 关节类型
  - `PhysicsPrismaticJoint`: 滑动关节（Stewart平台常用）
  - `PhysicsRevoluteJoint`: 旋转关节
  - `PhysicsSphericalJoint`: 球形关节
  - `PhysicsFixedJoint`: 固定关节
- **轴**: 关节运动方向
- **限制**: 关节运动范围（最小值，最大值）
- **连接体**: 关节连接的两个刚体

### 步骤2: Isaac Lab Articulation加载

脚本使用Isaac Lab的Articulation类加载模型：

```
============================================================
步骤2: 使用Isaac Lab Articulation加载模型
============================================================

============================================================
[INFO] Articulation关节信息
============================================================

总关节数量: 6

关节名称列表:
  [0] actuator_1
  [1] actuator_2
  [2] actuator_3
  [3] actuator_4
  [4] actuator_5
  [5] actuator_6

关节类型:
  [0] actuator_1: prismatic
  [1] actuator_2: prismatic
  [2] actuator_3: prismatic
  [3] actuator_4: prismatic
  [4] actuator_5: prismatic
  [5] actuator_6: prismatic

关节限制 (位置):
  [0] actuator_1: [-0.2000, 0.2000]
  [1] actuator_2: [-0.2000, 0.2000]
  [2] actuator_3: [-0.2000, 0.2000]
  [3] actuator_4: [-0.2000, 0.2000]
  [4] actuator_5: [-0.2000, 0.2000]
  [5] actuator_6: [-0.2000, 0.2000]

默认关节位置:
  [0] actuator_1: 0.0000
  [1] actuator_2: 0.0000
  [2] actuator_3: 0.0000
  [3] actuator_4: 0.0000
  [4] actuator_5: 0.0000
  [5] actuator_6: 0.0000

链接信息:
  [0] base
  [1] upper_leg_1
  [2] upper_leg_2
  [3] upper_leg_3
  [4] upper_leg_4
  [5] upper_leg_5
  [6] upper_leg_6
  [7] platform

根节点信息:
  根节点名称: base
  根节点路径: /World/Robot/base
```

**关键信息说明**：

- **总关节数量**: 模型中的关节数量
- **关节名称列表**: 所有关节的名称，用于配置环境
- **关节类型**: Isaac Lab识别的关节类型
- **关节限制**: 关节的位置限制
- **默认关节位置**: 模型的初始关节位置
- **链接信息**: 模型中的所有刚体
- **根节点信息**: 模型的根刚体

### 步骤3: 仿真运行

脚本启动仿真，可以在Isaac Sim中查看模型：

```
============================================================
步骤3: 仿真运行中...
============================================================
[INFO] 按 Ctrl+C 或关闭窗口退出
[INFO] 可以使用鼠标旋转、缩放查看模型
```

**交互操作**：
- 🖱️ 左键拖动：旋转视角
- 🖱️ 右键拖动：平移视角
- 🖱️ 滚轮：缩放
- 🖱️ 中键拖动：平移视角

---

## 常见问题

### Q1: 文件不存在

**错误信息**：
```
[ERROR] 文件不存在: assets/stewart_platform.usd
```

**解决方案**：
1. 检查文件路径是否正确
2. 确保文件已放置在 `assets/` 目录
3. 使用绝对路径：`--usd_path /full/path/to/model.usd`

### Q2: 未找到关节

**错误信息**：
```
[WARNING] 未在USD文件中找到关节信息
```

**可能原因**：
1. USD文件没有包含关节定义
2. 关节类型不被支持
3. 关节定义在子USD文件中

**解决方案**：
1. 使用Isaac Sim打开USD文件检查结构
2. 确保USD文件包含 `PhysicsJoint` 类型的Prim
3. 检查USD文件的引用关系

### Q3: 加载失败

**错误信息**：
```
[ERROR] 加载模型失败: ...
```

**可能原因**：
1. USD文件格式不正确
2. 缺少必要的物理属性
3. 模型过于复杂

**解决方案**：
1. 使用Isaac Sim验证USD文件
2. 简化模型（减少多边形数量）
3. 检查USD文件的物理属性

### Q4: 关节名称不匹配

**问题**：在环境配置中使用关节名称时找不到

**解决方案**：
1. 使用脚本输出的确切关节名称
2. 使用正则表达式匹配：`find_joints("actuator_.*")`
3. 检查关节名称的大小写

### Q5: 仿真卡顿

**问题**：仿真运行缓慢

**解决方案**：
1. 使用 `--headless` 模式
2. 减少模型复杂度
3. 调整仿真参数（dt、render_interval）

---

## 下一步

### 1. 记录关节信息

将脚本输出的关节信息记录下来，用于配置环境：

```python
# 在 stewart_test_env_cfg.py 中
actuator_names = [
    "actuator_1",
    "actuator_2", 
    "actuator_3",
    "actuator_4",
    "actuator_5",
    "actuator_6"
]
```

### 2. 创建机器人配置

根据关节信息创建ArticulationCfg：

```python
STEWART_PLATFORM_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/stewart_platform.usd",
    ),
    # ... 其他配置
)
```

### 3. 修改环境实现

在 `stewart_test_env.py` 中使用关节信息：

```python
def __init__(self, cfg: StewartTestEnvCfg, render_mode: str | None = None, **kwargs):
    super().__init__(cfg, render_mode, **kwargs)
    
    # 使用脚本输出的关节名称
    self._actuator_dof_idx = []
    for name in self.cfg.actuator_names:
        idx, _ = self.robot.find_joints(name)
        self._actuator_dof_idx.extend(idx)
```

### 4. 开始训练

```bash
python scripts/rsl_rl/train.py --task Template-Stewart-Test-Direct-v0
```

---

## 技术支持

如遇到问题，请：

1. 查看Isaac Lab官方文档
2. 检查Isaac Sim日志
3. 在GitHub提交Issue
4. 参考示例代码

---

## 附录

### USD文件检查清单

- [ ] 文件可以正常打开
- [ ] 包含RigidBody定义
- [ ] 包含Joint定义
- [ ] 关节有正确的连接关系
- [ ] 物理属性已设置（质量、惯性）
- [ ] 碰撞体已定义（可选）
- [ ] 材质已定义（可选）

### 关节类型对照表

| USD类型 | Isaac Lab类型 | 说明 |
|---------|---------------|------|
| PhysicsPrismaticJoint | prismatic | 滑动关节 |
| PhysicsRevoluteJoint | revolute | 旋转关节 |
| PhysicsSphericalJoint | spherical | 球形关节 |
| PhysicsFixedJoint | fixed | 固定关节 |

### 常用命令

```bash
# 检查USD文件结构
usdcat assets/stewart_platform.usd > structure.txt

# 使用Isaac Sim打开模型
./isaac-sim.sh assets/stewart_platform.usd

# 查看USD文件信息
usdcat assets/stewart_platform.usd | grep "class"
```

---

**祝您使用愉快！** 🚀
