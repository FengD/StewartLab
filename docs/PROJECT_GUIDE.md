# IsaacLab Stewart平台项目解读指南

> 当前代码已经从 Isaac Lab 模板任务演进为 Stewart 平台强化学习工程。最新任务、观测/动作、奖励、训练命令和资产路径请参考 `docs/STEWART_RL_TASKS.md`。本文档保留为项目结构和模板迁移背景说明。

本文档详细解读了IsaacLab生成的倒立摆样例工程，并指导如何将其替换为Stewart摇摆台模型。

---

## 目录

1. [项目结构概览](#项目结构概览)
2. [核心模块详解](#核心模块详解)
3. [倒立摆到Stewart平台的替换步骤](#倒立摆到stewart平台的替换步骤)
4. [关键配置说明](#关键配置说明)
5. [常见问题与解决方案](#常见问题与解决方案)

---

## 项目结构概览

```
stewart_test/
├── assets/                          # 模型资源目录（需创建）
│   └── your_stewart_model.usd       # Stewart平台USD模型
│
├── scripts/                         # 脚本目录
│   ├── load_usd_model.py           # 加载USD模型并查看关节信息的脚本
│   ├── list_envs.py                # 列出所有可用环境
│   ├── random_agent.py             # 随机智能体
│   ├── zero_agent.py               # 零智能体
│   ├── rsl_rl/                     # RSL-RL训练框架脚本
│   │   ├── train.py                # 训练脚本
│   │   ├── play.py                 # 推理/测试脚本
│   │   └── cli_args.py             # 命令行参数处理
│   └── skrl/                       # SKRL训练框架脚本
│       ├── train.py
│       └── play.py
│
├── source/                          # 源代码目录
│   └── stewart_test/               # 项目扩展模块
│       ├── config/
│       │   └── extension.toml      # 扩展配置文件
│       └── stewart_test/           # Python包
│           ├── __init__.py
│           └── tasks/              # 任务定义
│               └── direct/         # Direct RL环境
│                   ├── stewart_test/           # 单智能体任务
│                   │   ├── __init__.py         # 环境注册
│                   │   ├── stewart_test_env_cfg.py  # 环境配置
│                   │   ├── stewart_test_env.py      # 环境实现
│                   │   └── agents/             # 智能体配置
│                   │       ├── rsl_rl_ppo_cfg.py    # RSL-RL PPO配置
│                   │       └── skrl_ppo_cfg.yaml    # SKRL PPO配置
│                   └── stewart_test_marl/     # 多智能体任务
│
├── logs/                            # 训练日志目录
├── outputs/                         # 输出目录
└── pyproject.toml                   # 项目配置
```

---

## 核心模块详解

### 1. 环境配置模块 (`stewart_test_env_cfg.py`)

**作用**: 定义强化学习环境的配置参数

**关键组件**:

```python
@configclass
class StewartTestEnvCfg(DirectRLEnvCfg):
    # 环境基本参数
    decimation = 2                    # 仿真步数与控制步数的比率
    episode_length_s = 5.0            # 每个回合的时长（秒）
    action_space = 1                  # 动作空间维度
    observation_space = 4             # 观察空间维度
    
    # 仿真配置
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=decimation)
    
    # 机器人配置 - 这是替换模型的关键！
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0)
    
    # 关节名称 - 需要根据你的模型修改
    cart_dof_name = "slider_to_cart"  # 小车关节名
    pole_dof_name = "cart_to_pole"    # 摆杆关节名
    
    # 奖励权重
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    # ... 更多奖励参数
```

**替换Stewart平台需要修改**:
- `robot_cfg`: 指向你的Stewart平台USD模型
- `action_space`: Stewart平台通常有6个驱动关节
- `observation_space`: 根据需要观察的状态调整
- `cart_dof_name` / `pole_dof_name`: 改为Stewart平台的关节名称

---

### 2. 环境实现模块 (`stewart_test_env.py`)

**作用**: 实现强化学习环境的核心逻辑

**关键方法**:

| 方法 | 功能 | 说明 |
|------|------|------|
| `__init__` | 初始化环境 | 查找关节索引，初始化缓冲区 |
| `_setup_scene` | 设置场景 | 创建机器人、地面、灯光等 |
| `_pre_physics_step` | 物理步前处理 | 保存动作 |
| `_apply_action` | 应用动作 | 将动作应用到机器人关节 |
| `_get_observations` | 获取观察 | 返回当前状态观察 |
| `_get_rewards` | 计算奖励 | 根据任务目标计算奖励 |
| `_get_dones` | 判断终止 | 判断回合是否结束 |
| `_reset_idx` | 重置环境 | 重置指定环境到初始状态 |

**替换Stewart平台需要修改**:

```python
class StewartTestEnv(DirectRLEnv):
    def __init__(self, cfg: StewartTestEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # 修改：查找Stewart平台的关节
        # Stewart平台通常有6个驱动关节（电动缸）
        self._actuator_dof_idx, _ = self.robot.find_joints("actuator_.*")  # 使用正则匹配
        
    def _apply_action(self) -> None:
        # 修改：应用动作到Stewart平台的6个驱动关节
        self.robot.set_joint_effort_target(
            self.actions * self.cfg.action_scale, 
            joint_ids=self._actuator_dof_idx
        )
        
    def _get_observations(self) -> dict:
        # 修改：观察Stewart平台的状态
        # 例如：动平台位置、姿态、各关节角度等
        obs = torch.cat([
            self.robot.data.root_pos_w,           # 动平台位置
            self.robot.data.root_quat_w,          # 动平台姿态
            self.joint_pos[:, self._actuator_dof_idx],  # 关节位置
            self.joint_vel[:, self._actuator_dof_idx],  # 关节速度
        ], dim=-1)
        return {"policy": obs}
```

---

### 3. 环境注册模块 (`__init__.py`)

**作用**: 将环境注册到Gymnasium，使其可以通过名称调用

```python
gym.register(
    id="Template-Stewart-Test-Direct-v0",  # 环境ID
    entry_point=f"{__name__}.stewart_test_env:StewartTestEnv",  # 环境类
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stewart_test_env_cfg:StewartTestEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

---

### 4. 智能体配置模块 (`rsl_rl_ppo_cfg.py`)

**作用**: 配置PPO算法的超参数

```python
@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16        # 每个环境收集的步数
    max_iterations = 150          # 最大训练迭代次数
    save_interval = 50            # 保存间隔
    experiment_name = "cartpole_direct"  # 实验名称
    
    # 神经网络配置
    policy = RslRlPpoActorCriticCfg(
        actor_hidden_dims=[32, 32],    # Actor网络隐藏层
        critic_hidden_dims=[32, 32],   # Critic网络隐藏层
        activation="elu",               # 激活函数
    )
    
    # PPO算法参数
    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=1.0e-3,          # 学习率
        gamma=0.99,                     # 折扣因子
        lam=0.95,                       # GAE参数
        # ... 更多参数
    )
```

**替换Stewart平台需要修改**:
- `experiment_name`: 改为 "stewart_platform"
- `actor_hidden_dims`: 根据任务复杂度调整网络大小
- `learning_rate`: 可能需要调整

---

### 5. 训练脚本 (`train.py`)

**作用**: 启动强化学习训练

**使用方法**:
```bash
# 基本训练
python scripts/rsl_rl/train.py --task Template-Stewart-Test-Direct-v0

# 指定环境数量
python scripts/rsl_rl/train.py --task Template-Stewart-Test-Direct-v0 --num_envs 1024

# 使用GPU
python scripts/rsl_rl/train.py --task Template-Stewart-Test-Direct-v0 --device cuda:0
```

---

### 6. 推理脚本 (`play.py`)

**作用**: 加载训练好的模型进行测试

**使用方法**:
```bash
# 使用最新checkpoint
python scripts/rsl_rl/play.py --task Template-Stewart-Test-Direct-v0

# 指定checkpoint
python scripts/rsl_rl/play.py --task Template-Stewart-Test-Direct-v0 --checkpoint path/to/model.pt

# 录制视频
python scripts/rsl_rl/play.py --task Template-Stewart-Test-Direct-v0 --video
```

---

## 倒立摆到Stewart平台的替换步骤

### 步骤1: 准备USD模型

1. 将你的Stewart平台USD模型放入 `assets/` 目录
2. 使用 `load_usd_model.py` 脚本检查模型关节信息：

```bash
python scripts/load_usd_model.py --usd_path assets/stewart_platform.usd
```

3. 记录关节名称、类型和限制

### 步骤2: 创建机器人配置

在 `stewart_test_env_cfg.py` 中创建Stewart平台的配置：

```python
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

# Stewart平台配置
STEWART_PLATFORM_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/stewart_platform.usd",  # 你的模型路径
        activate_contact_sensors=False,
    ),
    init_state=sim_utils.ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # 设置初始关节位置
            "actuator_1": 0.0,
            "actuator_2": 0.0,
            # ... 其他关节
        },
    ),
    actuators={
        "actuators": sim_utils.ImplicitActuatorCfg(
            joint_names_expr=["actuator_.*"],  # 匹配所有驱动关节
            stiffness=100.0,   # 刚度
            damping=10.0,      # 阻尼
        ),
    },
)

@configclass
class StewartTestEnvCfg(DirectRLEnvCfg):
    # 修改动作和观察空间
    action_space = 6   # Stewart平台6个驱动关节
    observation_space = 18  # 位置(3) + 姿态(4) + 关节角度(6) + 关节速度(6) = 19
    
    # 使用Stewart平台配置
    robot_cfg: ArticulationCfg = STEWART_PLATFORM_CFG
    
    # Stewart平台关节名称
    actuator_names = ["actuator_1", "actuator_2", "actuator_3", 
                      "actuator_4", "actuator_5", "actuator_6"]
```

### 步骤3: 修改环境实现

在 `stewart_test_env.py` 中修改：

```python
def __init__(self, cfg: StewartTestEnvCfg, render_mode: str | None = None, **kwargs):
    super().__init__(cfg, render_mode, **kwargs)
    
    # 查找所有驱动关节
    self._actuator_dof_idx = []
    for name in self.cfg.actuator_names:
        idx, _ = self.robot.find_joints(name)
        self._actuator_dof_idx.extend(idx)

def _get_observations(self) -> dict:
    # Stewart平台观察：动平台位姿 + 关节状态
    obs = torch.cat([
        self.robot.data.root_pos_w,                    # 动平台位置 (3)
        self.robot.data.root_quat_w,                   # 动平台姿态 (4)
        self.joint_pos[:, self._actuator_dof_idx],     # 关节位置 (6)
        self.joint_vel[:, self._actuator_dof_idx],     # 关节速度 (6)
    ], dim=-1)
    return {"policy": obs}

def _get_rewards(self) -> torch.Tensor:
    # 定义Stewart平台的奖励函数
    # 例如：保持动平台水平、跟踪目标位置等
    target_pos = torch.tensor([0.0, 0.0, 0.5], device=self.device)
    pos_error = torch.sum((self.robot.data.root_pos_w - target_pos) ** 2, dim=1)
    reward = -pos_error  # 位置误差作为负奖励
    return reward
```

### 步骤4: 更新智能体配置

在 `rsl_rl_ppo_cfg.py` 中：

```python
@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    experiment_name = "stewart_platform"
    
    policy = RslRlPpoActorCriticCfg(
        actor_hidden_dims=[128, 128, 64],  # 更大的网络
        critic_hidden_dims=[128, 128, 64],
    )
```

### 步骤5: 测试和训练

```bash
# 测试环境是否正常
python scripts/random_agent.py --task Template-Stewart-Test-Direct-v0 --num_envs 1

# 开始训练
python scripts/rsl_rl/train.py --task Template-Stewart-Test-Direct-v0 --num_envs 4096
```

---

## 关键配置说明

### ArticulationCfg 参数详解

| 参数 | 说明 | 示例 |
|------|------|------|
| `prim_path` | USD场景中的路径 | `"/World/Robot"` |
| `spawn.usd_path` | USD模型文件路径 | `"assets/model.usd"` |
| `init_state.pos` | 初始位置 | `(0.0, 0.0, 0.5)` |
| `init_state.rot` | 初始姿态（欧拉角） | `(0.0, 0.0, 0.0)` |
| `actuators` | 执行器配置 | 见下表 |

### 执行器配置

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| `ImplicitActuatorCfg` | 隐式力控制 | 位置控制、力控制 |
| `ExplicitActuatorCfg` | 显式力控制 | 精确力控制 |
| `IdealPDActuatorCfg` | 理想PD控制 | 快速原型开发 |

---

## 常见问题与解决方案

### Q1: 找不到关节

**问题**: `ValueError: Could not find joint 'xxx'`

**解决方案**:
1. 使用 `load_usd_model.py` 检查实际的关节名称
2. 确保关节名称拼写正确
3. 使用正则表达式匹配：`find_joints("joint_.*")`

### Q2: 模型加载失败

**问题**: 模型无法加载或显示异常

**解决方案**:
1. 检查USD文件路径是否正确
2. 确保USD文件包含正确的物理属性
3. 使用Isaac Sim打开USD文件验证

### Q3: 训练不稳定

**问题**: 奖励不收敛或发散

**解决方案**:
1. 调整奖励函数的权重
2. 减小学习率
3. 增加网络容量
4. 检查动作空间归一化

### Q4: 仿真速度慢

**问题**: 训练速度太慢

**解决方案**:
1. 减少 `render_interval`
2. 使用GPU加速：`--device cuda`
3. 优化USD模型复杂度

---

## 参考资源

- [IsaacLab官方文档](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Sim文档](https://docs.omniverse.nvidia.com/isaacsim/latest/)
- [RSL-RL库](https://github.com/leggedrobotics/rsl_rl)
- [Stewart平台原理](https://en.wikipedia.org/wiki/Stewart_platform)

---

## 下一步

1. 将你的Stewart平台USD模型放入 `assets/` 目录
2. 运行 `load_usd_model.py` 查看关节信息
3. 根据关节信息修改环境配置
4. 定义适合Stewart平台的奖励函数
5. 开始训练！

如有问题，请参考IsaacLab官方文档或提交Issue。
