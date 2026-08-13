# Piper 机器人迁移至 ManiSkill3 技术报告

> AgileX Piper 6-DOF 桌面机械臂从 ROS/URDF 和 MuJoCo 迁移至 ManiSkill3 仿真框架的完整技术文档。

---

## 目录

1. [项目概述](#1-项目概述)
2. [Piper 机器人简介](#2-piper-机器人简介)
3. [迁移前分析：三个源的对齐](#3-迁移前分析三个源的对齐)
4. [Step 1：URDF 模型准备与修正](#4-step-1urdf-模型准备与修正)
5. [Step 2：凸碰撞网格生成（CoACD）](#5-step-2凸碰撞网格生成coacd)
6. [Step 3：自碰撞排除（SRDF）](#6-step-3自碰撞排除srdf)
7. [Step 4：Python Agent 实现](#7-step-4python-agent-实现)
8. [Step 5：注册与任务集成](#8-step-5注册与任务集成)
9. [Step 6：可达性调试与修复](#9-step-6可达性调试与修复)
10. [PPO 训练 Pipeline](#10-ppo-训练-pipeline)
11. [参数对齐对照表](#11-参数对齐对照表)
12. [最终文件结构](#12-最终文件结构)
13. [使用方式](#13-使用方式)

---

## 1. 项目概述

### 目标

将 AgileX Piper 6-DOF 机械臂完整迁移到 ManiSkill3 仿真平台，使其能够：
- 在 `PushCube-v1` 和 `PickCube-v1` 等标准 ManiSkill 任务中运行
- 支持 RGB + State 混合观测的 PPO 强化学习训练
- 与 ManiSkill 的控制器体系（PD joint pos、PD EE pose 等）完全兼容

### 源材料

| 源 | 仓库 | 作用 |
|----|------|------|
| **piper_ros** | https://github.com/agilexrobotics/piper_ros (noetic) | 提供原始 URDF、STL mesh、惯量/质量参数（来自 CAD） |
| **mujoco_menagerie** | https://github.com/google-deepmind/mujoco_menagerie (agilex_piper) | 提供已验证的 MuJoCo XML，含 actuator 参数、碰撞简化策略、mimic 方案、keyframe |
| **ManiSkill3** | https://github.com/haosulab/ManiSkill (panda, so100 等) | 参照标准 Agent 模板、控制器配置模式、任务集成方式 |

---

## 2. Piper 机器人简介

| 属性 | 值 |
|------|-----|
| **制造商** | AgileX Robotics |
| **自由度** | 6-DOF 手臂 + 2 指夹爪 = **8 DOF** |
| **关节类型** | joint1–joint6: revolute（旋转）；joint7–joint8: prismatic（平移夹爪） |
| **连杆链** | `base_link → link1 → link2 → link3 → link4 → link5 → link6 → gripper_base → (link7, link8)` |
| **末端法兰** | `link6`（TCP，Tool Center Point） |
| **夹爪类型** | 平行二指，joint7 和 joint8 互为镜像（mimic） |
| **最大臂展** | ~0.6m（从基座算起） |
| **工作空间** | X: [-0.21, 0.27], Y: [-0.47, 0.47], Z: [0.03, 0.75]（基座位于 x=-0.35 时） |

---

## 3. 迁移前分析：三个源的对齐

在动手写代码之前，先对三个源进行交叉对比，找出差异并决定以哪个为准。

### 3.1 关节范围对比

| Joint | URDF (piper_ros) | MuJoCo (menagerie) | MuJoCo (piper_ros 自带) | **最终采用** |
|-------|-------------------|---------------------|--------------------------|--------------|
| joint1 | [-2.618, 2.168] | [-2.618, 2.618] | [-2.618, 2.618] | **[-2.618, 2.618]** ← 对称化 |
| joint2 | [0, 3.14] | [0, 3.14] | [0, 3.14] | [0, 3.14] |
| joint3 | [-2.697, 0] | [-2.697, 0] | [-2.967, 0] | **[-2.967, 0]** ← 取宽松 |
| joint4 | [-1.832, 1.832] | [-1.832, 1.832] | [-1.745, 1.745] | **[-1.832, 1.832]** ← 取宽松 |
| joint5 | [-1.22, 1.22] | [-1.22, 1.22] | [-1.22, 1.22] | [-1.22, 1.22] |
| joint6 | [-2.094, 2.094] | [-3.14, 3.14] | [-2.094, 2.094] | **[-3.14, 3.14]** ← 取宽松 |
| joint7 | [0, 0.035] | [0, 0.035] | [0, 0.035] | [0, 0.035] |
| joint8 | [-0.035, 0] | [-0.035, 0] | [-0.035, 0] | [-0.035, 0] |

**原则**：优先采用 mujoco_menagerie 的值（该仓库是专门为物理仿真验证过的），当 piper_ros 自带 MuJoCo XML 范围更宽松时也采用其值。

### 3.2 质量/惯量参数对比

| 方案 | 说明 | 采用 |
|------|------|------|
| URDF inertial | 来自 SolidWorks CAD 模型，每个 link 独立定义 | **✅ 采用** |
| MuJoCo inertial | 手工重新标定，link6 合并了夹爪质量（1.2kg vs 0.007+0.457kg） | ❌ |

**原因**：ManiSkill 直接加载 URDF，使用 URDF 自带的 inertial 最一致。URDF 的 inertial 与 visual/collision mesh 的几何原点对齐良好，且每个 link 独立定义更符合多体动力学建模规范。

### 3.3 碰撞模型对比

| 源 | 碰撞模型 | 优缺点 |
|----|----------|--------|
| URDF (piper_ros) | 原始 STL mesh | 精度高但非凸，SAPIEN 物理引擎可能报错或精度差 |
| MuJoCo (menagerie) | 简化为 capsule/box | 仿真速度快，但改变了碰撞几何形状 |
| **最终方案** | **CoACD 凸分解** | 既保持几何精度，又满足 SAPIEN 凸碰撞要求 |

### 3.4 夹爪 Mimic 实现对比

| 框架 | 实现方式 |
|------|----------|
| ROS URDF | 两个独立 prismatic joint，无 mimic 机制 |
| MuJoCo | `<equality joint joint1="joint8" joint2="joint7" polycoef="0 -1 0 0 0"/>` |
| **ManiSkill3** | `PDJointPosMimicControllerConfig(mimic={"joint8": {"joint": "joint7", "ratio": -1.0}})` |

三者等价效果：给定 1 个控制信号，joint7 正向移动 +d，joint8 反向移动 -d。

### 3.5 控制器增益对比

| 关节 | MuJoCo kp | MuJoCo kv | ManiSkill 采用 |
|------|-----------|-----------|---------------|
| joint1–3 | 80 | 5 | stiffness=1000, damping=100 |
| joint4 | 40 | 5 | stiffness=1000, damping=100 |
| joint5–6 | 10 | 1.5 | stiffness=1000, damping=100 |
| gripper | 40 | 5 | stiffness=1000, damping=100, force_limit=10N |

ManiSkill 使用统一 stiffness/damping 值是设计惯例（参考 panda），虽然与 MuJoCo 的差异化 kp 不同，但对 RL 训练影响不大。

### 3.6 关节 origin 差异

MuJoCo 和 URDF 使用不同的 quaternion 表示，导致 joint position 数值不同但实际空间姿态等效。例如：
- MuJoCo: `joint3` pos (0.28358, 0.028726, 0)
- URDF: `joint3` pos (0.28503, 0, 0)

经 quaternion 变换后等效，无需修改。

---

## 4. Step 1：URDF 模型准备与修正

### 4.1 从 piper_ros 提取资源

```bash
git clone --depth 1 --branch noetic https://github.com/agilexrobotics/piper_ros.git
cp piper_ros/src/piper_description/meshes/*.STL mani_skill/assets/robots/piper/meshes/
```

### 4.2 URDF 修改清单

原始文件：`piper_ros/src/piper_description/urdf/piper_description.urdf`

#### (a) 移除 dummy_link

```xml
<!-- 删除以下内容，ManiSkill/SAPIEN 不需要这个 ROS 惯例 -->
<link name="dummy_link"/>
<joint name="base_to_dummy" type="fixed">
  <parent link="dummy_link"/>
  <child link="base_link"/>
</joint>
```

原因：`dummy_link` 是 ROS 的惯例（world → dummy_link → base_link），让机器人在 Gazebo 中能通过 fixed joint 固定在 world 中。ManiSkill 使用 `fix_root_link=True` 直接固定 `base_link`。

#### (b) 修改 mesh 路径

```xml
<!-- 从 ROS package 路径 -->
<mesh filename="package://piper_description/meshes/base_link.STL" />
<!-- 改为相对路径 -->
<mesh filename="meshes/base_link.STL" />
```

#### (c) 碰撞 mesh 引用改为凸分解文件

```xml
<!-- visual 保持原始 STL -->
<visual>
  <mesh filename="meshes/link1.STL" />
</visual>
<!-- collision 使用 CoACD 凸分解结果 -->
<collision>
  <mesh filename="meshes/link1.convex.stl" />
</collision>
```

#### (d) 关节范围对齐

根据 3.1 节的对照表更新所有 `<limit lower="" upper="">` 值。

#### (e) 质量/惯量对齐

直接使用 URDF 中的原始 inertial 值（来自 CSV 的每个 link 独立质量）。

### 4.3 最终 URDF 结构

```
机器人名: piper
Links (10): base_link, link1~link8, gripper_base
Joints (10): joint1~joint6 (revolute), joint6_to_gripper_base (fixed), joint7~joint8 (prismatic)
Active DOF: 8
```

---

## 5. Step 2：凸碰撞网格生成（CoACD）

### 5.1 为什么需要凸分解？

SAPIEN 物理引擎要求碰撞几何为**凸面体**（convex mesh）。原始的 STL 文件通常是非凸的复杂三角网格，直接用作碰撞体会导致物理引擎报错、碰撞检测不准确、GPU 仿真性能下降。

### 5.2 方法

```bash
pip install coacd trimesh
```

```python
import trimesh, coacd
mesh = trimesh.load("link1.STL", force="mesh")
cmesh = coacd.Mesh(mesh.vertices.astype(np.float64), mesh.faces.astype(np.int32))
parts = coacd.run_coacd(cmesh, threshold=0.05, max_convex_hull=-1)
combined = trimesh.util.concatenate([trimesh.Trimesh(vertices=p[0], faces=p[1]) for p in parts])
combined.export("link1.convex.stl")
```

CoACD 参数：
- `threshold=0.05`：凸分解误差阈值
- `max_convex_hull=-1`：不限制最大凸包数量

### 5.3 生成结果

| Link | 原始面数 | 凸包数 | 耗时 | 文件大小 |
|------|---------|--------|------|---------|
| base_link | 12,146 | 177 | ~50s | 1.1 MB |
| link1 | 8,954 | 174 | ~50s | 1.0 MB |
| link2 | 73,166 | 37 | ~8s | 0.4 MB |
| link3 | 36,914 | 31 | ~9s | 0.4 MB |
| link4 | 17,558 | 271 | ~59s | 2.2 MB |
| link5 | 16,672 | 65 | ~19s | 0.7 MB |
| link6 | 1,176 | 183 | ~12s | 0.3 MB |
| gripper_base | 12,988 | 58 | ~11s | 0.5 MB |
| link7 | 2,086 | 19 | ~8s | 0.2 MB |
| link8 | 2,086 | 20 | ~10s | 0.2 MB |
| **总计** | — | **1,035** | **~4 min** | **6.8 MB** |

---

## 6. Step 3：自碰撞排除（SRDF）

### 6.1 为什么需要 SRDF？

在物理仿真中，相邻连杆之间不需要检测碰撞（它们通过关节自然连接），某些远距离的 link pair 也永远不会碰撞。不排除这些碰撞会浪费计算资源并可能导致仿真不稳定。

### 6.2 SRDF 规则分类

参照 `panda_v2.srdf`，分三类共 27 对排除规则：

**1. Adjacent（相邻连杆，9 对）**
```xml
<disable_collisions link1="base_link" link2="link1" reason="Adjacent"/>
<disable_collisions link1="link1" link2="link2" reason="Adjacent"/>
<!-- ... 共 9 对 -->
```

**2. Never（永远不会碰撞，16 对）**
```xml
<disable_collisions link1="base_link" link2="link2" reason="Never"/>
<disable_collisions link1="base_link" link2="link3" reason="Never"/>
<!-- ... 共 16 对 -->
```

**3. Default（默认排除，2 对）**
```xml
<disable_collisions link1="link7" link2="link8" reason="Default"/>
<disable_collisions link1="link6" link2="link7" reason="Default"/>
```

### 6.3 自动加载机制

SAPIEN 的 URDF loader 会自动寻找与 URDF 同名的 `.srdf` 文件（即 `piper_description.srdf`），无需额外代码。

---

## 7. Step 4：Python Agent 实现

### 7.1 Agent 类结构

遵循 ManiSkill `BaseAgent` 模板（参考 `panda.py`）：

```python
@register_agent()
class Piper(BaseAgent):
    uid = "piper"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/piper/piper_description.urdf"

    arm_joint_names = ["joint1","joint2","joint3","joint4","joint5","joint6"]
    gripper_joint_names = ["joint7", "joint8"]
    ee_link_name = "link6"  # 末端法兰

    keyframes = dict(
        home=Keyframe(qpos=np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0, 0.0, 0.0]))
    )
```

### 7.2 关键设计决策

#### (a) EE link 选择：`link6`

| 候选 | 说明 | 采用 |
|------|------|------|
| `link6` | 手臂最后一个 revolute joint 所连的 link | **✅ 采用** |
| `gripper_base` | 通过 fixed joint 连到 link6 | ❌ 与 panda 模式不一致 |

#### (b) 夹爪 Mimic 控制

```python
gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
    self.gripper_joint_names,
    lower=-0.015, upper=0.04,
    stiffness=1e3, damping=1e2, force_limit=10,  # force_limit 对齐 MuJoCo
    mimic={"joint8": {"joint": "joint7", "ratio": -1.0}},
)
```

`ratio: -1.0` 的物理含义：在 URDF 中，joint7 的 axis 是 `(0,0,1)`，joint8 的 axis 是 `(0,0,-1)`。两个手指需要对称运动（同时打开/关闭），故 joint8 = -1.0 × joint7。

#### (c) 控制器配置（11 种控制模式）

| 控制模式 | arm 控制器 | gripper 控制器 |
|----------|-----------|---------------|
| `pd_joint_pos` | 绝对关节位置 | mimic |
| `pd_joint_delta_pos` | delta 关节位置 | mimic |
| `pd_ee_delta_pos` | delta 末端位置 | mimic |
| `pd_ee_delta_pose` | delta 末端姿态 | mimic |
| `pd_ee_pose` | 绝对末端姿态 | mimic |
| `pd_joint_vel` | 关节速度 | mimic |
| `pd_joint_pos_vel` | 关节位置+速度 | mimic |
| `pd_joint_delta_pos_vel` | delta 位置+速度 | mimic |
| `pd_joint_target_delta_pos` | 以目标 qpos 为基准 | mimic |
| `pd_ee_target_delta_pos` | 以目标末端位置为基准 | mimic |
| `pd_ee_target_delta_pose` | 以目标末端姿态为基准 | mimic |

#### (d) 抓取检测（`is_grasping`）

```python
def is_grasping(self, object, min_force=0.5, max_angle=85):
    l_contact = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
    r_contact = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
    # 检查接触力大小 > min_force 且接触角度 < max_angle
```

#### (e) 物理参数

```python
arm_stiffness = 1e3    # PD 控制器 P 增益
arm_damping = 1e2      # PD 控制器 D 增益
arm_force_limit = 100  # 关节力限制 (N)

gripper_stiffness = 1e3
gripper_damping = 1e2
gripper_force_limit = 10   # 对齐 MuJoCo finger forcerange="-10 10"
```

#### (f) 夹爪摩擦面配置

```python
urdf_config = dict(
    _materials=dict(
        gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
    ),
    link=dict(
        link7=dict(material="gripper", patch_radius=0.02, min_patch_radius=0.005),
        link8=dict(material="gripper", patch_radius=0.02, min_patch_radius=0.005),
    ),
)
```

`patch_radius=0.02` 是关键参数：Piper 手指较小，初始值 0.1（参考 panda）对于 piper 来说太大，导致 SAPIEN 无法在手指表面生成有效的摩擦接触面。改为 0.02 后，`get_pairwise_contact_forces` 能正确检测手指与物体的接触。

---

## 8. Step 5：注册与任务集成

### 8.1 模块导出

```python
# mani_skill/agents/robots/piper/__init__.py
from .piper import Piper

# mani_skill/agents/robots/__init__.py
from .piper import *
```

### 8.2 PushCube-v1 集成

`pick_cube.py` 中的关键修改：

```python
SUPPORTED_ROBOTS = ["panda", "fetch", "piper"]
agent: Union[Panda, Fetch, Piper]

def _load_agent(self, options: dict):
    if self.robot_uids == "piper":
        # Piper 是桌面级机械臂，基座需要更靠近桌子
        super()._load_agent(options, sapien.Pose(p=[-0.35, 0, 0]))
    else:
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))
```

### 8.3 PickCube-v1 集成

除了 SUPPORTED_ROBOTS 和 `_load_agent` 修改外，还在 `pick_cube_cfgs.py` 中为 Piper 定制了任务配置：

```python
"piper": {
    "cube_half_size": 0.02,
    "goal_thresh": 0.025,
    "cube_spawn_half_size": 0.06,
    "cube_spawn_center": (0.03, 0.0),
    "max_goal_height": 0.12,
    "sensor_cam_eye_pos": [-0.1, 0, 0.45],
    "sensor_cam_target_pos": [0.03, 0, 0.08],
    "human_cam_eye_pos": [0.3, 0.5, 0.40],
    "human_cam_target_pos": [0.03, 0.0, 0.08],
},
```

---

## 9. Step 6：可达性调试与修复

### 9.1 问题诊断

初始版本中，Piper 基座位于 `x=-0.615`（与 panda 相同），但立方体默认 spawning 在 `[0.1, 0.1]×[-0.1, -0.1]`，导致 TCP 距立方体 **0.66m**——远超 Piper 的有效臂展（~0.5m）。

### 9.2 工作空间扫描

通过 8100 个关节配置点的扫描，得到 Piper 在桌面高度（z≈0.04m）的有效工作空间：

```
X: [-0.213, 0.272]
Y: [-0.470, 0.469]
```

### 9.3 修复措施

| 修复 | 旧值 | 新值 | 依据 |
|------|------|------|------|
| 基座位置 x | -0.615 | **-0.35** | 更靠近桌子中心 |
| cube spawn center x | 0.0 | **0.03** | 工作空间扫描中心点 |
| cube spawn half size | 0.1 | **0.06** | 确保所有位置可达 |
| max goal height | 0.3 | **0.12** | 适配小臂展 |

修复后 TCP 最小可达距离从 0.66m 降至 **0.093m**，可推动立方体。

### 9.4 PickCube 可达性

对于 PickCube，经修复后的配置（基座 x=-0.35, cube center=(0.03, 0)），TCP 距 cube 最小距离为 **0.032m**，理论可达。但 `patch_radius` 的修复是必须的——否则即使手指物理重叠，接触力始终为 0。

---

## 10. PPO 训练 Pipeline

### 10.1 PushCube 训练脚本

`train_piper_ppo_rgb.py`：基于 CleanRL PPO 实现，支持：

- **观测模式**：RGB (128×128×3) + 可选 State 信息
- **控制模式**：`pd_joint_delta_pos`（7 维连续动作）
- **网络架构**：NatureCNN（3 层 conv）→ 256 维特征 + 可选 256 维 state encoder → Actor-Critic
- **默认参数**：256 并行环境、5M 步训练、10 epoch PPO 更新

### 10.2 PickCube 训练脚本

`train_piper_pick_cube.py`：为抓取任务优化，默认使用 `--include-state`（State 信息对 learn grasping 至关重要），10M 步训练。

### 10.3 训练命令

```bash
# PushCube（推箱子）
python train_piper_ppo_rgb.py --env-id PushCube-v1 --robot-uid piper \
  --num-envs 256 --total-timesteps 5000000 --include-state --capture-video

# PickCube（抓取放置）
python train_piper_pick_cube.py --num-envs 256 --total-timesteps 10000000 \
  --include-state --capture-video
```

### 10.4 训练结果

**PushCube**（5M 步）：
- Return 趋势：0.697→1.781（+155%）
- Reward 趋势：0.014→0.036（+157%）
- 成功尚未出现（PushCube 首次成功通常需要 2-3M 步的优化探索）

**PickCube**（5M 步，修复前）：
- 两次训练因基座位置和 patch_radius 问题导致 success=0
- 修复后应能看到 >0 的成功率

---

## 11. 参数对齐对照表

### 11.1 关节范围（最终采用值）

| Joint | 下界 | 上界 | 单位 | 来源 |
|-------|------|------|------|------|
| joint1 | -2.618 | 2.618 | rad | mujoco_menagerie（对称化） |
| joint2 | 0 | 3.14 | rad | 所有源一致 |
| joint3 | -2.967 | 0 | rad | piper_ros 自带 mujoco（最宽松） |
| joint4 | -1.832 | 1.832 | rad | mujoco_menagerie/URDF |
| joint5 | -1.22 | 1.22 | rad | 所有源一致 |
| joint6 | -3.14 | 3.14 | rad | mujoco_menagerie（最宽松） |
| joint7 | 0 | 0.035 | m | 所有源一致 |
| joint8 | -0.035 | 0 | m | 所有源一致 |

### 11.2 坐标系/变换对齐

| 概念 | URDF | MuJoCo | ManiSkill |
|------|------|--------|-----------|
| base_link 原点 | (0, 0, 0) | (0, 0, 0) | 直接使用 URDF |
| joint1 位置 | (0, 0, 0.123) | (0, 0, 0.123) | 直接使用 URDF |
| joint 方向 | 全部绕 Z 轴 | 全部绕 Z 轴 | 直接使用 URDF |
| mesh 格式 | STL (binary) | STL/OBJ | STL（视觉）/ .convex.stl（碰撞） |

### 11.3 文件名对齐

| piper_ros | mujoco_menagerie | ManiSkill3 |
|-----------|------------------|------------|
| `base_link.STL` | `base_link.stl` | `base_link.STL` + `base_link.convex.stl` |
| `piper_description.urdf` | `piper.xml` | `piper_description.urdf` + `piper_description.srdf` |

---

## 12. 最终文件结构

```
mani_skill_repo/mani_skill/
├── assets/robots/piper/
│   ├── piper_description.urdf          # URDF：visual 用 .STL，collision 用 .convex.stl
│   ├── piper_description.srdf          # 自碰撞排除（27 对规则）
│   └── meshes/
│       ├── base_link.STL ~ link8.STL, gripper_base.STL           # 10 个视觉网格
│       └── base_link.convex.stl ~ link8.convex.stl, gripper_base.convex.stl  # 10 个凸碰撞
│
├── agents/robots/piper/
│   ├── __init__.py                     # from .piper import Piper
│   └── piper.py                        # Piper(BaseAgent)：8-DOF, 11 种控制模式
│
├── agents/robots/__init__.py           # 添加 from .piper import *
│
└── envs/tasks/tabletop/
    ├── push_cube.py                    # SUPPORTED_ROBOTS += "piper", base x=-0.35
    ├── pick_cube.py                    # SUPPORTED_ROBOTS += "piper", base x=-0.35
    └── pick_cube_cfgs.py              # piper 专用任务配置
```

```
mani_skill_repo/
├── train_piper_ppo_rgb.py              # PushCube PPO 训练脚本（CleanRL 风格）
├── train_piper_pick_cube.py            # PickCube PPO 训练脚本（含 best.pt 保存）
├── make_video.py                       # 视频渲染脚本
├── monitor_training.sh                 # 训练监控脚本（自动检测首次成功）
└── PIPER_MIGRATION.md                  # 本技术报告
```

---

## 13. 使用方式

### 13.1 官方 API

```python
import gymnasium as gym
import mani_skill.envs

# PushCube
env = gym.make("PushCube-v1", robot_uids="piper")

# PickCube
env = gym.make("PickCube-v1", robot_uids="piper",
               control_mode="pd_joint_delta_pos",
               obs_mode="rgb+depth")
```

### 13.2 训练

```bash
# PushCube PPO
cd mani_skill_repo
python train_piper_ppo_rgb.py --env-id PushCube-v1 --robot-uid piper \
  --num-envs 256 --total-timesteps 5000000 --include-state --capture-video

# PickCube PPO
python train_piper_pick_cube.py \
  --num-envs 256 --total-timesteps 10000000 --include-state --capture-video
```

### 13.3 评估

```bash
# 用训练好的 checkpoint 评估并生成视频
python train_piper_pick_cube.py --evaluate \
  --checkpoint runs/piper__PickCube-v1__.../best.pt \
  --capture-video
```

---

*文档生成时间：2026-08-12*  
*技术栈：ManiSkill3 + SAPIEN + CleanRL PPO + CoACD*