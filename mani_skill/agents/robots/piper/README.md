# Piper (ManiSkill3)

Piper 6-DOF 机械臂 + 双指夹爪（AgileX Robotics），接入 ManiSkill3 的适配记录。

资产来自 `piper_ros` (noetic)，控制增益对齐 Gazebo。下面是接入过程中排查并修复的 bug，以及与 Gazebo 的参数对比。

---

## 一、Bugfix 记录

### 1. TCP 位姿修正

**现象**：夹爪总是用腕部怼 cube，推/抓奖励把腕部驱到目标，夹爪实际偏 0.1358m，卡在 cube 后方，success 一直为 0。

**原因**：`ee_link_name = "link6"`（腕部法兰）。但夹爪接触点在腕部前方 0.1358m 处。

**修复**：
- URDF 新增 `piper_tcp` 无质量 link（fixed joint，`origin xyz="0 0 0.1358"`，parent=gripper_base，类比 `panda_hand_tcp`）。
- `ee_link_name = "piper_tcp"`。

### 2. 夹爪抓取参数与判断逻辑

**现象**：夹爪一直闭合往下压，而不是张开去夹；即使夹住也判为未抓取。

**原因（5 个叠加 bug）**：

| # | bug | 修复 |
|---|-----|------|
| a | mimic 用 `"ratio": -1.0`，但控制器读 `multiplier` → 两指同向运动，无开合 | 改 `"multiplier": -1.0` |
| b | joint7=0 是**闭合**，0.035 是**张开**（与假设相反） | 反转映射方向 |
| c | 控制器 `lower=-0.015, upper=0.04` → 中性 action=0 映射到 36% 闭合 | 改 `lower=0.0, upper=0.035`，action=-1→闭合，action=1→张开 |
| d | home keyframe joint7=0（闭合） | 改 0.035（张开），scene_builder 强制初始张开 |
| e | `is_grasping` 用 finger link Y 轴（指向前方）做开合方向 → 角度 88° > 85° 阈值，恒判 False | 改用 Z 轴（真实开合方向），角度降到 ~2° |

**附带**：夹爪增益从 panda 的 k=1000,b=100（力限饱和）改为 Gazebo 的 k=100,b=10。

### 3. 手臂关节参数对齐 Gazebo

**现象**：能夹住 cube，但学不会提起来。

**原因**：`arm_stiffness=1000`（Gazebo 的 10 倍）→ TCP 峰值加速度 a_max=87.7 m/s² → 夹住的 cube 受 5.6N 惯性力，远超 1.2N 摩擦握力 → 探索阶段 cube 被甩飞 → 策略拿不到 place_reward → 学不会 lift。

**迭代过程**：

| 增益 | a_max | 跟踪 err@5 | 结论 |
|------|-------|-----------|------|
| k=1000, b=20 | 87.7 | 0.036 | 太快，甩飞 cube |
| k=1000, b=100 | 15.4 | 0.039 | 太慢（用户反馈） |
| k=100, b=20 | 5.8 | 0.121 | 过阻尼，夹不准 |
| **k=100, b=5（Gazebo）** | **38** | **0.036** | **跟踪好，最终采用** |

> **k=100, b=20 的判断依据**：训练视频里夹爪总是夹不准 cube（到位偏差大），据此推测过阻尼导致跟踪率差。后续实测确认：第 5 步跟踪误差 0.121 rad（远大于其他配置的 0.036），b=20 对轻连杆（腕关节）阻尼比 ζ 高达 14~30，严重过阻尼。降到 b=5 后跟踪恢复。

**修复**：`arm_stiffness=100, arm_damping=5`（对齐 Gazebo）。

**附带**：`ppo.py` 加 `--max-episode-steps`（PushCube 默认 50 步对慢速 piper 太短）。

---

## 二、与 Gazebo 参数对比

参考：`piper_ros` noetic 分支 `src/piper_sim/piper_gazebo/config/piper_gazebo_control.yaml`。

### 控制增益

| 项目 | Gazebo | Menagerie | ManiSkill piper |
|------|--------|-----------|-----------------|
| arm p / kp (joint1-6) | 100 | 80/80/80/40/10/10 | **100** |
| arm d / kv | 5 | 5/5/5/5/1.5/1.5 | **5** |
| arm 积分项 i | 0.01 | — | —（纯 PD） |
| arm force | — | ±100 | 100 |
| gripper p / kp (joint7-8) | 100 | 40 | **100** |
| gripper d / kv | 10 | 5 | **10** |
| gripper i | 0.001 | — | — |
| gripper force | — | ±10 | 10 |

### 关节限位

| 关节 | 范围 | 类型 |
|------|------|------|
| joint1 | [-2.618, 2.618] | revolute |
| joint2 | [0, 3.14] | revolute |
| joint3 | [-2.967, 0] | revolute |
| joint4 | [-1.832, 1.832] | revolute |
| joint5 | [-1.22, 1.22] | revolute |
| joint6 | [-3.14, 3.14] | revolute |
| joint7 | [0, 0.035] | prismatic（夹爪） |
| joint8 | [-0.035, 0] | prismatic（夹爪，mimic joint7） |

### 夹爪 mimic

| 来源 | 实现 |
|------|------|
| Gazebo | equality constraint `joint8 = -joint7` |
| Menagerie | equality `polycoef="0 -1 0 0 0"` |
| ManiSkill | `PDJointPosMimicControllerConfig` `mimic={"joint8": {"joint": "joint7", "multiplier": -1.0}}` |

### 关键差异

1. **纯 PD vs PID**：ManiSkill 控制器是纯 PD（无积分项），Gazebo 是完整 PID。稳态误差靠足够 stiffness 消除。
2. **无 armature**：SAPIEN 不支持 MuJoCo 的 armature（menagerie 靠 armature=0.005 稳定轻连杆）。Gazebo 的 d=5 在 SAPIEN 里大阶跃 a_max=38，探索动作不宜过剧烈。
3. **TCP**：Gazebo/Menagerie 无显式 TCP，ManiSkill 新增 `piper_tcp`（腕部前方 0.1358m，指尖接触点）。
4. **增益已对齐 Gazebo**：手臂 p=100/d=5、夹爪 p=100/d=10，与 Gazebo 一一对应。

---

## 三、关键物理参数

| 参数 | 值 |
|------|-----|
| 夹爪手指质量（link7/8） | 0.0304 kg |
| 夹爪满开间距 | 0.07 m |
| 夹爪满闭间距 | 0 m |
| 夹爪夹持力（k=100） | ~2 N/指 |
| TCP 偏移（link6 → piper_tcp） | 0.1358 m |
| PickCube cube 尺寸/质量 | 0.04 m / 0.064 kg |
| 控制频率 | 20 Hz |
| 仿真步长 | 0.01 s |

---

## 四、训练效果视频

### PushCube

<video src="./piper_push_cube.mp4" controls muted width="480"></video>

### PickCube

<video src="./piper_pick_cube.mp4" controls muted width="480"></video>
