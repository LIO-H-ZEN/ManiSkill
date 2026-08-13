#!/usr/bin/env python3
"""
Piper 可达性诊断脚本。
测试 Piper 在 PushCube 和 PickCube 任务中能否物理上完成任务。
"""
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs

def test_pushcube():
    """测试 PushCube：推方块到目标区域"""
    print("=" * 60)
    print("PushCube-v1 Piper 可达性测试")
    print("=" * 60)

    env = gym.make("PushCube-v1", obs_mode="state", robot_uids="piper", render_mode="rgb_array")
    obs, _ = env.reset()

    # 获取初始状态
    agent = env.unwrapped.agent
    tcp_pose = agent.tcp.pose
    obj_pose = env.unwrapped.obj.pose
    goal_pos = env.unwrapped.goal_region.pose.p

    print(f"\n--- 初始位置 ---")
    bp = agent.robot.pose.p[0].cpu().numpy()
    tp = tcp_pose.p[0].cpu().numpy()
    op = obj_pose.p[0].cpu().numpy()
    gp = goal_pos[0].cpu().numpy()
    print(f"Piper base at: [{bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.3f}]")
    print(f"TCP position: [{tp[0]:.3f}, {tp[1]:.3f}, {tp[2]:.3f}]")
    print(f"Cube position: [{op[0]:.3f}, {op[1]:.3f}, {op[2]:.3f}]")
    print(f"Goal position: [{gp[0]:.3f}, {gp[1]:.3f}, {gp[2]:.3f}]")
    tcp_to_cube = torch.norm(tcp_pose.p - obj_pose.p, dim=1).item()
    tcp_to_goal = torch.norm(tcp_pose.p - goal_pos, dim=1).item()
    cube_to_goal = torch.norm(obj_pose.p - goal_pos, dim=1).item()
    print(f"\nTCP → Cube dist: {tcp_to_cube:.3f}m")
    print(f"TCP → Goal dist: {tcp_to_goal:.3f}m")
    print(f"Cube → Goal dist: {cube_to_goal:.3f}m (target: +0.2 in X)")

    # 获取关节极限
    qpos = agent.robot.get_qpos()
    qlimit_low = agent.robot.get_qlimits()[0, :, 0]
    qlimit_high = agent.robot.get_qlimits()[0, :, 1]
    print(f"\n--- 关节范围 ---")
    joint_names = [f"joint{i+1}" for i in range(min(8, qpos.shape[1]))]
    for i, name in enumerate(joint_names[:6]):  # 只看前6个arm joint
        print(f"  {name}: [{qlimit_low[i]:.3f}, {qlimit_high[i]:.3f}] (current: {qpos[0,i]:.3f})")
    print(f"  gripper: [{qlimit_low[6]:.3f}, {qlimit_high[6]:.3f}]")

    # 尝试手动把 TCP 移到 cube 后方（PushCube reward 期望的位置）
    cube_pos = obj_pose.p[0].cpu().numpy()
    push_pos = cube_pos + np.array([-0.02 - 0.005, 0, 0])  # 在 cube 后方

    print(f"\n--- 目标到达 ---")
    print(f"PushCube reward 期望 TCP 在: [{push_pos[0]:.3f}, {push_pos[1]:.3f}, {push_pos[2]:.3f}]")
    print(f"初始 TCP 在: [{tp[0]:.3f}, {tp[1]:.3f}, {tp[2]:.3f}]")
    tcp_to_push = np.linalg.norm(tp - push_pos)
    print(f"TCP 到期望推位置的距离: {tcp_to_push:.3f}m")

    # 随机采样 5 个 episode 看 cube 位置分布
    print(f"\n--- 随机场景采样 ---")
    positions = []
    for i in range(10):
        obs, _ = env.reset()
        pos = env.unwrapped.obj.pose.p[0].cpu().numpy()
        goal = env.unwrapped.goal_region.pose.p[0].cpu().numpy()
        dist = np.linalg.norm(pos - goal)
        positions.append((pos, goal, dist))
        print(f"  Episode {i}: cube=({pos[0]:.2f},{pos[1]:.2f}), goal=({goal[0]:.2f},{goal[1]:.2f}), dist={dist:.2f}m")

    env.close()

def test_pickcube():
    """测试 PickCube：抓取方块移到目标"""
    print("\n" + "=" * 60)
    print("PickCube-v1 Piper 抓取测试")
    print("=" * 60)

    env = gym.make("PickCube-v1", obs_mode="state", robot_uids="piper", render_mode="rgb_array")
    obs, _ = env.reset()

    agent = env.unwrapped.agent
    cube = env.unwrapped.cube
    goal_site = env.unwrapped.goal_site

    print(f"\n--- 初始位置 ---")
    tp = agent.tcp.pose.p[0].cpu().numpy()
    cp = cube.pose.p[0].cpu().numpy()
    gsp = goal_site.pose.p[0].cpu().numpy()
    print(f"TCP: [{tp[0]:.3f}, {tp[1]:.3f}, {tp[2]:.3f}]")
    print(f"Cube: [{cp[0]:.3f}, {cp[1]:.3f}, {cp[2]:.3f}]")
    print(f"Goal: [{gsp[0]:.3f}, {gsp[1]:.3f}, {gsp[2]:.3f}]")
    tcp_to_cube = torch.norm(agent.tcp.pose.p - cube.pose.p, dim=1).item()
    cube_to_goal = torch.norm(goal_site.pose.p - cube.pose.p, dim=1).item()
    print(f"TCP→Cube: {tcp_to_cube:.3f}m, Cube→Goal: {cube_to_goal:.3f}m")

    # 检查 gripper 能否抓住 cube
    cube_half = env.unwrapped.cube_half_size
    gripper_open_dist = 0.06  # Piper gripper max open (0.03 + 0.03)
    cube_half = env.unwrapped.cube_half_size
    print(f"\nCube half_size: {cube_half:.3f}m")
    print(f"Gripper max opening: {gripper_open_dist:.3f}m")
    print(f"Can grasp? {'YES' if gripper_open_dist > cube_half * 2 else 'NO'} (gripper/2 > cube_size)")

    # 随机采样
    print(f"\n--- 随机场景采样 ---")
    for i in range(5):
        obs, _ = env.reset()
        cpos = env.unwrapped.cube.pose.p[0].cpu().numpy()
        gpos = env.unwrapped.goal_site.pose.p[0].cpu().numpy()
        tcp = agent.tcp.pose.p[0].cpu().numpy()
        dist_tcp_cube = np.linalg.norm(tcp - cpos)
        print(f"  Ep{i}: cube=({cpos[0]:.2f},{cpos[1]:.2f},{cpos[2]:.2f}), "
              f"TCP→cube={dist_tcp_cube:.2f}m, cube→goal={np.linalg.norm(cpos-gpos):.2f}m")

    env.close()

if __name__ == "__main__":
    test_pushcube()
    test_pickcube()