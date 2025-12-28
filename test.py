import os
import sys
import time
import numpy as np
import gymnasium as gym
import json
from datetime import datetime
from pathlib import Path
import torch

# 导入gymnasium_2048
import gymnasium_2048

from agents.dqn_agent import DQNAgent
from game.game_2048 import check_valid_actions_2048


def test_agent(agent, env_name, episodes=5, render=False, seed=42, save_logs=True):
    """
    测试已训练好的智能体
    
    参数:
        agent: 训练好的智能体
        env_name: 环境名称
        episodes: 测试回合数
        render: 是否渲染显示
        seed: 随机种子
        save_logs: 是否保存日志
    """
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)
    
    print("\n" + "=" * 60)
    print("Testing the trained DQN agent...")
    print(f"Environment: {env_name}")
    print(f"Episodes: {episodes}")
    print(f"Render mode: {'ON' if render else 'OFF'}")
    print(f"Seed: {seed}")
    print(f"Save logs: {'YES' if save_logs else 'NO'}")
    print("=" * 60)
    
    # 初始化统计列表
    total_rewards = []
    max_tiles = []
    steps_counts = []
    empty_cells_counts = []
    achieved_2048 = []
    
    # 创建日志目录
    if save_logs:
        log_dir = Path("2048_logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_board_from_env(env):
        """从环境中提取棋盘状态"""
        try:
            # 尝试直接获取棋盘
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'board'):
                return env.unwrapped.board.copy()
            
            # 尝试从observation_space获取
            if hasattr(env, 'observation_space'):
                obs_space = env.observation_space
                # 如果是Dict空间，尝试获取'board'或'observation'
                if hasattr(obs_space, 'spaces'):
                    for key in ['board', 'observation']:
                        if key in obs_space.spaces:
                            return env.unwrapped.get_state().get(key, np.zeros((4, 4), dtype=int))
            
            # 尝试从last_observation获取
            if hasattr(env, 'last_observation'):
                obs = env.last_observation
                if isinstance(obs, dict):
                    for key in ['board', 'observation']:
                        if key in obs:
                            return obs[key].copy()
                elif isinstance(obs, np.ndarray):
                    return obs.copy()
        except:
            pass
        
        # 如果以上方法都失败，返回空棋盘
        return np.zeros((4, 4), dtype=int)

    # 为每个episode测试
    for episode in range(episodes):
        # 重置环境
        episode_seed = seed + episode
        state, _ = env.reset(seed=episode_seed)
        episode_reward = 0
        done = False
        step_count = 0
        max_steps_per_episode = 2000
        
        # 创建当前episode的日志
        game_log = []
        
        # 获取初始棋盘
        initial_board = get_board_from_env(env)
        
        # 记录初始状态
        game_log.append({
            'step': 0,
            'board': initial_board.tolist(),
            'score': 0,
            'action': None,
            'reward': 0
        })
        
        # 开始游戏循环
        while not done and step_count < max_steps_per_episode:
            # 获取动作
            action = agent.get_action(
                state, 
                env=env, 
                deterministic=True, 
                use_k_step=True, 
                is_training=False
            )
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 获取当前棋盘状态
            current_board = get_board_from_env(env)
            
            # 确保reward是标量数值
            if isinstance(reward, np.ndarray):
                reward = float(reward.item() if reward.size == 1 else reward.sum())
            elif isinstance(reward, (list, tuple)):
                reward = float(sum(reward))
            else:
                reward = float(reward)
            
            # 更新累计奖励
            episode_reward += reward
            
            # 记录每一步
            game_log.append({
                'step': step_count + 1,
                'board': current_board.tolist(),
                'score': episode_reward,  # 累计分数
                'action': int(action),
                'reward': reward
            })
            
            # 更新状态
            state = next_state
            step_count += 1
            
            # 如果需要渲染，稍微延迟
            if render:
                time.sleep(0.05)
        
        # 收集统计数据
        total_rewards.append(episode_reward)
        steps_counts.append(step_count)
        
        # 获取最终棋盘
        final_board = get_board_from_env(env)
        max_tile = np.max(final_board)
        empty_cells = np.sum(final_board == 0)
        
        max_tiles.append(max_tile)
        empty_cells_counts.append(empty_cells)
        achieved_2048.append(1 if max_tile >= 2048 else 0)
        
        # 输出当前episode结果
        print(f"Test Episode {episode + 1}/{episodes}: "
              f"原始奖励 = {episode_reward:.2f}, "
              f"最大瓦片 = {max_tile}, "
              f"步数 = {step_count}, "
              f"空位 = {empty_cells}, "
              f"{'达到2048+' if max_tile >= 2048 else '未达到2048'}")
        
        # 保存当前episode的日志
        if save_logs:
            episode_log_data = {
                'episode': episode + 1,
                'seed': episode_seed,
                'final_score': float(episode_reward),
                'steps': step_count,
                'max_tile': int(max_tile),
                'empty_cells': int(empty_cells),
                'achieved_2048': bool(max_tile >= 2048),
                'game_log': game_log,
                'timestamp': timestamp
            }
            
            # 为每个episode创建单独的日志文件
            episode_log_filename = f"2048_ep{episode + 1}_seed{episode_seed}_{timestamp}.json"
            episode_log_path = log_dir / episode_log_filename
            
            with open(episode_log_path, 'w', encoding='utf-8') as f:
                json.dump(episode_log_data, f, ensure_ascii=False, indent=2)
            
            print(f"  Episode {episode + 1} 日志已保存: {episode_log_filename}")
    
    # 计算总体统计
    if episodes > 0:
        success_rate = np.mean(achieved_2048) * 100
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        avg_max_tile = np.mean(max_tiles)
        avg_steps = np.mean(steps_counts)
        avg_empty_cells = np.mean(empty_cells_counts)
        
        # 输出总体结果
        print("\n" + "=" * 60)
        print("测试完成！总体统计:")
        print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"平均最大瓦片: {avg_max_tile:.2f}")
        print(f"最大奖励: {np.max(total_rewards):.2f}")
        print(f"最小奖励: {np.min(total_rewards):.2f}")
        print(f"平均步数: {avg_steps:.1f}")
        print(f"平均空位数量: {avg_empty_cells:.1f}")
        print(f"达到2048+成功率: {success_rate:.1f}% ({sum(achieved_2048)}/{episodes})")
        print("=" * 60)
        
        # 保存汇总日志
        if save_logs and episodes > 1:
            summary_data = {
                'test_config': {
                    'env_name': env_name,
                    'episodes': episodes,
                    'seed': seed,
                    'render': render
                },
                'statistics': {
                    'total_rewards': [float(r) for r in total_rewards],
                    'max_tiles': [int(t) for t in max_tiles],
                    'steps_counts': steps_counts,
                    'empty_cells_counts': [int(c) for c in empty_cells_counts],
                    'achieved_2048': [bool(a) for a in achieved_2048],
                    'avg_reward': float(avg_reward),
                    'std_reward': float(std_reward),
                    'avg_max_tile': float(avg_max_tile),
                    'avg_steps': float(avg_steps),
                    'avg_empty_cells': float(avg_empty_cells),
                    'success_rate': float(success_rate)
                },
                'episode_logs': [f"2048_ep{ep+1}_seed{seed+ep}_{timestamp}.json" for ep in range(episodes)],
                'timestamp': timestamp
            }
            
            summary_filename = f"2048_summary_{episodes}episodes_seed{seed}_{timestamp}.json"
            summary_path = log_dir / summary_filename
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n汇总日志已保存: {summary_filename}")
    
    # 关闭环境
    env.close()
    
    # 返回测试结果
    return {
        'total_rewards': total_rewards,
        'max_tiles': max_tiles,
        'steps_counts': steps_counts,
        'empty_cells_counts': empty_cells_counts,
        'achieved_2048': achieved_2048,
        'success_rate': success_rate if episodes > 0 else 0
    }


def demo_agent(agent, env_name, episodes=3, render=False):
    """
    演示训练好的智能体
    """
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)
    
    print("\n" + "=" * 60)
    print("Starting demonstration of the trained DQN agent...")
    print(f"Render mode: {'ON' if render else 'OFF'}")
    
    total_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        max_steps_per_episode = 2000
        
        while not done and step_count < max_steps_per_episode:
            action = agent.get_action(state, env=env, deterministic=True, use_k_step=True, is_training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if render:
                time.sleep(0.1)
        
        total_rewards.append(episode_reward)
        print(f"Demonstration Episode {episode + 1}/{episodes}: "
              f"Reward = {episode_reward:.2f}, "
              f"Steps = {step_count}")
    
    env.close()
    
    print(f"Average demonstration reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print("=" * 60)


def test_dqn_2048():
    """
    简单测试2048 DQN代码是否能运行
    """
    import gymnasium_2048
    print("gymnasium-2048已安装，开始测试...")
    import gymnasium as gym
    env = gym.make("gymnasium_2048/TwentyFortyEight-v0")
    
    # 获取状态和动作空间信息
    if len(env.observation_space.shape) == 3:  # gymnasium-2048使用4x4x16
        state_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    elif len(env.observation_space.shape) == 2:  # gym-2048使用4x4网格
        state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 创建智能体
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-4,
        gamma=0.99,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=100,
        use_cuda=torch.cuda.is_available()
    )
    
    # 运行一个简单的测试episode
    state, _ = env.reset()
    print(f"初始状态示例: {state[:, :, :3] if len(state.shape) == 3 else state}")
    
    for step in range(10):  # 只运行10步测试
        action = agent.get_action(state, env=env)
        print(f"步骤 {step}: 动作 = {action}")
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 获取动作掩码
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'board'):
            board = env.unwrapped.board.copy()
            action_mask = check_valid_actions_2048(board)
        else:
            action_mask = np.ones(action_dim)
        
        # 存储经验
        agent.store_transition(state, action, reward, next_state, done, action_mask)
        
        state = next_state
        print(f"奖励: {reward}, 游戏结束: {done}")
        
        if done:
            print("游戏在步骤中结束，重置环境")
            state, _ = env.reset()
    
    print("简单测试完成，代码结构正常!")
    
    # 尝试进行一次更新
    if len(agent.replay_buffer) >= agent.batch_size:
        loss = agent.update()
        print(f"更新成功! Loss: {loss}")
    else:
        print(f"记忆库中的经验数量不足: {len(agent.replay_buffer)} < {agent.batch_size}")
    
    env.close()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_dqn_2048()
