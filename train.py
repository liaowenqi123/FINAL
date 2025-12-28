import os
import sys
import time
import argparse
import numpy as np
import gymnasium as gym
import torch

from agents.dqn_agent import DQNAgent
from utils.visualization import plot_training_results
from test import test_agent, demo_agent
from config import DQNConfig


def train_dqn():
    """使用默认参数的训练函数"""
    # 使用配置文件中的默认参数
    class Args:
        def __init__(self):
            self.env_name = DQNConfig.ENV_NAME
            self.episodes = DQNConfig.EPISODES
            self.max_steps = DQNConfig.MAX_STEPS
            self.lr = DQNConfig.LEARNING_RATE
            self.gamma = DQNConfig.GAMMA
            self.buffer_size = DQNConfig.BUFFER_SIZE
            self.batch_size = DQNConfig.BATCH_SIZE
            self.target_update_freq = DQNConfig.TARGET_UPDATE_FREQ
            self.epsilon_start = DQNConfig.EPSILON_START
            self.epsilon_end = DQNConfig.EPSILON_END
            self.epsilon_decay = DQNConfig.EPSILON_DECAY
            self.save_freq = DQNConfig.SAVE_FREQ
            self.eval_freq = DQNConfig.EVAL_FREQ
            self.test_model = ""
            self.test_episodes = DQNConfig.TEST_EPISODES
            self.render_test = DQNConfig.RENDER_TEST
            self.seed = DQNConfig.SEED
            self.render_demo = DQNConfig.RENDER_DEMO
            self.save_plot = DQNConfig.SAVE_PLOT
            self.use_k_step = DQNConfig.USE_K_STEP
            self.k_step = DQNConfig.K_STEP
            self.num_simulations = DQNConfig.NUM_SIMULATIONS
            self.use_k_step_train = DQNConfig.USE_K_STEP_TRAIN
            self.k_step_train = DQNConfig.K_STEP_TRAIN
            self.num_simulations_train = DQNConfig.NUM_SIMULATIONS_TRAIN
            self.k_step_temperature = DQNConfig.K_STEP_TEMPERATURE
    
    args = Args()
    
    # 执行训练
    _execute_training(args)


def train_dqn_with_args():
    """支持命令行参数的训练函数"""
    parser = argparse.ArgumentParser(description="DQN with Valid Action Mask for 2048")
    parser.add_argument("--env_name", type=str, default="gymnasium_2048/TwentyFortyEight-v0", help="环境名称")
    parser.add_argument("--episodes", type=int, default=1000, help="训练episode数量")  # 从3000增加到5000
    parser.add_argument("--max_steps", type=int, default=2000, help="每个episode最大步数")
    parser.add_argument("--lr", type=float, default=7e-4, help="学习率")  # 从3e-4提高到5e-4
    parser.add_argument("--gamma", type=float, default=0.92, help="折扣因子")  # 从0.95降低到0.92
    parser.add_argument("--buffer_size", type=int, default=50000, help="经验缓冲区大小")  # 从100000减少到50000
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")  # 从128减少到64
    parser.add_argument("--target_update_freq", type=int, default=500, help="目标网络更新频率")  # 从200降低到100
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="初始epsilon值")
    parser.add_argument("--epsilon_end", type=float, default=0.03, help="最终epsilon值")  # 从0.05降低到0.02
    parser.add_argument("--epsilon_decay", type=float, default=0.9998, help="epsilon衰减率")  # 从0.9995调整到0.9997
    parser.add_argument("--save_freq", type=int, default=500, help="模型保存频率")  # 从200增加到500
    parser.add_argument("--eval_freq", type=int, default=500, help="评估频率")  # 减少评估频率
    parser.add_argument("--test_model", type=str, default="", help="测试已保存的模型路径")
    parser.add_argument("--test_episodes", type=int, default=10, help="测试时的episode数量")
    parser.add_argument("--render_test", action="store_true", help="测试时渲染环境")
    parser.add_argument("--seed", type=int, default=42, help="测试时使用的随机种子")
    parser.add_argument("--render_demo", action="store_true", help="演示时渲染环境")
    parser.add_argument("--save_plot", action="store_true", help="保存训练曲线图")
    parser.add_argument("--use_k_step", action="store_true", default=True, help="使用K步前瞻")
    parser.add_argument("--k_step", type=int, default=15, help="K步前瞻步数（测试/验证时使用）")
    parser.add_argument("--num_simulations", type=int, default=25, help="蒙特卡洛模拟次数（测试/验证时使用）")

    # 训练时专用的k步搜索参数
    parser.add_argument("--use_k_step_train", type=bool, default=True, help="训练时使用K步前瞻")
    parser.add_argument("--k_step_train", type=int, default=2, help="训练时K步前瞻步数（浅搜索）")
    parser.add_argument("--num_simulations_train", type=int, default=4, help="训练时蒙特卡洛模拟次数（浅搜索）")
    
    # Softmax探索温度参数
    parser.add_argument("--k_step_temperature", type=float, default=0.3, help="K步搜索中的softmax温度参数")
    args = parser.parse_args()
    
    # 执行训练逻辑
    _execute_training(args)


def _execute_training(args):
    """执行训练逻辑的核心函数"""
    # 导入gymnasium_2048
    if 'gymnasium_2048' in args.env_name:
        import gymnasium_2048
    
    # 创建环境
    env = gym.make(args.env_name)
    
    # 获取状态和动作空间信息
    if len(env.observation_space.shape) == 3:  # gymnasium-2048使用4x4x16
        state_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    elif len(env.observation_space.shape) == 2:  # gym-2048使用4x4网格
        state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print("=" * 60)
    print("DQN with Valid Action Mask Training Parameters:")
    print(f"Environment: {args.env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Episodes: {args.episodes}")
    print(f"Learning Rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print(f"Buffer Size: {args.buffer_size}, Batch Size: {args.batch_size}")
    print(f"Target Update Frequency: {args.target_update_freq}")
    print(f"Epsilon: {args.epsilon_start} -> {args.epsilon_end} (decay: {args.epsilon_decay})")
    print(f"K-step Temperature: {args.k_step_temperature}")
    print("=" * 60)
    
    # 创建智能体
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        use_cuda=True,
        use_k_step_lookahead=args.use_k_step,  # 测试/验证时启用K步前瞻
        k_step=args.k_step,                   # 测试/验证时K步数
        num_simulations=args.num_simulations,  # 测试/验证时蒙特卡洛模拟次数
        use_k_step_lookahead_train=args.use_k_step_train,  # 训练时启用K步前瞻
        k_step_train=args.k_step_train,       # 训练时K步数（浅搜索）
        num_simulations_train=args.num_simulations_train,  # 训练时蒙特卡洛模拟次数（浅搜索）
        k_step_temperature=args.k_step_temperature  # K步搜索中的softmax温度参数
    )
    
    # 检查是否需要测试已保存的模型
    if args.test_model:
        print(f"正在加载模型: {args.test_model}")
        agent.load_model(args.test_model)
        print("模型加载完成，开始测试...")
        
        # 直接进行测试，使用指定的种子
        test_agent(agent, args.env_name, episodes=args.test_episodes, render=args.render_test, seed=args.seed)
        env.close()
        return agent, []
    
    # 初始化记录
    episode_rewards = []
    losses = []
    eval_rewards = []
    best_reward = -np.inf
    
    print("开始DQN训练...")
    start_time = time.time()
    
    # 训练循环
    for episode in range(args.episodes):
        # 训练一个episode
        episode_reward, episode_loss = agent.train_episode(
            env, max_steps=args.max_steps
        )
        
        # 记录结果
        episode_rewards.append(episode_reward)
        losses.append(episode_loss)
        
        # 更新最佳奖励
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # 在500 episode时清空经验池并切换奖励模式
        # if (episode + 1) == 500:
        #     print(f"【里程碑】Episode 500: 清空经验池并切换到原始奖励模式")
        #     agent.replay_buffer.clear()
        #     agent.use_enhanced_reward = False  # 切换到原始奖励模式
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward_last_10 = np.mean(episode_rewards[-10:])
            reward_mode = "增强" if getattr(agent, 'use_enhanced_reward', True) else "原始"
            print(f"Episode {episode + 1}/{args.episodes}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Avg(last 10): {avg_reward_last_10:.2f}, "
                  f"Loss: {episode_loss:.4f}, "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"奖励模式: {reward_mode}")
        
        # 定期评估
        if (episode + 1) % args.eval_freq == 0:
            eval_env = gym.make(args.env_name)
            avg_eval_reward, std_eval_reward = agent.evaluate(eval_env, episodes=1, use_k_step=args.use_k_step)
            eval_rewards.append(avg_eval_reward)
            eval_env.close()
            
            print(f"【评估】Episode {episode + 1}: "
                  f"Avg Reward = {avg_eval_reward:.2f} ± {std_eval_reward:.2f}")
        
        # 定期保存模型
        if (episode + 1) % args.save_freq == 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_dir = "models_discrete"
            os.makedirs(model_dir, exist_ok=True)
            safe_env_name = args.env_name.replace("/", "_").replace(":", "_")
            model_path = os.path.join(model_dir, f"dqn_valid_mask_{safe_env_name}_{timestamp}_ep{episode+1}.pth")
            agent.save_model(model_path)
    
    # 训练完成
    training_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Total training time: {training_time:.2f}s")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    print("=" * 60)
    
    # 保存最终模型
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_dir = "models_discrete"
    os.makedirs(model_dir, exist_ok=True)
    safe_env_name = args.env_name.replace("/", "_").replace(":", "_")
    final_model_path = os.path.join(model_dir, f"dqn_valid_mask_{safe_env_name}_final_{timestamp}.pth")
    agent.save_model(final_model_path)
    
    # 绘制训练结果
    plot_training_results(episode_rewards, losses, args.env_name, show_plot=args.save_plot)
    
    # 演示训练好的智能体
    demo_agent(agent, args.env_name, episodes=3, render=args.render_demo)
    
    env.close()
    return agent, episode_rewards


if __name__ == '__main__':
    train_dqn_with_args()