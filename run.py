#!/usr/bin/env python3
"""
DQN 2048游戏训练项目主入口文件
使用方法:
    python run.py train    # 开始训练
    python run.py test     # 测试已训练的模型
    python run.py demo     # 演示已训练的模型
    python run.py help     # 显示帮助信息
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from train import train_dqn
from test import test_agent, demo_agent, test_dqn_2048
from config import DQNConfig, PathConfig


def create_directories():
    """创建必要的目录"""
    dirs_to_create = [
        PathConfig.MODEL_DIR,
        PathConfig.LOG_DIR,
        PathConfig.PLOT_DIR
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"目录已创建/存在: {dir_path}")


def train_mode():
    """训练模式"""
    print("=" * 60)
    print("DQN 2048游戏训练模式")
    print("=" * 60)
    
    # 创建必要的目录
    create_directories()
    
    # 调用训练函数
    train_dqn()


def test_mode(model_path=None):
    """测试模式"""
    print("=" * 60)
    print("DQN 2048游戏测试模式")
    print("=" * 60)
    
    if model_path is None:
        # 查找最新的模型文件
        model_dir = Path(PathConfig.MODEL_DIR)
        model_files = list(model_dir.glob("*.pth"))
        
        if not model_files:
            print("错误: 未找到任何模型文件!")
            print(f"请检查 {model_dir} 目录是否存在.pth文件")
            return
        
        # 按修改时间排序，选择最新的
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        model_path = str(model_files[0])
        print(f"使用最新模型: {model_path}")
    
    # 导入必要的模块
    from agents.dqn_agent import DQNAgent
    import gymnasium as gym
    import torch
    
    # 创建环境和智能体
    env = gym.make(DQNConfig.ENV_NAME)
    
    # 获取状态和动作空间信息
    if len(env.observation_space.shape) == 3:  # gymnasium-2048使用4x4x16
        state_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    elif len(env.observation_space.shape) == 2:  # gym-2048使用4x4网格
        state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 创建智能体
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=DQNConfig.LEARNING_RATE,
        gamma=DQNConfig.GAMMA,
        buffer_size=DQNConfig.BUFFER_SIZE,
        batch_size=DQNConfig.BATCH_SIZE,
        target_update_freq=DQNConfig.TARGET_UPDATE_FREQ,
        epsilon_start=DQNConfig.EPSILON_START,
        epsilon_end=DQNConfig.EPSILON_END,
        epsilon_decay=DQNConfig.EPSILON_DECAY,
        use_cuda=torch.cuda.is_available(),
        use_k_step_lookahead=DQNConfig.USE_K_STEP,
        k_step=DQNConfig.K_STEP,
        num_simulations=DQNConfig.NUM_SIMULATIONS,
        use_k_step_lookahead_train=DQNConfig.USE_K_STEP_TRAIN,
        k_step_train=DQNConfig.K_STEP_TRAIN,
        num_simulations_train=DQNConfig.NUM_SIMULATIONS_TRAIN
    )
    
    # 加载模型
    agent.load_model(model_path)
    
    # 测试智能体
    test_agent(agent, DQNConfig.ENV_NAME, episodes=DQNConfig.TEST_EPISODES, 
               render=DQNConfig.RENDER_TEST, seed=DQNConfig.SEED)
    
    env.close()


def demo_mode(model_path=None):
    """演示模式"""
    print("=" * 60)
    print("DQN 2048游戏演示模式")
    print("=" * 60)
    
    if model_path is None:
        # 查找最新的模型文件
        model_dir = Path(PathConfig.MODEL_DIR)
        model_files = list(model_dir.glob("*.pth"))
        
        if not model_files:
            print("错误: 未找到任何模型文件!")
            print(f"请检查 {model_dir} 目录是否存在.pth文件")
            return
        
        # 按修改时间排序，选择最新的
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        model_path = str(model_files[0])
        print(f"使用最新模型: {model_path}")
    
    # 导入必要的模块
    from agents.dqn_agent import DQNAgent
    import gymnasium as gym
    import torch
    
    # 创建环境和智能体
    env = gym.make(DQNConfig.ENV_NAME)
    
    # 获取状态和动作空间信息
    if len(env.observation_space.shape) == 3:  # gymnasium-2048使用4x4x16
        state_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    elif len(env.observation_space.shape) == 2:  # gym-2048使用4x4网格
        state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 创建智能体
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=DQNConfig.LEARNING_RATE,
        gamma=DQNConfig.GAMMA,
        buffer_size=DQNConfig.BUFFER_SIZE,
        batch_size=DQNConfig.BATCH_SIZE,
        target_update_freq=DQNConfig.TARGET_UPDATE_FREQ,
        epsilon_start=DQNConfig.EPSILON_START,
        epsilon_end=DQNConfig.EPSILON_END,
        epsilon_decay=DQNConfig.EPSILON_DECAY,
        use_cuda=torch.cuda.is_available(),
        use_k_step_lookahead=DQNConfig.USE_K_STEP,
        k_step=DQNConfig.K_STEP,
        num_simulations=DQNConfig.NUM_SIMULATIONS,
        use_k_step_lookahead_train=DQNConfig.USE_K_STEP_TRAIN,
        k_step_train=DQNConfig.K_STEP_TRAIN,
        num_simulations_train=DQNConfig.NUM_SIMULATIONS_TRAIN
    )
    
    # 加载模型
    agent.load_model(model_path)
    
    # 演示智能体
    demo_agent(agent, DQNConfig.ENV_NAME, episodes=3, render=DQNConfig.RENDER_DEMO)
    
    env.close()


def simple_test_mode():
    """简单测试模式 - 测试代码是否能正常运行"""
    print("=" * 60)
    print("DQN 2048游戏简单测试模式")
    print("=" * 60)
    
    test_dqn_2048()


def show_help():
    """显示帮助信息"""
    help_text = """
DQN 2048游戏训练项目

使用方法:
    python run.py <命令> [选项]

命令:
    train               开始训练DQN智能体
    test [model_path]   测试已训练的模型(可选指定模型路径)
    demo [model_path]   演示已训练的模型(可选指定模型路径)
    simple_test         简单测试代码是否能正常运行
    help                显示此帮助信息

示例:
    python run.py train                           # 开始训练
    python run.py test                            # 测试最新模型
    python run.py test models/model.pth            # 测试指定模型
    python run.py demo                            # 演示最新模型
    python run.py demo models/model.pth            # 演示指定模型
    python run.py simple_test                     # 简单测试

配置:
    所有配置参数都在 config.py 文件中定义，可以修改该文件来调整训练参数。

输出目录:
    - models_discrete/: 保存训练的模型文件
    - 2048_logs/: 保存测试和评估日志
    - plots_discrete/: 保存训练曲线图
    """
    print(help_text)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DQN 2048游戏训练项目", add_help=False)
    parser.add_argument('command', nargs='?', default='help', 
                       choices=['train', 'test', 'demo', 'simple_test', 'help'],
                       help='要执行的命令')
    parser.add_argument('model_path', nargs='?', default=None,
                       help='模型文件路径(用于test和demo命令)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_mode()
    elif args.command == 'test':
        test_mode(args.model_path)
    elif args.command == 'demo':
        demo_mode(args.model_path)
    elif args.command == 'simple_test':
        simple_test_mode()
    elif args.command == 'help':
        show_help()
    else:
        print(f"未知命令: {args.command}")
        show_help()


if __name__ == '__main__':
    main()