import os
import time
import numpy as np
import matplotlib.pyplot as plt


def plot_training_results(episode_rewards, losses, env_name, show_plot=False):
    """
    绘制训练结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 计算移动平均
    window_size = min(50, len(episode_rewards))
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        moving_avg_x = range(window_size-1, len(episode_rewards))
    else:
        moving_avg = []
        moving_avg_x = []
    
    # 1. Reward curve
    axes[0, 0].plot(episode_rewards, alpha=0.3, color='blue', label='Raw Reward')
    if len(moving_avg) > 0:
        axes[0, 0].plot(moving_avg_x, moving_avg, color='red', linewidth=2, label=f'Moving Average (window={window_size})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title(f'DQN with Valid Action Mask - {env_name}\nTraining Reward Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Loss curve
    axes[0, 1].plot(losses, color='orange')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Reward distribution
    axes[1, 0].hist(episode_rewards, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(np.mean(episode_rewards), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(episode_rewards):.2f}')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Recent performance
    recent_episodes = 100
    if len(episode_rewards) >= recent_episodes:
        recent_rewards = episode_rewards[-recent_episodes:]
        axes[1, 1].plot(recent_rewards, color='green')
        axes[1, 1].axhline(np.mean(recent_rewards), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(recent_rewards):.2f}')
        axes[1, 1].set_xlabel(f'Last {recent_episodes} Episodes')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].set_title(f'Recent Performance (Last {recent_episodes} Episodes)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_dir = "plots_discrete"
    os.makedirs(plot_dir, exist_ok=True)
    safe_env_name = env_name.replace("/", "_").replace(":", "_")
    plot_path = os.path.join(plot_dir, f"dqn_valid_mask_training_{safe_env_name}_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {plot_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()