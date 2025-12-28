"""
DQN 2048游戏训练配置文件
包含所有可调参数的默认值
"""


class DQNConfig:
    """DQN训练配置"""
    
    # 环境配置
    ENV_NAME = "gymnasium_2048/TwentyFortyEight-v0"
    
    # 训练参数
    EPISODES = 5000
    MAX_STEPS = 2000
    LEARNING_RATE = 3e-4  # 降低学习率，提高稳定性
    GAMMA = 0.98  # 大幅降低，更关注短期奖励
    BUFFER_SIZE = 200000  # 增大缓冲区以适应8倍数据增强
    BATCH_SIZE = 128  # 增大批次大小以更好地利用数据增强
    TARGET_UPDATE_FREQ = 500
    
# 探索参数 - 更慢的衰减，更多探索
    EPSILON_START = 1.0
    EPSILON_END = 0.05  # 提高最终值，保持一定探索
    EPSILON_DECAY = 0.9998  # 更慢的衰减
    
    # 保存和评估参数
    SAVE_FREQ = 2500
    EVAL_FREQ = 10000
    TEST_EPISODES = 3
    
    # K步前瞻参数 - 降低测试时的计算量
    USE_K_STEP = True
    K_STEP = 0  # 从15降到8 #5
    NUM_SIMULATIONS = 0  # 从100降到20 #1000
    
# 训练时专用的K步前瞻参数
    USE_K_STEP_TRAIN = True  # 重新启用训练时的K步搜索
    K_STEP_TRAIN = 2  # 进一步降低训练时的深度
    NUM_SIMULATIONS_TRAIN = 50  # 进一步降低训练时的模拟次数
    
    # Softmax探索温度参数
    K_STEP_TEMPERATURE = 2  # <1偏向利用，>1偏向探索
    
    # 其他参数
    SEED = 42
    RENDER_TEST = True
    RENDER_DEMO = False
    SAVE_PLOT = False


class ModelConfig:
    """模型网络配置"""
    
    # 网络结构
    HIDDEN_UNITS = [384, 192, 96]
    
    # 初始化参数
    WEIGHT_INIT = "kaiming_uniform"
    BIAS_INIT = 0.01
    
    # 软更新参数
    TAU = 0.001


class TrainingConfig:
    """训练策略配置"""
    
    # 探索和利用的比例调整
    EXPLORATION_PHASE_RATIO = 0.2  # 前期探索阶段占比
    BALANCED_PHASE_RATIO = 0.6     # 中期平衡阶段占比
    EXPLOITATION_PHASE_RATIO = 0.2  # 后期利用阶段占比
    
    # 探索乘数
    EXPLORATION_MULTIPLIER = 2.0
    BALANCED_MULTIPLIER = 1.0
    EXPLOITATION_MULTIPLIER = 0.3


class RewardConfig:
    """奖励函数配置"""
    
    # 基础奖励权重
    BASE_REWARD_MULTIPLIER = 2.0
    
    # 空位奖励权重
    EMPTY_CELLS_WEIGHT = 8.0
    
    # 最大瓦片奖励权重
    MAX_TILE_WEIGHT = 0.15
    MAX_TILE_THRESHOLD = 64
    
    # 角落奖励
    CORNER_REWARD = 100
    
    # 单调性奖励权重
    MONOTONICITY_WEIGHT = 0.1
    
    # 归一化因子
    NORMALIZATION_FACTOR = 80.0


class PathConfig:
    """路径配置"""
    
    # 模型保存路径
    MODEL_DIR = "models_discrete"
    
    # 日志保存路径
    LOG_DIR = "2048_logs"
    
    # 图表保存路径
    PLOT_DIR = "plots_discrete"