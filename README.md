# DQN 2048游戏训练项目

这是一个使用深度Q网络(DQN)训练智能体来玩2048游戏的强化学习项目。项目支持动作掩码、K步前瞻搜索等高级特性。

## 项目结构

```
FINAL/
├── run.py                 # 主入口文件
├── config.py              # 配置文件
├── README.md              # 项目说明
├── __init__.py            # 项目初始化文件
├── models/                # 模型模块
│   ├── __init__.py
│   └── dqn_network.py     # DQN网络结构
├── agents/                # 智能体模块
│   ├── __init__.py
│   └── dqn_agent.py       # DQN智能体实现
├── utils/                 # 工具模块
│   ├── __init__.py
│   ├── replay_buffer.py   # 经验回放缓冲区
│   └── visualization.py   # 可视化工具
└── game/                  # 游戏模块
    ├── __init__.py
    └── game_2048.py       # 2048游戏逻辑
```

## 功能特性

- **动作掩码**: 只选择有效动作，提高学习效率
- **K步前瞻搜索**: 在测试时使用蒙特卡洛树搜索提高性能
- **Double DQN**: 减少Q值过高估计的问题
- **软更新**: 平滑更新目标网络
- **增强奖励函数**: 考虑空位、最大瓦片、角落位置等因素
- **动态探索策略**: 根据训练阶段调整探索程度

## 安装依赖

```bash
pip install torch gymnasium gymnasium-2048 numpy matplotlib
```

## 使用方法

### 1. 训练模型

```bash
python run.py train
```

### 2. 测试模型

```bash
# 测试最新模型
python run.py test

# 测试指定模型
python run.py test models_discrete/model.pth
```

### 3. 演示模型

```bash
# 演示最新模型
python run.py demo

# 演示指定模型
python run.py demo models_discrete/model.pth
```

### 4. 简单测试

```bash
python run.py simple_test
```

### 5. 查看帮助

```bash
python run.py help
```

## 配置参数

所有配置参数都在 `config.py` 文件中定义，包括：

- **训练参数**: 学习率、批次大小、训练回合数等
- **网络参数**: 隐藏层大小、初始化方法等
- **探索参数**: epsilon起始值、结束值、衰减率等
- **K步前瞻参数**: 搜索深度、模拟次数等
- **奖励函数参数**: 各项奖励的权重

## 输出文件

训练过程中会生成以下文件：

- `models_discrete/`: 保存训练的模型文件
- `2048_logs/`: 保存测试和评估日志
- `plots_discrete/`: 保存训练曲线图

## 性能指标

项目会记录以下性能指标：

- 每回合的奖励
- 最大瓦片值
- 达到2048的成功率
- 平均空位数量
- 训练损失

## 技术细节

### 网络结构

- 使用4层全连接网络：[384, 192, 96, action_dim]
- 使用ReLU激活函数
- Kaiming权重初始化

### 训练算法

- Double DQN
- 经验回放
- 软更新目标网络
- Huber损失函数

### 动作选择

- 训练时：epsilon-贪婪策略
- 测试时：确定性策略 + K步前瞻搜索

## 注意事项

1. 确保安装了所有必需的依赖包
2. 训练过程中会占用较多GPU内存
3. K步前瞻搜索会显著增加测试时间
4. 模型文件较大，注意磁盘空间

## 许可证

本项目仅供学习和研究使用。