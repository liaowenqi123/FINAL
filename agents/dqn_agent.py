import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from models.dqn_network import EnhancedDQNNetWithMask
from utils.replay_buffer import ReplayBuffer
from game.game_2048 import check_valid_actions_2048, simulate_move_2048


class DQNAgent:
    """
    DQN智能体（带动作掩码）
    专门用于2048游戏的强化学习
    """
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, 
                 buffer_size=100000, batch_size=64, target_update_freq=1000,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 use_cuda=True, use_k_step_lookahead=False, k_step=3, num_simulations=5,
                 use_k_step_lookahead_train=False, k_step_train=2, num_simulations_train=3,
                 k_step_temperature=0.3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.update_count = 0

        
        # K步前瞻参数（新增部分）
        self.use_k_step_lookahead = use_k_step_lookahead
        self.k_step = k_step
        self.num_simulations = num_simulations
        self.k_step_cache = {}  # 缓存K步前瞻结果
        
        # 训练时专用的K步前瞻参数
        self.use_k_step_lookahead_train = use_k_step_lookahead_train
        self.k_step_train = k_step_train
        self.num_simulations_train = num_simulations_train
        self.k_step_cache_train = {}  # 训练时专用的K步前瞻缓存
        
        # Softmax探索温度参数
        self.k_step_temperature = k_step_temperature  # K步搜索中的温度参数，<1偏向利用，>1偏向探索
        
        # 自适应深度策略
        self.base_k_step = k_step
        self.adaptive_depth = False  # 是否启用自适应深度
        
        # 软更新参数（新增部分）
        self.tau = 0.001  # 软更新参数
        # 创建主网络和目标网络（修改为增强版本）
        self.q_network = EnhancedDQNNetWithMask(state_dim, action_dim)
        self.target_network = EnhancedDQNNetWithMask(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 奖励模式标志
        self.use_enhanced_reward = True  # 默认使用增强奖励
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 设备选择
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"使用设备: {self.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")
        else:
            self.device = torch.device("cpu")
            print(f"使用设备: {self.device}")
        self.q_network.to(self.device)
        self.target_network.to(self.device)

    def get_action(self, state, env=None, deterministic=False, use_k_step=False, is_training=False):
        """
        根据当前状态选择动作（带动作掩码和K步前瞻）
        """
        # 将网络设置为eval模式
        original_train_mode = self.q_network.training
        self.q_network.eval()
    
        try:
            # 获取有效动作掩码
            if env is not None and hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'board'):
                try:
                    board = env.unwrapped.board.copy()
                    valid_actions = check_valid_actions_2048(board)
                    action_mask = torch.FloatTensor(valid_actions).to(self.device)
                    valid_indices = np.where(valid_actions == 1)[0]
                except:
                    action_mask = torch.ones(self.action_dim).to(self.device)
                    valid_indices = np.arange(self.action_dim)
            else:
                action_mask = torch.ones(self.action_dim).to(self.device)
                valid_indices = np.arange(self.action_dim)
        
            # 状态预处理
            if isinstance(state, np.ndarray):
                if len(state.shape) == 3:  # 4x4x16
                    state = state.flatten()
                elif len(state.shape) == 2:  # 4x4
                    state = state.flatten()
        
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
            # 使用K步前瞻，但仅在测试和评估时启用
            if use_k_step and env is not None and deterministic and len(valid_indices) > 0 and not is_training:
                # 对每个有效动作进行K步前瞻评估
                action_values = []
            
                for action in valid_indices:
                    # 获取基础Q值
                    with torch.no_grad():
                        q_values = self.q_network(state_tensor, action_mask)
                        base_q_value = q_values[0, action].item()
                
                    # 根据训练场景获取K步前瞻评估值
                    if is_training:
                        k_step_value = self.k_step_lookahead_evaluation_train(state, action, env)
                    else:
                        k_step_value = self.k_step_lookahead_evaluation(state, action, env)
                
                    # 结合Q值和K步前瞻值（可调整权重）
                    combined_value = 0.3 * base_q_value + 0.7 * k_step_value
                    action_values.append(combined_value)
            
                # 选择价值最高的动作
                best_idx = np.argmax(action_values)
                return valid_indices[best_idx]
        
            # 原始动作选择逻辑
            with torch.no_grad():
                q_values = self.q_network(state_tensor, action_mask)
            
                if deterministic:
                    # 选择有效动作中Q值最大的动作
                    masked_q_values = q_values * action_mask
                    action = torch.argmax(masked_q_values, dim=-1).item()
                else:
                    # epsilon-贪婪策略
                    if np.random.random() < self.epsilon:
                        # 以epsilon概率随机选择一个有效动作
                        valid_indices_tensor = torch.nonzero(action_mask).squeeze(-1)
                        if len(valid_indices_tensor) > 0:
                            action = valid_indices_tensor[torch.randint(len(valid_indices_tensor), (1,))].item()
                        else:
                            action = 0
                    else:
                        # 选择有效动作中Q值最大的动作
                        masked_q_values = q_values * action_mask
                        action = torch.argmax(masked_q_values, dim=-1).item()
        
            return action
        finally:
            # 恢复原来的训练模式
            self.q_network.train(original_train_mode)

    def augment_state_action(self, state, action, reward, next_state, done, action_mask):
        """
        简化的数据增强：只处理4x4游戏板的旋转和翻转
        
        Returns:
            List of (state, action, reward, next_state, done, action_mask) tuples
        """
        augmented_samples = []
        
        # 尝试从环境中获取4x4游戏板
        try:
            # 检查状态是否是4x4游戏板
            if hasattr(state, 'shape') and len(state.shape) == 2 and state.shape == (4, 4):
                board = state.copy()
                next_board = next_state.copy()
            else:
                # 如果不是4x4格式，直接返回原样本
                return [(state, action, reward, next_state, done, action_mask)]
        except:
            return [(state, action, reward, next_state, done, action_mask)]
        
        # 定义8种变换和对应的动作映射
        transformations = [
            # (变换函数, 动作映射)
            (lambda x: x, {0:0, 1:1, 2:2, 3:3}),  # 原始
            (lambda x: np.rot90(x, k=1), {0:1, 1:2, 2:3, 3:0}),  # 旋转90度
            (lambda x: np.rot90(x, k=2), {0:2, 1:3, 2:0, 3:1}),  # 旋转180度
            (lambda x: np.rot90(x, k=3), {0:3, 1:0, 2:1, 3:2}),  # 旋转270度
            (lambda x: np.fliplr(x), {0:1, 1:0, 2:3, 3:2}),  # 水平翻转
            (lambda x: np.flipud(x), {0:2, 1:3, 2:0, 3:1}),  # 垂直翻转
            (lambda x: np.transpose(x), {0:0, 1:3, 2:2, 3:1}),  # 主对角线翻转
            (lambda x: np.fliplr(np.rot90(x, k=1)), {0:2, 1:1, 2:0, 3:3}),  # 副对角线翻转
        ]
        
        for transform_func, action_mapping in transformations:
            # 变换游戏板
            transformed_board = transform_func(board)
            transformed_next_board = transform_func(next_board)
            
            # 映射动作
            transformed_action = action_mapping[action]
            
            # 变换动作掩码
            if action_mask is not None:
                transformed_action_mask = np.array([action_mask[action_mapping[i]] for i in range(4)])
            else:
                transformed_action_mask = action_mask
            
            augmented_samples.append((
                transformed_board, 
                transformed_action, 
                reward, 
                transformed_next_board, 
                done, 
                transformed_action_mask
            ))
        
        return augmented_samples

    def store_transition(self, state, action, reward, next_state, done, action_mask=None):
        """
        存储经验到回放缓冲区（带数据增强）
        """
        # 生成增强样本
        augmented_samples = self.augment_state_action(state, action, reward, next_state, done, action_mask)
        
        # 存储所有增强样本
        for sample in augmented_samples:
            aug_state, aug_action, aug_reward, aug_next_state, aug_done, aug_action_mask = sample
            self.replay_buffer.push(aug_state, aug_action, aug_reward, aug_next_state, aug_done, aug_action_mask)

    def update(self):
        """
        更新Q网络（优化奖励计算，使用Double DQN和软更新）
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 从缓冲区采样
        states, actions, rewards, next_states, dones, action_masks = self.replay_buffer.sample(self.batch_size)
        
        # 状态预处理
        if len(states.shape) > 2:  # 3D数组
            states = states.reshape(states.shape[0], -1)
        if len(next_states.shape) > 2:  # 3D数组
            next_states = next_states.reshape(next_states.shape[0], -1)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        action_masks = torch.FloatTensor(action_masks).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states, action_masks).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值（简化版，不使用Double DQN）
        with torch.no_grad():
            # 直接使用目标网络计算最大Q值
            next_q_values = self.target_network(next_states, action_masks).max(dim=1)[0]
            
            # 计算目标Q值
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # 使用Huber损失
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=0.5)  # 更严格的梯度裁剪
        self.optimizer.step()
        
        # 软更新目标网络（新增部分）
        self.update_count += 1
        # 每100次更新才做一次软更新
        if self.update_count % 100 == 0:
            for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        # 衰减epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def get_adaptive_depth(self, board):
        """
        根据游戏状态自适应调整搜索深度
        """
        if not self.adaptive_depth:
            return self.k_step
        
        max_tile = np.max(board)
        empty_cells = np.sum(board == 0)
        
        if max_tile <= 64:
            return max(self.k_step, 2)
        elif max_tile <= 256:
            return max(self.k_step, 4)
        elif max_tile <= 1024:
            return max(self.k_step, 8)
        else:
            return max(self.k_step, 16)
    
    def k_step_lookahead_evaluation(self, state, action, env=None):
        """
        多步探索K步前瞻评估：使用智能探索策略
        """
        # 如果禁用K步前瞻或没有环境，直接返回0
        if not self.use_k_step_lookahead or env is None:
            return 0
        
        # 获取当前游戏板状态
        if not hasattr(env, 'unwrapped') or not hasattr(env.unwrapped, 'board'):
            return 0
        
        current_board = env.unwrapped.board.copy()
        
        # 缓存键
        state_key = str(current_board.flatten()) + f"_a{action}"
        if state_key in self.k_step_cache:
            return self.k_step_cache[state_key]
        
        # 多步探索参数
        n_steps = 3  # 探索步数
        total_explorations = 100  # 总探索次数
        k = 3  # 每步考虑的top k个动作
        
        all_sequence_values = []
        
        for sim in range(self.num_simulations):
            # 复制当前游戏板
            board = current_board.copy()
            sequence_value = 0
            
            # 模拟执行当前动作
            new_board, immediate_reward = simulate_move_2048(board, action)
            sequence_value += immediate_reward
            
            # 如果移动无效，直接返回惩罚值
            if np.array_equal(new_board, board):
                all_sequence_values.append(sequence_value)
                continue
            
            # 多步前瞻探索
            for step in range(n_steps - 1):
                # 检查游戏是否可能结束
                if np.sum(new_board == 0) == 0:
                    # 检查是否还有合并可能
                    can_merge = False
                    for i in range(4):
                        for j in range(3):
                            if new_board[i, j] == new_board[i, j+1] and new_board[i, j] != 0:
                                can_merge = True
                                break
                            if new_board[j, i] == new_board[j+1, i] and new_board[j, i] != 0:
                                can_merge = True
                                break
                        if can_merge:
                            break
                    if not can_merge:
                        break
                
                # 获取有效动作
                valid_actions = check_valid_actions_2048(new_board)
                valid_indices = np.where(valid_actions == 1)[0]
                
                if len(valid_indices) == 0:
                    break
                
                # 使用多步探索选择动作
                if len(new_board.shape) == 2:
                    state_input = new_board.flatten()
                    # 对于2048游戏，需要将4x4网格转换为256维的one-hot编码
                    if len(state_input) != self.state_dim:
                        # 创建256维的one-hot编码
                        one_hot_state = np.zeros(256, dtype=np.float32)
                        for i in range(16):
                            tile_value = int(state_input[i])
                            if tile_value > 0:
                                # 计算log2值，因为2048游戏中的瓦片值是2的幂
                                log_value = int(np.log2(tile_value))
                                if log_value < 16:  # 16个不同的值（1, 2, 4, ..., 65536）
                                    one_hot_state[i * 16 + log_value] = 1.0
                        state_input = one_hot_state
                else:
                    state_input = new_board.flatten()
                
                # 使用Q网络和多步探索选择动作
                state_tensor = torch.FloatTensor(state_input).unsqueeze(0).to(self.device)
                action_mask_tensor = torch.FloatTensor(valid_actions).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.q_network(state_tensor, action_mask_tensor)
                    q_values_np = q_values.cpu().numpy()[0]
                    next_action = self.multi_step_exploration(q_values_np, valid_indices, n_steps, total_explorations, k)
                
                board = new_board.copy()
                new_board, step_reward = simulate_move_2048(board, next_action)
                # 归一化奖励
                normalized_step_reward = np.tanh(step_reward / 1000.0)
                sequence_value += normalized_step_reward * (self.gamma ** (step + 1))
            
            all_sequence_values.append(sequence_value)
        
        # 计算所有序列的平均值
        if len(all_sequence_values) > 0:
            avg_value = np.mean(all_sequence_values)
        else:
            avg_value = 0
        
        # 缓存结果
        self.k_step_cache[state_key] = avg_value
        
        # 限制缓存大小
        if len(self.k_step_cache) > 1000:
            # 移除最旧的条目
            keys = list(self.k_step_cache.keys())
            for key in keys[:100]:
                del self.k_step_cache[key]
        
        return avg_value
    
    def k_step_lookahead_evaluation_train(self, state, action, env=None):
        """
        训练时专用的K步前瞻评估：使用较浅的搜索深度和较少的模拟次数
        """
        # 如果禁用K步前瞻或没有环境，直接返回0
        if not self.use_k_step_lookahead_train or env is None:
            return 0
        
        # 获取当前游戏板状态
        if not hasattr(env, 'unwrapped') or not hasattr(env.unwrapped, 'board'):
            return 0
        
        current_board = env.unwrapped.board.copy()
        
        # 缓存键
        state_key = str(current_board.flatten()) + f"_train_a{action}"
        if state_key in self.k_step_cache_train:
            return self.k_step_cache_train[state_key]
        
        total_values = []
        
        # 使用较少的蒙特卡洛模拟次数（训练时）
        total_values = []
        
        for sim in range(self.num_simulations_train):
            # 复制当前游戏板
            board = current_board.copy()
            total_value = 0
            
            # 模拟执行当前动作
            new_board, immediate_reward = simulate_move_2048(board, action)
            # 归一化奖励
            normalized_reward = np.tanh(immediate_reward / 1000.0)
            total_value += normalized_reward
            
            # 如果移动无效，直接返回惩罚值
            if np.array_equal(new_board, board):
                total_values.append(total_value)
                continue
            
            # 使用较浅的搜索深度（训练时）
            for step in range(self.k_step_train - 1):
                # 检查游戏是否可能结束
                if np.sum(new_board == 0) == 0:
                    # 检查是否还有合并可能
                    can_merge = False
                    for i in range(4):
                        for j in range(3):
                            if new_board[i, j] == new_board[i, j+1] and new_board[i, j] != 0:
                                can_merge = True
                                break
                            if new_board[j, i] == new_board[j+1, i] and new_board[j, i] != 0:
                                can_merge = True
                                break
                        if can_merge:
                            break
                    if not can_merge:
                        break
                
                # 获取有效动作
                valid_actions = check_valid_actions_2048(new_board)
                valid_indices = np.where(valid_actions == 1)[0]
                
                if len(valid_indices) == 0:
                    break
                
                # 选择下一动作：始终使用agent的最优决策（已经包含掩码）
                # 训练时也使用最优决策，探索性通过epsilon-贪婪在主训练循环中处理
                # 将状态转换为网络输入格式
                if len(new_board.shape) == 2:
                    state_input = new_board.flatten()
                    # 对于2048游戏，需要将4x4网格转换为256维的one-hot编码
                    if len(state_input) != self.state_dim:
                        # 创建256维的one-hot编码
                        one_hot_state = np.zeros(256, dtype=np.float32)
                        for i in range(16):
                            tile_value = int(state_input[i])
                            if tile_value > 0:
                                # 计算log2值，因为2048游戏中的瓦片值是2的幂
                                log_value = int(np.log2(tile_value))
                                if log_value < 16:  # 16个不同的值（1, 2, 4, ..., 65536）
                                    one_hot_state[i * 16 + log_value] = 1.0
                        state_input = one_hot_state
                else:
                    state_input = new_board.flatten()
                
# 使用Q网络和多步探索选择动作
                state_tensor = torch.FloatTensor(state_input).unsqueeze(0).to(self.device)
                action_mask_tensor = torch.FloatTensor(valid_actions).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.q_network(state_tensor, action_mask_tensor)
                    q_values_np = q_values.cpu().numpy()[0]
                    # 使用多步探索选择动作
                    next_action = self.multi_step_exploration(q_values_np, valid_indices, n_steps=3, total_explorations=100, k=3)
                board = new_board.copy()
                new_board, step_reward = simulate_move_2048(board, next_action)
                # 归一化奖励
                normalized_step_reward = np.tanh(step_reward / 1000.0)
                total_value += normalized_step_reward * (self.gamma ** (step + 1))
            
            total_values.append(total_value)
        
        # 混合策略：主要用mean，但给max一定权重
        if len(total_values) > 0:
            avg_value = np.mean(total_values)
            max_value = np.max(total_values)
            # 80%期望 + 20%最乐观情况
            fraction=(16-self.k_step_train)**2/256
            combined_value = (1-fraction) * avg_value + fraction * max_value
        else:
            combined_value = 0
        
        # 缓存结果
        self.k_step_cache_train[state_key] = combined_value
        
        # 限制缓存大小
        if len(self.k_step_cache_train) > 500:  # 训练时使用较小的缓存
            # 移除最旧的条目
            keys = list(self.k_step_cache_train.keys())
            for key in keys[:50]:
                del self.k_step_cache_train[key]
        
        return avg_value
    
    def calculate_enhanced_reward(self, state, next_state, base_reward, env):
        """
        重新设计的奖励函数 - 以"活着"和"发展"为目标
        """
        enhanced_reward = 0.0  # 不依赖原始奖励

        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'board'):
            try:
                board = env.unwrapped.board.copy()
                
                # 1. 生存奖励 - 基础分
                enhanced_reward += 1.0  # 每步都给基础分，鼓励"活着"
                
                # 2. 空位奖励 - 最重要的生存指标
                empty_cells = np.sum(board == 0)
                enhanced_reward += empty_cells * 0.8  # 空位越多越好
                
                # 3. 最大瓦片奖励 - 使用对数，避免极端值
                max_tile = np.max(board)
                if max_tile > 1:
                    enhanced_reward += np.log2(max_tile) * 0.5  # 对数奖励
                
                # 4. 单调性奖励 - 鼓励有序排列
                monotonic_bonus = 0.0
                # 行单调性
                for i in range(4):
                    row = board[i, :]
                    if all(row[j] <= row[j+1] for j in range(3)) or all(row[j] >= row[j+1] for j in range(3)):
                        monotonic_bonus += 0.2
                # 列单调性  
                for j in range(4):
                    col = board[:, j]
                    if all(col[i] <= col[i+1] for i in range(3)) or all(col[i] >= col[i+1] for i in range(3)):
                        monotonic_bonus += 0.2
                
                enhanced_reward += monotonic_bonus
                
                # 5. 角落奖励 - 大瓦片在角落
                max_tile_pos = np.unravel_index(np.argmax(board), board.shape)
                corners = [(0,0), (0,3), (3,0), (3,3)]
                if max_tile_pos in corners and max_tile >= 64:
                    enhanced_reward += 1.0
                
                # 6. 合并奖励 - 但用对数压缩
                if base_reward > 0:
                    enhanced_reward += np.log1p(base_reward) * 0.3  # log1p避免log(0)
                
            except:
                enhanced_reward = 1.0  # 出错时给基础分
    
        return enhanced_reward

    def calculate_smoothness(self, board):
        """计算平滑度（相邻瓦片值的差异）"""
        smoothness = 0
        for i in range(4):
            for j in range(4):
                if i < 3 and board[i, j] > 0 and board[i+1, j] > 0:
                    diff = abs(np.log2(board[i, j]) - np.log2(board[i+1, j]))
                    smoothness += diff
                if j < 3 and board[i, j] > 0 and board[i, j+1] > 0:
                    diff = abs(np.log2(board[i, j]) - np.log2(board[i, j+1]))
                    smoothness += diff
        return smoothness
    
    def multi_step_exploration(self, q_values, valid_indices, n_steps=3, total_explorations=100, k=3):
        """
        多步探索：探索n步动作序列，根据e-greedy分配探索次数
        
        参数:
        q_values: Q值数组
        valid_indices: 有效动作索引
        n_steps: 探索步数
        total_explorations: 总探索次数
        k: 每步考虑的top k个动作
        
        返回:
        选择的动作索引
        """
        if len(valid_indices) == 0:
            return 0
        
        # 如果有效动作数小于等于k，简化处理
        if len(valid_indices) <= k:
            return valid_indices[np.argmax(q_values[valid_indices])]
        
        # 获取top k个动作及其Q值
        valid_q_values = q_values[valid_indices]
        top_k_indices = np.argsort(valid_q_values)[-k:]
        top_k_valid_indices = [valid_indices[i] for i in top_k_indices]
        top_k_q_values = [valid_q_values[i] for i in top_k_indices]
        
        # 计算e-greedy概率分布
        q_values_array = np.array(top_k_q_values)
        
        # 应用e-greedy：epsilon概率随机，1-epsilon概率按Q值加权
        epsilon = 0.3  # 探索概率
        if np.random.random() < epsilon:
            # 随机选择（探索）
            probabilities = np.ones(k) / k
        else:
            # 按Q值加权（利用）
            # 使用softmax进行软选择
            exp_q = np.exp(q_values_array - np.max(q_values_array))
            probabilities = exp_q / np.sum(exp_q)
        
        # 选择动作
        chosen_idx = np.random.choice(len(top_k_valid_indices), p=probabilities)
        return top_k_valid_indices[chosen_idx]
    
    def train_episode(self, env, max_steps=1000, update_freq=1, render=False, episode_num=0, total_episodes=1000):
        """
        改进的训练策略
        """
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        update_count = 0

        for step_count in range(max_steps):
            # 动态调整学习策略
            if episode_num < total_episodes * 0.2:  # 前期缩短到20%
                # 前期：更多探索
                exploration_multiplier = 2.0  # 从1.5提高到2.0
            elif episode_num < total_episodes * 0.6:  # 中期调整到60%
                # 中期：平衡探索利用
                exploration_multiplier = 1.0
            else:
            # 后期：更多利用
                exploration_multiplier = 0.3  # 从0.5降低到0.3
         # 动态epsilon策略
            current_epsilon = max(self.epsilon * exploration_multiplier, self.epsilon_end)
            
            # 选择动作
            if np.random.random() < current_epsilon:
                # 随机选择有效动作
                if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'board'):
                    board = env.unwrapped.board.copy()
                    valid_actions = check_valid_actions_2048(board)
                    valid_indices = np.where(valid_actions == 1)[0]
                    if len(valid_indices) > 0:
                        # 优先选择能产生合并的动作
                        action = self.prefer_merge_action(board, valid_indices)
                    else:
                        action = 0
                else:
                    action = env.action_space.sample()
            else:
                # 使用网络选择动作（暂时禁用k步搜索）
                action = self.get_action(state, env=env, deterministic=True, use_k_step=False, is_training=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 获取动作掩码
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'board'):
                try:
                    board = env.unwrapped.board.copy()
                    action_mask = check_valid_actions_2048(board)
                except:
                    action_mask = np.ones(self.action_dim)
            else:
                action_mask = np.ones(self.action_dim)
            
            # 根据模式选择奖励函数
            if getattr(self, 'use_enhanced_reward', True):
                enhanced_reward = self.calculate_enhanced_reward(state, next_state, reward, env)
            else:
                enhanced_reward = reward  # 使用原始奖励
            
            # 存储经验
            self.store_transition(state, action, enhanced_reward, next_state, done, action_mask)
            
            # 控制更新频率 - 每4步更新一次
            if len(self.replay_buffer) >= self.batch_size and step_count % 4 == 0:
                loss = self.update()
                if loss is not None:
                    episode_loss += loss
                    update_count += 1
            
            state = next_state
            episode_reward += reward
            
            if done:
                break

    # 更平缓的epsilon衰减
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        avg_loss = episode_loss / max(update_count, 1)
        
        return episode_reward, avg_loss

    def prefer_merge_action(self, board, valid_indices):
        """在随机探索时优先选择能产生合并的动作"""
        # 对每个有效动作进行简单评估
        action_scores = []
        for action in valid_indices:
            # 模拟移动
            new_board, reward = simulate_move_2048(board, action)
            # 基础分数：合并奖励 + 空位增加
            score = reward
            if reward > 0:  # 如果能合并
                score += 10  # 额外奖励
            action_scores.append(score)
    
    # 选择分数最高的动作
        if len(action_scores) > 0 and max(action_scores) > 0:
            return valid_indices[np.argmax(action_scores)]
        else:
            return np.random.choice(valid_indices)

    def evaluate(self, env, episodes=10, max_steps=1000, render=False, use_k_step=True):
        """
        评估智能体性能（添加K步前瞻选项）
        """
        total_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            step_count = 0
            
            while step_count < max_steps:
                if render:
                    env.render()
                
                # 使用K步前瞻选择动作
                action = self.get_action(state, env=env, deterministic=True, use_k_step=use_k_step, is_training=False)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
                step_count += 1
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
        
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        return avg_reward, std_reward

    def save_model(self, filepath):
        """
        保存模型
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma
        }, filepath)
        print(f"模型已保存到: {filepath}")

    def load_model(self, filepath):
        """
        加载模型
        """
        if not os.path.exists(filepath):
            print(f"模型文件不存在: {filepath}")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        print(f"模型已从 {filepath} 加载")