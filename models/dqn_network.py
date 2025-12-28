import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedDQNNetWithMask(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(EnhancedDQNNetWithMask, self).__init__()
        
        # 假设输入是4x4x16的one-hot编码
        # 第一层卷积：保持尺寸 (padding=1)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # 第二层卷积：缩小尺寸 (不使用padding)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        
        # 计算卷积后的尺寸：4x4 -> 4x4 -> 2x2
        conv_output_size = 64 * 2 * 2  # 64 channels * 2x2 spatial size
        
        # 全连接层
        self.fc1 = nn.Linear(conv_output_size, 64)
        
        # 输出层
        self.fc2 = nn.Linear(64, action_dim)
        
        # 更好的初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0.01)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0.01)

    def forward(self, state, action_mask=None):
        # 输入状态重塑为4x4x16
        batch_size = state.shape[0]
        x = state.view(batch_size, 16, 4, 4)  # (batch, channels, height, width)
        
        # 第一层卷积（保持4x4）
        x = F.relu(self.conv1(x))
        
        # 第二层卷积（缩小到2x2）
        x = F.relu(self.conv2(x))
        
        # 展平
        x = x.view(batch_size, -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)

        if action_mask is not None:
            # 确保action_mask是正确的类型和形状
            action_mask = action_mask.float()  # 转换为浮点类型
            # 使用乘法而不是masked_fill，避免极端值
            q_values = q_values * action_mask + (1 - action_mask) * (-1.0)

        return q_values