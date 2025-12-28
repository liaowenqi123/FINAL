import numpy as np


def check_valid_actions_2048(board):
    """
    检查2048游戏板的有效移动方向
    通过模拟每个方向的移动来检查是否有效
    
    参数:
    board: 4x4的游戏板状态
    
    返回:
    valid_actions: 长度为4的数组，每个元素表示对应动作是否有效 (0=无效, 1=有效)
    """
    valid_actions = np.zeros(4, dtype=np.int32)  # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    
    # 将2048游戏的移动逻辑实现为函数
    def move_left(row):
        """移动一行到左边"""
        # 过滤掉0值
        non_zero = [x for x in row if x != 0]
        new_row = []
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i+1]:
                # 合并相同数字
                new_row.append(non_zero[i] * 2)
                i += 2  # 跳过下一个数字，因为它已被合并
            else:
                new_row.append(non_zero[i])
                i += 1
        # 填充0值到行末
        while len(new_row) < 4:
            new_row.append(0)
        return new_row

    for action in range(4):
        # 创建当前游戏板的副本以进行模拟
        temp_board = board.copy()
        original_board = board.copy()
        
        if action == 0:  # UP
            # 对每一列执行左移操作
            for j in range(4):
                col = temp_board[:, j]
                new_col = move_left(col)
                temp_board[:, j] = new_col
        elif action == 1:  # RIGHT
            # 对每一行执行右移操作（翻转-左移-翻转）
            for i in range(4):
                row = temp_board[i, :]
                flipped_row = np.flip(row)
                moved_row = move_left(flipped_row)
                temp_board[i, :] = np.flip(moved_row)
        elif action == 2:  # DOWN
            # 对每一列执行下移操作（翻转-左移-翻转）
            for j in range(4):
                col = temp_board[:, j]
                flipped_col = np.flip(col)
                moved_col = move_left(flipped_col)
                temp_board[:, j] = np.flip(moved_col)
        elif action == 3:  # LEFT
            # 对每一行执行左移操作
            for i in range(4):
                temp_board[i, :] = move_left(temp_board[i, :])
        
        # 检查移动后游戏板是否与原游戏板不同（即移动有效）
        if not np.array_equal(temp_board, original_board):
            valid_actions[action] = 1
        else:
            valid_actions[action] = 0
    
    return valid_actions


def simulate_row_move(row):
    """模拟单行移动，返回新行和合并奖励"""
    # 移除零值，并添加防护措施
    non_zero = []
    for x in row:
        if x != 0 and isinstance(x, (int, np.integer, np.floating)) and x > 0:
            non_zero.append(int(x))
    
    new_row = []
    i = 0
    reward = 0
    
    # 添加循环计数器防止无限循环
    max_iterations = len(non_zero) * 2
    iteration_count = 0
    
    while i < len(non_zero) and iteration_count < max_iterations:
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
            # 合并
            merged_value = non_zero[i] * 2
            new_row.append(merged_value)
            reward += merged_value  # 合并奖励等于合并后的值
            i += 2
        else:
            new_row.append(non_zero[i])
            i += 1
        iteration_count += 1
    
    # 如果检测到异常，返回原始行
    if iteration_count >= max_iterations:
        return np.array(row, dtype=int), 0
    
    # 填充零值
    while len(new_row) < 4:
        new_row.append(0)
    
    return np.array(new_row, dtype=int), reward


def simulate_move_2048(board, action):
    """
    模拟2048游戏移动，返回新的游戏板和合并奖励
    """
    # 添加输入验证
    if not isinstance(board, np.ndarray) or board.shape != (4, 4):
        return np.zeros((4, 4), dtype=int), -10
    
    original_board = board.copy()
    new_board = board.copy().astype(int)  # 确保是整数类型
    reward = 0
    
    # 移动逻辑
    if action == 0:  # 上
        for j in range(4):
            col = new_board[:, j]
            new_col, col_reward = simulate_row_move(col)
            new_board[:, j] = new_col
            reward += col_reward
    elif action == 1:  # 右
        for i in range(4):
            row = new_board[i, :]
            flipped_row = np.flip(row)
            new_row, row_reward = simulate_row_move(flipped_row)
            new_board[i, :] = np.flip(new_row)
            reward += row_reward
    elif action == 2:  # 下
        for j in range(4):
            col = new_board[:, j]
            flipped_col = np.flip(col)
            new_col, col_reward = simulate_row_move(flipped_col)
            new_board[:, j] = np.flip(new_col)
            reward += col_reward
    elif action == 3:  # 左
        for i in range(4):
            row = new_board[i, :]
            new_row, row_reward = simulate_row_move(row)
            new_board[i, :] = new_row
            reward += row_reward
    
    # 如果移动无效，返回原始状态和惩罚
    if np.array_equal(new_board, original_board):
        return original_board, -10  # 惩罚无效移动
    
    return new_board, reward