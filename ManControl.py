import gymnasium as gym
import gymnasium_2048
import pygame
from pygame.locals import *
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import cv2
import os
import sys

class Simple2048:
    def __init__(self, seed=42, record=False, replay_file=None):
        """
        极简2048游戏
        
        参数:
            seed: 随机种子
            record: 是否记录游戏
            replay_file: 回放文件路径
        """
        # 创建环境
        self.env = gym.make("gymnasium_2048/TwentyFortyEight-v0", render_mode="human")
        self.seed = seed
        self.score = 0
        self.game_over = False
        self.record = record
        self.game_log = []
        
        # 按键映射
        self.key_to_action = {
            K_UP: 0, K_w: 0,    # 上
            K_RIGHT: 1, K_d: 1, # 右
            K_DOWN: 2, K_s: 2,  # 下
            K_LEFT: 3, K_a: 3   # 左
        }
        
        if replay_file:
            self.replay(replay_file)
        else:
            self.play()
    
    def get_board_array(self, state):
        """安全地获取棋盘数组"""
        if isinstance(state, np.ndarray):
            return state
        elif isinstance(state, dict):
            # 尝试常见键名
            for key in ['board', 'state', 'observation']:
                if key in state and isinstance(state[key], np.ndarray):
                    return state[key]
        return np.zeros((4, 4), dtype=int)
    
    def convert_board_values(self, board):
        """转换棋盘值：将索引值转换为2048瓦片值"""
        if isinstance(board, list):
            converted_board = []
            for row in board:
                converted_row = []
                for cell in row:
                    if isinstance(cell, (int, float)) and cell > 0:
                        # 如果是1-16之间的整数，可能是索引值
                        if 1 <= cell <= 16 and cell == int(cell):
                            # 转换为2的幂：1->2, 2->4, 3->8, 4->16, ...
                            converted_row.append(2 ** int(cell))
                        else:
                            # 已经是瓦片值，直接使用
                            converted_row.append(int(cell))
                    else:
                        converted_row.append(0)
                converted_board.append(converted_row)
            return converted_board
        return board
    
    def reset(self):
        """重置游戏"""
        state, _ = self.env.reset(seed=self.seed)
        self.board = self.get_board_array(state)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_log = []
        
        if self.record:
            self.game_log.append({
                'step': 0,
                'board': self.board.tolist(),
                'score': 0
            })
    
    def step(self, action):
        """执行一步"""
        if self.game_over:
            return
        
        # 记录移动前
        if self.record:
            board_before = self.board.copy()
            score_before = self.score
        
        # 执行动作
        result = self.env.step(action)
        
        # 解析结果
        if len(result) == 4:
            next_state, reward, terminated, _ = result
            self.game_over = terminated
        else:
            next_state, reward, terminated, truncated, _ = result
            self.game_over = terminated or truncated
        
        # 更新状态
        self.board = self.get_board_array(next_state)
        
        # 更新分数
        if isinstance(reward, (int, float, np.number)):
            self.score += float(reward)
        
        self.steps += 1
        
        # 记录
        if self.record:
            self.game_log.append({
                'step': self.steps,
                'board': self.board.tolist(),
                'score': self.score,
                'action': action,
                'reward': float(reward) if isinstance(reward, (int, float, np.number)) else 0
            })
    
    def _capture_frame(self, video_writer):
        """捕获pygame窗口并写入视频"""
        try:
            # 获取pygame窗口表面
            screen = pygame.display.get_surface()
            if screen is None:
                return
            
            # 将pygame表面转换为numpy数组
            frame = pygame.surfarray.array3d(screen)
            
            # 转换颜色顺序 (RGB -> BGR)
            frame = cv2.cvtColor(frame.swapaxes(0, 1), cv2.COLOR_RGB2BGR)
            
            # 写入视频
            video_writer.write(frame)
        except Exception as e:
            print(f"捕获帧失败: {e}")

    def save_log(self):
        """保存游戏记录"""
        if not self.record or not self.game_log:
            return
        
        log_dir = Path("2048_logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"2048_seed{self.seed}_{timestamp}.json"
        
        data = {
            'seed': self.seed,
            'final_score': self.score,
            'steps': self.steps,
            'max_tile': int(np.max(self.board)),
            'log': self.game_log,
            'timestamp': timestamp
        }
        
        filepath = log_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n游戏记录已保存: {filepath}")
    
    def play(self):
        """开始游戏"""
        print(f"2048游戏 - 种子: {self.seed}")
        print("控制: 方向键或WASD移动, ESC或者Q退出")
        print("=" * 40)
        
        self.reset()
        
        running = True
        while running:
            self.env.render()
            
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                
                elif event.type == KEYDOWN:
                    if event.key in (K_q, K_ESCAPE):
                        running = False
                    
                    elif event.key in self.key_to_action:
                        action = self.key_to_action[event.key]
                        
                        if not self.game_over:
                            self.step(action)
                            
                            if self.game_over:
                                print(f"\n游戏结束！最终分数: {self.score}")
                                if self.record:
                                    self.save_log()
                                # 等待退出
                                while True:
                                    self.env.render()
                                    for e in pygame.event.get():
                                        if e.type == QUIT or (e.type == KEYDOWN and e.key in (K_q, K_ESCAPE)):
                                            running = False
                                            break
                                    if not running:
                                        break
                                    pygame.time.Clock().tick(60)
            
            pygame.time.Clock().tick(60)
        
        self.env.close()
    
    def replay(self, replay_file, record_video=True):
        """回放游戏"""
        print(f"回放: {replay_file}")
        
        # 初始化video_writer变量
        video_writer = None
        video_path = None
        
        try:
            # 加载记录
            filepath = Path(replay_file)
            if not filepath.exists():
                filepath = Path("2048_logs") / replay_file
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            seed = data.get('seed', self.seed)
            # 兼容新旧格式：新格式使用game_log，旧格式使用log
            log = data.get('game_log', data.get('log', []))
            
            if not log:
                print("记录为空")
                return
            
            print(f"种子: {seed} | 分数: {data.get('final_score', 0)}")
            print("按任意键开始回放...")
            self.env.reset(seed=seed)
            
            # 视频录制初始化
            if record_video:
                try:
                    # 创建录制目录
                    video_dir = Path("2048_videos")
                    video_dir.mkdir(exist_ok=True)
                    
                    # 获取窗口大小
                    info = pygame.display.Info()
                    window_width, window_height = info.current_w, info.current_h
                    
                    # 生成视频文件名
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_filename = f"replay_seed{seed}_{timestamp}.mp4"
                    video_path = video_dir / video_filename
                    
                    # 初始化视频写入器
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 3  # 降低帧率，使视频更流畅
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (window_width, window_height))
                    
                    print(f"开始录制视频: {video_path}")
                except Exception as e:
                    print(f"视频录制初始化失败: {e}")
                    record_video = False
                    video_writer = None
            
            # 等待开始（不录制）
            waiting = True
            while waiting:
                self.env.render()
                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        waiting = False
                        # 开始录制
                        if record_video and video_writer:
                            print("开始录制游戏过程...")
            
            # 回放每一步
            import time
            for i, step in enumerate(log):
                if i == 0:  # 初始状态
                    # 显示初始状态并录制
                    self.env.render()
                    if record_video and video_writer:
                        self._capture_frame(video_writer)
                    continue
                
                action = step.get('action')
                if action is not None:
                    self.env.step(action)
                
                self.env.render()
                
                # 录制当前帧
                if record_video and video_writer:
                    self._capture_frame(video_writer)
                
                time.sleep(0.3)
                
                # 检查退出
                for event in pygame.event.get():
                    if event.type == KEYDOWN and event.key in (K_q, K_ESCAPE):
                        print("回放中断")
                        if video_writer:
                            video_writer.release()
                            if video_path:
                                print(f"视频已保存: {video_path}")
                        return
            
            print("回放完成！按Q退出...")
            
            # 停止录制
            if video_writer:
                video_writer.release()
                if video_path:
                    print(f"视频已保存: {video_path}")
                video_writer = None
            
            # 等待退出（不录制）
            while True:
                self.env.render()
                for event in pygame.event.get():
                    if event.type == KEYDOWN and event.key in (K_q, K_ESCAPE):
                        return
        
        except Exception as e:
            print(f"回放出错: {e}")
        
        finally:
            if video_writer:
                video_writer.release()
            self.env.close()

# 最简使用方式
if __name__ == "__main__":
    import sys
    
    # 如果有命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--replay":
            replay_file = sys.argv[2] if len(sys.argv) > 2 else None
            record_video = "--no-video" not in sys.argv  # 默认录制视频
            if replay_file:
                game = Simple2048(replay_file=replay_file)
                # 调用replay方法时传入录制参数
                game.replay(replay_file, record_video=record_video)
            else:
                print("请指定回放文件")
        else:
            try:
                seed = int(sys.argv[1])
                game = Simple2048(seed=seed, record=True)
            except ValueError:
                print(f"使用默认种子42")
                game = Simple2048(seed=42, record=True)
    else:
        # 默认：种子42，记录游戏
        game = Simple2048(seed=42, record=True)