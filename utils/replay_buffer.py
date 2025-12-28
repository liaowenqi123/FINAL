import numpy as np
import random


class ReplayBuffer:
    """
    经验回放缓冲区
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done, action_mask=None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, action_mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, action_mask = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, action_mask

    def clear(self):
        self.buffer = []
        self.position = 0