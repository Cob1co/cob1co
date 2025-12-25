"""经验回放缓冲区 - 复用开源代码的实现"""

import collections
import random
from typing import List, Tuple


class ReplayBuffer:
    """简单的先进先出经验回放缓冲区
    
    存储：(state, action, reward, next_state, done) 五元组
    """
    
    def __init__(self, capacity: int):
        """
        参数：
            capacity: 缓冲区最大容量
        """
        self.buffer = collections.deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """添加一条经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """均匀随机采样
        
        返回：
            (states, actions, rewards, next_states, dones)
        """
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done
    
    def size(self) -> int:
        """返回当前缓冲区大小"""
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()


if __name__ == "__main__":
    """测试经验回放"""
    import numpy as np
    
    print("=" * 60)
    print("测试经验回放缓冲区")
    print("=" * 60)
    
    buffer = ReplayBuffer(capacity=1000)
    
    # 添加一些假数据
    for i in range(100):
        state = np.random.rand(6)
        action = np.random.rand(1)
        reward = np.random.rand()
        next_state = np.random.rand(6)
        done = False
        buffer.add(state, action, reward, next_state, done)
    
    print(f"缓冲区大小: {buffer.size()}")
    
    # 采样
    batch_size = 32
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    print(f"\n采样批次大小: {len(states)}")
    print(f"状态形状: {np.array(states).shape}")
    print(f"动作形状: {np.array(actions).shape}")
    
    print("\n✓ 经验回放测试通过！")
