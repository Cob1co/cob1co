"""SAC智能体 - 基于开源代码改造为1维动作

主要改动：
1. 动作空间：6维 → 1维
2. PolicyNet简化：单个动作输出
3. 保留SAC核心算法不变
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path


class PolicyNet(nn.Module):
    """Actor网络：输出动作的均值和标准差"""
    
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int = 1):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        """
        输入：状态 x
        输出：动作 action ∈ [-1, 1]，对数概率 log_prob
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # 限制标准差范围
        std = torch.exp(log_std)
        
        # 重参数化采样
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # 可微采样
        log_prob = dist.log_prob(normal_sample)
        
        # tanh压缩到 [-1, 1]
        action = torch.tanh(normal_sample)
        
        # 修正对数概率（考虑tanh变换的雅可比行列式）
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob


class QValueNet(nn.Module):
    """Critic网络：评估 Q(s, a)"""
    
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int = 1):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, a):
        """
        输入：状态 x，动作 a
        输出：Q值
        """
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class SACAgent:
    """SAC算法智能体（处理连续动作）"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: dict,
        device: str = "cuda"
    ):
        """
        参数：
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            config: SAC配置（来自phase2_config.yaml）
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # SAC超参数
        sac_cfg = config["sac"]
        self.hidden_dim = sac_cfg["hidden_dim"]
        self.lr_actor = sac_cfg["lr_actor"]
        self.lr_critic = sac_cfg["lr_critic"]
        self.lr_alpha = sac_cfg["lr_alpha"]
        self.gamma = sac_cfg["gamma"]
        self.tau = sac_cfg["tau"]
        self.alpha_init = sac_cfg["alpha_init"]
        self.auto_alpha = sac_cfg["auto_alpha"]
        self.target_entropy = sac_cfg["target_entropy"]
        
        # 构建网络
        self.actor = PolicyNet(state_dim, self.hidden_dim, action_dim).to(self.device)
        self.critic_1 = QValueNet(state_dim, self.hidden_dim, action_dim).to(self.device)
        self.critic_2 = QValueNet(state_dim, self.hidden_dim, action_dim).to(self.device)
        
        # 目标网络
        self.target_critic_1 = QValueNet(state_dim, self.hidden_dim, action_dim).to(self.device)
        self.target_critic_2 = QValueNet(state_dim, self.hidden_dim, action_dim).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.lr_critic)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.lr_critic)
        
        # 熵系数（自动调节）
        if self.auto_alpha:
            self.log_alpha = torch.tensor(np.log(self.alpha_init), dtype=torch.float32, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_alpha)
        else:
            self.log_alpha = torch.tensor(np.log(self.alpha_init), dtype=torch.float32, device=self.device)
    
    def take_action(self, state, deterministic=False):
        """选择动作
        
        参数：
            state: 状态
            deterministic: 是否使用确定性策略（评估时用）
        
        返回：
            action: numpy数组
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if deterministic:
            # 确定性策略：直接用均值
            with torch.no_grad():
                x = F.relu(self.actor.fc1(state))
                x = F.relu(self.actor.fc2(x))
                mu = self.actor.fc_mu(x)
                action = torch.tanh(mu)
        else:
            # 随机策略：采样
            with torch.no_grad():
                action, _ = self.actor(state)
        
        return action.cpu().numpy()[0]
    
    def calc_target(self, rewards, next_states, dones):
        """计算目标Q值"""
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target
    
    def soft_update(self, net, target_net):
        """软更新目标网络"""
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def update(self, transition_dict):
        """更新网络参数
        
        参数：
            transition_dict: 包含 states, actions, rewards, next_states, dones
        """
        states = torch.FloatTensor(np.array(transition_dict['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(transition_dict['actions'])).reshape(-1, 1).to(self.device)
        rewards = torch.FloatTensor(np.array(transition_dict['rewards'])).reshape(-1, 1).to(self.device)
        next_states = torch.FloatTensor(np.array(transition_dict['next_states'])).to(self.device)
        dones = torch.FloatTensor(np.array(transition_dict['dones'])).reshape(-1, 1).to(self.device)
        
        # ========== 更新两个Critic网络 ==========
        td_target = self.calc_target(rewards, next_states, dones).detach()
        critic_1_loss = F.mse_loss(self.critic_1(states, actions), td_target)
        critic_2_loss = F.mse_loss(self.critic_2(states, actions), td_target)
        
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        
        # ========== 更新Actor网络 ==========
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ========== 更新熵系数alpha ==========
        if self.auto_alpha:
            # 标准 SAC：以 log_alpha 为参数，目标使平均熵接近 target_entropy
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # ========== 软更新目标网络 ==========
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
        
        return {
            'critic_1_loss': critic_1_loss.item(),
            'critic_2_loss': critic_2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.log_alpha.exp().item(),
        }
    
    def save(self, path: Path, expert_id: int):
        """保存模型"""
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path / f"expert_{expert_id}_actor.pth")
        torch.save(self.critic_1.state_dict(), path / f"expert_{expert_id}_critic1.pth")
        torch.save(self.critic_2.state_dict(), path / f"expert_{expert_id}_critic2.pth")
        print(f"✓ 模型已保存: {path}/expert_{expert_id}_*.pth")
    
    def load(self, path: Path, expert_id: int):
        """加载模型"""
        actor_state = torch.load(path / f"expert_{expert_id}_actor.pth", weights_only=True)
        critic1_state = torch.load(path / f"expert_{expert_id}_critic1.pth", weights_only=True)
        critic2_state = torch.load(path / f"expert_{expert_id}_critic2.pth", weights_only=True)

        self.actor.load_state_dict(actor_state)
        self.critic_1.load_state_dict(critic1_state)
        self.critic_2.load_state_dict(critic2_state)
        # 同步目标网络
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        print(f"✓ 模型已加载: {path}/expert_{expert_id}_*.pth")


if __name__ == "__main__":
    """测试SAC智能体"""
    import yaml
    from replay_buffer import ReplayBuffer
    
    print("=" * 60)
    print("测试SAC智能体")
    print("=" * 60)
    
    # 加载配置
    config_path = Path(__file__).parent / "phase2_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    state_dim = 6
    action_dim = 1
    
    agent = SACAgent(state_dim, action_dim, config, device="cpu")
    
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"隐藏层维度: {agent.hidden_dim}")
    print(f"设备: {agent.device}")
    
    # 测试前向传播
    state = np.random.rand(state_dim)
    action = agent.take_action(state)
    print(f"\n随机状态: {state}")
    print(f"输出动作: {action}")
    
    # 测试更新
    replay_buffer = ReplayBuffer(1000)
    for _ in range(100):
        replay_buffer.add(
            np.random.rand(state_dim),
            np.random.rand(action_dim),
            np.random.rand(),
            np.random.rand(state_dim),
            False
        )
    
    batch = replay_buffer.sample(32)
    transition_dict = {
        'states': batch[0],
        'actions': batch[1],
        'rewards': batch[2],
        'next_states': batch[3],
        'dones': batch[4]
    }
    
    losses = agent.update(transition_dict)
    print(f"\n更新损失:")
    for k, v in losses.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n✓ SAC智能体测试通过！")
