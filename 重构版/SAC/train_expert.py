"""训练单个专家策略的脚本

使用方法：
    python train_expert.py --expert_id 0
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from microgrid_env import MicrogridEnv
from sac_agent import SACAgent
from replay_buffer import ReplayBuffer


def train_expert(expert_id: int, config: dict, show_progress: bool = True):
    """训练单个专家
    
    参数：
        expert_id: 专家ID (0-4)
        config: 配置字典
        show_progress: 是否显示进度条
    """
    print("=" * 60)
    print(f"开始训练专家 {expert_id}")
    print("=" * 60)
    
    # ========== 1. 加载数据 ==========
    data_path_str = config["data"]["clustered"]
    # 统一约定：若给的是相对路径，则相对于本脚本所在的 SAC 目录解析
    data_path = Path(__file__).parent / data_path_str

    if not data_path.exists():
        raise FileNotFoundError(f"训练数据不存在: {data_path}\n请先运行 prepare_training_data.py")
    
    df = pd.read_csv(data_path)
    df_expert = df[df["Day_Label"] == expert_id].copy()
    
    if len(df_expert) == 0:
        raise ValueError(f"专家 {expert_id} 没有对应的数据")
    
    print(f"\n数据统计:")
    print(f"  总数据: {len(df)} 条")
    print(f"  专家 {expert_id} 数据: {len(df_expert)} 条")
    print(f"  对应天数: {len(df_expert) // 24} 天")
    
    # ========== 2. 创建环境和智能体 ==========
    env = MicrogridEnv(df_expert, config, expert_id=expert_id)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    
    device = "cuda" if config["training"]["use_gpu"] else "cpu"
    agent = SACAgent(state_dim, action_dim, config, device=device)
    
    print(f"\n环境配置:")
    print(f"  状态维度: {state_dim}")
    print(f"  动作维度: {action_dim}")
    print(f"  Episode长度: {env.max_steps} 步")
    print(f"  计算设备: {agent.device}")
    
    # 额外的GPU信息
    import torch
    if torch.cuda.is_available():
        print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"  ⚠️ CUDA不可用，使用CPU训练（速度会很慢）")
    
    # ========== 3. 训练参数 ==========
    sac_cfg = config["sac"]
    buffer_size = sac_cfg["buffer_size"]
    batch_size = sac_cfg["batch_size"]
    warmup_steps = sac_cfg["warmup_steps"]
    max_episodes = sac_cfg["max_episodes"]
    save_frequency = sac_cfg["save_frequency"]
    
    replay_buffer = ReplayBuffer(buffer_size)
    
    # 记录训练指标
    return_list = []
    cost_list = []
    curtail_list = []
    
    # ========== 4. 训练循环 ==========
    print(f"\n开始训练 ({max_episodes} episodes)...")
    
    total_steps = 0
    
    for episode in tqdm(range(1, max_episodes + 1), desc=f"Expert {expert_id}"):
        state = env.reset()
        episode_return = 0
        episode_cost = 0
        episode_curtail = 0
        
        done = False
        step = 0
        
        while not done:
            # 选择动作（前warmup_steps步用随机动作）
            if total_steps < warmup_steps:
                action = np.random.uniform(-1, 1, size=(action_dim,))
            else:
                action = agent.take_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action[0] if action_dim == 1 else action)
            
            # 存储经验
            replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_return += reward
            episode_cost = info.get("episode_cost", 0)
            episode_curtail = info.get("episode_curtail", 0)
            
            # 更新网络（如果buffer足够大）
            if replay_buffer.size() >= batch_size and total_steps >= warmup_steps:
                batch = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': batch[0],
                    'actions': batch[1],
                    'rewards': batch[2],
                    'next_states': batch[3],
                    'dones': batch[4]
                }
                agent.update(transition_dict)
            
            total_steps += 1
            step += 1
        
        # 记录episode指标
        return_list.append(episode_return)
        cost_list.append(episode_cost)
        curtail_list.append(episode_curtail)
        
        # 定期保存模型
        if episode % save_frequency == 0:
            model_dir = Path(config["training"]["model_dir"])
            agent.save(model_dir, expert_id)
            
            # 打印最近10个episode的平均指标
            if len(return_list) >= 10:
                avg_return = np.mean(return_list[-10:])
                avg_cost = np.mean(cost_list[-10:])
                avg_curtail = np.mean(curtail_list[-10:])
                tqdm.write(f"\nEpisode {episode}: "
                          f"Avg Return={avg_return:.2f}, "
                          f"Avg Cost={avg_cost:.2f}, "
                          f"Avg Curtail={avg_curtail:.2f}")
    
    # ========== 5. 最终保存 ==========
    model_dir = Path(config["training"]["model_dir"])
    agent.save(model_dir, expert_id)
    
    # 保存训练曲线数据
    log_dir = Path(config["training"]["log_dir"]) / f"expert_{expert_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    training_log = {
        "episode": list(range(1, len(return_list) + 1)),
        "return": return_list,
        "cost": cost_list,
        "curtail": curtail_list,
    }
    
    with open(log_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)
    
    # ========== 6. 绘制训练曲线 ==========
    plot_training_curve(return_list, cost_list, curtail_list, log_dir, expert_id)
    
    print(f"\n✓ 专家 {expert_id} 训练完成!")
    print(f"  模型保存: {model_dir}")
    print(f"  日志保存: {log_dir}")
    
    return agent, training_log


def plot_training_curve(returns, costs, curtails, save_dir, expert_id):
    """绘制训练曲线"""
    episodes = list(range(1, len(returns) + 1))
    
    # 计算移动平均
    window = 9
    if len(returns) >= window:
        returns_smooth = moving_average(returns, window)
        costs_smooth = moving_average(costs, window)
        curtails_smooth = moving_average(curtails, window)
    else:
        returns_smooth = returns
        costs_smooth = costs
        curtails_smooth = curtails
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # 回报曲线
    axes[0].plot(episodes, returns, alpha=0.3, label='Raw')
    axes[0].plot(episodes[:len(returns_smooth)], returns_smooth, label='Smoothed')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return')
    axes[0].set_title(f'Expert {expert_id} - Training Return')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 成本曲线
    axes[1].plot(episodes, costs, alpha=0.3, label='Raw')
    axes[1].plot(episodes[:len(costs_smooth)], costs_smooth, label='Smoothed')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Cost (CNY)')
    axes[1].set_title(f'Expert {expert_id} - Episode Cost')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 弃电曲线
    axes[2].plot(episodes, curtails, alpha=0.3, label='Raw')
    axes[2].plot(episodes[:len(curtails_smooth)], curtails_smooth, label='Smoothed')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Curtailment (MWh)')
    axes[2].set_title(f'Expert {expert_id} - Episode Curtailment')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150)
    plt.close()
    
    print(f"  训练曲线已保存: {save_dir / 'training_curves.png'}")


def moving_average(data, window_size):
    """计算移动平均"""
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练单个专家策略')
    parser.add_argument('--expert_id', type=int, required=True, help='专家ID (0-4)')
    parser.add_argument('--config', type=str, default='phase2_config.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 训练
    train_expert(args.expert_id, config)


if __name__ == "__main__":
    main()
