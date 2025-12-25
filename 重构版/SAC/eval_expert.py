"""评估已训练专家策略的脚本"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from microgrid_env import MicrogridEnv
from sac_agent import SACAgent


def _resolve_path(path_str: str) -> Path:
    """根据配置中的路径字符串返回可用路径"""
    p = Path(path_str)
    if p.exists():
        return p
    # 尝试相对于当前目录
    alt = Path(__file__).parent / path_str
    if alt.exists():
        return alt
    # 再尝试嵌套SAC目录（兼容旧路径）
    alt2 = Path(__file__).parent / "SAC" / path_str
    if alt2.exists():
        return alt2
    return p


def evaluate_expert(expert_id: int, config: dict, episodes: int = 20, deterministic: bool = True):
    """在环境中评估指定专家策略"""

    data_path = _resolve_path(config["data"]["clustered"])
    if not data_path.exists():
        raise FileNotFoundError(f"找不到训练数据: {data_path}")

    df = pd.read_csv(data_path)
    df_expert = df[df["Day_Label"] == expert_id].copy()
    if df_expert.empty:
        raise ValueError(f"专家 {expert_id} 没有对应的数据")

    env = MicrogridEnv(df_expert, config, expert_id=expert_id)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    model_dir_cfg = config["training"]["model_dir"]
    candidate_dirs = [
        _resolve_path(model_dir_cfg),
        Path(__file__).parent / model_dir_cfg,
        Path(__file__).parent / "SAC" / model_dir_cfg,
    ]

    model_dir = None
    for cand in candidate_dirs:
        actor_path = cand / f"expert_{expert_id}_actor.pth"
        if actor_path.exists():
            model_dir = cand
            break

    if model_dir is None:
        raise FileNotFoundError(
            f"找不到专家 {expert_id} 的模型权重，请确认目录配置是否正确"
        )

    agent = SACAgent(
        state_dim,
        action_dim,
        config,
        device="cuda" if config["training"]["use_gpu"] else "cpu",
    )
    agent.load(model_dir, expert_id)

    metrics = {
        "return": [],
        "cost": [],
        "curtail": [],
        "export": [],
        "import": [],
        "ramp": [],
    }

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        total_import = 0.0
        total_export = 0.0
        total_ramp = 0.0
        last_grid = None

        while not done:
            action = agent.take_action(state, deterministic=deterministic)
            next_state, reward, done, info = env.step(action[0])
            total_reward += reward
            total_import += info.get("import_mw", 0.0)
            total_export += info.get("export_mw", 0.0)
            if last_grid is not None:
                total_ramp += abs(info.get("grid_mw", 0.0) - last_grid)
            last_grid = info.get("grid_mw", 0.0)
            state = next_state

        metrics["return"].append(total_reward)
        metrics["cost"].append(info.get("episode_cost", 0.0))
        metrics["curtail"].append(info.get("episode_curtail", 0.0))
        metrics["import"].append(total_import)
        metrics["export"].append(total_export)
        metrics["ramp"].append(total_ramp)

        print(
            f"Episode {ep+1}/{episodes}: Return={total_reward:.2f}, Cost={info.get('episode_cost', 0.0):.2f}, "
            f"Curtail={info.get('episode_curtail', 0.0):.2f} MWh"
        )

    summary = {k: float(np.mean(v)) for k, v in metrics.items()}
    summary["episodes"] = episodes
    summary["deterministic"] = deterministic

    result_dir = Path("eval_results")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / f"expert_{expert_id}_eval.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nEvaluation Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.2f}")
    print(f"\n✓ 评估结果已保存: {output_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="评估专家策略")
    parser.add_argument("--expert_id", type=int, required=True, help="专家ID (0-4)")
    parser.add_argument("--episodes", type=int, default=20, help="评估episode数量")
    parser.add_argument("--config", type=str, default="phase2_config.yaml", help="配置文件路径")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="评估时使用随机策略（默认使用确定性策略）",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    evaluate_expert(args.expert_id, config, episodes=args.episodes, deterministic=not args.stochastic)


if __name__ == "__main__":
    main()
