"""批量训练所有5个专家策略

使用方法：
    python train_all_experts.py
"""

import yaml
from pathlib import Path
import time

from train_expert import train_expert


def main():
    """训练所有专家"""
    print("=" * 70)
    print(" " * 20 + "训练所有专家策略")
    print("=" * 70)
    
    # 加载配置：始终从脚本所在目录查找 phase2_config.yaml，避免工作目录不同导致找不到文件
    config_path = Path(__file__).parent / "phase2_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    num_experts = config["training"]["num_experts"]
    
    print(f"\n配置信息:")
    print(f"  专家数量: {num_experts}")
    print(f"  最大Episode: {config['sac']['max_episodes']}")
    print(f"  计算设备: {'GPU' if config['training']['use_gpu'] else 'CPU'}")
    
    # 训练每个专家
    start_time = time.time()
    
    for expert_id in range(num_experts):
        print(f"\n{'='*70}")
        print(f"训练专家 {expert_id}/{num_experts-1}")
        print(f"{'='*70}")
        
        expert_start = time.time()
        
        try:
            train_expert(expert_id, config, show_progress=True)
            expert_time = time.time() - expert_start
            print(f"\n✓ 专家 {expert_id} 训练完成，用时: {expert_time/60:.2f} 分钟")
        
        except Exception as e:
            print(f"\n✗ 专家 {expert_id} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"所有专家训练完成!")
    print(f"总用时: {total_time/60:.2f} 分钟 ({total_time/3600:.2f} 小时)")
    print(f"{'='*70}")
    
    # 输出模型位置
    model_dir = Path(config["training"]["model_dir"])
    print(f"\n模型保存位置: {model_dir}")
    print(f"模型文件列表:")
    if model_dir.exists():
        for f in sorted(model_dir.glob("expert_*.pth")):
            print(f"  - {f.name}")
    
    print(f"\n✓ 可以开始运行全年评估脚本: python eval_annual.py")


if __name__ == "__main__":
    main()
