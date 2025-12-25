"""测试Transformer训练脚本

使用之前收集的小规模测试数据（56个样本）
验证训练流程
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from LMPC.training.train_transformer import train_transformer


def main():
    print("="*70)
    print("🧪 测试Transformer训练流程")
    print("="*70)
    
    # 使用绝对路径
    project_root = Path(__file__).parent.parent
    data_path = str(project_root / 'LMPC' / 'data' / 'training_data_test.pkl')
    output_dir = str(project_root / 'LMPC' / 'models' / 'test')
    
    # 训练配置（小规模测试）
    train_transformer(
        data_path=data_path,
        output_dir=output_dir,
        batch_size=8,      # 小批次
        epochs=20,         # 少轮数
        learning_rate=1e-3,  # 稍高学习率
        val_split=0.2
    )
    
    print("\n✅ 训练流程测试完成！")


if __name__ == '__main__':
    main()
