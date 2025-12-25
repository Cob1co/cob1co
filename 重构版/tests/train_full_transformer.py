"""正式训练Transformer - 使用30天完整数据

使用完整数据集训练Transformer权重控制器
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from LMPC.training.train_transformer import train_transformer


def main():
    print("="*70)
    print("🚀 正式训练Transformer权重控制器")
    print("="*70)
    
    # 使用绝对路径
    project_root = Path(__file__).parent.parent
    data_path = str(project_root / 'LMPC' / 'data' / 'training_data_30days.pkl')
    output_dir = str(project_root / 'LMPC' / 'models' / 'production')
    
    print(f"\n📂 数据路径: {data_path}")
    print(f"📂 输出目录: {output_dir}")
    
    # 正式训练配置
    train_transformer(
        data_path=data_path,
        output_dir=output_dir,
        batch_size=32,         # 标准批次
        epochs=100,            # 完整训练
        learning_rate=1e-4,    # 标准学习率
        val_split=0.2          # 20%验证集
    )
    
    print("\n" + "="*70)
    print("✅ Transformer训练完成！")
    print("="*70)
    print("\n下一步：")
    print("1. 查看训练曲线和预测散点图")
    print("2. 使用训练好的模型进行系统评估")
    print("3. 对比有/无Transformer的性能差异")


if __name__ == '__main__':
    main()
