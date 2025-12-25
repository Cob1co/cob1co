"""数据收集器测试脚本

小规模测试：收集1天数据（约72个样本）
验证整个数据收集流程
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import pandas as pd
import pickle

from LMPC.training.data_collector import DataCollector


def main():
    print("="*70)
    print("🧪 数据收集器小规模测试")
    print("="*70)
    
    # 1. 加载配置
    print("\n📋 步骤1：加载配置")
    config_path = Path(__file__).parent.parent / 'LMPC' / 'phase3_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print("✅ 配置加载完成")
    
    # 2. 加载数据（2023年历史数据 + 对应训练预测数据）
    print("\n📋 步骤2：加载数据（2023训练数据）")
    data_dir = Path(__file__).parent.parent
    realtime_path = data_dir / 'data' / 'data2023.csv'
    realtime_df = pd.read_csv(realtime_path, parse_dates=['Time'])
    print(f"   历史数据(2023): {len(realtime_df)}行")
    
    # 训练用预测数据（8小时窗口，带误差）
    forecast_path = data_dir / 'LMPC' / 'data' / 'forecast_2023_8h_training.pkl'
    with open(forecast_path, 'rb') as f:
        forecast_data = pickle.load(f)
    print(f"   训练预测数据(2023): {len(forecast_data)}条")
    print("✅ 数据加载完成")
    
    # 3. 创建数据收集器
    print("\n📋 步骤3：创建数据收集器")
    collector = DataCollector(config)
    print("✅ 数据收集器创建完成")
    
    # 收集完整训练数据（2023年1月前30天）
    print("\n📋 步骤4：收集完整训练数据（2023-01-01 至 2023-01-30）")
    print("⚠️  注意：这将需要较长时间（约2小时）")
    print("   30天 × 96步 × 27种权重组合")
    print("   建议让程序在后台运行")
    
    output_path = str(data_dir / 'LMPC' / 'data' / 'training_data_30days.pkl')
    training_data = collector.collect_training_data(
        realtime_df=realtime_df,
        forecast_data=forecast_data,
        start_date='2023-01-01',
        end_date='2023-01-30',  # 30天
        output_path=output_path
    )
    
    # 5. 验证结果
    print("\n📋 步骤5：验证结果")
    if len(training_data['features']) > 0:
        print(f"✅ 成功收集 {len(training_data['features'])} 个样本")
        print(f"   特征形状: {training_data['features'][0].shape}")
        print(f"   标签形状: {training_data['labels'][0].shape}")
        print(f"   标签示例: {training_data['labels'][0]}")
    else:
        print("❌ 未收集到任何样本")
    
    print("\n" + "="*70)
    print("🎉 数据收集器测试完成！")
    print("="*70)


if __name__ == '__main__':
    main()
