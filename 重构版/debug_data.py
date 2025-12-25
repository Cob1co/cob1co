import pickle
import numpy as np
from pathlib import Path

# 读取数据
pkl_path = Path('LMPC/logs/eval_results_march.pkl')
print(f"📂 读取文件: {pkl_path}")

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

base_soc = np.array(data['baseline']['soc'])
lmpc_soc = np.array(data['phase3']['soc'])
weights = np.array(data['phase3']['weights'])

# 1. 检查数据长度
print(f"\n📏 数据长度: Base={len(base_soc)}, LMPC={len(lmpc_soc)}")

# 2. 计算 SOC 绝对差异总和
diff_soc = np.abs(base_soc - lmpc_soc)
total_diff = np.sum(diff_soc)
max_diff = np.max(diff_soc)

print(f"🔍 SOC 差异统计:")
print(f"   差异总和: {total_diff:.6f}")
print(f"   最大差异: {max_diff:.6f}")

if total_diff == 0:
    print("❌ 结论: 底层数据确实完全一样！可能是评估脚本保存逻辑有误。")
else:
    print("✅ 结论: 底层数据不一样！是 Streamlit 显示的问题（缓存或图表Bug）。")

# 3. 检查权重是否在变
print(f"\n🧠 权重检查 (前 10 步):")
print(f"   Alpha_SOC | Alpha_Grid | Alpha_Cost")
for i in range(10):
    print(f"   {weights[i][0]:.4f}    | {weights[i][1]:.4f}     | {weights[i][2]:.4f}")

std_soc_w = np.std(weights[:, 0])
if std_soc_w < 1e-6:
    print("⚠️ 警告: Alpha_SOC 权重完全没变过！")
else:
    print("✅ 确认: 权重在动态变化。")