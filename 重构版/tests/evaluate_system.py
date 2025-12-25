"""Phase 3系统评估

对比有/无Transformer的性能差异
修正：计算真实电费成本而非MPC目标函数值
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import yaml
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from LMPC.core.weather_classifier import WeatherClassifier
from LMPC.core.expert_interface import ExpertInterface
from LMPC.core.feature_extractor import FeatureExtractor
from LMPC.core.transformer_controller import TransformerController
from LMPC.core.mpc_controller import MPCController

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def run_baseline(config, realtime_df, forecast_data, start_idx, end_idx):
    """运行基线（固定权重α=1.0）"""
    
    print("\n🔵 运行基线：固定权重 α=[1.0, 1.0, 1.0]")
    
    # 初始化模块
    expert_interface = ExpertInterface(config)
    feature_extractor = FeatureExtractor(config)
    mpc = MPCController(config)
    weather_classifier = WeatherClassifier(config=config)
    
    # 记录结果
    results = {
        'soc': [],
        'grid_power': [],
        'cost': [], # 这里将存储真实人民币成本（购电成本）
        'soc_error': [],
        'grid_error': [],
        'curtail_energy': [],  # 弃电电量(MWh)
    }
    
    # 当前状态
    current_soc = 0.5  # 初始SOC为50%
    prev_grid_power = 0.0
    
    total_skipped = 0
    # 时间步长 (小时)，直接复用 MPCController 中的配置
    dt_hours = float(getattr(mpc, "dt_hours", 0.25))

    for t in tqdm(range(start_idx, end_idx), desc="基线推理"):
        # 历史数据（24步）
        history_start = max(0, t - 23)
        history_data = realtime_df.iloc[history_start:t+1]
        
        # 预测数据（8小时，32步）
        if t < len(forecast_data):
            forecast_item = forecast_data[t]
            forecast_8h = {
                'load': forecast_item['forecast']['load'],
                'pv': forecast_item['forecast']['solar'] / 1000.0 * 35.0,
                'wind': forecast_item['forecast']['wind'] / 10.0 * 20.0,
                'price': forecast_item['forecast']['price']
            }
        else:
            total_skipped += 1
            continue
        
        # 天气分类
        expert_id = weather_classifier.classify_from_history(history_data)
        
        # 获取专家计划
        current_state = {'soc': current_soc, 'grid_power': prev_grid_power}
        plan = expert_interface.get_plan(expert_id, current_state, forecast_8h)
        
        # MPC求解（固定权重）
        reference_plan = {
            'soc': plan['soc'],
            'grid_power': plan['grid_power']
        }
        
        mpc_forecast = {
            'load': forecast_8h['load'][:16] / 1000,
            'pv': forecast_8h['pv'][:16],
            'wind': forecast_8h['wind'][:16],
            'price': forecast_8h['price'][:16]
        }
        
        solution = mpc.solve(
            current_state=current_state,
            forecast=mpc_forecast,
            reference_plan=reference_plan,
            dynamic_weights=None  # 固定权重（α=1.0）
        )
        
        # 更新状态
        if solution['status'] == 'optimal':
            current_soc = solution['soc_plan'][0]
            current_grid = solution['grid_plan'][0]

            # --- 计算真实购电成本（优先使用实时电价） ---
            import_mw = max(0.0, -current_grid)
            # 电价 (元/kWh)：优先用实时数据，缺失时退回预测价
            if "Price_CNY_kWh" in realtime_df.columns:
                price_kwh = float(realtime_df.iloc[t]["Price_CNY_kWh"])
            else:
                price_kwh = float(mpc_forecast["price"][0])
            # 成本 = 购电 × 1000 × 元/kWh × 小时
            real_cost = import_mw * 1000.0 * price_kwh * dt_hours

            # 弃电电量(MWh)，取当前时刻的弃电功率
            curtail_plan = np.asarray(solution.get("P_curtail_plan", []), dtype=float)
            curtail_mw = float(curtail_plan[0]) if curtail_plan.size > 0 else 0.0
            curtail_energy = curtail_mw * dt_hours

            results['cost'].append(real_cost)
            # ---------------------------

            soc_error = abs(current_soc - plan['soc'][0])
            grid_error = abs(current_grid - plan['grid_power'][0])
            
            results['soc'].append(current_soc)
            results['grid_power'].append(current_grid)
            results['soc_error'].append(soc_error)
            results['grid_error'].append(grid_error)
            results['curtail_energy'].append(curtail_energy)

            prev_grid_power = current_grid
        else:
            if t == start_idx + 10:
                print(f"\n⚠️  基线MPC求解失败: 状态={solution['status']}")
    
    print(f"\n📊 基线统计:")
    print(f"   有效样本: {len(results['soc'])}")
    print(f"   跳过样本: {total_skipped}")
    return results


def _compute_forecast_error_eval(realtime_df, forecast_data, idx, key, window=4):
    errors = []
    for i in range(1, window + 1):
        t = idx - i
        if t < 0 or t >= len(realtime_df) or t >= len(forecast_data):
            continue
        f_item = forecast_data[t]
        f_dict = f_item.get("forecast", {})
        if key == "load":
            real = float(realtime_df.iloc[t]["Load_kW"])
            pred_arr = np.asarray(f_dict.get("load", []), dtype=float)
            if pred_arr.size == 0:
                continue
            pred = float(pred_arr[0])
        elif key == "solar":
            real = float(realtime_df.iloc[t]["Solar_W_m2"])
            pred_arr = np.asarray(f_dict.get("solar", []), dtype=float)
            if pred_arr.size == 0:
                continue
            pred = float(pred_arr[0])
        elif key == "wind":
            real = float(realtime_df.iloc[t]["Wind_Speed_m_s"])
            pred_arr = np.asarray(f_dict.get("wind", []), dtype=float)
            if pred_arr.size == 0:
                continue
            pred = float(pred_arr[0])
        else:
            continue
        if real <= 0:
            continue
        err = abs(real - pred) / real
        errors.append(err)
    return float(np.mean(errors)) if errors else 0.0


def run_mpc_only(config, realtime_df, forecast_data, start_idx, end_idx):
    """运行纯 MPC 基线（不使用专家参考计划）"""

    print("\n🔶 运行纯 MPC 基线（无专家参考）")

    mpc = MPCController(config)

    results = {
        'soc': [],
        'grid_power': [],
        'cost': [],
        'soc_error': [],
        'grid_error': [],
        'curtail_energy': [],  # 弃电电量(MWh)
    }

    current_soc = 0.5
    prev_grid_power = 0.0
    dt_hours = float(getattr(mpc, "dt_hours", 0.25))

    total_skipped = 0

    for t in tqdm(range(start_idx, end_idx), desc="纯MPC推理"):
        # 预测数据（8小时，32步）
        if t < len(forecast_data):
            forecast_item = forecast_data[t]
            forecast_8h = {
                'load': forecast_item['forecast']['load'],
                'pv': forecast_item['forecast']['solar'] / 1000.0 * 35.0,
                'wind': forecast_item['forecast']['wind'] / 10.0 * 20.0,
                'price': forecast_item['forecast']['price']
            }
        else:
            total_skipped += 1
            continue

        # 参考轨迹：SOC 恒定 0.5，电网功率参考为 0
        H = min(16, len(forecast_8h['load']))
        soc_ref = np.full(H, 0.5, dtype=float)
        grid_ref = np.zeros(H, dtype=float)

        reference_plan = {
            'soc': soc_ref,
            'grid_power': grid_ref,
        }

        mpc_forecast = {
            'load': forecast_8h['load'][:H] / 1000,
            'pv': forecast_8h['pv'][:H],
            'wind': forecast_8h['wind'][:H],
            'price': forecast_8h['price'][:H],
        }

        current_state = {'soc': current_soc, 'grid_power': prev_grid_power}
        solution = mpc.solve(
            current_state=current_state,
            forecast=mpc_forecast,
            reference_plan=reference_plan,
            dynamic_weights=None,
        )

        if solution.get('status') == 'optimal':
            current_soc = float(solution['soc_plan'][0])
            current_grid = float(solution['grid_plan'][0])

            # 真实购电成本
            import_mw = max(0.0, -current_grid)
            if "Price_CNY_kWh" in realtime_df.columns:
                price_kwh = float(realtime_df.iloc[t]["Price_CNY_kWh"])
            else:
                price_kwh = float(mpc_forecast["price"][0])
            real_cost = import_mw * 1000.0 * price_kwh * dt_hours

            # 弃电电量(MWh)
            curtail_plan = np.asarray(solution.get("P_curtail_plan", []), dtype=float)
            curtail_mw = float(curtail_plan[0]) if curtail_plan.size > 0 else 0.0
            curtail_energy = curtail_mw * dt_hours

            results['cost'].append(real_cost)

            soc_error = abs(current_soc - soc_ref[0])
            grid_error = abs(current_grid - grid_ref[0])

            results['soc'].append(current_soc)
            results['grid_power'].append(current_grid)
            results['soc_error'].append(soc_error)
            results['grid_error'].append(grid_error)
            results['curtail_energy'].append(curtail_energy)

            prev_grid_power = current_grid
        else:
            if t == start_idx + 10:
                print(f"\n⚠️  纯MPC求解失败: 状态={solution.get('status')}")

    print(f"\n📊 纯MPC统计:")
    print(f"   有效样本: {len(results['soc'])}")
    print(f"   跳过样本: {total_skipped}")
    
    if len(results['soc']) > 0:
        total_cost = float(np.sum(results['cost']))
        mean_soc_err = float(np.mean(results['soc_error']))
        mean_grid_err = float(np.mean(results['grid_error']))
        grid_std = float(np.std(np.diff(results['grid_power'])))
        total_curt = float(np.sum(results.get('curtail_energy', [])))

        print("\n" + "=" * 70)
        print("📊 纯MPC性能统计")
        print("=" * 70)
        print(f"总成本: {total_cost:.2f} (真实购电成本)")
        print(f"平均SOC跟踪误差: {mean_soc_err:.4f}")
        print(f"平均电网功率跟踪误差: {mean_grid_err:.4f} MW")
        print(f"电网功率波动标准差: {grid_std:.4f} MW")
        print(f"总弃电量: {total_curt:.2f} MWh")

    return results


def run_phase3(config, realtime_df, forecast_data, start_idx, end_idx, model_path):
    """运行Phase 3（Transformer动态权重）"""
    
    print("\n🟢 运行Phase 3：Transformer动态权重")
    
    # 初始化模块
    expert_interface = ExpertInterface(config)
    feature_extractor = FeatureExtractor(config)
    transformer = TransformerController(model_path=model_path, config=config)
    mpc = MPCController(config)
    weather_classifier = WeatherClassifier(config=config)
    
    # 记录结果
    results = {
        'soc': [],
        'grid_power': [],
        'cost': [], # 这里将存储真实人民币成本（净成本：购电-售电）
        'soc_error': [],
        'grid_error': [],
        'weights': [],
        'curtail_energy': [],  # 弃电电量(MWh)
    }
    
    # 当前状态
    current_soc = 0.5
    prev_grid_power = 0.0
    # 时间步长 (小时)，直接复用 MPCController 中的配置
    dt_hours = float(getattr(mpc, "dt_hours", 0.25))

    history_buffer = []
    history_states = []
    history_len = int(getattr(feature_extractor, "history_len", 24))
    prev_expert_id = None
    time_since_switch_h = 0.0
    total_skipped = 0

    for t in tqdm(range(start_idx, end_idx), desc="Phase 3推理"):
        # 获取数据
        history_start = max(0, t - 23)
        history_data = realtime_df.iloc[history_start:t+1]

        if t < len(forecast_data):
            forecast_item = forecast_data[t]
            forecast_8h = {
                'load': forecast_item['forecast']['load'],
                'pv': forecast_item['forecast']['solar'] / 1000.0 * 35.0,
                'wind': forecast_item['forecast']['wind'] / 10.0 * 20.0,
                'price': forecast_item['forecast']['price']
            }
        else:
            total_skipped += 1
            continue

        expert_id = weather_classifier.classify_from_history(history_data)

        if prev_expert_id is None or expert_id == prev_expert_id:
            time_since_switch_h += dt_hours
        else:
            time_since_switch_h = 0.0
        prev_expert_id = expert_id

        current_state = {'soc': current_soc, 'grid_power': prev_grid_power}
        plan = expert_interface.get_plan(expert_id, current_state, forecast_8h)

        # 构造历史特征
        history_states.append({
            'soc': current_soc,
            'grid_power': prev_grid_power,
            'time': realtime_df.iloc[t]['Time']
        })
        if len(history_states) > history_len:
            history_states.pop(0)

        load_err = _compute_forecast_error_eval(realtime_df, forecast_data, t, key="load", window=4)
        pv_err = _compute_forecast_error_eval(realtime_df, forecast_data, t, key="solar", window=4)
        wind_err = _compute_forecast_error_eval(realtime_df, forecast_data, t, key="wind", window=4)
        forecast_errors = {'load': load_err, 'pv': pv_err, 'wind': wind_err}

        features = feature_extractor.extract_features(
            history_states=history_states,
            expert_plan=plan,
            forecast_errors=forecast_errors,
            expert_id=expert_id,
            expert_switch_time=time_since_switch_h
        )

        history_buffer.append(features)
        if len(history_buffer) > history_len:
            history_buffer.pop(0)

        # Transformer预测
        if len(history_buffer) == history_len:
            feature_sequence = np.stack(history_buffer)
            weights = transformer.predict_weights(feature_sequence, apply_filter=True)
        else:
            weights = {'alpha_soc': 1.0, 'alpha_grid': 1.0, 'alpha_cost': 1.0}

        # MPC求解
        reference_plan = {
            'soc': plan['soc'],
            'grid_power': plan['grid_power']
        }
        mpc_forecast = {
            'load': forecast_8h['load'][:16] / 1000,
            'pv': forecast_8h['pv'][:16],
            'wind': forecast_8h['wind'][:16],
            'price': forecast_8h['price'][:16]
        }
        
        solution = mpc.solve(
            current_state=current_state,
            forecast=mpc_forecast,
            reference_plan=reference_plan,
            dynamic_weights=weights
        )
        
        # 更新状态
        if solution['status'] == 'optimal':
            current_soc = solution['soc_plan'][0]
            current_grid = solution['grid_plan'][0]

            # --- 计算真实购电成本（优先使用实时电价） ---
            import_mw = max(0.0, -current_grid)
            if "Price_CNY_kWh" in realtime_df.columns:
                price_kwh = float(realtime_df.iloc[t]["Price_CNY_kWh"])
            else:
                price_kwh = float(mpc_forecast["price"][0])
            real_cost = import_mw * 1000.0 * price_kwh * dt_hours

            # 弃电电量(MWh)
            curtail_plan = np.asarray(solution.get("P_curtail_plan", []), dtype=float)
            curtail_mw = float(curtail_plan[0]) if curtail_plan.size > 0 else 0.0
            curtail_energy = curtail_mw * dt_hours

            results['cost'].append(real_cost)
            # ---------------------------

            soc_error = abs(current_soc - plan['soc'][0])
            grid_error = abs(current_grid - plan['grid_power'][0])
            
            results['soc'].append(current_soc)
            results['grid_power'].append(current_grid)
            results['soc_error'].append(soc_error)
            results['grid_error'].append(grid_error)
            results['weights'].append([
                weights['alpha_soc'],
                weights['alpha_grid'],
                weights['alpha_cost']
            ])
            results['curtail_energy'].append(curtail_energy)

            prev_grid_power = current_grid
        else:
            if t == start_idx + 10:
                print(f"\n⚠️  Phase 3 MPC求解失败: 状态={solution['status']}")
    
    print(f"\n📊 Phase 3统计:")
    print(f"   有效样本: {len(results['soc'])}")
    print(f"   跳过样本: {total_skipped}")
    return results


def compare_results(baseline_results, phase3_results):
    """对比结果"""
    print("\n" + "="*70)
    print("📊 性能对比")
    print("="*70)
    
    if len(baseline_results['soc']) == 0 or len(phase3_results['soc']) == 0:
        print("\n❌ 错误：没有收集到有效样本！")
        return {}
    
    metrics = {}
    
    # 总成本 (现在是真实购电成本)
    baseline_cost = np.sum(baseline_results['cost'])
    phase3_cost = np.sum(phase3_results['cost'])
    cost_reduction = (baseline_cost - phase3_cost) / baseline_cost * 100
    
    metrics['总成本'] = {
        '基线': f'{baseline_cost:.2f}',
        'Phase3': f'{phase3_cost:.2f}',
        '改善': f'{cost_reduction:+.2f}%'
    }
    
    # SOC跟踪误差
    baseline_soc_error = np.mean(baseline_results['soc_error'])
    phase3_soc_error = np.mean(phase3_results['soc_error'])
    if baseline_soc_error > 0:
        soc_error_reduction = (baseline_soc_error - phase3_soc_error) / baseline_soc_error * 100
    else:
        soc_error_reduction = 0.0
    
    metrics['SOC跟踪误差'] = {
        '基线': f'{baseline_soc_error:.4f}',
        'Phase3': f'{phase3_soc_error:.4f}',
        '改善': f'{soc_error_reduction:+.2f}%'
    }
    
    # 电网功率跟踪误差
    baseline_grid_error = np.mean(baseline_results['grid_error'])
    phase3_grid_error = np.mean(phase3_results['grid_error'])
    grid_error_reduction = (baseline_grid_error - phase3_grid_error) / baseline_grid_error * 100
    
    metrics['电网跟踪误差'] = {
        '基线': f'{baseline_grid_error:.4f}',
        'Phase3': f'{phase3_grid_error:.4f}',
        '改善': f'{grid_error_reduction:+.2f}%'
    }
    
    # 电网功率波动
    baseline_grid_std = np.std(np.diff(baseline_results['grid_power']))
    phase3_grid_std = np.std(np.diff(phase3_results['grid_power']))
    grid_std_reduction = (baseline_grid_std - phase3_grid_std) / baseline_grid_std * 100
    
    metrics['电网功率波动'] = {
        '基线': f'{baseline_grid_std:.4f}',
        'Phase3': f'{phase3_grid_std:.4f}',
        '改善': f'{grid_std_reduction:+.2f}%'
    }

    # 总弃电量(MWh)
    baseline_curt = float(np.sum(baseline_results.get('curtail_energy', [])))
    phase3_curt = float(np.sum(phase3_results.get('curtail_energy', [])))
    if baseline_curt > 0:
        curt_reduction = (baseline_curt - phase3_curt) / baseline_curt * 100
    else:
        curt_reduction = 0.0

    metrics['总弃电量(MWh)'] = {
        '基线': f'{baseline_curt:.2f}',
        'Phase3': f'{phase3_curt:.2f}',
        '改善': f'{curt_reduction:+.2f}%'
    }
    
    print(f"\n{'指标':<15} {'基线':<15} {'Phase 3':<15} {'改善':<15}")
    print("-"*70)
    for metric_name, values in metrics.items():
        print(f"{metric_name:<15} {values['基线']:<15} {values['Phase3']:<15} {values['改善']:<15}")
    
    return metrics


def summarize_baseline(baseline_results):
    if len(baseline_results['soc']) == 0:
        return
    total_cost = float(np.sum(baseline_results['cost']))
    mean_soc_err = float(np.mean(baseline_results['soc_error']))
    mean_grid_err = float(np.mean(baseline_results['grid_error']))
    grid_std = float(np.std(np.diff(baseline_results['grid_power'])))
    total_curt = float(np.sum(baseline_results.get('curtail_energy', [])))

    print("\n" + "=" * 70)
    print("📊 基线性能统计")
    print("=" * 70)
    print(f"总成本: {total_cost:.2f} (真实购电成本)")
    print(f"平均SOC跟踪误差: {mean_soc_err:.4f}")
    print(f"平均电网功率跟踪误差: {mean_grid_err:.4f} MW")
    print(f"电网功率波动标准差: {grid_std:.4f} MW")
    print(f"总弃电量: {total_curt:.2f} MWh")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None, help="评估起始日期，例如 2024-03-01")
    parser.add_argument("--end", type=str, default=None, help="评估结束日期，例如 2024-03-31")
    args = parser.parse_args()

    print("="*70)
    print("🚀 Phase 3系统评估")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'LMPC' / 'phase3_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("📋 加载数据...")
    realtime_path = project_root / 'data' / 'realtime2024.csv'
    realtime_df = pd.read_csv(realtime_path, parse_dates=['Time'])
    
    forecast_path = project_root / 'LMPC' / 'data' / 'forecast_2024_8h_testing.pkl'
    with open(forecast_path, 'rb') as f:
        forecast_data = pickle.load(f)
    
    # 评估时段
    eval_cfg = config.get("evaluation", {})
    period_cfg = eval_cfg.get("period", {})
    mode = str(period_cfg.get("mode", "range")).lower()

    if mode == "full" and not (args.start and args.end):
        start_idx = 0
        end_idx = len(realtime_df) - 1
    else:
        if args.start and args.end:
            start_time = pd.Timestamp(args.start)
            end_time = pd.Timestamp(args.end)
        else:
            start_str = period_cfg.get("start")
            end_str = period_cfg.get("end")
            if not start_str or not end_str:
                # 默认 2024-03
                start_time = pd.Timestamp("2024-03-01")
                end_time = pd.Timestamp("2024-03-31")
            else:
                start_time = pd.Timestamp(start_str)
                end_time = pd.Timestamp(end_str)

        print(f"\n📋 评估时段: {start_time.date()} 至 {end_time.date()}")
        mask = (realtime_df["Time"] >= start_time) & (realtime_df["Time"] <= end_time)
        idxs = realtime_df.index[mask]
        start_idx = int(idxs[0])
        end_idx = int(idxs[-1])

    print(f"✅ 时间步: {start_idx} - {end_idx} (共{end_idx-start_idx+1}步)")
    
    # 模型路径
    models_cfg = config.get('models', {})
    rel_model_path = models_cfg.get('transformer_weights', 'LMPC/models/transformer_weights.pth')
    model_path = project_root / rel_model_path
    
    # 运行纯 MPC 基线
    mpc_only_results = run_mpc_only(config, realtime_df, forecast_data, start_idx, end_idx)

    # 运行当前基线（专家+固定权重 MPC）
    baseline_results = run_baseline(config, realtime_df, forecast_data, start_idx, end_idx)
    summarize_baseline(baseline_results)
    
    # 运行 Phase 3（专家+MPC+Transformer 动态权重）
    phase3_results = run_phase3(config, realtime_df, forecast_data, start_idx, end_idx, str(model_path))
    
    # 对比：仍以“当前基线 vs Phase3”为主
    metrics_summary = compare_results(baseline_results, phase3_results)

    # ==========================================
    # 新增：保存详细数据供 GUI 展示
    # ==========================================
    print("\n💾 正在保存详细结果用于可视化...")

    # 根据评估时间段动态命名结果文件，避免不同实验互相覆盖
    if args.start and args.end:
        period_label = f"{args.start}_to_{args.end}".replace("-", "")
    else:
        cfg_start = str(period_cfg.get("start", "")).strip()
        cfg_end = str(period_cfg.get("end", "")).strip()
        if mode == "full" and not (cfg_start and cfg_end):
            period_label = "full"
        elif cfg_start and cfg_end:
            period_label = f"{cfg_start}_to_{cfg_end}".replace("-", "")
        else:
            period_label = "unknown_period"

    filename = f"eval_results_{period_label}.pkl"
    save_path = project_root / 'LMPC' / 'logs' / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    time_index = realtime_df.iloc[start_idx:end_idx+1]['Time'].values

    viz_data = {
        'time': time_index[:len(baseline_results['soc'])],
        'baseline': baseline_results,
        'phase3': phase3_results,
        'metrics': metrics_summary
    }

    with open(save_path, 'wb') as f:
        pickle.dump(viz_data, f)

    print(f"✅ 结果已保存至: {save_path}")
    print("   现在可以运行: streamlit run dashboard.py 查看可视化界面")
    print("\n✅ 系统评估完成！")


if __name__ == '__main__':
    main()