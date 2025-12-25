import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_ideal_scenario(output_file='ideal_data_2023.csv'):
    # 1. 时间轴设置 (2023全年, 小时级)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31 23:00:00', freq='H')
    df = pd.DataFrame({'Time': dates})
    n = len(df)
    
    # 辅助变量
    hours = df['Time'].dt.hour
    day_of_year = df['Time'].dt.dayofyear
    month = df['Time'].dt.month

    print("正在生成数据...")

    # =====================================================
    # 2. 模拟环境数据 (温度 & 光照 & 风速)
    # =====================================================
    
    # --- 温度 (Temperature_C) ---
    # 逻辑：冬季寒冷(-15度)刺激供暖负荷，夏季炎热(35度)刺激制冷负荷
    # 季节性趋势 + 日内波动 + 随机噪声
    seasonal_temp = -10 * np.cos(2 * np.pi * day_of_year / 365) + 12 # 年均温12度
    daily_temp = 5 * np.sin(2 * np.pi * (hours - 9) / 24) # 下午3点最热
    noise_temp = np.random.normal(0, 2, n)
    df['Temperature_C'] = (seasonal_temp + daily_temp + noise_temp).round(2)

    # --- 光照 (Solar_W_m2) ---
    # 逻辑：支撑 70MW 光伏。夏天强，冬天弱。
    # 太阳高度角模拟
    # 简单的日照模型：最大值1000 W/m2
    # 季节影响：夏天系数1.2，冬天0.6
    season_solar_factor = 1 - 0.3 * np.cos(2 * np.pi * (day_of_year - 172) / 365) 
    
    # 日内分布 (6点到19点有光)
    # 使用 clip(0) 去除负值
    hour_solar = np.clip(np.sin(np.pi * (hours - 6) / 13), 0, 1)
    
    # 天气干扰 (模拟阴雨天，随机给某些天打折)
    weather_factor = np.random.choice([1.0, 0.8, 0.2], size=n, p=[0.7, 0.2, 0.1])
    
    # 最终光照
    df['Solar_W_m2'] = (1000 * season_solar_factor * hour_solar * weather_factor).round(2)

    # --- 风速 (Wind_Speed_m_s) ---
    # 逻辑：支撑 30MW 风电。这就要求平均风速在 6.5 m/s 左右。
    # 使用 Weibull 分布模拟真实风速，增加一点季节性（冬天风大）
    
    # 基础随机风速 (Weibull分布, shape=2, scale=7)
    raw_wind = np.random.weibull(2.2, n) * 7.5
    
    # 季节修正 (冬天风大一些)
    season_wind_mod = 1 + 0.2 * np.cos(2 * np.pi * day_of_year / 365)
    
    # 日内修正 (通常下午/晚上风大一点)
    daily_wind_mod = 1 + 0.1 * np.sin(2 * np.pi * (hours - 14) / 24)
    
    df['Wind_Speed_m_s'] = (raw_wind * season_wind_mod * daily_wind_mod).round(2)
    # 截断极端值，风机切出风速通常25m/s
    df['Wind_Speed_m_s'] = df['Wind_Speed_m_s'].clip(0, 25)


    # =====================================================
    # 3. 模拟 电价 (Price_CNY_kWh)
    # =====================================================
    # 典型的工业分时电价，价差要大，鼓励储能
    # 谷(0.30): 23-7 (8小时, 适合加热器全速开启)
    # 平(0.65): 7-10, 15-18, 21-23
    # 峰(1.20): 10-15, 18-21 (适合汽轮机放电)
    
    def get_price_vectorized(h):
        conditions = [
            (h >= 23) | (h < 7),  # 谷
            (h >= 10) & (h < 15) | (h >= 18) & (h < 21) # 峰
        ]
        choices = [0.30, 1.20]
        return np.select(conditions, choices, default=0.65)

    # 加一点随机浮动，显得真实
    price_base = get_price_vectorized(hours.values)
    price_noise = np.random.uniform(-0.01, 0.01, n)
    df['Price_CNY_kWh'] = (price_base + price_noise).round(3)


    # =====================================================
    # 4. 模拟 负荷 (Load_kW) - 关键步骤
    # =====================================================
    # 目标：让光伏(70MW)和风电(30MW)能覆盖大部分，
    # 且夜间负荷约为 20MW (汽轮机容量)，
    # 白天峰值负荷约为 50MW (光伏盈余 20MW 给加热器)。
    
    # A. 基础负荷 (Base Load): 工业连续生产，夜间也有
    base_load = 18000 # 18 MW
    
    # B. 工作负荷 (Work Load): 8:00 - 18:00
    # 正弦波模拟上班高峰
    work_curve = np.clip(np.sin(np.pi * (hours - 8) / 10), 0, 1)
    work_load = work_curve * 25000 # 峰值增加 25 MW -> 总计 43 MW
    
    # C. 生活晚高峰 (Living Load): 18:00 - 22:00
    living_curve = np.clip(np.sin(np.pi * (hours - 18) / 4), 0, 1)
    living_load = living_curve * 10000 # 增加 10 MW
    
    # D. 温控负荷 (HVAC): 
    # 你的储能有200MWh热容，说明可能还有供热需求或者电采暖
    # 设定：温度 < 10度 或 > 26度 时负荷增加
    hvac_load = np.zeros(n)
    # 供热 (冬季耗电极大)
    mask_heat = df['Temperature_C'] < 10
    hvac_load[mask_heat] = (10 - df['Temperature_C'][mask_heat]) * 800 # 每低1度增加0.8MW
    # 制冷 (夏季)
    mask_cool = df['Temperature_C'] > 26
    hvac_load[mask_cool] = (df['Temperature_C'][mask_cool] - 26) * 1200 # 每高1度增加1.2MW
    
    # 合成总负荷 + 随机波动
    total_load = base_load + work_load + living_load + hvac_load
    load_noise = np.random.normal(0, 1000, n) # 1MW 左右的随机波动
    
    df['Load_kW'] = (total_load + load_noise).round(2)
    # 确保负荷不为负
    df['Load_kW'] = df['Load_kW'].clip(lower=5000)

    # =====================================================
    # 5. 验证数据特征 (自检环节)
    # =====================================================
    print("-" * 30)
    print("数据生成完毕，特征自检：")
    print(f"1. 风速均值: {df['Wind_Speed_m_s'].mean():.2f} m/s (目标 > 6.5，支撑30MW风电)")
    print(f"2. 辐照均值: {df['Solar_W_m2'].mean():.2f} W/m2 (支撑70MW光伏)")
    print(f"3. 负荷均值: {df['Load_kW'].mean()/1000:.2f} MW")
    print(f"4. 负荷峰值: {df['Load_kW'].max()/1000:.2f} MW (目标 50-60MW)")
    print(f"5. 负荷谷值: {df['Load_kW'].min()/1000:.2f} MW (目标 15-20MW，匹配汽轮机)")
    print("-" * 30)

    # 保存
    df.to_csv(output_file, index=False)
    print(f"文件已保存至: {output_file}")

if __name__ == "__main__":
    generate_ideal_scenario()