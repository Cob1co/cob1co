"""基于2023年数据生成2024年15分钟级实时数据

特点：
1. 从1小时分辨率插值到15分钟分辨率
2. 添加一些天气变化（约10%的天数）
3. 保持列名格式与2023年一致
4. 添加随机小波动模拟真实性
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm


class Realtime2024Generator:
    """2024年实时数据生成器"""
    
    def __init__(self, data_2023_path='data/data2023.csv'):
        """
        参数:
            data_2023_path: 2023年数据路径
        """
        print(f"\n📂 加载2023年数据: {data_2023_path}")
        self.df_2023 = pd.read_csv(data_2023_path)
        print(f"   原始数据: {len(self.df_2023)} 行 (1小时分辨率)")
        
        # 解析时间
        self.df_2023['Time'] = pd.to_datetime(self.df_2023['Time'])
    
    def interpolate_to_15min(self):
        """
        将1小时数据插值到15分钟
        
        返回:
            df_15min: DataFrame, 15分钟分辨率数据
        """
        print(f"\n🔄 插值到15分钟分辨率...")
        
        # 创建15分钟时间序列
        start_time = self.df_2023['Time'].iloc[0]
        end_time = self.df_2023['Time'].iloc[-1]
        time_15min = pd.date_range(start=start_time, end=end_time, freq='15min')
        
        # 创建新的DataFrame
        df_15min = pd.DataFrame({'Time': time_15min})
        
        # 对每个变量进行线性插值
        for col in ['Temperature_C', 'Solar_W_m2', 'Wind_Speed_m_s', 'Load_kW', 'Price_CNY_kWh']:
            # 创建1小时数据的时间索引
            df_hourly = self.df_2023.set_index('Time')[col]
            
            # 重采样并线性插值
            df_resampled = df_hourly.reindex(
                df_hourly.index.union(time_15min)
            ).interpolate(method='linear')
            
            # 提取15分钟点
            df_15min[col] = df_resampled.reindex(time_15min).values
        
        print(f"   ✅ 插值完成: {len(df_15min)} 行 (15分钟分辨率)")
        return df_15min
    
    def add_small_variations(self, df):
        """
        添加小的随机波动，模拟真实数据
        
        参数:
            df: DataFrame
        
        返回:
            df: 添加波动后的DataFrame
        """
        print(f"\n🌊 添加随机小波动...")
        
        df = df.copy()
        
        # 为每个变量添加小波动
        # 负荷：±2%
        df['Load_kW'] *= (1 + np.random.normal(0, 0.02, len(df)))
        df['Load_kW'] = df['Load_kW'].clip(lower=0)
        
        # 温度：±0.5°C
        df['Temperature_C'] += np.random.normal(0, 0.5, len(df))
        
        # 辐照度：±5%
        df['Solar_W_m2'] *= (1 + np.random.normal(0, 0.05, len(df)))
        df['Solar_W_m2'] = df['Solar_W_m2'].clip(lower=0)
        
        # 风速：±3%
        df['Wind_Speed_m_s'] *= (1 + np.random.normal(0, 0.03, len(df)))
        df['Wind_Speed_m_s'] = df['Wind_Speed_m_s'].clip(lower=0)
        
        # 电价：±1%
        df['Price_CNY_kWh'] *= (1 + np.random.normal(0, 0.01, len(df)))
        df['Price_CNY_kWh'] = df['Price_CNY_kWh'].clip(lower=0.1)
        
        print(f"   ✅ 波动已添加")
        return df
    
    def add_weather_changes(self, df):
        """
        为约10%的天数添加天气变化
        
        参数:
            df: DataFrame
        
        返回:
            df: 修改后的DataFrame
        """
        print(f"\n🌦️  添加天气变化...")
        
        df = df.copy()
        df['Day'] = df['Time'].dt.date
        unique_days = df['Day'].unique()
        num_days = len(unique_days)
        
        # 随机选择10%的天数进行天气变化
        num_changes = int(num_days * 0.1)
        changed_days = np.random.choice(unique_days, num_changes, replace=False)
        
        weather_scenarios = [
            ('晴转多云', lambda row, hour: self._sunny_to_cloudy(row, hour)),
            ('多云转晴', lambda row, hour: self._cloudy_to_sunny(row, hour)),
            ('晴天转雨', lambda row, hour: self._sunny_to_rainy(row, hour)),
            ('雨转晴', lambda row, hour: self._rainy_to_sunny(row, hour)),
            ('多风转微风', lambda row, hour: self._windy_to_calm(row, hour)),
            ('微风转多风', lambda row, hour: self._calm_to_windy(row, hour)),
        ]
        
        change_log = []
        
        for day in tqdm(changed_days, desc="修改天气"):
            # 随机选择天气变化类型
            scenario_name, scenario_func = random.choice(weather_scenarios)
            
            # 获取当天的所有行
            day_mask = df['Day'] == day
            day_indices = df[day_mask].index
            
            # 应用天气变化
            for idx in day_indices:
                hour = df.loc[idx, 'Time'].hour + df.loc[idx, 'Time'].minute / 60
                df.loc[idx] = scenario_func(df.loc[idx], hour)
            
            change_log.append({
                'date': day,
                'scenario': scenario_name
            })
        
        print(f"   ✅ 已修改 {len(changed_days)} 天的天气")
        print(f"   示例变化:")
        for i, log in enumerate(change_log[:5]):
            print(f"      {log['date']}: {log['scenario']}")
        
        return df.drop(columns=['Day']), change_log
    
    def _sunny_to_cloudy(self, row, hour):
        """晴转多云：光照逐渐减弱"""
        row = row.copy()
        # 下午光照衰减40-60%
        if 12 <= hour <= 18:
            reduction = 0.4 + 0.2 * (hour - 12) / 6
            row['Solar_W_m2'] *= (1 - reduction)
        return row
    
    def _cloudy_to_sunny(self, row, hour):
        """多云转晴：光照逐渐增强"""
        row = row.copy()
        # 上午光照增强20-40%
        if 8 <= hour <= 14:
            increase = 0.2 + 0.2 * (hour - 8) / 6
            row['Solar_W_m2'] *= (1 + increase)
        return row
    
    def _sunny_to_rainy(self, row, hour):
        """晴天转雨：光照大幅减弱，温度下降"""
        row = row.copy()
        # 下午开始下雨
        if hour >= 14:
            row['Solar_W_m2'] *= 0.2  # 光照减少80%
            row['Temperature_C'] -= 5  # 温度下降5度
            row['Wind_Speed_m_s'] *= 1.3  # 风速增加
        return row
    
    def _rainy_to_sunny(self, row, hour):
        """雨转晴：光照恢复，温度回升"""
        row = row.copy()
        # 上午还在下雨，下午放晴
        if hour < 12:
            row['Solar_W_m2'] *= 0.3
            row['Temperature_C'] -= 3
        else:
            # 逐渐恢复
            recovery = (hour - 12) / 6
            row['Solar_W_m2'] *= (0.3 + 0.7 * recovery)
            row['Temperature_C'] -= 3 * (1 - recovery)
        return row
    
    def _windy_to_calm(self, row, hour):
        """多风转微风：风速减弱"""
        row = row.copy()
        # 全天风速逐渐减弱
        row['Wind_Speed_m_s'] *= (0.3 + 0.2 * np.random.random())
        return row
    
    def _calm_to_windy(self, row, hour):
        """微风转多风：风速增强"""
        row = row.copy()
        # 下午风速增强
        if hour >= 10:
            row['Wind_Speed_m_s'] *= (1.5 + 0.5 * np.random.random())
        return row
    
    def convert_to_2024(self, df):
        """
        将时间戳转换为2024年
        
        参数:
            df: DataFrame
        
        返回:
            df_2024: 2024年数据
        """
        print(f"\n📅 转换时间戳到2024年...")
        
        df_2024 = df.copy()
        df_2024['Time'] = df_2024['Time'].apply(
            lambda x: x.replace(year=2024)
        )
        
        print(f"   ✅ 时间范围: {df_2024['Time'].iloc[0]} 到 {df_2024['Time'].iloc[-1]}")
        return df_2024
    
    def round_to_match_2023(self, df):
        """
        四舍五入到与2023年相同的小数位数
        
        参数:
            df: DataFrame
        
        返回:
            df: 四舍五入后的DataFrame
        """
        print(f"\n🔢 四舍五入到标准精度...")
        
        df = df.copy()
        df['Temperature_C'] = df['Temperature_C'].round(2)    # 2位小数
        df['Solar_W_m2'] = df['Solar_W_m2'].round(1)          # 1位小数
        df['Wind_Speed_m_s'] = df['Wind_Speed_m_s'].round(1)  # 1位小数
        df['Price_CNY_kWh'] = df['Price_CNY_kWh'].round(3)    # 3位小数
        df['Load_kW'] = df['Load_kW'].round(3)                # 3位小数
        
        print(f"   ✅ 精度已统一")
        return df
    
    def generate(self, output_path='data/realtime2024.csv'):
        """
        生成完整的2024年实时数据
        
        参数:
            output_path: 输出文件路径
        
        返回:
            df_2024: 2024年15分钟数据
            change_log: 天气变化记录
        """
        # 1. 插值到15分钟
        df_15min = self.interpolate_to_15min()
        
        # 2. 添加小波动
        df_varied = self.add_small_variations(df_15min)
        
        # 3. 添加天气变化
        df_weather_changed, change_log = self.add_weather_changes(df_varied)
        
        # 4. 转换到2024年
        df_2024 = self.convert_to_2024(df_weather_changed)
        
        # 5. 四舍五入到标准精度
        df_2024 = self.round_to_match_2023(df_2024)
        
        # 6. 保存
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df_2024.to_csv(output_path, index=False)
        print(f"\n✅ 2024年实时数据已保存: {output_path}")
        print(f"   数据量: {len(df_2024)} 行")
        print(f"   时间跨度: {len(df_2024) / 96:.1f} 天")
        
        # 保存天气变化日志
        log_path = Path(output_path).parent / 'weather_changes_2024.csv'
        pd.DataFrame(change_log).to_csv(log_path, index=False)
        print(f"   天气变化日志: {log_path}")
        
        return df_2024, change_log


def main():
    """主函数"""
    # 确定项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # 路径
    data_2023_path = project_root / 'data' / 'data2023.csv'
    output_path = project_root / 'data' / 'realtime2024.csv'
    
    # 生成器
    generator = Realtime2024Generator(data_2023_path)
    
    # 生成数据
    df_2024, change_log = generator.generate(output_path)
    
    # 统计信息
    print("\n📊 数据统计:")
    print(f"   负荷范围: {df_2024['Load_kW'].min():.1f} - {df_2024['Load_kW'].max():.1f} kW")
    print(f"   温度范围: {df_2024['Temperature_C'].min():.1f} - {df_2024['Temperature_C'].max():.1f} °C")
    print(f"   辐照度范围: {df_2024['Solar_W_m2'].min():.1f} - {df_2024['Solar_W_m2'].max():.1f} W/m²")
    print(f"   风速范围: {df_2024['Wind_Speed_m_s'].min():.1f} - {df_2024['Wind_Speed_m_s'].max():.1f} m/s")
    print(f"   电价范围: {df_2024['Price_CNY_kWh'].min():.3f} - {df_2024['Price_CNY_kWh'].max():.3f} 元/kWh")
    
    print(f"\n🎉 完成！")


if __name__ == '__main__':
    main()
