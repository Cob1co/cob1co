"""生成带预测误差的数据

为历史数据添加预测误差，模拟真实预测场景
- 2023年数据：用于Transformer训练
- 2024年数据：用于实时测试

误差特征：
1. 误差随预测步长增加而增大
2. 光伏/风电误差 > 负荷误差
3. 添加随机波动
"""

import pandas as pd
import numpy as np
import yaml
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse


class ForecastDataGenerator:
    """预测数据生成器"""
    
    def __init__(self, config_path='phase3_config.yaml'):
        """
        参数:
            config_path: 配置文件路径
        """
        # 如果是相对路径，从LMPC目录查找
        config_file = Path(config_path)
        if not config_file.is_absolute() and not config_file.exists():
            # 尝试从脚本目录的上级目录查找
            script_dir = Path(__file__).parent
            config_file = script_dir.parent / config_path
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.error_cfg = self.config['forecast_error']
        self.horizon_steps = self.error_cfg['horizon_steps']  # 32步
        self.horizon_hours = self.error_cfg['horizon_hours']  # 8小时
        
        # 误差配置
        self.load_base = self.error_cfg['load']['base_std']
        self.load_growth = self.error_cfg['load']['growth_rate']
        
        self.pv_base = self.error_cfg['pv']['base_std']
        self.pv_growth = self.error_cfg['pv']['growth_rate']
        
        self.wind_base = self.error_cfg['wind']['base_std']
        self.wind_growth = self.error_cfg['wind']['growth_rate']
    
    def compute_error_std(self, step, base_std, growth_rate):
        """
        计算误差标准差（随步长增加）
        
        参数:
            step: 预测步长 (0-31)
            base_std: 基础标准差
            growth_rate: 增长率
        
        返回:
            当前步的标准差
        """
        # 线性增长: std = base + growth * (step / total_steps)
        hour_ahead = step * 0.25  # 15分钟步长
        return base_std + growth_rate * (hour_ahead / self.horizon_hours)
    
    def add_forecast_error(self, real_value, step, base_std, growth_rate, min_value=0.0, decimals=3):
        """
        为真实值添加预测误差
        
        参数:
            real_value: 真实值
            step: 预测步长
            base_std: 基础标准差
            growth_rate: 增长率
            min_value: 最小值约束
            decimals: 小数位数
        
        返回:
            预测值（带误差，四舍五入）
        """
        std = self.compute_error_std(step, base_std, growth_rate)
        error = np.random.normal(0, std)
        forecast = real_value * (1 + error)
        return round(max(min_value, forecast), decimals)
    
    def _detect_data_format(self, real_data):
        """
        检测数据格式（2023原始格式 vs 2024处理格式）
        
        返回:
            column_map: Dict, 列名映射
        """
        columns = real_data.columns.tolist()
        
        # 2023/2024年格式：Load_kW, Solar_W_m2, Wind_Speed_m_s, Price_CNY_kWh
        if 'Load_kW' in columns:
            return {
                'load': 'Load_kW',
                'solar': 'Solar_W_m2',
                'wind': 'Wind_Speed_m_s',
                'price': 'Price_CNY_kWh'
            }
        else:
            raise ValueError(f"未知的数据格式！列名: {columns}")
    
    def generate_forecast(self, real_data, output_path):
        """
        为整个数据集生成预测数据
        
        参数:
            real_data: DataFrame, 真实数据
            output_path: 输出文件路径
        
        返回:
            forecast_data: List[Dict], 预测数据列表
        """
        # 检测数据格式
        self.col_map = self._detect_data_format(real_data)
        
        print(f"\n🔮 开始生成预测数据...")
        print(f"   数据长度: {len(real_data)} 条")
        print(f"   预测窗口: {self.horizon_hours} 小时 ({self.horizon_steps} 步)")
        
        forecast_data = []
        
        # 确保有足够的未来数据
        valid_length = len(real_data) - self.horizon_steps
        
        for t in tqdm(range(valid_length), desc="生成预测"):
            # 获取未来8小时的真实数据
            real_future = real_data.iloc[t:t+self.horizon_steps].copy()
            
            # 为每个变量添加误差
            forecast_horizon = {
                'Load_Forecast': [],
                'Solar_Forecast': [],
                'Wind_Forecast': [],
                'Price_Forecast': []  # 电价通常预测较准，误差小
            }
            
            for step in range(self.horizon_steps):
                row = real_future.iloc[step]
                
                # 负荷预测 (3位小数)
                load_forecast = self.add_forecast_error(
                    row[self.col_map['load']], step, self.load_base, self.load_growth, decimals=3
                )
                forecast_horizon['Load_Forecast'].append(load_forecast)
                
                # 光伏/太阳预测 (1位小数)
                solar_forecast = self.add_forecast_error(
                    row[self.col_map['solar']], step, self.pv_base, self.pv_growth, decimals=1
                )
                forecast_horizon['Solar_Forecast'].append(solar_forecast)
                
                # 风电预测 (1位小数)
                wind_forecast = self.add_forecast_error(
                    row[self.col_map['wind']], step, self.wind_base, self.wind_growth, decimals=1
                )
                forecast_horizon['Wind_Forecast'].append(wind_forecast)
                
                # 电价预测（误差较小，2-5%，3位小数）
                price_forecast = self.add_forecast_error(
                    row[self.col_map['price']], step, 0.02, 0.03, min_value=0.1, decimals=3
                )
                forecast_horizon['Price_Forecast'].append(price_forecast)
            
            # 转为numpy数组
            forecast_data.append({
                'timestamp_origin': t,
                'forecast': {
                    'load': np.array(forecast_horizon['Load_Forecast']),
                    'solar': np.array(forecast_horizon['Solar_Forecast']),
                    'wind': np.array(forecast_horizon['Wind_Forecast']),
                    'price': np.array(forecast_horizon['Price_Forecast'])
                },
                # 保存真实值用于计算误差
                'real': {
                    'load': real_future[self.col_map['load']].values,
                    'solar': real_future[self.col_map['solar']].values,
                    'wind': real_future[self.col_map['wind']].values,
                    'price': real_future[self.col_map['price']].values
                }
            })
        
        # 保存
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(forecast_data, f)
        
        print(f"\n✅ 预测数据已保存: {output_path}")
        print(f"   样本数量: {len(forecast_data)}")
        
        # 统计误差
        self._print_error_statistics(forecast_data)
        
        return forecast_data
    
    def _print_error_statistics(self, forecast_data):
        """打印误差统计信息"""
        print("\n📊 误差统计 (MAPE - Mean Absolute Percentage Error):")
        
        for var_name in ['load', 'solar', 'wind']:
            errors_1h = []   # 1小时前预测误差
            errors_4h = []   # 4小时前预测误差
            errors_8h = []   # 8小时前预测误差
            
            for sample in forecast_data[:1000]:  # 随机抽样1000个
                real = sample['real'][var_name]
                forecast = sample['forecast'][var_name]
                
                # 计算不同步长的误差
                if real[4] > 0:  # 1小时
                    errors_1h.append(abs(forecast[4] - real[4]) / real[4])
                if real[16] > 0:  # 4小时
                    errors_4h.append(abs(forecast[16] - real[16]) / real[16])
                if real[31] > 0:  # 8小时
                    errors_8h.append(abs(forecast[31] - real[31]) / real[31])
            
            print(f"   {var_name:6s}: 1h={np.mean(errors_1h)*100:.1f}%  "
                  f"4h={np.mean(errors_4h)*100:.1f}%  "
                  f"8h={np.mean(errors_8h)*100:.1f}%")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成带预测误差的数据')
    parser.add_argument('--year', type=int, required=True, 
                       help='数据年份 (2023 or 2024)')
    parser.add_argument('--output', type=str, required=True,
                       help='输出文件路径 (.pkl)')
    parser.add_argument('--config', type=str, default='phase3_config.yaml',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 初始化生成器
    generator = ForecastDataGenerator(args.config)
    
    # 确定项目根目录（从脚本位置向上两级）
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # 加载真实数据
    if args.year == 2023:
        data_path = project_root / 'data' / 'data2023.csv'
    elif args.year == 2024:
        data_path = project_root / 'data' / 'realtime2024.csv'
    else:
        raise ValueError(f"不支持的年份: {args.year}")
    
    print(f"\n📂 加载数据: {data_path}")
    real_data = pd.read_csv(data_path)
    
    # 生成预测数据
    generator.generate_forecast(real_data, args.output)
    
    print(f"\n🎉 完成！")


if __name__ == '__main__':
    main()
