"""Phase 3 环境测试脚本

验证：
1. 目录结构是否正确
2. 配置文件是否可读
3. 依赖库是否安装
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_directory_structure():
    """测试目录结构"""
    print("\n📁 测试目录结构...")
    
    lmpc_dir = project_root / "LMPC"
    required_dirs = ['core', 'training', 'utils', 'scripts', 'data', 'models', 'logs']
    
    for dir_name in required_dirs:
        dir_path = lmpc_dir / dir_name
        if dir_path.exists():
            print(f"   ✅ {dir_name}/ 存在")
        else:
            print(f"   ❌ {dir_name}/ 不存在")
            return False
    
    return True


def test_config_file():
    """测试配置文件"""
    print("\n⚙️  测试配置文件...")
    
    try:
        import yaml
        config_path = project_root / "LMPC" / "phase3_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查关键配置项
        required_keys = ['data', 'models', 'capacity', 'mpc', 'transformer']
        for key in required_keys:
            if key in config:
                print(f"   ✅ {key} 配置存在")
            else:
                print(f"   ❌ {key} 配置缺失")
                return False
        
        return True
    
    except Exception as e:
        print(f"   ❌ 配置文件读取失败: {e}")
        return False


def test_dependencies():
    """测试依赖库"""
    print("\n📦 测试依赖库...")
    
    dependencies = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'yaml': 'PyYAML',
        'cvxpy': 'CVXPY (MPC求解器)',
        'sklearn': 'Scikit-learn',
        'tqdm': 'TQDM',
        'matplotlib': 'Matplotlib'
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} 未安装")
            all_ok = False
    
    return all_ok


def test_data_files():
    """测试数据文件"""
    print("\n📊 测试数据文件...")
    
    data_dir = project_root / "data"
    required_files = ['data2023.csv', 'realtime2024.csv']
    
    all_ok = True
    for filename in required_files:
        file_path = data_dir / filename
        if file_path.exists():
            import pandas as pd
            df = pd.read_csv(file_path)
            print(f"   ✅ {filename} (共 {len(df)} 行)")
        else:
            print(f"   ❌ {filename} 不存在")
            all_ok = False
    
    return all_ok


def test_sac_models():
    """测试SAC专家模型"""
    print("\n🤖 测试SAC专家模型...")
    
    sac_dir = project_root / "SAC" / "models"
    
    if not sac_dir.exists():
        print(f"   ❌ SAC模型目录不存在")
        return False
    
    all_ok = True
    for i in range(5):
        actor_path = sac_dir / f"expert_{i}_actor.pth"
        if actor_path.exists():
            print(f"   ✅ 专家{i}模型存在")
        else:
            print(f"   ❌ 专家{i}模型缺失")
            all_ok = False
    
    return all_ok


def main():
    """主测试函数"""
    print("="*60)
    print("Phase 3 环境测试")
    print("="*60)
    
    results = {
        '目录结构': test_directory_structure(),
        '配置文件': test_config_file(),
        '依赖库': test_dependencies(),
        '数据文件': test_data_files(),
        'SAC模型': test_sac_models()
    }
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 所有测试通过！环境配置正确。")
        print("\n下一步:")
        print("   1. 运行 python LMPC/scripts/generate_forecast_data.py --year 2023 --output data/forecast_2023_8h_training.pkl")
        print("   2. 运行 python LMPC/scripts/generate_forecast_data.py --year 2024 --output data/forecast_2024_8h_testing.pkl")
    else:
        print("\n⚠️ 部分测试失败，请检查环境配置。")
    
    print("="*60)


if __name__ == '__main__':
    main()
