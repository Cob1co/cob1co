"""模型参数加载模块。

所有设备相关的物理参数和经济参数统一从本目录下的 config.yaml 读取。
"""

from pathlib import Path
from typing import Any, Dict

import yaml


# 当前文件所在目录
_BASE_DIR = Path(__file__).resolve().parent
# 配置文件路径
_CONFIG_PATH = _BASE_DIR / "config.yaml"


def load_model_config() -> Dict[str, Any]:
    """加载模型配置。

    返回一个字典，包含 project/physics/assets/economics 四大块。
    """
    with _CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}
    return cfg


def get_section(name: str) -> Dict[str, Any]:
    """按名称获取单个一级配置块，例如 "assets"、"physics"。"""
    cfg = load_model_config()
    sec = cfg.get(name, {})
    if not isinstance(sec, dict):
        return {}
    return sec


def get_assets() -> Dict[str, Any]:
    """获取 assets 块。"""
    return get_section("assets")


def get_physics() -> Dict[str, Any]:
    """获取 physics 块。"""
    return get_section("physics")


def get_economics() -> Dict[str, Any]:
    """获取 economics 块。"""
    return get_section("economics")
