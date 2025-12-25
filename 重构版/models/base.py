"""设备模型基础抽象类。

所有具体设备模型都应继承 Component，并实现 reset / step 方法。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class Component(ABC):
    """设备组件抽象基类。"""

    @abstractmethod
    def reset(self, **kwargs: Any) -> None:
        """重置设备状态。

        参数使用关键字形式，保持灵活。
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, inputs: Dict[str, Any], dt_hours: float = 1.0) -> Dict[str, Any]:
        """单步仿真。

        inputs: 输入量字典，例如功率、环境量等。
        dt_hours: 当前时间步长（小时）。
        返回: 输出量字典。
        """
        raise NotImplementedError
