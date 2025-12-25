"""第一阶段容量配置的启发式优化器（MUGH 风格简化版）。

- 使用改进的粒子群算法（PSO）在容量空间中搜索；
- 速度更新公式中加入经济模块给出的边际效益向量 MUI；
- 目标函数来自 economics_engine.evaluate_capacity。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import random

import pandas as pd

from economics_engine import evaluate_capacity


CapacityDict = Dict[str, float]


@dataclass
class Particle:
    position: CapacityDict
    velocity: CapacityDict
    best_position: CapacityDict
    best_obj: float


class MUGHOptimizer:
    """基于边际效益修正的简单 PSO 优化器。"""

    def __init__(self, phase1_cfg: Dict[str, Any], scenarios: List[Tuple[pd.DataFrame, float]]):
        self.cfg = phase1_cfg
        self.scenarios = scenarios

        opt_cfg = phase1_cfg.get("optimizer", {})
        self.n_particles = int(opt_cfg.get("n_particles", 20))
        self.max_iter = int(opt_cfg.get("max_iter", 50))
        self.w = float(opt_cfg.get("inertia_w", 0.7))
        self.c1 = float(opt_cfg.get("c1", 1.4))
        self.c2 = float(opt_cfg.get("c2", 1.4))
        self.alpha_mu = float(opt_cfg.get("alpha_mu", 0.3))

        self.bounds = self.cfg.get("capacity_bounds", {})
        self.keys = ["wind_mw", "pv_mw", "ts_mwh", "eh_mw_th", "st_mw_e"]

    def _random_capacity(self) -> CapacityDict:
        cap: CapacityDict = {}
        for k in self.keys:
            b = self.bounds.get(k, {})
            vmin = float(b.get("min", 0.0))
            vmax = float(b.get("max", 0.0))
            if vmax <= vmin:
                cap[k] = vmin
            else:
                cap[k] = random.uniform(vmin, vmax)
        return cap

    def _zero_velocity(self) -> CapacityDict:
        return {k: 0.0 for k in self.keys}

    def _clip_capacity(self, cap: CapacityDict) -> CapacityDict:
        out: CapacityDict = {}
        for k in self.keys:
            v = float(cap.get(k, 0.0))
            b = self.bounds.get(k, {})
            vmin = float(b.get("min", 0.0))
            vmax = float(b.get("max", 0.0))
            if v < vmin:
                v = vmin
            if v > vmax:
                v = vmax
            out[k] = v
        return out

    def _add(self, a: CapacityDict, b: CapacityDict) -> CapacityDict:
        return {k: float(a.get(k, 0.0)) + float(b.get(k, 0.0)) for k in self.keys}

    def _sub(self, a: CapacityDict, b: CapacityDict) -> CapacityDict:
        return {k: float(a.get(k, 0.0)) - float(b.get(k, 0.0)) for k in self.keys}

    def _scale(self, a: CapacityDict, s: float) -> CapacityDict:
        return {k: float(a.get(k, 0.0)) * s for k in self.keys}

    def optimize(self) -> Tuple[CapacityDict, Dict[str, Any]]:
        """执行优化，返回最优容量向量和对应指标。"""
        # 初始化粒子群
        particles: List[Particle] = []
        global_best_pos: CapacityDict | None = None
        global_best_obj: float | None = None
        global_best_metrics: Dict[str, Any] = {}

        for _ in range(self.n_particles):
            pos = self._random_capacity()
            vel = self._zero_velocity()
            obj, metrics, _mui = evaluate_capacity(pos, self.scenarios, self.cfg)
            p = Particle(
                position=pos,
                velocity=vel,
                best_position=pos.copy(),
                best_obj=obj,
            )
            particles.append(p)

            if global_best_obj is None or obj < global_best_obj:
                global_best_obj = obj
                global_best_pos = pos.copy()
                global_best_metrics = metrics

        if global_best_pos is None:
            # 理论上不会发生
            return self._random_capacity(), {}

        # 迭代更新
        for _iter in range(self.max_iter):
            for p in particles:
                # 评估当前粒子
                obj, metrics, mui = evaluate_capacity(p.position, self.scenarios, self.cfg)

                # 更新个体最优
                if obj < p.best_obj:
                    p.best_obj = obj
                    p.best_position = p.position.copy()

                # 更新全局最优
                if global_best_obj is None or obj < global_best_obj:
                    global_best_obj = obj
                    global_best_pos = p.position.copy()
                    global_best_metrics = metrics

                # PSO 速度更新
                r1 = random.random()
                r2 = random.random()

                # 经典 PSO 部分
                cognitive = self._scale(self._sub(p.best_position, p.position), self.c1 * r1)
                social = self._scale(self._sub(global_best_pos, p.position), self.c2 * r2)
                v_inertia = self._scale(p.velocity, self.w)

                v_new: CapacityDict = {}
                for k in self.keys:
                    v_k = v_inertia[k] + cognitive[k] + social[k]
                    # 边际效益修正项：直接叠加 alpha_mu * mui[k]
                    v_k += self.alpha_mu * float(mui.get(k, 0.0))
                    v_new[k] = v_k

                p.velocity = v_new

                # 位置更新 + 边界截断
                new_pos = self._add(p.position, p.velocity)
                p.position = self._clip_capacity(new_pos)

        return global_best_pos, global_best_metrics
