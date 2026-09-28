"""
OKX-Dog 量化策略评估中枢 - 参数平原与生死线抗过拟合检验器
模块: okx-dog-ai/strategy_plateau_validator.py
融合自: oke_auto_trade (filter_good_param.py & bread_through_strategy_ga_target.py)

核心职责:
1. 生死线硬指标审查 (Hard Survival Gate):
   - 剔除 Top-3 暴利交易后的收益衰减测试 (防黑天鹅幸存者偏差，保留率须 >= 50%)
   - 手续费/毛利比红线审查 (严防高频磨损向交易所打工，费用率须 < 40%)
   - 双边 40bps (0.04%) 真实滑点与手续费摩擦压力测试 (抗滑点后期望须 > 0)
2. 参数平原检验 (Parameter Plateau Validation):
   - 杜绝“孤立尖峰”过拟合，检验核心参数在 +/-20%~30% 邻域扰动下的收益稳定性与胜率方差。
3. 产出统一对齐的 StrategyPlateauScore 契约评分。
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from schemas import StrategyPlateauScore
except ImportError:
    from pydantic import BaseModel, Field

    class StrategyPlateauScore(BaseModel):
        is_plateau_qualified: bool
        drop_top3_decay: float
        fee_to_gross_ratio: float
        friction_stress_pnl: float
        neighbor_stability_score: float


logger = logging.getLogger("StrategyPlateauValidator")


class StrategyPlateauValidator:
    """参数平原与生死线抗过拟合评估器"""

    @staticmethod
    def evaluate_strategy_survival(
        trades_pnl: List[float],
        trades_notional: List[float],
        total_fee_paid: float,
        slippage_stress_bps: float = 40.0,  # 双边 40bps (0.0040)
        min_top3_retention_ratio: float = 0.50,  # 剔除 Top3 后至少保留 50% 利润
        max_fee_ratio: float = 0.40,             # 手续费/毛利比上限 40%
        neighbor_pnl_list: Optional[List[float]] = None  # +/-20% 邻域参数组合的 PnL
    ) -> StrategyPlateauScore:
        """
        全面评估策略样本的实盘存活能力与参数平原质量。

        :param trades_pnl: 历史逐笔交易实现盈亏列表 (USDT)
        :param trades_notional: 历史逐笔交易名义金额列表 (USDT)
        :param total_fee_paid: 累计已付手续费 (USDT)
        :param slippage_stress_bps: 双边滑点摩擦压力基点 (bps)
        :param min_top3_retention_ratio: Top3 利润保留率阈值
        :param max_fee_ratio: 手续费占比上限
        :param neighbor_pnl_list: 邻域参数策略组合的 PnL 收益列表
        :return: StrategyPlateauScore
        """
        if not trades_pnl:
            return StrategyPlateauScore(
                is_plateau_qualified=False,
                drop_top3_decay=0.0,
                fee_to_gross_ratio=1.0,
                friction_stress_pnl=0.0,
                neighbor_stability_score=0.0
            )

        total_net_pnl = sum(trades_pnl)
        gross_profit = sum([p for p in trades_pnl if p > 0]) or 1e-9

        # -------------------------------------------------------------
        # 1. Top-3 暴利剔除衰减率检验 (防运气型黑天鹅)
        # -------------------------------------------------------------
        sorted_pnl = sorted(trades_pnl, reverse=True)
        top3_sum = sum(sorted_pnl[:3]) if len(sorted_pnl) >= 3 else sum(sorted_pnl)
        
        if total_net_pnl > 0:
            remaining_pnl = total_net_pnl - top3_sum
            drop_top3_retention = max(0.0, remaining_pnl / total_net_pnl)
        else:
            drop_top3_retention = 0.0

        pass_top3 = (total_net_pnl > 0) and (drop_top3_retention >= min_top3_retention_ratio)

        # -------------------------------------------------------------
        # 2. 手续费/毛利比检验 (拒绝给交易所打工)
        # -------------------------------------------------------------
        fee_ratio = total_fee_paid / gross_profit
        pass_fee = fee_ratio <= max_fee_ratio

        # -------------------------------------------------------------
        # 3. 真实滑点与摩擦压力测试 (双边 40bps 压力测试)
        # -------------------------------------------------------------
        total_notional = sum(trades_notional)
        slippage_cost = total_notional * (slippage_stress_bps / 10000.0)
        friction_stress_pnl = total_net_pnl - slippage_cost
        pass_stress = friction_stress_pnl > 0

        # -------------------------------------------------------------
        # 4. 邻域参数平原稳定性检验 (Parameter Plateau)
        # -------------------------------------------------------------
        # 检验在 +/-20% 参数微调下，策略是否依然能够稳定盈利且方差极小
        neighbor_score = 50.0  # 默认基准分
        pass_neighbors = True

        if neighbor_pnl_list and len(neighbor_pnl_list) >= 3:
            pos_ratio = sum([1 for p in neighbor_pnl_list if p > 0]) / len(neighbor_pnl_list)
            mean_neighbor = float(np.mean(neighbor_pnl_list))
            std_neighbor = float(np.std(neighbor_pnl_list)) if len(neighbor_pnl_list) > 1 else 0.0

            # 变异系数 CV = std / mean
            cv = (std_neighbor / abs(mean_neighbor)) if abs(mean_neighbor) > 1e-6 else 1.0
            stability_factor = max(0.0, 1.0 - min(1.0, cv))

            # 综合计算平原得分 (0~100)
            neighbor_score = round(pos_ratio * 60.0 + stability_factor * 40.0, 1)
            # 要求至少 70% 的邻居参数盈利且得分 >= 60
            pass_neighbors = (pos_ratio >= 0.70) and (neighbor_score >= 60.0)
        else:
            # 若未提供邻居，降级使用自身夏普与单笔稳定性推导
            if total_net_pnl > 0 and len(trades_pnl) >= 10:
                pnl_std = float(np.std(trades_pnl)) or 1.0
                pnl_mean = float(np.mean(trades_pnl))
                pseudo_sharpe = max(0.0, pnl_mean / pnl_std)
                neighbor_score = min(100.0, round(pseudo_sharpe * 40.0 + 30.0, 1))

        # -------------------------------------------------------------
        # 5. 综合合格判定
        # -------------------------------------------------------------
        is_qualified = bool(pass_top3 and pass_fee and pass_stress and pass_neighbors)

        return StrategyPlateauScore(
            is_plateau_qualified=is_qualified,
            drop_top3_decay=round(drop_top3_retention, 4),
            fee_to_gross_ratio=round(fee_ratio, 4),
            friction_stress_pnl=round(friction_stress_pnl, 2),
            neighbor_stability_score=neighbor_score
        )
