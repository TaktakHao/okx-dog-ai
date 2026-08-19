"""
OKX-Dog 量化交易员专用量化与风控计算工具箱
模块: okx-dog-ai/agent/tools.py
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("okx_dog.ai.agent.tools")


def calculate_risk_reward_ratio(
    action: str,
    entry_range: List[float],
    stop_loss_price: float,
    take_profit_levels: List[Dict[str, Any]],
) -> Tuple[float, Optional[str]]:
    """
    计算并校验实际理论盈亏比 (R:R Ratio) 与方向逻辑。
    返回: (rr_ratio, error_message)
    """
    if not entry_range or len(entry_range) != 2:
        return 0.0, "入场区间格式非法，必须为 [最低价, 最高价]"

    if not take_profit_levels:
        return 0.0, "未配置止盈目标点位"

    entry_avg = (entry_range[0] + entry_range[1]) / 2.0
    tp1_price = float(take_profit_levels[0].get("price", 0.0))

    if entry_avg <= 0 or stop_loss_price <= 0 or tp1_price <= 0:
        return 0.0, "价格参数必须大于 0"

    risk_dist = abs(entry_avg - stop_loss_price)
    if risk_dist <= 1e-6:
        return 0.0, "止损价与入场均价距离过近或相同，无法构成有效风控区间"

    reward_dist = abs(tp1_price - entry_avg)
    rr_ratio = round(reward_dist / risk_dist, 2)

    # 方向逻辑严密性检验
    if action == "BUY_LONG":
        if stop_loss_price >= entry_avg:
            return rr_ratio, f"做多操作中，止损价 ({stop_loss_price}) 必须低于入场均价 ({entry_avg:.2f})"
        if tp1_price <= entry_avg:
            return rr_ratio, f"做多操作中，第一止盈价 ({tp1_price}) 必须高于入场均价 ({entry_avg:.2f})"
    elif action == "SELL_SHORT":
        if stop_loss_price <= entry_avg:
            return rr_ratio, f"做空操作中，止损价 ({stop_loss_price}) 必须高于入场均价 ({entry_avg:.2f})"
        if tp1_price >= entry_avg:
            return rr_ratio, f"做空操作中，第一止盈价 ({tp1_price}) 必须低于入场均价 ({entry_avg:.2f})"

    return rr_ratio, None


def verify_hard_risk_compliance(
    action: str,
    entry_price: float,
    stop_loss_price: float,
    suggested_leverage: int,
    account_balance_usdt: float,
    risk_limits: Dict[str, Any],
) -> List[str]:
    """
    针对交易员输出的草案进行硬风控物理边界拦截审查。
    返回: 违规项列表 (若为空列表则表示 100% 审查通过)
    """
    violations: List[str] = []

    if action in ["HOLD_WAIT", "CLOSE_POSITION"]:
        return violations

    max_leverage = int(risk_limits.get("max_leverage", 5))
    max_order_usdt = float(risk_limits.get("max_order_usdt", 500.0))
    max_daily_loss = float(risk_limits.get("max_daily_loss_usdt", 200.0))

    # 1. 杠杆审查
    if suggested_leverage > max_leverage:
        violations.append(
            f"建议杠杆 ({suggested_leverage}x) 超出系统允许的最大杠杆上限 ({max_leverage}x)"
        )

    # 2. 止损幅度与单笔名义风险评估
    if entry_price > 0 and stop_loss_price > 0:
        stop_loss_dist_pct = abs(entry_price - stop_loss_price) / entry_price
        if stop_loss_dist_pct > 0.15:
            violations.append(
                f"单笔止损幅度过宽 ({stop_loss_dist_pct * 100:.2f}%)，严重偏离量化风控合理范围 (建议不超过 5%~10%)"
            )

        # 估算最大委托名义价值下的亏损敞口
        estimated_max_loss = max_order_usdt * stop_loss_dist_pct * (suggested_leverage if suggested_leverage <= 3 else 1.0)
        if estimated_max_loss > max_daily_loss:
            violations.append(
                f"预估单笔可能最大亏损 ({estimated_max_loss:.2f} USDT) 逼近或超过单日最大亏损熔断阈值 ({max_daily_loss:.2f} USDT)"
            )

    return violations


def derive_dynamic_atr_stops(
    current_price: float,
    atr_14: float,
    action: str,
    multiplier: float = 2.0,
) -> Tuple[float, float, float]:
    """
    基于 ATR(14) 动态推导建议硬止损与 TP1/TP2 目标。
    返回: (stop_loss, tp1, tp2)
    """
    safe_atr = max(atr_14, current_price * 0.005)
    stop_distance = safe_atr * multiplier

    if action == "BUY_LONG":
        stop_loss = round(current_price - stop_distance, 2 if current_price > 10 else 4)
        tp1 = round(current_price + (stop_distance * 1.6), 2 if current_price > 10 else 4)
        tp2 = round(current_price + (stop_distance * 2.8), 2 if current_price > 10 else 4)
    elif action == "SELL_SHORT":
        stop_loss = round(current_price + stop_distance, 2 if current_price > 10 else 4)
        tp1 = round(current_price - (stop_distance * 1.6), 2 if current_price > 10 else 4)
        tp2 = round(current_price - (stop_distance * 2.8), 2 if current_price > 10 else 4)
    else:
        stop_loss = 0.0
        tp1 = 0.0
        tp2 = 0.0

    return stop_loss, tp1, tp2
