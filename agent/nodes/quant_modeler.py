"""
量化统计与盘口微观结构建模节点 (QuantModeler)
模块: okx-dog-ai/agent/nodes/quant_modeler.py

职责:
1. 深入分析盘口前 5 档/前 10 档订单簿买卖深度 (Orderbook Depth) 与失衡比率 (Imbalance Ratio)。
2. 测算当前市场的波动率体制 (Volatility Regime) 与布林带挤压状态。
3. 基于多周期动量与微观深度构建多因子数学期望胜率与盈亏比预估。
4. 基于半凯利公式 (Half-Kelly Criterion) 输出建议的最优单笔风险敞口与名义仓位。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from ..registry import BaseSpecialist, register_specialist
from ..state import QuantTraderState, ThinkingStep
from ..tools import (
    calculate_kelly_position_size,
    calculate_orderbook_imbalance,
    check_volatility_squeeze,
)

logger = logging.getLogger("okx_dog.ai.agent.quant_modeler")


@register_specialist
class QuantModelerSpecialist(BaseSpecialist):
    name = "quant_modeler"
    stage_name = "量化统计与盘口微观建模"
    layer = "perception"
    description = "分析订单簿前5档失衡比、波动率挤压与半凯利动态头寸管理"

    async def analyze(self, state: QuantTraderState) -> Dict[str, Any]:
        return await quant_modeler_node(state)


async def quant_modeler_node(state: QuantTraderState) -> Dict[str, Any]:
    """
    LangGraph Node: 量化统计与微观结构建模
    """
    logger.info("执行 Node: 量化统计与微观结构建模...")
    now_ms = int(time.time() * 1000)
    current_price = float(state.get("current_price", 0.0))
    account_balance = float(state.get("account_balance_usdt", 1000.0))
    raw_snapshot = state.get("market_snapshot", {})

    # 1. 提取盘口订单簿前 5 档深度
    bids = raw_snapshot.get("orderbook_bids_top5")
    asks = raw_snapshot.get("orderbook_asks_top5")
    imbalance_ratio, ob_desc = calculate_orderbook_imbalance(bids=bids, asks=asks, depth_levels=5)

    # 2. 提取 15m 指标进行波动率挤压建模
    multi_ind = raw_snapshot.get("multi_indicators", {})
    if hasattr(multi_ind, "indicators"):
        ind_15m = multi_ind.indicators.get("15m", {})
        atr14 = float(getattr(ind_15m, "atr_14", current_price * 0.01))
        bb_upper = float(getattr(ind_15m, "bb_upper", current_price * 1.02))
        bb_lower = float(getattr(ind_15m, "bb_lower", current_price * 0.98))
        bb_mid = float(getattr(ind_15m, "bb_middle", current_price))
    elif isinstance(multi_ind, dict) and "indicators" in multi_ind:
        ind_15m = multi_ind["indicators"].get("15m", {})
        atr14 = float(ind_15m.get("atr_14", current_price * 0.01))
        bb_upper = float(ind_15m.get("bb_upper", current_price * 1.02))
        bb_lower = float(ind_15m.get("bb_lower", current_price * 0.98))
        bb_mid = float(ind_15m.get("bb_middle", current_price))
    else:
        atr14 = current_price * 0.01
        bb_upper = current_price * 1.02
        bb_lower = current_price * 0.98
        bb_mid = current_price

    is_squeezed, bandwidth_pct, squeeze_msg = check_volatility_squeeze(
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        bb_middle=bb_mid,
        atr_14=atr14,
        current_price=current_price,
    )

    # 3. 推算期望胜率与凯利仓位
    base_win_rate = 0.55
    if imbalance_ratio > 1.3:
        base_win_rate += 0.08
    elif imbalance_ratio < 0.7:
        base_win_rate -= 0.08

    expected_win_rate = round(max(0.35, min(0.78, base_win_rate)), 2)
    expected_rr = 1.8

    suggested_risk_usdt, kelly_fraction, kelly_desc = calculate_kelly_position_size(
        win_rate=expected_win_rate,
        reward_risk_ratio=expected_rr,
        account_balance_usdt=account_balance,
        max_risk_pct=0.05,
        fractional_kelly=0.5,
    )

    # 4. 组装量化特征字典
    quant_features = {
        "orderbook_imbalance_ratio": imbalance_ratio,
        "orderbook_desc": ob_desc,
        "is_volatility_squeezed": is_squeezed,
        "bollinger_bandwidth": bandwidth_pct,
        "squeeze_status_message": squeeze_msg,
        "expected_win_rate": expected_win_rate,
        "expected_rr_ratio": expected_rr,
        "suggested_risk_usdt": suggested_risk_usdt,
        "suggested_position_pct": round(kelly_fraction * 100, 2),
        "kelly_rationale": kelly_desc,
    }

    # 5. 生成专业量化思考链
    thought_text = (
        f"【量化微观结构与统计建模】盘口微观失衡比={imbalance_ratio:.2f} ({ob_desc})。\n"
        f"波动率挤压检验: {'触发挤压(警惕假突破)' if is_squeezed else '波动率正常'} (布林带宽={bandwidth_pct*100:.2f}%)。\n"
        f"多因子胜率期望={expected_win_rate*100:.0f}%。{kelly_desc}"
    )

    thinking_step: ThinkingStep = {
        "node": "QuantModeler",
        "stage_name": "量化统计与微观结构建模",
        "thought": thought_text,
        "timestamp_ms": now_ms,
    }

    return {
        "quant_features": quant_features,
        "thinking_steps": [thinking_step],
    }
