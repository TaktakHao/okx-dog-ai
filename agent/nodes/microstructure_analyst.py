"""
微观流动性与订单簿冲击分析专家 (MicrostructureAnalyst)
模块: okx-dog-ai/agent/nodes/microstructure_analyst.py
角色: 微观流动性与智能执行交易员 (liquidity-execution-microstructure-trader)

职责:
1. 深度分析买卖前 20 档订单簿的累积流动性厚度与买卖价差 (Spread)；
2. 预估目标头寸在当前盘口下的市场冲击成本 (Market Impact) 与滑点风险；
3. 输出智能执行模式建议 (LIMIT 区间限价 / POST_ONLY 被动做市 / TWAP 算法拆单)。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from ..registry import BaseSpecialist, register_specialist
from ..state import QuantTraderState, ThinkingStep
from ..tools import analyze_orderbook_liquidity

logger = logging.getLogger("okx_dog.ai.agent.microstructure_analyst")


@register_specialist
class MicrostructureAnalystSpecialist(BaseSpecialist):
    name = "microstructure_analyst"
    stage_name = "微观流动性与订单簿冲击分析"
    layer = "perception"
    description = "分析盘口有效深度、买卖价差与冲击成本，推荐最优执行路由"

    async def analyze(self, state: QuantTraderState) -> Dict[str, Any]:
        return await microstructure_analyst_node(state)


async def microstructure_analyst_node(state: QuantTraderState) -> Dict[str, Any]:
    """
    LangGraph Node: 微观流动性与订单簿冲击分析
    """
    logger.info("执行 Node: 微观流动性与订单簿冲击分析...")
    now_ms = int(time.time() * 1000)
    current_price = float(state.get("current_price", 0.0))
    raw_snapshot = state.get("market_snapshot", {})

    bids = raw_snapshot.get("orderbook_bids_top5") or raw_snapshot.get("orderbook_bids") or []
    asks = raw_snapshot.get("orderbook_asks_top5") or raw_snapshot.get("orderbook_asks") or []

    risk_limits = state.get("risk_limits", {})
    std_order_usdt = float(risk_limits.get("max_order_usdt", 500.0))

    microstructure_result = analyze_orderbook_liquidity(
        bids=bids,
        asks=asks,
        current_price=current_price,
        standard_order_usdt=std_order_usdt,
        depth_levels=20,
    )

    thought_text = (
        f"【微观流动性与订单冲击分析】买卖价差: {microstructure_result['spread_bps']:.1f} bps，"
        f"预估冲击成本: {microstructure_result['estimated_impact_bps']:.1f} bps。"
        f"买盘有效深度: ${microstructure_result['bid_liquidity_depth_usdt']/1e3:.1f}K, "
        f"卖盘有效深度: ${microstructure_result['ask_liquidity_depth_usdt']/1e3:.1f}K。"
        f"推荐订单路由: 【{microstructure_result['recommended_execution_mode']}】({microstructure_result['execution_advice']})。"
    )

    thinking_step: ThinkingStep = {
        "node": "MicrostructureAnalyst",
        "stage_name": "微观流动性与订单簿冲击分析",
        "thought": thought_text,
        "timestamp_ms": now_ms,
    }

    return {
        "microstructure_data": microstructure_result,
        "thinking_steps": [thinking_step],
    }
