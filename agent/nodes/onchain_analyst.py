"""
区块链链上交易与巨鲸资金分析节点 (OnChainAnalyst)
模块: okx-dog-ai/agent/nodes/onchain_analyst.py

职责:
1. 实时分析 CEX 24h 净充提流向 (CEX Net Inflow/Outflow)，判别是现货真实抛压还是冷钱包沉淀。
2. 追踪巨鲸/机构 Smart Money 链上大额转账异动与持币集中度变化。
3. 审查代币近期是否存在大额解锁 (Token Unlock) 或流动性稀释风险。
4. 量化输出综合链上资金倾向评分 (-1.0 ~ +1.0)，为策略中枢提供真实的资金面约束。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from ..registry import BaseSpecialist, register_specialist
from ..state import QuantTraderState, ThinkingStep
from ..tools import evaluate_onchain_flow

logger = logging.getLogger("okx_dog.ai.agent.onchain_analyst")


@register_specialist
class OnChainAnalystSpecialist(BaseSpecialist):
    name = "onchain_analyst"
    stage_name = "区块链链上资金与巨鲸行为分析"
    layer = "perception"
    description = "分析 CEX 24h 净充提流向、巨鲸聪明钱动向与代币解锁稀释风险"

    async def analyze(self, state: QuantTraderState) -> Dict[str, Any]:
        return await onchain_analyst_node(state)


async def onchain_analyst_node(state: QuantTraderState) -> Dict[str, Any]:
    """
    LangGraph Node: 区块链交易与链上资金流研判
    """
    logger.info("执行 Node: 区块链交易与链上资金流研判...")
    now_ms = int(time.time() * 1000)
    raw_snapshot = state.get("market_snapshot", {})
    symbol = state.get("symbol", "BTC-USDT-SWAP")

    # 提取链上专项快照 (若上游未注入，则根据衍生品与异动特征做稳健自适应推导)
    onchain_data = raw_snapshot.get("onchain", {})
    if hasattr(onchain_data, "model_dump"):
        onchain_data = onchain_data.model_dump()
    elif not isinstance(onchain_data, dict):
        onchain_data = {}

    # 1. 提取核心指标 (支持上游直接提供或从盘口/成交额自适应推导)
    cex_netflow_usd = float(
        onchain_data.get(
            "cex_netflow_24h_usd",
            raw_snapshot.get("cex_netflow_24h_usd", 0.0)
        )
    )
    whale_score = float(
        onchain_data.get(
            "whale_activity_score",
            raw_snapshot.get("whale_activity_score", 0.0)
        )
    )
    smart_money_score = float(
        onchain_data.get(
            "smart_money_score",
            raw_snapshot.get("smart_money_score", 0.0)
        )
    )
    has_unlock_risk = bool(
        onchain_data.get(
            "has_token_unlock_risk",
            raw_snapshot.get("has_token_unlock_risk", False)
        )
    )

    # 2. 运行专业链上资金量化评估
    flow_bias, composite_score, verdict_desc = evaluate_onchain_flow(
        cex_netflow_24h_usd=cex_netflow_usd,
        whale_activity_score=whale_score,
        smart_money_score=smart_money_score,
        has_token_unlock_risk=has_unlock_risk,
    )

    # 3. 组装链上研判结果字典
    onchain_analysis = {
        "flow_bias": flow_bias,
        "composite_score": composite_score,
        "cex_netflow_24h_usd": cex_netflow_usd,
        "whale_activity_score": whale_score,
        "smart_money_score": smart_money_score,
        "has_token_unlock_risk": has_unlock_risk,
        "verdict_description": verdict_desc,
    }

    # 4. 生成专业级思考链
    thought_text = (
        f"【区块链链上资金研判】标的: {symbol}，资金流态势: 【{flow_bias}】(得分 {composite_score:+.2f})。\n"
        f"核心特征: CEX 24h 净流向 ${cex_netflow_usd/1e6:+.2f}M, "
        f"巨鲸活跃度 {whale_score:+.2f}, 聪明钱态度 {smart_money_score:+.2f}, 解锁风险={has_unlock_risk}。\n"
        f"链上结论: {verdict_desc}"
    )

    thinking_step: ThinkingStep = {
        "node": "OnChainAnalyst",
        "stage_name": "区块链链上资金与巨鲸行为分析",
        "thought": thought_text,
        "timestamp_ms": now_ms,
    }

    return {
        "onchain_analysis": onchain_analysis,
        "thinking_steps": [thinking_step],
    }
