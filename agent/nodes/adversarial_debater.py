"""
红蓝对抗博弈辩论节点 (AdversarialDebater)
模块: okx-dog-ai/agent/nodes/adversarial_debater.py
角色: 多头进攻辩护专家 (BullAdvocate) vs 空头风控挑刺红队 (adversarial-red-team-bear-critic)

职责:
1. 汇聚 6 大感知专家产物（宏观多周期、链上资金、量化微观、衍生品情绪、宏观日历、流动性结构）；
2. 多头进攻辩护专家 (BullAdvocate): 搜寻均线多头排列、放量突破、链上沉淀与买盘托底证据；
3. 空头风控红队专家 (BearCritic): 严苛挑刺，搜寻上方密集套牢筹码、顶背离诱多、资金费率过热及宏观黑天鹅风险；
4. 输出多空双方置信度与辩论陈述，为首席仲裁官提供结构化博弈论据。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from ..state import QuantTraderState, ThinkingStep

logger = logging.getLogger("okx_dog.ai.agent.adversarial_debater")


async def adversarial_debate_node(state: QuantTraderState) -> Dict[str, Any]:
    """
    LangGraph Node: 红蓝对抗博弈辩论中枢
    """
    logger.info("执行 Node: 红蓝对抗博弈辩论 (多头进攻 vs 空头挑刺红队)...")
    now_ms = int(time.time() * 1000)
    current_price = float(state.get("current_price", 0.0))
    macro_regime = state.get("market_regime", "RANGING")
    onchain_data = state.get("onchain_analysis", {})
    quant_data = state.get("quant_features", {})
    deriv_data = state.get("derivatives_sentiment", {})
    macro_event = state.get("macro_event_risk", {})
    micro_data = state.get("microstructure_data", {})

    onchain_score = float(onchain_data.get("composite_score", 0.0))
    onchain_bias = onchain_data.get("flow_bias", "NEUTRAL")
    imbalance_ratio = float(quant_data.get("orderbook_imbalance_ratio", 1.0))
    deriv_score = float(deriv_data.get("sentiment_score", 0.0))
    funding_bias = deriv_data.get("funding_rate_bias", "NEUTRAL")
    event_risk_level = macro_event.get("event_risk_level", "LOW")
    spread_bps = float(micro_data.get("spread_bps", 1.0))

    # -------------------------------------------------------------
    # 1. 多头进攻辩护专家 (Bull Specialist Advocate) 评估
    # -------------------------------------------------------------
    bull_confidence = 0.50
    bull_args: List[str] = []
    bull_warnings: List[str] = []

    if macro_regime in ["TRENDING_UP", "VOLATILE_BREAKOUT"]:
        bull_confidence += 0.20
        bull_args.append(f"宏观技术面呈多头进攻态势 ({macro_regime})，均线系统呈多头排列")
    if onchain_score > 0.2:
        bull_confidence += 0.15
        bull_args.append(f"链上巨鲸资金呈主动吸筹沉淀态势 (得分 {onchain_score:+.2f})，现货筹码锁仓良好")
    if imbalance_ratio > 1.2:
        bull_confidence += 0.10
        bull_args.append(f"订单簿买盘挂单厚实 (买卖比 {imbalance_ratio:.2f})，下方存在强支撑托底")
    if deriv_score > 0.2 and funding_bias != "EXTREME_POSITIVE":
        bull_confidence += 0.08
        bull_args.append("衍生品市场多头情绪温和健康，持仓量稳步递增")

    bull_stance = "BULLISH" if bull_confidence >= 0.65 else "NEUTRAL"
    bull_opinion = {
        "role_name": "多头进攻辩护专家 (Bull Specialist)",
        "stance": bull_stance,
        "confidence": round(min(0.95, bull_confidence), 2),
        "key_arguments": bull_args or ["均线维持多头排列", "逢回调支撑位有承接"],
        "risk_warnings": bull_warnings or ["需关注上方密集套牢盘阻力"],
    }

    # -------------------------------------------------------------
    # 2. 空头风控红队挑刺专家 (Bear Critic Red-Team) 苛刻审查
    # -------------------------------------------------------------
    bear_confidence = 0.50
    bear_args: List[str] = []
    bear_warnings: List[str] = []

    if funding_bias == "EXTREME_POSITIVE":
        bear_confidence += 0.25
        bear_args.append("资金费率极度过热，多头杠杆极度拥挤，随时可能引发多头踩踏爆仓插针")
    if macro_regime == "TRENDING_DOWN":
        bear_confidence += 0.25
        bear_args.append("宏观处于空头破位下行通道，反弹均为诱多洗盘")
    if onchain_score < -0.2:
        bear_confidence += 0.15
        bear_args.append(f"链上呈现大额 CEX 充币出货抛压 (得分 {onchain_score:+.2f})")
    if imbalance_ratio < 0.8:
        bear_confidence += 0.10
        bear_args.append(f"盘口上方存在密集卖单挂盘压制 (买卖比 {imbalance_ratio:.2f})")
    if event_risk_level in ["HIGH", "CRITICAL"]:
        bear_confidence += 0.20
        bear_args.append(f"宏观日历处于高危事件敏感窗口 ({event_risk_level})，盲目开仓极易遭受流动性真空双向打脸")
    if spread_bps > 4.0:
        bear_warnings.append(f"微观价差过大 ({spread_bps:.1f} bps)，滑点冲击成本偏高")

    bear_stance = "BEARISH" if bear_confidence >= 0.65 else "NEUTRAL"
    bear_opinion = {
        "role_name": "空头风控红队专家 (Bear Critic Red-Team)",
        "stance": bear_stance,
        "confidence": round(min(0.95, bear_confidence), 2),
        "key_arguments": bear_args or ["上方阻力位抛压聚集", "市场缺乏增量流动性"],
        "risk_warnings": bear_warnings or ["关注突发黑天鹅与假突破风险"],
    }

    debate_summary = (
        f"红蓝多空博弈辩论完成: 多头置信度 {bull_opinion['confidence']*100:.0f}% ({bull_stance})，"
        f"空头红队置信度 {bear_opinion['confidence']*100:.0f}% ({bear_stance})。"
    )

    thought_text = (
        f"【红蓝对抗博弈辩论】\n"
        f"🔵 多头论点 ({bull_opinion['confidence']*100:.0f}%): {'; '.join(bull_opinion['key_arguments'])}\n"
        f"🔴 空头挑刺 ({bear_opinion['confidence']*100:.0f}%): {'; '.join(bear_opinion['key_arguments'])}\n"
        f"⚖️ 状态摘要: {debate_summary}"
    )

    thinking_step: ThinkingStep = {
        "node": "AdversarialDebater",
        "stage_name": "红蓝对抗博弈辩论",
        "thought": thought_text,
        "timestamp_ms": now_ms,
    }

    return {
        "bull_opinion": bull_opinion,
        "bear_opinion": bear_opinion,
        "debate_summary": debate_summary,
        "thinking_steps": [thinking_step],
    }
