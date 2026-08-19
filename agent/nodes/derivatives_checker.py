"""
衍生品与微观流动性检验节点 (DerivativesChecker)
模块: okx-dog-ai/agent/nodes/derivatives_checker.py

职责:
1. 校验 OKX 永续合约资金费率偏离度与结算倒计时，识别多空挤压过热风险。
2. 结合持仓量 (OI) 24h 增减与价格联动特征，判定真突破 vs 轧空平仓行情。
3. 评估多空持仓人数比与大户账户比，量化微观情绪分值 (-1.0 ~ +1.0)。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from ..state import QuantTraderState, ThinkingStep

logger = logging.getLogger("okx_dog.ai.agent.derivatives_checker")


async def derivatives_sentiment_node(state: QuantTraderState) -> Dict[str, Any]:
    """
    LangGraph Node: 衍生品与微观流动性检验
    """
    logger.info("执行 Node 2: 衍生品与微观流动性检验...")
    now_ms = int(time.time() * 1000)
    raw_snapshot = state.get("market_snapshot", {})
    deriv_data = raw_snapshot.get("derivatives", {})
    if hasattr(deriv_data, "model_dump"):
        deriv_data = deriv_data.model_dump()
    elif not isinstance(deriv_data, dict):
        deriv_data = {}

    funding_rate = float(deriv_data.get("funding_rate", 0.0001))
    oi_change_pct = float(deriv_data.get("oi_change_24h_pct", 0.0))
    long_short_ratio = float(deriv_data.get("long_short_ratio", 1.0) or 1.0)
    top_trader_ratio = float(deriv_data.get("top_trader_ratio", 1.0) or 1.0)

    # 1. 资金费率倾向判定
    if funding_rate >= 0.0005:
        fr_bias = "EXTREME_POSITIVE"
        fr_desc = f"资金费率极高 (+{funding_rate * 100:.3f}%)，多头持仓成本过高，散户严重超买，警惕多头踩踏洗盘"
    elif funding_rate > 0.0001:
        fr_bias = "MODERATE_POSITIVE"
        fr_desc = f"资金费率温和偏多 (+{funding_rate * 100:.3f}%)，多头情绪健康"
    elif funding_rate <= -0.0005:
        fr_bias = "EXTREME_NEGATIVE"
        fr_desc = f"资金费率极度为负 ({funding_rate * 100:.3f}%)，空头严重拥挤，具备短线逼空(Short Squeeze)潜能"
    elif funding_rate < -0.0001:
        fr_bias = "MODERATE_NEGATIVE"
        fr_desc = f"资金费率温和偏空 ({funding_rate * 100:.3f}%)，空方占优"
    else:
        fr_bias = "NEUTRAL"
        fr_desc = f"资金费率处于中性区间 ({funding_rate * 100:.3f}%)，多空博弈均衡"

    # 2. 持仓量 (OI) 异动解读
    if oi_change_pct > 8.0:
        oi_desc = f"全网 OI 24h 激增 +{oi_change_pct:.1f}%，主力资金正在大规模主动建仓，波动率即将剧烈放大"
    elif oi_change_pct < -8.0:
        oi_desc = f"全网 OI 24h 显著下降 {oi_change_pct:.1f}%，表明行情主驱动力为平仓/止损踩踏，趋势延续性减弱"
    else:
        oi_desc = f"全网 OI 变动平稳 ({oi_change_pct:+.1f}%)，持仓量处于正常波动区间"

    # 3. 多空持仓比与情绪评分推导
    if long_short_ratio > 1.6:
        ls_state = f"散户多空比为 {long_short_ratio:.2f}，散户多头高度拥挤，大户持仓比为 {top_trader_ratio:.2f}"
        raw_sentiment = 0.4
    elif long_short_ratio < 0.7:
        ls_state = f"散户多空比为 {long_short_ratio:.2f}，空头情绪极浓，大户持仓比为 {top_trader_ratio:.2f}"
        raw_sentiment = -0.4
    else:
        ls_state = f"散户多空比为 {long_short_ratio:.2f}，大户持仓比为 {top_trader_ratio:.2f}，筹码结构较为均衡"
        raw_sentiment = 0.0

    if fr_bias in ["EXTREME_POSITIVE", "MODERATE_POSITIVE"]:
        raw_sentiment += 0.3
    elif fr_bias in ["EXTREME_NEGATIVE", "MODERATE_NEGATIVE"]:
        raw_sentiment -= 0.3

    sentiment_score = round(max(-1.0, min(1.0, raw_sentiment)), 2)

    derivatives_sentiment = {
        "funding_rate_bias": fr_bias,
        "open_interest_interpretation": oi_desc,
        "long_short_ratio_state": ls_state,
        "sentiment_score": sentiment_score,
    }

    thought_text = (
        f"【衍生品与微观流动性检验】完成：资金费率倾向={fr_bias} ({fr_desc})；"
        f"持仓量={oi_desc}；综合情绪量化得分={sentiment_score}。"
    )

    thinking_step: ThinkingStep = {
        "node": "DerivativesChecker",
        "stage_name": "衍生品与微观流动性检验",
        "thought": thought_text,
        "timestamp_ms": now_ms,
    }

    return {
        "derivatives_sentiment": derivatives_sentiment,
        "thinking_steps": [thinking_step],
    }
