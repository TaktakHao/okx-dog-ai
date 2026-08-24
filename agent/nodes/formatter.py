"""
契约模型格式化与思维链收敛节点 (ResponseFormatter)
模块: okx-dog-ai/agent/nodes/formatter.py
角色: 契约与思维链收敛官

职责:
1. 将 StateGraph 全生命周期的所有研判（6 专家感知 + 红蓝对抗 + 首席仲裁 + 硬风控审查）严格组装为符合 AIAnalysisResponse 标准契约的对象。
2. 将全链路 ThinkingStep 渲染为专业级分层 Markdown/CoT 思考轨迹。
3. 计算总执行延迟并标记使用的 Multi-Agent 架构版本。
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict

from ..state import QuantTraderState, ThinkingStep

try:
    from ...schemas import (
        AIAnalysisResponse,
        AISignal,
        DerivativesSentiment,
        MarketRegime,
        RiskAssessment,
        TakeProfitLevel,
        TimeframeAnalysis,
        TimeframeDetail,
        TradePlan,
    )
except (ImportError, ValueError):
    try:
        from okx_dog_ai.schemas import (
            AIAnalysisResponse,
            AISignal,
            DerivativesSentiment,
            MarketRegime,
            RiskAssessment,
            TakeProfitLevel,
            TimeframeAnalysis,
            TimeframeDetail,
            TradePlan,
        )
    except (ImportError, ValueError):
        from schemas import (
            AIAnalysisResponse,
            AISignal,
            DerivativesSentiment,
            MarketRegime,
            RiskAssessment,
            TakeProfitLevel,
            TimeframeAnalysis,
            TimeframeDetail,
            TradePlan,
        )

logger = logging.getLogger("okx_dog.ai.agent.formatter")


async def response_formatter_node(state: QuantTraderState) -> Dict[str, Any]:
    """
    LangGraph Node: 契约模型格式化与思维链收敛
    """
    logger.info("执行 Node: 契约模型格式化与思维链收敛...")
    now_ms = int(time.time() * 1000)
    start_ts = state.get("timestamp", now_ms)
    latency_ms = max(1, now_ms - start_ts)

    analysis_id = state.get("analysis_id") or str(uuid.uuid4())
    symbol = state.get("symbol", "BTC-USDT-SWAP")
    market_regime_str = state.get("market_regime", "RANGING")

    # 1. 组装多周期细节
    tf_data = state.get("timeframe_analysis", {})
    tf_analysis = TimeframeAnalysis(
        tf_15m=TimeframeDetail(**tf_data.get("tf_15m", {
            "trend": "NEUTRAL_CHOPPY",
            "key_indicators_summary": "15M 平衡震荡",
            "support_level": 0.0,
            "resistance_level": 0.0,
        })),
        tf_1h=TimeframeDetail(**tf_data.get("tf_1h", {
            "trend": "NEUTRAL_CHOPPY",
            "key_indicators_summary": "1H 平衡震荡",
            "support_level": 0.0,
            "resistance_level": 0.0,
        })),
        tf_4h=TimeframeDetail(**tf_data.get("tf_4h", {
            "trend": "NEUTRAL_CHOPPY",
            "key_indicators_summary": "4H 平衡震荡",
            "support_level": 0.0,
            "resistance_level": 0.0,
        })),
        tf_1d=TimeframeDetail(**tf_data.get("tf_1d", {
            "trend": "NEUTRAL_CHOPPY",
            "key_indicators_summary": "1D 平衡震荡",
            "support_level": 0.0,
            "resistance_level": 0.0,
        })),
    )

    # 2. 组装衍生品情绪
    deriv_data = state.get("derivatives_sentiment", {})
    derivatives_sentiment = DerivativesSentiment(
        funding_rate_bias=deriv_data.get("funding_rate_bias", "NEUTRAL"),
        open_interest_interpretation=deriv_data.get("open_interest_interpretation", "持仓量平稳"),
        long_short_ratio_state=deriv_data.get("long_short_ratio_state", "多空比例均衡"),
        sentiment_score=float(deriv_data.get("sentiment_score", 0.0)),
    )

    # 3. 组装信号
    signal_data = state.get("signal", {})
    signal = AISignal(
        action=signal_data.get("action", "HOLD_WAIT"),
        confidence=float(signal_data.get("confidence", 0.5)),
        urgency=signal_data.get("urgency", "LOW"),
    )

    # 4. 组装交易计划
    plan_data = state.get("trade_plan", {})
    tp_raw_list = plan_data.get("take_profit_levels", [])
    tp_levels = [
        TakeProfitLevel(
            price=float(tp.get("price", 0.0)),
            percentage=float(tp.get("percentage", 0.5)),
            description=str(tp.get("description", "分批止盈")),
        )
        for tp in tp_raw_list
    ] if tp_raw_list else [
        TakeProfitLevel(price=0.0, percentage=1.0, description="无止盈计划")
    ]

    trade_plan = TradePlan(
        entry_range=plan_data.get("entry_range", [0.0, 0.0]),
        take_profit_levels=tp_levels,
        stop_loss_price=float(plan_data.get("stop_loss_price", 0.0)),
        risk_reward_ratio=float(plan_data.get("risk_reward_ratio", 1.5)),
        suggested_leverage=int(plan_data.get("suggested_leverage", 1)),
        order_type=plan_data.get("order_type", "LIMIT"),
    )

    # 5. 组装风控评估
    risk_data = state.get("risk_assessment", {})
    risk_assessment = RiskAssessment(
        key_risks=risk_data.get("key_risks", ["市场突发波动风险"]),
        invalidation_condition=risk_data.get("invalidation_condition", "突破关键位置失效"),
        max_holding_time_hours=float(risk_data.get("max_holding_time_hours", 24.0)),
    )

    # 6. 整合思维链步骤为完整思考文本
    steps = state.get("thinking_steps", [])
    thinking_lines = ["<think>"]
    thinking_lines.append("### 【OKX-Dog 机构级多智能体量化决策推导流】")
    for s in steps:
        thinking_lines.append(f"\n#### [{s.get('stage_name', s.get('node'))}]")
        thinking_lines.append(s.get("thought", ""))
    thinking_lines.append("\n</think>")
    full_thinking_process = "\n".join(thinking_lines)

    # 7. 实例化生产契约对象
    model_name = state.get("llm_config", {}).get("model_name") or "okx-dog-multi-agent-v2"

    response = AIAnalysisResponse(
        analysis_id=analysis_id,
        symbol=symbol,
        timestamp=now_ms,
        market_regime=MarketRegime(market_regime_str),
        timeframe_analysis=tf_analysis,
        derivatives_sentiment=derivatives_sentiment,
        signal=signal,
        trade_plan=trade_plan,
        risk_assessment=risk_assessment,
        reasoning_summary=state.get("reasoning_summary", "量化研判完成"),
        reasoning_details=state.get("reasoning_details", "多周期共振、红蓝辩论与硬风控校验通过"),
        model_used=model_name,
        latency_ms=latency_ms,
        thinking_process=full_thinking_process,
    )

    last_step: ThinkingStep = {
        "node": "ResponseFormatter",
        "stage_name": "契约模型格式化与思维链收敛",
        "thought": f"【格式化完成】成功生成合规交易契约: Action={signal.action}, 置信度={signal.confidence:.2f}, 耗时={latency_ms}ms。",
        "timestamp_ms": now_ms,
    }

    return {
        "final_response": response.model_dump(),
        "model_used": model_name,
        "latency_ms": latency_ms,
        "thinking_steps": [last_step],
    }
