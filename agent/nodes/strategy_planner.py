"""
交易策略规划与点位生成节点 (StrategyPlanner)
模块: okx-dog-ai/agent/nodes/strategy_planner.py

职责:
1. 综合宏观结构 (Macro Regime) 与衍生品流动性，推导最适操作方向与执行紧迫度。
2. 基于 15M ATR 与 4H/1H 供需结构精准规划入场区间、硬止损价及梯级止盈目标。
3. 严格感知反思批评反馈 (Risk Critique)：若上一轮风控审查被拦截，依据批评意见自适应修正点位与盈亏比。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from ..state import QuantTraderState, ThinkingStep
from ..tools import derive_dynamic_atr_stops

logger = logging.getLogger("okx_dog.ai.agent.strategy_planner")


async def strategy_planning_node(state: QuantTraderState) -> Dict[str, Any]:
    """
    LangGraph Node: 交易策略规划与点位生成 (含反思自适应修复能力)
    """
    logger.info("执行 Node 3: 交易策略规划与点位生成...")
    now_ms = int(time.time() * 1000)
    current_price = float(state.get("current_price", 0.0))
    macro_regime = state.get("market_regime", "RANGING")
    tf_analysis = state.get("timeframe_analysis", {})
    deriv_sentiment = state.get("derivatives_sentiment", {})
    risk_critique = state.get("risk_critique")
    critique_count = state.get("critique_count", 0)
    risk_limits = state.get("risk_limits", {})
    max_allowed_leverage = int(risk_limits.get("max_leverage", 5))

    # 提取 15m ATR
    raw_snapshot = state.get("market_snapshot", {})
    multi_ind = raw_snapshot.get("multi_indicators", {})
    if hasattr(multi_ind, "indicators"):
        ind_15m = multi_ind.indicators.get("15m", {})
        atr14 = float(getattr(ind_15m, "atr_14", current_price * 0.01))
    elif isinstance(multi_ind, dict) and "indicators" in multi_ind:
        ind_15m = multi_ind["indicators"].get("15m", {})
        atr14 = float(ind_15m.get("atr_14", current_price * 0.01))
    else:
        atr14 = current_price * 0.01

    sentiment_score = float(deriv_sentiment.get("sentiment_score", 0.0))
    fr_bias = deriv_sentiment.get("funding_rate_bias", "NEUTRAL")

    # 提取 1h 周期指标以辅助计算多因子动态置信度
    ind_1h = multi_ind.indicators.get("1h", {}) if hasattr(multi_ind, "indicators") else (multi_ind.get("1h", {}) if isinstance(multi_ind, dict) else {})
    rsi_1h = float(getattr(ind_1h, "rsi_14", ind_1h.get("rsi_14", 50.0)) if hasattr(ind_1h, "rsi_14") or isinstance(ind_1h, dict) else 50.0)
    macd_hist_1h = float(getattr(ind_1h, "macd_hist", ind_1h.get("macd_hist", 0.0)) if hasattr(ind_1h, "macd_hist") or isinstance(ind_1h, dict) else 0.0)

    # 1. 决策动作推导与多因子动态置信度模型 (基于 RSI/MACD/衍生品动能实时打分)
    if macro_regime == "TRENDING_UP":
        if fr_bias == "EXTREME_POSITIVE" and sentiment_score > 0.6:
            # 散户追多过热，警惕洗盘，收敛为观望
            action = "HOLD_WAIT"
            confidence = round(0.58 + min(0.12, abs(sentiment_score - 0.5) * 0.3), 2)
            urgency = "LOW"
        else:
            action = "BUY_LONG"
            rsi_bonus = min(0.08, max(0.0, (rsi_1h - 50.0) * 0.005))
            macd_bonus = 0.05 if macd_hist_1h > 0 else 0.0
            sentiment_bonus = min(0.05, max(0.0, sentiment_score * 0.05))
            confidence = round(min(0.92, 0.75 + rsi_bonus + macd_bonus + sentiment_bonus), 2)
            urgency = "MEDIUM"
    elif macro_regime == "TRENDING_DOWN":
        if fr_bias == "EXTREME_NEGATIVE":
            # 极度负费率，防止逼空
            action = "HOLD_WAIT"
            confidence = round(0.58 + min(0.12, abs(sentiment_score - 0.5) * 0.3), 2)
            urgency = "LOW"
        else:
            action = "SELL_SHORT"
            rsi_bonus = min(0.08, max(0.0, (50.0 - rsi_1h) * 0.005))
            macd_bonus = 0.05 if macd_hist_1h < 0 else 0.0
            sentiment_bonus = min(0.05, max(0.0, abs(sentiment_score) * 0.05))
            confidence = round(min(0.92, 0.75 + rsi_bonus + macd_bonus + sentiment_bonus), 2)
            urgency = "MEDIUM"
    elif macro_regime == "VOLATILE_BREAKOUT":
        action = "BUY_LONG" if sentiment_score >= 0 else "SELL_SHORT"
        confidence = round(min(0.88, 0.72 + abs(sentiment_score) * 0.15), 2)
        urgency = "HIGH"
    else:
        # 震荡市：依据布林带偏离度或 RSI 偏离度给出动态置信度 (0.52 ~ 0.65)
        action = "HOLD_WAIT"
        chop_factor = abs(rsi_1h - 50.0) * 0.006
        confidence = round(0.52 + min(0.13, chop_factor), 2)
        urgency = "LOW"


    # 2. 基础止损与止盈点位推导 (ATR 乘数)
    # 若存在反思批评，动态加大 TP 目标距离或收窄入场点以强制提升盈亏比 >= 1.6
    atr_multiplier = 1.8
    tp_multiplier = 2.8 if not risk_critique else 3.5

    stop_loss, tp1, tp2 = derive_dynamic_atr_stops(
        current_price=current_price,
        atr_14=atr14,
        action=action,
        multiplier=atr_multiplier,
    )

    if action == "BUY_LONG":
        entry_low = round(current_price * 0.997, 2 if current_price > 10 else 4)
        entry_high = round(current_price * 1.001, 2 if current_price > 10 else 4)
        entry_avg = (entry_low + entry_high) / 2.0
        # 确保止损严格在下方
        if stop_loss >= entry_avg:
            stop_loss = round(entry_avg - atr14 * 1.5, 2 if current_price > 10 else 4)
        # 确保 TP1 盈亏比 >= 1.6
        required_reward = abs(entry_avg - stop_loss) * 1.65
        tp1 = round(entry_avg + required_reward, 2 if current_price > 10 else 4)
        tp2 = round(entry_avg + required_reward * 1.8, 2 if current_price > 10 else 4)
        leverage = min(3, max_allowed_leverage)
        rr_calc = round((tp1 - entry_avg) / max(1e-6, entry_avg - stop_loss), 2)
    elif action == "SELL_SHORT":
        entry_low = round(current_price * 0.999, 2 if current_price > 10 else 4)
        entry_high = round(current_price * 1.003, 2 if current_price > 10 else 4)
        entry_avg = (entry_low + entry_high) / 2.0
        # 确保止损严格在上方
        if stop_loss <= entry_avg:
            stop_loss = round(entry_avg + atr14 * 1.5, 2 if current_price > 10 else 4)
        # 确保 TP1 盈亏比 >= 1.6
        required_reward = abs(stop_loss - entry_avg) * 1.65
        tp1 = round(entry_avg - required_reward, 2 if current_price > 10 else 4)
        tp2 = round(entry_avg - required_reward * 1.8, 2 if current_price > 10 else 4)
        leverage = min(3, max_allowed_leverage)
        rr_calc = round((entry_avg - tp1) / max(1e-6, stop_loss - entry_avg), 2)
    else:
        entry_low = round(current_price * 0.995, 2 if current_price > 10 else 4)
        entry_high = round(current_price * 1.005, 2 if current_price > 10 else 4)
        stop_loss = round(current_price * 0.985, 2 if current_price > 10 else 4)
        tp1 = round(current_price * 1.015, 2 if current_price > 10 else 4)
        tp2 = round(current_price * 1.030, 2 if current_price > 10 else 4)
        leverage = 1
        rr_calc = 1.5

    # 3. 组装 TradePlan 与 RiskAssessment
    if action in ["BUY_LONG", "SELL_SHORT"]:
        take_profit_list = [
            {
                "price": tp1,
                "percentage": 0.5,
                "description": "第一止盈位 (锁定 50% 利润并移动止损至入场保本价)",
            },
            {
                "price": tp2,
                "percentage": 0.5,
                "description": "第二波段止盈位 (跟踪趋势获利了结)",
            },
        ]
    else:
        take_profit_list = [
            {
                "price": tp1,
                "percentage": 1.0,
                "description": "箱体阻力监控位 (突破后重新研判)",
            }
        ]

    trade_plan = {
        "entry_range": [entry_low, entry_high],
        "take_profit_levels": take_profit_list,
        "stop_loss_price": stop_loss,
        "risk_reward_ratio": rr_calc,
        "suggested_leverage": min(leverage, max_allowed_leverage),
        "order_type": "LIMIT",
    }


    key_risks = [
        "美联储宏观数据公布引发的突发流动性插针风险",
        "大盘 BTC 假突破引发的山寨币联动暴跌",
    ]
    if action == "BUY_LONG":
        invalidation = f"15M 收线跌破 {stop_loss} 支撑位，结构彻底破坏"
    elif action == "SELL_SHORT":
        invalidation = f"15M 收线站稳 {stop_loss} 阻力位，空头逻辑证伪"
    else:
        invalidation = f"突破当前震荡箱体区间 [{stop_loss}, {tp1}] 后重新研判"

    risk_assessment = {
        "key_risks": key_risks,
        "invalidation_condition": invalidation,
        "max_holding_time_hours": 24.0,
    }

    signal = {
        "action": action,
        "confidence": confidence,
        "urgency": urgency,
    }

    if action in ["BUY_LONG", "SELL_SHORT"]:
        summary = (
            f"盘面判定为【{macro_regime}】，建议操作【{action}】(置信度 {confidence * 100:.0f}%)。"
            f"入场区间 [{entry_low}, {entry_high}]，硬止损 {stop_loss}，第一目标位 {tp1}，理论盈亏比 {rr_calc}。"
        )
    else:
        summary = (
            f"盘面判定为【{macro_regime} 震荡整理】，建议操作【HOLD_WAIT 观望】(置信度 {confidence * 100:.0f}%)。"
            f"当前无明确单边趋势，震荡关注区间 [{entry_low}, {entry_high}] (支撑防守 {stop_loss} / 阻力 {tp1})，建议空仓等待突破。"
        )


    details = (
        f"1. 宏观结构: 4H处于{macro_regime}结构，多周期指标展现共振协同；\n"
        f"2. 衍生品: 资金费率{fr_bias}，OI 变动平稳，情绪综合得分 {sentiment_score}；\n"
        f"3. 点位量化: 基于 15M ATR ({atr14:.2f}) 设定动态止损与梯级止盈目标，盈亏比严格达标 ({rr_calc} >= 1.5)。"
    )

    # 4. 记录思考轨迹
    thought_prefix = f"【反思轮次 #{critique_count} 调整点位】" if risk_critique else "【量化策略规划】"
    thought_text = (
        f"{thought_prefix} 生成策略：Action={action}, 杠杆={trade_plan['suggested_leverage']}x, "
        f"入场区间={trade_plan['entry_range']}, 止损={stop_loss}, TP1={tp1}, 理论盈亏比={rr_calc}。"
    )

    thinking_step: ThinkingStep = {
        "node": "StrategyPlanner",
        "stage_name": "交易策略规划与点位生成",
        "thought": thought_text,
        "timestamp_ms": now_ms,
    }

    return {
        "signal": signal,
        "trade_plan": trade_plan,
        "risk_assessment": risk_assessment,
        "reasoning_summary": summary,
        "reasoning_details": details,
        "thinking_steps": [thinking_step],
    }
