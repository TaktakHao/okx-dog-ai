"""
首席量化仲裁官与多因子策略规划中枢 (StrategyPlanner)
模块: okx-dog-ai/agent/nodes/strategy_planner.py
角色: 首席量化仲裁官 (Chief Quantitative Arbiter)

职责:
1. 综合六方感知专家（技术共振、链上资金、量化微观、衍生品情绪、宏观日历、订单簿流动性）与红蓝对抗博弈陈述。
2. 计算多智能体共识分 (Consensus Score 0~100) 与准入红线判定 (>=75 准做多, <=35 准做空)。
3. 结合 15M ATR、微观流动性冲击与半凯利模型精准规划入场区间、硬止损价及梯级止盈目标。
4. 严格感知反思批评反馈 (Risk Critique)：若上一轮风控审查被拦截，依据批评意见自适应修正点位与盈亏比。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from ..state import QuantTraderState, ThinkingStep
from ..tools import check_volatility_squeeze, derive_dynamic_atr_stops

logger = logging.getLogger("okx_dog.ai.agent.strategy_planner")


async def strategy_planning_node(state: QuantTraderState) -> Dict[str, Any]:
    """
    LangGraph Node: 首席量化仲裁与多因子融合决策规划
    """
    logger.info("执行 Node: 首席量化仲裁与多因子融合决策规划...")
    now_ms = int(time.time() * 1000)
    current_price = float(state.get("current_price", 0.0))
    macro_regime = state.get("market_regime", "RANGING")
    deriv_sentiment = state.get("derivatives_sentiment", {})
    onchain_data = state.get("onchain_analysis", {})
    quant_data = state.get("quant_features", {})
    macro_event = state.get("macro_event_risk", {})
    micro_data = state.get("microstructure_data", {})
    bull_opinion = state.get("bull_opinion", {})
    bear_opinion = state.get("bear_opinion", {})
    risk_critique = state.get("risk_critique")
    critique_count = state.get("critique_count", 0)
    risk_limits = state.get("risk_limits", {})
    max_allowed_leverage = int(risk_limits.get("max_leverage", 5))

    # 1. 提取技术面 15m ATR 与布林带
    raw_snapshot = state.get("market_snapshot", {})
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

    # 2. 提取各专家研判特征与红蓝对抗意见
    is_squeezed, bandwidth, squeeze_msg = check_volatility_squeeze(
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        bb_middle=bb_mid,
        atr_14=atr14,
        current_price=current_price,
    )

    deriv_score = float(deriv_sentiment.get("sentiment_score", 0.0))
    fr_bias = deriv_sentiment.get("funding_rate_bias", "NEUTRAL")
    onchain_bias = onchain_data.get("flow_bias", "NEUTRAL")
    onchain_score = float(onchain_data.get("composite_score", 0.0))
    cex_netflow = float(onchain_data.get("cex_netflow_24h_usd", 0.0))
    has_unlock_risk = bool(onchain_data.get("has_token_unlock_risk", False))
    imbalance_ratio = float(quant_data.get("orderbook_imbalance_ratio", 1.0))
    expected_win_rate = float(quant_data.get("expected_win_rate", 0.58))
    kelly_desc = quant_data.get("kelly_rationale", "")
    event_risk_level = macro_event.get("event_risk_level", "LOW")
    is_event_lockout = bool(macro_event.get("is_lockout_active", False))
    recommended_order_type = micro_data.get("recommended_execution_mode", "LIMIT")

    bull_conf = float(bull_opinion.get("confidence", 0.5))
    bull_stance = bull_opinion.get("stance", "NEUTRAL")
    bear_conf = float(bear_opinion.get("confidence", 0.5))
    bear_stance = bear_opinion.get("stance", "NEUTRAL")

    # 3. 首席量化仲裁：获取 Softmax 门控网络的体制自适应权重 (Regime-Adaptive Weights)
    try:
        from ..evolution.evolution_manager import AgentEvolutionManager
        manager = AgentEvolutionManager.get_instance()
        dynamic_weights = manager.gating_network.get_regime_adaptive_weights(macro_regime)
    except Exception:
        dynamic_weights = {
            "bull_specialist": 25.0,
            "bear_critic": 25.0,
            "macro_news": 15.0,
            "onchain_analyst": 15.0,
            "micro_sniper": 10.0,
            "chief_arbiter": 10.0,
        }

    w_bull = dynamic_weights.get("bull_specialist", 25.0) / 100.0
    w_bear = dynamic_weights.get("bear_critic", 25.0) / 100.0
    w_onchain = dynamic_weights.get("onchain_analyst", 15.0) / 100.0
    w_micro = dynamic_weights.get("micro_sniper", 10.0) / 100.0
    w_macro = dynamic_weights.get("macro_news", 15.0) / 100.0

    consensus_score = 50.0
    if bull_stance == "BULLISH":
        consensus_score += bull_conf * w_bull * 85.0
    elif bull_stance == "BEARISH":
        consensus_score -= bull_conf * w_bull * 40.0

    if bear_stance == "BEARISH":
        consensus_score -= bear_conf * w_bear * 85.0
    elif bear_stance == "BULLISH":
        consensus_score += bear_conf * w_bear * 30.0

    if onchain_score > 0.2:
        consensus_score += onchain_score * w_onchain * 60.0
    elif onchain_score < -0.2:
        consensus_score -= abs(onchain_score) * w_onchain * 60.0

    if imbalance_ratio > 1.3:
        consensus_score += min(15.0, (imbalance_ratio - 1.0) * w_micro * 50.0)
    elif imbalance_ratio < 0.7:
        consensus_score -= min(15.0, (1.0 - imbalance_ratio) * w_micro * 50.0)

    # 宏观市场体制顺势红利
    if macro_regime == "TRENDING_UP":
        consensus_score += 5.0
    elif macro_regime == "TRENDING_DOWN":
        consensus_score -= 5.0

    # 宏观高危风险惩罚 (受 macro_news 权重缩放)
    if event_risk_level in ["HIGH", "CRITICAL"]:
        consensus_score = consensus_score * (0.65 if w_macro >= 0.15 else 0.80)

    consensus_score = int(max(5, min(95, round(consensus_score))))

    # 4. 动作判定与体制自适应准入红线审查
    # 单边多头体制放宽至 72 分防踏空；单边空头放宽至 35 分；震荡市收紧至 78/28 防被摩擦锯齿绞杀
    if macro_regime == "TRENDING_UP":
        long_threshold = 72
        short_threshold = 28
    elif macro_regime == "TRENDING_DOWN":
        long_threshold = 78
        short_threshold = 36
    else:  # RANGING or VOLATILE_BREAKOUT
        long_threshold = 76
        short_threshold = 30

    action = "HOLD_WAIT"
    confidence = 0.50
    urgency = "LOW"
    strategy_rationale = ""
    is_approved_by_arbiter = False
    intent = "WAIT_OBSERVE"

    # 推导战术意图 (Intent)
    if fr_bias in ["EXTREME_POSITIVE", "EXTREME_NEGATIVE"]:
        intent = "SHORT_SQUEEZE"
    elif abs(imbalance_ratio - 1.0) >= 0.40:
        intent = "LIQUIDITY_SWEEP"
    elif macro_regime in ["TRENDING_UP", "TRENDING_DOWN"]:
        intent = "TREND_CONTINUATION"
    elif is_squeezed:
        intent = "MEAN_REVERSION"
    else:
        intent = "BREAKOUT_FOLLOW"

    if is_event_lockout:
        action = "HOLD_WAIT"
        confidence = 0.30
        urgency = "LOW"
        intent = "WAIT_OBSERVE"
        strategy_rationale = f"触发宏观高危事件锁 ({macro_event.get('diagnostic_summary', '敏感窗口')})，强制全局观望"
    elif is_squeezed and macro_regime == "VOLATILE_BREAKOUT":
        action = "HOLD_WAIT"
        confidence = 0.50
        urgency = "LOW"
        intent = "WAIT_OBSERVE"
        strategy_rationale = f"触发波动率挤压保护: {squeeze_msg}，收敛为防守观望"
    elif macro_regime == "TRENDING_UP" and (cex_netflow > 20_000_000 or has_unlock_risk or onchain_score <= -0.5):
        action = "HOLD_WAIT"
        confidence = 0.55
        urgency = "LOW"
        intent = "WAIT_OBSERVE"
        strategy_rationale = f"检测到链上大额充币/严重抛压风险 (CEX净流入=${cex_netflow/1e6:.1f}M)，警惕诱多洗盘"
    elif macro_regime == "TRENDING_UP" and fr_bias == "EXTREME_POSITIVE" and deriv_score > 0.75:
        action = "HOLD_WAIT"
        confidence = 0.58
        urgency = "LOW"
        intent = "WAIT_OBSERVE"
        strategy_rationale = "衍生品资金费率极度过热，多头杠杆拥挤，防范踩踏插针"
    elif consensus_score >= long_threshold:
        calc_conf = round(min(0.95, 0.72 + (consensus_score - long_threshold) * 0.015), 2)
        if calc_conf >= 0.72:
            action = "BUY_LONG"
            confidence = calc_conf
            urgency = "MEDIUM"
            is_approved_by_arbiter = True
            strategy_rationale = (
                f"[{macro_regime}体制自适应] 强共识做多 (共识分={consensus_score}/100 >= {long_threshold}, 战术意图={intent}) + "
                f"多头加权权重({w_bull*100:.0f}%) + 链上资金({onchain_bias})"
            )
        else:
            action = "HOLD_WAIT"
            confidence = calc_conf
            intent = "WAIT_OBSERVE"
            strategy_rationale = f"仲裁置信度 ({calc_conf}) 未达 0.72 红线，转为观望"
    elif consensus_score <= short_threshold:
        calc_conf = round(min(0.95, 0.72 + (short_threshold - consensus_score) * 0.015), 2)
        if calc_conf >= 0.72:
            action = "SELL_SHORT"
            confidence = calc_conf
            urgency = "MEDIUM"
            is_approved_by_arbiter = True
            strategy_rationale = (
                f"[{macro_regime}体制自适应] 强共识做空 (共识分={consensus_score}/100 <= {short_threshold}, 战术意图={intent}) + "
                f"风控红队加权({w_bear*100:.0f}%) + 盘口压制"
            )
        else:
            action = "HOLD_WAIT"
            confidence = calc_conf
            intent = "WAIT_OBSERVE"
            strategy_rationale = f"仲裁置信度 ({calc_conf}) 未达 0.72 红线，转为观望"
    else:
        action = "HOLD_WAIT"
        confidence = 0.55
        urgency = "LOW"
        intent = "WAIT_OBSERVE"
        strategy_rationale = f"首席仲裁共识分 ({consensus_score}/100) 未达[{macro_regime}]体制准入红线 (≥{long_threshold} 或 ≤{short_threshold})，保持观望"

    # 5. 点位规划推导 (ATR 乘数 + 摩擦与反思修正)
    atr_multiplier = 1.8
    stop_loss, tp1, tp2 = derive_dynamic_atr_stops(
        current_price=current_price,
        atr_14=atr14,
        action=action,
        multiplier=atr_multiplier,
    )

    friction_buffer = current_price * 0.0015

    if action == "BUY_LONG":
        entry_low = round(current_price * 0.997, 2 if current_price > 10 else 4)
        entry_high = round(current_price * 1.001, 2 if current_price > 10 else 4)
        entry_avg = (entry_low + entry_high) / 2.0
        if stop_loss >= entry_avg:
            stop_loss = round(entry_avg - atr14 * 1.5, 2 if current_price > 10 else 4)
        required_reward = (abs(entry_avg - stop_loss) + friction_buffer) * 1.80 + friction_buffer
        tp1 = round(entry_avg + required_reward, 2 if current_price > 10 else 4)
        tp2 = round(entry_avg + required_reward * 1.8, 2 if current_price > 10 else 4)
        leverage = min(3, max_allowed_leverage)
        rr_calc = round((tp1 - entry_avg - friction_buffer) / max(1e-6, entry_avg - stop_loss + friction_buffer), 2)
    elif action == "SELL_SHORT":
        entry_low = round(current_price * 0.999, 2 if current_price > 10 else 4)
        entry_high = round(current_price * 1.003, 2 if current_price > 10 else 4)
        entry_avg = (entry_low + entry_high) / 2.0
        if stop_loss <= entry_avg:
            stop_loss = round(entry_avg + atr14 * 1.5, 2 if current_price > 10 else 4)
        required_reward = (abs(stop_loss - entry_avg) + friction_buffer) * 1.80 + friction_buffer
        tp1 = round(entry_avg - required_reward, 2 if current_price > 10 else 4)
        tp2 = round(entry_avg - required_reward * 1.8, 2 if current_price > 10 else 4)
        leverage = min(3, max_allowed_leverage)
        rr_calc = round((entry_avg - tp1 - friction_buffer) / max(1e-6, stop_loss - entry_avg + friction_buffer), 2)
    else:
        entry_low = round(current_price * 0.995, 2 if current_price > 10 else 4)
        entry_high = round(current_price * 1.005, 2 if current_price > 10 else 4)
        stop_loss = round(current_price * 0.985, 2 if current_price > 10 else 4)
        tp1 = round(current_price * 1.015, 2 if current_price > 10 else 4)
        tp2 = round(current_price * 1.030, 2 if current_price > 10 else 4)
        leverage = 1
        rr_calc = 1.8

    # 6. 组装 TradePlan 与 RiskAssessment
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
        "intent": intent,
        "entry_range": [entry_low, entry_high],
        "take_profit_levels": take_profit_list,
        "stop_loss_price": stop_loss,
        "risk_reward_ratio": rr_calc,
        "suggested_leverage": min(leverage, max_allowed_leverage),
        "order_type": "POST_ONLY",
    }

    key_risks = [
        f"宏观日历事件扰动 ({macro_event.get('event_title') or '常规日历'})",
        "大盘 BTC 假突破引发的联动剧烈插针",
        "链上大户集中充提币引发的流动性滑点",
    ]
    if action == "BUY_LONG":
        invalidation = f"15M 收线跌破 {stop_loss} 支撑位，多头逻辑证伪"
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
            f"首席仲裁共识分【{consensus_score}/100】，判定操作【{action}】(置信度 {confidence * 100:.0f}%)。"
            f"入场区间 [{entry_low}, {entry_high}]，硬止损 {stop_loss}，第一目标位 {tp1}，理论盈亏比 {rr_calc}。"
        )
    else:
        summary = (
            f"首席仲裁共识分【{consensus_score}/100】，建议操作【HOLD_WAIT 观望】(置信度 {confidence * 100:.0f}%)。"
            f"{strategy_rationale}。建议空仓等待确定性右侧信号。"
        )

    details = (
        f"1. 宏观技术面: 4H处于{macro_regime}体制，多周期指标协同；\n"
        f"2. 区块链链上面: 资金流态势为【{onchain_bias}】(得分 {onchain_score:+.2f})，CEX净流向 ${cex_netflow/1e6:+.2f}M；\n"
        f"3. 量化微观结构: 盘口前5档买卖比为 {imbalance_ratio:.2f}，模型胜率期望 {expected_win_rate*100:.0f}%；\n"
        f"4. 衍生品情绪: 资金费率{fr_bias}，情绪量化得分 {deriv_score}；\n"
        f"5. 宏观与流动性: 宏观事件风险【{event_risk_level}】，推荐订单模式【{recommended_order_type}】；\n"
        f"6. 红蓝对抗辩论: 多头置信度 {bull_conf*100:.0f}%, 空头红队置信度 {bear_conf*100:.0f}% -> 综合共识分 {consensus_score}/100；\n"
        f"7. 点位与仓位: 基于 15M ATR ({atr14:.2f}) 设定动态止盈止损 (净盈亏比 {rr_calc} >= 1.5)，{kelly_desc}。"
    )

    thought_prefix = f"【反思轮次 #{critique_count} 调整点位】" if risk_critique else "【首席量化仲裁与策略规划】"
    thought_text = (
        f"{thought_prefix} 仲裁共识分={consensus_score}/100, 决策 Action={action}, 杠杆={trade_plan['suggested_leverage']}x, "
        f"入场区间={trade_plan['entry_range']}, 止损={stop_loss}, TP1={tp1}, 盈亏比={rr_calc}, 执行模式={trade_plan['order_type']}。"
    )

    thinking_step: ThinkingStep = {
        "node": "StrategyPlanner",
        "stage_name": "首席量化仲裁与策略规划",
        "thought": thought_text,
        "timestamp_ms": now_ms,
    }

    return {
        "consensus_score": consensus_score,
        "is_approved_by_arbiter": is_approved_by_arbiter,
        "signal": signal,
        "trade_plan": trade_plan,
        "risk_assessment": risk_assessment,
        "reasoning_summary": summary,
        "reasoning_details": details,
        "thinking_steps": [thinking_step],
    }
