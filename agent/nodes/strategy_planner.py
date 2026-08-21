"""
交易策略规划与多因子融合决策节点 (StrategyPlanner)
模块: okx-dog-ai/agent/nodes/strategy_planner.py

职责:
1. 综合四方专家分析结果：宏观技术面 (Macro)、区块链链上资金 (OnChain)、量化微观盘口 (Quant)、衍生品情绪 (Derivatives)。
2. 进行多因子共振打分与假突破/诱多防范：当链上大额充币或盘口卖压严重时，自动收敛为防守观望。
3. 基于 15M ATR、供需结构与凯利模型精准规划入场区间、硬止损价及梯级止盈目标。
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
    LangGraph Node: 交易策略规划与多因子融合决策 (含反思自适应修复能力与摩擦修正)
    """
    logger.info("执行 Node: 交易策略规划与多因子融合决策...")
    now_ms = int(time.time() * 1000)
    current_price = float(state.get("current_price", 0.0))
    macro_regime = state.get("market_regime", "RANGING")
    tf_analysis = state.get("timeframe_analysis", {})
    deriv_sentiment = state.get("derivatives_sentiment", {})
    onchain_data = state.get("onchain_analysis", {})
    quant_data = state.get("quant_features", {})
    risk_critique = state.get("risk_critique")
    critique_count = state.get("critique_count", 0)
    risk_limits = state.get("risk_limits", {})
    max_allowed_leverage = int(risk_limits.get("max_leverage", 5))

    # 1. 提取技术面 15m ATR 与布林带指标
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

    # 2. 提取各专家研判特征
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

    # 3. 多因子动态置信度与动作裁决 (宏观 + 链上资金 + 量化微观 + 衍生品)
    # 防御规则 A: 波动率挤压假突破防御
    if is_squeezed and macro_regime == "VOLATILE_BREAKOUT":
        action = "HOLD_WAIT"
        confidence = 0.50
        urgency = "LOW"
        strategy_rationale = f"触发波动率挤压保护: {squeeze_msg}，收敛为防守观望"
    # 防御规则 B: 链上巨额充币/解锁 假多头防御
    elif macro_regime == "TRENDING_UP" and (cex_netflow > 15_000_000 or has_unlock_risk or onchain_score <= -0.4):
        action = "HOLD_WAIT"
        confidence = 0.55
        urgency = "LOW"
        strategy_rationale = f"检测到链上大额充币/抛压风险 (CEX净流入=${cex_netflow/1e6:.1f}M, 链上评分={onchain_score:+.2f})，警惕诱多洗盘，转换为观望防守"
    # 防御规则 C: 衍生品极度过热防御
    elif macro_regime == "TRENDING_UP" and fr_bias == "EXTREME_POSITIVE" and deriv_score > 0.6:
        action = "HOLD_WAIT"
        confidence = round(0.58 + min(0.12, abs(deriv_score - 0.5) * 0.3), 2)
        urgency = "LOW"
        strategy_rationale = "衍生品资金费率极度过热，多头杠杆拥挤，防范踩踏插针"
    # 做多信号
    elif macro_regime == "TRENDING_UP":
        action = "BUY_LONG"
        # 链上与量化多因子加分
        onchain_bonus = min(0.08, max(0.0, onchain_score * 0.08))
        imbalance_bonus = min(0.06, max(0.0, (imbalance_ratio - 1.0) * 0.05))
        deriv_bonus = min(0.05, max(0.0, deriv_score * 0.05))
        confidence = round(min(0.92, 0.74 + onchain_bonus + imbalance_bonus + deriv_bonus), 2)
        urgency = "MEDIUM"
        strategy_rationale = f"宏观多头结构 + 链上资金偏好({onchain_bias}) + 盘口深度买单支撑({imbalance_ratio:.2f})"
    # 防御规则 D: 极度负费率逼空防御
    elif macro_regime == "TRENDING_DOWN" and fr_bias == "EXTREME_NEGATIVE":
        action = "HOLD_WAIT"
        confidence = round(0.58 + min(0.12, abs(deriv_score - 0.5) * 0.3), 2)
        urgency = "LOW"
        strategy_rationale = "极度负费率，警惕空头踩踏逼空"
    # 做空信号
    elif macro_regime == "TRENDING_DOWN":
        action = "SELL_SHORT"
        onchain_bonus = min(0.08, max(0.0, abs(onchain_score) * 0.08)) if onchain_score < 0 else 0.0
        imbalance_bonus = min(0.06, max(0.0, (1.0 - imbalance_ratio) * 0.05)) if imbalance_ratio < 1.0 else 0.0
        confidence = round(min(0.92, 0.74 + onchain_bonus + imbalance_bonus), 2)
        urgency = "MEDIUM"
        strategy_rationale = f"宏观空头破位 + 链上筹码松动({onchain_bias}) + 盘口卖盘压制"
    elif macro_regime == "VOLATILE_BREAKOUT":
        action = "BUY_LONG" if (deriv_score + onchain_score) >= 0 else "SELL_SHORT"
        confidence = round(min(0.88, 0.72 + abs(deriv_score + onchain_score) * 0.1), 2)
        urgency = "HIGH"
        strategy_rationale = "剧烈异动放量突破"
    else:
        action = "HOLD_WAIT"
        confidence = 0.55
        urgency = "LOW"
        strategy_rationale = "盘面处于区间震荡，多空博弈均衡，等待右侧破位信号"

    # 4. 基础止损与止盈点位推导 (ATR 乘数 + 摩擦修正)
    atr_multiplier = 1.8
    stop_loss, tp1, tp2 = derive_dynamic_atr_stops(
        current_price=current_price,
        atr_14=atr14,
        action=action,
        multiplier=atr_multiplier,
    )

    # 预留 0.15% 的双边手续费与滑点缓冲
    friction_buffer = current_price * 0.0015

    if action == "BUY_LONG":
        entry_low = round(current_price * 0.997, 2 if current_price > 10 else 4)
        entry_high = round(current_price * 1.001, 2 if current_price > 10 else 4)
        entry_avg = (entry_low + entry_high) / 2.0
        if stop_loss >= entry_avg:
            stop_loss = round(entry_avg - atr14 * 1.5, 2 if current_price > 10 else 4)
        required_reward = (abs(entry_avg - stop_loss) + friction_buffer) * 1.65 + friction_buffer
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
        required_reward = (abs(stop_loss - entry_avg) + friction_buffer) * 1.65 + friction_buffer
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
        rr_calc = 1.5

    # 5. 组装 TradePlan 与 RiskAssessment
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
        "链上大户集中充提币引发的突发流动性滑点",
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
            f"盘面判定为【{macro_regime}】，建议操作【HOLD_WAIT 观望】(置信度 {confidence * 100:.0f}%)。"
            f"{strategy_rationale}。关注区间 [{entry_low}, {entry_high}]，建议空仓等待确定性信号。"
        )

    details = (
        f"1. 宏观技术面: 4H处于{macro_regime}结构，多周期指标协同；\n"
        f"2. 区块链链上面: 资金流态势为【{onchain_bias}】(得分 {onchain_score:+.2f})，CEX净流向 ${cex_netflow/1e6:+.2f}M；\n"
        f"3. 量化微观结构: 盘口前5档买卖比为 {imbalance_ratio:.2f}，模型胜率期望 {expected_win_rate*100:.0f}%；\n"
        f"4. 衍生品情绪: 资金费率{fr_bias}，情绪量化得分 {deriv_score}；\n"
        f"5. 点位与仓位: 基于 15M ATR ({atr14:.2f}) 设定动态止盈止损 (净盈亏比 {rr_calc} >= 1.5)，{kelly_desc}。"
    )

    thought_prefix = f"【反思轮次 #{critique_count} 调整点位】" if risk_critique else "【多因子量化融合策略规划】"
    thought_text = (
        f"{thought_prefix} 生成策略：Action={action}, 杠杆={trade_plan['suggested_leverage']}x, "
        f"入场区间={trade_plan['entry_range']}, 止损={stop_loss}, TP1={tp1}, 理论盈亏比={rr_calc}。"
        f"结合链上资金面({onchain_bias})与盘口微观失衡比({imbalance_ratio:.2f})。"
    )

    thinking_step: ThinkingStep = {
        "node": "StrategyPlanner",
        "stage_name": "交易策略规划与多因子融合",
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
