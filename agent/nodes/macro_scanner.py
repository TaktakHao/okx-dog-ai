"""
宏观多周期共振扫描节点 (MacroTrendScanner)
模块: okx-dog-ai/agent/nodes/macro_scanner.py

职责:
1. 自上而下 (1D -> 4H -> 1H -> 15M) 扫描技术指标与均线排列状态。
2. 提取各周期关键支撑/阻力价位。
3. 判定宏观市场体制 (Market Regime: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE_BREAKOUT)。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from ..registry import BaseSpecialist, register_specialist
from ..state import QuantTraderState, ThinkingStep

logger = logging.getLogger("okx_dog.ai.agent.macro_scanner")


@register_specialist
class MacroTrendScannerSpecialist(BaseSpecialist):
    name = "macro_scanner"
    stage_name = "宏观多周期共振扫描"
    layer = "perception"
    description = "自上而下分析 1D/4H/1H/15M 多周期均线排列、指标强弱与关键供需位"

    async def analyze(self, state: QuantTraderState) -> Dict[str, Any]:
        return await macro_trend_scan_node(state)


def _analyze_single_tf(tf_name: str, ind: Dict[str, Any], current_price: float) -> Dict[str, Any]:
    """对单一周期进行量化指标研判与关键位提取"""
    ema20 = float(ind.get("ema_20", current_price))
    ema50 = float(ind.get("ema_50", current_price))
    ema200 = float(ind.get("ema_200", current_price) or current_price)
    rsi14 = float(ind.get("rsi_14", 50.0))
    bb_upper = float(ind.get("bb_upper", current_price * 1.02))
    bb_middle = float(ind.get("bb_middle", current_price))
    bb_lower = float(ind.get("bb_lower", current_price * 0.98))
    macd_hist = float(ind.get("macd_hist", 0.0))
    atr14 = float(ind.get("atr_14", current_price * 0.01))

    # 1. 趋势与形态研判 (结合 EMA 相对强弱、RSI 动能与 MACD 柱体综合打分)
    if rsi14 >= 72:
        trend = "OVERBOUGHT"
        summary = f"RSI({rsi14:.1f})严重超买，上轨压制逼近"
    elif rsi14 <= 28:
        trend = "OVERSOLD"
        summary = f"RSI({rsi14:.1f})处于极度超卖区，存在技术性反弹动能"
    elif (ema20 >= ema50 and (current_price >= ema20 or rsi14 >= 50.0)) or (rsi14 >= 56.0 and macd_hist >= 0):
        trend = "BULLISH"
        summary = f"EMA多头占优(EMA20={ema20:.1f}, EMA50={ema50:.1f})，RSI={rsi14:.1f}，MACD柱={macd_hist:+.2f}"
    elif (ema20 < ema50 and (current_price <= ema20 or rsi14 <= 50.0)) or (rsi14 <= 44.0 and macd_hist <= 0):
        trend = "BEARISH"
        summary = f"EMA空头占优(EMA20={ema20:.1f}, EMA50={ema50:.1f})，RSI={rsi14:.1f}，MACD柱={macd_hist:+.2f}"
    else:
        trend = "NEUTRAL_CHOPPY"
        summary = f"均线系统交织缠绕，震荡整理中 (RSI={rsi14:.1f})"

    # 2. 关键支撑与阻力推导
    support = round(min(bb_lower, current_price - atr14 * 1.5), 2 if current_price > 10 else 4)
    resistance = round(max(bb_upper, current_price + atr14 * 1.5), 2 if current_price > 10 else 4)

    return {
        "trend": trend,
        "key_indicators_summary": summary,
        "support_level": max(0.0, support),
        "resistance_level": max(0.0, resistance),
    }


async def macro_trend_scan_node(state: QuantTraderState) -> Dict[str, Any]:
    """
    LangGraph Node: 宏观多周期共振扫描
    """
    logger.info("执行 Node: 宏观多周期共振扫描...")
    now_ms = int(time.time() * 1000)
    current_price = float(state.get("current_price", 0.0))
    raw_snapshot = state.get("market_snapshot", {})

    # 提取多周期指标字典
    multi_ind_dict = raw_snapshot.get("multi_indicators", {})
    if hasattr(multi_ind_dict, "indicators"):
        ind_map = {k: (v.model_dump() if hasattr(v, "model_dump") else dict(v)) for k, v in multi_ind_dict.indicators.items()}
    elif isinstance(multi_ind_dict, dict) and "indicators" in multi_ind_dict:
        ind_map = multi_ind_dict["indicators"]
    elif isinstance(multi_ind_dict, dict):
        ind_map = multi_ind_dict
    else:
        ind_map = {}

    timeframe_analysis = {}
    for tf in ["15m", "1h", "4h", "1d"]:
        raw_tf_ind = ind_map.get(tf, {})
        if hasattr(raw_tf_ind, "model_dump"):
            raw_tf_ind = raw_tf_ind.model_dump()
        timeframe_analysis[f"tf_{tf}"] = _analyze_single_tf(tf, raw_tf_ind, current_price)

    # 综合判定 Market Regime (大周期权重优先 1d/4h > 1h > 15m)
    tf_1d_trend = timeframe_analysis["tf_1d"]["trend"]
    tf_4h_trend = timeframe_analysis["tf_4h"]["trend"]
    tf_1h_trend = timeframe_analysis["tf_1h"]["trend"]
    tf_15m_trend = timeframe_analysis["tf_15m"]["trend"]

    bull_count = sum(1 for t in [tf_1d_trend, tf_4h_trend, tf_1h_trend, tf_15m_trend] if t in ["BULLISH", "OVERBOUGHT"])
    bear_count = sum(1 for t in [tf_1d_trend, tf_4h_trend, tf_1h_trend, tf_15m_trend] if t in ["BEARISH", "OVERSOLD"])

    if bull_count >= 2 and bear_count == 0:
        market_regime = "TRENDING_UP"
    elif bear_count >= 2 and bull_count == 0:
        market_regime = "TRENDING_DOWN"
    elif bull_count > bear_count:
        market_regime = "TRENDING_UP"
    elif bear_count > bull_count:
        market_regime = "TRENDING_DOWN"
    else:
        # 检验是否为剧烈突破
        tf_15m_ind = ind_map.get("15m", {})
        bb_width = float(tf_15m_ind.get("bb_width_pct", 2.0) or 2.0)
        if bb_width > 4.0:
            market_regime = "VOLATILE_BREAKOUT"
        else:
            market_regime = "RANGING"

    thought_text = (
        f"【宏观多周期共振扫描】完成：当前标的现价 {current_price}。"
        f"1D大趋势={tf_1d_trend}, 4H中期结构={tf_4h_trend}, 1H波段={tf_1h_trend}。"
        f"宏观盘面体制判定为: {market_regime}。"
        f"4H核心支撑位={timeframe_analysis['tf_4h']['support_level']}, 阻力位={timeframe_analysis['tf_4h']['resistance_level']}。"
    )

    thinking_step: ThinkingStep = {
        "node": "MacroTrendScanner",
        "stage_name": "宏观多周期共振扫描",
        "thought": thought_text,
        "timestamp_ms": now_ms,
    }

    return {
        "market_regime": market_regime,
        "timeframe_analysis": timeframe_analysis,
        "thinking_steps": [thinking_step],
    }
