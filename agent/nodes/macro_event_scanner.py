"""
全球宏观日历与突发事件驱动分析专家 (MacroEventScanner)
模块: okx-dog-ai/agent/nodes/macro_event_scanner.py
角色: 全球宏观与事件驱动策略师 (financial-macro-event-strategist)

职责:
1. 监控重大宏观经济日历（美联储 FOMC 利率决议、美国 CPI/PPI、非农就业数据）倒计时窗口；
2. 提取并量化全网突发快讯 NLP 情绪与黑天鹅风险；
3. 在处于高危敏感窗口（如事件前 30 分钟）时触发一票否决安全锁，防范极端流动性真空插针。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from ..registry import BaseSpecialist, register_specialist
from ..state import QuantTraderState, ThinkingStep
from ..tools import evaluate_macro_event_risk

logger = logging.getLogger("okx_dog.ai.agent.macro_event_scanner")


@register_specialist
class MacroEventScannerSpecialist(BaseSpecialist):
    name = "macro_event_scanner"
    stage_name = "全球宏观事件与突发舆情扫描"
    layer = "perception"
    description = "扫描全球宏观日历发布窗口、突发快讯与系统性黑天鹅事件"

    async def analyze(self, state: QuantTraderState) -> Dict[str, Any]:
        return await macro_event_scanner_node(state)


async def macro_event_scanner_node(state: QuantTraderState) -> Dict[str, Any]:
    """
    LangGraph Node: 全球宏观日历与突发事件驱动扫描
    """
    logger.info("执行 Node: 全球宏观日历与突发事件驱动扫描...")
    now_ms = int(time.time() * 1000)
    raw_snapshot = state.get("market_snapshot", {})

    # 提取快照中的宏观日历与新闻情绪字段
    macro_event_info = raw_snapshot.get("macro_calendar_event") or {}
    minutes_to_event = macro_event_info.get("minutes_to_event")
    event_title = macro_event_info.get("event_title")
    news_sentiment_avg = float(raw_snapshot.get("news_sentiment_avg", 0.0))
    has_black_swan = bool(raw_snapshot.get("has_breaking_black_swan", False))

    event_risk_level, diagnostic_msg, is_lockout = evaluate_macro_event_risk(
        minutes_to_high_impact_event=minutes_to_event,
        event_title=event_title,
        news_sentiment_avg=news_sentiment_avg,
        has_breaking_black_swan=has_black_swan,
    )

    macro_event_data = {
        "event_risk_level": event_risk_level,
        "is_lockout_active": is_lockout,
        "minutes_to_event": minutes_to_event,
        "event_title": event_title,
        "news_sentiment_avg": news_sentiment_avg,
        "diagnostic_summary": diagnostic_msg,
    }

    thought_text = (
        f"【全球宏观日历与事件扫描】风险等级: {event_risk_level} (锁仓状态: {'强制锁仓' if is_lockout else '正常'})。"
        f"{diagnostic_msg}"
    )

    thinking_step: ThinkingStep = {
        "node": "MacroEventScanner",
        "stage_name": "全球宏观事件与突发舆情扫描",
        "thought": thought_text,
        "timestamp_ms": now_ms,
    }

    return {
        "macro_event_risk": macro_event_data,
        "thinking_steps": [thinking_step],
    }
