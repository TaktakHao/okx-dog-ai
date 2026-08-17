"""
OKX-Dog 容错自愈解析器与思维链提取单元测试
模块: okx-dog-ai/tests/test_parser.py
"""

import json
import pytest
from okx_dog_ai.parser import RobustJSONParser, ThoughtStreamExtractor
from okx_dog_ai.schemas import AIAnalysisResponse, SignalAction, MarketRegime


SAMPLE_STANDARD_JSON = """
{
  "analysis_id": "c1f7a8b2-3e4d-4a11-8f92-5b9e7c102a01",
  "symbol": "BTC-USDT-SWAP",
  "timestamp": 1755216000000,
  "market_regime": "TRENDING_UP",
  "timeframe_analysis": {
    "tf_15m": {"trend": "BULLISH", "key_indicators_summary": "多头排列", "support_level": 94380.0, "resistance_level": 95200.0},
    "tf_1h": {"trend": "BULLISH", "key_indicators_summary": "金叉放量", "support_level": 94200.0, "resistance_level": 96500.0},
    "tf_4h": {"trend": "BULLISH", "key_indicators_summary": "突破回踩", "support_level": 94200.0, "resistance_level": 97000.0},
    "tf_1d": {"trend": "BULLISH", "key_indicators_summary": "日线主升", "support_level": 91500.0, "resistance_level": 98500.0}
  },
  "derivatives_sentiment": {
    "funding_rate_bias": "MODERATE_POSITIVE",
    "open_interest_interpretation": "主力做多",
    "long_short_ratio_state": "多空比健康",
    "sentiment_score": 0.75
  },
  "signal": {
    "action": "BUY_LONG",
    "confidence": 0.88,
    "urgency": "MEDIUM"
  },
  "trade_plan": {
    "entry_range": [94500.0, 94700.0],
    "take_profit_levels": [
      {"price": 96800.0, "percentage": 0.5, "description": "TP1"},
      {"price": 98500.0, "percentage": 0.5, "description": "TP2"}
    ],
    "stop_loss_price": 93800.0,
    "risk_reward_ratio": 2.75,
    "suggested_leverage": 5,
    "order_type": "LIMIT"
  },
  "risk_assessment": {
    "key_risks": ["美股波动"],
    "invalidation_condition": "跌破93800",
    "max_holding_time_hours": 24.0
  },
  "reasoning_summary": "建议做多BTC",
  "reasoning_details": "4H突破回踩确认"
}
"""


def test_thought_stream_extractor():
    """测试流式思维链增量提取与状态机"""
    extractor = ThoughtStreamExtractor()

    chunks = [
        "<th",
        "ink>正在分析日线走势...",
        "4H突破94200阻力，回踩确认",
        "</think>\n```json\n",
        SAMPLE_STANDARD_JSON[:100],
        SAMPLE_STANDARD_JSON[100:],
        "\n```",
    ]

    all_events = []
    for c in chunks:
        events = extractor.feed_chunk(c)
        all_events.extend(events)

    thinking, content = extractor.get_final_result()

    assert "正在分析日线走势" in thinking
    assert "4H突破94200阻力" in thinking
    assert "analysis_id" in content
    assert "BTC-USDT-SWAP" in content

    # 验证事件流分类准确
    event_types = [e[0] for e in all_events]
    assert "think" in event_types
    assert "content" in event_types


def test_parser_layer1_native():
    """Layer 1: 原生标准 JSON 解析测试"""
    resp = RobustJSONParser.parse(SAMPLE_STANDARD_JSON, symbol="BTC-USDT-SWAP")
    assert isinstance(resp, AIAnalysisResponse)
    assert resp.symbol == "BTC-USDT-SWAP"
    assert resp.signal.action == SignalAction.BUY_LONG
    assert resp.signal.confidence == 0.88
    assert resp.trade_plan.stop_loss_price == 93800.0


def test_parser_layer2_markdown_blocks():
    """Layer 2: Markdown 代码块包裹解析测试"""
    md_text = f"这是为您生成的研判：\n```json\n{SAMPLE_STANDARD_JSON}\n```\n祝您交易愉快！"
    resp = RobustJSONParser.parse(md_text, symbol="BTC-USDT-SWAP")
    assert isinstance(resp, AIAnalysisResponse)
    assert resp.signal.action == SignalAction.BUY_LONG


def test_parser_layer3_outermost_json():
    """Layer 3: 提取最外层大括号解析测试"""
    wrapped_text = f"前置废话 123... {SAMPLE_STANDARD_JSON} 后置废话 456..."
    resp = RobustJSONParser.parse(wrapped_text, symbol="BTC-USDT-SWAP")
    assert isinstance(resp, AIAnalysisResponse)
    assert resp.signal.action == SignalAction.BUY_LONG


def test_parser_layer4_defect_repair():
    """Layer 4: 缺陷自愈修复测试 (单引号、缺引号、尾随逗号、注释、未闭合括号)"""
    defective_json = """
    // 这是一个带有注释的不规范 JSON
    {
      'analysis_id': "c1f7a8b2-3e4d-4a11-8f92-5b9e7c102a01",
      symbol: "ETH-USDT-SWAP",
      market_regime: "TRENDING_DOWN",
      timeframe_analysis: {
        tf_15m: { trend: "BEARISH", key_indicators_summary: "下行", support_level: 3300, resistance_level: 3500 },
        tf_1h: { trend: "BEARISH", key_indicators_summary: "下行", support_level: 3300, resistance_level: 3500 },
        tf_4h: { trend: "BEARISH", key_indicators_summary: "下行", support_level: 3300, resistance_level: 3500 },
        tf_1d: { trend: "BEARISH", key_indicators_summary: "下行", support_level: 3300, resistance_level: 3500 },
      },
      derivatives_sentiment: {
        funding_rate_bias: "EXTREME_POSITIVE",
        open_interest_interpretation: "过热",
        long_short_ratio_state: "追多严重",
        sentiment_score: -0.7,
      },
      signal: {
        action: "SELL_SHORT",
        confidence: 0.85,
        urgency: "HIGH",
      },
      trade_plan: {
        entry_range: [3480.0, 3495.0],
        take_profit_levels: [{ price: 3390.0, percentage: 1.0, description: "TP1" }],
        stop_loss_price: 3525.0,
        risk_reward_ratio: 2.3,
        suggested_leverage: 3,
        order_type: "LIMIT",
      },
      risk_assessment: {
        key_risks: ["假突破插针", ],
        invalidation_condition: "站稳3525",
        max_holding_time_hours: 12.0
      },
      reasoning_summary: "建议做空ETH",
      reasoning_details: "顶背离形成"
    """  # 故意缺少最后一个闭合大括号 }

    resp = RobustJSONParser.parse(defective_json, symbol="ETH-USDT-SWAP")
    assert isinstance(resp, AIAnalysisResponse)
    assert resp.symbol == "ETH-USDT-SWAP"
    assert resp.signal.action == SignalAction.SELL_SHORT
    assert resp.signal.confidence == 0.85
    assert resp.trade_plan.suggested_leverage == 3


def test_parser_layer5_missing_fields_patch_and_safe_fallback():
    """Layer 5: 字段智能补齐与完全损毁时的安全 HOLD_WAIT 熔断兜底"""
    # 1. 简略残缺字典智能补齐
    incomplete_json = """
    {
      "symbol": "SOL-USDT-SWAP",
      "signal": {"action": "BUY_LONG", "confidence": 0.75}
    }
    """
    resp_patched = RobustJSONParser.parse(incomplete_json, symbol="SOL-USDT-SWAP")
    assert isinstance(resp_patched, AIAnalysisResponse)
    assert resp_patched.symbol == "SOL-USDT-SWAP"
    assert resp_patched.signal.action == SignalAction.BUY_LONG
    assert resp_patched.timeframe_analysis.tf_15m is not None
    assert resp_patched.trade_plan is not None

    # 2. 完全乱码损毁 -> 触发绝对安全熔断兜底
    garbage_text = "这是一段完全无法识别的损坏文本，既不是 JSON 也没有结构。"
    fallback = RobustJSONParser.parse(garbage_text, symbol="BTC-USDT-SWAP")
    assert isinstance(fallback, AIAnalysisResponse)
    assert fallback.signal.action == SignalAction.HOLD_WAIT
    assert fallback.signal.confidence == 0.0
    assert fallback.trade_plan.stop_loss_price == 0.0
    assert "安全熔断" in fallback.reasoning_summary
