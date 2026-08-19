"""
OKX-Dog LangGraph 资深量化交易员 Agent 核心测试套件
测试模块: okx-dog-ai/tests/test_quant_agent.py
"""

from __future__ import annotations

import pytest
import asyncio
from typing import Any, Dict

from okx_dog_ai.schemas import (
    AIAnalysisResponse,
    DerivativesMetrics,
    FundingRateBias,
    HardRiskLimits,
    MarketContextSnapshot,
    MarketRegime,
    MultiPeriodIndicators,
    SignalAction,
    SinglePeriodIndicators,
)
from okx_dog_ai.agent import (
    QuantTraderAgentRunner,
    create_quant_trader_graph,
    calculate_risk_reward_ratio,
    verify_hard_risk_compliance,
    derive_dynamic_atr_stops,
)
from okx_dog_ai.agent.nodes.macro_scanner import macro_trend_scan_node
from okx_dog_ai.agent.nodes.derivatives_checker import derivatives_sentiment_node
from okx_dog_ai.agent.nodes.strategy_planner import strategy_planning_node
from okx_dog_ai.agent.nodes.risk_critic import risk_critic_node
from okx_dog_ai.agent.nodes.formatter import response_formatter_node


def _create_mock_snapshot(current_price: float = 65000.0, is_bullish: bool = True) -> MarketContextSnapshot:
    """创建标准化的行情快照测试夹具"""
    if is_bullish:
        ema20, ema50, ema200 = 65200.0, 64800.0, 63000.0
        rsi = 62.0
        macd_hist = 25.0
        fr = 0.00015
    else:
        ema20, ema50, ema200 = 64500.0, 65000.0, 66500.0
        rsi = 38.0
        macd_hist = -25.0
        fr = -0.0002

    tf_ind = SinglePeriodIndicators(
        timeframe="1h",
        ema_20=ema20,
        ema_50=ema50,
        ema_200=ema200,
        macd_dif=10.0,
        macd_dea=5.0,
        macd_hist=macd_hist,
        rsi_14=rsi,
        bb_upper=current_price * 1.03,
        bb_middle=current_price,
        bb_lower=current_price * 0.97,
        bb_width_pct=3.0,
        atr_14=600.0,
    )

    multi = MultiPeriodIndicators(
        symbol="BTC-USDT-SWAP",
        timestamp=1700000000000,
        indicators={
            "15m": tf_ind,
            "1h": tf_ind,
            "4h": tf_ind,
            "1d": tf_ind,
        }
    )

    deriv = DerivativesMetrics(
        symbol="BTC-USDT-SWAP",
        funding_rate=fr,
        open_interest=50000.0,
        oi_change_24h_pct=5.5,
        long_short_ratio=1.2,
        top_trader_ratio=1.1,
        sentiment_score=0.3 if is_bullish else -0.3,
        timestamp=1700000000000,
    )

    return MarketContextSnapshot(
        symbol="BTC-USDT-SWAP",
        current_price=current_price,
        change_24h_pct=3.2 if is_bullish else -2.5,
        multi_indicators=multi,
        derivatives=deriv,
        account_balance_usdt=2000.0,
        risk_limits=HardRiskLimits(
            max_order_usdt=500.0,
            max_daily_loss_usdt=200.0,
            max_leverage=5,
        ),
    )


def test_quant_tools_risk_reward():
    """测试盈亏比工具与方向性逻辑校验"""
    # 1. 正常做多，R:R >= 1.5
    rr, err = calculate_risk_reward_ratio(
        action="BUY_LONG",
        entry_range=[65000.0, 65200.0],
        stop_loss_price=64000.0,
        take_profit_levels=[{"price": 67000.0, "percentage": 0.5}],
    )
    assert err is None
    assert rr >= 1.5

    # 2. 做多但止损在入场价上方 (逻辑错误)
    rr_err1, err1 = calculate_risk_reward_ratio(
        action="BUY_LONG",
        entry_range=[65000.0, 65200.0],
        stop_loss_price=66000.0,
        take_profit_levels=[{"price": 67000.0}],
    )
    assert err1 is not None

    # 3. 正常做空，R:R >= 1.5
    rr_short, err_short = calculate_risk_reward_ratio(
        action="SELL_SHORT",
        entry_range=[65000.0, 65200.0],
        stop_loss_price=66100.0,
        take_profit_levels=[{"price": 63200.0, "percentage": 0.5}],
    )
    assert err_short is None
    assert rr_short >= 1.5


def test_quant_tools_hard_risk_compliance():
    """测试硬风控边界拦截"""
    risk_limits = {"max_leverage": 5, "max_order_usdt": 500.0, "max_daily_loss_usdt": 200.0}

    # 1. 杠杆违规
    violations = verify_hard_risk_compliance(
        action="BUY_LONG",
        entry_price=65000.0,
        stop_loss_price=64000.0,
        suggested_leverage=10,  # 超出 5x
        account_balance_usdt=1000.0,
        risk_limits=risk_limits,
    )
    assert any("杠杆" in v for v in violations)

    # 2. 完全合规
    valid_violations = verify_hard_risk_compliance(
        action="BUY_LONG",
        entry_price=65000.0,
        stop_loss_price=64000.0,
        suggested_leverage=3,
        account_balance_usdt=1000.0,
        risk_limits=risk_limits,
    )
    assert len(valid_violations) == 0


@pytest.mark.asyncio
async def test_macro_scanner_node():
    """测试宏观多周期共振扫描节点"""
    snap = _create_mock_snapshot(current_price=65000.0, is_bullish=True)
    state = {
        "symbol": "BTC-USDT-SWAP",
        "current_price": 65000.0,
        "market_snapshot": snap.model_dump(),
        "thinking_steps": [],
    }
    out = await macro_trend_scan_node(state)
    assert out["market_regime"] in ["TRENDING_UP", "RANGING", "VOLATILE_BREAKOUT"]
    assert "tf_1h" in out["timeframe_analysis"]
    assert len(out["thinking_steps"]) == 1


@pytest.mark.asyncio
async def test_derivatives_checker_node():
    """测试衍生品与微观流动性检验节点"""
    snap = _create_mock_snapshot(current_price=65000.0, is_bullish=True)
    state = {
        "market_snapshot": snap.model_dump(),
        "thinking_steps": [],
    }
    out = await derivatives_sentiment_node(state)
    assert "derivatives_sentiment" in out
    assert out["derivatives_sentiment"]["funding_rate_bias"] in ["MODERATE_POSITIVE", "NEUTRAL", "EXTREME_POSITIVE"]
    assert len(out["thinking_steps"]) == 1


@pytest.mark.asyncio
async def test_risk_critic_reflection_and_correction():
    """测试当方案存在缺陷时，RiskCritic 触发反思回退，StrategyPlanner 成功修复"""
    # 模拟一个盈亏比不足 1.5 的策略草案
    state = {
        "symbol": "BTC-USDT-SWAP",
        "current_price": 65000.0,
        "market_regime": "TRENDING_UP",
        "signal": {"action": "BUY_LONG", "confidence": 0.8, "urgency": "MEDIUM"},
        "trade_plan": {
            "entry_range": [65000.0, 65000.0],
            "stop_loss_price": 64000.0,  # risk = 1000
            "take_profit_levels": [{"price": 65500.0, "percentage": 0.5}],  # reward = 500 -> R:R = 0.5 (不合格!)
            "suggested_leverage": 3,
        },
        "risk_limits": {"max_leverage": 5, "max_order_usdt": 500.0, "max_daily_loss_usdt": 200.0},
        "account_balance_usdt": 1000.0,
        "critique_count": 0,
        "thinking_steps": [],
    }

    # 1. 触发 Critic 审查
    critic_out = await risk_critic_node(state)
    assert critic_out["risk_passed"] is False
    assert critic_out["critique_count"] == 1
    assert "盈亏比不足 1.5" in critic_out["risk_critique"]

    # 2. 将批评意见送入 StrategyPlanner
    state.update(critic_out)
    plan_out = await strategy_planning_node(state)
    state.update(plan_out)

    # 3. 再次由 Critic 审查 -> 此时应自适应修复并放行
    second_critic_out = await risk_critic_node(state)
    assert second_critic_out["risk_passed"] is True
    assert second_critic_out["risk_critique"] is None


@pytest.mark.asyncio
async def test_quant_trader_agent_runner_end_to_end():
    """测试 LangGraph QuantTraderAgentRunner 全链路端到端运行"""
    snap = _create_mock_snapshot(current_price=65000.0, is_bullish=True)
    runner = QuantTraderAgentRunner()
    
    response = await runner.run(snap)
    assert isinstance(response, AIAnalysisResponse)
    assert response.symbol == "BTC-USDT-SWAP"
    assert response.signal.action in [SignalAction.BUY_LONG, SignalAction.SELL_SHORT, SignalAction.HOLD_WAIT]
    assert response.trade_plan.risk_reward_ratio >= 1.5
    assert response.thinking_process is not None
    assert "<think>" in response.thinking_process
    assert len(response.reasoning_summary) > 0


@pytest.mark.asyncio
async def test_quant_trader_agent_runner_stream():
    """测试 LangGraph QuantTraderAgentRunner SSE 流式输出"""
    snap = _create_mock_snapshot(current_price=65000.0, is_bullish=True)
    runner = QuantTraderAgentRunner()

    events = []
    async for chunk in runner.run_stream(snap):
        events.append(chunk)

    event_types = [e.event for e in events]
    assert "start" in event_types
    assert "think" in event_types
    assert "done" in event_types

    done_event = next(e for e in events if e.event == "done")
    assert isinstance(done_event.structured_output, AIAnalysisResponse)
