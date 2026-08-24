"""
OKX-Dog LangGraph 机构级量化智能体决策大脑核心测试套件
测试模块: okx-dog-ai/tests/test_quant_agent.py
"""

from __future__ import annotations

import pytest
import asyncio
from typing import Any, Dict

try:
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
        AgentRoleRegistry,
        BaseSpecialist,
        register_specialist,
        QuantTraderAgentRunner,
        create_quant_trader_graph,
        calculate_risk_reward_ratio,
        verify_hard_risk_compliance,
        derive_dynamic_atr_stops,
    )
    from okx_dog_ai.agent.tools import (
        calculate_orderbook_imbalance,
        evaluate_onchain_flow,
        calculate_kelly_position_size,
        evaluate_macro_event_risk,
        analyze_orderbook_liquidity,
    )
    from okx_dog_ai.agent.nodes.macro_scanner import macro_trend_scan_node
    from okx_dog_ai.agent.nodes.onchain_analyst import onchain_analyst_node
    from okx_dog_ai.agent.nodes.quant_modeler import quant_modeler_node
    from okx_dog_ai.agent.nodes.derivatives_checker import derivatives_sentiment_node
    from okx_dog_ai.agent.nodes.macro_event_scanner import macro_event_scanner_node
    from okx_dog_ai.agent.nodes.microstructure_analyst import microstructure_analyst_node
    from okx_dog_ai.agent.nodes.adversarial_debater import adversarial_debate_node
    from okx_dog_ai.agent.nodes.strategy_planner import strategy_planning_node
    from okx_dog_ai.agent.nodes.risk_critic import risk_critic_node
    from okx_dog_ai.agent.nodes.formatter import response_formatter_node
except (ImportError, ValueError):
    from schemas import (
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
    from agent import (
        AgentRoleRegistry,
        BaseSpecialist,
        register_specialist,
        QuantTraderAgentRunner,
        create_quant_trader_graph,
        calculate_risk_reward_ratio,
        verify_hard_risk_compliance,
        derive_dynamic_atr_stops,
    )
    from agent.tools import (
        calculate_orderbook_imbalance,
        evaluate_onchain_flow,
        calculate_kelly_position_size,
        evaluate_macro_event_risk,
        analyze_orderbook_liquidity,
    )
    from agent.nodes.macro_scanner import macro_trend_scan_node
    from agent.nodes.onchain_analyst import onchain_analyst_node
    from agent.nodes.quant_modeler import quant_modeler_node
    from agent.nodes.derivatives_checker import derivatives_sentiment_node
    from agent.nodes.macro_event_scanner import macro_event_scanner_node
    from agent.nodes.microstructure_analyst import microstructure_analyst_node
    from agent.nodes.adversarial_debater import adversarial_debate_node
    from agent.nodes.strategy_planner import strategy_planning_node
    from agent.nodes.risk_critic import risk_critic_node
    from agent.nodes.formatter import response_formatter_node


def _create_mock_snapshot(current_price: float = 65000.0, is_bullish: bool = True) -> MarketContextSnapshot:
    """创建标准化的行情快照测试夹具 (含链上、盘口、宏观日历与微观特征)"""
    if is_bullish:
        ema20, ema50, ema200 = 65200.0, 64800.0, 63000.0
        rsi = 62.0
        macd_hist = 25.0
        fr = 0.00015
        bids = [[64990.0, 15.0], [64980.0, 25.0], [64950.0, 40.0], [64900.0, 50.0], [64850.0, 60.0]]
        asks = [[65010.0, 8.0], [65020.0, 12.0], [65050.0, 15.0], [65100.0, 20.0], [65150.0, 25.0]]
    else:
        ema20, ema50, ema200 = 64500.0, 65000.0, 66500.0
        rsi = 38.0
        macd_hist = -25.0
        fr = -0.0002
        bids = [[64990.0, 5.0], [64980.0, 8.0], [64950.0, 10.0], [64900.0, 15.0], [64850.0, 20.0]]
        asks = [[65010.0, 20.0], [65020.0, 30.0], [65050.0, 45.0], [65100.0, 55.0], [65150.0, 70.0]]

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
        orderbook_bids_top5=bids,
        orderbook_asks_top5=asks,
        imbalance_ratio=2.4 if is_bullish else 0.4,
    )


def test_quant_tools_risk_reward():
    """测试盈亏比工具与方向性逻辑校验"""
    rr, err = calculate_risk_reward_ratio(
        action="BUY_LONG",
        entry_range=[65000.0, 65200.0],
        stop_loss_price=64000.0,
        take_profit_levels=[{"price": 67000.0, "percentage": 0.5}],
    )
    assert err is None
    assert rr >= 1.5


def test_quant_tools_orderbook_imbalance():
    """测试订单簿买卖失衡比计算"""
    bids = [[65000.0, 10.0], [64990.0, 20.0]]
    asks = [[65010.0, 5.0], [65020.0, 5.0]]
    ratio, desc = calculate_orderbook_imbalance(bids, asks, depth_levels=2)
    assert ratio == 3.0
    assert "买盘" in desc


def test_quant_tools_macro_event_and_liquidity():
    """测试宏观事件与微观流动性评估工具"""
    # 1. 宏观日历事件前 15 分钟触发强制锁仓
    risk_level, msg, is_locked = evaluate_macro_event_risk(
        minutes_to_high_impact_event=15,
        event_title="美联储FOMC利率决议",
    )
    assert risk_level == "CRITICAL"
    assert is_locked is True
    assert "美联储FOMC" in msg

    # 2. 订单簿微观流动性与冲击成本
    bids = [[65000.0, 10.0], [64990.0, 20.0]]
    asks = [[65001.0, 10.0], [65010.0, 20.0]]
    liq = analyze_orderbook_liquidity(bids, asks, current_price=65000.0, standard_order_usdt=500.0)
    assert "spread_bps" in liq
    assert "recommended_execution_mode" in liq
    assert liq["recommended_execution_mode"] in ["LIMIT", "POST_ONLY", "TWAP"]


def test_role_registry_extensibility():
    """测试可插拔角色注册中心与扩展性 (模拟新增第三方分析专家)"""
    initial_specs = AgentRoleRegistry.get_specialists_by_layer("perception")
    assert len(initial_specs) >= 6

    # 动态注册一个自定义专家
    @register_specialist
    class MockTwitterSentimentSpecialist(BaseSpecialist):
        name = "mock_twitter_specialist"
        stage_name = "推特社媒舆情分析专家"
        layer = "perception"
        description = "实时分析推特热度与情绪偏向"

        async def analyze(self, state):
            return {
                "twitter_sentiment": {"score": 0.85, "viral_topic": "BTC_ETF"},
                "thinking_steps": [{
                    "node": self.name,
                    "stage_name": self.stage_name,
                    "thought": "推特社媒多头情绪异常高涨",
                    "timestamp_ms": 1700000000000,
                }],
            }

    updated_specs = AgentRoleRegistry.get_specialists_by_layer("perception")
    assert any(s.name == "mock_twitter_specialist" for s in updated_specs)

    # 验证图编译器能直接动态包含该新角色
    graph = create_quant_trader_graph()
    assert graph is not None


@pytest.mark.asyncio
async def test_macro_event_scanner_node():
    """测试宏观事件扫描专家节点"""
    snap = _create_mock_snapshot(current_price=65000.0, is_bullish=True)
    state = {
        "symbol": "BTC-USDT-SWAP",
        "market_snapshot": snap.model_dump(),
        "thinking_steps": [],
    }
    out = await macro_event_scanner_node(state)
    assert "macro_event_risk" in out
    assert "event_risk_level" in out["macro_event_risk"]
    assert len(out["thinking_steps"]) == 1


@pytest.mark.asyncio
async def test_microstructure_analyst_node():
    """测试微观流动性分析专家节点"""
    snap = _create_mock_snapshot(current_price=65000.0, is_bullish=True)
    state = {
        "symbol": "BTC-USDT-SWAP",
        "current_price": 65000.0,
        "market_snapshot": snap.model_dump(),
        "thinking_steps": [],
    }
    out = await microstructure_analyst_node(state)
    assert "microstructure_data" in out
    assert "recommended_execution_mode" in out["microstructure_data"]
    assert len(out["thinking_steps"]) == 1


@pytest.mark.asyncio
async def test_adversarial_debater_node():
    """测试红蓝对抗博弈辩论节点"""
    snap = _create_mock_snapshot(current_price=65000.0, is_bullish=True)
    state = {
        "symbol": "BTC-USDT-SWAP",
        "current_price": 65000.0,
        "market_regime": "TRENDING_UP",
        "onchain_analysis": {"composite_score": 0.4, "flow_bias": "ACCUMULATING"},
        "quant_features": {"orderbook_imbalance_ratio": 1.8},
        "derivatives_sentiment": {"sentiment_score": 0.3, "funding_rate_bias": "MODERATE_POSITIVE"},
        "macro_event_risk": {"event_risk_level": "LOW"},
        "microstructure_data": {"spread_bps": 1.2},
        "thinking_steps": [],
    }
    out = await adversarial_debate_node(state)
    assert "bull_opinion" in out
    assert "bear_opinion" in out
    assert out["bull_opinion"]["stance"] in ["BULLISH", "NEUTRAL"]
    assert out["bear_opinion"]["stance"] in ["BEARISH", "NEUTRAL"]
    assert len(out["thinking_steps"]) == 1


@pytest.mark.asyncio
async def test_risk_critic_reflection_and_correction():
    """测试当方案存在缺陷时，RiskCritic 触发反思回退，StrategyPlanner 成功修复"""
    state = {
        "symbol": "BTC-USDT-SWAP",
        "current_price": 65000.0,
        "market_regime": "TRENDING_UP",
        "bull_opinion": {"confidence": 0.8, "stance": "BULLISH"},
        "bear_opinion": {"confidence": 0.4, "stance": "NEUTRAL"},
        "signal": {"action": "BUY_LONG", "confidence": 0.8, "urgency": "MEDIUM"},
        "trade_plan": {
            "entry_range": [65000.0, 65000.0],
            "stop_loss_price": 64000.0,
            "take_profit_levels": [{"price": 65500.0, "percentage": 0.5}],
            "suggested_leverage": 3,
        },
        "risk_limits": {"max_leverage": 5, "max_order_usdt": 500.0, "max_daily_loss_usdt": 200.0},
        "account_balance_usdt": 1000.0,
        "critique_count": 0,
        "thinking_steps": [],
    }

    # 1. 触发 Critic 审查 (缺陷拦截)
    critic_out = await risk_critic_node(state)
    assert critic_out["risk_passed"] is False
    assert critic_out["critique_count"] == 1

    # 2. 将批评意见送入 StrategyPlanner 反思自适应修复
    state.update(critic_out)
    plan_out = await strategy_planning_node(state)
    state.update(plan_out)

    # 3. 再次由 Critic 审查 -> 此时应自适应修复并放行
    second_critic_out = await risk_critic_node(state)
    assert second_critic_out["risk_passed"] is True
    assert second_critic_out["risk_critique"] is None


@pytest.mark.asyncio
async def test_quant_trader_agent_runner_end_to_end():
    """测试机构级分层 LangGraph 全链路端到端运行"""
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
    """测试机构级分层 LangGraph SSE 流式输出"""
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
