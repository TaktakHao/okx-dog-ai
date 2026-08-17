"""
OKX-Dog Prompt 组装器与动态 Token 裁剪单元测试
模块: okx-dog-ai/tests/test_prompt_builder.py
"""

import pytest
from okx_dog_ai.prompt_builder import MarketPromptBuilder, TokenEstimator
from okx_dog_ai.schemas import (
    MarketContextSnapshot,
    MultiPeriodIndicators,
    SinglePeriodIndicators,
    DerivativesMetrics,
    PositionSnapshot,
    HardRiskLimits,
)


def create_sample_snapshot():
    """构造完整测试行情快照"""
    ind_15m = SinglePeriodIndicators(
        timeframe="15m",
        ema_20=94500.0,
        ema_50=94200.0,
        ema_200=93800.0,
        macd_dif=120.0,
        macd_dea=95.0,
        macd_hist=50.0,
        rsi_14=58.5,
        bb_upper=95200.0,
        bb_middle=94500.0,
        bb_lower=93800.0,
        bb_width_pct=1.48,
        atr_14=250.0,
    )
    ind_1h = SinglePeriodIndicators(
        timeframe="1h",
        ema_20=94200.0,
        ema_50=93800.0,
        ema_200=92000.0,
        macd_dif=210.0,
        macd_dea=180.0,
        macd_hist=60.0,
        rsi_14=62.0,
        bb_upper=95500.0,
        bb_middle=94000.0,
        bb_lower=92500.0,
        bb_width_pct=3.19,
        atr_14=420.0,
    )
    ind_4h = SinglePeriodIndicators(
        timeframe="4h",
        ema_20=93500.0,
        ema_50=92000.0,
        ema_200=89000.0,
        macd_dif=450.0,
        macd_dea=400.0,
        macd_hist=100.0,
        rsi_14=65.0,
        bb_upper=96000.0,
        bb_middle=93000.0,
        bb_lower=90000.0,
        bb_width_pct=6.45,
        atr_14=850.0,
    )
    ind_1d = SinglePeriodIndicators(
        timeframe="1d",
        ema_20=91000.0,
        ema_50=88000.0,
        ema_200=82000.0,
        macd_dif=900.0,
        macd_dea=800.0,
        macd_hist=200.0,
        rsi_14=68.0,
        bb_upper=98000.0,
        bb_middle=90000.0,
        bb_lower=82000.0,
        bb_width_pct=17.78,
        atr_14=1600.0,
    )

    multi = MultiPeriodIndicators(
        symbol="BTC-USDT-SWAP",
        timestamp=1755216000000,
        indicators={"15m": ind_15m, "1h": ind_1h, "4h": ind_4h, "1d": ind_1d},
    )

    deriv = DerivativesMetrics(
        symbol="BTC-USDT-SWAP",
        funding_rate=0.0001,
        funding_countdown_min=180,
        open_interest=50000.0,
        open_interest_usdt=50000.0 * 94650.0,
        oi_change_24h_pct=5.5,
        long_short_ratio=1.15,
        top_trader_ratio=1.20,
        sentiment_score=0.65,
    )

    pos = PositionSnapshot(
        symbol="BTC-USDT-SWAP",
        side="long",
        leverage=3,
        contracts=10.0,
        notional_usd=946500.0,
        entry_price=94000.0,
        mark_price=94650.0,
        unrealized_pnl=6500.0,
        pnl_percentage=6.91,
    )

    return MarketContextSnapshot(
        symbol="BTC-USDT-SWAP",
        current_price=94650.0,
        change_24h_pct=2.35,
        multi_indicators=multi,
        derivatives=deriv,
        active_position=pos,
        account_balance_usdt=5000.0,
        risk_limits=HardRiskLimits(max_order_usdt=500.0, max_daily_loss_usdt=200.0, max_leverage=5),
        imbalance_ratio=1.45,
        orderbook_bids_top5=[[94640.0, 5.0], [94630.0, 10.0], [94620.0, 8.0]],
        orderbook_asks_top5=[[94660.0, 4.0], [94670.0, 6.0], [94680.0, 7.0]],
    )


def test_token_estimator():
    """测试 Token 快速估算器"""
    text_en = "Hello world, this is a test for OpenAI GPT-4."
    tokens_en = TokenEstimator.estimate_tokens(text_en)
    assert 5 <= tokens_en <= 20

    text_cn = "OKX-Dog 个人量化副驾驶系统，支持多周期指标与硬风控拦截。"
    tokens_cn = TokenEstimator.estimate_tokens(text_cn)
    assert 15 <= tokens_cn <= 45


def test_prompt_builder_system_prompt():
    """测试 System Prompt 组装与约束注入"""
    builder = MarketPromptBuilder()
    sys_prompt = builder.build_system_prompt(include_few_shot=True, custom_constraints=["禁止做空 BTC"])
    
    assert "OKX-Dog 个人量化智能副驾驶" in sys_prompt
    assert "多周期共振原则" in sys_prompt
    assert "禁止做空 BTC" in sys_prompt
    assert "Few-Shot 优质决策参考样本" in sys_prompt
    assert "OKXDogAIAnalysisResponse" in sys_prompt


def test_prompt_builder_budget_pruning():
    """测试 P0~P3 自底向上 Token 动态裁剪算法"""
    builder = MarketPromptBuilder()
    snapshot = create_sample_snapshot()

    # 1. 充足预算 (4000 tokens) -> 完整保留 P0 + P1 + P2 + P3
    full_prompt = builder.build_user_prompt(snapshot, scenario="standard", max_tokens=4000)
    assert "<p0_critical_context>" in full_prompt
    assert "<p1_core_indicators>" in full_prompt
    assert "<p2_macro_and_sentiment>" in full_prompt
    assert "<p3_microstructure_details>" in full_prompt

    # 2. 紧凑预算 (250 tokens) -> 裁剪 P3 和 P2，压缩 P1，但强制保留 P0
    pruned_prompt = builder.build_user_prompt(snapshot, scenario="standard", max_tokens=250)
    assert "<p0_critical_context>" in pruned_prompt
    assert "<p3_microstructure_details>" not in pruned_prompt
    # 验证 P0 核心字段永不丢失
    assert "BTC-USDT-SWAP" in pruned_prompt
    assert "94650.0" in pruned_prompt
    assert "4H主趋势状态" in pruned_prompt


def test_prompt_builder_scenarios():
    """测试不同场景的 User Prompt 组装"""
    builder = MarketPromptBuilder()
    snapshot = create_sample_snapshot()

    # 常规场景
    std_prompt = builder.build_user_prompt(snapshot, scenario="standard")
    assert "多周期技术指标与衍生品数据进行深度共振推导" in std_prompt

    # 异动场景
    snapshot.is_anomaly_mode = True
    snapshot.anomaly_desc = "短线买单扫盘放量突破"
    anomaly_prompt = builder.build_user_prompt(snapshot, scenario="anomaly")
    assert "短线买单扫盘放量突破" in anomaly_prompt
    assert "判定本次异动是真实放量突破还是主力假突破" in anomaly_prompt

    # 持仓管理场景
    pos_prompt = builder.build_user_prompt(snapshot, scenario="position_manage")
    assert "针对当前持仓状态提供专业量化调仓决策" in pos_prompt


def test_build_messages():
    """测试 build_messages 导出规范的 messages 数组"""
    builder = MarketPromptBuilder(default_token_budget=3000)
    snapshot = create_sample_snapshot()

    messages = builder.build_messages(snapshot, scenario="standard", include_few_shot=False)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "BTC-USDT-SWAP" in messages[1]["content"]
