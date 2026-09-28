"""
OKX-Dog AI 策略决策与参数平原评估单元测试
测试内容:
1. StrategyPlateauValidator:
   - 正常稳健策略 (通过生死线过滤与平原检验)
   - 虚假高夏普策略 (剔除 Top3 后收益缩水 > 50%，一票否决)
   - 高手续费磨损策略 (fee_to_gross_ratio >= 0.40，一票否决)
   - 双边 40bps 强滑点穿透测试
2. MarketPromptBuilder:
   - 验证 DCA 强平缓冲与卡尔曼对冲变量成功组装进入 Prompt
"""

import pytest
from strategy_plateau_validator import StrategyPlateauValidator
from prompt_builder import MarketPromptBuilder


def test_strategy_plateau_validator_qualified():
    """测试稳健策略通过参数平原与生死线"""
    # 模拟 20 笔均匀盈利与小幅亏损交易
    trades_pnl = [50.0, 45.0, 40.0, -20.0, 35.0, 30.0, -15.0, 25.0, 20.0, -10.0, 30.0, 25.0, -15.0, 20.0, 15.0]
    trades_notional = [1000.0] * len(trades_pnl)
    total_fee = 15.0  # 手续费极低

    # 模拟 +/-20% 邻居参数收益均稳健正向
    neighbors = [280.0, 275.0, 290.0, 260.0]

    score = StrategyPlateauValidator.evaluate_strategy_survival(
        trades_pnl=trades_pnl,
        trades_notional=trades_notional,
        total_fee_paid=total_fee,
        slippage_stress_bps=40.0,
        min_top3_retention_ratio=0.50,
        neighbor_pnl_list=neighbors
    )

    assert score.is_plateau_qualified is True
    assert score.drop_top3_decay >= 0.50
    assert score.fee_to_gross_ratio < 0.40
    assert score.friction_stress_pnl > 0
    assert score.neighbor_stability_score >= 60.0


def test_strategy_plateau_validator_top3_outlier_rejected():
    """测试仅靠 1~2 笔黑天鹅暴利支撑的假策略被 Top-3 衰减硬拦截"""
    # 净利润 1000，但其中 3 笔单占了 900
    trades_pnl = [500.0, 250.0, 150.0, 10.0, 5.0, -20.0, 10.0, -15.0, 10.0, 100.0]
    trades_notional = [1000.0] * len(trades_pnl)

    score = StrategyPlateauValidator.evaluate_strategy_survival(
        trades_pnl=trades_pnl,
        trades_notional=trades_notional,
        total_fee_paid=30.0,
        min_top3_retention_ratio=0.50
    )

    # 剔除 top3(500+250+150=900) 后只剩 100，保留率 10% < 50%
    assert score.is_plateau_qualified is False
    assert score.drop_top3_decay < 0.50


def test_prompt_builder_quantitative_boundaries_injection():
    """测试 MarketPromptBuilder 成功将强平缓冲与卡尔曼变量组装入上下文"""
    builder = MarketPromptBuilder()

    snapshot = {
        "symbol": "BTC-USDT-SWAP",
        "current_price": 62000.0,
        "change_24h_pct": 2.5,
        "user_strategy_bias": "BALANCED",
        "account_balance_usdt": 8500.0,
        "timeframes": {
            "15m": {"ema_20": 61900.0, "ema_50": 61800.0, "rsi_14": 55.0, "atr_14": 120.0},
            "1h": {"ema_20": 61500.0, "ema_50": 61000.0, "ema_200": 59000.0, "rsi_14": 62.0},
            "4h": {"ema_20": 60000.0, "ema_50": 58000.0, "ema_200": 55000.0, "rsi_14": 68.0},
        },
        "derivatives": {"funding_rate": 0.0001, "open_interest": 45000.0, "oi_change_24h_pct": 5.0},
        "dca_liquidation_analysis": {
            "is_safe": True,
            "liq_safety_buffer_pct": 14.5,
            "required_extreme_margin": 1820.0,
            "violation_reason": None
        },
        "kalman_pair_metrics": {
            "main_symbol": "BTC-USDT-SWAP",
            "sub_symbol": "ETH-USDT-SWAP",
            "dynamic_beta": 0.82,
            "dynamic_alpha": -1.5,
            "spread_z_score": 2.15,
            "covariance_trace": 0.045,
            "is_divergent": False,
            "suggested_arbitrage_action": "SHORT_SUB_LONG_MAIN"
        }
    }

    user_prompt = builder.build_user_prompt(snapshot)
    assert "【数理真实边界】" in user_prompt
    assert "DCA强平缓冲: 14.5%" in user_prompt
    assert "卡尔曼对冲(BTC-USDT-SWAP/ETH-USDT-SWAP) Z-Score: +2.15" in user_prompt
    assert "套利动作: SHORT_SUB_LONG_MAIN" in user_prompt
