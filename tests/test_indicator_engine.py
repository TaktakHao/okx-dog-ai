"""
OKX-Dog 量化指标引擎单元测试
模块: okx-dog-ai/tests/test_indicator_engine.py
"""

import math
import pytest
from okx_dog_ai.indicator_engine import IndicatorEngine, KlineBar
from okx_dog_ai.schemas import FundingRateBias


def generate_synthetic_klines(count: int = 250, start_price: float = 90000.0, trend: float = 10.0):
    """生成测试用合成 K 线序列"""
    klines = []
    base_ts = 1750000000000
    price = start_price
    for i in range(count):
        ts = base_ts + i * 900000  # 15m 间隔
        o = price
        h = price + 50.0 + (i % 5) * 5.0
        l = price - 40.0 - (i % 3) * 5.0
        c = price + trend + (math.sin(i / 5.0) * 30.0)
        v = 100.0 + (i % 10) * 10.0
        vc = v * c
        klines.append({
            "timestamp": ts,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "vol_currency": vc,
            "is_closed": True,
        })
        price = c
    return klines


def test_indicator_engine_warmup_and_calculation():
    """测试指标引擎历史全量预热与各项指标输出"""
    engine = IndicatorEngine(symbol="BTC-USDT-SWAP")
    klines = generate_synthetic_klines(count=220, start_price=90000.0, trend=15.0)

    # 1. 预热 15m 历史
    engine.feed_kline_history("15m", klines)

    # 2. 获取多周期指标
    multi = engine.get_multi_period_indicators()
    assert "15m" in multi.indicators
    ind_15m = multi.indicators["15m"]

    # 3. 验证 EMA
    assert ind_15m.ema_20 > 90000.0
    assert ind_15m.ema_50 > 90000.0
    assert ind_15m.ema_200 is not None
    assert ind_15m.ema_200 > 90000.0
    # 上升趋势下短期 EMA 通常大于长期 EMA
    assert ind_15m.ema_20 > ind_15m.ema_50

    # 4. 验证 MACD
    assert ind_15m.macd_dif is not None
    assert ind_15m.macd_dea is not None
    assert ind_15m.macd_hist is not None

    # 5. 验证 RSI 在合理区间 [0, 100]
    assert 0.0 <= ind_15m.rsi_14 <= 100.0

    # 6. 验证布林带 上轨 > 中轨 > 下轨
    assert ind_15m.bb_upper > ind_15m.bb_middle > ind_15m.bb_lower
    assert ind_15m.bb_width_pct > 0.0

    # 7. 验证 ATR 为正数
    assert ind_15m.atr_14 > 0.0


def test_indicator_engine_incremental_o1():
    """测试 O(1) 增量更新（未收线实时 Tick 与收线确认）"""
    engine = IndicatorEngine(symbol="ETH-USDT-SWAP")
    klines = generate_synthetic_klines(count=50, start_price=3000.0, trend=2.0)
    engine.feed_kline_history("1h", klines)

    last_closed_ema20 = engine.states["1h"].ema_20
    assert last_closed_ema20 is not None

    # 模拟未收线的实时 Tick (价格大幅拉升至 3500)
    tick_bar = KlineBar(
        timestamp=1750000000000 + 51 * 3600000,
        open=3100.0,
        high=3510.0,
        low=3090.0,
        close=3500.0,
        volume=50.0,
        vol_currency=175000.0,
        is_closed=False,
    )
    realtime_ind = engine.update_kline_bar("1h", tick_bar)

    # 瞬时 EMA20 应该随 3500 拉升
    assert realtime_ind.ema_20 > last_closed_ema20
    # 但是底层已收线基准不能被未收线 Tick 污染！
    assert engine.states["1h"].ema_20 == last_closed_ema20

    # 模拟该 Bar 最终以 3480 收线确认
    closed_bar = KlineBar(
        timestamp=1750000000000 + 51 * 3600000,
        open=3100.0,
        high=3510.0,
        low=3090.0,
        close=3480.0,
        volume=60.0,
        vol_currency=208800.0,
        is_closed=True,
    )
    new_confirmed_ind = engine.update_kline_bar("1h", closed_bar)

    # 现在底层已收线基准应被正式更新
    assert round(engine.states["1h"].ema_20, 4) == new_confirmed_ind.ema_20
    assert engine.states["1h"].ema_20 > last_closed_ema20


def test_derivatives_monitoring_and_anomaly():
    """测试衍生品资金费率倾向评估与异常突增检测"""
    engine = IndicatorEngine(symbol="BTC-USDT-SWAP")

    # 1. 资金费率偏向评估
    assert engine.evaluate_funding_bias(0.0006) == FundingRateBias.EXTREME_POSITIVE
    assert engine.evaluate_funding_bias(0.00015) == FundingRateBias.MODERATE_POSITIVE
    assert engine.evaluate_funding_bias(0.00002) == FundingRateBias.NEUTRAL
    assert engine.evaluate_funding_bias(-0.0002) == FundingRateBias.MODERATE_NEGATIVE
    assert engine.evaluate_funding_bias(-0.0007) == FundingRateBias.EXTREME_NEGATIVE

    # 2. 模拟 OI 历史与突增检测
    base_ts = 1750000000000
    # 模拟 24h 前的 OI 为 10000
    engine.oi_history.append((base_ts - 24 * 3600 * 1000, 10000.0))
    # 模拟 15m 前的 OI 为 11000
    engine.oi_history.append((base_ts - 900 * 1000, 11000.0))

    # 更新当前指标: OI 激增到 12000 (15m 内增长 > 2.5%), 费率过热 0.0006, 盘口买卖比失衡 3.2
    deriv = engine.update_derivatives(
        funding_rate=0.0006,
        open_interest=12000.0,
        open_interest_usdt=12000.0 * 95000.0,
        long_short_ratio=2.5,
        imbalance_ratio=3.2,
        timestamp_ms=base_ts,
    )

    assert deriv.oi_change_24h_pct == pytest.approx(20.0, rel=1e-2)
    assert deriv.sentiment_score > 0.5

    # 3. 验证异常检测
    is_anomaly, reasons = engine.detect_anomalies()
    assert is_anomaly is True
    assert len(reasons) >= 2  # 包含 OI 突增、极端费率、盘口失衡等


def test_snapshot_dict_export():
    """测试统一指标快照字典导出格式"""
    engine = IndicatorEngine(symbol="SOL-USDT-SWAP")
    klines = generate_synthetic_klines(count=30, start_price=180.0, trend=0.5)
    engine.feed_kline_history("15m", klines)
    engine.feed_kline_history("1h", klines)

    snapshot = engine.get_snapshot_dict()
    assert snapshot["symbol"] == "SOL-USDT-SWAP"
    assert "indicators" in snapshot
    assert "15m" in snapshot["indicators"]
    assert "1h" in snapshot["indicators"]
    assert "derivatives" in snapshot
    assert "funding_bias" in snapshot
