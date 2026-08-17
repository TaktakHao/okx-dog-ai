"""
OKX-Dog 量化指标计算引擎 - 高性能多周期技术指标与衍生品监控引擎
模块: okx-dog-ai/indicator_engine.py

特性:
1. 支持 15m, 1h, 4h, 1d 等多周期 K 线数据维护与重采样。
2. 计算核心量化指标：
   - EMA (20, 50, 200) 均线系统
   - MACD (12, 26, 9) 异同移动平均线 (DIF, DEA, HIST)
   - RSI (14) 相对强弱指标 (Wilder's Smoothing)
   - 布林带 Bollinger Bands (20, 2.0: Upper, Middle, Lower, Bandwidth, %B)
   - ATR (14) 真实波幅
3. 毫秒级 O(1) 增量递推更新机制：支持在实时 Tick/Bar 变动时无需重算历史序列。
4. 衍生品专项监控：资金费率倾向评估、全网 OI 24h 变化率与短期异动突增检测。
5. 统一输出符合 schemas.py / models.py 契约的结构化多周期指标快照。
"""

from __future__ import annotations

import math
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from .schemas import (
        FundingRateBias,
        SinglePeriodIndicators,
        MultiPeriodIndicators,
        DerivativesMetrics,
    )
except ImportError:
    from okx_dog_ai.schemas import (
        FundingRateBias,
        SinglePeriodIndicators,
        MultiPeriodIndicators,
        DerivativesMetrics,
    )

logger = logging.getLogger("okx_dog.ai.indicator_engine")


@dataclass
class KlineBar:
    """单个 K 线 Bar 数据结构"""
    timestamp: int            # 开盘时间戳 (毫秒)
    open: float              # 开盘价
    high: float              # 最高价
    low: float               # 最低价
    close: float             # 收盘价
    volume: float = 0.0      # 成交量 (币数/张数)
    vol_currency: float = 0.0 # 成交额 (USDT)
    is_closed: bool = True   # 是否已收线确认


@dataclass
class TimeframeState:
    """单一周期的指标运行状态与增量缓存"""
    timeframe: str
    capacity: int = 500
    bars: deque[KlineBar] = field(default_factory=lambda: deque(maxlen=500))
    recent_closes_20: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    
    # 已收线闭合指标状态 (基准值)
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None
    
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    macd_dif: Optional[float] = None
    macd_dea: Optional[float] = None
    macd_hist: Optional[float] = None
    
    smma_gain: Optional[float] = None
    smma_loss: Optional[float] = None
    rsi_14: Optional[float] = None
    
    atr_14: Optional[float] = None
    last_closed_bar: Optional[KlineBar] = None
    
    # 实时未收线指标快照
    latest_indicators: Optional[SinglePeriodIndicators] = None


class IndicatorEngine:
    """
    高性能多周期量化技术指标计算引擎
    """

    # 默认各周期环形缓冲区容量
    RING_BUFFER_CAPACITIES = {
        "15m": 500,
        "1h": 500,
        "4h": 300,
        "1d": 250,
    }

    def __init__(self, symbol: str = "BTC-USDT-SWAP"):
        self.symbol = symbol
        self.states: Dict[str, TimeframeState] = {}
        for tf, cap in self.RING_BUFFER_CAPACITIES.items():
            self.states[tf] = TimeframeState(timeframe=tf, capacity=cap, bars=deque(maxlen=cap))
        
        # 衍生品与情绪监控历史
        self.oi_history: deque[Tuple[int, float]] = deque(maxlen=2880)  # 记录最近48小时的 (timestamp_ms, oi)
        self.latest_funding_rate: float = 0.0
        self.predicted_funding_rate: Optional[float] = None
        self.next_funding_time: Optional[int] = None
        self.latest_oi: float = 0.0
        self.latest_oi_usdt: Optional[float] = None
        self.long_short_ratio: Optional[float] = 1.0
        self.top_trader_ratio: Optional[float] = 1.0
        self.imbalance_ratio: float = 1.0

    # =========================================================================
    # 1. 历史全量预热 (Feed History)
    # =========================================================================

    def feed_kline_history(self, timeframe: str, klines: List[Union[Dict[str, Any], List[Any], KlineBar]]) -> None:
        """
        全量预热历史 K 线数据，重置并建立该周期的增量递推基准状态。
        klines: 按时间由远及近升序排列的 K 线列表。
        """
        if timeframe not in self.states:
            cap = self.RING_BUFFER_CAPACITIES.get(timeframe, 500)
            self.states[timeframe] = TimeframeState(timeframe=timeframe, capacity=cap, bars=deque(maxlen=cap))
        
        state = self.states[timeframe]
        state.bars.clear()
        state.recent_closes_20.clear()
        
        # 重置状态
        state.ema_20 = None
        state.ema_50 = None
        state.ema_200 = None
        state.ema_12 = None
        state.ema_26 = None
        state.macd_dif = None
        state.macd_dea = None
        state.macd_hist = None
        state.smma_gain = None
        state.smma_loss = None
        state.rsi_14 = None
        state.atr_14 = None
        state.last_closed_bar = None
        state.latest_indicators = None

        if not klines:
            return

        parsed_bars: List[KlineBar] = []
        for item in klines:
            bar = self._parse_kline_item(item)
            if bar is not None:
                parsed_bars.append(bar)

        if not parsed_bars:
            return

        # 逐根 Bar 顺序计算预热已闭合状态
        for i, bar in enumerate(parsed_bars):
            is_last = (i == len(parsed_bars) - 1)
            # 若是最后一根且未闭合，作为当前未收线 Bar 处理，否则全部作为收线 Bar
            if is_last and not bar.is_closed:
                self.update_kline_bar(timeframe, bar)
            else:
                self._apply_closed_bar(state, bar)

        # 确保最新指标已生成
        if state.last_closed_bar is not None and state.latest_indicators is None:
            state.latest_indicators = self._calculate_indicators_from_state(state, state.last_closed_bar.close, state.last_closed_bar.high, state.last_closed_bar.low)

    # =========================================================================
    # 2. O(1) 增量更新核心逻辑
    # =========================================================================

    def update_kline_bar(self, timeframe: str, kline: Union[Dict[str, Any], List[Any], KlineBar]) -> SinglePeriodIndicators:
        """
        处理单根 K 线 Bar 的增量更新（支持收线确认或未收线实时 Tick 更新）。
        时间复杂度 O(1)。
        """
        if timeframe not in self.states:
            cap = self.RING_BUFFER_CAPACITIES.get(timeframe, 500)
            self.states[timeframe] = TimeframeState(timeframe=timeframe, capacity=cap, bars=deque(maxlen=cap))
        
        state = self.states[timeframe]
        bar = self._parse_kline_item(kline)
        if bar is None:
            if state.latest_indicators:
                return state.latest_indicators
            return self._build_empty_indicators(timeframe)

        if bar.is_closed:
            # 1. 确认收线：更新确定性基准状态并入队
            self._apply_closed_bar(state, bar)
            ind = self._calculate_indicators_from_state(state, bar.close, bar.high, bar.low)
            state.latest_indicators = ind
            return ind
        else:
            # 2. 实时未收线：基于上次已收线的确定状态计算瞬时指标，不污染已收线基准
            if state.last_closed_bar is None:
                # 若无历史收线基准，临时基于当前 Bar 初始化
                ind = self._calculate_initial_indicators(timeframe, bar)
                state.latest_indicators = ind
                return ind

            ind = self._calculate_realtime_indicators(state, bar)
            state.latest_indicators = ind
            return ind

    def _apply_closed_bar(self, state: TimeframeState, bar: KlineBar) -> None:
        """应用收线确认的 Bar，严格推进 O(1) 递推状态机"""
        state.bars.append(bar)
        state.recent_closes_20.append(bar.close)
        prev_bar = state.last_closed_bar

        # --- 1. EMA 20, 50, 200 增量计算 ---
        alpha_20 = 2.0 / (20.0 + 1.0)
        alpha_50 = 2.0 / (50.0 + 1.0)
        alpha_200 = 2.0 / (200.0 + 1.0)

        if state.ema_20 is None:
            state.ema_20 = bar.close
        else:
            state.ema_20 = alpha_20 * bar.close + (1.0 - alpha_20) * state.ema_20

        if state.ema_50 is None:
            state.ema_50 = bar.close
        else:
            state.ema_50 = alpha_50 * bar.close + (1.0 - alpha_50) * state.ema_50

        if len(state.bars) >= 200 or state.ema_200 is not None:
            if state.ema_200 is None:
                # 初始 200 根 EMA 可先用前 200 根均值或第 200 根价格
                state.ema_200 = bar.close
            else:
                state.ema_200 = alpha_200 * bar.close + (1.0 - alpha_200) * state.ema_200

        # --- 2. MACD (12, 26, 9) 增量计算 ---
        alpha_12 = 2.0 / (12.0 + 1.0)
        alpha_26 = 2.0 / (26.0 + 1.0)
        alpha_9 = 2.0 / (9.0 + 1.0)

        if state.ema_12 is None:
            state.ema_12 = bar.close
        else:
            state.ema_12 = alpha_12 * bar.close + (1.0 - alpha_12) * state.ema_12

        if state.ema_26 is None:
            state.ema_26 = bar.close
        else:
            state.ema_26 = alpha_26 * bar.close + (1.0 - alpha_26) * state.ema_26

        state.macd_dif = state.ema_12 - state.ema_26

        if state.macd_dea is None:
            state.macd_dea = state.macd_dif
        else:
            state.macd_dea = alpha_9 * state.macd_dif + (1.0 - alpha_9) * state.macd_dea

        state.macd_hist = (state.macd_dif - state.macd_dea) * 2.0

        # --- 3. RSI (14) Wilder's Smoothing 增量计算 ---
        if prev_bar is not None:
            diff = bar.close - prev_bar.close
            gain = max(diff, 0.0)
            loss = max(-diff, 0.0)

            if state.smma_gain is None or state.smma_loss is None:
                state.smma_gain = gain
                state.smma_loss = loss
            else:
                state.smma_gain = (state.smma_gain * 13.0 + gain) / 14.0
                state.smma_loss = (state.smma_loss * 13.0 + loss) / 14.0

            if state.smma_loss < 1e-9:
                state.rsi_14 = 100.0 if state.smma_gain > 0 else 50.0
            else:
                rs = state.smma_gain / state.smma_loss
                state.rsi_14 = 100.0 - (100.0 / (1.0 + rs))
        else:
            state.smma_gain = 0.0
            state.smma_loss = 0.0
            state.rsi_14 = 50.0

        # --- 4. ATR (14) 增量计算 ---
        if prev_bar is not None:
            tr = max(
                bar.high - bar.low,
                abs(bar.high - prev_bar.close),
                abs(bar.low - prev_bar.close)
            )
            if state.atr_14 is None:
                state.atr_14 = tr
            else:
                state.atr_14 = (state.atr_14 * 13.0 + tr) / 14.0
        else:
            state.atr_14 = bar.high - bar.low

        state.last_closed_bar = bar

    def _calculate_realtime_indicators(self, state: TimeframeState, current_bar: KlineBar) -> SinglePeriodIndicators:
        """针对未收线的瞬时 Tick/Bar 进行 O(1) 实时无锁推算"""
        prev_bar = state.last_closed_bar
        curr_price = current_bar.close

        # EMA 瞬时推算
        alpha_20 = 2.0 / 21.0
        alpha_50 = 2.0 / 51.0
        alpha_200 = 2.0 / 201.0

        ema_20 = alpha_20 * curr_price + (1.0 - alpha_20) * (state.ema_20 if state.ema_20 is not None else curr_price)
        ema_50 = alpha_50 * curr_price + (1.0 - alpha_50) * (state.ema_50 if state.ema_50 is not None else curr_price)
        ema_200 = (alpha_200 * curr_price + (1.0 - alpha_200) * state.ema_200) if state.ema_200 is not None else None

        # MACD 瞬时推算
        alpha_12 = 2.0 / 13.0
        alpha_26 = 2.0 / 27.0
        alpha_9 = 2.0 / 10.0

        ema_12 = alpha_12 * curr_price + (1.0 - alpha_12) * (state.ema_12 if state.ema_12 is not None else curr_price)
        ema_26 = alpha_26 * curr_price + (1.0 - alpha_26) * (state.ema_26 if state.ema_26 is not None else curr_price)
        dif = ema_12 - ema_26
        dea = alpha_9 * dif + (1.0 - alpha_9) * (state.macd_dea if state.macd_dea is not None else dif)
        hist = (dif - dea) * 2.0

        # RSI 瞬时推算
        if prev_bar is not None:
            diff = curr_price - prev_bar.close
            gain = max(diff, 0.0)
            loss = max(-diff, 0.0)
            base_gain = state.smma_gain if state.smma_gain is not None else 0.0
            base_loss = state.smma_loss if state.smma_loss is not None else 0.0
            cur_smma_gain = (base_gain * 13.0 + gain) / 14.0
            cur_smma_loss = (base_loss * 13.0 + loss) / 14.0

            if cur_smma_loss < 1e-9:
                rsi = 100.0 if cur_smma_gain > 0 else 50.0
            else:
                rs = cur_smma_gain / cur_smma_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi = 50.0

        # 布林带 (20, 2.0)
        bb_upper, bb_middle, bb_lower, bb_width = self._calculate_bollinger(state.recent_closes_20, curr_price)

        # ATR 瞬时推算
        if prev_bar is not None:
            tr = max(
                current_bar.high - current_bar.low,
                abs(current_bar.high - prev_bar.close),
                abs(current_bar.low - prev_bar.close)
            )
            base_atr = state.atr_14 if state.atr_14 is not None else tr
            atr = (base_atr * 13.0 + tr) / 14.0
        else:
            atr = current_bar.high - current_bar.low

        is_golden = (state.ema_20 is not None and state.ema_50 is not None and state.ema_20 <= state.ema_50 and ema_20 > ema_50)
        is_death = (state.ema_20 is not None and state.ema_50 is not None and state.ema_20 >= state.ema_50 and ema_20 < ema_50)

        return SinglePeriodIndicators(
            timeframe=state.timeframe,
            ema_20=round(ema_20, 4),
            ema_50=round(ema_50, 4),
            ema_200=round(ema_200, 4) if ema_200 is not None else None,
            macd_dif=round(dif, 4),
            macd_dea=round(dea, 4),
            macd_hist=round(hist, 4),
            rsi_14=round(rsi, 2),
            bb_upper=round(bb_upper, 4),
            bb_middle=round(bb_middle, 4),
            bb_lower=round(bb_lower, 4),
            bb_width_pct=round(bb_width, 2),
            atr_14=round(atr, 4),
            is_golden_cross=is_golden,
            is_death_cross=is_death,
        )

    def _calculate_indicators_from_state(self, state: TimeframeState, close: float, high: float, low: float) -> SinglePeriodIndicators:
        """从已确定的状态生成结构化指标对象"""
        bb_upper, bb_middle, bb_lower, bb_width = self._calculate_bollinger(state.recent_closes_20, None)
        ema_20 = state.ema_20 if state.ema_20 is not None else close
        ema_50 = state.ema_50 if state.ema_50 is not None else close
        ema_200 = state.ema_200

        dif = state.macd_dif if state.macd_dif is not None else 0.0
        dea = state.macd_dea if state.macd_dea is not None else 0.0
        hist = state.macd_hist if state.macd_hist is not None else 0.0
        rsi = state.rsi_14 if state.rsi_14 is not None else 50.0
        atr = state.atr_14 if state.atr_14 is not None else (high - low)

        return SinglePeriodIndicators(
            timeframe=state.timeframe,
            ema_20=round(ema_20, 4),
            ema_50=round(ema_50, 4),
            ema_200=round(ema_200, 4) if ema_200 is not None else None,
            macd_dif=round(dif, 4),
            macd_dea=round(dea, 4),
            macd_hist=round(hist, 4),
            rsi_14=round(rsi, 2),
            bb_upper=round(bb_upper, 4),
            bb_middle=round(bb_middle, 4),
            bb_lower=round(bb_lower, 4),
            bb_width_pct=round(bb_width, 2),
            atr_14=round(atr, 4),
            is_golden_cross=False,
            is_death_cross=False,
        )

    def _calculate_bollinger(self, recent_closes: deque[float], current_price: Optional[float] = None) -> Tuple[float, float, float, float]:
        """
        计算布林带 (周期 20, 标准差 2.0)。
        若传入 current_price，模拟将当前价加入窗口。
        返回 (Upper, Middle, Lower, Bandwidth%)
        """
        prices = list(recent_closes)
        if current_price is not None:
            if len(prices) >= 20:
                prices = prices[1:] + [current_price]
            else:
                prices = prices + [current_price]

        if not prices:
            p = current_price if current_price is not None else 0.0
            return p, p, p, 0.0

        n = len(prices)
        mean = sum(prices) / n
        if n < 2:
            return mean, mean, mean, 0.0

        # 计算总体标准差
        variance = sum((x - mean) ** 2 for x in prices) / n
        std = math.sqrt(variance)

        upper = mean + 2.0 * std
        lower = mean - 2.0 * std
        bandwidth = ((upper - lower) / mean * 100.0) if mean > 1e-9 else 0.0

        return upper, mean, lower, bandwidth

    def _calculate_initial_indicators(self, timeframe: str, bar: KlineBar) -> SinglePeriodIndicators:
        """当只有单根 Bar 时生成初始指标"""
        return SinglePeriodIndicators(
            timeframe=timeframe,
            ema_20=bar.close,
            ema_50=bar.close,
            ema_200=bar.close,
            macd_dif=0.0,
            macd_dea=0.0,
            macd_hist=0.0,
            rsi_14=50.0,
            bb_upper=bar.close,
            bb_middle=bar.close,
            bb_lower=bar.close,
            bb_width_pct=0.0,
            atr_14=max(bar.high - bar.low, 0.01),
            is_golden_cross=False,
            is_death_cross=False,
        )

    def _build_empty_indicators(self, timeframe: str) -> SinglePeriodIndicators:
        return SinglePeriodIndicators(
            timeframe=timeframe,
            ema_20=0.0,
            ema_50=0.0,
            ema_200=None,
            macd_dif=0.0,
            macd_dea=0.0,
            macd_hist=0.0,
            rsi_14=50.0,
            bb_upper=0.0,
            bb_middle=0.0,
            bb_lower=0.0,
            bb_width_pct=0.0,
            atr_14=0.0,
            is_golden_cross=False,
            is_death_cross=False,
        )

    def _parse_kline_item(self, item: Any) -> Optional[KlineBar]:
        """将不同格式的 K 线输入标准化解析为 KlineBar"""
        if isinstance(item, KlineBar):
            return item
        if isinstance(item, dict):
            # 支持 OKX 格式与通用格式
            ts = int(item.get("timestamp") or item.get("time") or item.get("ts") or 0)
            o = float(item.get("open") or item.get("o") or 0.0)
            h = float(item.get("high") or item.get("h") or 0.0)
            l = float(item.get("low") or item.get("l") or 0.0)
            c = float(item.get("close") or item.get("c") or 0.0)
            v = float(item.get("volume") or item.get("vol") or item.get("v") or 0.0)
            vc = float(item.get("vol_currency") or item.get("volCcy") or 0.0)
            is_closed = bool(item.get("is_closed", item.get("confirmed", True)))
            # OKX WebSocket 的 confirm 字段: "1" 表示收线, "0" 表示未收线
            if "confirm" in item:
                is_closed = (str(item["confirm"]) == "1")
            return KlineBar(timestamp=ts, open=o, high=h, low=l, close=c, volume=v, vol_currency=vc, is_closed=is_closed)
        if isinstance(item, (list, tuple)) and len(item) >= 5:
            # [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
            ts = int(item[0])
            o = float(item[1])
            h = float(item[2])
            l = float(item[3])
            c = float(item[4])
            v = float(item[5]) if len(item) > 5 else 0.0
            vc = float(item[6]) if len(item) > 6 else 0.0
            is_closed = True
            if len(item) >= 9:
                is_closed = (str(item[8]) == "1" or str(item[8]).lower() == "true")
            return KlineBar(timestamp=ts, open=o, high=h, low=l, close=c, volume=v, vol_currency=vc, is_closed=is_closed)
        return None

    # =========================================================================
    # 3. 衍生品监控与 OI / 资金费率异动检测
    # =========================================================================

    def update_derivatives(
        self,
        funding_rate: float,
        open_interest: float,
        predicted_funding_rate: Optional[float] = None,
        next_funding_time: Optional[int] = None,
        open_interest_usdt: Optional[float] = None,
        long_short_ratio: Optional[float] = None,
        top_trader_ratio: Optional[float] = None,
        imbalance_ratio: float = 1.0,
        timestamp_ms: Optional[int] = None,
    ) -> DerivativesMetrics:
        """更新衍生品持仓与情绪指标"""
        now_ms = timestamp_ms or int(datetime.utcnow().timestamp() * 1000)
        self.latest_funding_rate = funding_rate
        self.predicted_funding_rate = predicted_funding_rate
        self.next_funding_time = next_funding_time
        self.latest_oi = open_interest
        self.latest_oi_usdt = open_interest_usdt
        self.long_short_ratio = long_short_ratio
        self.top_trader_ratio = top_trader_ratio
        self.imbalance_ratio = imbalance_ratio

        self.oi_history.append((now_ms, open_interest))

        # 计算 24h OI 变化率
        oi_change_24h_pct = self._calculate_oi_change_pct(now_ms, target_hours=24.0)

        # 资金费率结算倒计时
        countdown_min: Optional[int] = None
        if next_funding_time and next_funding_time > now_ms:
            countdown_min = max(0, int((next_funding_time - now_ms) / 60000))

        # 情绪综合评分 (-1.0 到 +1.0)
        sentiment_score = self._calculate_sentiment_score(funding_rate, oi_change_24h_pct, long_short_ratio)

        annualized_pct = funding_rate * 3 * 365 * 100.0  # 8小时结算一次，年化 = 费率 * 3 * 365 * 100%

        return DerivativesMetrics(
            symbol=self.symbol,
            funding_rate=funding_rate,
            predicted_funding_rate=predicted_funding_rate,
            funding_rate_annualized_pct=round(annualized_pct, 2),
            next_funding_time=next_funding_time,
            funding_countdown_min=countdown_min,
            open_interest=open_interest,
            open_interest_usdt=open_interest_usdt,
            oi_change_24h_pct=round(oi_change_24h_pct, 2),
            long_short_ratio=round(long_short_ratio, 2) if long_short_ratio is not None else None,
            top_trader_ratio=round(top_trader_ratio, 2) if top_trader_ratio is not None else None,
            sentiment_score=round(sentiment_score, 2),
            timestamp=now_ms,
        )

    def _calculate_oi_change_pct(self, current_time_ms: int, target_hours: float = 24.0) -> float:
        """基于历史队列计算指定小时跨度的 OI 百分比变动"""
        if not self.oi_history or len(self.oi_history) < 2:
            return 0.0

        target_past_ms = current_time_ms - int(target_hours * 3600 * 1000)
        # 寻找最接近 target_past_ms 的历史记录
        closest_oi: Optional[float] = None
        min_diff = float("inf")

        for ts, oi_val in self.oi_history:
            diff = abs(ts - target_past_ms)
            if diff < min_diff:
                min_diff = diff
                closest_oi = oi_val

        if closest_oi is None or closest_oi <= 0:
            closest_oi = self.oi_history[0][1]

        if closest_oi <= 0:
            return 0.0

        return ((self.latest_oi - closest_oi) / closest_oi) * 100.0

    def _calculate_sentiment_score(self, funding_rate: float, oi_change_24h: float, ls_ratio: Optional[float]) -> float:
        """量化计算衍生品综合情绪得分 (-1.0 极度恐慌/看空 到 +1.0 极度贪婪/看多)"""
        # 资金费率得分: 0.0001 (0.01%) 为基准中性, 0.0005 为极度看多 (+0.6)
        fr_score = max(min(funding_rate / 0.0005, 1.0), -1.0) * 0.4

        # OI 变化得分: +10% 对应 +0.3
        oi_score = max(min(oi_change_24h / 10.0, 1.0), -1.0) * 0.3

        # 多空人数比得分: 1.0 为中性, 2.0 对应散户狂热 (+0.3)
        ls_score = 0.0
        if ls_ratio is not None and ls_ratio > 0:
            ls_score = max(min((ls_ratio - 1.0) / 1.0, 1.0), -1.0) * 0.3

        total = fr_score + oi_score + ls_score
        return max(min(total, 1.0), -1.0)

    def evaluate_funding_bias(self, funding_rate: float) -> FundingRateBias:
        """评估资金费率偏向分类"""
        if funding_rate >= 0.0005:
            return FundingRateBias.EXTREME_POSITIVE
        elif funding_rate >= 0.0001:
            return FundingRateBias.MODERATE_POSITIVE
        elif funding_rate <= -0.0005:
            return FundingRateBias.EXTREME_NEGATIVE
        elif funding_rate <= -0.0001:
            return FundingRateBias.MODERATE_NEGATIVE
        else:
            return FundingRateBias.NEUTRAL

    def detect_anomalies(self) -> Tuple[bool, List[str]]:
        """
        检测是否存在突发盘面异动（如 OI 瞬时激增、极端费率、买卖盘深度极度失衡）
        返回 (is_anomaly, anomaly_reasons)
        """
        reasons: List[str] = []
        now_ms = int(datetime.utcnow().timestamp() * 1000)

        # 1. 检查 15 分钟内 OI 突增 (超过 2.5%)
        short_oi_change = self._calculate_oi_change_pct(now_ms, target_hours=0.25)
        if abs(short_oi_change) >= 2.5:
            reasons.append(f"15分钟 OI 突增 {short_oi_change:+.2f}%")

        # 2. 检查资金费率极端过热
        if abs(self.latest_funding_rate) >= 0.0005:
            bias_str = "极端看多过热" if self.latest_funding_rate > 0 else "极端看空拥挤"
            reasons.append(f"资金费率触发 {bias_str} ({self.latest_funding_rate * 100:.4f}%)")

        # 3. 检查盘口买卖失衡
        if self.imbalance_ratio >= 2.5 or self.imbalance_ratio <= 0.4:
            reasons.append(f"盘口深度严重失衡 (买/卖比={self.imbalance_ratio:.2f})")

        # 4. 检查 15m RSI 极端超买/超卖
        tf_15m = self.states.get("15m")
        if tf_15m and tf_15m.latest_indicators:
            rsi = tf_15m.latest_indicators.rsi_14
            if rsi >= 80.0:
                reasons.append(f"15M RSI 达到极端超买区 ({rsi:.1f})")
            elif rsi <= 20.0:
                reasons.append(f"15M RSI 达到极端超卖区 ({rsi:.1f})")

        return (len(reasons) > 0, reasons)

    # =========================================================================
    # 4. 快照聚合与输出
    # =========================================================================

    def get_multi_period_indicators(self) -> MultiPeriodIndicators:
        """获取所有周期的最新指标汇总包"""
        indicators_map: Dict[str, SinglePeriodIndicators] = {}
        for tf, state in self.states.items():
            if state.latest_indicators:
                indicators_map[tf] = state.latest_indicators
            else:
                indicators_map[tf] = self._build_empty_indicators(tf)

        return MultiPeriodIndicators(
            symbol=self.symbol,
            timestamp=int(datetime.utcnow().timestamp() * 1000),
            indicators=indicators_map
        )

    def get_derivatives_metrics(self) -> DerivativesMetrics:
        """获取最新衍生品监控指标"""
        now_ms = int(datetime.utcnow().timestamp() * 1000)
        oi_change = self._calculate_oi_change_pct(now_ms, target_hours=24.0)
        countdown_min = max(0, int((self.next_funding_time - now_ms) / 60000)) if self.next_funding_time and self.next_funding_time > now_ms else None
        sentiment = self._calculate_sentiment_score(self.latest_funding_rate, oi_change, self.long_short_ratio)
        annualized = self.latest_funding_rate * 3 * 365 * 100.0

        return DerivativesMetrics(
            symbol=self.symbol,
            funding_rate=self.latest_funding_rate,
            predicted_funding_rate=self.predicted_funding_rate,
            funding_rate_annualized_pct=round(annualized, 2),
            next_funding_time=self.next_funding_time,
            funding_countdown_min=countdown_min,
            open_interest=self.latest_oi,
            open_interest_usdt=self.latest_oi_usdt,
            oi_change_24h_pct=round(oi_change, 2),
            long_short_ratio=round(self.long_short_ratio, 2) if self.long_short_ratio is not None else None,
            top_trader_ratio=round(self.top_trader_ratio, 2) if self.top_trader_ratio is not None else None,
            sentiment_score=round(sentiment, 2),
            timestamp=now_ms,
        )

    def get_snapshot_dict(self) -> Dict[str, Any]:
        """输出统一标准的结构化指标快照字典，供 Prompt 组装器直接消费"""
        multi_ind = self.get_multi_period_indicators()
        deriv = self.get_derivatives_metrics()
        is_anomaly, anomaly_reasons = self.detect_anomalies()

        # 提取最新价格
        current_price = 0.0
        for tf in ["15m", "1h", "4h", "1d"]:
            state = self.states.get(tf)
            if state and state.last_closed_bar:
                current_price = state.last_closed_bar.close
                break

        return {
            "symbol": self.symbol,
            "timestamp": multi_ind.timestamp,
            "current_price": current_price,
            "indicators": {tf: ind.model_dump() for tf, ind in multi_ind.indicators.items()},
            "derivatives": deriv.model_dump(),
            "is_anomaly": is_anomaly,
            "anomaly_reasons": anomaly_reasons,
            "funding_bias": self.evaluate_funding_bias(deriv.funding_rate).value,
        }
