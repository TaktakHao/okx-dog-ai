"""
OKX-Dog 量化与 AI 研判决策中枢 - 核心数据模式与契约定义
模块: okx-dog-ai/schemas.py
100% 严格对齐 AI_SCHEMA.json, models.py 与 TYPES_CONTRACT.ts
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, field_serializer


# =============================================================================
# 1. 核心领域枚举 (Domain Enums)
# =============================================================================

class MarketRegime(str, Enum):
    """宏观盘面结构分类 (严格对齐 AI_SCHEMA.json)"""
    TRENDING_UP = "TRENDING_UP"              # 多头趋势
    TRENDING_DOWN = "TRENDING_DOWN"          # 空头趋势
    RANGING = "RANGING"                      # 震荡整理
    VOLATILE_BREAKOUT = "VOLATILE_BREAKOUT"  # 剧烈突破/异动


class TimeframeTrend(str, Enum):
    """单一周期趋势研判 (严格对齐 AI_SCHEMA.json definitions.TimeframeDetail)"""
    BULLISH = "BULLISH"                      # 多头排列/强劲上涨
    BEARISH = "BEARISH"                      # 空头排列/破位下跌
    NEUTRAL_CHOPPY = "NEUTRAL_CHOPPY"        # 均线缠绕/宽幅震荡
    OVERBOUGHT = "OVERBOUGHT"                # 超买顶背离
    OVERSOLD = "OVERSOLD"                    # 超卖底背离


class FundingRateBias(str, Enum):
    """资金费率倾向评估 (严格对齐 AI_SCHEMA.json properties.derivatives_sentiment)"""
    EXTREME_POSITIVE = "EXTREME_POSITIVE"    # 极端看多 (费率过热风险)
    MODERATE_POSITIVE = "MODERATE_POSITIVE"  # 健康看多
    NEUTRAL = "NEUTRAL"                      # 中性平衡
    MODERATE_NEGATIVE = "MODERATE_NEGATIVE"  # 健康看空
    EXTREME_NEGATIVE = "EXTREME_NEGATIVE"    # 极端看空 (空头挤压风险)


class SignalAction(str, Enum):
    """AI 核心操作动作 (100% 对齐 AI_SCHEMA.json)"""
    BUY_LONG = "BUY_LONG"                    # 做多 / 买入开多
    SELL_SHORT = "SELL_SHORT"                # 做空 / 卖出开空
    CLOSE_POSITION = "CLOSE_POSITION"        # 平仓离场
    HOLD_WAIT = "HOLD_WAIT"                  # 空仓观望 / 维持现有持仓


class SignalUrgency(str, Enum):
    """交易信号执行紧迫度"""
    LOW = "LOW"                              # 挂单等待回调入场
    MEDIUM = "MEDIUM"                        # 常规限价单挂单
    HIGH = "HIGH"                            # 市价即时成交 / 突破紧急入场或紧急止损


class TradePlanOrderType(str, Enum):
    """建议交易计划委托类型"""
    LIMIT = "LIMIT"                          # 限价单
    MARKET = "MARKET"                        # 市价单
    TRIGGER_LIMIT = "TRIGGER_LIMIT"          # 条件计划限价单


class LLMProvider(str, Enum):
    """支持的大模型服务商"""
    ANTIGRAVITY = "antigravity"
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


# =============================================================================
# 2. 多周期技术指标与衍生品数据模型 (Indicator & Derivatives Models)
# =============================================================================

class SinglePeriodIndicators(BaseModel):
    """单个时间周期的量化技术指标集合"""
    model_config = ConfigDict(populate_by_name=True)

    timeframe: str = Field(..., description="周期 (15m, 1h, 4h, 1d)")
    ema_20: float = Field(..., description="EMA 20 指数移动平均线")
    ema_50: float = Field(..., description="EMA 50 指数移动平均线")
    ema_200: Optional[float] = Field(None, description="EMA 200 指数移动平均线")
    macd_dif: float = Field(..., description="MACD DIF 快线")
    macd_dea: float = Field(..., description="MACD DEA 慢线")
    macd_hist: float = Field(..., description="MACD 柱状图")
    rsi_14: float = Field(..., description="RSI 14 相对强弱指标")
    bb_upper: float = Field(..., description="布林带上轨")
    bb_middle: float = Field(..., description="布林带中轨")
    bb_lower: float = Field(..., description="布林带下轨")
    bb_width_pct: Optional[float] = Field(None, description="布林带带宽百分比 ((上轨-下轨)/中轨 * 100)")
    atr_14: float = Field(..., description="ATR 14 真实波幅")
    is_golden_cross: Optional[bool] = Field(False, description="EMA20 是否上穿 EMA50 金叉")
    is_death_cross: Optional[bool] = Field(False, description="EMA20 是否下穿 EMA50 死叉")


class MultiPeriodIndicators(BaseModel):
    """多周期指标汇总聚合包"""
    model_config = ConfigDict(populate_by_name=True)

    symbol: str = Field(..., description="标的代码，如 BTC-USDT-SWAP")
    timestamp: int = Field(..., description="计算毫秒时间戳")
    indicators: Dict[str, SinglePeriodIndicators] = Field(..., description="各周期指标映射，例如 {'15m': ..., '1h': ...}")


class DerivativesMetrics(BaseModel):
    """衍生品专项监控指标"""
    model_config = ConfigDict(populate_by_name=True)

    symbol: str = Field(..., description="标的代码")
    funding_rate: float = Field(..., description="当前资金费率 (例如 0.0001)")
    predicted_funding_rate: Optional[float] = Field(None, description="预测下期资金费率")
    funding_rate_annualized_pct: Optional[float] = Field(None, description="年化资金费率百分比")
    next_funding_time: Optional[int] = Field(None, description="下次结算时间戳 (ms)")
    funding_countdown_min: Optional[int] = Field(None, description="距离下次结算倒计时分钟数")
    open_interest: float = Field(..., description="全网未平仓合约量 (张数/币数)")
    open_interest_usdt: Optional[float] = Field(None, description="全网未平仓合约名义估值 (USDT)")
    oi_change_24h_pct: float = Field(0.0, description="24小时 OI 变动百分比")
    long_short_ratio: Optional[float] = Field(None, description="散户多空持仓人数比")
    top_trader_ratio: Optional[float] = Field(None, description="大户持仓多空比")
    sentiment_score: Optional[float] = Field(0.0, description="情绪量化得分 (-1.0 ~ +1.0)")
    timestamp: Optional[int] = Field(None, description="指标生成时间戳 (ms)")


class PositionSnapshot(BaseModel):
    """当前持仓简要快照"""
    model_config = ConfigDict(populate_by_name=True)

    symbol: str = Field(..., description="标的代码")
    side: str = Field(..., description="持仓方向 ('long', 'short', 'net')")
    leverage: int = Field(1, description="杠杆倍数")
    contracts: float = Field(..., description="持仓张数或币数")
    notional_usd: float = Field(..., description="名义价值 (USDT)")
    entry_price: float = Field(..., description="开仓均价")
    mark_price: float = Field(..., description="当前标记价")
    unrealized_pnl: float = Field(0.0, description="未实现盈亏 (USDT)")
    pnl_percentage: float = Field(0.0, description="收益率百分比")
    liquidation_price: Optional[float] = Field(None, description="预估强平价")
    margin_ratio: Optional[float] = Field(None, description="保证金率")
    holding_hours: Optional[float] = Field(0.0, description="持仓时长 (小时)")


class HardRiskLimits(BaseModel):
    """硬风控参数快照"""
    model_config = ConfigDict(populate_by_name=True)

    max_order_usdt: float = Field(500.0, description="单笔最大委托名义价值 (USDT)")
    max_daily_loss_usdt: float = Field(200.0, description="单日累计最大亏损熔断阈值 (USDT)")
    max_leverage: int = Field(5, description="最高允许杠杆倍数")
    max_slippage_pct: float = Field(0.5, description="最大允许滑点百分比 (0.5%)")


class MarketContextSnapshot(BaseModel):
    """送入 Prompt 组装器的结构化行情全景快照"""
    model_config = ConfigDict(populate_by_name=True)

    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="研判跟踪 UUID")
    symbol: str = Field(..., description="标的代码")
    timestamp: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp() * 1000), description="毫秒时间戳")
    current_price: float = Field(..., description="最新标记价/成交价")
    change_24h_pct: float = Field(0.0, description="24小时涨跌幅百分比")
    high_24h: Optional[float] = Field(None, description="24小时最高价")
    low_24h: Optional[float] = Field(None, description="24小时最低价")
    volume_24h_usdt: Optional[float] = Field(None, description="24小时成交额 (USDT)")
    multi_indicators: MultiPeriodIndicators = Field(..., description="多周期技术指标")
    derivatives: DerivativesMetrics = Field(..., description="衍生品指标")
    active_position: Optional[PositionSnapshot] = Field(None, description="交易员当前持仓状态")
    account_balance_usdt: float = Field(1000.0, description="账户可用 USDT 余额")
    risk_limits: HardRiskLimits = Field(default_factory=HardRiskLimits, description="硬风控规则参数")
    user_strategy_bias: str = Field("BALANCED", description="用户交易风格 (CONSERVATIVE_TREND / AGGRESSIVE_SCALP / BALANCED)")
    imbalance_ratio: Optional[float] = Field(1.0, description="盘口买卖失衡比 (Bids/Asks)")
    orderbook_bids_top5: Optional[List[List[float]]] = Field(None, description="盘口前5买档 [[price, size], ...]")
    orderbook_asks_top5: Optional[List[List[float]]] = Field(None, description="盘口前5卖档 [[price, size], ...]")
    is_anomaly_mode: bool = Field(False, description="是否处于突发异动模式")
    anomaly_desc: Optional[str] = Field(None, description="异动类型描述")


# =============================================================================
# 3. AI 决策契约模型 (100% 严格对齐 AI_SCHEMA.json)
# =============================================================================

class TimeframeDetail(BaseModel):
    """单一周期技术指标与形态研判细节 (AI_SCHEMA definitions.TimeframeDetail)"""
    model_config = ConfigDict(populate_by_name=True)

    trend: TimeframeTrend = Field(..., description="周期趋势：BULLISH, BEARISH, NEUTRAL_CHOPPY, OVERBOUGHT, OVERSOLD")
    key_indicators_summary: str = Field(..., description="该周期关键指标表现描述")
    support_level: float = Field(..., ge=0.0, description="识别出的第一关键支撑价位")
    resistance_level: float = Field(..., ge=0.0, description="识别出的第一关键阻力价位")


class TimeframeAnalysis(BaseModel):
    """多周期技术面共振分析字典 (AI_SCHEMA properties.timeframe_analysis)"""
    model_config = ConfigDict(populate_by_name=True)

    tf_15m: TimeframeDetail = Field(..., description="15分钟周期研判细节")
    tf_1h: TimeframeDetail = Field(..., description="1小时周期研判细节")
    tf_4h: TimeframeDetail = Field(..., description="4小时周期研判细节")
    tf_1d: TimeframeDetail = Field(..., description="日线周期研判细节")


class DerivativesSentiment(BaseModel):
    """衍生品与市场情绪专项研判 (AI_SCHEMA properties.derivatives_sentiment)"""
    model_config = ConfigDict(populate_by_name=True)

    funding_rate_bias: FundingRateBias = Field(..., description="资金费率倾向")
    open_interest_interpretation: str = Field(..., description="全网未平仓合约量 (OI) 异动解读")
    long_short_ratio_state: str = Field(..., description="多空持仓比与大户状态分析")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="综合情绪量化得分 (-1.0 ~ +1.0)")


class AISignal(BaseModel):
    """最终交易决策信号与置信度 (AI_SCHEMA properties.signal)"""
    model_config = ConfigDict(populate_by_name=True)

    action: SignalAction = Field(..., description="核心操作动作：BUY_LONG, SELL_SHORT, CLOSE_POSITION, HOLD_WAIT")
    confidence: float = Field(..., ge=0.0, le=1.0, description="决策置信度打分 [0.0, 1.0]")
    urgency: SignalUrgency = Field(..., description="执行紧迫度：LOW, MEDIUM, HIGH")


class TakeProfitLevel(BaseModel):
    """分阶段止盈点位与平仓比例 (AI_SCHEMA definitions.TakeProfitLevel)"""
    model_config = ConfigDict(populate_by_name=True)

    price: float = Field(..., ge=0.0, description="目标止盈价格")
    percentage: float = Field(..., ge=0.01, le=1.0, description="该点位建议平仓比例 (0.01 ~ 1.0)")
    description: str = Field(..., description="止盈依据说明")


class TradePlan(BaseModel):
    """建议交易计划与量化点位参数 (AI_SCHEMA properties.trade_plan)"""
    model_config = ConfigDict(populate_by_name=True)

    entry_range: List[float] = Field(..., min_length=2, max_length=2, description="建议入场价格区间 [最低入场价, 最高入场价]")
    take_profit_levels: List[TakeProfitLevel] = Field(..., min_length=1, max_length=4, description="分批止盈目标列表")
    stop_loss_price: float = Field(..., ge=0.0, description="硬止损价格")
    risk_reward_ratio: float = Field(..., ge=0.0, description="理论盈亏比 (R:R Ratio)")
    suggested_leverage: int = Field(..., ge=1, le=20, description="建议杠杆倍数 (1 ~ 20)")
    order_type: TradePlanOrderType = Field(TradePlanOrderType.LIMIT, description="建议委托类型：LIMIT, MARKET, TRIGGER_LIMIT")


class RiskAssessment(BaseModel):
    """风险评估与逻辑失效边界 (AI_SCHEMA properties.risk_assessment)"""
    model_config = ConfigDict(populate_by_name=True)

    key_risks: List[str] = Field(..., min_length=1, max_length=5, description="面临的核心风险因素列表")
    invalidation_condition: str = Field(..., description="交易逻辑彻底失效的技术形态或价格条件")
    max_holding_time_hours: float = Field(..., ge=0.5, le=720.0, description="建议最长持仓时间 (小时)")


class AIAnalysisResponse(BaseModel):
    """AI 大模型完整结构化研判响应 (100% 严格对齐 AI_SCHEMA.json 生产标准)"""
    model_config = ConfigDict(populate_by_name=True)

    analysis_id: str = Field(..., description="研判唯一标识符 (UUID v4)")
    symbol: str = Field(..., description="交易标的代码，例如 'BTC-USDT-SWAP'")
    timestamp: int = Field(..., description="研判生成时的 Unix 毫秒时间戳 (ms)")
    market_regime: MarketRegime = Field(..., description="宏观盘面结构分类")
    timeframe_analysis: TimeframeAnalysis = Field(..., description="多周期（15m, 1h, 4h, 1d）共振分析")
    derivatives_sentiment: DerivativesSentiment = Field(..., description="衍生品与市场情绪专项研判")
    signal: AISignal = Field(..., description="最终交易决策信号与置信度")
    trade_plan: TradePlan = Field(..., description="建议交易计划与量化点位参数")
    risk_assessment: RiskAssessment = Field(..., description="风险评估与逻辑失效边界")
    reasoning_summary: str = Field(..., description="精炼中文研判结论与核心逻辑摘要 (150字以内)")
    reasoning_details: str = Field(..., description="详尽的量化逻辑推导过程")

    # 扩展属性 (运行时网关填充)
    model_used: Optional[str] = Field(None, description="实际调用的 LLM 模型")
    latency_ms: Optional[int] = Field(0, description="推理耗时 (ms)")
    thinking_process: Optional[str] = Field(None, description="提取的思维链内容 (<think>...</think>)")


# =============================================================================
# 4. SSE 流式通信块模型 (SSE Stream Models)
# =============================================================================

class SSEStreamChunk(BaseModel):
    """SSE 流式传输数据块"""
    model_config = ConfigDict(populate_by_name=True)

    event: str = Field("chunk", description="事件类型: start, think, content, json_patch, done, error")
    data: Union[str, AIAnalysisResponse, Dict[str, Any]] = Field(..., description="数据载荷")
    reasoning_content: Optional[str] = Field(None, description="思维链增量文本")
    structured_output: Optional[AIAnalysisResponse] = Field(None, description="解析出的完整研判对象")
    error_message: Optional[str] = Field(None, description="错误信息")


# 别名兼容
AICoTStreamEvent = SSEStreamChunk
