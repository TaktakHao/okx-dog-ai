"""
OKX-Dog AI 契约与 Few-Shot 样本验证测试
模块: okx-dog-ai/tests/test_contract_and_few_shots.py
"""

import json
import pytest
from okx_dog_ai.parser import RobustJSONParser
from okx_dog_ai.schemas import AIAnalysisResponse, SignalAction


FEW_SHOT_1_JSON = """
{
  "analysis_id": "c1f7a8b2-3e4d-4a11-8f92-5b9e7c102a01",
  "symbol": "BTC-USDT-SWAP",
  "timestamp": 1755216000000,
  "market_regime": "TRENDING_UP",
  "timeframe_analysis": {
    "tf_15m": {
      "trend": "BULLISH",
      "key_indicators_summary": "EMA20>50>200发散多头排列；RSI=58.2健康运行；布林通道向上倾斜，价格稳定在中轨上方。",
      "support_level": 94380.0,
      "resistance_level": 95200.0
    },
    "tf_1h": {
      "trend": "BULLISH",
      "key_indicators_summary": "EMA20上穿EMA50金叉，MACD零轴上方红柱再次放量，RSI=63.5动能强劲。",
      "support_level": 94200.0,
      "resistance_level": 96500.0
    },
    "tf_4h": {
      "trend": "BULLISH",
      "key_indicators_summary": "成功放量突破94200平台阻力并缩量回踩确认，均线组呈标准多头排列，上升结构完好。",
      "support_level": 94200.0,
      "resistance_level": 97000.0
    },
    "tf_1d": {
      "trend": "BULLISH",
      "key_indicators_summary": "日线级别处于大上升通道主升浪，EMA200位于84000提供坚实宏观底部支撑。",
      "support_level": 91500.0,
      "resistance_level": 98500.0
    }
  },
  "derivatives_sentiment": {
    "funding_rate_bias": "MODERATE_POSITIVE",
    "open_interest_interpretation": "价格破位上行伴随全网持仓量(OI)稳步增长6.8%，体现主力资金主动建仓做多意愿强烈，行情非虚假轧空。",
    "long_short_ratio_state": "多空人数比为1.12，大户持仓比1.25，多头持仓结构健康，无散户极端狂热追多情绪。",
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
      {
        "price": 96800.0,
        "percentage": 0.5,
        "description": "TP1: 触及前期高点强阻力区平仓50%锁定利润，并将剩余仓位止损上移至开仓价保本。"
      },
      {
        "price": 98500.0,
        "percentage": 0.5,
        "description": "TP2: 日线波段斐波那契1.618扩展目标位，全平离场。"
      }
    ],
    "stop_loss_price": 93800.0,
    "risk_reward_ratio": 2.75,
    "suggested_leverage": 5,
    "order_type": "LIMIT"
  },
  "risk_assessment": {
    "key_risks": [
      "美股开盘时段可能的宏观流动性扰动",
      "若94200回踩被放量下穿则演变为假突破陷阱"
    ],
    "invalidation_condition": "若1小时K线实体收盘跌破93800支撑位，则突破多头逻辑彻底失效，应立即止损离场。",
    "max_holding_time_hours": 24.0
  },
  "reasoning_summary": "4H突破关键阻力位并回踩确认，1H金叉共振，OI增仓且费率健康，建议在94500-94700分批做多，盈亏比2.75。",
  "reasoning_details": "日线与4小时周期大趋势向上，4H放量突破94200箱体上沿后出现缩量整理，构成经典突破回踩确认形态。1小时级别EMA均线呈多头排列，MACD零轴上方再次金叉放量，RSI=63.5动能充足。衍生品端OI上升6.8%验证增量资金入场，资金费率处于0.008%健康区间。综合建议在94500-94700挂限价多单，止损设于破位结构下方的93800，第一止盈看96800，理论盈亏比高达2.75。"
}
"""

FEW_SHOT_2_JSON = """
{
  "analysis_id": "e8a2b3c4-5d6e-4f78-90ab-cdef12345678",
  "symbol": "ETH-USDT-SWAP",
  "timestamp": 1755219600000,
  "market_regime": "TRENDING_DOWN",
  "timeframe_analysis": {
    "tf_15m": {
      "trend": "BEARISH",
      "key_indicators_summary": "冲高受阻收出长上影阴线Pinbar；RSI快速自76超买区向下拐头跌破50；短期跌破EMA20中轨支撑。",
      "support_level": 3430.0,
      "resistance_level": 3495.0
    },
    "tf_1h": {
      "trend": "OVERBOUGHT",
      "key_indicators_summary": "出现清晰的RSI与MACD顶背离（价格创3495新高，RSI高点由78降至64），MACD死叉在即。",
      "support_level": 3390.0,
      "resistance_level": 3500.0
    },
    "tf_4h": {
      "trend": "NEUTRAL_CHOPPY",
      "key_indicators_summary": "受制于3500整数大关与布林带上轨压制，多次上冲未果形成三顶结构，上行阻力极大。",
      "support_level": 3320.0,
      "resistance_level": 3500.0
    },
    "tf_1d": {
      "trend": "BULLISH",
      "key_indicators_summary": "日线处于高位震荡整理期，RSI=68接近超买边界，存在强烈的回踩均线修复需求。",
      "support_level": 3200.0,
      "resistance_level": 3600.0
    }
  },
  "derivatives_sentiment": {
    "funding_rate_bias": "EXTREME_POSITIVE",
    "open_interest_interpretation": "高位持仓量居高不下且资金费率高达+0.075%，呈现极端的多头拥挤状态，多头持仓成本过高极易引发连锁踩踏平仓。",
    "long_short_ratio_state": "散户多空人数比飙升至2.85，大户账户比逆向转空，盘口买单虽大但无法推高价格，显示主力正在暗中派发。",
    "sentiment_score": -0.70
  },
  "signal": {
    "action": "SELL_SHORT",
    "confidence": 0.84,
    "urgency": "MEDIUM"
  },
  "trade_plan": {
    "entry_range": [3480.0, 3495.0],
    "take_profit_levels": [
      {
        "price": 3390.0,
        "percentage": 0.6,
        "description": "TP1: 触及1小时周期EMA50均线与前期平台支撑，平仓60%锁定利润并移止损至开仓价。"
      },
      {
        "price": 3280.0,
        "percentage": 0.4,
        "description": "TP2: 4小时布林带下轨及关键需求区，全平结利。"
      }
    ],
    "stop_loss_price": 3525.0,
    "risk_reward_ratio": 2.37,
    "suggested_leverage": 4,
    "order_type": "LIMIT"
  },
  "risk_assessment": {
    "key_risks": [
      "主力在3500关口发起最后一波快速诱多插针扫损",
      "BTC若出现非理性暴力拉升可能带动ETH跟风突破"
    ],
    "invalidation_condition": "若15分钟收盘站稳3525上方，则顶背离反转逻辑证伪，多头趋势延续，必须坚决止损。",
    "max_holding_time_hours": 16.0
  },
  "reasoning_summary": "1H出现严重量价顶背离，资金费率+0.075%极端过热，散户扎堆做多，建议在3480-3495分批做空，目标3390，盈亏比2.37。",
  "reasoning_details": "价格在3500关口连续受阻，1小时图表呈现显著的RSI与MACD顶背离特征。资金费率攀升至+0.075%的历史极端值，且多空人数比高达2.85，表明散户情绪狂热追多，多头拥挤度极高。盘口观察到虽然买单厚重但价格滞涨，符合主力吸收流动性派发特征。建议在3480-3495区间限价开空，硬止损设于3525，首要目标3390，预期盈亏比2.37。"
}
"""

FEW_SHOT_3_JSON = """
{
  "analysis_id": "a9b8c7d6-e5f4-4a3b-8c2d-1e0f9a8b7c6d",
  "symbol": "SOL-USDT-SWAP",
  "timestamp": 1755223200000,
  "market_regime": "RANGING",
  "timeframe_analysis": {
    "tf_15m": {
      "trend": "NEUTRAL_CHOPPY",
      "key_indicators_summary": "EMA20/50/200均线严重缠绕粘合；布林带极限收口至1.2%（即将变盘）；RSI=49.8处于绝对中性无方向状态。",
      "support_level": 180.5,
      "resistance_level": 184.5
    },
    "tf_1h": {
      "trend": "NEUTRAL_CHOPPY",
      "key_indicators_summary": "价格处于175-190宽幅震荡箱体中轨，MACD柱体在零轴附近微幅震荡无动量，量能极度萎缩。",
      "support_level": 178.0,
      "resistance_level": 186.0
    },
    "tf_4h": {
      "trend": "NEUTRAL_CHOPPY",
      "key_indicators_summary": "中周期处于无趋势震荡筑底阶段，上下轨边界清晰（支撑175，阻力190），中间区域缺乏博弈优势。",
      "support_level": 175.0,
      "resistance_level": 190.0
    },
    "tf_1d": {
      "trend": "NEUTRAL_CHOPPY",
      "key_indicators_summary": "日线EMA20横向走平，整体处于大级别三角收敛末端，等待方向选择。",
      "support_level": 170.0,
      "resistance_level": 195.0
    }
  },
  "derivatives_sentiment": {
    "funding_rate_bias": "NEUTRAL",
    "open_interest_interpretation": "全网未平仓合约量(OI)下降4.2%，显示场内资金交投平淡，主力处于静默观望状态，无方向性资金沉淀。",
    "long_short_ratio_state": "多空人数比为1.02，散户与大户持仓比例均衡，市场缺乏明确情绪倾向。",
    "sentiment_score": 0.0
  },
  "signal": {
    "action": "HOLD_WAIT",
    "confidence": 0.25,
    "urgency": "LOW"
  },
  "trade_plan": {
    "entry_range": [0.0, 0.0],
    "take_profit_levels": [
      {
        "price": 0.0,
        "percentage": 1.0,
        "description": "当前无持仓计划，不设止盈目标。"
      }
    ],
    "stop_loss_price": 0.0,
    "risk_reward_ratio": 0.0,
    "suggested_leverage": 1,
    "order_type": "LIMIT"
  },
  "risk_assessment": {
    "key_risks": [
      "箱体中轨盲目开仓极易遭遇双向扫损",
      "布林带收口后可能出现无量假突破诱多/诱空"
    ],
    "invalidation_condition": "若价格放量有效突破190.00阻力或放量跌破175.00支撑，则震荡状态打破，届时应重新发起研判。",
    "max_holding_time_hours": 48.0
  },
  "reasoning_summary": "多周期均线全面粘合，处于175-190箱体正中央，量能萎缩且OI下降，无任何交易优势，坚决保持空仓观望。",
  "reasoning_details": "15分钟与1小时级别均线呈严重粘合状态，RSI指标停留在50中轴附近，成交量显著萎缩。4小时级别上，价格正处于175-190震荡箱体的正中轴位置（182.50），无论做多或做空，距离上下边界距离均等，无法构建盈亏比大于1.5的有效交易计划。衍生品端OI下降4.2%且资金费率完全中性，反映主力资金离场观望。建议严格遵守风控纪律，保持空仓，静待价格测试箱体边界或放量真突破后再做入场规划。"
}
"""


def test_few_shot_1_validation():
    """验证 Few-Shot 样本 1 (看多多周期共振)"""
    resp = RobustJSONParser.parse(FEW_SHOT_1_JSON)
    assert isinstance(resp, AIAnalysisResponse)
    assert resp.symbol == "BTC-USDT-SWAP"
    assert resp.signal.action == SignalAction.BUY_LONG
    assert resp.signal.confidence == 0.88
    assert resp.trade_plan.risk_reward_ratio == 2.75


def test_few_shot_2_validation():
    """验证 Few-Shot 样本 2 (看空背离反转)"""
    resp = RobustJSONParser.parse(FEW_SHOT_2_JSON)
    assert isinstance(resp, AIAnalysisResponse)
    assert resp.symbol == "ETH-USDT-SWAP"
    assert resp.signal.action == SignalAction.SELL_SHORT
    assert resp.signal.confidence == 0.84
    assert resp.trade_plan.stop_loss_price == 3525.0


def test_few_shot_3_validation():
    """验证 Few-Shot 样本 3 (震荡观望)"""
    resp = RobustJSONParser.parse(FEW_SHOT_3_JSON)
    assert isinstance(resp, AIAnalysisResponse)
    assert resp.symbol == "SOL-USDT-SWAP"
    assert resp.signal.action == SignalAction.HOLD_WAIT
    assert resp.signal.confidence == 0.25
    assert resp.trade_plan.stop_loss_price == 0.0
