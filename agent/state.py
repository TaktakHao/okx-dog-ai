"""
OKX-Dog 量化交易员 Agent 运行状态机模型 (机构级多智能体分层状态机)
模块: okx-dog-ai/agent/state.py
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


def merge_dict_reducer(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """字典合并 Reducer (支持多个并发节点安全 Fan-Out 写入同一字典键)"""
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class ThinkingStep(TypedDict):
    """单阶段思考轨迹记录"""
    node: str
    stage_name: str
    thought: str
    timestamp_ms: int


class QuantTraderState(TypedDict, total=False):
    """
    LangGraph 资深量化交易员 StateGraph 全生命周期状态
    支持 6 大专家并行感知 + 红蓝对抗博弈 + 首席量化仲裁 + 严格硬风控反思闭环
    """
    # 1. 输入上下文与基础参数
    symbol: str
    current_price: float
    analysis_id: str
    timestamp: int
    market_snapshot: Dict[str, Any]
    account_balance_usdt: float
    risk_limits: Dict[str, Any]
    llm_config: Dict[str, Any]
    
    # 2. 感知层 6 大专业量化分析产物 (Perception Layer)
    market_regime: str                    # MacroScanner (宏观多周期共振)
    timeframe_analysis: Dict[str, Any]    # MacroScanner
    onchain_analysis: Dict[str, Any]      # OnChainAnalyst (链上巨鲸资金)
    quant_features: Dict[str, Any]        # QuantModeler (盘口微观与半凯利)
    derivatives_sentiment: Dict[str, Any] # DerivativesChecker (衍生品资金费率与OI)
    macro_event_risk: Dict[str, Any]      # MacroEventScanner (全球宏观日历与突发舆情)
    microstructure_data: Dict[str, Any]   # MicrostructureAnalyst (微观流动性与订单冲击)
    
    # 3. 动态角色扩展挂载槽 (使用 merge_dict_reducer 支持并发多节点无冲突挂载)
    specialist_outputs: Annotated[Dict[str, Any], merge_dict_reducer]
    
    # 4. 对抗层红蓝辩论产物 (Adversarial Debate Layer)
    bull_opinion: Dict[str, Any]          # 多头进攻辩护专家意见
    bear_opinion: Dict[str, Any]          # 空头风控红队挑刺专家意见
    debate_summary: str                   # 多空对抗陈述摘要
    
    # 5. 首席仲裁与多因子策略规划产物 (Arbitration & Strategy Planning Layer)
    consensus_score: int                  # 多智能体仲裁共识分 (0~100)
    is_approved_by_arbiter: bool          # 仲裁准入红线是否放行 (共识分>=75 或 <=35)
    signal: Dict[str, Any]                # 交易信号 (action, confidence, urgency)
    trade_plan: Dict[str, Any]            # 交易计划 (entry_range, tp, sl, rr, leverage, order_type)
    risk_assessment: Dict[str, Any]       # 风控评估
    reasoning_summary: str
    reasoning_details: str
    
    # 6. 硬风控与反思审查控制状态 (Risk Critic & Reflection Loop)
    risk_passed: bool                     # 是否通过硬风控审查
    risk_critique: Optional[str]          # 风控专家批评意见
    critique_count: int                   # 反思重试计数器 (最大 2 轮)
    force_fallback_hold: bool             # 是否触发强制安全熔断
    
    # 7. 思考链与多阶段推理轨迹 (Reducer 自动累加)
    thinking_steps: Annotated[List[ThinkingStep], operator.add]
    
    # 8. 最终合规契约输出 (Formatting Layer)
    final_response: Optional[Dict[str, Any]]
    model_used: Optional[str]
    latency_ms: Optional[int]
    error: Optional[str]
