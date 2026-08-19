"""
OKX-Dog 量化交易员 Agent 运行状态机模型
模块: okx-dog-ai/agent/state.py
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class ThinkingStep(TypedDict):
    """单阶段思考轨迹记录"""
    node: str
    stage_name: str
    thought: str
    timestamp_ms: int


class QuantTraderState(TypedDict, total=False):
    """
    LangGraph 资深量化交易员 StateGraph 全生命周期状态
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
    
    # 2. 宏观多周期研判产物 (Node 1: MacroScanner)
    market_regime: str
    timeframe_analysis: Dict[str, Any]
    
    # 3. 衍生品与微观结构产物 (Node 2: DerivativesChecker)
    derivatives_sentiment: Dict[str, Any]
    
    # 4. 交易策略与点位规划初稿 (Node 3: StrategyPlanner)
    signal: Dict[str, Any]
    trade_plan: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    reasoning_summary: str
    reasoning_details: str
    
    # 5. 硬风控与反思审查状态 (Node 4: RiskCritic)
    risk_passed: bool
    risk_critique: Optional[str]
    critique_count: int
    force_fallback_hold: bool
    
    # 6. 思考链与多阶段推理轨迹 (Reducer 累加)
    thinking_steps: Annotated[List[ThinkingStep], operator.add]
    
    # 7. 最终合规契约输出 (Node 5: Formatter)
    final_response: Optional[Dict[str, Any]]
    model_used: Optional[str]
    latency_ms: Optional[int]
    error: Optional[str]
