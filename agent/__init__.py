"""
OKX-Dog LangGraph 机构级量化智能体决策大脑核心包
"""

from .registry import AgentRoleRegistry, BaseSpecialist, register_specialist
from .state import QuantTraderState, ThinkingStep
from .graph import create_quant_trader_graph
from .runner import QuantTraderAgentRunner
from .tools import (
    calculate_risk_reward_ratio,
    verify_hard_risk_compliance,
    derive_dynamic_atr_stops,
    calculate_orderbook_imbalance,
    evaluate_onchain_flow,
    calculate_kelly_position_size,
    evaluate_macro_event_risk,
    analyze_orderbook_liquidity,
)

__all__ = [
    "AgentRoleRegistry",
    "BaseSpecialist",
    "register_specialist",
    "QuantTraderState",
    "ThinkingStep",
    "create_quant_trader_graph",
    "QuantTraderAgentRunner",
    "calculate_risk_reward_ratio",
    "verify_hard_risk_compliance",
    "derive_dynamic_atr_stops",
    "calculate_orderbook_imbalance",
    "evaluate_onchain_flow",
    "calculate_kelly_position_size",
    "evaluate_macro_event_risk",
    "analyze_orderbook_liquidity",
]

from .evolution import (
    RewardEngine,
    SoftmaxGatingNetwork,
    AgentEvolutionManager,
)
