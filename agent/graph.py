"""
OKX-Dog LangGraph 资深量化交易员图编排与状态机构建器
模块: okx-dog-ai/agent/graph.py
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from .nodes import (
    derivatives_sentiment_node,
    macro_trend_scan_node,
    response_formatter_node,
    risk_critic_node,
    strategy_planning_node,
)
from .state import QuantTraderState

logger = logging.getLogger("okx_dog.ai.agent.graph")


def route_after_risk_critic(state: QuantTraderState) -> Literal["strategy_planner", "formatter"]:
    """
    风控审查后的条件边路由：
    - 若审查未通过 (risk_passed == False)，带着批评意见回退到 strategy_planner 重新优化点位；
    - 若审查通过 (risk_passed == True)，流转至 formatter 封装最终契约。
    """
    risk_passed = state.get("risk_passed", True)
    if not risk_passed:
        logger.info("条件边路由: 风控审查未通过，回退至 strategy_planner 进行自适应反思调整...")
        return "strategy_planner"
    logger.info("条件边路由: 风控审查通过，流转至 formatter 进行标准契约输出...")
    return "formatter"


def create_quant_trader_graph() -> StateGraph:
    """
    创建并编译 OKX-Dog 资深量化交易员 StateGraph 决策图
    """
    builder = StateGraph(QuantTraderState)

    # 1. 注册核心节点
    builder.add_node("macro_scanner", macro_trend_scan_node)
    builder.add_node("derivatives_checker", derivatives_sentiment_node)
    builder.add_node("strategy_planner", strategy_planning_node)
    builder.add_node("risk_critic", risk_critic_node)
    builder.add_node("formatter", response_formatter_node)

    # 2. 编排基础有向边
    builder.add_edge(START, "macro_scanner")
    builder.add_edge("macro_scanner", "derivatives_checker")
    builder.add_edge("derivatives_checker", "strategy_planner")
    builder.add_edge("strategy_planner", "risk_critic")

    # 3. 编排硬风控反思条件边 (Critic & Reflection Loop)
    builder.add_conditional_edges(
        "risk_critic",
        route_after_risk_critic,
        {
            "strategy_planner": "strategy_planner",
            "formatter": "formatter",
        },
    )

    # 4. 终结边
    builder.add_edge("formatter", END)

    # 5. 编译执行图
    graph = builder.compile()
    logger.info("OKX-Dog LangGraph 资深量化交易员状态机成功编译完成")
    return graph
