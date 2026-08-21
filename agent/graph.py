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
    onchain_analyst_node,
    quant_modeler_node,
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
    创建并编译 OKX-Dog 资深量化交易员 StateGraph 决策图 (4 专家并行研判 + 反思回路)
    """
    builder = StateGraph(QuantTraderState)

    # 1. 注册核心节点
    builder.add_node("macro_scanner", macro_trend_scan_node)
    builder.add_node("onchain_analyst", onchain_analyst_node)
    builder.add_node("quant_modeler", quant_modeler_node)
    builder.add_node("derivatives_checker", derivatives_sentiment_node)
    builder.add_node("strategy_planner", strategy_planning_node)
    builder.add_node("risk_critic", risk_critic_node)
    builder.add_node("formatter", response_formatter_node)

    # 2. 编排 4 专家并行分发与汇聚 (Fan-Out / Fan-In)
    # START 并行分发至 4 大分析专家
    builder.add_edge(START, "macro_scanner")
    builder.add_edge(START, "onchain_analyst")
    builder.add_edge(START, "quant_modeler")
    builder.add_edge(START, "derivatives_checker")

    # 4 大专家研判产物汇聚至策略规划与多因子融合中枢
    builder.add_edge("macro_scanner", "strategy_planner")
    builder.add_edge("onchain_analyst", "strategy_planner")
    builder.add_edge("quant_modeler", "strategy_planner")
    builder.add_edge("derivatives_checker", "strategy_planner")

    # 3. 策略规划流转至硬风控审查
    builder.add_edge("strategy_planner", "risk_critic")

    # 4. 编排硬风控反思条件边 (Critic & Reflection Loop)
    builder.add_conditional_edges(
        "risk_critic",
        route_after_risk_critic,
        {
            "strategy_planner": "strategy_planner",
            "formatter": "formatter",
        },
    )

    # 5. 终结边
    builder.add_edge("formatter", END)

    # 6. 编译执行图
    graph = builder.compile()
    logger.info("OKX-Dog LangGraph 4 专家并行决策状态机成功编译完成")
    return graph
