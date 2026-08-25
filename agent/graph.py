"""
OKX-Dog LangGraph 资深量化交易员分层图编排与状态机构建器
模块: okx-dog-ai/agent/graph.py
角色: 多智能体交易协同架构师 (agency-multi-agent-systems-architect)

分层决策拓扑:
1. 感知层 (Perception Layer): 动态加载注册表中所有感知专家 (Fan-Out 并行执行)
2. 对抗层 (Adversarial Layer): 红蓝对抗博弈辩论 (Fan-In 汇聚)
3. 仲裁规划层 (Arbitration Layer): 首席量化仲裁与多因子 ATR 点位规划
4. 风控审计层 (Risk & Reflection Layer): 硬风控审查与自适应反思回退回路
5. 契约格式化层 (Formatting Layer): 标准契约收敛与思维链输出 -> END
"""

from __future__ import annotations

import logging
try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    # 允许在没有外部 langgraph 库的基础轻量环境下安全导入
    END, START, StateGraph = "END", "START", object

from .nodes import (
    adversarial_debate_node,
    response_formatter_node,
    risk_critic_node,
    strategy_planning_node,
)
from .registry import AgentRoleRegistry
from .state import QuantTraderState

logger = logging.getLogger("okx_dog.ai.agent.graph")


def route_after_risk_critic(state: QuantTraderState) -> Literal["strategy_planner", "formatter"]:
    """
    风控审查后的条件边路由：
    - 若审查未通过 (risk_passed == False)，带着批评意见回退到 strategy_planner 重新自适应优化点位；
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
    创建并编译 OKX-Dog 机构级量化交易员 StateGraph 分层决策状态机
    """
    builder = StateGraph(QuantTraderState)

    # 1. 从注册中心动态获取感知层专家 (Perception Layer)
    perception_specialists = AgentRoleRegistry.get_specialists_by_layer("perception")
    if not perception_specialists:
        raise RuntimeError("AgentRoleRegistry 中未发现任何已注册的感知层专家！")

    perception_node_names = []
    for spec in perception_specialists:
        node_name = spec.name
        builder.add_node(node_name, spec.get_node_function())
        perception_node_names.append(node_name)
        logger.info("已将感知层专家节点 [%s] 并入决策图", node_name)

    # 2. 注册对抗层、仲裁规划层、风控层与格式化层节点
    builder.add_node("adversarial_debater", adversarial_debate_node)
    builder.add_node("strategy_planner", strategy_planning_node)
    builder.add_node("risk_critic", risk_critic_node)
    builder.add_node("formatter", response_formatter_node)

    # 3. 编排感知层并行分发 (Fan-Out)
    # START 并行流向所有已注册的感知专家
    for node_name in perception_node_names:
        builder.add_edge(START, node_name)

    # 4. 感知层产物汇聚至红蓝对抗博弈中枢 (Fan-In)
    for node_name in perception_node_names:
        builder.add_edge(node_name, "adversarial_debater")

    # 5. 红蓝对抗辩论产物流转至首席量化仲裁与规划中枢
    builder.add_edge("adversarial_debater", "strategy_planner")

    # 6. 策略规划流转至硬风控审查
    builder.add_edge("strategy_planner", "risk_critic")

    # 7. 编排硬风控反思条件边 (Critic & Reflection Loop)
    builder.add_conditional_edges(
        "risk_critic",
        route_after_risk_critic,
        {
            "strategy_planner": "strategy_planner",
            "formatter": "formatter",
        },
    )

    # 8. 终结边
    builder.add_edge("formatter", END)

    # 9. 编译执行图
    graph = builder.compile()
    logger.info("OKX-Dog LangGraph 机构级分层决策状态机成功编译完成 (感知专家数=%d)", len(perception_node_names))
    return graph
