"""
硬风控与数学盈亏比审查节点 (RiskCritic)
模块: okx-dog-ai/agent/nodes/risk_critic.py

职责:
1. 严格计算理论盈亏比 (R:R Ratio)，强制要求 R:R >= 1.5 且方向逻辑无悖论。
2. 拦截违规杠杆与超额风险敞口 (verify_hard_risk_compliance)。
3. 若发现缺陷且重试次数 < 2，生成精确批评反馈 (Risk Critique) 触发条件边回退重调；
4. 若超过最大重试次数，执行绝对安全熔断机制，强制收敛为 HOLD_WAIT。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from ..state import QuantTraderState, ThinkingStep
from ..tools import calculate_risk_reward_ratio, verify_hard_risk_compliance

logger = logging.getLogger("okx_dog.ai.agent.risk_critic")


async def risk_critic_node(state: QuantTraderState) -> Dict[str, Any]:
    """
    LangGraph Node: 硬风控与数学盈亏比审查
    """
    logger.info("执行 Node 4: 硬风控与数学盈亏比审查...")
    now_ms = int(time.time() * 1000)
    signal = state.get("signal", {})
    action = signal.get("action", "HOLD_WAIT")
    trade_plan = state.get("trade_plan", {})
    risk_limits = state.get("risk_limits", {})
    account_balance = float(state.get("account_balance_usdt", 1000.0))
    critique_count = int(state.get("critique_count", 0))

    violations: List[str] = []

    # 仅对真实开仓操作进行严密数学审查
    if action in ["BUY_LONG", "SELL_SHORT"]:
        entry_range = trade_plan.get("entry_range", [])
        stop_loss = float(trade_plan.get("stop_loss_price", 0.0))
        tp_levels = trade_plan.get("take_profit_levels", [])
        leverage = int(trade_plan.get("suggested_leverage", 1))

        # 1. 净盈亏比 (扣除手续费与滑点摩擦) 与点位方向审查
        net_rr_ratio, rr_err = calculate_risk_reward_ratio(
            action=action,
            entry_range=entry_range,
            stop_loss_price=stop_loss,
            take_profit_levels=tp_levels,
        )

        if rr_err:
            violations.append(f"点位逻辑错误: {rr_err}")
        elif net_rr_ratio < 1.5:
            violations.append(
                f"扣除手续费与滑点摩擦后的真实净盈亏比不足 1.5 (当前计算为 {net_rr_ratio:.2f})，不符合实盘进场底线要求"
            )
        else:
            # 回填精确校验后的净盈亏比
            trade_plan["risk_reward_ratio"] = net_rr_ratio

        # 2. 硬风控边界审查 (杠杆与最大亏损)
        entry_avg = (entry_range[0] + entry_range[1]) / 2.0 if len(entry_range) == 2 else 0.0
        risk_errs = verify_hard_risk_compliance(
            action=action,
            entry_price=entry_avg,
            stop_loss_price=stop_loss,
            suggested_leverage=leverage,
            account_balance_usdt=account_balance,
            risk_limits=risk_limits,
        )
        violations.extend(risk_errs)

    # 3. 审查结果裁决与状态机决策
    if violations:
        critique_msg = "；".join(violations)
        if critique_count < 2:
            logger.warning(
                f"风控审查未通过 [轮次 #{critique_count + 1}]: {critique_msg}，触发反思回退..."
            )
            thought_text = (
                f"【硬风控审查被拦截】(第 {critique_count + 1} 轮): 发现以下违规项: {critique_msg}。"
                f"将反思意见注入状态机，触发条件边回退至 StrategyPlanner 重新计算点位。"
            )
            thinking_step: ThinkingStep = {
                "node": "RiskCritic",
                "stage_name": "硬风控与数学盈亏比审查 (拦截反思)",
                "thought": thought_text,
                "timestamp_ms": now_ms,
            }
            return {
                "risk_passed": False,
                "risk_critique": critique_msg,
                "critique_count": critique_count + 1,
                "thinking_steps": [thinking_step],
            }
        else:
            # 达到最大重试次数，执行强制安全熔断
            logger.error(
                f"风控审查在 2 次反思后依然未通过: {critique_msg}，执行绝对安全熔断 -> 强制 HOLD_WAIT"
            )
            thought_text = (
                f"【硬风控强制熔断】在经历了 2 轮反思调整后点位依然无法满足严格风控标准 ({critique_msg})。"
                f"为保护资金安全，强制将信号转换为 HOLD_WAIT (空仓观望)，置信度归零。"
            )
            fallback_signal = {
                "action": "HOLD_WAIT",
                "confidence": 0.0,
                "urgency": "LOW",
            }
            thinking_step: ThinkingStep = {
                "node": "RiskCritic",
                "stage_name": "硬风控与数学盈亏比审查 (安全熔断)",
                "thought": thought_text,
                "timestamp_ms": now_ms,
            }
            return {
                "risk_passed": True,
                "force_fallback_hold": True,
                "signal": fallback_signal,
                "risk_critique": None,
                "thinking_steps": [thinking_step],
            }

    # 4. 审查完全通过
    thought_text = (
        f"【硬风控审查 100% 通过】：交易计划动作={action}，盈亏比={trade_plan.get('risk_reward_ratio', 1.5):.2f} (>= 1.5)，"
        f"杠杆={trade_plan.get('suggested_leverage', 1)}x，止损距离与单笔风险敞口均在安全合规范围内。"
    )
    thinking_step: ThinkingStep = {
        "node": "RiskCritic",
        "stage_name": "硬风控与数学盈亏比审查 (合规放行)",
        "thought": thought_text,
        "timestamp_ms": now_ms,
    }

    return {
        "risk_passed": True,
        "risk_critique": None,
        "trade_plan": trade_plan,
        "thinking_steps": [thinking_step],
    }
