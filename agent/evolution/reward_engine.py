"""
OKX-Dog 多智能体复合强化奖励计算引擎 (Multi-Agent Reward Engine)
模块: okx-dog-ai/agent/evolution/reward_engine.py
角色: AI 与量化算法工程师 (agency-ai-engineer)

核心功能:
1. 在每笔交易平仓或研判生命周期到期时，执行逐 Agent 多维强化奖励分解；
2. 结合实现盈亏比 (Realized R:R)、MFE/MAE 走势偏移预测精度、风控避险贡献度与回撤惩罚；
3. 输出结构化奖励事件 MultiAgentRewardOutcome，供门控网络与员工档案系统实时增量学习。
"""

import logging
import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("okx_dog.ai.evolution.reward_engine")


class AgentRewardBreakdown(BaseModel):
    """单个 Agent 员工的奖励分拆解"""
    role_id: str = Field(..., description="Agent 员工角色 ID")
    role_name: str = Field(..., description="员工全称")
    stance: str = Field(..., description="当时陈述立场 (BULLISH/BEARISH/NEUTRAL)")
    confidence: float = Field(..., description="当时置信度 (0.0 ~ 1.0)")
    pnl_contribution: float = Field(default=0.0, description="盈亏比贡献得分")
    accuracy_contribution: float = Field(default=0.0, description="走势方向预测精度得分")
    risk_defense_bonus: float = Field(default=0.0, description="风控排雷避险立功奖励")
    drawdown_penalty: float = Field(default=0.0, description="逆势与深回撤惩罚分")
    total_reward: float = Field(..., description="本次单笔综合强化奖励分")
    achievement_note: Optional[str] = Field(default=None, description="本次立功/失误简要归因说明")


class MultiAgentRewardOutcome(BaseModel):
    """单次平仓/复盘后的多角色奖励结算结果"""
    trade_id: str = Field(..., description="对应交易或研判 ID")
    symbol: str = Field(..., description="标的代码")
    realized_pnl: float = Field(..., description="最终实现盈亏 USDT")
    realized_rr: float = Field(..., description="实现盈亏比")
    direction_correct: bool = Field(..., description="方向是否正确预测")
    breakdowns: List[AgentRewardBreakdown] = Field(default_factory=list, description="各角色的奖惩明细")
    summary: str = Field(..., description="结算评语摘要")


class RewardEngine:
    """复合强化奖励计算器"""

    @classmethod
    def evaluate_trade_outcome(
        cls,
        trade_id: str,
        symbol: str,
        pos_side: str,  # "LONG" / "SHORT" / "FLAT"
        entry_price: float,
        exit_price: float,
        sl_price: float,
        realized_pnl: float,
        mfe_pct: float,
        mae_pct: float,
        agent_opinions: Dict[str, Dict[str, Any]],
        is_crisis_defended: bool = False
    ) -> MultiAgentRewardOutcome:
        """
        计算多角色强化学习奖励
        """
        # 1. 计算实际盈亏比 Realized R:R
        risk_dist = abs(entry_price - sl_price) if abs(entry_price - sl_price) > 1e-6 else entry_price * 0.01
        price_delta = (exit_price - entry_price) if pos_side == "LONG" else (entry_price - exit_price)
        raw_rr = price_delta / risk_dist
        clamped_rr = max(-2.0, min(4.0, raw_rr))

        # 2. 预测精度得分 (MFE / MAE 归一化)
        total_move = abs(mfe_pct) + abs(mae_pct) + 1e-5
        accuracy_score = (abs(mfe_pct) - abs(mae_pct)) / total_move  # [-1.0, 1.0]

        is_win = realized_pnl > 0 or clamped_rr > 0.3
        is_dir_correct = (accuracy_score > 0) if pos_side != "FLAT" else True

        breakdowns: List[AgentRewardBreakdown] = []

        for role_id, op in agent_opinions.items():
            role_name = op.get("role_name", role_id)
            stance = op.get("stance", "NEUTRAL").upper()
            conf = float(op.get("confidence", 0.6))

            pnl_score = 0.0
            acc_score = 0.0
            risk_bonus = 0.0
            dd_penalty = 0.0
            note = None

            # A. 多头分析师 (Bull Specialist)
            if "bull" in role_id.lower():
                if pos_side == "LONG":
                    pnl_score = clamped_rr * conf * 10.0
                    acc_score = accuracy_score * conf * 8.0
                    if is_win:
                        note = f"精准捕捉多头主升浪，贡献 +{clamped_rr:.2f}R 收益"
                    else:
                        dd_penalty = abs(mae_pct) * 2.0
                        note = "多头研判遭遇回踩，受到轻微回撤扣分"
                elif pos_side == "SHORT" and stance == "BULLISH":
                    pnl_score = -5.0 * conf
                    note = "逆势看多导致误判"

            # B. 空头风控挑刺官 (Bear Critic)
            elif "bear" in role_id.lower():
                if pos_side == "SHORT":
                    pnl_score = clamped_rr * conf * 10.0
                    acc_score = accuracy_score * conf * 8.0
                    if is_win:
                        note = f"精准看空狙击下行波段，贡献 +{clamped_rr:.2f}R 收益"
                elif is_crisis_defended or (pos_side == "LONG" and not is_win):
                    # 在多头亏损或系统识别插针时，空头的挑刺预警获得立功奖励
                    risk_bonus = 12.0 * conf
                    note = "🛡️ 成功挑刺提示高位风险，获排雷立功加分"
                elif is_win and stance == "BEARISH":
                    pnl_score = -3.0 * conf
                    note = "在顺畅多头行情中偏保守挑刺"

            # C. 宏观与突发情报员 (Macro News Scout)
            elif "macro" in role_id.lower() or "news" in role_id.lower():
                acc_score = accuracy_score * conf * 5.0
                if is_crisis_defended:
                    risk_bonus = 10.0 * conf
                    note = "🌐 宏观情报及时预警黑天鹅风险，保护本金安全"
                elif is_win:
                    pnl_score = clamped_rr * conf * 4.0
                    note = "宏观背景研判与趋势方向一致"

            # D. 链上主力侦察员 (On-Chain Analyst)
            elif "onchain" in role_id.lower() or "chain" in role_id.lower():
                acc_score = accuracy_score * conf * 6.0
                if is_win:
                    pnl_score = clamped_rr * conf * 6.0
                    note = "⛓️ 准确捕捉交易所大额资金异动与主力建仓信号"
                elif is_crisis_defended:
                    risk_bonus = 8.0 * conf
                    note = "⛓️ 提前监测到巨鲸异常充币抛售信号，保护本金安全"

            # E. 微观盘口狙击手 (Micro Sniper)
            elif "micro" in role_id.lower() or "sniper" in role_id.lower():
                acc_score = accuracy_score * conf * 8.0
                if is_win:
                    pnl_score = clamped_rr * conf * 6.0
                    note = "⚡ 盘口买卖失衡捕捉精准，入场摩擦损耗极小"
                else:
                    dd_penalty = 2.0

            # E. 首席量化仲裁官 (Chief Arbiter)
            elif "arbiter" in role_id.lower() or "chief" in role_id.lower():
                pnl_score = clamped_rr * 8.0
                acc_score = accuracy_score * 6.0
                if is_win:
                    note = "⚖️ 把关决策严明，签发高质量交易指令"
                else:
                    note = "未能有效阻断亏损单，扣减统筹积分"

            # 通用兜底
            else:
                pnl_score = clamped_rr * conf * 5.0
                acc_score = accuracy_score * conf * 5.0

            total_reward = round(pnl_score + acc_score + risk_bonus - dd_penalty, 2)

            breakdowns.append(
                AgentRewardBreakdown(
                    role_id=role_id,
                    role_name=role_name,
                    stance=stance,
                    confidence=conf,
                    pnl_contribution=round(pnl_score, 2),
                    accuracy_contribution=round(acc_score, 2),
                    risk_defense_bonus=round(risk_bonus, 2),
                    drawdown_penalty=round(dd_penalty, 2),
                    total_reward=total_reward,
                    achievement_note=note or "平稳执行既定研判流程"
                )
            )

        summary_text = f"交易平仓结算完成 (盈亏: {realized_pnl:+.2f} USDT, 盈亏比: {clamped_rr:+.2f}R)。各决策员工强化奖励已完成结算。"

        return MultiAgentRewardOutcome(
            trade_id=trade_id,
            symbol=symbol,
            realized_pnl=round(realized_pnl, 2),
            realized_rr=round(clamped_rr, 2),
            direction_correct=is_dir_correct,
            breakdowns=breakdowns,
            summary=summary_text
        )
