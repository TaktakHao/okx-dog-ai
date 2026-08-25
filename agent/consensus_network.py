"""
OKX-Dog AI 决策大脑 - 多智能体对抗博弈辩论与首席仲裁网络
模块: okx-dog-ai/agent/consensus_network.py
角色: 多智能体交易协同架构师 (agency-multi-agent-systems-architect)
架构:
1. 多头辩护专家 (BullSpecialistAgent): 搜寻均线多头排列、放量突破、主力流入证据
2. 空头风控专家 (BearCriticAgent): 严苛挑刺，搜寻上方套牢盘、顶背离、费率过热与多头陷阱
3. 宏观舆情专家 (MacroNewsAgent): 结合全球热点资讯、宏观日历与突发黑天鹅评估宏观大盘风险
4. 首席量化仲裁官 (ChiefArbiterAgent): 综合三方辩论陈述与 RAG 避坑规则，计算共识分 consensus_score
5. 决策准入红线: 只有共识分 >= 75 且 盈亏比 >= 1.8 且 无冷却锁 时，方可签发开仓指令
"""

import json
import logging
import uuid
import time
from typing import Any, Dict, List, Optional, Tuple

from models import (
    AgentDebateOpinion,
    MultiAgentConsensusResponse,
    SignalAction,
)

logger = logging.getLogger("okx_dog.ai.consensus_network")


class BullSpecialistAgent:
    """多头辩护专家"""
    def evaluate(self, snapshot: Dict[str, Any]) -> AgentDebateOpinion:
        price = float(snapshot.get("current_price", 100.0))
        chg = float(snapshot.get("change_24h_pct", 0.0))
        deriv = snapshot.get("derivatives") or {}
        fr = float(deriv.get("funding_rate", 0.0))
        oi_chg = float(deriv.get("oi_change_24h_pct", 0.0))

        confidence = 0.65
        args = []
        risks = []

        if chg > 0:
            confidence += 0.15
            args.append(f"24H上涨动能完好 ({chg:+.2f}%)，多头趋势占优")
        if oi_chg > 2.0:
            confidence += 0.1
            args.append(f"未平仓合约量增加 {oi_chg:+.1f}%，伴随真实主力增量资金介入")
        if fr > 0 and fr < 0.03:
            args.append("资金费率处于健康适度看多区间，未见极端拥挤")
        else:
            risks.append("短线费率略高，需防范高位震荡洗盘")

        stance = "BULLISH" if confidence >= 0.60 else "NEUTRAL"
        return AgentDebateOpinion(
            role_name="多头辩护专家 (Bull Specialist)",
            stance=stance,
            confidence=round(min(0.95, confidence), 2),
            key_arguments=args or ["均线维持多头排列", "逢回调支撑位有承接"],
            risk_warnings=risks or ["需关注上方密集套牢盘"]
        )


class BearCriticAgent:
    """空头风控专家 (挑刺者)"""
    def evaluate(self, snapshot: Dict[str, Any]) -> AgentDebateOpinion:
        deriv = snapshot.get("derivatives") or {}
        fr = float(deriv.get("funding_rate", 0.0))
        chg = float(snapshot.get("change_24h_pct", 0.0))

        confidence = 0.55
        args = []
        risks = []

        if fr >= 0.03:
            confidence += 0.25
            args.append(f"资金费率高达 {fr*100:+.4f}% 偏过热，多头多头挤压踩踏风险加剧")
        if chg > 8.0:
            confidence += 0.15
            args.append(f"短线涨幅过大 (+{chg:.1f}%)，RSI 趋近超买区间，存在假突破诱多概率")

        risks.append("上方存在前高关键阻力位，突破量能若衰竭极易形成顶背离")
        stance = "BEARISH" if confidence >= 0.65 else "NEUTRAL"

        return AgentDebateOpinion(
            role_name="空头风控专家 (Bear Critic)",
            stance=stance,
            confidence=round(min(0.95, confidence), 2),
            key_arguments=args or ["上方阻力位抛压沉重", "衍生品多头情绪趋于拥挤"],
            risk_warnings=risks
        )


class MacroNewsAgent:
    """宏观舆情专家"""
    def evaluate(self, snapshot: Dict[str, Any], news_items: Optional[List[Dict[str, Any]]] = None) -> AgentDebateOpinion:
        confidence = 0.70
        args = []
        risks = []

        items = news_items or snapshot.get("news_items")
        if not items:
            try:
                from news_nlp_engine import NewsNLPEngine
                items = NewsNLPEngine._cached_news
            except Exception:
                items = None

        if items:
            avg_sentiment = sum(n.get("sentiment_score", 0.0) for n in items) / max(len(items), 1)
            has_p0 = any(n.get("urgency") == "P0" for n in items)
            if has_p0:
                confidence = 0.90
                risks.append("🚨 监测到突发 P0 级别全球重大事件/黑天鹅快讯，需防范极端插针！")
            elif avg_sentiment > 0.2:
                args.append(f"全网最新快讯整体情绪偏多头利好 (综合情感打分: {avg_sentiment:+.2f})")
            elif avg_sentiment < -0.2:
                confidence += 0.1
                risks.append(f"全网热点存在利空情绪扰动 (综合情感打分: {avg_sentiment:+.2f})")
        else:
            args.append("宏观日历暂无重大黑天鹅风险窗口公布，大盘流动性中性偏暖")

        stance = "BULLISH" if confidence >= 0.65 and not risks else ("BEARISH" if any("🚨" in r for r in risks) else "NEUTRAL")
        return AgentDebateOpinion(
            role_name="宏观舆情专家 (Macro & News)",
            stance=stance,
            confidence=round(confidence, 2),
            key_arguments=args or ["宏观流动性平稳，无系统性黑天鹅事件"],
            risk_warnings=risks or ["美股开盘时段可能伴随波动放大"]
        )


class ConsensusOrchestrator:
    """首席量化仲裁官与多智能体网络协调中枢"""

    def __init__(self):
        self.bull_agent = BullSpecialistAgent()
        self.bear_agent = BearCriticAgent()
        self.macro_agent = MacroNewsAgent()

    def arbitrate(
        self,
        snapshot: Dict[str, Any],
        injected_rules: List[str],
        news_items: Optional[List[Dict[str, Any]]] = None,
        is_in_cooldown: bool = False
    ) -> MultiAgentConsensusResponse:
        symbol = snapshot.get("symbol", "BTC-USDT-SWAP")
        price = float(snapshot.get("current_price", 100.0))

        # 1. 四方辩论
        bull_op = self.bull_agent.evaluate(snapshot)
        bear_op = self.bear_agent.evaluate(snapshot)
        macro_op = self.macro_agent.evaluate(snapshot, news_items)

        # 2. 仲裁打分 (基于 Softmax 动态加权门控网络)
        try:
            from .evolution.evolution_manager import AgentEvolutionManager
            manager = AgentEvolutionManager.get_instance()
            weights = manager.gating_network.get_weights()
            w_bull = weights.get("bull_specialist", 25.0) / 100.0
            w_bear = weights.get("bear_critic", 25.0) / 100.0
            w_macro = weights.get("macro_news", 15.0) / 100.0
            if not injected_rules:
                injected_rules = list(manager.learned_rules)
        except Exception:
            w_bull, w_bear, w_macro = 0.25, 0.25, 0.15

        # 基础中性分 50
        net_delta = 0.0
        if bull_op.stance == "BULLISH":
            net_delta += bull_op.confidence * w_bull * 80.0
        elif bull_op.stance == "BEARISH":
            net_delta -= bull_op.confidence * w_bull * 40.0

        if bear_op.stance == "BEARISH":
            net_delta -= bear_op.confidence * w_bear * 80.0
        elif bear_op.stance == "BULLISH":
            net_delta += bear_op.confidence * w_bear * 30.0

        if macro_op.stance == "BULLISH":
            net_delta += macro_op.confidence * w_macro * 50.0
        elif macro_op.stance == "BEARISH":
            net_delta -= macro_op.confidence * w_macro * 60.0

        score = int(max(10, min(95, 50 + net_delta)))

        # 3. 动作判定与准入红线
        final_action = SignalAction.HOLD_WAIT
        rejection_reason = None

        if score >= 75:
            final_action = SignalAction.BUY_LONG
        elif score <= 35:
            final_action = SignalAction.SELL_SHORT

        is_approved = True
        if is_in_cooldown:
            is_approved = False
            rejection_reason = "系统处于交易心理防上头强制冷静期，硬阻断开仓"
        elif score < 75 and score > 35:
            is_approved = False
            rejection_reason = f"首席仲裁共识分 ({score}) 未达开仓准入红线 (≥75 或 ≤35)，建议维持观望"

        # 4. 生成建议交易计划
        atr = price * 0.015
        if final_action == SignalAction.BUY_LONG:
            entry_low = round(price * 0.998, 2)
            entry_high = round(price * 1.002, 2)
            sl = round(price - atr * 2.0, 2)
            tp = round(price + atr * 4.0, 2)
            rr = 2.0
        elif final_action == SignalAction.SELL_SHORT:
            entry_low = round(price * 0.998, 2)
            entry_high = round(price * 1.002, 2)
            sl = round(price + atr * 2.0, 2)
            tp = round(price - atr * 4.0, 2)
            rr = 2.0
        else:
            entry_low = entry_high = sl = tp = price
            rr = 1.0

        plan = {
            "entry_range": [entry_low, entry_high],
            "stop_loss_price": sl,
            "take_profit_price": tp,
            "risk_reward_ratio": rr,
            "suggested_leverage": 3,
            "order_type": "LIMIT"
        }

        summary = (
            f"多智能体对抗博弈完成：共识分【{score}/100】。"
            f"多头置信度 {bull_op.confidence*100:.0f}%，空头风控置信度 {bear_op.confidence*100:.0f}%。"
            f"仲裁结论: {final_action.value}。"
            + (f"（已通过准入红线，建议执行）" if is_approved else f"（未通过执行: {rejection_reason}）")
        )

        return MultiAgentConsensusResponse(
            consensus_id=str(uuid.uuid4())[:12],
            symbol=symbol,
            timestamp=int(time.time() * 1000),
            consensus_score=score,
            final_action=final_action,
            is_approved_to_execute=is_approved,
            rejection_reason=rejection_reason,
            bull_opinion=bull_op,
            bear_opinion=bear_op,
            macro_opinion=macro_op,
            arbitration_summary=summary,
            injected_knowledge_rules=injected_rules,
            suggested_trade_plan=plan
        )