"""
OKX-Dog AI 决策大脑 - Paperclip 虚拟量化公司控制平面适配器
模块: okx-dog-ai/agent/paperclip_adapter.py
角色: 后端架构师与 AI 系统工程师 (agency-backend-architect & agency-ai-engineer)

核心功能:
1. 提供虚拟量化公司 6 大员工组织架构 (Org Chart) 的标准数据接口。
2. 将本地 Python 多智能体共识网络 (ConsensusNetwork) 包装为 Paperclip 标准 Action / Tool。
3. 接收技术形态与指标异动事件 (MarketAnomalyTriggerEvent)，执行投决会红蓝辩论并产出仲裁工单。
4. 统计本次任务消耗的思考 Token，并与 CFO 战绩激励引擎联动。
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .consensus_network import ConsensusOrchestrator
try:
    from models import SignalAction
except ImportError:
    import sys
    from pathlib import Path
    backend_dir = Path(__file__).resolve().parent.parent.parent / "okx-dog-backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    from models import SignalAction


logger = logging.getLogger("okx_dog.ai.paperclip_adapter")


class PaperclipEmployeeInfo(BaseModel):
    role: str
    name: str
    title: str
    department: str
    base_weight: float
    kpi_target: str
    is_executive: bool
    avatar_icon: str


class PaperclipCompanyOrgResponse(BaseModel):
    company_name: str = "OKX-Dog Autonomous Quant Fund"
    ceo_role: str = "CHIEF_ARBITER"
    governance_mode: str = "FULL_AUTONOMOUS"
    evaluation_window_days: int = 7
    employees: List[PaperclipEmployeeInfo]


class MarketAnomalyTriggerRequest(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}")
    symbol: str = "BTC-USDT-SWAP"
    trigger_type: str = "TREND_BREAKOUT"  # TREND_BREAKOUT / OVERSOLD_DIVERGENCE / DERIVATIVES_OI_SPIKE
    price: float = 0.0
    anomaly_description: str = ""
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    snapshot_context: Optional[Dict[str, Any]] = None


class PaperclipArbitrationTaskResult(BaseModel):
    task_id: str
    symbol: str
    action: str
    consensus_score: int
    expected_rr_ratio: float
    approved_by_ceo: bool
    bonus_tokens_consumed: int
    reasoning_summary: str
    debate_opinions: List[Dict[str, Any]] = []


class PaperclipAdapter:
    """
    Paperclip 控制平面适配中枢
    负责向外部 Paperclip 调度器暴露组织架构与决策执行端点
    """

    def __init__(self):
        self.orchestrator = ConsensusOrchestrator()
        self._init_org_chart()

    def _init_org_chart(self):
        self.employees = [
            PaperclipEmployeeInfo(
                role="CHIEF_ARBITER",
                name="诸葛仲裁",
                title="首席量化仲裁官 (执行 CEO)",
                department="投决管理委员会",
                base_weight=30.0,
                kpi_target="全盘夏普比率 >= 1.8，组合最大回撤 < 8%",
                is_executive=True,
                avatar_icon="Crown"
            ),
            PaperclipEmployeeInfo(
                role="BULL_SPECIALIST",
                name="冲锋多头",
                title="资深多头趋势研究员",
                department="量化策略投研部",
                base_weight=20.0,
                kpi_target="多头开仓盈亏比 >= 1.8，顺势波段捕获率 >= 65%",
                is_executive=False,
                avatar_icon="TrendingUp"
            ),
            PaperclipEmployeeInfo(
                role="BEAR_CRITIC",
                name="铁血风控",
                title="红队首席合规挑刺官",
                department="风险管理合规部",
                base_weight=20.0,
                kpi_target="假突破与顶背离排雷成功率 >= 70%",
                is_executive=False,
                avatar_icon="ShieldAlert"
            ),
            PaperclipEmployeeInfo(
                role="MACRO_NEWS",
                name="全球眼",
                title="全球宏观与黑天鹅策略师",
                department="全球情报研究部",
                base_weight=10.0,
                kpi_target="美联储/CPI关键窗口零失误，P0突发黑天鹅零误判",
                is_executive=False,
                avatar_icon="Globe"
            ),
            PaperclipEmployeeInfo(
                role="MICROSTRUCTURE",
                name="盘口狙击",
                title="微观流动性交易员",
                department="交易撮合执行部",
                base_weight=10.0,
                kpi_target="Smart Pegging Maker 成交比例 >= 60%，滑点损耗压降",
                is_executive=False,
                avatar_icon="Crosshair"
            ),
            PaperclipEmployeeInfo(
                role="CFO_AUDITOR",
                name="公道伯",
                title="财务与战绩审计总监 (CFO)",
                department="财务与绩效激励部",
                base_weight=10.0,
                kpi_target="战绩积分与算力奖励 100% 准确核算，守护长期复利",
                is_executive=True,
                avatar_icon="Coins"
            ),
        ]

    def get_company_org(self) -> PaperclipCompanyOrgResponse:
        """获取虚拟公司组织架构与在册员工资料"""
        return PaperclipCompanyOrgResponse(employees=self.employees)

    async def handle_anomaly_event(self, req: MarketAnomalyTriggerRequest) -> PaperclipArbitrationTaskResult:
        """
        处理前哨指标哨兵推送的异动事件，唤醒虚拟投决会对抗辩论
        """
        task_id = f"task_{uuid.uuid4().hex[:10]}"
        symbol = req.symbol
        price = req.price if req.price > 0 else 100.0

        snapshot = req.snapshot_context or {
            "symbol": symbol,
            "current_price": price,
            "change_24h_pct": 2.5,
            "market_regime": "TRENDING_UP",
            "derivatives": {
                "funding_rate": 0.0001,
                "oi_change_24h_pct": 3.8
            }
        }

        # 1. 尝试从 CFO 模块获取当前是否有员工已解锁「深度思考」特权
        deep_thinking_active = False
        bonus_consumed = 0
        try:
            from .evolution.cfo_performance_manager import CfoPerformanceManager
            cfo = CfoPerformanceManager.get_instance()
            # 若有员工拥有战绩奖励，核销本次任务消耗
            bonus_consumed = cfo.consume_bonus_quota_for_task(symbol, amount=1200)
            deep_thinking_active = bonus_consumed > 0
        except Exception:
            deep_thinking_active = False
            bonus_consumed = 0

        # 2. 执行投决会仲裁
        injected_rules = [f"异动触发事件: {req.trigger_type} - {req.anomaly_description}"]
        arbitration = self.orchestrator.arbitrate(
            snapshot=snapshot,
            injected_rules=injected_rules
        )

        score = arbitration.consensus_score
        final_action_val = arbitration.final_action.value if hasattr(arbitration.final_action, "value") else str(arbitration.final_action)
        approved = arbitration.is_approved_to_execute

        # 组织辩论摘要
        opinions_list = []
        for op in [arbitration.bull_opinion, arbitration.bear_opinion, arbitration.macro_opinion]:
            if op:
                opinions_list.append({
                    "role": op.role_name,
                    "stance": op.stance,
                    "confidence": op.confidence,
                    "key_arguments": op.key_arguments,
                    "risk_warnings": op.risk_warnings
                })

        summary = (
            f"【CEO 投决会裁定】标的: {symbol}, 仲裁分: {score}/100, 决议动作: {final_action_val}, "
            f"准入状态: {'批准执行' if approved else '暂缓观望'}, "
            f"{'🌟已激活战绩奖金深度思考' if deep_thinking_active else '⚡标准精简推理'}"
        )

        plan = arbitration.suggested_trade_plan or {}
        rr_ratio = float(plan.get("risk_reward_ratio", 1.8))

        return PaperclipArbitrationTaskResult(
            task_id=task_id,
            symbol=symbol,
            action=final_action_val,
            consensus_score=score,
            expected_rr_ratio=rr_ratio,
            approved_by_ceo=approved,
            bonus_tokens_consumed=bonus_consumed,
            reasoning_summary=summary,
            debate_opinions=opinions_list
        )



# 单例实例
paperclip_adapter = PaperclipAdapter()
