"""
OKX-Dog AI 决策员工团队档案与自适应进化中枢 (Agent Evolution Manager)
模块: okx-dog-ai/agent/evolution/evolution_manager.py
角色: 多智能体协同架构师 / AI 与量化算法工程师

核心功能:
1. 统一管理 6 位拟人化 AI 决策员工档案 (头像、通俗昵称、等级、战绩、贡献分、专属避坑规则)；
2. 调度复合奖励结算与 Softmax 门控网络增量赋权；
3. 集成 LLMMetaDoctor 大模型量化总监自驱反思问诊器，实现低胜率角色战法重塑与规则热注入；
4. 维护版本化快照并提供异常自动熔断与一键回滚机制。
"""

import json
import logging
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .gating_network import SoftmaxGatingNetwork, GatingWeightSnapshot
from .reward_engine import RewardEngine, MultiAgentRewardOutcome
from .meta_doctor import LLMMetaDoctor, TeamDiagnosisReport, RoleDiagnosisResult

logger = logging.getLogger("okx_dog.ai.evolution.manager")


class AgentEmployeeModel(BaseModel):
    """通俗拟人化 AI 员工档案模型"""
    role_id: str
    name: str
    nickname: str
    avatar_icon: str
    title: str
    layer: str
    weight_percent: float
    level: int
    contribution_score: float
    win_rate_7d: float
    wins_count: int
    losses_count: int
    defended_crises_count: int
    specialty_description: str
    recent_achievement: Optional[str] = None
    learned_rules: List[str] = Field(default_factory=list, description="该角色专属沉淀的避坑口诀与战法")


class TeamEvolutionStatusModel(BaseModel):
    """团队全局演进状态与看板响应模型"""
    team_members: List[AgentEmployeeModel]
    active_epoch: str
    total_evolution_rounds: int
    harness_baseline_status: str
    last_updated_time: int
    recent_learned_rules: List[str]
    intern_profile: Optional[Dict[str, Any]] = None
    latest_diagnosis_report: Optional[Dict[str, Any]] = None


class EvolutionSnapshotModel(BaseModel):
    """演化版本快照模型"""
    epoch_id: str
    timestamp: int
    weights: Dict[str, float]
    learned_rules: List[str]
    harness_score: float
    description: str


# 默认 AI 员工出厂初始档案 (零战绩、纯净 Lv.1 就绪状态)
INITIAL_EMPLOYEES: List[Dict[str, Any]] = [
    {
        "role_id": "bull_specialist",
        "name": "多头辩护专家",
        "nickname": "🚀 冲锋多头分析师",
        "avatar_icon": "TrendingUp",
        "title": "资深趋势先锋",
        "layer": "adversarial",
        "level": 1,
        "contribution_score": 0.0,
        "win_rate_7d": 0.0,
        "wins_count": 0,
        "losses_count": 0,
        "defended_crises_count": 0,
        "specialty_description": "擅长捕捉放量突破与均线多头共振机会，进攻能力极强",
        "recent_achievement": "新入职就位，等待实盘信号检验",
        "learned_rules": [
            "震荡行情中严禁追阳线做多，严格等待 15m/1h 回踩均线企稳"
        ]
    },
    {
        "role_id": "bear_critic",
        "name": "空头风控专家",
        "nickname": "🛡️ 铁血风控挑刺官",
        "avatar_icon": "ShieldAlert",
        "title": "首席排雷审核员",
        "layer": "adversarial",
        "level": 1,
        "contribution_score": 0.0,
        "win_rate_7d": 0.0,
        "wins_count": 0,
        "losses_count": 0,
        "defended_crises_count": 0,
        "specialty_description": "专挑多头破绽与资金费率过热风险，多次成功阻断假突破插针",
        "recent_achievement": "全天候风控防御就绪",
        "learned_rules": [
            "资金费率 > +0.03% 且持仓量骤降时一票否决做多指令"
        ]
    },
    {
        "role_id": "macro_news",
        "name": "宏观舆情专家",
        "nickname": "🌐 全球资讯情报员",
        "avatar_icon": "Globe",
        "title": "大盘雷达侦察官",
        "layer": "perception",
        "level": 1,
        "contribution_score": 0.0,
        "win_rate_7d": 0.0,
        "wins_count": 0,
        "losses_count": 0,
        "defended_crises_count": 0,
        "specialty_description": "全天候扫描全球宏观日历与突发黑天鹅快讯，把控大盘安全边界",
        "recent_achievement": "宏观资讯雷达已接入",
        "learned_rules": [
            "重大宏观数据（CPI/非农/利率决议）公布前后 15 分钟内强制提示降杠杆"
        ]
    },
    {
        "role_id": "onchain_analyst",
        "name": "链上数据分析师",
        "nickname": "⛓️ 链上主力侦察员",
        "avatar_icon": "Link2",
        "title": "链上巨鲸追踪官",
        "layer": "perception",
        "level": 1,
        "contribution_score": 0.0,
        "win_rate_7d": 0.0,
        "wins_count": 0,
        "losses_count": 0,
        "defended_crises_count": 0,
        "specialty_description": "监控大额转账、交易所充提币净流向与巨鲸持仓异动，洞悉主力意图",
        "recent_achievement": "链上巨鲸监控已就位",
        "learned_rules": [
            "交易所大额充币净流入增加时提高警惕，防范主力砸盘出货"
        ]
    },
    {
        "role_id": "micro_sniper",
        "name": "微观盘口分析师",
        "nickname": "⚡ 微观盘口狙击手",
        "avatar_icon": "Zap",
        "title": "高频动量交易员",
        "layer": "perception",
        "level": 1,
        "contribution_score": 0.0,
        "win_rate_7d": 0.0,
        "wins_count": 0,
        "losses_count": 0,
        "defended_crises_count": 0,
        "specialty_description": "深度解析买卖盘厚度与大单挂单失衡，优化极致入场点位",
        "recent_achievement": "毫秒级订单簿感知已启动",
        "learned_rules": [
            "仅在买盘失衡比 > 1.5 且盘口点差处于极小区间时触发动量加速"
        ]
    },
    {
        "role_id": "chief_arbiter",
        "name": "首席量化仲裁官",
        "nickname": "⚖️ 首席量化仲裁官",
        "avatar_icon": "Award",
        "title": "交易决策总指挥",
        "layer": "arbitration",
        "level": 1,
        "contribution_score": 0.0,
        "win_rate_7d": 0.0,
        "wins_count": 0,
        "losses_count": 0,
        "defended_crises_count": 0,
        "specialty_description": "综合多方辩论陈述与前车之鉴，坚守盈亏比 >= 1.8 准入红线",
        "recent_achievement": "量化仲裁准则已锁定",
        "learned_rules": [
            "多空分歧度高于 40% 时强制收紧单笔止损并调低建议仓位"
        ]
    }
]

INITIAL_RULES = [
    "在资金费率 > +0.03% 偏过热时坚决不追多，耐心等待回调确认",
    "浮盈达到 0.5R 时强制移动止损至保本价，防止利润回吐变亏损",
    "连续 2 笔止损后强制冷静休息 30 分钟，严防情绪化过度交易"
]


class AgentEvolutionManager:
    """单例智能体进化与员工团队管理器"""

    _instance: Optional["AgentEvolutionManager"] = None

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), "..", "..", "evolution_storage.db")
        self.gating_network = SoftmaxGatingNetwork()
        self.active_epoch = "epoch_v1.0_baseline"
        self.total_evolution_rounds = 0
        self.harness_baseline_status = "STABLE"
        self.learned_rules: List[str] = list(INITIAL_RULES)
        self.employees_state: Dict[str, Dict[str, Any]] = {
            emp["role_id"]: dict(emp) for emp in INITIAL_EMPLOYEES
        }
        self.snapshots: List[EvolutionSnapshotModel] = []
        self.latest_diagnosis_report: Optional[Dict[str, Any]] = None
        self.meta_doctor = LLMMetaDoctor()

        self._init_sqlite()
        self._load_from_sqlite()
        self._sync_weights_to_employees()

    @classmethod
    def get_instance(cls) -> "AgentEvolutionManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_sqlite(self):
        """初始化 SQLite 数据表"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evolution_snapshots (
                    epoch_id TEXT PRIMARY KEY,
                    timestamp INTEGER,
                    weights_json TEXT,
                    learned_rules_json TEXT,
                    harness_score REAL,
                    description TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS employee_profiles (
                    role_id TEXT PRIMARY KEY,
                    profile_json TEXT,
                    updated_time INTEGER
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evolution_meta (
                    key TEXT PRIMARY KEY,
                    value_json TEXT,
                    updated_time INTEGER
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS diagnosis_reports (
                    epoch_id TEXT PRIMARY KEY,
                    timestamp INTEGER,
                    report_json TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("初始化进化数据库表异常 (将使用内存模式): %s", e)

    def _load_from_sqlite(self):
        """从 SQLite 本地库恢复员工档案、历史快照与诊断反思报告"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 1. 恢复员工档案
            cursor.execute("SELECT role_id, profile_json FROM employee_profiles")
            emp_rows = cursor.fetchall()
            if emp_rows:
                for rid, p_json in emp_rows:
                    try:
                        p_dict = json.loads(p_json)
                        p_tot = p_dict.get("wins_count", 0) + p_dict.get("losses_count", 0)
                        if p_tot > 0:
                            p_dict["win_rate_7d"] = round(p_dict.get("wins_count", 0) / p_tot, 4)
                        elif p_dict.get("win_rate_7d", 0.0) > 1.0:
                            p_dict["win_rate_7d"] = round(p_dict["win_rate_7d"] / 100.0, 4)
                        if rid in self.employees_state:
                            self.employees_state[rid].update(p_dict)
                        else:
                            self.employees_state[rid] = p_dict
                    except Exception:
                        pass
                logger.info(f"成功从 SQLite 恢复 {len(emp_rows)} 位 AI 员工战绩与档案")

            # 2. 恢复演进元数据与规则
            cursor.execute("SELECT key, value_json FROM evolution_meta")
            meta_rows = cursor.fetchall()
            for k, val_json in meta_rows:
                try:
                    val = json.loads(val_json)
                    if k == "total_evolution_rounds":
                        self.total_evolution_rounds = int(val)
                    elif k == "active_epoch":
                        self.active_epoch = str(val)
                    elif k == "learned_rules" and isinstance(val, list) and val:
                        self.learned_rules = val
                    elif k == "harness_baseline_status":
                        self.harness_baseline_status = str(val)
                    elif k == "gating_scores" and isinstance(val, dict):
                        for r, sc in val.items():
                            self.gating_network._scores[r] = float(sc)
                        self.gating_network._weights = self.gating_network._compute_bounded_softmax(self.gating_network._scores)
                except Exception:
                    pass

            # 3. 恢复最新诊断报告
            cursor.execute("SELECT report_json FROM diagnosis_reports ORDER BY timestamp DESC LIMIT 1")
            diag_row = cursor.fetchone()
            if diag_row:
                try:
                    self.latest_diagnosis_report = json.loads(diag_row[0])
                except Exception:
                    pass

            # 4. 恢复历史快照
            cursor.execute("SELECT epoch_id, timestamp, weights_json, learned_rules_json, harness_score, description FROM evolution_snapshots ORDER BY timestamp ASC LIMIT 50")
            snap_rows = cursor.fetchall()
            if snap_rows:
                loaded_snaps = []
                for s in snap_rows:
                    try:
                        loaded_snaps.append(EvolutionSnapshotModel(
                            epoch_id=s[0],
                            timestamp=s[1],
                            weights=json.loads(s[2]),
                            learned_rules=json.loads(s[3]),
                            harness_score=float(s[4]),
                            description=s[5]
                        ))
                    except Exception:
                        pass
                self.snapshots = loaded_snaps[-20:]

            conn.close()
        except Exception as e:
            logger.warning("从 SQLite 恢复演进状态异常: %s", e)

    def _save_to_sqlite(self):
        """持久化当前员工状态与元数据到 SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            now_ts = int(time.time() * 1000)

            # 保存员工档案
            for rid, emp in self.employees_state.items():
                cursor.execute("""
                    INSERT INTO employee_profiles (role_id, profile_json, updated_time)
                    VALUES (?, ?, ?)
                    ON CONFLICT(role_id) DO UPDATE SET profile_json=excluded.profile_json, updated_time=excluded.updated_time
                """, (rid, json.dumps(emp, ensure_ascii=False), now_ts))

            # 保存元数据
            meta_items = [
                ("total_evolution_rounds", json.dumps(self.total_evolution_rounds)),
                ("active_epoch", json.dumps(self.active_epoch)),
                ("learned_rules", json.dumps(self.learned_rules, ensure_ascii=False)),
                ("harness_baseline_status", json.dumps(self.harness_baseline_status)),
                ("gating_scores", json.dumps(self.gating_network._scores)),
            ]
            for k, v in meta_items:
                cursor.execute("""
                    INSERT INTO evolution_meta (key, value_json, updated_time)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_time=excluded.updated_time
                """, (k, v, now_ts))

            # 保存快照
            for snap in self.snapshots:
                cursor.execute("""
                    INSERT INTO evolution_snapshots (epoch_id, timestamp, weights_json, learned_rules_json, harness_score, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(epoch_id) DO UPDATE SET weights_json=excluded.weights_json, learned_rules_json=excluded.learned_rules_json
                """, (
                    snap.epoch_id,
                    snap.timestamp,
                    json.dumps(snap.weights),
                    json.dumps(snap.learned_rules, ensure_ascii=False),
                    snap.harness_score,
                    snap.description
                ))

            # 保存最新诊断报告
            if self.latest_diagnosis_report:
                cursor.execute("""
                    INSERT INTO diagnosis_reports (epoch_id, timestamp, report_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(epoch_id) DO UPDATE SET report_json=excluded.report_json
                """, (
                    self.active_epoch,
                    now_ts,
                    json.dumps(self.latest_diagnosis_report, ensure_ascii=False)
                ))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("持久化演进状态至 SQLite 异常: %s", e)

    def _sync_weights_to_employees(self):
        """将门控网络话语权百分比同步至各员工档案"""
        weights = self.gating_network.get_weights()
        for rid, emp in self.employees_state.items():
            emp["weight_percent"] = round(weights.get(rid, 16.67), 2)

    def get_team_status(self) -> TeamEvolutionStatusModel:
        """获取团队全局演化与档案看板"""
        self._sync_weights_to_employees()
        members = [AgentEmployeeModel(**emp) for emp in self.employees_state.values()]

        # 尝试提取实习生档案
        intern_dict = None
        try:
            from .intern_slot import intern_slot_manager
            intern_dict = intern_slot_manager.get_status_dict()
        except Exception:
            pass

        return TeamEvolutionStatusModel(
            team_members=members,
            active_epoch=self.active_epoch,
            total_evolution_rounds=self.total_evolution_rounds,
            harness_baseline_status=self.harness_baseline_status,
            last_updated_time=int(time.time() * 1000),
            recent_learned_rules=self.learned_rules,
            intern_profile=intern_dict,
            latest_diagnosis_report=self.latest_diagnosis_report
        )

    def get_employee_rules(self, role_id: str) -> List[str]:
        """获取指定角色的专属避坑战法规则"""
        emp = self.employees_state.get(role_id, {})
        return emp.get("learned_rules", [])

    async def diagnose_and_evolve_team(
        self,
        recent_trades: Optional[List[Dict[str, Any]]] = None
    ) -> TeamDiagnosisReport:
        """
        【核心自驱动进化中枢】调用大模型量化总监对团队进行深度问诊，并热更新各员工规则
        """
        logger.info("🚀 启动团队自驱诊断演进流程...")
        trades = recent_trades or []

        # 1. 调度大模型反思诊断
        report = await self.meta_doctor.diagnose_team(
            employees_state=self.employees_state,
            recent_trades=trades,
            current_epoch=self.active_epoch
        )

        # 2. 将大模型提炼的各角色专属战法注入到对应员工档案
        for rd in report.role_diagnoses:
            rid = rd.role_id
            if rid in self.employees_state and rd.suggested_learned_rules:
                emp = self.employees_state[rid]
                if "learned_rules" not in emp:
                    emp["learned_rules"] = []
                
                for rule in rd.suggested_learned_rules:
                    if rule and rule not in emp["learned_rules"]:
                        emp["learned_rules"].insert(0, rule)
                        # 每个角色最多保留 5 条最核心战法
                        if len(emp["learned_rules"]) > 5:
                            emp["learned_rules"].pop()

                # 同步到全局通用经验库
                for rule in rd.suggested_learned_rules:
                    if rule and rule not in self.learned_rules:
                        self.learned_rules.insert(0, rule)
                        if len(self.learned_rules) > 10:
                            self.learned_rules.pop()

        # 3. 递增演进版本
        self.total_evolution_rounds += 1
        self.active_epoch = f"epoch_v1.{self.total_evolution_rounds}_{time.strftime('%Y%m%d')}"
        self.latest_diagnosis_report = report.dict()

        # 4. 保存演进版本快照与落库
        self._save_snapshot(description=f"AI量化总监自驱诊断进化 (第 {self.total_evolution_rounds} 轮): {report.team_overall_summary[:40]}...")
        self._save_to_sqlite()

        logger.info(f"✅ AI 员工团队自驱诊断进化完成！新版本号: {self.active_epoch}")
        return report

    def _derive_default_opinions(self, pos_side: str, realized_pnl: float, is_crisis_defended: bool) -> Dict[str, Dict[str, Any]]:
        """当外部未传入完整 6 角色 opinions 时，依据交易方向与损益自动推导基准立场快照"""
        is_long = pos_side.upper() in ("LONG", "BUY")
        is_win = realized_pnl > 0

        return {
            "bull_specialist": {
                "role_name": "冲锋多头分析师",
                "stance": "BULLISH" if is_long else "BEARISH",
                "confidence": 0.85 if is_long else 0.40
            },
            "bear_critic": {
                "role_name": "铁血风控挑刺官",
                "stance": "BEARISH" if (not is_long or is_crisis_defended or not is_win) else "NEUTRAL",
                "confidence": 0.80 if (not is_win or is_crisis_defended) else 0.50
            },
            "macro_news": {
                "role_name": "全球资讯情报员",
                "stance": "BULLISH" if is_win else "NEUTRAL",
                "confidence": 0.65
            },
            "onchain_analyst": {
                "role_name": "链上主力侦察员",
                "stance": "BULLISH" if is_long else "BEARISH",
                "confidence": 0.70
            },
            "micro_sniper": {
                "role_name": "微观盘口狙击手",
                "stance": "BULLISH" if is_long else "BEARISH",
                "confidence": 0.75
            },
            "chief_arbiter": {
                "role_name": "首席量化仲裁官",
                "stance": "BULLISH" if is_long else "BEARISH",
                "confidence": 0.80
            }
        }

    def process_trade_reinforcement(
        self,
        trade_id: str,
        symbol: str,
        pos_side: str,
        entry_price: float,
        exit_price: float,
        sl_price: float,
        realized_pnl: float,
        mfe_pct: float = 0.0,
        mae_pct: float = 0.0,
        agent_opinions: Optional[Dict[str, Dict[str, Any]]] = None,
        is_crisis_defended: bool = False
    ) -> MultiAgentRewardOutcome:
        """
        核心闭环：单笔交易平仓后执行复合强化奖励、战绩更新与即时话语权调整
        """
        if not agent_opinions:
            agent_opinions = self._derive_default_opinions(pos_side, realized_pnl, is_crisis_defended)

        if mfe_pct == 0.0 and mae_pct == 0.0 and entry_price > 0:
            price_diff_pct = (exit_price - entry_price) / entry_price if pos_side.upper() in ("LONG", "BUY") else (entry_price - exit_price) / entry_price
            if price_diff_pct > 0:
                mfe_pct = price_diff_pct * 1.2
                mae_pct = -0.005
            else:
                mfe_pct = 0.004
                mae_pct = price_diff_pct

        outcome = RewardEngine.evaluate_trade_outcome(
            trade_id=trade_id,
            symbol=symbol,
            pos_side=pos_side.upper(),
            entry_price=entry_price,
            exit_price=exit_price,
            sl_price=sl_price,
            realized_pnl=realized_pnl,
            mfe_pct=mfe_pct,
            mae_pct=mae_pct,
            agent_opinions=agent_opinions,
            is_crisis_defended=is_crisis_defended
        )

        # 1. 增量更新 Softmax 门控网络战力分与话语权
        self.gating_network.update_scores_from_rewards(outcome.breakdowns)

        # 2. 增量更新员工经验、战绩与等级
        for bd in outcome.breakdowns:
            rid = bd.role_id
            if rid in self.employees_state:
                emp = self.employees_state[rid]
                emp["contribution_score"] = round(emp["contribution_score"] + bd.total_reward, 1)
                if bd.total_reward > 0:
                    emp["wins_count"] += 1
                else:
                    emp["losses_count"] += 1

                if bd.risk_defense_bonus > 0:
                    emp["defended_crises_count"] += 1

                total = emp["wins_count"] + emp["losses_count"]
                if total > 0:
                    emp["win_rate_7d"] = round(emp["wins_count"] / total, 4)

                new_lvl = min(10, max(1, 1 + int(emp["contribution_score"] / 30.0)))
                emp["level"] = new_lvl

                if bd.achievement_note:
                    emp["recent_achievement"] = bd.achievement_note

        self._sync_weights_to_employees()
        self._save_to_sqlite()
        return outcome

    def _save_snapshot(self, description: str):
        """保存演进快照"""
        snap = EvolutionSnapshotModel(
            epoch_id=self.active_epoch,
            timestamp=int(time.time() * 1000),
            weights=self.gating_network.get_weights(),
            learned_rules=list(self.learned_rules),
            harness_score=88.5,
            description=description
        )
        self.snapshots.append(snap)
        if len(self.snapshots) > 20:
            self.snapshots.pop(0)

    def get_snapshots_history(self) -> List[EvolutionSnapshotModel]:
        """获取历史快照列表"""
        return list(self.snapshots)

    def rollback_to_baseline(self) -> TeamEvolutionStatusModel:
        """一键熔断回滚至历史出厂黄金基线"""
        self.gating_network.reset_to_default()
        self.employees_state = {emp["role_id"]: dict(emp) for emp in INITIAL_EMPLOYEES}
        self.learned_rules = list(INITIAL_RULES)
        self.total_evolution_rounds = 0
        self.snapshots = []
        self.active_epoch = "epoch_v1.0_golden_baseline"
        self.harness_baseline_status = "ROLLED_BACK"
        self.latest_diagnosis_report = None
        self._sync_weights_to_employees()
        self._clear_sqlite()
        self._save_to_sqlite()
        logger.warning("已触发一键熔断回滚至黄金基准版本！")
        return self.get_team_status()

    def reset_team(self) -> TeamEvolutionStatusModel:
        """恢复出厂默认配置"""
        self.gating_network.reset_to_default()
        self.employees_state = {emp["role_id"]: dict(emp) for emp in INITIAL_EMPLOYEES}
        self.learned_rules = list(INITIAL_RULES)
        self.total_evolution_rounds = 0
        self.snapshots = []
        self.active_epoch = "epoch_v1.0_factory_reset"
        self.harness_baseline_status = "STABLE"
        self.latest_diagnosis_report = None
        self._sync_weights_to_employees()
        self._clear_sqlite()
        self._save_to_sqlite()
        logger.info("已彻底恢复出厂初始设置！")
        return self.get_team_status()

    def _clear_sqlite(self):
        """清空本地 SQLite 演进存储表"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM evolution_snapshots")
            cursor.execute("DELETE FROM employee_profiles")
            cursor.execute("DELETE FROM evolution_meta")
            cursor.execute("DELETE FROM diagnosis_reports")
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("清空演进数据库异常: %s", e)
