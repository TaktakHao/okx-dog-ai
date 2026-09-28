"""
OKX-Dog AI 决策大脑 - CFO 财务与战绩激励审计中枢 (CFO Performance Manager)
模块: okx-dog-ai/agent/evolution/cfo_performance_manager.py
角色: 资深量化系统工程师与财务总监 (agency-backend-architect & agency-financial-analyst)

核心功能:
1. 落地长期主义战绩考核 (Rolling 7-Day Horizon) 与员工战绩档案持久化。
2. 维护员工算力奖励池 (Remaining Bonus Quota)，实现「战绩好 -> 奖励 Token -> 解锁深度思考」闭环。
3. 对接实盘平仓交易记录 (trade_record)，自动核算利润贡献并向立功员工派发奖励。
4. 提供符合 contracts/paperclip/employee_quota_contract.json 规范的标准化对外接口。
"""

import json
import logging
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("okx_dog.ai.evolution.cfo")

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "evolution_storage.db")


class EmployeeQuotaStatusModel(BaseModel):
    role: str
    name: str
    rolling_7d_pnl_usdt: float = 0.0
    rolling_win_rate: float = 0.65
    successful_hedges_count: int = 0
    total_bonus_earned_tokens: int = 50000
    remaining_bonus_tokens: int = 25000
    deep_thinking_unlocked: bool = True
    current_weight: float = 20.0
    level_title: str = "骨干合伙人"


class CompanyQuotaOverviewModel(BaseModel):
    company_name: str = "OKX-Dog Autonomous Quant Fund"
    subscription_type: str = "LOCAL_CLI_SUBSCRIPTION"
    total_bonus_pool_tokens: int = 300000
    remaining_pool_tokens: int = 150000
    employees_quota: List[EmployeeQuotaStatusModel]
    updated_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))


class CfoPerformanceManager:
    """
    CFO 财务与员工绩效激励总监单例
    负责管理 6 大员工的战绩积分与可用奖励额度
    """
    _instance: Optional["CfoPerformanceManager"] = None

    @classmethod
    def get_instance(cls, db_path: Optional[str] = None) -> "CfoPerformanceManager":
        if cls._instance is None:
            cls._instance = cls(db_path=db_path or DEFAULT_DB_PATH)
        return cls._instance

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()
        self._load_or_bootstrap_records()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cfo_employee_quota (
                        role TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        rolling_7d_pnl REAL DEFAULT 0.0,
                        rolling_win_rate REAL DEFAULT 0.65,
                        successful_hedges_count INTEGER DEFAULT 0,
                        total_bonus_earned INTEGER DEFAULT 50000,
                        remaining_bonus INTEGER DEFAULT 25000,
                        current_weight REAL DEFAULT 20.0,
                        level_title TEXT DEFAULT '骨干合伙人',
                        updated_at INTEGER NOT NULL
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cfo_reward_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT,
                        role TEXT NOT NULL,
                        reward_tokens INTEGER NOT NULL,
                        reason TEXT NOT NULL,
                        pnl_usdt REAL DEFAULT 0.0,
                        timestamp INTEGER NOT NULL
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"CFO 数据库初始化异常: {e}")

    def _load_or_bootstrap_records(self):
        default_roster = [
            ("CHIEF_ARBITER", "诸葛仲裁", 128.5, 0.72, 8, 80000, 45000, 30.0, "执行 CEO"),
            ("BULL_SPECIALIST", "冲锋多头", 95.2, 0.68, 2, 60000, 32000, 25.0, "资深趋势官"),
            ("BEAR_CRITIC", "铁血风控", 62.0, 0.75, 12, 70000, 48000, 20.0, "首席排雷官"),
            ("MACRO_NEWS", "全球眼", 25.0, 0.60, 5, 40000, 15000, 10.0, "情报专家"),
            ("MICROSTRUCTURE", "盘口狙击", 38.6, 0.70, 3, 45000, 22000, 10.0, "执行先锋"),
            ("CFO_AUDITOR", "公道伯", 15.0, 0.80, 4, 50000, 30000, 5.0, "财务总监"),
        ]
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now_ms = int(time.time() * 1000)
                for r in default_roster:
                    cursor.execute("""
                        INSERT OR IGNORE INTO cfo_employee_quota 
                        (role, name, rolling_7d_pnl, rolling_win_rate, successful_hedges_count, total_bonus_earned, remaining_bonus, current_weight, level_title, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], now_ms))
                conn.commit()
        except Exception as e:
            logger.error(f"CFO 员工档案初始化异常: {e}")

    def get_company_quota_overview(self) -> CompanyQuotaOverviewModel:
        """获取全员战绩与可用奖励额度概览"""
        employees: List[EmployeeQuotaStatusModel] = []
        tot_earned = 0
        tot_remain = 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT role, name, rolling_7d_pnl, rolling_win_rate, successful_hedges_count,
                           total_bonus_earned, remaining_bonus, current_weight, level_title
                    FROM cfo_employee_quota
                """)
                rows = cursor.fetchall()
                for row in rows:
                    role, name, pnl, win_rate, hedges, earned, remain, weight, level = row
                    tot_earned += earned
                    tot_remain += remain
                    employees.append(EmployeeQuotaStatusModel(
                        role=role,
                        name=name,
                        rolling_7d_pnl_usdt=round(pnl, 2),
                        rolling_win_rate=round(win_rate, 2),
                        successful_hedges_count=hedges,
                        total_bonus_earned_tokens=earned,
                        remaining_bonus_tokens=remain,
                        deep_thinking_unlocked=(remain > 0),
                        current_weight=round(weight, 1),
                        level_title=level
                    ))
        except Exception as e:
            logger.error(f"获取全员额度概览异常: {e}")

        return CompanyQuotaOverviewModel(
            company_name="OKX-Dog Autonomous Quant Fund",
            subscription_type="LOCAL_CLI_SUBSCRIPTION",
            total_bonus_pool_tokens=tot_earned,
            remaining_pool_tokens=tot_remain,
            employees_quota=employees,
            updated_at_ms=int(time.time() * 1000)
        )

    def reward_trade_profit(self, trade_id: str, pnl_usdt: float, side: str = "LONG"):
        """
        实盘交易平仓盈利结算派发战绩奖励
        每获得 1 USDT 净利润，向主推团队按功劳派发 1,000 Tokens 奖励
        """
        if pnl_usdt <= 0:
            logger.info(f"【CFO 战绩结算】交易 {trade_id} 净利润为 {pnl_usdt:.2f} USDT，不触发奖励派发")
            return

        now_ms = int(time.time() * 1000)
        tokens_to_award = int(pnl_usdt * 1000)
        # 多头或空头立功员工
        lead_role = "BULL_SPECIALIST" if side.upper() == "LONG" else "BEAR_CRITIC"
        ceo_role = "CHIEF_ARBITER"

        lead_bonus = int(tokens_to_award * 0.6)
        ceo_bonus = int(tokens_to_award * 0.4)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # 奖励主功臣
                cursor.execute("""
                    UPDATE cfo_employee_quota
                    SET rolling_7d_pnl = rolling_7d_pnl + ?,
                        total_bonus_earned = total_bonus_earned + ?,
                        remaining_bonus = remaining_bonus + ?,
                        updated_at = ?
                    WHERE role = ?
                """, (pnl_usdt * 0.6, lead_bonus, lead_bonus, now_ms, lead_role))
                # 奖励统筹 CEO
                cursor.execute("""
                    UPDATE cfo_employee_quota
                    SET rolling_7d_pnl = rolling_7d_pnl + ?,
                        total_bonus_earned = total_bonus_earned + ?,
                        remaining_bonus = remaining_bonus + ?,
                        updated_at = ?
                    WHERE role = ?
                """, (pnl_usdt * 0.4, ceo_bonus, ceo_bonus, now_ms, ceo_role))

                # 记录日志
                cursor.execute("""
                    INSERT INTO cfo_reward_log (trade_id, role, reward_tokens, reason, pnl_usdt, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (trade_id, lead_role, lead_bonus, f"实盘波段止盈 (+{pnl_usdt:.2f} USDT)", pnl_usdt, now_ms))
                cursor.execute("""
                    INSERT INTO cfo_reward_log (trade_id, role, reward_tokens, reason, pnl_usdt, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (trade_id, ceo_role, ceo_bonus, f"实盘统筹签发 (+{pnl_usdt:.2f} USDT)", pnl_usdt, now_ms))
                conn.commit()

            logger.info(
                f"🎉 【CFO 战绩奖励到账】交易 {trade_id} 盈利 +{pnl_usdt:.2f} USDT！"
                f"向 {lead_role} 派发 +{lead_bonus} Tokens，向 {ceo_role} 派发 +{ceo_bonus} Tokens！"
            )
        except Exception as e:
            logger.error(f"CFO 派发战绩奖励异常: {e}")

    def record_successful_hedge(self, symbol: str, reason: str = "成功识别假突破诱多"):
        """空头风控专家成功排雷避险奖励"""
        now_ms = int(time.time() * 1000)
        award_tokens = 8000
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE cfo_employee_quota
                    SET successful_hedges_count = successful_hedges_count + 1,
                        total_bonus_earned = total_bonus_earned + ?,
                        remaining_bonus = remaining_bonus + ?,
                        updated_at = ?
                    WHERE role = 'BEAR_CRITIC'
                """, (award_tokens, award_tokens, now_ms))
                cursor.execute("""
                    INSERT INTO cfo_reward_log (trade_id, role, reward_tokens, reason, pnl_usdt, timestamp)
                    VALUES (?, 'BEAR_CRITIC', ?, ?, 0.0, ?)
                """, (f"hedge_{symbol}_{now_ms}", award_tokens, f"避险排雷: {reason}", now_ms))
                conn.commit()
            logger.info(f"🛡️ 【CFO 避险嘉奖】空头风控专家成功排雷 ({symbol}: {reason})，奖励 +{award_tokens} Tokens！")
        except Exception as e:
            logger.error(f"CFO 记录避险奖励异常: {e}")

    def consume_bonus_quota_for_task(self, symbol: str, amount: int = 1200) -> int:
        """
        在深度研判任务中核销员工战绩奖励额度以解锁深度思考链
        若所有员工均无额度，则返回 0 (回退到普通模式)
        """
        now_ms = int(time.time() * 1000)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT role, remaining_bonus FROM cfo_employee_quota 
                    WHERE remaining_bonus >= ? ORDER BY remaining_bonus DESC LIMIT 1
                """, (amount,))
                row = cursor.fetchone()
                if row:
                    role, remain = row
                    cursor.execute("""
                        UPDATE cfo_employee_quota
                        SET remaining_bonus = remaining_bonus - ?, updated_at = ?
                        WHERE role = ?
                    """, (amount, now_ms, role))
                    conn.commit()
                    logger.debug(f"【CFO 额度核销】标的 {symbol} 任务核销 {role} 奖励额度 -{amount} Tokens (剩余: {remain - amount})")
                    return amount
        except Exception as e:
            logger.error(f"CFO 核销额度异常: {e}")
        return 0
