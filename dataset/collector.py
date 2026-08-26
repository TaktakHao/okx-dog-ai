"""
OKX-Dog 量化决策数据集沉淀与盘口真实反馈收集器
模块: okx-dog-ai/dataset/collector.py
角色: AI 与量化算法工程师 (AI & Quant Engineer)

功能:
1. 自动记录每次触发决策时的完整多周期特征快照、6 专家论证与资深思考链 (<think>)；
2. 异步跟踪记录盘口后续真实的 K 线走势、最大回撤与实际盈亏结算；
3. 按照量化现实标准打上正负标签 (Chosen / Rejected)，为 SFT 与 DPO 蒸馏提供高质量数据源。
"""

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("okx_dog.ai.dataset.collector")

DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = DATASET_DIR / "decision_dataset.db"


class DatasetCollector:
    """
    量化决策数据收集与真实盘口收益沉淀器
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        self._init_sqlite()

    def _get_connection(self) -> sqlite3.Connection:
        """获取 SQLite 连接并配置超时"""
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self):
        """初始化数据集 SQLite 表结构"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decision_samples (
                    sample_id TEXT PRIMARY KEY,
                    analysis_id TEXT,
                    timestamp INTEGER,
                    symbol TEXT,
                    current_price REAL,
                    market_features_json TEXT,
                    senior_analysis_json TEXT,
                    raw_thinking_steps_json TEXT,
                    refined_cot TEXT,
                    actual_outcome_json TEXT,
                    label TEXT DEFAULT 'PENDING',
                    pnl_pct REAL DEFAULT 0.0,
                    is_qualified_sft INTEGER DEFAULT 1,
                    created_at INTEGER
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_samples_symbol_time 
                ON decision_samples(symbol, timestamp);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_samples_label 
                ON decision_samples(label, is_qualified_sft);
            """)
            conn.commit()
            conn.close()
            logger.info("数据集 SQLite 存储库就绪: %s", self.db_path)
        except Exception as e:
            logger.error("初始化数据集 SQLite 异常: %s", e)

    def record_decision_sample(
        self,
        symbol: str,
        current_price: float,
        market_snapshot: Dict[str, Any],
        final_response: Dict[str, Any],
        thinking_steps: Optional[List[Dict[str, Any]]] = None,
        analysis_id: Optional[str] = None,
    ) -> str:
        """
        记录一次决策发生的完整上下文样本
        """
        sample_id = f"samp_{uuid.uuid4().hex[:12]}"
        now_ms = int(time.time() * 1000)
        analysis_id = analysis_id or final_response.get("analysis_id") or str(uuid.uuid4())

        # 提炼核心技术特征（剔除巨量重复字段，保留高价值指标）
        clean_features = self._extract_clean_features(market_snapshot)

        # 检查是否满足基本的 SFT 质量初筛（非全空决策，JSON 契约完整）
        action = final_response.get("action", "HOLD")
        confidence = float(final_response.get("confidence", 0.0))
        is_qualified = 1 if (action in ["BUY", "SELL", "HOLD"] and confidence >= 0.0) else 0

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO decision_samples (
                    sample_id, analysis_id, timestamp, symbol, current_price,
                    market_features_json, senior_analysis_json,
                    raw_thinking_steps_json, refined_cot, actual_outcome_json,
                    label, pnl_pct, is_qualified_sft, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 'PENDING', 0.0, ?, ?)
            """, (
                sample_id,
                analysis_id,
                now_ms,
                symbol,
                current_price,
                json.dumps(clean_features, ensure_ascii=False),
                json.dumps(final_response, ensure_ascii=False),
                json.dumps(thinking_steps or [], ensure_ascii=False),
                is_qualified,
                now_ms,
            ))
            conn.commit()
            conn.close()
            logger.info("成功捕获量化决策样本: sample_id=%s, symbol=%s, action=%s", sample_id, symbol, action)
            return sample_id
        except Exception as e:
            logger.error("记录决策样本异常: %s", e)
            return ""

    def update_sample_outcome(
        self,
        sample_id_or_analysis_id: str,
        pnl_pct: float,
        max_drawdown_pct: float = 0.0,
        hit_tp: bool = False,
        hit_sl: bool = False,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        根据盘口后续结算结果更新样本真实盈亏标签 (用于 DPO 偏好对齐)
        - 盈利且风控合规 -> 'CHOSEN' (正样本)
        - 亏损打止损或假突破被套 -> 'REJECTED' (负样本)
        """
        outcome_data = {
            "pnl_pct": pnl_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "hit_tp": hit_tp,
            "hit_sl": hit_sl,
            "settled_time": int(time.time() * 1000),
            **(extra_metrics or {})
        }

        # 智能判定正负样本标签
        if hit_tp or pnl_pct >= 0.015:  # 盈利 > 1.5% 或触碰止盈
            label = "CHOSEN"
        elif hit_sl or pnl_pct <= -0.010:  # 触碰止损或亏损 > 1.0%
            label = "REJECTED"
        else:
            label = "NEUTRAL"

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE decision_samples
                SET actual_outcome_json = ?,
                    label = ?,
                    pnl_pct = ?
                WHERE sample_id = ? OR analysis_id = ?
            """, (
                json.dumps(outcome_data, ensure_ascii=False),
                label,
                pnl_pct,
                sample_id_or_analysis_id,
                sample_id_or_analysis_id,
            ))
            updated_count = cursor.rowcount
            conn.commit()
            conn.close()
            logger.info(
                "更新样本真实收益反馈: id=%s, label=%s, pnl=%.2f%%, updated=%d",
                sample_id_or_analysis_id, label, pnl_pct * 100, updated_count
            )
            return updated_count > 0
        except Exception as e:
            logger.error("更新样本真实收益异常: %s", e)
            return False

    def get_unrefined_samples(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取尚未经过大模型精简提炼的原始样本"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM decision_samples
                WHERE refined_cot IS NULL AND is_qualified_sft = 1
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            results = [dict(r) for r in rows]
            conn.close()
            return results
        except Exception as e:
            logger.error("查询待提炼样本异常: %s", e)
            return []

    def update_refined_cot(self, sample_id: str, refined_cot: str) -> bool:
        """更新经大模型提炼后的高质量思考链"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE decision_samples
                SET refined_cot = ?
                WHERE sample_id = ?
            """, (refined_cot, sample_id))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("保存提炼思考链异常: %s", e)
            return False

    def _extract_clean_features(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """从原始行情快照提取高密度结构化特征"""
        clean = {
            "symbol": snapshot.get("symbol", "BTC-USDT-SWAP"),
            "current_price": snapshot.get("current_price", 0.0),
            "funding_rate": snapshot.get("funding_rate", 0.0),
            "open_interest": snapshot.get("open_interest", 0.0),
        }
        # 提取指标
        if "indicators" in snapshot and isinstance(snapshot["indicators"], dict):
            clean["indicators"] = snapshot["indicators"]
        # 提取微观盘口
        if "microstructure" in snapshot and isinstance(snapshot["microstructure"], dict):
            clean["orderbook_imbalance"] = snapshot["microstructure"].get("imbalance_ratio", 0.0)
            clean["spread_bps"] = snapshot["microstructure"].get("spread_bps", 0.0)
        # 提取链上与宏观
        if "onchain" in snapshot and isinstance(snapshot["onchain"], dict):
            clean["whale_net_flow"] = snapshot["onchain"].get("net_flow_usd", 0.0)
        if "macro" in snapshot and isinstance(snapshot["macro"], dict):
            clean["macro_risk_level"] = snapshot["macro"].get("risk_level", "NORMAL")
        return clean


# 单例实例
dataset_collector = DatasetCollector()
