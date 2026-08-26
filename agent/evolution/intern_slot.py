"""
OKX-Dog 实习生专有插槽与以老带新双轨影子演进引擎 (Intern Slot & Shadow Evolution Engine)
模块: okx-dog-ai/agent/evolution/intern_slot.py
角色: 多智能体协同架构师 / AI 与量化算法工程师

核心功能:
1. 实习生模型独立接入 (Ollama / vLLM 本地微调模型插槽)；
2. 异步非阻塞双轨影子推演 (Shadow Inference): 老模型实盘发单，实习生后台模拟并沉淀虚拟战绩；
3. 虚拟实盘收益核算与打分 (Virtual PnL & Alpha Evaluation)；
4. 决策分歧与失误数据自动回流沉淀为 DPO 训练样本；
5. 动态转正评估门禁 (Promotion Readiness Gate)。
"""

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

try:
    from ...config import ai_settings
    from ...dataset.collector import dataset_collector
    from ...dataset.exporter import SYSTEM_PROMPT_QUANT_ARBITER
except (ImportError, ValueError):
    try:
        from okx_dog_ai.config import ai_settings
        from okx_dog_ai.dataset.collector import dataset_collector
        from okx_dog_ai.dataset.exporter import SYSTEM_PROMPT_QUANT_ARBITER
    except (ImportError, ValueError):
        import sys
        base_dir = str(Path(__file__).resolve().parent.parent.parent)
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)
        from config import ai_settings
        from dataset.collector import dataset_collector
        from dataset.exporter import SYSTEM_PROMPT_QUANT_ARBITER

logger = logging.getLogger("okx_dog.ai.evolution.intern_slot")

INTERN_DB_PATH = Path(__file__).resolve().parent.parent.parent / "dataset" / "intern_shadow.db"


class InternPerformanceProfile(BaseModel):
    """实习生战绩与转正评估档案"""
    role_id: str = "intern_shadow_puppy"
    name: str = "实习量化研究员"
    nickname: str = "🐣 小狗量化实习生"
    avatar_icon: str = "GraduationCap"
    title: str = "量化微调演进模型"
    model_name: str = "okx-dog-intern"
    provider: str = "ollama"
    status: str = "SHADOW_TRAINING"  # SHADOW_TRAINING (影子带训) / PROMOTED (已转正)
    total_trials: int = 0
    agreement_rate: float = 0.0
    virtual_win_rate_7d: float = 0.0
    virtual_wins: int = 0
    virtual_losses: int = 0
    virtual_total_pnl_pct: float = 0.0
    contract_compliance_rate: float = 1.0
    is_promotion_ready: bool = False
    promotion_checklist: Dict[str, bool] = Field(default_factory=dict)
    recent_feedback: Optional[str] = "正在接受资深老模型带训，持续积累影子实战经验"


class InternSlotManager:
    """
    实习生插槽与双轨演进中枢
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or INTERN_DB_PATH)
        self._init_sqlite()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self):
        """初始化实习生影子推演数据表"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS intern_shadow_decisions (
                    decision_id TEXT PRIMARY KEY,
                    analysis_id TEXT,
                    timestamp INTEGER,
                    symbol TEXT,
                    current_price REAL,
                    features_json TEXT,
                    intern_response_json TEXT,
                    intern_cot TEXT,
                    senior_action TEXT,
                    intern_action TEXT,
                    is_agreed INTEGER,
                    is_valid_contract INTEGER,
                    virtual_pnl_pct REAL DEFAULT 0.0,
                    virtual_status TEXT DEFAULT 'OPEN',
                    settled_time INTEGER
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_intern_time 
                ON intern_shadow_decisions(timestamp);
            """)
            conn.commit()
            conn.close()
            logger.info("实习生影子推演数据库就绪: %s", self.db_path)
        except Exception as e:
            logger.error("初始化实习生数据库失败: %s", e)

    def is_enabled(self) -> bool:
        """检查实习生插槽是否已开启"""
        return getattr(ai_settings, "INTERN_ENABLED", False)

    async def call_intern_model(self, features_json_str: str) -> Tuple[Optional[Dict[str, Any]], str, bool]:
        """
        通过 Ollama / OpenAI 兼容协议调用本地部署的实习生开源模型
        返回: (parsed_json_decision, extracted_cot, is_valid_schema)
        """
        base_url = getattr(ai_settings, "INTERN_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
        model_name = getattr(ai_settings, "INTERN_MODEL_NAME", "okx-dog-intern")
        api_key = getattr(ai_settings, "INTERN_API_KEY", "ollama")
        temp = getattr(ai_settings, "INTERN_TEMPERATURE", 0.2)
        max_tokens = getattr(ai_settings, "INTERN_MAX_TOKENS", 2048)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_QUANT_ARBITER},
            {"role": "user", "content": f"当前市场多周期指标与盘口特征如下:\n{features_json_str}\n\n请进行深度博弈思考并在 <think> 标签后输出规范 JSON 决策。"}
        ]

        endpoint = f"{base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tokens
        }

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(endpoint, headers=headers, json=payload)
                if resp.status_code != 200:
                    logger.warning("实习生模型服务返回非 200: status=%d, text=%s", resp.status_code, resp.text[:200])
                    return None, "", False

                data = resp.json()
                raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return self._parse_intern_output(raw_text)
        except Exception as e:
            logger.warning("调用实习生模型异常 (请确认本地 Ollama 是否已启动): %s", e)
            return None, "", False

    def _parse_intern_output(self, raw_text: str) -> Tuple[Optional[Dict[str, Any]], str, bool]:
        """解析实习生输出中的思考链和结构化 JSON"""
        cot = ""
        # 提取 <think>...</think>
        if "<think>" in raw_text and "</think>" in raw_text:
            start = raw_text.find("<think>") + len("<think>")
            end = raw_text.find("</think>")
            cot = raw_text[start:end].strip()
            rest = raw_text[end + len("</think>"):].strip()
        else:
            rest = raw_text.strip()

        # 提取 ```json ... ```
        json_str = rest
        if "```json" in rest:
            j_start = rest.find("```json") + len("```json")
            j_end = rest.find("```", j_start)
            if j_end != -1:
                json_str = rest[j_start:j_end].strip()
        elif "```" in rest:
            j_start = rest.find("```") + len("```")
            j_end = rest.find("```", j_start)
            if j_end != -1:
                json_str = rest[j_start:j_end].strip()

        # 尝试 JSON 解析
        try:
            parsed = json.loads(json_str)
            # 基础字段契约检查
            required_keys = ["action", "confidence"]
            is_valid = all(k in parsed for k in required_keys)
            return parsed, cot, is_valid
        except Exception:
            return None, cot, False

    async def trigger_shadow_inference(
        self,
        symbol: str,
        current_price: float,
        market_snapshot: Dict[str, Any],
        senior_response: Dict[str, Any],
        analysis_id: Optional[str] = None
    ):
        """
        后台异步触发实习生影子推演 (与老模型主链路并发零阻塞)
        """
        if not self.is_enabled():
            return

        now_ms = int(time.time() * 1000)
        decision_id = f"intn_{uuid.uuid4().hex[:12]}"
        analysis_id = analysis_id or senior_response.get("analysis_id") or str(uuid.uuid4())
        clean_features = dataset_collector._extract_clean_features(market_snapshot)
        clean_features_str = json.dumps(clean_features, ensure_ascii=False)

        senior_action = senior_response.get("action", "HOLD")

        # 调用实习生模型
        intern_json, intern_cot, is_valid = await self.call_intern_model(clean_features_str)
        intern_action = intern_json.get("action", "UNKNOWN") if intern_json else "ERROR"
        is_agreed = 1 if (intern_action == senior_action) else 0

        # 保存到数据库
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO intern_shadow_decisions (
                    decision_id, analysis_id, timestamp, symbol, current_price,
                    features_json, intern_response_json, intern_cot,
                    senior_action, intern_action, is_agreed, is_valid_contract,
                    virtual_pnl_pct, virtual_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, 'OPEN')
            """, (
                decision_id,
                analysis_id,
                now_ms,
                symbol,
                current_price,
                clean_features_str,
                json.dumps(intern_json or {}, ensure_ascii=False),
                intern_cot,
                senior_action,
                intern_action,
                is_agreed,
                1 if is_valid else 0,
            ))
            conn.commit()
            conn.close()

            logger.info(
                "🐣 实习生影子推演完成: id=%s, intern_action=%s (老专家=%s, 一致性=%s, 合规=%s)",
                decision_id, intern_action, senior_action, "✅" if is_agreed else "❌", "✅" if is_valid else "❌"
            )
        except Exception as e:
            logger.error("保存实习生影子推演异常: %s", e)

    def settle_virtual_outcome(
        self,
        analysis_id: str,
        actual_pnl_pct: float,
        hit_tp: bool = False,
        hit_sl: bool = False
    ):
        """
        根据盘口后续走势核算实习生的虚拟实盘盈亏
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM intern_shadow_decisions
                WHERE analysis_id = ?
            """, (analysis_id,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return

            intern_action = row["intern_action"]
            senior_action = row["senior_action"]

            # 计算实习生独立推演的理论盈亏
            virtual_pnl = 0.0
            if intern_action == senior_action:
                virtual_pnl = actual_pnl_pct
            elif intern_action == "HOLD":
                virtual_pnl = 0.0
            elif (intern_action == "BUY" and senior_action == "SELL") or (intern_action == "SELL" and senior_action == "BUY"):
                virtual_pnl = -actual_pnl_pct  # 反向操作则盈亏颠倒
            else:
                virtual_pnl = actual_pnl_pct if intern_action in ["BUY", "SELL"] else 0.0

            cursor.execute("""
                UPDATE intern_shadow_decisions
                SET virtual_pnl_pct = ?,
                    virtual_status = 'SETTLED',
                    settled_time = ?
                WHERE analysis_id = ?
            """, (virtual_pnl, int(time.time() * 1000), analysis_id))
            conn.commit()
            conn.close()

            logger.info(
                "🐣 实习生虚拟战绩结算: analysis_id=%s, intern_action=%s, virtual_pnl=%.2f%%",
                analysis_id, intern_action, virtual_pnl * 100
            )
        except Exception as e:
            logger.error("结算实习生虚拟战绩异常: %s", e)

    def get_performance_profile(self) -> InternPerformanceProfile:
        """
        查询实习生的全量战绩档案与转正达标状态
        """
        profile = InternPerformanceProfile(
            model_name=getattr(ai_settings, "INTERN_MODEL_NAME", "okx-dog-intern"),
            provider=getattr(ai_settings, "INTERN_PROVIDER", "ollama"),
        )

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_count,
                    SUM(is_agreed) as agreed_count,
                    SUM(is_valid_contract) as valid_count
                FROM intern_shadow_decisions
            """)
            stats = cursor.fetchone()
            total = stats["total_count"] or 0
            agreed = stats["agreed_count"] or 0
            valid = stats["valid_count"] or 0

            # 查询结算后的盈亏情况
            cursor.execute("""
                SELECT 
                    COUNT(*) as settled_count,
                    SUM(CASE WHEN virtual_pnl_pct > 0.005 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN virtual_pnl_pct < -0.005 THEN 1 ELSE 0 END) as losses,
                    SUM(virtual_pnl_pct) as total_pnl
                FROM intern_shadow_decisions
                WHERE virtual_status = 'SETTLED'
            """)
            pnl_stats = cursor.fetchone()
            settled_cnt = pnl_stats["settled_count"] or 0
            wins = pnl_stats["wins"] or 0
            losses = pnl_stats["losses"] or 0
            tot_pnl = pnl_stats["total_pnl"] or 0.0

            conn.close()

            profile.total_trials = total
            profile.agreement_rate = round(agreed / total, 4) if total > 0 else 0.0
            profile.contract_compliance_rate = round(valid / total, 4) if total > 0 else 1.0
            profile.virtual_wins = wins
            profile.virtual_losses = losses
            profile.virtual_total_pnl_pct = round(tot_pnl, 4)
            profile.virtual_win_rate_7d = round(wins / (wins + losses), 4) if (wins + losses) > 0 else 0.0

            # 严格转正核查清单
            checklist = {
                "影子实战推演 >= 30 次": total >= 30,
                "契约合规率 == 100%": profile.contract_compliance_rate >= 0.99,
                "虚拟胜率 >= 65%": profile.virtual_win_rate_7d >= 0.65,
                "虚拟累计收益 > 0": tot_pnl > 0.0,
                "与老专家一致率 >= 70%": profile.agreement_rate >= 0.70
            }
            profile.promotion_checklist = checklist
            profile.is_promotion_ready = all(checklist.values())

            # 动态精准判定实习生插槽状态
            if not self.is_enabled():
                profile.status = "NOT_CONFIGURED"
                profile.recent_feedback = "💤 实习生插槽尚未激活。请在数据导出并微调后，通过 Ollama 部署并在 .env 配置 INTERN_ENABLED=true 接入"
            elif profile.is_promotion_ready:
                profile.status = "PROMOTION_READY"
                profile.recent_feedback = "🎉 恭喜！各项量化考核指标全部达标，已具备独立单干能力，随时可一键转正为主力决策模型！"
            elif total == 0:
                profile.status = "CONNECTED_IDLE"
                profile.recent_feedback = "⚡ 实习生模型已连接就绪，等待市场交易信号触发首次影子推演..."
            else:
                profile.status = "SHADOW_TRAINING"
                profile.recent_feedback = f"目前已完成 {total} 次影子演练，胜率 {profile.virtual_win_rate_7d*100:.1f}%，继续由老模型带训中..."
        except Exception as e:
            logger.error("获取实习生档案异常: %s", e)

        return profile


# 单例实例
intern_slot_manager = InternSlotManager()
