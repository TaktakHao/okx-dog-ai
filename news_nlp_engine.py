"""
OKX-Dog AI 决策大脑 - 突发新闻 NLP 结构化解析与多维度情感打分引擎
模块: okx-dog-ai/news_nlp_engine.py
角色: AI 与量化算法工程师 (agency-ai-engineer) / 提示词工程师 (agency-prompt-engineer)
功能:
1. 极速新闻 NLP 结构化特征提取
2. 情感偏向打分 (-1.0 极度利空 ~ +1.0 极度利好)
3. 突发事件时效评级 (P0 突发黑天鹅/暴利好, P1 重要, P2 一般)
4. 谣言/FUD 与不实噪音识别过滤
5. 精炼中文一句话核心提炼
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("okx_dog.ai.news_nlp")

NEWS_ANALYSIS_SYSTEM_PROMPT = """你是一个资深加密货币与全球宏观量化分析师。
你的任务是对抓取到的最新财经与加密热点资讯进行极速 NLP 结构化解析，评估其对行情的影响。
必须严格输出满足以下 JSON Schema 的 JSON 对象，禁止输出任何多余 Markdown 格式或解释说明。

JSON 字段要求:
1. sentiment_score: 浮点数，范围 -1.0 到 +1.0 (-1.0 极度利空，0.0 中性，+1.0 极度利好)
2. urgency: 字符串，"P0" (突发黑天鹅/暴利好/监管重磅), "P1" (重要宏观/行业大事件), "P2" (常规新闻/日常快讯)
3. related_symbols: 字符串数组，如 ["BTC", "ETH", "SOL", "USDT"]
4. is_fud_or_rumor: 布尔值，是否为未证实传闻、小道消息或明显恶意 FUD
5. summary_zh: 字符串，用精炼简明的一句话提炼该新闻对交易的核心影响 (40字以内)
"""

class NewsNLPEngine:
    """新闻 NLP 情感与事件分析引擎"""

    @staticmethod
    def create_prompt(title: str, content: Optional[str] = None) -> str:
        """构造分析 Prompt"""
        text = f"新闻标题: {title}
"
        if content:
            text += f"正文/摘要: {content}
"
        return text

    @staticmethod
    def parse_llm_output(raw_output: str) -> Dict[str, Any]:
        """解析并校验 LLM 输出的结构化情感结果"""
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("
")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "
".join(lines).strip()

        try:
            data = json.loads(cleaned)
            score = float(data.get("sentiment_score", 0.0))
            score = max(-1.0, min(1.0, score))
            urgency = str(data.get("urgency", "P2")).upper()
            if urgency not in ["P0", "P1", "P2"]:
                urgency = "P2"
            related = data.get("related_symbols", [])
            if not isinstance(related, list):
                related = []
            is_fud = bool(data.get("is_fud_or_rumor", False))
            summary_zh = str(data.get("summary_zh", ""))

            return {
                "sentiment_score": score,
                "urgency": urgency,
                "related_symbols": related,
                "is_fud_or_rumor": is_fud,
                "summary_zh": summary_zh
            }
        except Exception as e:
            logger.warning(f"解析新闻 LLM 情感打分失败，降级为规则启发式打分: {e}")
            return NewsNLPEngine.heuristic_fallback(raw_output)

    @staticmethod
    def heuristic_fallback(title: str) -> Dict[str, Any]:
        """轻量启发式规则备用打分"""
        lower = title.lower()
        score = 0.0
        urgency = "P2"
        is_fud = False
        symbols = []

        bull_words = ["surge", "bull", "breakout", "all-time high", "ath", "etf approved", "adoption", "rally", "rate cut", "降息", "大涨", "突破", "新高", "获批"]
        bear_words = ["crash", "dump", "hack", "lawsuit", "sec charges", "ban", "rate hike", "liquidation", "加息", "大跌", "暴跌", "被盗", "起诉", "破产", "清算"]
        p0_words = ["sec charges", "hack", "ban", "emergency rate", "fomc cut 50bps", "black swan", "黑天鹅", "突发暴跌", "崩盘", "暂停提款"]

        for w in bull_words:
            if w in lower:
                score += 0.4
        for w in bear_words:
            if w in lower:
                score -= 0.4
        for w in p0_words:
            if w in lower:
                urgency = "P0"

        if any(s in lower for s in ["btc", "bitcoin", "大饼"]):
            symbols.append("BTC")
        if any(s in lower for s in ["eth", "ethereum", "以太"]):
            symbols.append("ETH")
        if any(s in lower for s in ["sol", "solana"]):
            symbols.append("SOL")

        score = max(-1.0, min(1.0, score))
        return {
            "sentiment_score": round(score, 2),
            "urgency": urgency,
            "related_symbols": symbols,
            "is_fud_or_rumor": is_fud,
            "summary_zh": title[:40]
        }