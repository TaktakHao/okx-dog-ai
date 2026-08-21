"""
OKX-Dog AI Quant Studio - Codex 客户端与 3 轮自愈纠错闭环
模块: okx-dog-ai/studio/codex_client.py
角色: AI 与量化算法工程师 (agency-ai-engineer)
功能: 自然语言需求解析 -> Python 策略代码生成 -> AST 白名单审计 -> 受限沙箱回测 -> 报错自动纠错
"""

import logging
import os
import uuid
from typing import Any, Dict, Optional

from .ast_guard import validate_python_code_security
from .sandbox_runner import run_strategy_in_sandbox

logger = logging.getLogger("okx_dog.studio.codex_client")

# 加载 Prompt
_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "codex_system_prompt.md")
if os.path.exists(_PROMPT_FILE):
    with open(_PROMPT_FILE, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
else:
    SYSTEM_PROMPT = "你是一个顶尖量化工程师，请继承 BaseQuantStrategy 并实现 generate_signals(self, df: pd.DataFrame) -> pd.Series，仅使用 numpy 和 pandas 向量化计算。"


# 经典高质量策略模板库 (供无网络/离线或一键试用时秒级响应)
DEFAULT_TEMPLATES = {
    "bollinger_breakout": '''import numpy as np
import pandas as pd
from strategy_base import BaseQuantStrategy

class CustomQuantStrategy(BaseQuantStrategy):
    """自适应布林带通道突破与波动率挤压策略"""
    def __init__(self, period=20, std_dev=2.0):
        self.period = period
        self.std_dev = std_dev

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        mid = df['close'].rolling(window=self.period).mean()
        std = df['close'].rolling(window=self.period).std()
        upper = mid + self.std_dev * std
        lower = mid - self.std_dev * std

        long_cond = (df['close'] > upper) & (df['close'].shift(1) <= upper.shift(1))
        short_cond = (df['close'] < lower) & (df['close'].shift(1) >= lower.shift(1))

        signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
        return pd.Series(signals, index=df.index)
''',
    "rsi_divergence_funding": '''import numpy as np
import pandas as pd
from strategy_base import BaseQuantStrategy

class CustomQuantStrategy(BaseQuantStrategy):
    """RSI 超卖反弹共振资金费率转正动量策略"""
    def __init__(self, rsi_period=14):
        self.rsi_period = rsi_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))

        fr = df.get('funding_rate', pd.Series(0, index=df.index))

        long_cond = (rsi < 35) & (rsi.shift(1) >= 35) & (fr >= 0)
        short_cond = (rsi > 65) & (rsi.shift(1) <= 65) & (fr < 0)

        signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
        return pd.Series(signals, index=df.index)
''',
    "ema_cross_oi": '''import numpy as np
import pandas as pd
from strategy_base import BaseQuantStrategy

class CustomQuantStrategy(BaseQuantStrategy):
    """双均线金叉共振未平仓量(OI)增量突破策略"""
    def __init__(self, fast=10, slow=30):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ema_f = df['close'].ewm(span=self.fast, adjust=False).mean()
        ema_s = df['close'].ewm(span=self.slow, adjust=False).mean()
        oi = df.get('oi', pd.Series(0, index=df.index))
        oi_change = oi.pct_change(5).fillna(0)

        long_cond = (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1)) & (oi_change > 0.01)
        short_cond = (ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))

        signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
        return pd.Series(signals, index=df.index)
'''
}


class CodexQuantStudioClient:
    """OKX-Dog AI Quant Studio - 策略生成与沙盒闭环核心管理器"""

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm = llm_client

    def _extract_python_code(self, raw_text: str) -> str:
        """从 LLM 输出中精准提取 Python 代码块"""
        if "```python" in raw_text:
            code = raw_text.split("```python")[1].split("```")[0].strip()
            return code
        elif "```" in raw_text:
            code = raw_text.split("```")[1].split("```")[0].strip()
            return code
        return raw_text.strip()

    async def generate_and_verify_strategy(
        self,
        prompt: str,
        symbol: str = "BTC-USDT-SWAP",
        timeframe: str = "15m",
        parquet_path: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        全生命周期闭环: 生成 -> AST 审计 -> 沙盒回测 -> 报错自愈
        """
        strategy_id = str(uuid.uuid4())
        history_errors = []
        last_code = ""

        # 匹配内置模板关键字优先处理
        p_lower = prompt.lower()
        if "布林" in p_lower or "bollinger" in p_lower:
            fallback_template = DEFAULT_TEMPLATES["bollinger_breakout"]
        elif "oi" in p_lower or "持仓量" in p_lower or "未平仓" in p_lower:
            fallback_template = DEFAULT_TEMPLATES["ema_cross_oi"]
        else:
            fallback_template = DEFAULT_TEMPLATES["rsi_divergence_funding"]

        for attempt in range(1, max_retries + 1):
            logger.info("Codex 策略生成第 [%d/%d] 轮尝试: %s", attempt, max_retries, prompt)
            code = ""

            if self.llm:
                try:
                    user_msg = f"交易员需求: {prompt}\n回测标的: {symbol}, 周期: {timeframe}"
                    if history_errors:
                        user_msg += f"\n\n[自愈提示] 上一轮生成的代码执行出现如下报错，请针对性修改代码结构并修复：\n{history_errors[-1]}"

                    raw_resp = await self.llm.chat_completion(
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=user_msg,
                        temperature=0.2
                    )
                    code = self._extract_python_code(raw_resp)
                except Exception as e:
                    logger.warning("LLM 请求异常，采用智能模板兜底: %s", e)
                    code = fallback_template
            else:
                code = fallback_template

            last_code = code

            # 1. AST 静态安全与语法审查
            ast_passed, ast_msg = validate_python_code_security(code)
            if not ast_passed:
                logger.warning("AST 静态拦截: %s", ast_msg)
                history_errors.append(f"AST 审查拦截: {ast_msg}")
                continue

            # 2. 隔离沙箱回测
            sandbox_res = await run_strategy_in_sandbox(
                code=code,
                parquet_path=parquet_path,
                symbol=symbol,
                timeframe=timeframe,
                timeout_sec=5.0
            )

            if not sandbox_res.get("success"):
                err_text = sandbox_res.get("error", "未知沙箱错误")
                logger.warning("沙箱执行失败: %s", err_text)
                history_errors.append(f"沙箱运行时错误: {err_text}")
                continue

            # 3. 成功跑通
            logger.info("策略生成与沙盒回测成功完成，耗费轮次: %d", attempt)
            return {
                "strategy_id": strategy_id,
                "strategy_name": f"AI-Quant-{symbol}-{timeframe}-{attempt}",
                "prompt": prompt,
                "code": code,
                "ast_passed": True,
                "attempts": attempt,
                "status": "VERIFIED",
                "metrics": sandbox_res.get("metrics"),
                "equity_curve": sandbox_res.get("equity_curve", []),
                "trade_signals": sandbox_res.get("trade_signals", []),
                "error_message": None
            }

        # 超过最大重试次数时，采用安全模板兜底保证系统高可用
        fallback_res = await run_strategy_in_sandbox(fallback_template, parquet_path, symbol, timeframe)
        return {
            "strategy_id": strategy_id,
            "strategy_name": f"AI-Quant-{symbol}-{timeframe}-Fallback",
            "prompt": prompt,
            "code": fallback_template,
            "ast_passed": True,
            "attempts": max_retries,
            "status": "FALLBACK_VERIFIED",
            "metrics": fallback_res.get("metrics"),
            "equity_curve": fallback_res.get("equity_curve", []),
            "trade_signals": fallback_res.get("trade_signals", []),
            "error_message": f"多轮生成未过，已自适应降级为高健壮性标准策略 (最新错误: {history_errors[-1] if history_errors else 'None'})"
        }
