"""
OKX-Dog 容错自愈解析器与思维链提取中枢
模块: okx-dog-ai/parser.py

特性:
1. 5 层容错自愈解析器 (RobustJSONParser):
   - Layer 1: 原生标准 json.loads
   - Layer 2: Markdown 代码块提取 (```json ... ``` 或 ``` ...)
   - Layer 3: 正则/边界提取最外层 {...} JSON 实体
   - Layer 4: 缺陷语法自愈修复器 (去除注释、修复尾逗号、补齐引号、截断修复与字符栈自动闭合)
   - Layer 5: Pydantic 数据模型严格校验与安全兜底 (HOLD_WAIT 绝对安全回退)
2. ThoughtStreamExtractor: 针对 SSE 流式生成将 <think>...</think> 思维链与 JSON 决策内容干净分离。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

try:
    from okx_dog_ai.schemas import (
        AIAnalysisResponse,
        TimeframeAnalysis,
        TimeframeDetail,
        TimeframeTrend,
        DerivativesSentiment,
        FundingRateBias,
        AISignal,
        SignalAction,
        SignalUrgency,
        TradePlan,
        TakeProfitLevel,
        TradePlanOrderType,
        RiskAssessment,
        MarketRegime,
    )
except (ImportError, Exception):
    try:
        from schemas import (
            AIAnalysisResponse,
            TimeframeAnalysis,
            TimeframeDetail,
            TimeframeTrend,
            DerivativesSentiment,
            FundingRateBias,
            AISignal,
            SignalAction,
            SignalUrgency,
            TradePlan,
            TakeProfitLevel,
            TradePlanOrderType,
            RiskAssessment,
            MarketRegime,
        )
    except (ImportError, Exception):
        from .schemas import (
            AIAnalysisResponse,
            TimeframeAnalysis,
            TimeframeDetail,
            TimeframeTrend,
            DerivativesSentiment,
            FundingRateBias,
            AISignal,
            SignalAction,
            SignalUrgency,
            TradePlan,
            TakeProfitLevel,
            TradePlanOrderType,
            RiskAssessment,
            MarketRegime,
        )

logger = logging.getLogger("okx_dog.ai.parser")


class ThoughtStreamExtractor:
    """
    针对 SSE 流式推送的思维链 (<think>...</think>) 增量分离状态机
    """

    def __init__(self):
        self.buffer = ""
        self.in_think = False
        self.think_finished = False
        self.accumulated_thinking: List[str] = []
        self.accumulated_content: List[str] = []

    def feed_chunk(self, chunk: str) -> List[Tuple[str, str]]:
        """
        输入新的文本 chunk，返回分类后的增量事件列表: [("think", delta), ("content", delta), ...]
        """
        if not chunk:
            return []

        self.buffer += chunk
        events: List[Tuple[str, str]] = []

        while self.buffer:
            if not self.in_think and not self.think_finished:
                # 处于初始状态，寻找 <think> 标签
                think_start = self.buffer.find("<think>")
                if think_start != -1:
                    # <think> 之前若有内容，作为 content 发送
                    before = self.buffer[:think_start]
                    if before:
                        self.accumulated_content.append(before)
                        events.append(("content", before))
                    self.in_think = True
                    self.buffer = self.buffer[think_start + 7:]
                    continue
                else:
                    # 未发现 <think> 标签，检查是否有可能被截断的 "<think"
                    possible_tag = False
                    for i in range(1, 7):
                        if self.buffer.endswith("<think>"[:i]):
                            possible_tag = True
                            break
                    if possible_tag:
                        # 暂时留在缓冲区等待下一个 chunk
                        break
                    else:
                        # 没有 <think> 标签，全部作为 content 发出
                        self.accumulated_content.append(self.buffer)
                        events.append(("content", self.buffer))
                        self.buffer = ""
                        break

            elif self.in_think:
                # 处于 <think> 内部，寻找 </think>
                think_end = self.buffer.find("</think>")
                if think_end != -1:
                    think_text = self.buffer[:think_end]
                    if think_text:
                        self.accumulated_thinking.append(think_text)
                        events.append(("think", think_text))
                    self.in_think = False
                    self.think_finished = True
                    self.buffer = self.buffer[think_end + 8:]
                    continue
                else:
                    # 检查尾部是否有被截断的 "</think"
                    possible_tag_len = 0
                    for i in range(1, 8):
                        if self.buffer.endswith("</think>"[:i]):
                            possible_tag_len = i
                            break
                    if possible_tag_len > 0:
                        emit_part = self.buffer[:-possible_tag_len]
                        if emit_part:
                            self.accumulated_thinking.append(emit_part)
                            events.append(("think", emit_part))
                        self.buffer = self.buffer[-possible_tag_len:]
                        break
                    else:
                        self.accumulated_thinking.append(self.buffer)
                        events.append(("think", self.buffer))
                        self.buffer = ""
                        break

            else:
                # 思考已完成，全部属于 JSON content
                self.accumulated_content.append(self.buffer)
                events.append(("content", self.buffer))
                self.buffer = ""
                break

        return events

    def get_final_result(self) -> Tuple[str, str]:
        """流结束时提取完整的 (thinking_content, json_content)"""
        if self.buffer:
            if self.in_think:
                self.accumulated_thinking.append(self.buffer)
            else:
                self.accumulated_content.append(self.buffer)
            self.buffer = ""

        full_thinking = "".join(self.accumulated_thinking).strip()
        full_content = "".join(self.accumulated_content).strip()
        return full_thinking, full_content


class RobustJSONParser:
    """
    5 层容错自愈 JSON 解析器
    """

    @classmethod
    def extract_think_and_content(cls, raw_text: str) -> Tuple[str, str]:
        """静态分离 <think>...</think> 与纯净主体文本"""
        if not raw_text:
            return "", ""

        think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else ""
        clean = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        # 处理可能只有 <think> 没有闭合标签的情况
        if "<think>" in clean and "</think>" not in clean:
            parts = clean.split("<think>", 1)
            thinking = parts[1].strip()
            clean = parts[0].strip()

        return thinking, clean

    @classmethod
    def parse(
        cls,
        raw_text: str,
        symbol: str = "BTC-USDT-SWAP",
        model_used: Optional[str] = None,
        latency_ms: Optional[int] = 0,
        preset_thinking: Optional[str] = None,
    ) -> AIAnalysisResponse:
        """
        执行 5 层容错解析并将结果校验为生产级 AIAnalysisResponse。
        """
        extracted_thinking, content = cls.extract_think_and_content(raw_text)
        final_thinking = preset_thinking or extracted_thinking or None

        parsed_dict: Optional[Dict[str, Any]] = None
        last_error_msg: str = ""

        # --- Layer 1: 原生标准 json.loads ---
        try:
            parsed_dict = json.loads(content)
            logger.debug("Layer 1 原生 json.loads 解析成功")
        except Exception as e:
            last_error_msg = f"Layer 1 失败: {e}"

        # --- Layer 2: Markdown 代码块提取 ---
        if parsed_dict is None:
            md_content = cls._extract_markdown_code_block(content)
            if md_content:
                try:
                    parsed_dict = json.loads(md_content)
                    logger.debug("Layer 2 Markdown 代码块解析成功")
                except Exception as e:
                    last_error_msg = f"Layer 2 失败: {e}"
                    content = md_content

        # --- Layer 3: 正则/边界提取最外层 {...} ---
        if parsed_dict is None:
            boundary_content = cls._extract_outermost_json(content)
            if boundary_content:
                try:
                    parsed_dict = json.loads(boundary_content)
                    logger.debug("Layer 3 边界截取解析成功")
                except Exception as e:
                    last_error_msg = f"Layer 3 失败: {e}"
                    content = boundary_content

        # --- Layer 4: 缺陷自愈修复器 ---
        if parsed_dict is None:
            try:
                repaired_json_str = cls._repair_json_syntax(content)
                parsed_dict = json.loads(repaired_json_str)
                logger.debug("Layer 4 缺陷自愈修复器解析成功")
            except Exception as e:
                last_error_msg = f"Layer 4 自愈修复失败: {e}"

        # --- Layer 5: Pydantic 模型严格校验与安全兜底 ---
        if parsed_dict is not None and isinstance(parsed_dict, dict):
            try:
                normalized_dict = cls._normalize_fields(parsed_dict, symbol)
                response_obj = AIAnalysisResponse.model_validate(normalized_dict)
                response_obj.model_used = model_used
                response_obj.latency_ms = latency_ms
                response_obj.thinking_process = final_thinking
                return response_obj
            except ValidationError as val_err:
                logger.warning("Layer 5 Pydantic 校验失败，尝试智能补全: %s", val_err)
                try:
                    patched_dict = cls._patch_missing_fields(parsed_dict, symbol)
                    response_obj = AIAnalysisResponse.model_validate(patched_dict)
                    response_obj.model_used = model_used
                    response_obj.latency_ms = latency_ms
                    response_obj.thinking_process = final_thinking
                    return response_obj
                except Exception as patch_err:
                    last_error_msg = f"Layer 5 智能补全失败: {patch_err}"

        # 若全部层级均不可恢复，输出绝对安全的系统降级响应
        logger.error("JSON 无法恢复，触发 Layer 5 安全熔断兜底: %s", last_error_msg)
        fallback = cls.create_safe_fallback_response(symbol=symbol, error_msg=last_error_msg)
        fallback.model_used = model_used
        fallback.latency_ms = latency_ms
        fallback.thinking_process = final_thinking
        return fallback

    # =========================================================================
    # 2. 私有自愈修复逻辑 (Layer 2 ~ Layer 4)
    # =========================================================================

    @classmethod
    def _extract_markdown_code_block(cls, text: str) -> Optional[str]:
        """提取 ```json ... ``` 或 ``` ... ``` 内部代码"""
        if "```json" in text:
            parts = text.split("```json", 1)[1]
            if "```" in parts:
                return parts.split("```", 1)[0].strip()
        if "```" in text:
            parts = text.split("```", 1)[1]
            if "```" in parts:
                return parts.split("```", 1)[0].strip()
        return None

    @classmethod
    def _extract_outermost_json(cls, text: str) -> Optional[str]:
        """定位首个 { 与最后一个 } 截取纯 JSON 字符串"""
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1].strip()
        return None

    @classmethod
    def _repair_json_syntax(cls, text: str) -> str:
        """应用多重语法修复规则"""
        if not text:
            return "{}"

        s = text.strip()

        # 1. 移除单行 // 与多行 /* ... */ 注释
        s = re.sub(r"//.*?\n", "\n", s)
        s = re.sub(r"//.*?$", "", s)
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)

        # 2. 确保以 { 开头
        start_idx = s.find("{")
        if start_idx != -1:
            s = s[start_idx:]

        # 3. 将单引号字符串转为双引号字符串: 'abc' -> "abc"
        s = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', s)

        # 4. 修复缺少引号的 Key: { action: "BUY_LONG" } -> { "action": "BUY_LONG" }
        s = re.sub(r'([{,\n\r\t ]+)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', s)

        # 5. 移除尾随逗号: [1, 2, ] -> [1, 2], {"a": 1, } -> {"a": 1}
        s = re.sub(r",\s*([\]}])", r"\1", s)

        # 6. 处理尾部被截断的不完整键值
        s = cls._strip_trailing_truncated(s)

        # 7. 字符栈自动闭合括号与引号
        s = cls._stack_repair(s)

        # 8. 再次清理闭合后产生的多余逗号
        s = re.sub(r",\s*([\]}])", r"\1", s)

        return s

    @classmethod
    def _strip_trailing_truncated(cls, s: str) -> str:
        """剥离尾部截断的未完成键或值"""
        s = s.rstrip()
        # 如果以逗号结尾，剥离它
        s = re.sub(r',\s*$', '', s)
        # 如果尾部形如 `"some_key": ` 或 `"some_key`
        s = re.sub(r',\s*"[a-zA-Z0-9_]+"\s*:\s*$', '', s)
        s = re.sub(r',\s*"[a-zA-Z0-9_]*$', '', s)
        return s

    @classmethod
    def _stack_repair(cls, s: str) -> str:
        """基于字符栈自动闭合未匹配的双引号、大括号与中括号"""
        stack: List[str] = []
        in_string = False
        escape = False

        for char in s:
            if char == '"' and not escape:
                in_string = not in_string
            elif not in_string:
                if char in "{[":
                    stack.append(char)
                elif char == "}" and stack and stack[-1] == "{":
                    stack.pop()
                elif char == "]" and stack and stack[-1] == "[":
                    stack.pop()
            escape = (char == "\\" and not escape)

        # 若字符串本身未闭合，先闭合双引号
        if in_string:
            s += '"'

        # 弹出栈中所有未闭合的括号
        while stack:
            top = stack.pop()
            s += "}" if top == "{" else "]"

        return s

    # =========================================================================
    # 3. 字段规范化与补全 (Layer 5)
    # =========================================================================

    @classmethod
    def _normalize_fields(cls, d: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """将字段名映射至标准名称"""
        res = dict(d)

        # 别名映射
        if "analysis_uuid" in res and "analysis_id" not in res:
            res["analysis_id"] = res.pop("analysis_uuid")
        if "analysis_id" not in res:
            res["analysis_id"] = str(uuid.uuid4())

        if "symbol" not in res:
            res["symbol"] = symbol

        if "timestamp" not in res:
            res["timestamp"] = int(datetime.utcnow().timestamp() * 1000)

        if "market_state" in res and "market_regime" not in res:
            res["market_regime"] = res.pop("market_state")

        if "action_plan" in res and "trade_plan" not in res:
            res["trade_plan"] = res.pop("action_plan")

        # 预先处理 trade_plan 中的 percentage
        if "trade_plan" in res and isinstance(res["trade_plan"], dict):
            tp_list = res["trade_plan"].get("take_profit_levels")
            if isinstance(tp_list, list):
                for item in tp_list:
                    if isinstance(item, dict) and "percentage" in item:
                        try:
                            p_val = float(item["percentage"])
                            if p_val > 1.0:
                                item["percentage"] = round(p_val / 100.0, 4)
                        except Exception:
                            item["percentage"] = 0.5

        # 预先处理 funding_rate_bias 模糊枚举
        if "derivatives_sentiment" in res and isinstance(res["derivatives_sentiment"], dict):
            bias_str = str(res["derivatives_sentiment"].get("funding_rate_bias", "")).upper()
            if bias_str not in ["EXTREME_POSITIVE", "MODERATE_POSITIVE", "NEUTRAL", "MODERATE_NEGATIVE", "EXTREME_NEGATIVE"]:
                if "EXTREME" in bias_str and ("POS" in bias_str or "LONG" in bias_str or "OVERHEAT" in bias_str):
                    res["derivatives_sentiment"]["funding_rate_bias"] = "EXTREME_POSITIVE"
                elif "POS" in bias_str or "LONG" in bias_str or "BULL" in bias_str:
                    res["derivatives_sentiment"]["funding_rate_bias"] = "MODERATE_POSITIVE"
                elif "EXTREME" in bias_str and ("NEG" in bias_str or "SHORT" in bias_str):
                    res["derivatives_sentiment"]["funding_rate_bias"] = "EXTREME_NEGATIVE"
                elif "NEG" in bias_str or "SHORT" in bias_str or "BEAR" in bias_str:
                    res["derivatives_sentiment"]["funding_rate_bias"] = "MODERATE_NEGATIVE"
                else:
                    res["derivatives_sentiment"]["funding_rate_bias"] = "NEUTRAL"

        if "rationale" in res and "reasoning_details" not in res:
            res["reasoning_details"] = res.pop("rationale")

        return res

    @classmethod
    def _patch_missing_fields(cls, d: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """智能补全缺失的关键结构"""
        res = cls._normalize_fields(d, symbol)

        # market_regime
        if res.get("market_regime") not in ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE_BREAKOUT"]:
            res["market_regime"] = "RANGING"

        # timeframe_analysis
        tf_analysis = res.get("timeframe_analysis")
        if not isinstance(tf_analysis, dict):
            tf_analysis = {}
        for tf in ["tf_15m", "tf_1h", "tf_4h", "tf_1d"]:
            if tf not in tf_analysis or not isinstance(tf_analysis[tf], dict):
                tf_analysis[tf] = {
                    "trend": "NEUTRAL_CHOPPY",
                    "key_indicators_summary": "自动补全中性指标",
                    "support_level": 0.0,
                    "resistance_level": 0.0,
                }
            else:
                item = tf_analysis[tf]
                if "trend" not in item:
                    item["trend"] = "NEUTRAL_CHOPPY"
                if "key_indicators_summary" not in item:
                    item["key_indicators_summary"] = "正常"
                item["support_level"] = float(item.get("support_level", 0.0))
                item["resistance_level"] = float(item.get("resistance_level", 0.0))
        res["timeframe_analysis"] = tf_analysis

        # derivatives_sentiment
        deriv = res.get("derivatives_sentiment")
        if not isinstance(deriv, dict):
            deriv = {}
        bias_str = str(deriv.get("funding_rate_bias", "")).upper()
        if bias_str not in ["EXTREME_POSITIVE", "MODERATE_POSITIVE", "NEUTRAL", "MODERATE_NEGATIVE", "EXTREME_NEGATIVE"]:
            if "EXTREME" in bias_str and ("POS" in bias_str or "LONG" in bias_str or "OVERHEAT" in bias_str):
                deriv["funding_rate_bias"] = "EXTREME_POSITIVE"
            elif "POS" in bias_str or "LONG" in bias_str or "BULL" in bias_str:
                deriv["funding_rate_bias"] = "MODERATE_POSITIVE"
            elif "EXTREME" in bias_str and ("NEG" in bias_str or "SHORT" in bias_str):
                deriv["funding_rate_bias"] = "EXTREME_NEGATIVE"
            elif "NEG" in bias_str or "SHORT" in bias_str or "BEAR" in bias_str:
                deriv["funding_rate_bias"] = "MODERATE_NEGATIVE"
            else:
                deriv["funding_rate_bias"] = "NEUTRAL"
        if "open_interest_interpretation" not in deriv:
            deriv["open_interest_interpretation"] = "数据正常，中性观察"
        if "long_short_ratio_state" not in deriv:
            deriv["long_short_ratio_state"] = "持仓比例均衡"
        deriv["sentiment_score"] = float(deriv.get("sentiment_score", 0.0))
        res["derivatives_sentiment"] = deriv

        # signal
        sig = res.get("signal")
        if not isinstance(sig, dict):
            sig = {}
        if sig.get("action") not in ["BUY_LONG", "SELL_SHORT", "CLOSE_POSITION", "HOLD_WAIT"]:
            sig["action"] = "HOLD_WAIT"
        sig["confidence"] = float(sig.get("confidence", 0.0))
        if sig.get("urgency") not in ["LOW", "MEDIUM", "HIGH"]:
            sig["urgency"] = "LOW"
        res["signal"] = sig

        # trade_plan
        tp = res.get("trade_plan")
        if not isinstance(tp, dict):
            tp = {}
        er = tp.get("entry_range")
        if not isinstance(er, list) or len(er) != 2:
            tp["entry_range"] = [0.0, 0.0]
        else:
            tp["entry_range"] = [float(er[0]), float(er[1])]

        tp_levels = tp.get("take_profit_levels")
        if not isinstance(tp_levels, list) or not tp_levels:
            tp["take_profit_levels"] = [{"price": 0.0, "percentage": 1.0, "description": "无具体计划"}]
        else:
            cleaned_tpl = []
            for item in tp_levels:
                if isinstance(item, dict):
                    pct = float(item.get("percentage", 1.0))
                    if pct > 1.0:
                        pct = pct / 100.0
                    pct = max(0.01, min(1.0, pct))
                    cleaned_tpl.append({
                        "price": float(item.get("price", 0.0)),
                        "percentage": round(pct, 4),
                        "description": str(item.get("description", "止盈目标")),
                    })
            tp["take_profit_levels"] = cleaned_tpl or [{"price": 0.0, "percentage": 1.0, "description": "无具体计划"}]

        tp["stop_loss_price"] = float(tp.get("stop_loss_price", 0.0))
        tp["risk_reward_ratio"] = float(tp.get("risk_reward_ratio", 0.0))
        tp["suggested_leverage"] = max(1, min(int(tp.get("suggested_leverage", 1)), 20))
        if tp.get("order_type") not in ["LIMIT", "MARKET", "TRIGGER_LIMIT"]:
            tp["order_type"] = "LIMIT"
        res["trade_plan"] = tp

        # risk_assessment
        ra = res.get("risk_assessment")
        if not isinstance(ra, dict):
            ra = {}
        kr = ra.get("key_risks")
        if not isinstance(kr, list) or not kr:
            ra["key_risks"] = ["市场正常波动风险"]
        else:
            ra["key_risks"] = [str(x) for x in kr]
        if "invalidation_condition" not in ra:
            ra["invalidation_condition"] = "触及止损或结构改变"
        ra["max_holding_time_hours"] = float(ra.get("max_holding_time_hours", 24.0))
        res["risk_assessment"] = ra

        # reasoning
        if "reasoning_summary" not in res:
            res["reasoning_summary"] = "系统完成综合研判，维持纪律执行。"
        if "reasoning_details" not in res:
            res["reasoning_details"] = "多周期指标与衍生品综合推导完成。"

        return res

    # =========================================================================
    # 4. 安全熔断兜底对象生成
    # =========================================================================

    @classmethod
    def create_safe_fallback_response(cls, symbol: str, error_msg: str) -> AIAnalysisResponse:
        """当发生不可恢复解析错误时的绝对安全兜底契约 (HOLD_WAIT, confidence=0)"""
        now_ms = int(datetime.utcnow().timestamp() * 1000)
        return AIAnalysisResponse(
            analysis_id=str(uuid.uuid4()),
            symbol=symbol,
            timestamp=now_ms,
            market_regime=MarketRegime.RANGING,
            timeframe_analysis=TimeframeAnalysis(
                tf_15m=TimeframeDetail(trend=TimeframeTrend.NEUTRAL_CHOPPY, key_indicators_summary="系统降级保护", support_level=0.0, resistance_level=0.0),
                tf_1h=TimeframeDetail(trend=TimeframeTrend.NEUTRAL_CHOPPY, key_indicators_summary="系统降级保护", support_level=0.0, resistance_level=0.0),
                tf_4h=TimeframeDetail(trend=TimeframeTrend.NEUTRAL_CHOPPY, key_indicators_summary="系统降级保护", support_level=0.0, resistance_level=0.0),
                tf_1d=TimeframeDetail(trend=TimeframeTrend.NEUTRAL_CHOPPY, key_indicators_summary="系统降级保护", support_level=0.0, resistance_level=0.0),
            ),
            derivatives_sentiment=DerivativesSentiment(
                funding_rate_bias=FundingRateBias.NEUTRAL,
                open_interest_interpretation="数据流降级，维持中性",
                long_short_ratio_state="中性",
                sentiment_score=0.0,
            ),
            signal=AISignal(
                action=SignalAction.HOLD_WAIT,
                confidence=0.0,
                urgency=SignalUrgency.LOW,
            ),
            trade_plan=TradePlan(
                entry_range=[0.0, 0.0],
                take_profit_levels=[TakeProfitLevel(price=0.0, percentage=1.0, description="无计划")],
                stop_loss_price=0.0,
                risk_reward_ratio=0.0,
                suggested_leverage=1,
                order_type=TradePlanOrderType.LIMIT,
            ),
            risk_assessment=RiskAssessment(
                key_risks=[f"AI 响应解析触发降级保护: {error_msg}"],
                invalidation_condition="系统自动降级保护生效",
                max_holding_time_hours=1.0,
            ),
            reasoning_summary="AI 研判输出解析失败，系统启动安全熔断保护，强制输出 HOLD_WAIT 观望指令。",
            reasoning_details=f"底层异常信息: {error_msg}。为确保账户资金安全，硬风控中枢已锁定开仓动作。",
        )
