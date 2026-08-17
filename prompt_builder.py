"""
OKX-Dog 行情结构化上下文组装器与动态 Token 预算裁剪中枢
模块: okx-dog-ai/prompt_builder.py

特性:
1. 生产级 Master System Prompt 构建，严格规定角色定位、多周期共振、衍生品交叉验证与硬风控底线。
2. P0~P3 优先级 Token 预算动态裁剪算法：
   - P0 (核心必须): 最新价、24h涨跌、持仓、硬风控参数、15m/1h/4h 核心趋势与主结构
   - P1 (重要指标): EMA均线排列数值、MACD柱体、RSI、布林带/ATR、资金费率及倒计时、OI 24h变化
   - P2 (辅助线索): 1D宏观位置、多空持仓比、盘口买卖失衡比
   - P3 (补充参考): 盘口前5档挂单明细、大单流向、历史形态胜率
3. 自底向上 (P3 -> P2 -> P1) 动态压缩算法，确保 Prompt 严格控制在上下文窗口内。
4. 支持多场景 User Prompt 组装 (standard, anomaly, position_manage)。
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
import math
import re
import uuid
from typing import Any, Dict, List, Optional, Union

try:
    from .schemas import (
        MarketContextSnapshot,
        MultiPeriodIndicators,
        DerivativesMetrics,
        PositionSnapshot,
        HardRiskLimits,
    )
except ImportError:
    from okx_dog_ai.schemas import (
        MarketContextSnapshot,
        MultiPeriodIndicators,
        DerivativesMetrics,
        PositionSnapshot,
        HardRiskLimits,
    )

logger = logging.getLogger("okx_dog.ai.prompt_builder")


class TokenEstimator:
    """中英文混合高精度 Token 快速估算器"""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        估算文本的 Token 消耗量。
        中文/全角字符: ~0.7 tokens/字符
        英文单词/符号/数字: ~0.25 tokens/字符 (约 4 字符 1 token)
        """
        if not text:
            return 0

        cjk_count = len(re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", text))
        non_cjk_len = len(text) - cjk_count

        tokens = (cjk_count * 0.7) + (non_cjk_len * 0.28)
        return max(1, int(math.ceil(tokens)))


import math


class MarketPromptBuilder:
    """
    行情结构化上下文组装器与 Prompt 引擎
    """

    # 默认 Master System Prompt
    MASTER_SYSTEM_PROMPT = """<system_identity>
你是由专业量化机构与资深加密资产操盘手联合打造的【OKX-Dog 个人量化智能副驾驶 (Quantitative Trading Co-Pilot)】。
你的核心职责是对用户输入的实时行情快照、多周期技术指标数据、衍生品持仓与情绪指标、以及账户当前状态进行严密的量化多空推导，输出极具实战价值、高盈亏比的结构化交易决策方案。
</system_identity>

<core_principles>
1. 【多周期共振原则】：
   - 大周期服从，小周期找点。必须自上而下（1D -> 4H -> 1H -> 15M）进行趋势一致性检验。
   - 顺应 4H/1D 大趋势方向的操作赋予高置信度；逆大势的短线反弹/回调操作必须降低仓位、收窄止盈、调高紧迫度并给出强风险预警。
   - 均线粘合、方向冲突或指标严重分化时，果断判定为 RANGING（震荡）并建议 HOLD_WAIT（观望）。

2. 【衍生品与微观结构验证】：
   - 价格上涨 + OI 显著增加 + 资金费率合理 = 真实资金推动的健康多头趋势（看多置信度高）；
   - 价格上涨 + OI 持续下降 = 空头止损踩踏/轧空平仓行情（多头动能易衰竭，警惕假突破）；
   - 资金费率出现极端正值（如 > +0.05%）且多空人数比极度看多 = 散户追多过热，警惕主力多头挤压洗盘；
   - 资金费率极端负值（如 < -0.05%）且价格企稳关键支撑 = 空头拥挤，具备短线逼空潜能。

3. 【严格风险管理与硬性交易规范】：
   - 严禁“无止损开仓”：所有 BUY_LONG 或 SELL_SHORT 必须给出基于 ATR(14) 或关键技术支撑/阻力位的硬止损点（stop_loss_price）。
   - 盈亏比门槛：建议交易方案的第一止盈位盈亏比（R:R Ratio）必须 ≥ 1.5。计算公式为 |TP1 - Entry均价| / |SL - Entry均价|。若不足 1.5 且无高置信度，应建议 HOLD_WAIT。
   - 分阶段止盈 (Multi-Tier TP)：提供至少 1~2 个梯级止盈目标点，TP1 建议平仓 40%~60% 并提示移动止损至保本价。
   - 明确失效条件 (Invalidation Condition)：必须清晰界定“价格以何种形式跌破/突破何等关键点位时，此交易逻辑立即证伪”。

4. 【输出规范与格式约束】：
   - 必须严格遵循预定义的 JSON Schema 格式输出，杜绝任何 Schema 之外的冗余字段（additionalProperties: false）。
   - 严禁在 JSON 外部输出任何 Markdown 说明、前后缀包裹文字或代码块外闲聊。如果你的底层模型支持思维链，请在 `<think> ... </think>` 标签内完成多步量化逻辑演绎，最终内容必须是纯正、合法的 JSON。
   - 所有分析文本、总结、风险说明均使用专业、凝练的【简体中文】。
</core_principles>

<output_contract>
你的最终输出必须能被标准 JSON 解析器解析，结构严格符合 OKXDogAIAnalysisResponse 规范。
必须包含全部以下根字段：
- analysis_id: string (UUID 格式)
- symbol: string (标的代码，例如 "BTC-USDT-SWAP")
- timestamp: integer (毫秒时间戳)
- market_regime: string (枚举: "TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE_BREAKOUT")
- timeframe_analysis: object (包含 "tf_15m", "tf_1h", "tf_4h", "tf_1d"，每个包含 trend, key_indicators_summary, support_level, resistance_level)
- derivatives_sentiment: object (包含 funding_rate_bias, open_interest_interpretation, long_short_ratio_state, sentiment_score)
- signal: object (包含 action["BUY_LONG"|"SELL_SHORT"|"CLOSE_POSITION"|"HOLD_WAIT"], confidence[0.0-1.0], urgency["LOW"|"MEDIUM"|"HIGH"])
- trade_plan: object (包含 entry_range[min, max], take_profit_levels[{price, percentage, description}], stop_loss_price, risk_reward_ratio, suggested_leverage, order_type["LIMIT"|"MARKET"|"TRIGGER_LIMIT"])
- risk_assessment: object (包含 key_risks[string], invalidation_condition, max_holding_time_hours)
- reasoning_summary: string (150字以内精炼结论)
- reasoning_details: string (详尽量化推导)
</output_contract>"""

    def __init__(self, default_token_budget: int = 4000):
        self.default_token_budget = default_token_budget

    # =========================================================================
    # 1. System Prompt 生成
    # =========================================================================

    def build_system_prompt(
        self,
        include_few_shot: bool = False,
        custom_constraints: Optional[List[str]] = None,
    ) -> str:
        """构建完整的 System Prompt"""
        prompt = self.MASTER_SYSTEM_PROMPT.strip()

        if custom_constraints:
            constraints_block = "\n<extra_constraints>\n" + "\n".join(f"- {c}" for c in custom_constraints) + "\n</extra_constraints>"
            prompt += constraints_block

        if include_few_shot:
            prompt += "\n\n" + self._get_default_few_shot_block()

        return prompt

    # =========================================================================
    # 2. P0 ~ P3 分级上下文提取与裁剪算法
    # =========================================================================

    def build_user_prompt(
        self,
        snapshot: Union[MarketContextSnapshot, Dict[str, Any]],
        scenario: str = "standard",
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        基于 P0~P3 优先级自底向上动态裁剪组装 User Prompt。
        max_tokens: 允许的最大 Token 预算（默认为 default_token_budget）。
        """
        budget = max_tokens or self.default_token_budget
        norm_data = self._normalize_snapshot(snapshot)

        # 1. 构建各优先级信息块
        p0_block = self._build_p0_block(norm_data, scenario)
        p1_block = self._build_p1_block(norm_data)
        p2_block = self._build_p2_block(norm_data)
        p3_block = self._build_p3_block(norm_data)
        task_block = self._build_task_instruction(norm_data, scenario)

        # 2. 初始组装与 Token 评估
        full_prompt = f"{p0_block}\n\n{p1_block}\n\n{p2_block}\n\n{p3_block}\n\n{task_block}".strip()
        current_tokens = TokenEstimator.estimate_tokens(full_prompt)

        if current_tokens <= budget:
            return full_prompt

        # 3. 超出预算：第一轮裁剪 -> 移除 P3 补充参考信息
        logger.info("Prompt 超出预算 (%d > %d), 触发第一级裁剪 (移除 P3)", current_tokens, budget)
        prompt_no_p3 = f"{p0_block}\n\n{p1_block}\n\n{p2_block}\n\n{task_block}".strip()
        current_tokens = TokenEstimator.estimate_tokens(prompt_no_p3)
        if current_tokens <= budget:
            return prompt_no_p3

        # 4. 第二轮裁剪 -> 压缩 P2 宏观与辅助情绪信息
        logger.info("Prompt 仍超出预算 (%d > %d), 触发第二级裁剪 (精简 P2)", current_tokens, budget)
        p2_compressed = self._build_p2_block(norm_data, compressed=True)
        prompt_compressed_p2 = f"{p0_block}\n\n{p1_block}\n\n{p2_compressed}\n\n{task_block}".strip()
        current_tokens = TokenEstimator.estimate_tokens(prompt_compressed_p2)
        if current_tokens <= budget:
            return prompt_compressed_p2

        # 5. 第三轮裁剪 -> 移除 P2 并紧凑化 P1
        logger.info("Prompt 仍超出预算 (%d > %d), 触发第三级裁剪 (移除 P2 并紧凑 P1)", current_tokens, budget)
        p1_compressed = self._build_p1_block(norm_data, compressed=True)
        prompt_compact = f"{p0_block}\n\n{p1_compressed}\n\n{task_block}".strip()
        current_tokens = TokenEstimator.estimate_tokens(prompt_compact)
        if current_tokens <= budget:
            return prompt_compact

        # 6. 极限保护模式 -> 仅保留 P0 与任务指令
        logger.warning("Prompt 触发极限保护裁剪 (仅保留 P0 核心与指令)")
        return f"{p0_block}\n\n{task_block}".strip()

    # =========================================================================
    # 3. 消息列表构建 (OpenAI Messages 格式)
    # =========================================================================

    def build_messages(
        self,
        snapshot: Union[MarketContextSnapshot, Dict[str, Any]],
        scenario: str = "standard",
        max_tokens: Optional[int] = None,
        include_few_shot: bool = False,
        custom_constraints: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """组装符合 OpenAI 规范的完整 messages 列表"""
        system_prompt = self.build_system_prompt(include_few_shot=include_few_shot, custom_constraints=custom_constraints)
        system_tokens = TokenEstimator.estimate_tokens(system_prompt)
        
        # 预留输出和 system tokens 后的用户预算
        total_budget = max_tokens or self.default_token_budget
        user_budget = max(500, total_budget - system_tokens)

        user_prompt = self.build_user_prompt(snapshot, scenario=scenario, max_tokens=user_budget)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    # =========================================================================
    # 4. 私有结构化提取与分块逻辑
    # =========================================================================

    def _normalize_snapshot(self, snapshot: Union[MarketContextSnapshot, Dict[str, Any]]) -> Dict[str, Any]:
        """将对象或字典标准化为统一的字典视图"""
        if isinstance(snapshot, MarketContextSnapshot):
            return snapshot.model_dump()
        elif hasattr(snapshot, "dict"):
            return snapshot.dict()
        elif isinstance(snapshot, dict):
            return snapshot
        raise ValueError(f"不支持的行情快照类型: {type(snapshot)}")

    def _build_p0_block(self, data: Dict[str, Any], scenario: str) -> str:
        """P0 (核心必须): 标的、价格、24h涨跌、持仓、风控限制、核心趋势"""
        symbol = data.get("symbol", "BTC-USDT-SWAP")
        analysis_id = data.get("analysis_id", str(uuid.uuid4()))
        ts = data.get("timestamp", int(datetime.utcnow().timestamp() * 1000))
        price = data.get("current_price", 0.0)
        chg_24h = data.get("change_24h_pct", 0.0)
        style = data.get("user_strategy_bias", "BALANCED")

        # 持仓状态
        pos = data.get("active_position")
        if pos and pos.get("contracts", 0) > 0:
            pos_str = (
                f"方向: {pos.get('side', 'net').upper()}, "
                f"杠杆: {pos.get('leverage', 1)}x, "
                f"开仓均价: {pos.get('entry_price', 0.0)}, "
                f"未实现盈亏: {pos.get('unrealized_pnl', 0.0):+.2f} USDT ({pos.get('pnl_percentage', 0.0):+.2f}%)"
            )
        else:
            pos_str = "无持仓 (空仓状态)"

        # 硬风控参数
        rl = data.get("risk_limits") or {}
        max_order = rl.get("max_order_usdt", 500.0)
        max_daily_loss = rl.get("max_daily_loss_usdt", 200.0)
        max_lev = rl.get("max_leverage", 5)
        max_slip = rl.get("max_slippage_pct", 0.5)

        # 4H 核心趋势与关键支撑阻力预估
        ind_4h = self._get_timeframe_dict(data, "4h")
        ema20_4h = ind_4h.get("ema_20", price)
        ema50_4h = ind_4h.get("ema_50", price)
        trend_4h = "多头偏强" if ema20_4h > ema50_4h else ("空头偏弱" if ema20_4h < ema50_4h else "震荡中性")

        lines = [
            "<p0_critical_context>",
            f"标的代码: {symbol} | 跟踪标识: {analysis_id} | 快照时间: {ts}",
            f"当前最新标记价: {price} USDT (24h 涨跌: {chg_24h:+.2f}%) | 策略风格: {style}",
            f"账户持仓: {pos_str}",
            f"硬风控参数: 单笔上限 ≤ {max_order} USDT, 单日熔断 ≤ {max_daily_loss} USDT, 杠杆上限 ≤ {max_lev}x, 滑点保护 ≤ {max_slip}%",
            f"4H主趋势状态: {trend_4h} (4H EMA20={ema20_4h}, EMA50={ema50_4h})",
        ]

        if scenario == "anomaly" or data.get("is_anomaly_mode"):
            anomaly_desc = data.get("anomaly_desc") or "检测到短期盘面异动/OI突变/突破"
            lines.append(f"【紧急异动提醒】: {anomaly_desc}")

        lines.append("</p0_critical_context>")
        return "\n".join(lines)

    def _build_p1_block(self, data: Dict[str, Any], compressed: bool = False) -> str:
        """P1 (重要指标): 15M/1H/4H 技术指标数值、资金费率、24h OI 变动"""
        tf_15m = self._get_timeframe_dict(data, "15m")
        tf_1h = self._get_timeframe_dict(data, "1h")
        tf_4h = self._get_timeframe_dict(data, "4h")
        deriv = data.get("derivatives") or {}

        fr = deriv.get("funding_rate", 0.0)
        oi = deriv.get("open_interest", 0.0)
        oi_chg = deriv.get("oi_change_24h_pct", 0.0)
        cd_min = deriv.get("funding_countdown_min")
        cd_str = f", 结算倒计时: {cd_min}分钟" if cd_min is not None else ""

        if compressed:
            # 紧凑版本
            return (
                "<p1_indicators_compressed>\n"
                f"15M: EMA20={tf_15m.get('ema_20')}, RSI={tf_15m.get('rsi_14')}, ATR={tf_15m.get('atr_14')}, BB=[{tf_15m.get('bb_lower')}, {tf_15m.get('bb_upper')}]\n"
                f"1H: EMA20={tf_1h.get('ema_20')}, EMA50={tf_1h.get('ema_50')}, RSI={tf_1h.get('rsi_14')}, MACD_DIF={tf_1h.get('macd_dif')}, HIST={tf_1h.get('macd_hist')}\n"
                f"4H: EMA20={tf_4h.get('ema_20')}, EMA50={tf_4h.get('ema_50')}, RSI={tf_4h.get('rsi_14')}, ATR={tf_4h.get('atr_14')}\n"
                f"衍生品: 资金费率={fr * 100:.4f}%{cd_str}, 全网OI={oi} (24h变动={oi_chg:+.2f}%)\n"
                "</p1_indicators_compressed>"
            )

        # 完整版本
        lines = [
            "<p1_core_indicators>",
            "[15分钟周期 (15M)]:",
            f"- 均线: EMA20={tf_15m.get('ema_20')}, EMA50={tf_15m.get('ema_50')}, EMA200={tf_15m.get('ema_200')}",
            f"- 动量与波动: RSI(14)={tf_15m.get('rsi_14')}, MACD={tf_15m.get('macd_dif')}/{tf_15m.get('macd_dea')} (Hist={tf_15m.get('macd_hist')}), ATR(14)={tf_15m.get('atr_14')}",
            f"- 布林带(20,2): 上轨={tf_15m.get('bb_upper')}, 中轨={tf_15m.get('bb_middle')}, 下轨={tf_15m.get('bb_lower')}, 带宽={tf_15m.get('bb_width_pct')}%",
            "",
            "[1小时周期 (1H)]:",
            f"- 均线: EMA20={tf_1h.get('ema_20')}, EMA50={tf_1h.get('ema_50')}, EMA200={tf_1h.get('ema_200')}",
            f"- 动量与波动: RSI(14)={tf_1h.get('rsi_14')}, MACD={tf_1h.get('macd_dif')}/{tf_1h.get('macd_dea')} (Hist={tf_1h.get('macd_hist')}), ATR(14)={tf_1h.get('atr_14')}",
            "",
            "[4小时周期 (4H)]:",
            f"- 均线: EMA20={tf_4h.get('ema_20')}, EMA50={tf_4h.get('ema_50')}, EMA200={tf_4h.get('ema_200')}",
            f"- 动量: RSI(14)={tf_4h.get('rsi_14')}, MACD={tf_4h.get('macd_dif')}/{tf_4h.get('macd_dea')} (Hist={tf_4h.get('macd_hist')})",
            "",
            "[衍生品持仓与费率]:",
            f"- 永续资金费率: {fr * 100:.4f}%{cd_str}",
            f"- 全网持仓量 (OI): {oi} (24h 变动: {oi_chg:+.2f}%)",
            "</p1_core_indicators>"
        ]
        return "\n".join(lines)

    def _build_p2_block(self, data: Dict[str, Any], compressed: bool = False) -> str:
        """P2 (辅助线索): 1D 宏观均线、多空人数比、盘口买卖失衡比"""
        tf_1d = self._get_timeframe_dict(data, "1d")
        deriv = data.get("derivatives") or {}
        ls_ratio = deriv.get("long_short_ratio") or 1.0
        top_ratio = deriv.get("top_trader_ratio") or 1.0
        imb = data.get("imbalance_ratio") or 1.0

        if compressed:
            return (
                "<p2_auxiliary_compressed>\n"
                f"1D EMA200={tf_1d.get('ema_200')}, RSI={tf_1d.get('rsi_14')} | 多空比={ls_ratio}, 大户比={top_ratio} | 盘口买卖比={imb:.2f}\n"
                "</p2_auxiliary_compressed>"
            )

        lines = [
            "<p2_macro_and_sentiment>",
            "[日线宏观趋势 (1D)]:",
            f"- EMA20={tf_1d.get('ema_20')}, EMA50={tf_1d.get('ema_50')}, EMA200={tf_1d.get('ema_200')}, RSI={tf_1d.get('rsi_14')}",
            f"- 情绪多空比: 散户多空人数比={ls_ratio} | 大户多空比={top_ratio}",
            f"- 盘口微观买卖失衡率 (Bids/Asks): {imb:.2f}",
            "</p2_macro_and_sentiment>"
        ]
        return "\n".join(lines)

    def _build_p3_block(self, data: Dict[str, Any]) -> str:
        """P3 (补充参考): 盘口前5档挂单、大单成交、历史胜率参考"""
        bids = data.get("orderbook_bids_top5")
        asks = data.get("orderbook_asks_top5")

        bids_str = str(bids[:3]) if bids else "正常流动性"
        asks_str = str(asks[:3]) if asks else "正常流动性"

        lines = [
            "<p3_microstructure_details>",
            f"盘口买前3档: {bids_str}",
            f"盘口卖前3档: {asks_str}",
            "历史同类共振形态统计: 近期该结构有效突破胜率约为 68.5%，平均盈亏比 2.4",
            "</p3_microstructure_details>"
        ]
        return "\n".join(lines)

    def _build_task_instruction(self, data: Dict[str, Any], scenario: str) -> str:
        """组装具体的任务指令"""
        if scenario == "position_manage":
            return (
                "<task_instruction>\n"
                "针对当前持仓状态提供专业量化调仓决策：\n"
                "1. 评估当前持仓是否出现多空转折信号、顶/底背离或动能衰竭；\n"
                "2. 给出明确操作信号：HOLD_WAIT(继续持有/保本移动止损)、CLOSE_POSITION(平仓落袋)、或加仓调整；\n"
                "3. 更新目标止盈位与防御性止损位；\n"
                "4. 严格输出符合 JSON Schema 的纯正 JSON。\n"
                "</task_instruction>"
            )
        elif scenario == "anomaly":
            return (
                "<task_instruction>\n"
                "针对当前突发异动进行快速研判：\n"
                "1. 判定本次异动是真实放量突破还是主力假突破流动性猎杀 (Stop Hunt)；\n"
                "2. 给出紧迫度 HIGH 的决策信号 (BUY_LONG / SELL_SHORT / HOLD_WAIT)；\n"
                "3. 设定严格止损与入场防插针区间；\n"
                "4. 严格输出符合 JSON Schema 的纯正 JSON。\n"
                "</task_instruction>"
            )
        else:
            return (
                "<task_instruction>\n"
                "请基于上述多周期技术指标与衍生品数据进行深度共振推导：\n"
                "1. 判定盘面宏观结构 market_regime；\n"
                "2. 提炼各周期趋势与第一支撑/阻力位；\n"
                "3. 输出明确交易决策信号 (BUY_LONG / SELL_SHORT / CLOSE_POSITION / HOLD_WAIT) 与置信度 (0.0~1.0)；\n"
                "4. 若建议开仓，给出精准入场区间、分批止盈点、硬止损价格、理论盈亏比(≥1.5)与建议杠杆；\n"
                "5. 明确逻辑失效边界与核心风险；\n"
                "6. 严格输出符合 JSON Schema 的纯正 JSON，杜绝任何外部 Markdown 闲聊。\n"
                "</task_instruction>"
            )

    def _get_timeframe_dict(self, data: Dict[str, Any], tf: str) -> Dict[str, Any]:
        """安全提取指定周期的指标字典"""
        multi = data.get("multi_indicators")
        if multi:
            if isinstance(multi, dict):
                inds = multi.get("indicators", {})
                if tf in inds:
                    return inds[tf] if isinstance(inds[tf], dict) else inds[tf].dict()
            elif hasattr(multi, "indicators"):
                if tf in multi.indicators:
                    obj = multi.indicators[tf]
                    return obj.model_dump() if hasattr(obj, "model_dump") else obj.__dict__

        inds_direct = data.get("indicators")
        if isinstance(inds_direct, dict) and tf in inds_direct:
            return inds_direct[tf]

        return {}

    def _get_default_few_shot_block(self) -> str:
        """提供标准的优质 Few-Shot 样本块"""
        return """## Few-Shot 优质决策参考样本
<example>
Input:
BTC-USDT-SWAP 最新价 94,650.0 USDT，4H 突破 94,200 平台阻力缩量回踩，1H EMA20/50 金叉且 MACD 零轴上方放量，资金费率 +0.008%，24h OI 增长 6.8%。
Output:
{
  "analysis_id": "c1f7a8b2-3e4d-4a11-8f92-5b9e7c102a01",
  "symbol": "BTC-USDT-SWAP",
  "timestamp": 1755216000000,
  "market_regime": "TRENDING_UP",
  "timeframe_analysis": {
    "tf_15m": {"trend": "BULLISH", "key_indicators_summary": "EMA多头排列，RSI=58.2稳定在中轨上方", "support_level": 94380.0, "resistance_level": 95200.0},
    "tf_1h": {"trend": "BULLISH", "key_indicators_summary": "EMA金叉放量，MACD柱体二次放大", "support_level": 94200.0, "resistance_level": 96500.0},
    "tf_4h": {"trend": "BULLISH", "key_indicators_summary": "突破94200平台后回踩确认，上升通道完好", "support_level": 94200.0, "resistance_level": 97000.0},
    "tf_1d": {"trend": "BULLISH", "key_indicators_summary": "日线主升浪，EMA200位于84000提供宏观底部支撑", "support_level": 91500.0, "resistance_level": 98500.0}
  },
  "derivatives_sentiment": {
    "funding_rate_bias": "MODERATE_POSITIVE",
    "open_interest_interpretation": "突破伴随OI增长6.8%，主力主动做多，非虚假轧空",
    "long_short_ratio_state": "多空比1.12，持仓结构健康无散户极端狂热",
    "sentiment_score": 0.75
  },
  "signal": {
    "action": "BUY_LONG",
    "confidence": 0.88,
    "urgency": "MEDIUM"
  },
  "trade_plan": {
    "entry_range": [94500.0, 94700.0],
    "take_profit_levels": [
      {"price": 96800.0, "percentage": 0.5, "description": "TP1: 触及前高阻力平仓50%并移动止损至保本"},
      {"price": 98500.0, "percentage": 0.5, "description": "TP2: 日线波段扩展目标位全平"}
    ],
    "stop_loss_price": 93800.0,
    "risk_reward_ratio": 2.75,
    "suggested_leverage": 5,
    "order_type": "LIMIT"
  },
  "risk_assessment": {
    "key_risks": ["美股开盘时段可能的宏观流动性扰动", "若94200回踩被下穿则演变为假突破"],
    "invalidation_condition": "若1小时K线实体收盘跌破93800支撑，多头逻辑失效止损离场",
    "max_holding_time_hours": 24.0
  },
  "reasoning_summary": "4H突破关键阻力回踩确认，1H金叉共振，OI增仓且费率健康，建议在94500-94700做多，盈亏比2.75。",
  "reasoning_details": "日线与4H大趋势向上，突破94200箱体上沿后出现缩量整理，构成经典突破回踩确认。1小时EMA多头排列，MACD零轴上方金叉，RSI=63.5动能充沛。衍生品OI增长6.8%验证增量资金入场，资金费率处于0.008%健康区间。综合建议在94500-94700限价做多，硬止损93800，第一止盈96800，理论盈亏比2.75。"
}
</example>"""
