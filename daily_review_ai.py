"""
OKX-Dog AI 决策大脑 - 每日盘后深度复盘与经验演化 Prompt 引擎
模块: okx-dog-ai/daily_review_ai.py
角色: AI 与量化算法工程师 (agency-ai-engineer) / 提示词工程师 (agency-prompt-engineer)
功能:
1. 组装每日战报、逐笔成交流水与 MFE/MAE 统计
2. 引导 LLM 生成资深交易员视角的深度 Markdown 复盘战报
3. 严谨提炼 3 条今日最佳操作与 3 条核心避坑教训
4. 格式化输出策略权重参数优化建议
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("okx_dog.ai.daily_review")

DAILY_REVIEW_SYSTEM_PROMPT = """你是一名资深顶级加密量化基金首席操盘手与交易教练。
你的职责是对系统今日的所有实盘/模拟盘交易表现进行残酷、严谨、客观的盘后深度复盘。
从胜率、盈亏比、手续费磨损、MFE（最大有利偏移/是否过早卖飞）、MAE（最大不利偏移/是否扛单过深）等多维度进行解剖，
提炼出最具实战价值的“避坑教训与行动准则”，以帮助交易大脑在明日规避同类失误。

输出要求:
必须输出符合规范的 JSON 结构，包含以下字段:
1. summary_markdown: 完整的专业 Markdown 格式复盘报告（包含【今日战报概述】、【逐笔核心交易解剖】、【MFE/MAE与保本效率评估】、【得失归因总结】、【明日实战指引】）。
2. lessons_learned: 字符串数组，提炼出 2~4 条具体的避坑教训（格式要求: "在[具体行情特征]时，坚决[避坑行动准则]"）。
3. strategy_adjustments: 字典对象，给出针对明日行情的参数自适应微调建议（如 {"recommended_leverage": 3, "trailing_stop_ratio": 0.5, "focus_timeframe": "1h"}）。
"""

class DailyReviewAIEngine:
    """每日复盘 AI 生成引擎"""

    @staticmethod
    def build_review_prompt(
        review_date: str,
        stats: Dict[str, Any],
        trades: List[Dict[str, Any]]
    ) -> str:
        """组装复盘输入上下文"""
        prompt = f"【复盘日期】: {review_date}

"
        prompt += "【今日战报核心数据】:
"
        prompt += f"- 总交易笔数: {stats.get('total_trades', 0)}
"
        prompt += f"- 胜单数: {stats.get('winning_trades', 0)} / 负单数: {stats.get('losing_trades', 0)}
"
        prompt += f"- 当日胜率: {stats.get('win_rate', 0.0) * 100:.1f}%
"
        prompt += f"- 累计已实现盈亏: {stats.get('realized_pnl', 0.0):+.2f} USDT
"
        prompt += f"- 累计手续费磨损: {stats.get('fee_paid', 0.0):.2f} USDT
"
        prompt += f"- 当日最大回撤: {stats.get('max_drawdown', 0.0):.2f}%
"
        prompt += f"- 最佳单笔盈利: {stats.get('best_trade_pnl', 0.0):+.2f} USDT
"
        prompt += f"- 最差单笔亏损: {stats.get('worst_trade_pnl', 0.0):+.2f} USDT

"

        prompt += "【今日平仓交易逐笔流水 (前 15 笔)】:
"
        if not trades:
            prompt += "今日无平仓记录（维持空仓观望或持仓未变动）。
"
        else:
            for idx, t in enumerate(trades[:15], 1):
                prompt += (
                    f"{idx}. [{t.get('symbol')}] 方向:{t.get('pos_side')} "
                    f"入场:{t.get('entry_price')} -> 出场:{t.get('exit_price')} "
                    f"盈亏:{t.get('realized_pnl', 0.0):+.2f} USDT (ROI: {t.get('roi_pct', 0.0):+.2f}%) "
                    f"原因:{t.get('reason', '自动平仓')}
"
                )

        prompt += "
请根据以上数据进行全方位深度复盘，并给出提炼的避坑教训与 Markdown 报告。"
        return prompt

    @staticmethod
    def parse_review_response(raw_output: str, default_stats: Dict[str, Any]) -> Dict[str, Any]:
        """解析 LLM 生成的复盘结果"""
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
            return {
                "summary_markdown": data.get("summary_markdown", "# 每日复盘报告
无内容"),
                "lessons_learned": data.get("lessons_learned", []),
                "strategy_adjustments": data.get("strategy_adjustments", {})
            }
        except Exception as e:
            logger.warning(f"解析复盘 LLM 响应失败，生成格式化备用报告: {e}")
            pnl = default_stats.get("realized_pnl", 0.0)
            status_text = "🎉 今日实现盈利，策略运行稳健" if pnl >= 0 else "⚠️ 今日产生回撤，需严格执行防上头与保本纪律"
            md = f"""# 每日量化交易深度复盘报告 ({default_stats.get('review_date')})

## 1. 战报概述与收益归因
- **当日综合盈亏**: `{pnl:+.2f} USDT` ({status_text})
- **交易胜率**: `{default_stats.get('win_rate', 0.0) * 100:.1f}%` (总计 {default_stats.get('total_trades', 0)} 笔)
- **摩擦成本**: 手续费磨损 `{default_stats.get('fee_paid', 0.0):.2f} USDT`

## 2. 核心得失与避坑纪律
1. **严格遵守移动保本**: 浮盈达到 0.5R 时强制移动止损线至开仓成本价，防止利润回吐变亏损。
2. **严禁在费率极端过热时追高**: 资金费率超过 0.05% 时优先观望或逢高轻仓做空。
3. **连亏防上头机制**: 触发连续 2 笔止损后坚决强制休息 60 分钟，严防情绪化冲动交易。

## 3. 明日实战指引
保持主周期 (1h/4h) 顺势交易，坚守盈亏比 $\ge 1:1.8$ 准入红线。
"""
            return {
                "summary_markdown": md,
                "lessons_learned": [
                    "在资金费率 > 0.05% 且 4H RSI 超买时坚决不追多",
                    "在浮盈达标 0.5R 时强制锁定移动保本"
                ],
                "strategy_adjustments": {"max_leverage": 3, "breakeven_ratio": 0.5}
            }