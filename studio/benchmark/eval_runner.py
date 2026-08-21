"""
OKX-Dog AI Quant Studio - 黄金测试集基准与模型防退化评测中枢
模块: okx-dog-ai/studio/benchmark/eval_runner.py
角色: 量化回测与现实逻辑检验师 (agency-reality-checker)
功能: 针对典型历史极端行情 (插针/轧空/横盘假突破/暴跌) 批量评测 Prompt 与策略生成的合规率与回测表现
"""

import asyncio
import logging
from typing import Any, Dict, List

from ..ast_guard import validate_python_code_security
from ..sandbox_runner import run_strategy_in_sandbox
from ..codex_client import CodexQuantStudioClient, DEFAULT_TEMPLATES

logger = logging.getLogger("okx_dog.studio.benchmark")

BENCHMARK_PROMPTS = [
    "构建 15m 布林带带宽挤压且放量突破上轨时的开多策略",
    "基于 RSI 底背离且资金费率由负转正时的反弹博弈策略",
    "双均线 EMA10/30 金叉伴随未平仓合约量(OI)显著增加的动量突破因子",
    "ATR 动态波动率通道突破并附带移动止损逻辑",
    "多周期均线空头排列下的反弹遇阻开空策略"
]


class BenchmarkRunner:
    """自动化模型与 Prompt 防退化评测器"""

    def __init__(self, studio_client: CodexQuantStudioClient):
        self.studio_client = studio_client

    async def run_benchmark_suite(self) -> Dict[str, Any]:
        """运行标准基准测试套件"""
        results: List[Dict[str, Any]] = []
        passed_count = 0
        total_sharpe = 0.0

        for prompt in BENCHMARK_PROMPTS:
            res = await self.studio_client.generate_and_verify_strategy(prompt)
            success = res.get("status") in ["VERIFIED", "FALLBACK_VERIFIED"]
            if success:
                passed_count += 1
                metrics = res.get("metrics") or {}
                total_sharpe += float(metrics.get("sharpe_ratio", 0.0))

            results.append({
                "prompt": prompt,
                "status": res.get("status"),
                "attempts": res.get("attempts"),
                "ast_passed": res.get("ast_passed"),
                "metrics": res.get("metrics")
            })

        n = len(BENCHMARK_PROMPTS)
        pass_rate = (passed_count / n) * 100
        avg_sharpe = total_sharpe / max(1, passed_count)

        report = {
            "total_cases": n,
            "passed_cases": passed_count,
            "pass_rate_pct": round(pass_rate, 2),
            "avg_sharpe_ratio": round(avg_sharpe, 2),
            "benchmark_passed": pass_rate >= 80.0,
            "details": results
        }
        return report
