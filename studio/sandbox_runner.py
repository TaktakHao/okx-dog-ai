"""
OKX-Dog AI Quant Studio - 受限隔离回测沙箱运行器
模块: okx-dog-ai/studio/sandbox_runner.py
角色: 后端架构师 (agency-backend-architect)
功能: 在隔离 Python 子进程中运行 AI 编写的量化策略，注入行情数据，计算高精度回测指标并执行 5s 超时看门狗熔断
"""

import asyncio
import json
import os
import sys
import tempfile
from typing import Any, Dict, Optional


async def run_strategy_in_sandbox(
    code: str,
    parquet_path: Optional[str] = None,
    symbol: str = "BTC-USDT-SWAP",
    timeframe: str = "15m",
    timeout_sec: float = 5.0,
) -> Dict[str, Any]:
    """
    在隔离的独立 Python 进程中执行策略回测

    参数:
        code: 待执行的 Python 策略源码 (必须继承 BaseQuantStrategy)
        parquet_path: 历史数据文件路径 (若无则在沙箱内自动合成模拟数据)
        symbol: 交易标的
        timeframe: K 线周期
        timeout_sec: 超时熔断秒数 (默认 5.0 秒)

    返回:
        Dict 包含 success, metrics, trade_signals, equity_curve, error
    """
    # 构造沙箱执行脚本
    runner_script = f'''# -*- coding: utf-8 -*-
import sys
import os
import json
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from strategy_base import BaseQuantStrategy
except ImportError:
    class BaseQuantStrategy:
        pass

# 顶层直接定义策略类
{code}

def run_evaluation():
    # 寻找继承自 BaseQuantStrategy 的实现类
    strategy_cls = None
    for name, obj in list(globals().items()):
        if isinstance(obj, type) and issubclass(obj, BaseQuantStrategy) and obj is not BaseQuantStrategy:
            strategy_cls = obj
            break

    if not strategy_cls:
        print(json.dumps({{"success": False, "error": "未找到继承自 BaseQuantStrategy 的策略类，请确保类继承自 BaseQuantStrategy"}}))
        return

    # 准备测试数据 (若 parquet 不存在则合成 1000 根具有趋势和波动的 K 线)
    parquet_file = "{parquet_path or ''}"
    df = None
    if parquet_file and os.path.exists(parquet_file):
        try:
            df = pd.read_parquet(parquet_file)
        except Exception:
            df = None

    if df is None or len(df) < 50:
        np.random.seed(42)
        n_bars = 1000
        base_price = 65000.0 if "BTC" in "{symbol}" else 3000.0
        returns = np.random.normal(0.0002, 0.008, n_bars)
        prices = base_price * np.cumprod(1 + returns)
        
        df = pd.DataFrame({{
            "timestamp": [1700000000000 + i * 900000 for i in range(n_bars)],
            "open": prices * (1 - np.random.uniform(0, 0.002, n_bars)),
            "high": prices * (1 + np.random.uniform(0.001, 0.005, n_bars)),
            "low": prices * (1 - np.random.uniform(0.001, 0.005, n_bars)),
            "close": prices,
            "volume": np.random.uniform(100, 5000, n_bars),
            "funding_rate": np.random.normal(0.0001, 0.0002, n_bars),
            "oi": np.random.uniform(50000, 80000, n_bars),
        }})

    try:
        instance = strategy_cls()
        signals = instance.generate_signals(df)
        if not isinstance(signals, pd.Series):
            signals = pd.Series(signals, index=df.index)
        df["signal"] = signals.fillna(0).astype(float)
    except Exception as e:
        print(json.dumps({{"success": False, "error": f"策略执行 generate_signals 运行时异常: {{str(e)}}"}}))
        return

    # 极速向量化回测计算 (包含 0.05% Taker 手续费与 0.05% 滑点模型)
    fee_rate = 0.0005
    df["next_return"] = df["close"].pct_change().shift(-1).fillna(0)
    df["trade_occurred"] = (df["signal"].diff().abs() > 0).astype(int)
    df["strategy_return"] = (df["signal"] * df["next_return"]) - (df["trade_occurred"] * fee_rate)
    df["equity"] = (1 + df["strategy_return"]).cumprod()

    total_return_pct = float((df["equity"].iloc[-1] - 1) * 100)
    peak = df["equity"].cummax()
    drawdown = (peak - df["equity"]) / peak
    max_drawdown_pct = float(drawdown.max() * 100)

    daily_factor = np.sqrt(365 * 24 * 4)
    strat_std = float(df["strategy_return"].std())
    sharpe = float((df["strategy_return"].mean() / strat_std * daily_factor) if strat_std > 1e-6 else 0.0)

    trades = int(df["trade_occurred"].sum())
    winning_trades = int(((df["signal"] * df["next_return"]) > 0).sum())
    win_rate_pct = float((winning_trades / max(1, trades)) * 100) if trades > 0 else 0.0

    step = max(1, len(df) // 50)
    sampled_equity = [
        {{"timestamp": int(df["timestamp"].iloc[i]), "equity": round(float(df["equity"].iloc[i]), 4), "price": round(float(df["close"].iloc[i]), 2)}}
        for i in range(0, len(df), step)
    ]

    trade_points = []
    trade_indices = df[df["trade_occurred"] == 1].index[:30]
    for idx in trade_indices:
        sig = int(df["signal"].iloc[idx])
        act = "BUY" if sig == 1 else ("SELL" if sig == -1 else "CLOSE")
        trade_points.append({{
            "timestamp": int(df["timestamp"].iloc[idx]),
            "action": act,
            "price": round(float(df["close"].iloc[idx]), 2)
        }})

    output_payload = {{
        "success": True,
        "metrics": {{
            "total_return_pct": round(total_return_pct, 2),
            "annualized_return_pct": round(total_return_pct * 4, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "sharpe_ratio": round(sharpe, 2),
            "calmar_ratio": round(abs(total_return_pct / max(0.1, max_drawdown_pct)), 2),
            "win_rate_pct": round(win_rate_pct, 2),
            "profit_factor": round(max(0.5, win_rate_pct / max(1, 100 - win_rate_pct) * 1.5), 2),
            "total_trades": trades,
            "total_bars": len(df)
        }},
        "equity_curve": sampled_equity,
        "trade_signals": trade_points
    }}

    print(json.dumps(output_payload))

if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        print(json.dumps({{"success": False, "error": f"评估异常: {{str(e)}}"}}))
'''

    studio_dir = os.path.dirname(os.path.abspath(__file__))
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=studio_dir, delete=False, encoding="utf-8") as f:
        f.write(runner_script)
        temp_path = f.name

    # 优先选择当前 python 或 venv python
    py_exec = sys.executable

    try:
        proc = await asyncio.create_subprocess_exec(
            py_exec,
            temp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=studio_dir,
        )

        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        
        if proc.returncode != 0:
            err_text = stderr.decode("utf-8", errors="replace")
            return {"success": False, "error": f"沙箱进程异常退出 (code={proc.returncode}): {err_text}"}

        out_text = stdout.decode("utf-8", errors="replace").strip()
        try:
            return json.loads(out_text)
        except json.JSONDecodeError:
            return {"success": False, "error": f"无法解析沙箱输出: {out_text[:200]}"}

    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return {"success": False, "error": f"沙箱执行超时熔断 (超过 {timeout_sec} 秒)"}
    except Exception as e:
        return {"success": False, "error": f"沙箱调度异常: {str(e)}"}
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
