"""
测试套件: Paperclip 适配网关与 CFO 战绩激励引擎
文件: okx-dog-ai/tests/test_cfo_performance_and_paperclip.py
"""

import os
import tempfile
import pytest
from unittest.mock import MagicMock

from agent.paperclip_adapter import PaperclipAdapter, MarketAnomalyTriggerRequest
from agent.evolution.cfo_performance_manager import CfoPerformanceManager


@pytest.fixture
def temp_cfo_manager():
    """创建隔离的测试用 CFO 数据库管理器"""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_evolution.db")
    cfo = CfoPerformanceManager(db_path=db_path)
    yield cfo


def test_paperclip_adapter_org_chart():
    """验证 Paperclip 组织架构能正确暴露 6 大员工资料"""
    adapter = PaperclipAdapter()
    org = adapter.get_company_org()

    assert org.company_name == "OKX-Dog Autonomous Quant Fund"
    assert org.ceo_role == "CHIEF_ARBITER"
    assert len(org.employees) == 6

    roles = [e.role for e in org.employees]
    assert "CHIEF_ARBITER" in roles
    assert "BULL_SPECIALIST" in roles
    assert "BEAR_CRITIC" in roles
    assert "MACRO_NEWS" in roles
    assert "MICROSTRUCTURE" in roles
    assert "CFO_AUDITOR" in roles


def test_cfo_reward_trade_profit_and_quota(temp_cfo_manager):
    """验证实盘交易盈利能正确为多头冲锋员与 CEO 派发奖励额度"""
    cfo = temp_cfo_manager

    # 获取初始额度
    overview_before = cfo.get_company_quota_overview()
    bull_before = next(e for e in overview_before.employees_quota if e.role == "BULL_SPECIALIST")
    ceo_before = next(e for e in overview_before.employees_quota if e.role == "CHIEF_ARBITER")

    init_bull_remain = bull_before.remaining_bonus_tokens
    init_ceo_remain = ceo_before.remaining_bonus_tokens

    # 模拟一笔盈利 +50 USDT 的交易结算
    cfo.reward_trade_profit(trade_id="tr_test_001", pnl_usdt=50.0, side="LONG")

    # 再次查询额度
    overview_after = cfo.get_company_quota_overview()
    bull_after = next(e for e in overview_after.employees_quota if e.role == "BULL_SPECIALIST")
    ceo_after = next(e for e in overview_after.employees_quota if e.role == "CHIEF_ARBITER")

    # 总派发 50,000 Tokens (多头 30,000, CEO 20,000)
    assert bull_after.remaining_bonus_tokens == init_bull_remain + 30000
    assert ceo_after.remaining_bonus_tokens == init_ceo_remain + 20000
    assert bull_after.deep_thinking_unlocked is True


def test_cfo_successful_hedge_award(temp_cfo_manager):
    """验证空头风控专家排雷避险成功获得奖励"""
    cfo = temp_cfo_manager
    overview_before = cfo.get_company_quota_overview()
    bear_before = next(e for e in overview_before.employees_quota if e.role == "BEAR_CRITIC")
    init_hedges = bear_before.successful_hedges_count
    init_remain = bear_before.remaining_bonus_tokens

    cfo.record_successful_hedge(symbol="BTC-USDT-SWAP", reason="顶背离诱多排雷")

    overview_after = cfo.get_company_quota_overview()
    bear_after = next(e for e in overview_after.employees_quota if e.role == "BEAR_CRITIC")

    assert bear_after.successful_hedges_count == init_hedges + 1
    assert bear_after.remaining_bonus_tokens == init_remain + 8000


def test_cfo_consume_bonus_quota(temp_cfo_manager):
    """验证在深度研判任务中核销员工战绩额度"""
    cfo = temp_cfo_manager
    overview_before = cfo.get_company_quota_overview()
    init_total_remain = overview_before.remaining_pool_tokens

    consumed = cfo.consume_bonus_quota_for_task(symbol="ETH-USDT-SWAP", amount=2000)
    assert consumed == 2000

    overview_after = cfo.get_company_quota_overview()
    assert overview_after.remaining_pool_tokens == init_total_remain - 2000


@pytest.mark.asyncio
async def test_paperclip_adapter_anomaly_event_handling():
    """验证异动事件能触发投决会仲裁工单输出"""
    adapter = PaperclipAdapter()
    req = MarketAnomalyTriggerRequest(
        symbol="BTC-USDT-SWAP",
        trigger_type="TREND_BREAKOUT",
        price=68500.0,
        anomaly_description="15m/1h 均线金叉放量，突破前高阻力位"
    )

    res = await adapter.handle_anomaly_event(req)
    assert res.task_id.startswith("task_")
    assert res.symbol == "BTC-USDT-SWAP"
    assert res.consensus_score >= 0 and res.consensus_score <= 100
    assert len(res.debate_opinions) == 3  # 多头、空头、宏观
