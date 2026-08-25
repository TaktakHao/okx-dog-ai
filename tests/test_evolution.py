"""
OKX-Dog 多智能体强化奖励与自适应进化中枢单元测试
模块: okx-dog-ai/tests/test_evolution.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.evolution.reward_engine import RewardEngine
from agent.evolution.gating_network import SoftmaxGatingNetwork
from agent.evolution.evolution_manager import AgentEvolutionManager


class TestEvolutionEngine(unittest.TestCase):

    def test_reward_engine_bull_win(self):
        """测试多头盈利时多头分析师获得高奖励"""
        opinions = {
            "bull_specialist": {"role_name": "多头辩护专家", "stance": "BULLISH", "confidence": 0.8},
            "bear_critic": {"role_name": "空头风控专家", "stance": "BEARISH", "confidence": 0.5},
            "macro_news": {"role_name": "宏观舆情专家", "stance": "BULLISH", "confidence": 0.7},
        }
        outcome = RewardEngine.evaluate_trade_outcome(
            trade_id="test_01",
            symbol="BTC-USDT-SWAP",
            pos_side="LONG",
            entry_price=95000.0,
            exit_price=97000.0,
            sl_price=94000.0,
            realized_pnl=200.0,
            mfe_pct=0.025,
            mae_pct=0.002,
            agent_opinions=opinions,
            is_crisis_defended=False
        )
        self.assertGreater(outcome.realized_rr, 1.5)
        self.assertTrue(outcome.direction_correct)
        
        bull_bd = next(b for b in outcome.breakdowns if b.role_id == "bull_specialist")
        self.assertGreater(bull_bd.total_reward, 10.0)

    def test_reward_engine_crisis_defense(self):
        """测试插针风险被阻断时风控挑刺官获得排雷立功奖励"""
        opinions = {
            "bull_specialist": {"role_name": "多头辩护专家", "stance": "BULLISH", "confidence": 0.9},
            "bear_critic": {"role_name": "空头风控专家", "stance": "BEARISH", "confidence": 0.8},
        }
        outcome = RewardEngine.evaluate_trade_outcome(
            trade_id="test_02",
            symbol="BTC-USDT-SWAP",
            pos_side="FLAT",
            entry_price=95000.0,
            exit_price=95000.0,
            sl_price=94000.0,
            realized_pnl=0.0,
            mfe_pct=0.0,
            mae_pct=0.03,
            agent_opinions=opinions,
            is_crisis_defended=True
        )
        bear_bd = next(b for b in outcome.breakdowns if b.role_id == "bear_critic")
        self.assertGreater(bear_bd.risk_defense_bonus, 0.0)

    def test_softmax_gating_bounds(self):
        """测试 Softmax 门控权重始终在 [8%, 35%] 范围内且总和为 100%"""
        gating = SoftmaxGatingNetwork(min_weight=0.08, max_weight=0.35)
        weights = gating.get_weights()
        self.assertEqual(len(weights), 6)
        self.assertAlmostEqual(sum(weights.values()), 100.0, places=0)
        for r, w in weights.items():
            self.assertTrue(8.0 <= w <= 35.0, f"角色 {r} 权重 {w} 超出边界 [8, 35]")

    def test_evolution_manager_lifecycle(self):
        """测试员工演进管理器的生命周期与一键回滚"""
        manager = AgentEvolutionManager.get_instance()
        status = manager.get_team_status()
        self.assertEqual(len(status.team_members), 6)
        self.assertEqual(status.harness_baseline_status, "STABLE")

        # 执行一次模拟结算
        opinions = {
            "bull_specialist": {"role_name": "多头辩护专家", "stance": "BULLISH", "confidence": 0.8},
            "bear_critic": {"role_name": "空头风控专家", "stance": "BEARISH", "confidence": 0.6},
            "macro_news": {"role_name": "宏观舆情专家", "stance": "BULLISH", "confidence": 0.7},
            "micro_sniper": {"role_name": "微观盘口分析师", "stance": "BULLISH", "confidence": 0.75},
            "chief_arbiter": {"role_name": "首席量化仲裁官", "stance": "BULLISH", "confidence": 0.85},
        }
        outcome = manager.process_trade_reinforcement(
            trade_id="test_03",
            symbol="BTC-USDT-SWAP",
            pos_side="LONG",
            entry_price=95000.0,
            exit_price=97000.0,
            sl_price=94000.0,
            realized_pnl=150.0,
            mfe_pct=0.02,
            mae_pct=0.003,
            agent_opinions=opinions
        )
        self.assertEqual(outcome.realized_pnl, 150.0)
        self.assertGreater(manager.total_evolution_rounds, 12)

        # 测试一键回滚
        rollback_status = manager.rollback_to_baseline()
        self.assertEqual(rollback_status.harness_baseline_status, "ROLLED_BACK")


if __name__ == "__main__":
    unittest.main()
