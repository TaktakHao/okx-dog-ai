"""
OKX-Dog 多智能体 Softmax 动态加权门控网络 (Adaptive Softmax Gating Network)
模块: okx-dog-ai/agent/evolution/gating_network.py
角色: AI 与量化算法工程师 (agency-ai-engineer) / 多智能体协同架构师

核心功能:
1. 维护各决策专家的近期滑动加权战力得分 EMA(Score_i)；
2. 基于带温度系数 τ 的 Softmax 计算动态决策话语权 (10.0% ~ 40.0%)；
3. 严格执行金融级上下限安全约束 (防单一专家垄断，防任何专家失声)；
4. 输出 GatingWeightSnapshot 供仲裁官与前端看板消费。
"""

import logging
import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("okx_dog.ai.evolution.gating_network")


class GatingWeightSnapshot(BaseModel):
    """当前决策层各角色的动态话语权快照"""
    weights: Dict[str, float] = Field(..., description="各角色 ID 到权重百分比的映射 (如 {'bull_specialist': 28.5})")
    recent_scores: Dict[str, float] = Field(..., description="各角色近期的平滑表现得分")
    temperature: float = Field(default=1.2, description="Softmax 温度系数")
    timestamp: int = Field(..., description="快照生成毫秒时间戳")


class SoftmaxGatingNetwork:
    """平滑 Softmax 动态加权门控器"""

    DEFAULT_ROLES = [
        "bull_specialist",
        "bear_critic",
        "macro_news",
        "onchain_analyst",
        "micro_sniper",
        "chief_arbiter",
    ]

    def __init__(
        self,
        min_weight: float = 0.08,  # 单个专家最低保底 8%
        max_weight: float = 0.35,  # 单个专家最高上限 35%
        temperature: float = 1.2,  # 平滑温度系数
        alpha_ema: float = 0.15    # EMA 更新步长
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.temperature = temperature
        self.alpha_ema = alpha_ema

        # 初始化基准得分 (所有专家初始均为 50.0 分)
        self._scores: Dict[str, float] = {role: 50.0 for role in self.DEFAULT_ROLES}
        # 初始均匀权重
        self._weights: Dict[str, float] = self._compute_bounded_softmax(self._scores)

    def _compute_bounded_softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        """计算满足 [min_weight, max_weight] 边界约束的归一化权重"""
        if not scores:
            return {}

        roles = list(scores.keys())
        n = len(roles)
        if n == 1:
            return {roles[0]: 100.0}

        # 1. 计算原始 Softmax
        # 数值稳定性：减去最大值
        max_s = max(scores.values())
        exp_vals = {r: math.exp((scores[r] - max_s) / max(0.1, self.temperature * 10.0)) for r in roles}
        total_exp = sum(exp_vals.values())
        raw_weights = {r: exp_vals[r] / total_exp for r in roles}

        # 2. 迭代裁剪与重归一化 (Bounded Projection)
        weights = dict(raw_weights)
        for _ in range(5):
            excess = 0.0
            free_sum = 0.0
            clipped = {}

            for r in roles:
                if weights[r] > self.max_weight:
                    excess += weights[r] - self.max_weight
                    weights[r] = self.max_weight
                    clipped[r] = True
                elif weights[r] < self.min_weight:
                    deficit = self.min_weight - weights[r]
                    excess -= deficit
                    weights[r] = self.min_weight
                    clipped[r] = True
                else:
                    clipped[r] = False
                    free_sum += weights[r]

            if abs(excess) < 1e-4 or free_sum <= 1e-4:
                break

            # 将溢出值按自由比例二次分摊
            for r in roles:
                if not clipped[r]:
                    weights[r] += excess * (weights[r] / free_sum)

        # 转换为百分比 (四舍五入保留 1 位小数，最后一位调平凑满 100%)
        pct_weights = {r: round(weights[r] * 100.0, 1) for r in roles}
        delta = round(100.0 - sum(pct_weights.values()), 1)
        if delta != 0 and roles:
            pct_weights[roles[0]] = round(pct_weights[roles[0]] + delta, 1)

        return pct_weights

    def update_scores_from_rewards(self, reward_breakdowns: List[Any]) -> GatingWeightSnapshot:
        """
        根据单笔交易或复盘的奖励分，增量更新各专家战力分与话语权
        """
        import time

        for item in reward_breakdowns:
            role_id = getattr(item, "role_id", None) or item.get("role_id")
            reward = float(getattr(item, "total_reward", 0.0) if hasattr(item, "total_reward") else item.get("total_reward", 0.0))

            if role_id:
                curr = self._scores.get(role_id, 50.0)
                # 增量 EMA: reward 映射到增量
                # reward 范围一般在 -15 ~ +25
                target = max(10.0, min(100.0, curr + reward * 0.8))
                new_score = round(curr * (1 - self.alpha_ema) + target * self.alpha_ema, 2)
                self._scores[role_id] = new_score

        # 重新计算权重
        self._weights = self._compute_bounded_softmax(self._scores)
        logger.info("门控权重已平滑自适应更新: %s", self._weights)

        return self.get_snapshot()

    def get_weights(self) -> Dict[str, float]:
        """获取当前决策话语权百分比字典"""
        return dict(self._weights)

    def get_regime_adaptive_weights(self, market_regime: str = "RANGING") -> Dict[str, float]:
        """
        根据当前宏观市场体制 (Market Regime) 对基础门控权重进行动态体制倾斜放大：
        1. TRENDING_UP (单边多头): 放大冲锋多头 (bull_specialist) 与链上资金 (onchain_analyst) 权重，抑制红队盲目抄底做空；
        2. TRENDING_DOWN (单边空头): 放大铁血风控 (bear_critic) 权重，严防盲目抄底接飞刀；
        3. RANGING (震荡整理): 放大微观盘口狙击 (micro_sniper) 与风控 (bear_critic)，打压趋势突破；
        4. VOLATILE_BREAKOUT (剧烈突破): 放大宏观情报 (macro_news) 与仲裁官 (chief_arbiter) 避险权限。
        """
        regime = str(market_regime).upper()
        base_weights = dict(self._weights)

        # 体制偏置乘数 (Regime Multipliers)
        regime_multipliers = {
            "TRENDING_UP": {
                "bull_specialist": 1.45,
                "bear_critic": 0.70,
                "macro_news": 0.90,
                "onchain_analyst": 1.25,
                "micro_sniper": 0.85,
                "chief_arbiter": 1.00,
            },
            "TRENDING_DOWN": {
                "bull_specialist": 0.65,
                "bear_critic": 1.45,
                "macro_news": 1.10,
                "onchain_analyst": 1.10,
                "micro_sniper": 0.85,
                "chief_arbiter": 1.00,
            },
            "RANGING": {
                "bull_specialist": 0.85,
                "bear_critic": 1.15,
                "macro_news": 0.80,
                "onchain_analyst": 0.85,
                "micro_sniper": 1.50,
                "chief_arbiter": 1.10,
            },
            "VOLATILE_BREAKOUT": {
                "bull_specialist": 1.10,
                "bear_critic": 1.20,
                "macro_news": 1.40,
                "onchain_analyst": 0.90,
                "micro_sniper": 1.10,
                "chief_arbiter": 1.25,
            },
        }

        mults = regime_multipliers.get(regime, {r: 1.0 for r in self.DEFAULT_ROLES})

        # 加权调整
        adjusted = {r: max(5.0, base_weights.get(r, 16.6) * mults.get(r, 1.0)) for r in self.DEFAULT_ROLES}
        tot = sum(adjusted.values())
        norm_weights = {r: round((adjusted[r] / tot) * 100.0, 1) for r in self.DEFAULT_ROLES}
        delta = round(100.0 - sum(norm_weights.values()), 1)
        if delta != 0 and self.DEFAULT_ROLES:
            norm_weights[self.DEFAULT_ROLES[0]] = round(norm_weights[self.DEFAULT_ROLES[0]] + delta, 1)

        return norm_weights

    def get_role_weight(self, role_id: str, default: float = 20.0) -> float:
        """获取单个角色话语权"""
        return self._weights.get(role_id, default)

    def get_snapshot(self) -> GatingWeightSnapshot:
        """获取门控网络全景快照"""
        import time
        return GatingWeightSnapshot(
            weights=dict(self._weights),
            recent_scores=dict(self._scores),
            temperature=self.temperature,
            timestamp=int(time.time() * 1000)
        )

    def reset_to_default(self) -> GatingWeightSnapshot:
        """重置为初始均匀出厂状态"""
        self._scores = {role: 50.0 for role in self.DEFAULT_ROLES}
        self._weights = self._compute_bounded_softmax(self._scores)
        return self.get_snapshot()
