"""
OKX-Dog 多智能体自适应强化奖励、动态权重门控与 AI 员工档案中枢
模块: okx-dog-ai/agent/evolution/__init__.py
"""

from .reward_engine import (
    RewardEngine,
    MultiAgentRewardOutcome,
    AgentRewardBreakdown,
)
from .gating_network import (
    SoftmaxGatingNetwork,
    GatingWeightSnapshot,
)
from .evolution_manager import (
    AgentEvolutionManager,
    AgentEmployeeModel,
    TeamEvolutionStatusModel,
    EvolutionSnapshotModel,
)

__all__ = [
    "RewardEngine",
    "MultiAgentRewardOutcome",
    "AgentRewardBreakdown",
    "SoftmaxGatingNetwork",
    "GatingWeightSnapshot",
    "AgentEvolutionManager",
    "AgentEmployeeModel",
    "TeamEvolutionStatusModel",
    "EvolutionSnapshotModel",
]
