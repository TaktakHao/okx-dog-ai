"""
OKX-Dog AI 与量化算法决策核心包
"""

import sys
from pathlib import Path
_ai_dir = Path(__file__).resolve().parent
if str(_ai_dir) not in sys.path:
    sys.path.insert(0, str(_ai_dir))

try:
    from okx_dog_ai.schemas import (
        MarketRegime,
        TimeframeTrend,
        FundingRateBias,
        SignalAction,
        SignalUrgency,
        TradePlanOrderType,
        LLMProvider,
        SinglePeriodIndicators,
        MultiPeriodIndicators,
        DerivativesMetrics,
        PositionSnapshot,
        HardRiskLimits,
        MarketContextSnapshot,
        TimeframeDetail,
        TimeframeAnalysis,
        DerivativesSentiment,
        AISignal,
        TakeProfitLevel,
        TradePlan,
        RiskAssessment,
        AIAnalysisResponse,
        SSEStreamChunk,
        AICoTStreamEvent,
    )
    from okx_dog_ai.indicator_engine import KlineBar, TimeframeState, IndicatorEngine
    from okx_dog_ai.prompt_builder import MarketPromptBuilder, TokenEstimator
    from okx_dog_ai.llm_client import LLMClient, LLMInferenceError
    from okx_dog_ai.parser import RobustJSONParser, ThoughtStreamExtractor
    from okx_dog_ai.config import AIModelConfig, ai_settings
    from okx_dog_ai.antigravity_bridge import AntigravityBridge, antigravity_bridge
    from okx_dog_ai.agent import (
        QuantTraderState,
        ThinkingStep,
        create_quant_trader_graph,
        QuantTraderAgentRunner,
    )
except Exception:
    from schemas import (
        MarketRegime,
        TimeframeTrend,
        FundingRateBias,
        SignalAction,
        SignalUrgency,
        TradePlanOrderType,
        LLMProvider,
        SinglePeriodIndicators,
        MultiPeriodIndicators,
        DerivativesMetrics,
        PositionSnapshot,
        HardRiskLimits,
        MarketContextSnapshot,
        TimeframeDetail,
        TimeframeAnalysis,
        DerivativesSentiment,
        AISignal,
        TakeProfitLevel,
        TradePlan,
        RiskAssessment,
        AIAnalysisResponse,
        SSEStreamChunk,
        AICoTStreamEvent,
    )
    from indicator_engine import KlineBar, TimeframeState, IndicatorEngine
    from prompt_builder import MarketPromptBuilder, TokenEstimator
    from llm_client import LLMClient, LLMInferenceError
    from parser import RobustJSONParser, ThoughtStreamExtractor
    from config import AIModelConfig, ai_settings
    from antigravity_bridge import AntigravityBridge, antigravity_bridge
    from agent import (
        QuantTraderState,
        ThinkingStep,
        create_quant_trader_graph,
        QuantTraderAgentRunner,
    )

__all__ = [
    "MarketRegime",
    "TimeframeTrend",
    "FundingRateBias",
    "SignalAction",
    "SignalUrgency",
    "TradePlanOrderType",
    "LLMProvider",
    "SinglePeriodIndicators",
    "MultiPeriodIndicators",
    "DerivativesMetrics",
    "PositionSnapshot",
    "HardRiskLimits",
    "MarketContextSnapshot",
    "TimeframeDetail",
    "TimeframeAnalysis",
    "DerivativesSentiment",
    "AISignal",
    "TakeProfitLevel",
    "TradePlan",
    "RiskAssessment",
    "AIAnalysisResponse",
    "SSEStreamChunk",
    "AICoTStreamEvent",
    "KlineBar",
    "TimeframeState",
    "IndicatorEngine",
    "MarketPromptBuilder",
    "TokenEstimator",
    "LLMClient",
    "LLMInferenceError",
    "RobustJSONParser",
    "ThoughtStreamExtractor",
    "AIModelConfig",
    "ai_settings",
    "AntigravityBridge",
    "antigravity_bridge",
    "QuantTraderState",
    "ThinkingStep",
    "create_quant_trader_graph",
    "QuantTraderAgentRunner",
]

