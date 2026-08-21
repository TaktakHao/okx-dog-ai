"""
OKX-Dog LangGraph 资深量化交易员执行器与 SSE 流式生成器
模块: okx-dog-ai/agent/runner.py
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Optional, Union

from .graph import create_quant_trader_graph
from .state import QuantTraderState

try:
    from ..schemas import AIAnalysisResponse, MarketContextSnapshot, SSEStreamChunk
except (ImportError, ValueError):
    try:
        from okx_dog_ai.schemas import AIAnalysisResponse, MarketContextSnapshot, SSEStreamChunk
    except (ImportError, ValueError):
        from schemas import AIAnalysisResponse, MarketContextSnapshot, SSEStreamChunk

logger = logging.getLogger("okx_dog.ai.agent.runner")


class QuantTraderAgentRunner:
    """
    OKX-Dog 资深量化交易员 Agent 高层异步执行器
    """

    def __init__(self, default_llm_config: Optional[Dict[str, Any]] = None):
        self.default_llm_config = default_llm_config or {}
        self.graph = create_quant_trader_graph()

    def _prepare_initial_state(
        self,
        snapshot: Union[MarketContextSnapshot, Dict[str, Any]],
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> QuantTraderState:
        """从输入快照组装初始 State"""
        if hasattr(snapshot, "model_dump"):
            snap_dict = snapshot.model_dump()
        else:
            snap_dict = dict(snapshot)

        now_ms = int(time.time() * 1000)
        current_price = float(snap_dict.get("current_price", 0.0))
        symbol = snap_dict.get("symbol", "BTC-USDT-SWAP")
        analysis_id = snap_dict.get("analysis_id") or str(uuid.uuid4())
        account_bal = float(snap_dict.get("account_balance_usdt", 1000.0))
        risk_limits = snap_dict.get("risk_limits", {})
        if hasattr(risk_limits, "model_dump"):
            risk_limits = risk_limits.model_dump()

        effective_llm_config = {**self.default_llm_config, **(llm_config or {})}

        initial_state: QuantTraderState = {
            "symbol": symbol,
            "current_price": current_price,
            "analysis_id": analysis_id,
            "timestamp": now_ms,
            "market_snapshot": snap_dict,
            "account_balance_usdt": account_bal,
            "risk_limits": risk_limits,
            "llm_config": effective_llm_config,
            "critique_count": 0,
            "risk_passed": False,
            "thinking_steps": [],
        }
        return initial_state

    async def run(
        self,
        snapshot: Union[MarketContextSnapshot, Dict[str, Any]],
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> AIAnalysisResponse:
        """
        异步非流式执行量化交易员决策状态机
        """
        initial_state = self._prepare_initial_state(snapshot, llm_config)
        logger.info("启动 LangGraph 量化交易员状态机推理: symbol=%s", initial_state["symbol"])

        final_state = await self.graph.ainvoke(initial_state)

        final_dict = final_state.get("final_response")
        if not final_dict:
            raise RuntimeError("LangGraph 状态机未成功生成 final_response")

        if isinstance(final_dict, AIAnalysisResponse):
            return final_dict
        return AIAnalysisResponse(**final_dict)

    async def run_stream(
        self,
        snapshot: Union[MarketContextSnapshot, Dict[str, Any]],
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[SSEStreamChunk, None]:
        """
        异步 SSE 流式执行，逐步推送各节点思维链与最终决策结果
        """
        initial_state = self._prepare_initial_state(snapshot, llm_config)
        symbol = initial_state["symbol"]
        logger.info("启动 LangGraph 量化交易员 SSE 流式推理: symbol=%s", symbol)

        yield SSEStreamChunk(
            event="start",
            data=f"🚀 OKX-Dog 资深量化交易员 Agent 启动研判: {symbol}",
        )

        final_response_obj: Optional[AIAnalysisResponse] = None

        # 逐节点流式执行
        async for output in self.graph.astream(initial_state, stream_mode="updates"):
            for node_name, node_state_delta in output.items():
                steps = node_state_delta.get("thinking_steps", [])
                for step in steps:
                    thought_text = step.get("thought", "")
                    stage_name = step.get("stage_name", node_name)
                    yield SSEStreamChunk(
                        event="think",
                        data=f"【{stage_name}】: {thought_text}",
                        reasoning_content=f"[{stage_name}]\n{thought_text}\n\n",
                    )

                if "final_response" in node_state_delta:
                    final_data = node_state_delta["final_response"]
                    if isinstance(final_data, AIAnalysisResponse):
                        final_response_obj = final_data
                    else:
                        final_response_obj = AIAnalysisResponse(**final_data)

        if final_response_obj:
            yield SSEStreamChunk(
                event="done",
                data=final_response_obj,
                structured_output=final_response_obj,
            )
        else:
            yield SSEStreamChunk(
                event="error",
                data="Agent 执行未能生成合规结果",
                error_message="未能生成合规的 AIAnalysisResponse",
            )
