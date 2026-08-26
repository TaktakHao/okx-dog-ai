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
    from ..dataset.collector import dataset_collector
    from .evolution.intern_slot import intern_slot_manager
except (ImportError, ValueError):
    try:
        from okx_dog_ai.schemas import AIAnalysisResponse, MarketContextSnapshot, SSEStreamChunk
        from okx_dog_ai.dataset.collector import dataset_collector
        from okx_dog_ai.agent.evolution.intern_slot import intern_slot_manager
    except (ImportError, ValueError):
        from schemas import AIAnalysisResponse, MarketContextSnapshot, SSEStreamChunk
        try:
            from dataset.collector import dataset_collector
            from agent.evolution.intern_slot import intern_slot_manager
        except Exception:
            dataset_collector = None
            intern_slot_manager = None

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

    def _dispatch_shadow_and_collection(
        self,
        snapshot: Dict[str, Any],
        final_response_obj: AIAnalysisResponse,
        thinking_steps: Optional[List[Dict[str, Any]]] = None,
    ):
        """异步分发数据收集与实习生影子推演 (零阻塞主决策链路)"""
        try:
            resp_dict = final_response_obj.model_dump() if hasattr(final_response_obj, "model_dump") else dict(final_response_obj)
            current_price = float(snapshot.get("current_price", 0.0))
            symbol = snapshot.get("symbol", "BTC-USDT-SWAP")

            # 1. 沉淀数据
            if dataset_collector:
                dataset_collector.record_decision_sample(
                    symbol=symbol,
                    current_price=current_price,
                    market_snapshot=snapshot,
                    final_response=resp_dict,
                    thinking_steps=thinking_steps,
                    analysis_id=resp_dict.get("analysis_id"),
                )

            # 2. 触发实习生影子推演
            if intern_slot_manager and intern_slot_manager.is_enabled():
                import asyncio
                asyncio.create_task(
                    intern_slot_manager.trigger_shadow_inference(
                        symbol=symbol,
                        current_price=current_price,
                        market_snapshot=snapshot,
                        senior_response=resp_dict,
                        analysis_id=resp_dict.get("analysis_id"),
                    )
                )
        except Exception as e:
            logger.warning("数据沉淀或实习生推演触发失败 (不影响主链路): %s", e)

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
            resp_obj = final_dict
        else:
            resp_obj = AIAnalysisResponse(**final_dict)

        # 异步沉淀与触发实习生推演
        snap_dict = snapshot.model_dump() if hasattr(snapshot, "model_dump") else dict(snapshot)
        self._dispatch_shadow_and_collection(
            snapshot=snap_dict,
            final_response_obj=resp_obj,
            thinking_steps=final_state.get("thinking_steps", []),
        )

        return resp_obj

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
        accumulated_steps: List[Dict[str, Any]] = []

        # 逐节点流式执行
        async for output in self.graph.astream(initial_state, stream_mode="updates"):
            for node_name, node_state_delta in output.items():
                steps = node_state_delta.get("thinking_steps", [])
                for step in steps:
                    accumulated_steps.append(step)
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
            # 异步沉淀与触发实习生推演
            snap_dict = snapshot.model_dump() if hasattr(snapshot, "model_dump") else dict(snapshot)
            self._dispatch_shadow_and_collection(
                snapshot=snap_dict,
                final_response_obj=final_response_obj,
                thinking_steps=accumulated_steps,
            )

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
