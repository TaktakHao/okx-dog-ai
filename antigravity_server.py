"""
Antigravity CLI HTTP / WebSocket API 服务网关
模块: okx-dog-ai/antigravity_server.py

特性:
1. 封装本地 Antigravity CLI 为标准 HTTP RESTful 与 WebSocket 双工微服务。
2. 兼容 OpenAI 标准 API 规范 (/v1/chat/completions)，支持流式 (SSE) 与非流式输出。
3. 专用 WebSocket 双工流式通道 (/ws/antigravity/stream)，毫秒级推送思考过程与 Token。
4. 原生支持 JSON Schema 强制结构化校验与思维链 (thinking) 剥离。
5. 自带 CLI 启动入口与健康检查。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

try:
    from .antigravity_bridge import AntigravityBridge, antigravity_bridge
except ImportError:
    from antigravity_bridge import AntigravityBridge, antigravity_bridge

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("okx_dog.antigravity_server")

# 初始化 FastAPI 应用
app = FastAPI(
    title="Antigravity CLI API Gateway",
    description="将 Google Antigravity CLI (agy) 封装为兼容 OpenAI 协议与 WebSocket 的高性能 AI 网关",
    version="1.0.0",
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Pydantic 请求与响应模型 (兼容 OpenAI 协议与原生扩展)
# =============================================================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="角色: system | user | assistant")
    content: str = Field(..., description="消息内容")


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field("gemini-3.7-flash", description="模型名称")
    messages: List[ChatMessage] = Field(..., description="对话历史消息列表")
    temperature: Optional[float] = Field(0.2, description="采样温度")
    max_tokens: Optional[int] = Field(None, description="最大生成 Token 数")
    stream: Optional[bool] = Field(False, description="是否流式输出")
    response_format: Optional[Dict[str, Any]] = Field(None, description="结构化输出格式或 schema")
    effort: Optional[str] = Field("medium", description="推理深度: low | medium | high")


class NativeGenerateRequest(BaseModel):
    prompt: str = Field(..., description="提示词文本")
    system_instruction: Optional[str] = Field(None, description="系统指令")
    schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema 结构化约束")
    effort: Optional[str] = Field("medium", description="推理深度: low | medium | high")
    model: Optional[str] = Field(None, description="指定模型")
    timeout_seconds: Optional[float] = Field(60.0, description="超时时间(秒)")


# =============================================================================
# 辅助函数: 消息列表转 Prompt 文本
# =============================================================================

def _messages_to_prompt(messages: List[ChatMessage], response_format: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
    """将 OpenAI 风格的 messages 列表转换为统一的 Prompt 文本与 schema"""
    prompt_parts: List[str] = []
    schema = None

    if response_format and isinstance(response_format, dict):
        if response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", {}).get("schema")
        elif "schema" in response_format:
            schema = response_format["schema"]

    for msg in messages:
        if msg.role == "system":
            prompt_parts.append(f"[系统指令/System Instruction]:\n{msg.content}\n")
        elif msg.role == "user":
            prompt_parts.append(f"[用户请求/User]:\n{msg.content}\n")
        elif msg.role == "assistant":
            prompt_parts.append(f"[助手回复/Assistant]:\n{msg.content}\n")

    return "\n".join(prompt_parts).strip(), schema


# =============================================================================
# 1. 基础健康检查与状态
# =============================================================================

@app.get("/health", summary="服务健康检查与 CLI 状态检测")
@app.get("/api/v1/antigravity/health", summary="健康检查接口")
async def health_check():
    cli_ready = antigravity_bridge.is_available()
    return {
        "status": "healthy" if cli_ready else "degraded",
        "service": "Antigravity CLI API Gateway",
        "cli_path": antigravity_bridge.cli_path,
        "cli_available": cli_ready,
        "timestamp": int(time.time()),
    }


# =============================================================================
# 2. OpenAI 兼容接口: POST /v1/chat/completions
# =============================================================================

@app.post("/v1/chat/completions", summary="OpenAI 兼容 Chat Completions 接口")
async def chat_completions(req: ChatCompletionRequest):
    """
    完全兼容 OpenAI 协议的对话接口。
    支持非流式 (JSON) 与流式 (text/event-stream)。
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages 不能为空")

    prompt, schema = _messages_to_prompt(req.messages, req.response_format)
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())
    model_name = req.model or "gemini-3.7-flash"

    # A. 流式输出 (Server-Sent Events)
    if req.stream:
        async def event_stream_generator() -> AsyncGenerator[str, None]:
            try:
                # 首先发送起始 chunk
                first_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

                # 流式产生增量
                async for text_delta, think_delta in antigravity_bridge.generate_stream(
                    prompt=prompt,
                    response_schema=schema,
                    effort=req.effort or "medium",
                    model=model_name,
                ):
                    delta_dict: Dict[str, Any] = {}
                    if text_delta:
                        delta_dict["content"] = text_delta
                    if think_delta:
                        delta_dict["reasoning_content"] = think_delta

                    if delta_dict:
                        chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": delta_dict,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                # 结束 chunk
                end_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as exc:
                logger.error(f"[SSE Stream] 发生异常: {exc}", exc_info=True)
                err_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"\n[Error: {exc}]"},
                            "finish_reason": "error",
                        }
                    ],
                }
                yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream_generator(), media_type="text/event-stream")

    # B. 非流式输出 (标准 JSON)
    try:
        content, thinking, latency_ms, actual_model = await antigravity_bridge.generate(
            prompt=prompt,
            response_schema=schema,
            effort=req.effort or "medium",
            model=model_name,
        )

        message_obj: Dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if thinking:
            message_obj["reasoning_content"] = thinking

        response_data = {
            "id": chat_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": actual_model,
            "choices": [
                {
                    "index": 0,
                    "message": message_obj,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(content) // 4,
                "total_tokens": (len(prompt) + len(content)) // 4,
            },
            "latency_ms": latency_ms,
        }
        return JSONResponse(content=response_data)

    except Exception as exc:
        logger.error(f"[ChatCompletions] 调用异常: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# =============================================================================
# 3. 原生结构化接口: POST /api/v1/antigravity/generate
# =============================================================================

@app.post("/api/v1/antigravity/generate", summary="原生结构化推理生成接口")
async def native_generate(req: NativeGenerateRequest):
    full_prompt = req.prompt
    if req.system_instruction:
        full_prompt = f"{req.system_instruction}\n\n{req.prompt}"

    try:
        content, thinking, latency_ms, model_used = await antigravity_bridge.generate(
            prompt=full_prompt,
            response_schema=req.schema,
            effort=req.effort or "medium",
            model=req.model,
            timeout_seconds=req.timeout_seconds,
        )

        structured_data = None
        if req.schema:
            try:
                # 尝试解析为结构化 JSON
                cleaned = content.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.split("\n")
                    if len(lines) >= 2:
                        cleaned = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
                structured_data = json.loads(cleaned)
            except Exception:
                structured_data = None

        return {
            "code": 0,
            "msg": "success",
            "data": {
                "content": content,
                "thinking": thinking,
                "structured": structured_data,
                "latency_ms": latency_ms,
                "model_used": model_used,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 4. 原生 SSE 流式接口: GET /api/v1/antigravity/stream
# =============================================================================

@app.get("/api/v1/antigravity/stream", summary="原生 SSE 流式推理生成接口")
async def native_stream(
    prompt: str = Query(..., description="Prompt 提示词"),
    effort: str = Query("medium", description="推理深度"),
    model: Optional[str] = Query(None, description="指定模型"),
):
    async def sse_gen():
        try:
            async for text_delta, think_delta in antigravity_bridge.generate_stream(
                prompt=prompt,
                effort=effort,
                model=model,
            ):
                payload = {
                    "text": text_delta,
                    "thinking": think_delta,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(sse_gen(), media_type="text/event-stream")


# =============================================================================
# 5. 全双工 WebSocket 流式接口: /ws/antigravity/stream
# =============================================================================

@app.websocket("/ws/antigravity/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    全双工 WebSocket 流式传输通道。
    客户端发送 JSON: {"prompt": "...", "schema": {...}, "effort": "medium"}
    服务端实时推送: {"type": "delta", "text": "...", "thinking": "..."}
                   {"type": "done", "latency_ms": 1234}
    """
    await websocket.accept()
    logger.info("[WebSocket] 客户端已成功连接")

    try:
        while True:
            raw_msg = await websocket.receive_text()
            try:
                msg_data = json.loads(raw_msg)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "非法的 JSON 请求格式"})
                continue

            prompt = msg_data.get("prompt")
            if not prompt:
                await websocket.send_json({"type": "error", "message": "prompt 字段为必填项"})
                continue

            schema = msg_data.get("schema")
            effort = msg_data.get("effort", "medium")
            model = msg_data.get("model")

            start_t = time.time()
            try:
                async for text_delta, think_delta in antigravity_bridge.generate_stream(
                    prompt=prompt,
                    response_schema=schema,
                    effort=effort,
                    model=model,
                ):
                    if text_delta or think_delta:
                        await websocket.send_json({
                            "type": "delta",
                            "text": text_delta,
                            "thinking": think_delta,
                        })

                latency_ms = int((time.time() - start_t) * 1000)
                await websocket.send_json({
                    "type": "done",
                    "latency_ms": latency_ms,
                })
            except Exception as e:
                logger.error(f"[WebSocket] 推理流发生错误: {e}", exc_info=True)
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        logger.info("[WebSocket] 客户端断开连接")
    except Exception as e:
        logger.warning(f"[WebSocket] 异常关闭: {e}")



# =============================================================================
# Paperclip 虚拟量化公司与 CFO 战绩配额控制平面路由
# =============================================================================

@app.get("/api/v1/paperclip/org", summary="获取虚拟量化公司组织架构与 6 大员工资料")
async def get_paperclip_org():
    """返回符合 Paperclip 规范的组织架构、CEO 及在册员工清单"""
    try:
        from agent.paperclip_adapter import paperclip_adapter
    except ImportError:
        from .agent.paperclip_adapter import paperclip_adapter
    return paperclip_adapter.get_company_org()


@app.get("/api/v1/paperclip/quota/overview", summary="获取团队及员工剩余可用奖励额度概览")
async def get_paperclip_quota_overview():
    """返回长期主义战绩积分、累计奖金池与当前员工剩余额度"""
    try:
        from agent.evolution.cfo_performance_manager import CfoPerformanceManager
    except ImportError:
        from .agent.evolution.cfo_performance_manager import CfoPerformanceManager
    cfo = CfoPerformanceManager.get_instance()
    return cfo.get_company_quota_overview()


@app.post("/api/v1/paperclip/task/anomaly", summary="接收指标哨兵异动事件并触发投决会仲裁工单")
async def trigger_paperclip_anomaly_task(request: Request):
    """
    接收来自 OKX-Dog 哨兵的异动事件，唤醒 6 大员工红蓝对抗博弈，
    若有员工拥有战绩奖励则自动解锁深度思考链，输出仲裁决议。
    """
    try:
        from agent.paperclip_adapter import paperclip_adapter, MarketAnomalyTriggerRequest
    except ImportError:
        from .agent.paperclip_adapter import paperclip_adapter, MarketAnomalyTriggerRequest

    body = await request.json()
    req = MarketAnomalyTriggerRequest(**body)
    return await paperclip_adapter.handle_anomaly_event(req)


@app.post("/api/v1/paperclip/reward/trade", summary="实盘平仓盈利向立功员工派发战绩奖励")
async def reward_trade_profit(request: Request):
    """平仓产生真实利润时，向多头冲锋员与执行 CEO 派发 Tokens 奖励"""
    try:
        from agent.evolution.cfo_performance_manager import CfoPerformanceManager
    except ImportError:
        from .agent.evolution.cfo_performance_manager import CfoPerformanceManager
    cfo = CfoPerformanceManager.get_instance()
    body = await request.json()
    trade_id = body.get("trade_id", f"tr_{int(time.time())}")
    pnl_usdt = float(body.get("pnl_usdt", 0.0))
    side = body.get("side", "LONG")
    cfo.reward_trade_profit(trade_id=trade_id, pnl_usdt=pnl_usdt, side=side)
    return {"status": "ok", "message": f"战绩奖励派发结算完成: +{pnl_usdt:.2f} USDT"}


@app.post("/api/v1/paperclip/reward/hedge", summary="空头风控专家成功排雷避险奖励")
async def reward_hedge(request: Request):
    """成功识别假突破、排雷阻断重大亏损时，向空头专家派发避险奖金"""
    try:
        from agent.evolution.cfo_performance_manager import CfoPerformanceManager
    except ImportError:
        from .agent.evolution.cfo_performance_manager import CfoPerformanceManager
    cfo = CfoPerformanceManager.get_instance()
    body = await request.json()
    symbol = body.get("symbol", "BTC-USDT-SWAP")
    reason = body.get("reason", "成功识别假突破")
    cfo.record_successful_hedge(symbol=symbol, reason=reason)
    return {"status": "ok", "message": f"避险排雷奖励派发完成: {symbol}"}


# =============================================================================
# CLI 独立启动入口
# =============================================================================

def run_server():
    """启动独立微服务"""
    port = int(os.getenv("ANTIGRAVITY_PORT", "8001"))
    host = os.getenv("ANTIGRAVITY_HOST", "0.0.0.0")
    print(f"🚀 启动 Antigravity API Gateway 服务: http://{host}:{port}")
    uvicorn.run("antigravity_server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
