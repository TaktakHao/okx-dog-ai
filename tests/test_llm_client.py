"""
OKX-Dog LLM 客户端与多模型适配单元测试
模块: okx-dog-ai/tests/test_llm_client.py
"""

import json
import pytest
import httpx
from okx_dog_ai.llm_client import LLMClient, LLMInferenceError


def test_llm_model_adapter_payload():
    """测试多大模型参数抹平与适配"""
    client = LLMClient(
        provider="deepseek",
        base_url="https://api.deepseek.com",
        api_key="sk-test",
        model="deepseek-reasoner",
    )

    messages = [{"role": "user", "content": "hello"}]
    schema = {"type": "object", "properties": {"signal": {"type": "string"}}}

    # 1. 测试 DeepSeek-R1 (deepseek-reasoner)
    payload_r1 = client._adapt_request_payload(
        model="deepseek-reasoner",
        messages=messages,
        temperature=0.6,
        max_tokens=4096,
        response_schema=schema,
        stream=True,
    )
    assert payload_r1["model"] == "deepseek-reasoner"
    assert payload_r1["temperature"] == 0.6
    assert payload_r1["max_tokens"] == 4096
    # reasoner 目前不强制加 response_format
    assert "response_format" not in payload_r1

    # 2. 测试 GPT-4o
    payload_gpt = client._adapt_request_payload(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
        max_tokens=2048,
        response_schema=schema,
        stream=False,
    )
    assert payload_gpt["model"] == "gpt-4o"
    assert payload_gpt["temperature"] == 0.2
    assert "response_format" in payload_gpt
    assert payload_gpt["response_format"]["json_schema"]["strict"] is True

    # 3. 测试 o1-mini
    payload_o1 = client._adapt_request_payload(
        model="o1-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=8192,
        stream=False,
    )
    assert payload_o1["model"] == "o1-mini"
    assert "temperature" not in payload_o1
    assert payload_o1["max_completion_tokens"] == 8192
    assert payload_o1["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_llm_client_mock_non_stream():
    """使用 MockTransport 测试非流式正常调用与 reasoning_content 提取"""
    async def handler(request: httpx.Request):
        resp_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"action": "BUY_LONG", "confidence": 0.9}',
                        "reasoning_content": "4H突破前高阻力，准备做多",
                    }
                }
            ],
        }
        return httpx.Response(200, json=resp_data)

    client = LLMClient(provider="deepseek", base_url="https://api.deepseek.com", api_key="sk-test", model="deepseek-reasoner")
    client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    content, thinking, latency_ms, actual_m = await client.generate(
        messages=[{"role": "user", "content": "analyze"}]
    )

    assert "BUY_LONG" in content
    assert "4H突破前高阻力" in thinking
    assert actual_m == "deepseek-reasoner"
    assert latency_ms >= 0
    await client.close()


@pytest.mark.asyncio
async def test_llm_client_mock_streaming():
    """使用 MockTransport 测试流式 SSE 逐 chunk 提取"""
    async def stream_handler(request: httpx.Request):
        sse_lines = [
            'data: {"choices":[{"delta":{"reasoning_content":"分析日线趋势..."}}]}\n\n',
            'data: {"choices":[{"delta":{"reasoning_content":"4H均线多头排列"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"{\\"action\\": "}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"\\"BUY_LONG\\"}"}}]}\n\n',
            'data: [DONE]\n\n',
        ]
        return httpx.Response(200, content="".join(sse_lines).encode("utf-8"), headers={"Content-Type": "text/event-stream"})

    client = LLMClient(provider="deepseek", base_url="https://api.deepseek.com", api_key="sk-test", model="deepseek-reasoner")
    client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(stream_handler))

    events = []
    async for event in client.generate_stream(messages=[{"role": "user", "content": "analyze"}]):
        events.append(event)

    event_types = [e["type"] for e in events]
    assert "start" in event_types
    assert "think" in event_types
    assert "content" in event_types
    assert "done" in event_types

    done_event = [e for e in events if e["type"] == "done"][0]
    assert done_event["full_content"] == '{"action": "BUY_LONG"}'
    assert "分析日线趋势" in done_event["full_thinking"]
    await client.close()


@pytest.mark.asyncio
async def test_llm_client_fallback_on_failure():
    """测试主模型多次重试失败后自动触发备用模型降级"""
    attempt_count = 0

    async def fallback_handler(request: httpx.Request):
        nonlocal attempt_count
        attempt_count += 1
        if "deepseek" in str(request.url):
            # 主模型模拟 500 报错
            return httpx.Response(500, text="Internal Server Error")
        elif "openai" in str(request.url):
            # 备用模型返回 200 成功
            resp_data = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action": "HOLD_WAIT", "confidence": 0.5}',
                        }
                    }
                ]
            }
            return httpx.Response(200, json=resp_data)
        return httpx.Response(404)

    mock_transport = httpx.MockTransport(fallback_handler)
    client = LLMClient(
        provider="deepseek",
        base_url="https://api.deepseek.com",
        api_key="sk-main",
        model="deepseek-reasoner",
        max_retries=0,
        fallback_config={
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-backup",
            "model": "gpt-4o",
        },
    )
    client._http_client = httpx.AsyncClient(transport=mock_transport)

    content, thinking, latency_ms, actual_m = await client.generate(
        messages=[{"role": "user", "content": "analyze"}]
    )

    assert "HOLD_WAIT" in content
    assert actual_m == "gpt-4o"
    assert attempt_count >= 2
    await client.close()
