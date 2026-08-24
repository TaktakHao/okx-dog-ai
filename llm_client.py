"""
OKX-Dog 大模型统一客户端 - 兼容 Antigravity CLI 与 OpenAI 协议的多模型适配器
模块: okx-dog-ai/llm_client.py

特性:
1. 深度原生集成 Google Antigravity CLI (agy)，免 API Key 驱动本地极速研判。
2. 兼容 OpenAI 标准 API 规范，支持 DeepSeek-R1 (deepseek-reasoner), DeepSeek-V3, GPT-4o, Claude 等。
3. 支持异步非流式生成 (generate) 与异步 SSE 流式生成 (generate_stream)。
4. 原生支持 DeepSeek-R1 / Antigravity thinking 思维链增量流式捕获。
5. 30s 响应超时控制与指数退避重试机制 (Exponential Backoff with Jitter)。
6. 备用模型自动降级熔断切换 (Fallback Redundancy)。
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import httpx

try:
    from okx_dog_ai.config import AIModelConfig, ai_settings
    from okx_dog_ai.antigravity_bridge import AntigravityBridge, antigravity_bridge
except (ImportError, ModuleNotFoundError):
    try:
        from .config import AIModelConfig, ai_settings
        from .antigravity_bridge import AntigravityBridge, antigravity_bridge
    except (ImportError, ModuleNotFoundError, ValueError):
        import sys
        from pathlib import Path
        ai_dir = str(Path(__file__).resolve().parent)
        if ai_dir not in sys.path:
            sys.path.insert(0, ai_dir)
        import importlib
        try:
            _ai_cfg = importlib.import_module("okx-dog-ai.config")
        except Exception:
            _ai_cfg = None
        if not _ai_cfg or not hasattr(_ai_cfg, "ai_settings"):
            import config as _local_cfg
            if hasattr(_local_cfg, "ai_settings"):
                _ai_cfg = _local_cfg
            else:
                # 显式从当前目录加载
                import importlib.util
                spec = importlib.util.spec_from_file_location("ai_config_mod", Path(ai_dir) / "config.py")
                _ai_cfg = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_ai_cfg)

        AIModelConfig = getattr(_ai_cfg, "AIModelConfig", None)
        ai_settings = getattr(_ai_cfg, "ai_settings", None)

        try:
            import antigravity_bridge as _agy_mod
            AntigravityBridge = getattr(_agy_mod, "AntigravityBridge", None)
            antigravity_bridge = getattr(_agy_mod, "antigravity_bridge", None)
        except Exception:
            import importlib.util
            spec_b = importlib.util.spec_from_file_location("ai_bridge_mod", Path(ai_dir) / "antigravity_bridge.py")
            _agy_mod = importlib.util.module_from_spec(spec_b)
            spec_b.loader.exec_module(_agy_mod)
            AntigravityBridge = getattr(_agy_mod, "AntigravityBridge", None)
            antigravity_bridge = getattr(_agy_mod, "antigravity_bridge", None)

logger = logging.getLogger("okx_dog.ai.llm_client")


class LLMInferenceError(Exception):
    """LLM 推理调用异常基类"""
    def __init__(self, message: str, status_code: Optional[int] = None, raw_response: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.raw_response = raw_response


class LLMClient:
    """
    Antigravity CLI / OpenAI 协议兼容的高可用多大模型适配器客户端
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        fallback_config: Optional[Dict[str, Any]] = None,
    ):
        self.provider = (provider or getattr(ai_settings, "PROVIDER", "antigravity")).lower()
        self.base_url = (base_url or ai_settings.BASE_URL).rstrip("/")
        self.api_key = api_key or ai_settings.API_KEY
        self.model = model or ai_settings.MODEL_NAME
        self.temperature = temperature if temperature is not None else ai_settings.TEMPERATURE
        self.max_tokens = max_tokens or ai_settings.MAX_TOKENS
        self.timeout_seconds = timeout_seconds or ai_settings.REQUEST_TIMEOUT_SECONDS
        self.max_retries = max_retries
        self.fallback_config = fallback_config

        # 自动装配默认备用容灾引擎 (如 DeepSeek-v4-pro / DeepSeek-R1)
        if not self.fallback_config and getattr(ai_settings, "FALLBACK_PROVIDER", "none") != "none":
            self.fallback_config = {
                "provider": getattr(ai_settings, "FALLBACK_PROVIDER", "deepseek"),
                "base_url": getattr(ai_settings, "FALLBACK_BASE_URL", "https://api.deepseek.com"),
                "api_key": getattr(ai_settings, "FALLBACK_API_KEY", "") or self.api_key,
                "model": getattr(ai_settings, "FALLBACK_MODEL_NAME", "deepseek-reasoner"),
            }

        # 如果没有配置 API_KEY 且默认是 deepseek/openai，自动将 provider 切换为 antigravity
        if not self.api_key and self.provider in ("deepseek", "openai"):
            if antigravity_bridge.is_available():
                logger.info("未检测到 API Key，自动启用 Antigravity CLI 作为首选驱动引擎")
                self.provider = "antigravity"

        # 初始化 HTTP 客户端连接池 (供外部 OpenAI 兼容网关请求使用)
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )

    async def close(self) -> None:
        """关闭底层连接池"""
        if not self._http_client.is_closed:
            await self._http_client.aclose()

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """将 messages 列表格式化为统一的 prompt 字符串"""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"【系统量化分析角色与准则】\n{content}\n")
            elif role == "user":
                parts.append(f"【行情数据与研判任务】\n{content}\n")
            else:
                parts.append(f"【历史研判参考】\n{content}\n")
        return "\n".join(parts).strip()

    # =========================================================================
    # 1. 异步非流式生成 (generate)
    # =========================================================================

    async def generate(
        self,
        messages: List[Dict[str, str]],
        response_schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Tuple[str, str, int, str]:
        """
        异步非流式生成完整响应。
        返回四元组: (content, thinking_process, latency_ms, actual_model_used)
        """
        target_model = model or self.model
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens

        # ---------------------------------------------------------------------
        # 分支 A: Antigravity CLI 本地直驱模式 (免 API Key)
        # ---------------------------------------------------------------------
        if self.provider == "antigravity":
            prompt = self._messages_to_prompt(messages)
            try:
                content, thinking, latency_ms, actual_model = await antigravity_bridge.generate(
                    prompt=prompt,
                    response_schema=response_schema,
                    effort=getattr(ai_settings, "ANTIGRAVITY_EFFORT", "medium"),
                    model=target_model if target_model != "deepseek-reasoner" else None,
                    timeout_seconds=self.timeout_seconds or 60.0,
                )
                return content, thinking, latency_ms, actual_model
            except Exception as agy_err:
                logger.warning(f"Antigravity CLI 执行失败: {agy_err}，尝试检查备用通道...")
                if self.fallback_config:
                    return await self._execute_fallback(messages, response_schema, temp, tokens)
                raise LLMInferenceError(f"Antigravity CLI 研判失败: {agy_err}") from agy_err

        # ---------------------------------------------------------------------
        # 分支 B: OpenAI 兼容 HTTP 网关模式 (DeepSeek / OpenAI / Custom)
        # ---------------------------------------------------------------------
        start_time = time.time()
        attempt = 0

        while attempt <= self.max_retries:
            attempt += 1
            try:
                payload = self._adapt_request_payload(
                    model=target_model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                    response_schema=response_schema,
                    stream=False,
                )
                headers = self._build_headers()

                url = f"{self.base_url}/chat/completions"
                logger.info("发起 LLM 非流式请求 [尝试 %d/%d]: URL=%s, Model=%s", attempt, self.max_retries + 1, url, target_model)

                response = await self._http_client.post(url, json=payload, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    choice = data["choices"][0]
                    message_obj = choice.get("message", {})

                    content = message_obj.get("content", "") or ""
                    # 提取 reasoning_content (DeepSeek-R1)
                    thinking = message_obj.get("reasoning_content", "") or ""

                    latency_ms = int((time.time() - start_time) * 1000)
                    logger.info("LLM 请求成功: Model=%s, 耗时=%dms", target_model, latency_ms)
                    return content, thinking, latency_ms, target_model

                # 遇到 429 频控或 5xx 服务端错误时触发重试
                if response.status_code in [429, 500, 502, 503, 504]:
                    raw_text = response.text
                    logger.warning("LLM 请求返回服务端错误 (HTTP %d): %s", response.status_code, raw_text[:200])
                    if attempt <= self.max_retries:
                        backoff = (2 ** (attempt - 1)) * 1.0 + random.uniform(0.1, 0.5)
                        logger.info("触发指数退避等待 %.2f 秒后重试...", backoff)
                        await asyncio.sleep(backoff)
                        continue
                    raise LLMInferenceError(f"HTTP {response.status_code}: {raw_text}", status_code=response.status_code, raw_response=raw_text)

                # 其他客户端错误 (如 400, 401, 403) 不可重试，直接抛出
                raw_text = response.text
                raise LLMInferenceError(f"LLM 请求失败 (HTTP {response.status_code}): {raw_text}", status_code=response.status_code, raw_response=raw_text)

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                logger.warning("LLM 请求网络/超时异常 [尝试 %d]: %s", attempt, str(exc))
                if attempt <= self.max_retries:
                    backoff = (2 ** (attempt - 1)) * 1.0 + random.uniform(0.1, 0.5)
                    await asyncio.sleep(backoff)
                    continue
                break
            except LLMInferenceError:
                if attempt > self.max_retries:
                    break

        # 主模型调用全部失败，尝试备用模型降级
        if self.fallback_config:
            logger.warning("主模型 %s 多次重试失败，触发备用模型降级: %s", target_model, self.fallback_config.get("model"))
            return await self._execute_fallback(messages, response_schema, temperature, max_tokens)

        latency_ms = int((time.time() - start_time) * 1000)
        raise LLMInferenceError(f"LLM 请求在重试 {self.max_retries} 次后最终失败 (耗时 {latency_ms}ms)")

    # =========================================================================
    # 2. 异步 SSE 流式生成 (generate_stream)
    # =========================================================================

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        response_schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        异步流式生成。
        产出统一事件字典:
        - {"type": "start", "model": ...}
        - {"type": "think", "delta": ...} (思维链增量)
        - {"type": "content", "delta": ...} (JSON 内容增量)
        - {"type": "done", "full_content": ..., "full_thinking": ..., "latency_ms": ..., "model": ...}
        - {"type": "error", "error": ...}
        """
        target_model = model or self.model
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens
        start_time = time.time()

        yield {"type": "start", "model": target_model, "timestamp": int(start_time * 1000)}

        # ---------------------------------------------------------------------
        # 分支 A: Antigravity CLI 本地流式直驱模式
        # ---------------------------------------------------------------------
        if self.provider == "antigravity":
            prompt = self._messages_to_prompt(messages)
            full_content_chunks: List[str] = []
            full_thinking_chunks: List[str] = []

            try:
                async for text_delta, think_delta in antigravity_bridge.generate_stream(
                    prompt=prompt,
                    response_schema=response_schema,
                    effort=getattr(ai_settings, "ANTIGRAVITY_EFFORT", "medium"),
                    model=target_model if target_model != "deepseek-reasoner" else None,
                    timeout_seconds=self.timeout_seconds or 60.0,
                ):
                    if think_delta:
                        full_thinking_chunks.append(think_delta)
                        yield {"type": "think", "delta": think_delta}
                    if text_delta:
                        full_content_chunks.append(text_delta)
                        yield {"type": "content", "delta": text_delta}

                latency_ms = int((time.time() - start_time) * 1000)
                yield {
                    "type": "done",
                    "full_content": "".join(full_content_chunks),
                    "full_thinking": "".join(full_thinking_chunks),
                    "latency_ms": latency_ms,
                    "model": target_model,
                }
                return

            except Exception as agy_exc:
                logger.error("Antigravity CLI 流式执行异常: %s", str(agy_exc))
                if self.fallback_config:
                    try:
                        logger.warning("Antigravity CLI 异常，触发备用模型非流式补偿...")
                        content, thinking, latency_ms, actual_m = await self._execute_fallback(messages, response_schema, temp, tokens)
                        if thinking:
                            yield {"type": "think", "delta": thinking}
                        yield {"type": "content", "delta": content}
                        yield {
                            "type": "done",
                            "full_content": content,
                            "full_thinking": thinking,
                            "latency_ms": latency_ms,
                            "model": actual_m,
                        }
                        return
                    except Exception as fb_exc:
                        logger.error("备用模型也发生异常: %s", str(fb_exc))

                yield {"type": "error", "error": str(agy_exc)}
                return

        # ---------------------------------------------------------------------
        # 分支 B: OpenAI 兼容 HTTP 网关流式
        # ---------------------------------------------------------------------
        payload = self._adapt_request_payload(
            model=target_model,
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
            response_schema=response_schema,
            stream=True,
        )
        headers = self._build_headers()
        url = f"{self.base_url}/chat/completions"

        full_content_chunks: List[str] = []
        full_thinking_chunks: List[str] = []

        try:
            async with self._http_client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error("LLM 流式请求失败 (HTTP %d): %s", response.status_code, error_text.decode(errors="ignore"))
                    if self.fallback_config:
                        logger.warning("流式请求失败，尝试降级到备用模型...")
                        content, thinking, latency_ms, actual_m = await self._execute_fallback(messages, response_schema, temp, tokens)
                        if thinking:
                            yield {"type": "think", "delta": thinking}
                        yield {"type": "content", "delta": content}
                        yield {
                            "type": "done",
                            "full_content": content,
                            "full_thinking": thinking,
                            "latency_ms": latency_ms,
                            "model": actual_m,
                        }
                        return
                    else:
                        yield {"type": "error", "error": f"HTTP {response.status_code}: {error_text.decode(errors='ignore')}"}
                        return

                # 逐行读取 SSE 响应
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            choices = chunk_data.get("choices", [])
                            if not choices:
                                continue

                            delta = choices[0].get("delta", {})

                            # 1. 捕获 DeepSeek-R1 reasoning_content
                            reasoning_delta = delta.get("reasoning_content")
                            if reasoning_delta:
                                full_thinking_chunks.append(reasoning_delta)
                                yield {"type": "think", "delta": reasoning_delta}

                            # 2. 捕获常规 content
                            content_delta = delta.get("content")
                            if content_delta:
                                full_content_chunks.append(content_delta)
                                yield {"type": "content", "delta": content_delta}

                        except json.JSONDecodeError:
                            continue

            latency_ms = int((time.time() - start_time) * 1000)
            full_content = "".join(full_content_chunks)
            full_thinking = "".join(full_thinking_chunks)

            yield {
                "type": "done",
                "full_content": full_content,
                "full_thinking": full_thinking,
                "latency_ms": latency_ms,
                "model": target_model,
            }

        except Exception as exc:
            logger.error("LLM 流式传输异常: %s", str(exc))
            if self.fallback_config:
                try:
                    logger.warning("流式异常，触发备用模型非流式补偿...")
                    content, thinking, latency_ms, actual_m = await self._execute_fallback(messages, response_schema, temp, tokens)
                    if thinking:
                        yield {"type": "think", "delta": thinking}
                    yield {"type": "content", "delta": content}
                    yield {
                        "type": "done",
                        "full_content": content,
                        "full_thinking": thinking,
                        "latency_ms": latency_ms,
                        "model": actual_m,
                    }
                    return
                except Exception as fb_exc:
                    logger.error("备用模型也发生异常: %s", str(fb_exc))

            yield {"type": "error", "error": str(exc)}

    # =========================================================================
    # 3. 备用模型降级执行
    # =========================================================================

    async def _execute_fallback(
        self,
        messages: List[Dict[str, str]],
        response_schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, str, int, str]:
        """使用备用配置执行单次请求"""
        fb = self.fallback_config or {}
        fb_base_url = fb.get("base_url", "https://api.openai.com/v1").rstrip("/")
        fb_api_key = fb.get("api_key", self.api_key)
        fb_model = fb.get("model", "gpt-4o")

        payload = self._adapt_request_payload(
            model=fb_model,
            messages=messages,
            temperature=temperature if temperature is not None else 0.2,
            max_tokens=max_tokens or 4096,
            response_schema=response_schema,
            stream=False,
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {fb_api_key}",
        }

        start_t = time.time()
        url = f"{fb_base_url}/chat/completions"
        logger.info("执行备用模型请求: URL=%s, Model=%s", url, fb_model)

        resp = await self._http_client.post(url, json=payload, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            choice = data["choices"][0]
            msg = choice.get("message", {})
            content = msg.get("content", "") or ""
            thinking = msg.get("reasoning_content", "") or ""
            latency = int((time.time() - start_t) * 1000)
            return content, thinking, latency, fb_model

        raise LLMInferenceError(f"备用模型请求失败 (HTTP {resp.status_code}): {resp.text}", status_code=resp.status_code)

    # =========================================================================
    # 4. 私有辅助方法与请求参数适配
    # =========================================================================

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _adapt_request_payload(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """抹平不同供应商/模型参数差异"""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        is_o_series = model.startswith("o1") or model.startswith("o3")
        is_deepseek_reasoner = "reasoner" in model or "r1" in model.lower()

        if is_o_series:
            payload["max_completion_tokens"] = max_tokens
            payload["reasoning_effort"] = "medium"
        elif is_deepseek_reasoner:
            payload["temperature"] = temperature if temperature is not None else 0.6
            payload["max_tokens"] = max_tokens
        else:
            payload["temperature"] = temperature
            payload["max_tokens"] = max_tokens

        if response_schema and not is_deepseek_reasoner:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "OKXDogAIAnalysisResponse",
                    "strict": True,
                    "schema": response_schema,
                },
            }

        return payload
