"""
OKX-Dog 量化与 AI 研判决策中枢 - 模型配置模块
模块: okx-dog-ai/config.py
支持 OpenAI 规范多厂商 (DeepSeek / GPT-4o / Qwen / Claude) 统一参数管理
"""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class AIModelConfig(BaseSettings):
    """AI 大模型与提示词引擎全局配置"""

    PROVIDER: Literal["antigravity", "deepseek", "openai", "anthropic", "custom"] = "antigravity"
    BASE_URL: str = Field(default="https://api.deepseek.com", description="兼容 OpenAI 协议的 Base URL")
    API_KEY: str = Field(default="", description="大模型 API Key")
    MODEL_NAME: str = Field(default="gemini-3.7-flash", description="推理模型或对话模型名称")

    # Antigravity CLI 配置
    ANTIGRAVITY_CLI_PATH: str = Field(default="", description="Antigravity CLI (agy) 可执行文件路径")
    ANTIGRAVITY_EFFORT: str = Field(default="medium", description="Antigravity 推理深度 (low/medium/high)")
    ANTIGRAVITY_TIMEOUT: float = Field(default=60.0, description="Antigravity 推理超时时间")
    ANTIGRAVITY_SERVER_URL: str = Field(default="http://127.0.0.1:8001", description="Antigravity 微服务地址")
    ANTIGRAVITY_ISOLATE_ENV: bool = Field(default=True, description="是否启用专属独立隔离环境 (剥离无关 Skills 与 Rules)")
    ANTIGRAVITY_ENV_DIR: str = Field(default="", description="Antigravity 隔离环境目录 (默认 okx-dog-ai/.antigravity_env)")

    TEMPERATURE: float = Field(default=0.2, ge=0.0, le=2.0)
    MAX_TOKENS: int = Field(default=4096, ge=512, le=16384)
    REQUEST_TIMEOUT_SECONDS: float = Field(default=30.0, description="请求超时时间")

    # 是否开启思维链解析
    ENABLE_COT_STREAMING: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


ai_settings = AIModelConfig()
