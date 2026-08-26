"""
OKX-Dog 量化与 AI 研判决策中枢 - 模型配置模块
模块: okx-dog-ai/config.py
支持 Antigravity CLI 免 Key 首选与 DeepSeek / OpenAI / Claude 多厂商统一参数与容灾管理
"""

from pathlib import Path
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

AI_DIR = Path(__file__).resolve().parent
ROOT_DIR = AI_DIR.parent


class AIModelConfig(BaseSettings):
    """AI 大模型与提示词引擎全局配置 (支持首选驱动与容灾自愈备用)"""

    # 主驱动引擎 (默认首选 Antigravity 本地免 Key 极速驱动)
    PROVIDER: Literal["antigravity", "deepseek", "openai", "anthropic", "custom"] = "antigravity"
    BASE_URL: str = Field(default="http://127.0.0.1:8001", description="兼容 OpenAI 协议的 Base URL")
    API_KEY: str = Field(default="", description="大模型 API Key")
    MODEL_NAME: str = Field(default="gemini-3.7-flash", description="推理模型或对话模型名称")

    # 容灾备用引擎 (当 Antigravity CLI 异常/限流时 0 毫秒无缝接管)
    FALLBACK_PROVIDER: Literal["deepseek", "openai", "custom", "none"] = "deepseek"
    FALLBACK_BASE_URL: str = Field(default="https://api.deepseek.com", description="备用 API Base URL")
    FALLBACK_API_KEY: str = Field(default="", description="备用 DeepSeek / OpenAI API Key")
    FALLBACK_MODEL_NAME: str = Field(default="deepseek-reasoner", description="备用模型名称 (如 deepseek-reasoner / deepseek-chat)")

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

    # 实习生专有插槽 (Intern Slot - 用于开源微调模型通过 Ollama / 本地部署接入进行老带新)
    INTERN_ENABLED: bool = Field(default=False, description="是否启用实习生模型影子推演")
    INTERN_PROVIDER: Literal["ollama", "openai", "custom"] = Field(default="ollama", description="实习生模型服务提供方")
    INTERN_BASE_URL: str = Field(default="http://127.0.0.1:11434/v1", description="实习生模型 Ollama / vLLM 接口地址")
    INTERN_API_KEY: str = Field(default="ollama", description="实习生模型 API Key (Ollama 可随意填写)")
    INTERN_MODEL_NAME: str = Field(default="okx-dog-intern", description="实习生微调模型名称")
    INTERN_TEMPERATURE: float = Field(default=0.2, ge=0.0, le=2.0)
    INTERN_MAX_TOKENS: int = Field(default=2048, ge=256, le=8192)
    INTERN_SHADOW_MODE: bool = Field(default=True, description="是否开启影子推演模式 (只记录虚拟战绩不直接发单)")

    model_config = SettingsConfigDict(
        env_file=[str(ROOT_DIR / ".env"), str(AI_DIR / ".env"), ".env"],
        env_file_encoding="utf-8",
        extra="ignore"
    )


ai_settings = AIModelConfig()
