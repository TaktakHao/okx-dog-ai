"""
pytest 全局测试夹具与模块别名注入
模块: okx-dog-ai/tests/conftest.py
"""

import sys
from pathlib import Path

# 将 okx-dog-ai 根目录加入 sys.path
ai_root = Path(__file__).resolve().parent.parent
if str(ai_root) not in sys.path:
    sys.path.insert(0, str(ai_root))

# 建立 okx_dog_ai -> 当前模块映射
import schemas
import indicator_engine
import parser
import prompt_builder
import llm_client
import agent

class ModuleProxy:
    pass

import types
if "okx_dog_ai" not in sys.modules:
    okx_pkg = types.ModuleType("okx_dog_ai")
    okx_pkg.schemas = schemas
    okx_pkg.indicator_engine = indicator_engine
    okx_pkg.parser = parser
    okx_pkg.prompt_builder = prompt_builder
    okx_pkg.llm_client = llm_client
    okx_pkg.agent = agent
    sys.modules["okx_dog_ai"] = okx_pkg
    sys.modules["okx_dog_ai.schemas"] = schemas
    sys.modules["okx_dog_ai.indicator_engine"] = indicator_engine
    sys.modules["okx_dog_ai.parser"] = parser
    sys.modules["okx_dog_ai.prompt_builder"] = prompt_builder
    sys.modules["okx_dog_ai.llm_client"] = llm_client
    sys.modules["okx_dog_ai.agent"] = agent
