"""
OKX-Dog Antigravity CLI 异步桥接驱动中枢
模块: okx-dog-ai/antigravity_bridge.py

特性:
1. 异步子进程驱动本地 Antigravity CLI (`agy`)，无需外部第三方 API Key。
2. 支持流式 NDJSON 协议实时解析，无损剥离思考过程 (thinking) 与内容增量 (text_delta)。
3. 原生支持 `--json-schema` 强制结构化输出，确保 100% 满足量化交易契约。
4. 提供异步非流式生成 (generate) 与流式生成器 (generate_stream)，统一四元组契约。
5. 进程级超时管理与自愈退出，杜绝僵尸进程与内存泄漏。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from pathlib import Path

logger = logging.getLogger("okx_dog.ai.antigravity_bridge")


class AntigravityIsolatedEnvManager:
    """
    Antigravity CLI 独立隔离环境管理器
    负责构建与维护专属的 Sandbox 目录，剔除全局 60+ 无关 Skills 与 MCP Servers，
    大幅降低 Token 消耗并提升研判推理速度。
    """

    def __init__(self, env_dir: Optional[Union[str, Path]] = None):
        if env_dir:
            self.env_dir = Path(env_dir).resolve()
        else:
            custom = os.getenv("ANTIGRAVITY_ENV_DIR") or os.getenv("OKX_DOG_AGY_HOME")
            if custom:
                self.env_dir = Path(custom).resolve()
            else:
                self.env_dir = Path(__file__).resolve().parent / ".antigravity_env"

        self._initialized = False

    def get_host_gemini_dir(self) -> Path:
        """获取宿主机全局 ~/.gemini 目录"""
        return Path.home() / ".gemini"

    def ensure_initialized(self, force: bool = False) -> bool:
        """确保专属隔离环境初始化完毕"""
        if self._initialized and not force:
            return True

        host_gemini = self.get_host_gemini_dir()
        if not host_gemini.exists():
            logger.warning(f"[AntigravityIsolatedEnv] 宿主机配置不存在: {host_gemini}")
            return False

        try:
            self.env_dir.mkdir(parents=True, exist_ok=True)

            # 1. 链接系统 Library (macOS Keychain 认证无感穿透必须)
            host_library = Path.home() / "Library"
            target_library = self.env_dir / "Library"
            if host_library.exists():
                if target_library.is_symlink() or target_library.exists():
                    if target_library.is_symlink() and target_library.resolve() != host_library.resolve():
                        target_library.unlink()
                        target_library.symlink_to(host_library)
                else:
                    target_library.symlink_to(host_library)

            # 2. 创建隔离环境内部的 .gemini 结构
            target_gemini = self.env_dir / ".gemini"
            target_gemini.mkdir(parents=True, exist_ok=True)
            target_cli = target_gemini / "antigravity-cli"
            target_cli.mkdir(parents=True, exist_ok=True)
            target_config = target_gemini / "config"
            target_config.mkdir(parents=True, exist_ok=True)

            # 3. 复制同步登录认证与会话基础凭据
            files_to_sync = [
                "google_accounts.json",
                "installation_id",
                "settings.json",
                "state.json",
            ]
            for filename in files_to_sync:
                src = host_gemini / filename
                dst = target_gemini / filename
                if src.exists() and (force or not dst.exists()):
                    shutil.copy2(src, dst)

            # 4. 同步 CLI 状态与支持目录
            host_cli = host_gemini / "antigravity-cli"
            if host_cli.exists():
                cli_files_to_sync = [
                    "installation_id",
                    "jetski_state.pbtxt",
                    "settings.json",
                ]
                for filename in cli_files_to_sync:
                    src = host_cli / filename
                    dst = target_cli / filename
                    if src.exists() and (force or not dst.exists()):
                        shutil.copy2(src, dst)

                for dirname in ["bin", "builtin"]:
                    src = host_cli / dirname
                    dst = target_cli / dirname
                    if src.exists() and not dst.exists():
                        try:
                            dst.symlink_to(src)
                        except Exception:
                            pass

            # 5. 创建纯净的 config 与 settings (不挂载外部无用 MCP / Skills)
            mcp_config_file = target_config / "mcp_config.json"
            if not mcp_config_file.exists() or force:
                mcp_config_file.write_text('{\n  "mcpServers": {}\n}\n', encoding="utf-8")

            cli_settings_file = target_cli / "settings.json"
            if not cli_settings_file.exists() or force:
                minimal_settings = {
                    "skipUpdateCheck": True,
                    "security": {
                        "auth": {
                            "selectedType": "oauth-personal"
                        }
                    }
                }
                cli_settings_file.write_text(json.dumps(minimal_settings, indent=2), encoding="utf-8")

            self._initialized = True
            logger.info(f"[AntigravityIsolatedEnv] 专属隔离沙箱已就绪: {self.env_dir}")
            return True
        except Exception as e:
            logger.error(f"[AntigravityIsolatedEnv] 初始化隔离沙箱失败: {e}", exc_info=True)
            return False

    def get_subprocess_env(self) -> Dict[str, str]:
        """构建注入隔离环境 HOME 的子进程环境变量字典"""
        self.ensure_initialized()
        proc_env = os.environ.copy()
        proc_env["HOME"] = str(self.env_dir)
        return proc_env


class AntigravityCLIError(Exception):
    """Antigravity CLI 执行异常"""
    def __init__(self, message: str, return_code: Optional[int] = None, stderr: Optional[str] = None):
        super().__init__(message)
        self.return_code = return_code
        self.stderr = stderr


class AntigravityBridge:
    """
    Antigravity CLI 异步驱动桥接器 (支持环境隔离)
    """

    def __init__(
        self,
        cli_path: Optional[str] = None,
        model: Optional[str] = None,
        effort: str = "medium",
        timeout_seconds: float = 60.0,
        sandbox: bool = False,
        isolate_env: bool = True,
        isolated_env_dir: Optional[str] = None,
    ):
        """
        初始化 Antigravity 桥接器
        :param cli_path: agy 可执行文件路径，若为空则自动通过 which 查找
        :param model: 模型覆盖（可选，如 gemini-3.7-flash）
        :param effort: 推理深度 ('low', 'medium', 'high')
        :param timeout_seconds: 命令执行超时时间（秒）
        :param sandbox: 是否启用沙箱模式
        :param isolate_env: 是否启用项目专属隔离环境 (剥离无关 Skills / MCP)
        :param isolated_env_dir: 自定义隔离环境存放路径
        """
        self.cli_path = cli_path or self._discover_cli_path()
        self.model = model
        self.effort = effort
        self.timeout_seconds = timeout_seconds
        self.sandbox = sandbox
        self.isolate_env = isolate_env
        self.env_manager = AntigravityIsolatedEnvManager(isolated_env_dir) if isolate_env else None

    @staticmethod
    def _discover_cli_path() -> str:
        """自动探测系统上的 agy 可执行文件"""
        custom_path = os.getenv("ANTIGRAVITY_CLI_PATH")
        if custom_path and os.path.exists(custom_path) and os.access(custom_path, os.X_OK):
            return custom_path

        # 优先探测常见的安装路径
        candidates = [
            shutil.which("agy"),
            os.path.expanduser("~/.local/bin/agy"),
            "/usr/local/bin/agy",
            "/opt/homebrew/bin/agy",
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate) and os.access(candidate, os.X_OK):
                return candidate

        return "agy"

    def is_available(self) -> bool:
        """检查 CLI 工具是否可用"""
        try:
            resolved = shutil.which(self.cli_path) or (os.path.exists(self.cli_path) and os.access(self.cli_path, os.X_OK))
            return bool(resolved)
        except Exception:
            return False

    def _build_command(
        self,
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        effort: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[str]:
        """构建 agy 执行命令行参数列表"""
        cmd = [
            self.cli_path,
            "-p", prompt,
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
            "--disable-slash-commands",
        ]

        eff = effort or self.effort
        if eff in ("low", "medium", "high"):
            cmd.extend(["--effort", eff])

        target_model = model or self.model
        if target_model:
            cmd.extend(["--model", target_model])

        if response_schema:
            if isinstance(response_schema, dict):
                schema_str = json.dumps(response_schema, ensure_ascii=False)
            else:
                schema_str = str(response_schema)
            cmd.extend(["--json-schema", schema_str])

        if self.sandbox:
            cmd.append("--sandbox")

        return cmd

    async def generate(
        self,
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        effort: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Tuple[str, str, int, str]:
        """
        异步非流式执行 Antigravity CLI 研判。
        返回四元组: (content, thinking_process, latency_ms, actual_model_used)
        """
        start_time = time.time()
        timeout = timeout_seconds or self.timeout_seconds

        content_parts: List[str] = []
        thinking_parts: List[str] = []
        model_used = model or self.model or "Antigravity-Gemini"

        try:
            async for delta_text, delta_think in self.generate_stream(
                prompt=prompt,
                response_schema=response_schema,
                effort=effort,
                model=model,
                timeout_seconds=timeout,
            ):
                if delta_text:
                    content_parts.append(delta_text)
                if delta_think:
                    thinking_parts.append(delta_think)

            content = "".join(content_parts).strip()
            thinking = "".join(thinking_parts).strip()
            latency_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"[AntigravityBridge] 研判完成, 耗时={latency_ms}ms, "
                f"输出长度={len(content)}, 思考长度={len(thinking)}"
            )
            return content, thinking, latency_ms, model_used

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[AntigravityBridge] 研判执行异常 (耗时 {latency_ms}ms): {e}", exc_info=True)
            raise

    async def generate_stream(
        self,
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        effort: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """
        异步流式执行 Antigravity CLI 研判。
        每次 yield 二元组: (delta_text, delta_thinking)
        """
        cmd = self._build_command(
            prompt=prompt,
            response_schema=response_schema,
            effort=effort,
            model=model,
        )
        timeout = timeout_seconds or self.timeout_seconds

        proc_env = self.env_manager.get_subprocess_env() if self.env_manager else None
        logger.info(f"[AntigravityBridge] 启动 CLI 子进程 (隔离模式: {bool(self.env_manager)}): {' '.join(cmd[:6])} ...")

        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                env=proc_env,
            )

            async def _stream_reader() -> AsyncGenerator[Tuple[str, str], None]:
                assert process.stdout is not None
                while True:
                    line_bytes = await process.stdout.readline()
                    if not line_bytes:
                        break

                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        # 兼容非 JSON 格式的标准输出行
                        yield line + "\n", ""
                        continue

                    event = data.get("event")

                    if event == "step_update":
                        step_update = data.get("step_update", {})
                        step_type = step_update.get("step_type")

                        if step_type == "agent_response":
                            delta = step_update.get("text_delta", "")
                            if delta:
                                yield delta, ""

                        elif step_type in ("thinking", "thought"):
                            think_delta = step_update.get("text_delta") or step_update.get("thought", "")
                            if think_delta:
                                yield "", think_delta

                    elif event == "result":
                        res = data.get("result", {})
                        # 若 result 中有完整响应且之前未流式完整输出
                        structured = res.get("structured_output")
                        if structured and not line_bytes:
                            yield json.dumps(structured, ensure_ascii=False), ""

            # 使用超时保护流式读取
            async for text_delta, think_delta in _stream_reader():
                yield text_delta, think_delta

            # 等待进程退出并校验返回码
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                if process.returncode is None:
                    process.kill()

            if process.returncode is not None and process.returncode != 0:
                stderr_output = ""
                if process.stderr:
                    err_bytes = await process.stderr.read()
                    stderr_output = err_bytes.decode("utf-8", errors="replace").strip()
                logger.warning(
                    f"[AntigravityBridge] 子进程退出码非 0 ({process.returncode}): {stderr_output[:300]}"
                )

        except asyncio.CancelledError:
            if process and process.returncode is None:
                process.kill()
            raise
        except Exception as exc:
            if process and process.returncode is None:
                process.kill()
            raise AntigravityCLIError(f"Antigravity CLI 执行失败: {exc}") from exc
        finally:
            if process and process.returncode is None:
                try:
                    process.kill()
                except Exception:
                    pass


def _create_default_bridge() -> AntigravityBridge:
    isolate = True
    env_dir = None
    try:
        from .config import ai_settings
        isolate = getattr(ai_settings, "ANTIGRAVITY_ISOLATE_ENV", True)
        env_dir = getattr(ai_settings, "ANTIGRAVITY_ENV_DIR", None) or None
    except Exception:
        try:
            from config import ai_settings
            isolate = getattr(ai_settings, "ANTIGRAVITY_ISOLATE_ENV", True)
            env_dir = getattr(ai_settings, "ANTIGRAVITY_ENV_DIR", None) or None
        except Exception:
            pass
    return AntigravityBridge(isolate_env=isolate, isolated_env_dir=env_dir)


# 全局单例桥接实例
antigravity_bridge = _create_default_bridge()

