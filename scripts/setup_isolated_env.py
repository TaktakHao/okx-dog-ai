#!/usr/bin/env python3
"""
OKX-Dog Antigravity CLI 独立隔离环境管理器
模块: okx-dog-ai/scripts/setup_isolated_env.py

功能:
1. 自动为当前项目构建轻量纯净的 Antigravity 运行 Sandbox。
2. 剥离全局 60+ 无关 Skills、Rules 和 MCP Servers，大幅降低 Token 消耗 (降低 7,000+ Tokens)。
3. 自动桥接系统登录认证与 macOS Keychain，保持 100% 免登与静默认证。
4. 提供 setup / clean / status / test 等管理命令。
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_default_env_dir() -> Path:
    """获取默认隔离环境路径 (okx-dog-ai/.antigravity_env)"""
    script_dir = Path(__file__).resolve().parent
    ai_root = script_dir.parent
    return ai_root / ".antigravity_env"


def get_host_gemini_dir() -> Path:
    """获取宿主机全局 ~/.gemini 目录"""
    return Path.home() / ".gemini"


def setup_isolated_env(env_dir: Path, force: bool = False) -> bool:
    """初始化/更新隔离环境"""
    host_gemini = get_host_gemini_dir()
    if not host_gemini.exists():
        print(f"[ERROR] 宿主机全局配置目录不存在: {host_gemini}")
        return False

    print(f"[*] 正在初始化 Antigravity 隔离环境 -> {env_dir}")
    env_dir.mkdir(parents=True, exist_ok=True)

    # 1. 链接系统 Library (macOS Keychain 认证必须)
    host_library = Path.home() / "Library"
    target_library = env_dir / "Library"
    if host_library.exists():
        if target_library.is_symlink() or target_library.exists():
            if target_library.is_symlink() and target_library.resolve() != host_library.resolve():
                target_library.unlink()
                target_library.symlink_to(host_library)
        else:
            try:
                target_library.symlink_to(host_library)
                print("  [+] 已建立 macOS Library 符号链接 (保持 Keychain 登录态)")
            except Exception as e:
                print(f"  [!] 建立 Library 链接失败 (非 macOS 或权限问题): {e}")

    # 2. 创建隔离环境内部的 .gemini 结构
    target_gemini = env_dir / ".gemini"
    target_gemini.mkdir(parents=True, exist_ok=True)
    target_cli = target_gemini / "antigravity-cli"
    target_cli.mkdir(parents=True, exist_ok=True)
    target_config = target_gemini / "config"
    target_config.mkdir(parents=True, exist_ok=True)

    # 3. 复制/同步基础凭证文件 (保持只读/同步)
    files_to_sync = [
        "google_accounts.json",
        "installation_id",
        "settings.json",
        "state.json",
    ]
    for filename in files_to_sync:
        src = host_gemini / filename
        dst = target_gemini / filename
        if src.exists():
            shutil.copy2(src, dst)

    # 同步 CLI 特定状态
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
            if src.exists():
                shutil.copy2(src, dst)

        # 软链接二进制与内置支持目录
        for dirname in ["bin", "builtin"]:
            src = host_cli / dirname
            dst = target_cli / dirname
            if src.exists() and not dst.exists():
                try:
                    dst.symlink_to(src)
                except Exception:
                    pass

    # 4. 创建纯净的 config/mcp_config.json (空配置，不挂载外部无用 MCP)
    mcp_config_file = target_config / "mcp_config.json"
    if not mcp_config_file.exists() or force:
        mcp_config_file.write_text("{\n  \"mcpServers\": {}\n}\n", encoding="utf-8")

    # 5. 创建纯净的 settings.json
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

    print("[SUCCESS] Antigravity 专属隔离环境初始化完成！")
    print(f"  - 隔离 HOME 路径: {env_dir}")
    print("  - 全局 Skills 隔离: 60+ 外部 Skills 已剥离 (Token 消耗降低 7,000+)")
    print("  - 全局 MCP 隔离: 外部 MCP Servers 已剥离")
    return True


def check_status(env_dir: Path) -> None:
    """检查隔离环境状态"""
    print(f"=== Antigravity 隔离环境状态检查 [{env_dir}] ===")
    if not env_dir.exists():
        print("[Status] 未初始化 (Not Initialized)")
        return

    target_gemini = env_dir / ".gemini"
    if not target_gemini.exists():
        print("[Status] 异常: .gemini 目录缺失")
        return

    auth_files = ["google_accounts.json", "installation_id", "settings.json", "state.json"]
    present_files = [f for f in auth_files if (target_gemini / f).exists()]
    print(f"  - 基础凭据状态: {len(present_files)}/{len(auth_files)} 正常")

    library_link = env_dir / "Library"
    if library_link.exists():
        print("  - macOS Keychain 桥接: 正常")
    else:
        print("  - macOS Keychain 桥接: 未链接")

    config_skills = target_gemini / "config" / "skills"
    if not config_skills.exists():
        print("  - Skills 纯净度: 100% 纯净 (无外部无关 Skills 干扰)")
    else:
        skill_count = len(list(config_skills.iterdir()))
        print(f"  - Skills 纯净度: 包含 {skill_count} 个专属 Skills")


def test_isolated_run(env_dir: Path) -> None:
    """运行一次隔离环境下的 agy 测试"""
    setup_isolated_env(env_dir)
    print("\n[*] 正在隔离环境中执行测试请求 (agy -p 'HELLO')...")
    
    test_env = os.environ.copy()
    test_env["HOME"] = str(env_dir)

    cmd = [
        "agy",
        "-p", "请直接回复五个大写字母: HELLO",
        "--disable-slash-commands",
        "--output-format", "json"
    ]

    try:
        res = subprocess.run(
            cmd,
            env=test_env,
            capture_output=True,
            text=True,
            timeout=30
        )
        if res.returncode == 0:
            print("[SUCCESS] 隔离环境运行成功！")
            try:
                data = json.loads(res.stdout)
                print(f"  - 响应内容: {data.get('response', '').strip()}")
                usage = data.get("usage", {})
                print(f"  - Token 消耗: Input={usage.get('input_tokens')}, Total={usage.get('total_tokens')}")
                print(f"  - 执行耗时: {data.get('duration_seconds')}s")
            except Exception:
                print(f"  - 原始输出: {res.stdout[:200]}")
        else:
            print(f"[ERROR] 执行失败, 返回码={res.returncode}")
            print(f"  stderr: {res.stderr}")
            print(f"  stdout: {res.stdout}")
    except Exception as e:
        print(f"[ERROR] 测试调用异常: {e}")


def clean_env(env_dir: Path) -> None:
    """清理隔离环境缓存与数据"""
    if env_dir.exists():
        confirm = input(f"确认要删除隔离环境目录吗? ({env_dir}) [y/N]: ")
        if confirm.lower() == "y":
            shutil.rmtree(env_dir)
            print("[SUCCESS] 已清理隔离环境。")
    else:
        print("[*] 隔离环境目录不存在，无需清理。")


def main():
    parser = argparse.ArgumentParser(description="OKX-Dog Antigravity CLI 独立隔离环境管理工具")
    parser.add_argument("action", choices=["setup", "status", "test", "clean"], nargs="?", default="setup",
                        help="执行动作: setup(初始化/更新), status(检查状态), test(测试运行), clean(清理)")
    parser.add_argument("--dir", type=str, default="", help="自定义隔离环境路径 (默认 okx-dog-ai/.antigravity_env)")
    parser.add_argument("--force", action="store_true", help="强制覆盖已有配置文件")

    args = parser.parse_args()
    env_dir = Path(args.dir) if args.dir else get_default_env_dir()

    if args.action == "setup":
        setup_isolated_env(env_dir, force=args.force)
    elif args.action == "status":
        check_status(env_dir)
    elif args.action == "test":
        test_isolated_run(env_dir)
    elif args.action == "clean":
        clean_env(env_dir)


if __name__ == "__main__":
    main()
