#!/usr/bin/env bash
# OKX-Dog 专属隔离环境 Antigravity CLI 包装执行脚本
# 用法: ./okx-dog-ai/scripts/agy_isolated.sh [agy flags/prompt]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_DIR="$(dirname "$SCRIPT_DIR")"
ISOLATED_HOME="${AI_DIR}/.antigravity_env"

# 自动确保隔离环境已初始化
if [ ! -d "$ISOLATED_HOME/.gemini" ]; then
    echo "[*] 首次运行，正在自动初始化隔离环境..."
    python3 "$SCRIPT_DIR/setup_isolated_env.py" setup --dir "$ISOLATED_HOME"
fi

# 注入隔离 HOME 并执行 agy
export HOME="$ISOLATED_HOME"
exec agy "$@"
