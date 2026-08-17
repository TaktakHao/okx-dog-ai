# OKX-Dog AI - 智能量化研判与大模型决策中枢

OKX-Dog AI 是面向高频量化与衍生品交易的决策大脑，负责多周期量化指标计算、大模型 Structured Outputs 结构化输出协议与思维链（CoT）流式解析。

## 核心能力

1. **多周期指标实时计算 (`indicators.py`)**:
   - 15m, 1h, 4h, 1d 多周期 EMA, MACD, RSI, Bollinger Bands, ATR, 资金费率与持仓量 (OI) 专项计算
2. **大模型网关与协议契约 (`gateway.py` / `schemas.py`)**:
   - 适配 OpenAI 兼容格式模型（DeepSeek-R1 / V3, GPT-4o 等）
   - JSON Schema 契约校验 (`AI_SCHEMA.json`)，确保 100% 格式对齐
3. **思维链流式提取 (`prompts.py` / `SYSTEM_PROMPT.md`)**:
   - 实时解析 `<think>...</think>` 推理思维链并支持 SSE 流式推流
4. **量化自愈与风控保护**:
   - 模型异常时无缝降级至量化规则自愈决策，确保决策流永不中断

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt
```
