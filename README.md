# OKX-Dog AI - 智能量化研判与大模型决策中枢

OKX-Dog AI 是面向加密资产与 TradFi 衍生品交易的决策大脑，负责多周期量化指标计算、动态 Token 高保真压缩、Antigravity CLI / OpenAI 多模型统一驱动与思维链（CoT）流式解析。

## 核心能力与技术架构

1. **高保真动态 Token 压缩中枢 (`prompt_builder.py`)**:
   - 高信息密度 Master System Prompt 结构，剔除冗余套话；
   - 自适应价格与指标数值精度截断 (`_fmt_p`, `_fmt_pct`)，消除浮点 Token 浪费；
   - 紧凑行式上下文序列化，在保留 100% 量化指标前提下将 Token 消耗压降 **45% ~ 60%**；
   - P0~P3 优先级动态预算裁剪与保护机制。
2. **Antigravity CLI 本地极速引擎与 OpenAI 协议适配 (`llm_client.py` / `antigravity_bridge.py`)**:
   - 原生支持 Google Antigravity CLI (`agy`)，免 API Key 驱动本地极速研判；
   - 智能自适应 Effort 协商：自动对齐 Flash 系列 (`low/medium/high`) 与 Pro 系列 (`low/high`) 思考预算，杜绝参数不兼容；
   - 健壮的子进程生命周期管控与非零退出码精准错误透传，彻底杜绝假阳性；
   - 兼容 DeepSeek-R1 (deepseek-reasoner)、DeepSeek-V3、GPT-4o、Claude 等外部网关；
   - 进程级隔离沙盒 (`AntigravityIsolatedEnvManager`)，剔除全局无关 MCP/Skills 干扰。
3. **多周期指标实时计算与微观特征工程 (`indicator_engine.py`)**:
   - 15m, 1h, 4h, 1d 多周期 EMA (20/50/200), MACD, RSI, Bollinger Bands, ATR, 资金费率与全网持仓量 (OI)；
   - 资产类别自适应识别（TradFi 黄金/白银/美股代币/RWA/加密资产）。
4. **结构化决策契约与量化自愈保护 (`schemas.py` / `parser.py`)**:
   - 强类型 JSON Schema 契约校验，确保 100% 格式对齐；
   - 毫秒级思维链 `<think>...</think>` 流式捕获；
   - 模型超时与网络异常时无缝降级至高保真量化规则算法，确保系统 7x24h 永不中断。
5. **多智能体自适应强化奖励与 Softmax 动态门控中枢 (`agent/evolution/`)**:
   - **复合强化奖励分解 (`reward_engine.py`)**：结合实现盈亏比 ($S_{rr}$)、MFE/MAE 走势预测精度 ($S_{acc}$)、排雷避险立功加分 ($S_{risk}$) 与回撤惩罚 ($P_{dd}$)；
   - **Softmax 动态加权门控 (`gating_network.py`)**：带 $[8\%, 35\%]$ 上下限约束的动态话语权分配，避免单一角色垄断；
   - **AI 员工档案与演进中枢 (`evolution_manager.py`)**：管理 6 位拟人化量化员工（冲锋多头、铁血风控、全球情报、链上巨鲸、盘口狙击、量化仲裁）档案、等级晋升与自主避坑口诀库；
   - **Harness 防退化守护与自动熔断**：500+ 极端行情黄金基准校验，连续 3 笔亏损秒级回滚至黄金稳定基线。
6. **模型微调数据沉淀、两级精炼与实习生插槽 (`dataset/` & `agent/evolution/intern_slot.py`)**:
   - **自动化数据收集器 (`dataset/collector.py`)**：沉淀多周期特征快照、6 专家论证与真实盘口盈亏 (Chosen / Rejected)，全面支持 `BUY_LONG`, `SELL_SHORT`, `HOLD_WAIT`, `CLOSE_POSITION` 等全量动作契约；内置 `heal_and_backfill_historical_samples()` 数据自愈机制，自动修复与补全历史样本；
   - **大模型数据精炼师 (`dataset/refiner.py`)**：消除冗余客套、提炼高密度 `<think>` 思考链并注入事后诸葛亮 (Hindsight) 归因复盘；
   - **标准数据集导出器 (`dataset/exporter.py`)**：一键生成 Alpaca SFT 与 DPO 对齐 JSONL，具备数据自愈守护与自适应配对，供 Google Colab / Unsloth 极速微调；
   - **Ollama 实习生专有插槽 (`agent/evolution/intern_slot.py`)**：即插即用接入微调后的开源模型，采用双轨影子推演 (Shadow Mode) 异步模拟实战并核算虚拟战绩，支撑以老带新持续演进。

7. **大模型自驱动反思问诊与角色战法自驱进化 (`agent/evolution/meta_doctor.py`)**:
   - **大模型量化投资总监 (`LLMMetaDoctor`)**：每日盘后或一键触发时，调度 DeepSeek / GPT-4o 顶层大模型自动对 6 位 AI 员工进行失败案例病理剖析；
   - **角色专属避坑硬约束 (`learned_rules`)**：自动为低胜率或失误角色生成精炼的专属战法口诀并落库，后续盘面决策与红蓝博弈（`adversarial_debater.py`）中作为硬约束热注入生效；
   - **版本化自驱迭代**：演进版本号自动递增（`epoch_v1.x`），并提供历史完整反思长文报告查阅与安全回滚门禁。

## 快速启动与依赖

```bash
# 安装依赖
pip install -r requirements.txt
```

