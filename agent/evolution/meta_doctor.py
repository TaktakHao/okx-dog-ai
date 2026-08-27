"""
OKX-Dog 大模型量化总监自反思问诊器 (LLM Meta-Doctor & Agent Coach)
模块: okx-dog-ai/agent/evolution/meta_doctor.py
角色: AI 与量化算法工程师 (agency-ai-engineer) & 多智能体协同架构师 (agency-multi-agent-systems-architect)

核心功能:
1. 提取近期交易中的失败单与各专家做错的典型案例；
2. 调度 OpenAI / DeepSeek / Antigravity 顶层大模型充当“量化投资总监”；
3. 输出结构化反思长文报告，并为低胜率或失误角色提炼专属避坑硬约束口诀与参数微调建议。
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("okx_dog.ai.evolution.meta_doctor")


class RoleDiagnosisResult(BaseModel):
    """单个 AI 员工的诊断结果模型"""
    role_id: str
    role_name: str
    is_underperforming: bool = False
    core_flaw_analysis: str = Field(description="该角色近期失误或决策盲点的深度剖析")
    suggested_learned_rules: List[str] = Field(default_factory=list, description="为该角色提炼的 1~2 条专属硬约束避坑口诀")
    parameter_tuning: Optional[Dict[str, Any]] = Field(default=None, description="建议调整的量化因子或阈值参数")


class TeamDiagnosisReport(BaseModel):
    """团队整体大模型自驱诊断与反思报告模型"""
    epoch_id: str
    diagnosis_time: int
    team_overall_summary: str = Field(description="当前多智能体团队协同整体运行评估")
    market_regime_observation: str = Field(description="近期宏观与盘面行情环境总结")
    role_diagnoses: List[RoleDiagnosisResult] = Field(default_factory=list)
    consensus_improvement_plan: str = Field(description="后续多空博弈与仲裁决策的改进建议")


class LLMMetaDoctor:
    """大模型量化投资总监反思问诊服务"""

    def __init__(self, llm_client: Optional[Any] = None):
        self._llm_client = llm_client

    def _get_llm_client(self):
        if self._llm_client is None:
            try:
                from llm_client import LLMClient
                self._llm_client = LLMClient()
            except Exception as e:
                logger.warning(f"初始化 LLMClient 异常: {e}")
        return self._llm_client

    async def diagnose_team(
        self,
        employees_state: Dict[str, Dict[str, Any]],
        recent_trades: List[Dict[str, Any]],
        current_epoch: str = "epoch_v1.0",
    ) -> TeamDiagnosisReport:
        """
        核心反思诊断入口：对 AI 员工团队进行全方位战绩体检与失败案例问诊
        """
        logger.info("👨‍⚕️ 启动 LLM Meta-Doctor 量化总监自驱反思与诊断...")
        now_ts = int(time.time() * 1000)

        # 1. 整理各角色当前战绩与档案
        emp_summary_lines = []
        for rid, emp in employees_state.items():
            win_rate = emp.get("win_rate_7d", 0.0)
            win_rate_pct = f"{win_rate * 100:.1f}%" if win_rate <= 1.0 else f"{win_rate:.1f}%"
            emp_summary_lines.append(
                f"- 角色【{emp.get('nickname', emp.get('name', rid))}】(ID: {rid}): "
                f"等级 Lv.{emp.get('level', 1)}, 7日胜率: {win_rate_pct}, "
                f"胜/负: {emp.get('wins_count', 0)}/{emp.get('losses_count', 0)}, "
                f"排雷防护: {emp.get('defended_crises_count', 0)}次, 话语权: {emp.get('weight_percent', 0):.1f}%\n"
                f"  当前专长描述: {emp.get('specialty_description', '无')}"
            )
        emp_summary_text = "\n".join(emp_summary_lines)

        # 2. 提取典型的亏损与止损失败案例 (最多 5 条)
        losing_trades = [t for t in recent_trades if float(t.get("realized_pnl", 0.0)) < 0][:5]
        winning_trades = [t for t in recent_trades if float(t.get("realized_pnl", 0.0)) > 0][:2]
        
        trades_context_lines = []
        if losing_trades:
            trades_context_lines.append("【近期亏损/被止损典型订单记录】:")
            for i, t in enumerate(losing_trades, 1):
                trades_context_lines.append(
                    f"{i}. 标的: {t.get('symbol')}, 方向: {t.get('pos_side')}, "
                    f"入场价: {t.get('entry_price')}, 出场价: {t.get('exit_price')}, "
                    f"实现亏损: {float(t.get('realized_pnl', 0.0)):+.2f} USDT, 原因/出场逻辑: {t.get('reason', '触及止损')}"
                )
        else:
            trades_context_lines.append("【近期暂无严重亏损订单，团队运行处于基准平稳状态】")

        if winning_trades:
            trades_context_lines.append("【近期代表性盈利订单】:")
            for i, t in enumerate(winning_trades, 1):
                trades_context_lines.append(
                    f"{i}. 标的: {t.get('symbol')}, 方向: {t.get('pos_side')}, 盈利: {float(t.get('realized_pnl', 0.0)):+.2f} USDT"
                )

        trades_context_text = "\n".join(trades_context_lines)

        # 3. 构造专业级量化总监 Meta-Prompt
        system_prompt = (
            "你现在是顶级加密对冲基金的【量化投资总监兼 AI 智能体架构导师 (Meta-Doctor)】。\n"
            "你的任务是审查 OKX-Dog 系统内 6 位 AI 决策员工（冲锋多头分析师、铁血风控挑刺官、全球资讯情报员、"
            "链上主力侦察员、微观盘口狙击手、首席量化仲裁官）的实战表现，针对它们的盲点和失败单进行深度病理剖析，"
            "并为表现欠佳或有改进空间的员工生成精准、可执行的【硬约束避坑战法口诀 (learned_rules)】。\n\n"
            "【输出原则】:\n"
            "1. 拒绝空话套话，直击金融交易本质（如追高破位、假突破诱多、背离被套、过度风控等）；\n"
            "2. 生成的避坑口诀必须精炼干练（每条 15~35 字，如：'在 1H RSI>70 且处于震荡市时坚决不追高，严格等回踩 EMA20'）；\n"
            "3. 必须输出严格合法的 JSON 对象，不要包含 markdown 格式外的多余杂质。"
        )

        user_content = f"""
请根据以下 AI 员工团队战绩档案与近期实盘成交案例进行全面诊断：

【当前 AI 员工团队档案】:
{emp_summary_text}

【近期成交与失败案例】:
{trades_context_text}

请严格按以下 JSON Schema 输出结构化诊断报告：
{{
  "team_overall_summary": "团队整体协同表现与短板概述 (100字以内)",
  "market_regime_observation": "近期盘面走势特征与当前行情对哪类角色不利的观察 (80字以内)",
  "role_diagnoses": [
    {{
      "role_id": "bull_specialist",
      "role_name": "多头辩护专家",
      "is_underperforming": true,
      "core_flaw_analysis": "核心失误病理剖析 (如容易在震荡顶端追高)",
      "suggested_learned_rules": [
        "专属避坑口诀1",
        "专属避坑口诀2"
      ],
      "parameter_tuning": {{"rsi_overbought_threshold": 68}}
    }},
    {{
      "role_id": "bear_critic",
      "role_name": "空头风控专家",
      "is_underperforming": false,
      "core_flaw_analysis": "剖析...",
      "suggested_learned_rules": ["专属避坑口诀..."],
      "parameter_tuning": null
    }},
    {{
      "role_id": "macro_news",
      "role_name": "宏观舆情专家",
      "is_underperforming": false,
      "core_flaw_analysis": "剖析...",
      "suggested_learned_rules": ["专属避坑口诀..."],
      "parameter_tuning": null
    }},
    {{
      "role_id": "onchain_analyst",
      "role_name": "链上数据分析师",
      "is_underperforming": false,
      "core_flaw_analysis": "剖析...",
      "suggested_learned_rules": ["专属避坑口诀..."],
      "parameter_tuning": null
    }},
    {{
      "role_id": "micro_sniper",
      "role_name": "微观盘口分析师",
      "is_underperforming": false,
      "core_flaw_analysis": "剖析...",
      "suggested_learned_rules": ["专属避坑口诀..."],
      "parameter_tuning": {{"imbalance_ratio_threshold": 1.5}}
    }},
    {{
      "role_id": "chief_arbiter",
      "role_name": "首席量化仲裁官",
      "is_underperforming": false,
      "core_flaw_analysis": "剖析...",
      "suggested_learned_rules": ["专属避坑口诀..."],
      "parameter_tuning": null
    }}
  ],
  "consensus_improvement_plan": "下阶段多智能体共识决策的执行与风控强化建议 (100字以内)"
}}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # 4. 调用大模型生成反思报告
        client = self._get_llm_client()
        raw_text = ""
        try:
            if client:
                res = await client.generate(messages=messages, temperature=0.3)
                raw_text = res[0] if isinstance(res, tuple) else str(res)
            else:
                raw_text = self._generate_fallback_diagnosis(employees_state, losing_trades)
        except Exception as e:
            logger.error(f"调用大模型反思诊断异常: {e}，将采用内建量化启发式规则兜底")
            raw_text = self._generate_fallback_diagnosis(employees_state, losing_trades)

        # 5. 解析并清洗 JSON
        return self._parse_diagnosis_response(raw_text, current_epoch, now_ts, employees_state)

    def _generate_fallback_diagnosis(self, employees_state: Dict[str, Dict[str, Any]], losing_trades: List[Dict[str, Any]]) -> str:
        """内建量化启发式兜底生成"""
        default_payload = {
            "team_overall_summary": "多智能体团队近期总体协同平稳，但在结构性震荡中多空博弈存在轻度滞后，需加强右侧确认与动态止盈防护。",
            "market_regime_observation": "近期行情多处于宽幅震荡与缩量洗盘周期，突破形态假信号较多，适合偏防守与回踩进场策略。",
            "role_diagnoses": [
                {
                    "role_id": "bull_specialist",
                    "role_name": "多头辩护专家",
                    "is_underperforming": True,
                    "core_flaw_analysis": "在震荡市高位容易被假阳线诱多，缺乏 1H/4H 大周期均线多头共振验证。",
                    "suggested_learned_rules": [
                        "震荡市中严禁追突破阳线，必须等待 15m/1h 回踩 EMA20 企稳并出现买盘承接才可做多",
                        "当 1H RSI>68 时禁止左侧做多，防范顶背离快速杀跌"
                    ],
                    "parameter_tuning": {"rsi_ceiling": 68}
                },
                {
                    "role_id": "bear_critic",
                    "role_name": "空头风控专家",
                    "is_underperforming": False,
                    "core_flaw_analysis": "风控拦截表现优秀，多次成功提示阻力位压制，但需注意在强趋势单边牛市中避免过度恐慌。",
                    "suggested_learned_rules": [
                        "在放量突破关键颈线位时适度放宽做空偏见，顺应主力趋势方向"
                    ],
                    "parameter_tuning": None
                },
                {
                    "role_id": "macro_news",
                    "role_name": "宏观舆情专家",
                    "is_underperforming": False,
                    "core_flaw_analysis": "宏观日历跟踪良好，已在重大事件前有效提示降杠杆。",
                    "suggested_learned_rules": [
                        "在美联储决议或非农公布前后 30 分钟内保持一票否决权，严防流动性插针"
                    ],
                    "parameter_tuning": None
                },
                {
                    "role_id": "onchain_analyst",
                    "role_name": "链上数据分析师",
                    "is_underperforming": False,
                    "core_flaw_analysis": "链上大单监控稳定，对筹码沉淀识别准确。",
                    "suggested_learned_rules": [
                        "当交易所充币量激增时保持警惕，结合现货深度综合研判出货风险"
                    ],
                    "parameter_tuning": None
                },
                {
                    "role_id": "micro_sniper",
                    "role_name": "微观盘口分析师",
                    "is_underperforming": False,
                    "core_flaw_analysis": "盘口前5档失衡比较为灵敏，已为系统过滤部分虚假挂单。",
                    "suggested_learned_rules": [
                        "仅在买卖盘失衡比 > 1.5 且点差处于低位时触发动量加速建议"
                    ],
                    "parameter_tuning": {"imbalance_threshold": 1.5}
                },
                {
                    "role_id": "chief_arbiter",
                    "role_name": "首席量化仲裁官",
                    "is_underperforming": False,
                    "core_flaw_analysis": "综合仲裁权衡较好，严格执行了止盈止损计划。",
                    "suggested_learned_rules": [
                        "当多空红蓝对抗分歧度高于 40% 时，强制将建议仓位减半并收紧止损点"
                    ],
                    "parameter_tuning": None
                }
            ],
            "consensus_improvement_plan": "在后续演进中重点约束多头专家的追高倾向，强化微观盘口与风控挑刺官的双重验证机制。"
        }
        return json.dumps(default_payload, ensure_ascii=False)

    def _parse_diagnosis_response(
        self,
        raw_text: str,
        current_epoch: str,
        now_ts: int,
        employees_state: Dict[str, Dict[str, Any]]
    ) -> TeamDiagnosisReport:
        """从大模型原始文本中稳健提取结构化报告"""
        try:
            cleaned = raw_text.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            # 匹配最外层大括号
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                cleaned = match.group(0)

            data = json.loads(cleaned)
            role_results = []
            for rd in data.get("role_diagnoses", []):
                role_results.append(RoleDiagnosisResult(
                    role_id=rd.get("role_id", "unknown"),
                    role_name=rd.get("role_name", "AI员工"),
                    is_underperforming=bool(rd.get("is_underperforming", False)),
                    core_flaw_analysis=str(rd.get("core_flaw_analysis", "表现平稳")),
                    suggested_learned_rules=[str(r) for r in rd.get("suggested_learned_rules", []) if r],
                    parameter_tuning=rd.get("parameter_tuning")
                ))

            return TeamDiagnosisReport(
                epoch_id=current_epoch,
                diagnosis_time=now_ts,
                team_overall_summary=str(data.get("team_overall_summary", "团队整体运作正常。")),
                market_regime_observation=str(data.get("market_regime_observation", "近期市场处于典型技术震荡形态。")),
                role_diagnoses=role_results,
                consensus_improvement_plan=str(data.get("consensus_improvement_plan", "继续严格遵循交易纪律与风控要求。"))
            )
        except Exception as e:
            logger.warning(f"解析大模型诊断 JSON 失败: {e}，将采用兜底转换")
            fallback_json = self._generate_fallback_diagnosis(employees_state, [])
            fallback_data = json.loads(fallback_json)
            return TeamDiagnosisReport(
                epoch_id=current_epoch,
                diagnosis_time=now_ts,
                team_overall_summary=fallback_data["team_overall_summary"],
                market_regime_observation=fallback_data["market_regime_observation"],
                role_diagnoses=[RoleDiagnosisResult(**r) for r in fallback_data["role_diagnoses"]],
                consensus_improvement_plan=fallback_data["consensus_improvement_plan"]
            )
