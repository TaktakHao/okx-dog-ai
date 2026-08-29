"""
OKX-Dog 大模型数据精炼师与事后诸葛亮归因复盘器 (LLM Data Refiner & Hindsight Distiller)
模块: okx-dog-ai/dataset/refiner.py
角色: AI 与量化算法工程师 / Prompt 提示词工程师

功能:
1. 冗余消除与密度提炼: 将 6 角色的冗长长篇大论提炼为一段 300~500 字的资深思考链 (<think>)；
2. 多角色能力内化: 将“6 个人外部开会”重构为“资深交易员脑内的多维度博弈自省”；
3. 事后诸葛亮复盘 (Hindsight): 结合盘口真实盈亏走势，强化因果归因（为什么该入场 / 为什么成功避坑）；
4. 批量异步处理并沉淀为高质量 SFT / DPO 训练语料。
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

try:
    from ..llm_client import LLMClient
    from .collector import dataset_collector
except Exception:
    try:
        from okx_dog_ai.llm_client import LLMClient
        from okx_dog_ai.dataset.collector import dataset_collector
    except Exception:
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from llm_client import LLMClient
            from dataset.collector import dataset_collector
        except Exception:
            LLMClient = None
            try:
                from .collector import dataset_collector
            except Exception:
                from dataset.collector import dataset_collector

logger = logging.getLogger("okx_dog.ai.dataset.refiner")

REFINER_SYSTEM_PROMPT = """你是一名资深量化交易大模型数据架构师与知识蒸馏专家。
你的任务是将多智能体系统的“多角色长篇分析日志”与“事后真实盘口盈亏走势”精炼重构为一段高信息密度、逻辑严密、行文干练的【资深交易员内部博弈思考链 (<think>)】。

【提炼重构原则】：
1. **消除冗余与客套**：严禁出现“尊敬的仲裁官”、“多头专家汇报如下”等角色分工样板废话。
2. **内化为单一大脑的自省博弈**：将多头进攻、空头风控挑刺、微观盘口挂单厚度、宏观情绪整合为一个资深交易员在开仓前的综合权衡（例如：虽然突破 EMA25，但注意到上方密集抛压且资金费率偏高，故放弃市价追多）。
3. **事后诸葛亮归因增强 (Hindsight)**：如果提供了事后真实盈亏（例如实际涨了3%或被打止损），在思考链中以极其自然的量化因果逻辑强化“关键有效信号”并归纳“风险警示特征”。
4. **字数控制**：思考链字数控制在 250 ~ 450 字之间，直击要害。

【输出格式要求】：
请只输出提炼后的思考链内容（无需输出额外的 JSON 或包裹标签，直接输出纯文本思考正文）。"""


class LLMDataRefiner:
    """
    大模型数据精炼师
    """

    def __init__(self, llm_client: Optional[Any] = None):
        if llm_client is not None:
            self.llm_client = llm_client
        elif LLMClient is not None:
            self.llm_client = LLMClient()
        else:
            self.llm_client = None

    async def refine_sample(self, sample: Dict[str, Any]) -> str:
        """
        对单个原始样本进行思考链精炼重构
        """
        if not self.llm_client:
            logger.warning("LLMClient 未初始化，跳过大模型精炼")
            return ""
        sample_id = sample.get("sample_id", "")
        features_str = sample.get("market_features_json", "{}")
        senior_analysis_str = sample.get("senior_analysis_json", "{}")
        thinking_steps_str = sample.get("raw_thinking_steps_json", "[]")
        outcome_str = sample.get("actual_outcome_json") or "{}"

        user_content = f"""【原始行情特征】:
{features_str}

【原始多角色分析与决策】:
{senior_analysis_str}

【原始各步骤思维记录】:
{thinking_steps_str}

【事后真实盘口走势与盈亏 (Ground Truth)】:
{outcome_str}

请根据以上材料，提炼出一份高质量、高信息密度的资深交易员博弈思考链："""

        messages = [
            {"role": "system", "content": REFINER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        try:
            refined_cot = await asyncio.wait_for(
                self.llm_client.generate(messages=messages, temperature=0.3),
                timeout=8.0
            )
            refined_cot = refined_cot.strip()
            # 保存到数据库
            if refined_cot:
                dataset_collector.update_refined_cot(sample_id, refined_cot)
                logger.info("样本 %s 思考链提炼成功 (长度: %d 字符)", sample_id, len(refined_cot))
                return refined_cot
        except asyncio.TimeoutError:
            logger.warning("样本 %s 思考链提炼调用超时 (8s)，自动降级跳过", sample_id)
        except Exception as e:
            logger.error("提炼样本 %s 失败: %s", sample_id, e)

        return ""

    async def batch_refine(self, batch_size: int = 20, concurrency: int = 3, total_timeout: float = 15.0) -> int:
        """
        批量并发提炼待处理样本 (带全局总超时保护)
        """
        samples = dataset_collector.get_unrefined_samples(limit=batch_size)
        if not samples:
            logger.info("没有待提炼的样本")
            return 0

        logger.info("开始批量提炼 %d 条样本，并发数=%d，总超时=%.1fs", len(samples), concurrency, total_timeout)
        semaphore = asyncio.Semaphore(concurrency)

        async def _worker(s: Dict[str, Any]):
            async with semaphore:
                try:
                    return await self.refine_sample(s)
                except Exception as w_err:
                    logger.debug("单样本提炼任务异常: %s", w_err)
                    return ""

        try:
            tasks = [_worker(s) for s in samples]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=total_timeout
            )
            success_count = sum(1 for r in results if isinstance(r, str) and len(r) > 0)
            logger.info("批量提炼完成: 成功 %d / %d", success_count, len(samples))
            return success_count
        except asyncio.TimeoutError:
            logger.warning("批量提炼达到全局总超时 (%.1fs)，自动截断并返回已完成项", total_timeout)
            return 0
        except Exception as e:
            logger.error("批量提炼异常: %s", e)
            return 0


llm_data_refiner = LLMDataRefiner()
