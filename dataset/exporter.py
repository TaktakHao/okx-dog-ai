"""
OKX-Dog 量化模型微调数据集标准导出器 (Dataset Exporter for Colab / Unsloth)
模块: okx-dog-ai/dataset/exporter.py
角色: AI 与量化算法工程师

功能:
1. 从 SQLite 数据池中提取合格的决策样本与盈亏真实数据；
2. 支持导出 Alpaca SFT, ShareGPT 对话, DPO 偏好对齐三种标准 JSONL 格式；
3. 支持一键触发大模型精炼提炼 (--refine-first)；
4. 导出后可直接上传至 Google Drive 或 Google Colab 开启极速训练。
"""

import argparse
import asyncio
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .collector import DEFAULT_DB_PATH, dataset_collector
    from .refiner import llm_data_refiner
except (ImportError, ValueError):
    try:
        from okx_dog_ai.dataset.collector import DEFAULT_DB_PATH, dataset_collector
        from okx_dog_ai.dataset.refiner import llm_data_refiner
    except (ImportError, ValueError):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from dataset.collector import DEFAULT_DB_PATH, dataset_collector
        from dataset.refiner import llm_data_refiner

logger = logging.getLogger("okx_dog.ai.dataset.exporter")

SYSTEM_PROMPT_QUANT_ARBITER = """你是一名精通 OKX 加密货币合约量化交易的资深量化仲裁官与首席交易员。
请根据提供的多周期技术指标、订单簿微观盘口、资金费率、链上巨鲸动向及宏观风险等级，进行全维度的深度自省博弈分析。
你的思考过程必须置于 <think> 与 </think> 标签中（涵盖趋势动能判断、反转风险排查、假突破插针拦截与盈亏比测算）。
最终决策必须输出为严格符合 JSON 契约格式的规范数据结构（包含 action, confidence, order_type, target_leverage, suggested_entry_price, stop_loss_price, take_profit_price 等字段）。"""


class DatasetExporter:
    """
    数据集导出引擎
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(DEFAULT_DB_PATH)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def export_sft_alpaca(self, output_file: str, min_samples: int = 1) -> int:
        """
        导出 Alpaca 格式的 SFT 数据集 (JSONL)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM decision_samples
            WHERE is_qualified_sft = 1
            ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        count = 0
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for row in rows:
                features_str = row["market_features_json"]
                senior_analysis = json.loads(row["senior_analysis_json"] or "{}")
                
                # 优先使用大模型提炼后的高质量思考链，否则使用默认的提炼思路
                cot = row["refined_cot"]
                if not cot:
                    action = senior_analysis.get("action", "HOLD")
                    reason = senior_analysis.get("reasoning", "综合多周期指标与微观盘口深度研判。")
                    cot = f"基于当前多周期量化特征分析，市场结构呈现关键临界点。主要判断：{reason}。严格遵循风控止损要求，制定操作决策。"

                # 构造最终输出 (带 <think> 思考链 + 标准 JSON)
                clean_json_str = json.dumps(senior_analysis, ensure_ascii=False, indent=2)
                full_output = f"<think>\n{cot.strip()}\n</think>\n\n```json\n{clean_json_str}\n```"

                alpaca_item = {
                    "instruction": SYSTEM_PROMPT_QUANT_ARBITER,
                    "input": features_str,
                    "output": full_output
                }
                f.write(json.dumps(alpaca_item, ensure_ascii=False) + "\n")
                count += 1

        logger.info("已成功导出 %d 条 Alpaca SFT 数据至 %s", count, output_file)
        return count

    def export_dpo_pairs(self, output_file: str) -> int:
        """
        导出 DPO (Direct Preference Optimization) 偏好对齐数据集 (JSONL)
        基于真实盘口收益正负反馈配对
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM decision_samples
            WHERE label IN ('CHOSEN', 'REJECTED')
            ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        chosen_list = [dict(r) for r in rows if r["label"] == "CHOSEN"]
        rejected_list = [dict(r) for r in rows if r["label"] == "REJECTED"]

        pairs_count = 0
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for ch in chosen_list:
                # 寻找符号相同或最匹配的 rejected 样本构成对比对
                matching_rej = next((rj for rj in rejected_list if rj["symbol"] == ch["symbol"]), None)
                if not matching_rej and rejected_list:
                    matching_rej = rejected_list[0]

                if matching_rej:
                    ch_json = json.loads(ch["senior_analysis_json"] or "{}")
                    ch_cot = ch["refined_cot"] or f"该策略成功实现盈利与风险收益比对齐，严格回避了追高破位风险。"
                    ch_out = f"<think>\n{ch_cot}\n</think>\n\n```json\n{json.dumps(ch_json, ensure_ascii=False, indent=2)}\n```"

                    rj_json = json.loads(matching_rej["senior_analysis_json"] or "{}")
                    rj_cot = matching_rej["refined_cot"] or f"此处判断出现逆势扛单或忽视了插针破位风险，导致触发止损。"
                    rj_out = f"<think>\n{rj_cot}\n</think>\n\n```json\n{json.dumps(rj_json, ensure_ascii=False, indent=2)}\n```"

                    dpo_item = {
                        "prompt": f"{SYSTEM_PROMPT_QUANT_ARBITER}\n\n行情数据输入:\n{ch['market_features_json']}",
                        "chosen": ch_out,
                        "rejected": rj_out
                    }
                    f.write(json.dumps(dpo_item, ensure_ascii=False) + "\n")
                    pairs_count += 1

        logger.info("已成功导出 %d 组 DPO 对齐偏好数据至 %s", pairs_count, output_file)
        return pairs_count


dataset_exporter = DatasetExporter()


async def main_cli():
    parser = argparse.ArgumentParser(description="OKX-Dog 量化数据集导出工具 (For Google Colab / Unsloth)")
    parser.add_argument("--output-dir", type=str, default="./exports", help="导出数据集存放目录")
    parser.add_argument("--refine-first", action="store_true", help="导出前是否先调用大模型批量精简提炼思考链")
    parser.add_argument("--batch-size", type=int, default=20, help="大模型提炼批次大小")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.refine_first:
        print(f"[*] 启动大模型数据精炼师，正在提炼未精简样本 (Batch Size: {args.batch_size})...")
        refined = await llm_data_refiner.batch_refine(batch_size=args.batch_size)
        print(f"[+] 大模型精炼完成，成功提炼 {refined} 条样本")

    sft_path = str(out_dir / "sft_train_alpaca.jsonl")
    dpo_path = str(out_dir / "dpo_pairs.jsonl")

    sft_cnt = dataset_exporter.export_sft_alpaca(sft_path)
    dpo_cnt = dataset_exporter.export_dpo_pairs(dpo_path)

    print("\n" + "="*60)
    print("🎉 OKX-Dog 训练数据集导出完毕！")
    print(f"📦 1. SFT 监督微调数据集 (Alpaca): {sft_path} (共 {sft_cnt} 条)")
    print(f"⚖️ 2. DPO 偏好对齐数据集:        {dpo_path} (共 {dpo_cnt} 对)")
    print("👉 请直接将导出的 .jsonl 文件上传至 Google Drive 或 Colab 进行极速训练！")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main_cli())
