"""
OKX-Dog AI Quant Studio - 标准量化策略与因子抽象基类契约
模块: okx-dog-ai/studio/strategy_base.py
角色: AI 与量化算法工程师 (agency-ai-engineer)
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseQuantStrategy(ABC):
    """
    OKX-Dog 标准量化策略基类
    所有由 Codex 或交易员编写的 Python 策略代码必须继承此类并实现 generate_signals 方法。
    """

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        根据历史市场切片 DataFrame 生成交易信号序列

        参数:
            df (pd.DataFrame): 包含以下标准列的 K 线与衍生品时序数据:
                - timestamp (int): 毫秒时间戳
                - open (float): 开盘价
                - high (float): 最高价
                - low (float): 最低价
                - close (float): 收盘价
                - volume (float): 成交量 (张/币)
                - funding_rate (float, optional): 资金费率 (小数形式，如 0.0001)
                - oi (float, optional): 未平仓合约量 (Open Interest)

        返回:
            pd.Series: 与 df 等长的整型/浮点型信号序列，取值规范:
                 1 : 开多 (BUY_LONG)
                -1 : 开空 (SELL_SHORT)
                 0 : 观望 / 平仓 (FLAT)
        """
        pass
