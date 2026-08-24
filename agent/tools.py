"""
OKX-Dog 量化交易员专用量化与风控计算工具箱 (专业机构级工具库)
模块: okx-dog-ai/agent/tools.py
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("okx_dog.ai.agent.tools")


def calculate_risk_reward_ratio(
    action: str,
    entry_range: List[float],
    stop_loss_price: float,
    take_profit_levels: List[Dict[str, Any]],
    taker_fee_pct: float = 0.0005,  # 单边 Taker 0.05% (双边 0.10%)
    estimated_slippage_pct: float = 0.0005,  # 预估滑点 0.05%
) -> Tuple[float, Optional[str]]:
    """
    计算并校验扣除真实交易摩擦 (双边手续费 + 预期滑点) 后的【净盈亏比 (Net R:R Ratio)】与方向逻辑。
    返回: (net_rr_ratio, error_message)
    """
    if not entry_range or len(entry_range) != 2:
        return 0.0, "入场区间格式非法，必须为 [最低价, 最高价]"

    if not take_profit_levels:
        return 0.0, "未配置止盈目标点位"

    entry_avg = (entry_range[0] + entry_range[1]) / 2.0
    tp1_price = float(take_profit_levels[0].get("price", 0.0))

    if entry_avg <= 0 or stop_loss_price <= 0 or tp1_price <= 0:
        return 0.0, "价格参数必须大于 0"

    raw_risk_dist = abs(entry_avg - stop_loss_price)
    if raw_risk_dist <= 1e-6:
        return 0.0, "止损价与入场均价距离过近或相同，无法构成有效风控区间"

    raw_reward_dist = abs(tp1_price - entry_avg)

    # 计入交易摩擦成本 (双边手续费 + 滑点)
    total_friction_pct = (taker_fee_pct * 2.0) + estimated_slippage_pct
    friction_cost = entry_avg * total_friction_pct

    # 净盈利距离 = 名义盈利 - 摩擦成本；净风险距离 = 名义风险 + 摩擦成本
    net_reward_dist = max(0.0, raw_reward_dist - friction_cost)
    net_risk_dist = raw_risk_dist + friction_cost

    net_rr_ratio = round(net_reward_dist / net_risk_dist, 2)

    # 方向逻辑严密性检验
    if action == "BUY_LONG":
        if stop_loss_price >= entry_avg:
            return net_rr_ratio, f"做多操作中，止损价 ({stop_loss_price}) 必须低于入场均价 ({entry_avg:.2f})"
        if tp1_price <= entry_avg:
            return net_rr_ratio, f"做多操作中，第一止盈价 ({tp1_price}) 必须高于入场均价 ({entry_avg:.2f})"
    elif action == "SELL_SHORT":
        if stop_loss_price <= entry_avg:
            return net_rr_ratio, f"做空操作中，止损价 ({stop_loss_price}) 必须高于入场均价 ({entry_avg:.2f})"
        if tp1_price >= entry_avg:
            return net_rr_ratio, f"做空操作中，第一止盈价 ({tp1_price}) 必须低于入场均价 ({entry_avg:.2f})"

    return net_rr_ratio, None


def check_volatility_squeeze(
    bb_upper: float,
    bb_lower: float,
    bb_middle: float,
    atr_14: float,
    current_price: float,
) -> Tuple[bool, float, str]:
    """
    波动率挤压 (Volatility Squeeze) 过滤器
    检测布林带带宽是否严重收窄（处于变盘临界或假突破高发区）。
    返回: (is_squeezed, bandwidth_pct, diagnostic_msg)
    """
    if bb_middle <= 0 or current_price <= 0:
        return False, 0.0, "参数异常"

    bandwidth = (bb_upper - bb_lower) / bb_middle
    atr_ratio = atr_14 / current_price

    # 带宽 < 1.5% 或 ATR 占比 < 0.35% 判定为极度挤压震荡
    if bandwidth < 0.015 or atr_ratio < 0.0035:
        msg = f"检测到市场处于极度波动率挤压期 (布林带宽={bandwidth*100:.2f}%, ATR比率={atr_ratio*100:.2f}%)，假突破风险极高，建议保持观望。"
        return True, bandwidth, msg

    return False, bandwidth, "波动率正常"


def verify_hard_risk_compliance(
    action: str,
    entry_price: float,
    stop_loss_price: float,
    suggested_leverage: int,
    account_balance_usdt: float,
    risk_limits: Dict[str, Any],
) -> List[str]:
    """
    针对交易员输出的草案进行硬风控物理边界拦截审查。
    返回: 违规项列表 (若为空列表则表示 100% 审查通过)
    """
    violations: List[str] = []

    if action in ["HOLD_WAIT", "CLOSE_POSITION"]:
        return violations

    max_leverage = int(risk_limits.get("max_leverage", 5))
    max_order_usdt = float(risk_limits.get("max_order_usdt", 500.0))
    max_daily_loss = float(risk_limits.get("max_daily_loss_usdt", 200.0))

    # 1. 杠杆审查
    if suggested_leverage > max_leverage:
        violations.append(
            f"建议杠杆 ({suggested_leverage}x) 超出系统允许的最大杠杆上限 ({max_leverage}x)"
        )

    # 2. 止损幅度与单笔名义风险评估
    if entry_price > 0 and stop_loss_price > 0:
        stop_loss_dist_pct = abs(entry_price - stop_loss_price) / entry_price
        if stop_loss_dist_pct > 0.15:
            violations.append(
                f"单笔止损幅度过宽 ({stop_loss_dist_pct * 100:.2f}%)，严重偏离量化风控合理范围 (建议不超过 5%~10%)"
            )

        # 估算最大委托名义价值下的亏损敞口
        estimated_max_loss = max_order_usdt * stop_loss_dist_pct * (suggested_leverage if suggested_leverage <= 3 else 1.0)
        if estimated_max_loss > max_daily_loss:
            violations.append(
                f"预估单笔可能最大亏损 ({estimated_max_loss:.2f} USDT) 逼近或超过单日最大亏损熔断阈值 ({max_daily_loss:.2f} USDT)"
            )

    return violations


def derive_dynamic_atr_stops(
    current_price: float,
    atr_14: float,
    action: str,
    multiplier: float = 2.0,
) -> Tuple[float, float, float]:
    """
    基于 ATR(14) 动态推导建议硬止损与 TP1/TP2 目标。
    返回: (stop_loss, tp1, tp2)
    """
    safe_atr = max(atr_14, current_price * 0.005)
    stop_distance = safe_atr * multiplier

    if action == "BUY_LONG":
        stop_loss = round(current_price - stop_distance, 2 if current_price > 10 else 4)
        tp1 = round(current_price + (stop_distance * 1.6), 2 if current_price > 10 else 4)
        tp2 = round(current_price + (stop_distance * 2.8), 2 if current_price > 10 else 4)
    elif action == "SELL_SHORT":
        stop_loss = round(current_price + stop_distance, 2 if current_price > 10 else 4)
        tp1 = round(current_price - (stop_distance * 1.6), 2 if current_price > 10 else 4)
        tp2 = round(current_price - (stop_distance * 2.8), 2 if current_price > 10 else 4)
    else:
        stop_loss = 0.0
        tp1 = 0.0
        tp2 = 0.0

    return stop_loss, tp1, tp2


def calculate_orderbook_imbalance(
    bids: Optional[List[List[float]]],
    asks: Optional[List[List[float]]],
    depth_levels: int = 5,
) -> Tuple[float, str]:
    """
    计算订单簿买卖深度失衡度 (Orderbook Depth Imbalance Ratio)。
    公式: Imbalance = Total Bids Volume / (Total Asks Volume + 1e-6)
    """
    if not bids or not asks:
        return 1.0, "订单簿深度数据暂不可用，按中性 1.0 评估"

    top_bids = bids[:depth_levels]
    top_asks = asks[:depth_levels]

    bid_vol = sum(float(row[1]) for row in top_bids if len(row) >= 2)
    ask_vol = sum(float(row[1]) for row in top_asks if len(row) >= 2)

    if ask_vol <= 1e-6:
        ratio = 3.0 if bid_vol > 0 else 1.0
    else:
        ratio = round(bid_vol / ask_vol, 2)

    if ratio >= 1.5:
        desc = f"买盘前 {depth_levels} 档深度是卖盘的 {ratio:.2f} 倍，多头挂单承接强劲，下方支撑坚固"
    elif ratio > 1.15:
        desc = f"买盘深度温和占优 (买/卖比={ratio:.2f})，短线偏多"
    elif ratio <= 0.67:
        desc = f"卖盘前 {depth_levels} 档深度是买盘的 {(1.0 / max(0.01, ratio)):.2f} 倍，上方抛压挂单密集"
    elif ratio < 0.87:
        desc = f"卖盘深度温和占优 (买/卖比={ratio:.2f})，短线偏空"
    else:
        desc = f"买卖盘口深度均衡 (买/卖比={ratio:.2f})，多空挂单均衡博弈"

    return ratio, desc


def evaluate_onchain_flow(
    cex_netflow_24h_usd: float = 0.0,
    whale_activity_score: float = 0.0,
    smart_money_score: float = 0.0,
    has_token_unlock_risk: bool = False,
) -> Tuple[str, float, str]:
    """
    量化评估区块链链上资金流向与巨鲸行为。
    返回: (flow_bias, composite_score, summary_desc)
    """
    score = 0.0

    # 1. 交易所净流入流出贡献
    if cex_netflow_24h_usd < -10_000_000:
        score += 0.4
        flow_str = f"CEX 24h 呈大额净提币流出 (${abs(cex_netflow_24h_usd)/1e6:.1f}M)，现货筹码沉淀显著"
    elif cex_netflow_24h_usd < -2_000_000:
        score += 0.2
        flow_str = f"CEX 24h 呈温和净提币 (${abs(cex_netflow_24h_usd)/1e6:.1f}M)"
    elif cex_netflow_24h_usd > 10_000_000:
        score -= 0.45
        flow_str = f"CEX 24h 呈现大额充币流入 (+${cex_netflow_24h_usd/1e6:.1f}M)，警惕主力现货集中抛盘"
    elif cex_netflow_24h_usd > 2_000_000:
        score -= 0.2
        flow_str = f"CEX 24h 呈现温和充币流入 (+${cex_netflow_24h_usd/1e6:.1f}M)"
    else:
        flow_str = "CEX 资金进出基本平衡"

    # 2. 聪明钱与巨鲸得分调节
    score += max(-0.35, min(0.35, smart_money_score * 0.35))
    score += max(-0.25, min(0.25, whale_activity_score * 0.25))

    # 3. 代币解锁惩罚
    if has_token_unlock_risk:
        score -= 0.3
        flow_str += "；⚠️ 近期存在大额代币解锁风险，需防范二级市场稀释"

    final_score = round(max(-1.0, min(1.0, score)), 2)

    if final_score >= 0.35:
        flow_bias = "ACCUMULATING"
        verdict = f"链上巨鲸资金呈主动吸筹沉淀格局 (得分={final_score:+.2f})，{flow_str}"
    elif final_score <= -0.35:
        flow_bias = "DISTRIBUTING"
        verdict = f"链上呈现大户出货/抛压转移迹象 (得分={final_score:+.2f})，{flow_str}"
    else:
        flow_bias = "NEUTRAL"
        verdict = f"链上整体资金流态势中性平稳 (得分={final_score:+.2f})，{flow_str}"

    return flow_bias, final_score, verdict


def calculate_kelly_position_size(
    win_rate: float,
    reward_risk_ratio: float,
    account_balance_usdt: float,
    max_risk_pct: float = 0.05,
    fractional_kelly: float = 0.5,
) -> Tuple[float, float, str]:
    """
    基于半凯利公式 (Half-Kelly Criterion) 计算最优单笔名义仓位与风险敞口。
    """
    if win_rate <= 0 or reward_risk_ratio <= 0:
        return 0.0, 0.0, "参数异常，保持 0 仓位"

    p = max(0.01, min(0.99, win_rate))
    q = 1.0 - p
    b = max(0.1, reward_risk_ratio)

    kelly_f = (p * b - q) / b
    if kelly_f <= 0:
        return 0.0, 0.0, f"在当前胜率 ({p*100:.1f}%) 与盈亏比 ({b:.2f}) 下数学期望为负，凯利建议空仓观望"

    adjusted_f = min(max_risk_pct, kelly_f * fractional_kelly)
    risk_usdt = round(account_balance_usdt * adjusted_f, 2)
    pos_pct = round(adjusted_f * 100, 2)

    desc = (
        f"半凯利模型评估: 胜率期望 {p*100:.1f}%, 盈亏比 {b:.2f} -> "
        f"建议单笔风险敞口为账户的 {pos_pct:.1f}% (~{risk_usdt} USDT)"
    )
    return risk_usdt, adjusted_f, desc


# =========================================================================
# 【新增专业工具 1】全球宏观日历与突发事件驱动评估工具
# =========================================================================

def evaluate_macro_event_risk(
    minutes_to_high_impact_event: Optional[int] = None,
    event_title: Optional[str] = None,
    news_sentiment_avg: float = 0.0,
    has_breaking_black_swan: bool = False,
) -> Tuple[str, str, bool]:
    """
    评估宏观日历事件敏感窗口与全网快讯突发风险。
    返回: (event_risk_level, diagnostic_summary, is_lockout_active)
    - event_risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
    - is_lockout_active: 是否激活防插针交易锁（强制锁仓观望）
    """
    if has_breaking_black_swan:
        return (
            "CRITICAL",
            "⚠️ 检测到突发黑天鹅/系统性突发监管利空事件，激活全局一票否决安全锁，强制观望防守！",
            True,
        )

    # 处于美联储 FOMC / CPI / 非农重大数据发布前 30 分钟或发布后 15 分钟内
    if minutes_to_high_impact_event is not None:
        if 0 <= minutes_to_high_impact_event <= 30:
            return (
                "CRITICAL",
                f"🚨 距重大宏观事件【{event_title or '重磅宏观数据'}】仅剩 {minutes_to_high_impact_event} 分钟，"
                f"流动性极易发生双向巨额插针，激活事件前强制冷静锁仓机制！",
                True,
            )
        elif 30 < minutes_to_high_impact_event <= 120:
            return (
                "HIGH",
                f"距重大宏观事件【{event_title or '重磅宏观数据'}】约 {minutes_to_high_impact_event} 分钟，"
                f"建议严格缩减头寸敞口并调紧止损距离。",
                False,
            )

    if news_sentiment_avg < -0.4:
        return (
            "MEDIUM",
            f"全网最新快讯情绪偏悲观 (平均情感得分: {news_sentiment_avg:+.2f})，需警惕短期突发空头情绪宣泄。",
            False,
        )
    elif news_sentiment_avg > 0.4:
        return (
            "LOW",
            f"全网快讯情绪积极乐观 (平均情感得分: {news_sentiment_avg:+.2f})，宏观流动性环境良好。",
            False,
        )

    return ("LOW", "宏观日历与突发舆情平稳，暂无重大高危风险窗口。", False)


# =========================================================================
# 【新增专业工具 2】微观流动性与订单簿冲击成本评估工具
# =========================================================================

def analyze_orderbook_liquidity(
    bids: Optional[List[List[float]]],
    asks: Optional[List[List[float]]],
    current_price: float,
    standard_order_usdt: float = 1000.0,
    depth_levels: int = 20,
) -> Dict[str, Any]:
    """
    微观流动性与订单簿冲击成本评估工具。
    计算买卖前 20 档的有效深度、价差 (Spread) 与预期滑点，推导最优订单执行路由模式。
    """
    if not bids or not asks or current_price <= 0:
        return {
            "spread_bps": 1.0,
            "bid_liquidity_depth_usdt": 50000.0,
            "ask_liquidity_depth_usdt": 50000.0,
            "estimated_impact_bps": 2.0,
            "recommended_execution_mode": "LIMIT",
            "execution_advice": "盘口数据暂不可用，采用标准限价委托 (LIMIT)",
        }

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    spread = max(0.0, best_ask - best_bid)
    spread_bps = round((spread / current_price) * 10000.0, 2)  # 基点 (bps)

    top_bids = bids[:depth_levels]
    top_asks = asks[:depth_levels]

    bid_depth_usdt = sum(float(r[0]) * float(r[1]) for r in top_bids if len(r) >= 2)
    ask_depth_usdt = sum(float(r[0]) * float(r[1]) for r in top_asks if len(r) >= 2)

    # 预估名义金额的冲击成本 (基点)
    avg_side_depth = max(100.0, (bid_depth_usdt + ask_depth_usdt) / 2.0)
    impact_ratio = min(0.05, standard_order_usdt / avg_side_depth)
    estimated_impact_bps = round(max(0.5, impact_ratio * 100.0 * spread_bps), 2)

    # 推导推荐执行模式
    if spread_bps > 5.0 or estimated_impact_bps > 10.0:
        exec_mode = "TWAP"
        advice = f"盘口价差偏大 ({spread_bps:.1f} bps) 或流动性偏薄，建议采用 TWAP 算法分批挂单以降低冲击成本。"
    elif spread_bps <= 1.5:
        exec_mode = "POST_ONLY"
        advice = f"盘口极其紧凑 ({spread_bps:.1f} bps)，流动性充沛，建议使用 Post-Only 被动做市挂单赚取手续费返佣并消除滑点。"
    else:
        exec_mode = "LIMIT"
        advice = f"盘口流动性适中 (价差 {spread_bps:.1f} bps)，采用标准区间 LIMIT 限价单挂单。"

    return {
        "spread_bps": spread_bps,
        "bid_liquidity_depth_usdt": round(bid_depth_usdt, 2),
        "ask_liquidity_depth_usdt": round(ask_depth_usdt, 2),
        "estimated_impact_bps": estimated_impact_bps,
        "recommended_execution_mode": exec_mode,
        "execution_advice": advice,
    }
