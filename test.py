# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import baostock as bs
from tqdm import tqdm
import akshare as ak



# =========================
# 0. 清理代理残留
# =========================
for _k in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy"
]:
    os.environ.pop(_k, None)


# =========================
# 1. 参数区
# =========================

SYMBOL = "600900"                 # 6位A股代码
TEN_YEAR_BOND_YIELD = 1.81        # 十年国债利率，单位 %
CURRENT_POSITION_WEIGHT = 0.00    # 当前仓位占比
VALUATION_METHOD = "pb_pe_avg"    # pb_only / pe_only / pb_pe_avg
ROE_FULL_SCORE = 20.0             # ROE达到20%记满分（百分数口径）
MAX_DIVIDEND_SPREAD = 5.0
POSITION_MAX = 0.10
BAOSTOCK_ADJUSTFLAG = "2"         # 1 后复权, 2 前复权, 3 不复权
START_YEARS_BACK = 6              # 为1/3/5年估值分位准备6年数据

# 最稳方案：手工填当前股息率，例如 3.8 表示 3.8%
# 若设为 None，则自动用 AKShare 分红数据估算
MANUAL_DIVIDEND_YIELD: Optional[float] = None


# =========================
# 2. 数据结构
# =========================

@dataclass
class ScoreResult:
    stock: str
    valuation_score: float
    dividend_spread_score: float
    price_position_score: float
    position_score: float
    roe_score: float
    total_score: float
    details: Dict[str, float]


# =========================
# 3. 通用函数
# =========================

def clamp(x: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, x))


def percentile_low_is_better(series: pd.Series, current_value: float) -> float:
    """
    返回百分位(0~1): 当前值在历史序列中所处位置
    值越低越便宜，则百分位越低越好
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[np.isfinite(s)]
    if s.empty or pd.isna(current_value) or not np.isfinite(current_value):
        return np.nan
    return float((s <= current_value).mean())


def percentile_score_low_is_better(series: pd.Series, current_value: float) -> float:
    p = percentile_low_is_better(series, current_value)
    if pd.isna(p):
        return np.nan
    return clamp((1.0 - p) * 100.0)


def linear_score_higher_is_better(value: float, full_score_value: float) -> float:
    if pd.isna(value) or not np.isfinite(value):
        return np.nan
    if value <= 0:
        return 0.0
    if value >= full_score_value:
        return 100.0
    return clamp(value / full_score_value * 100.0)


def linear_score_lower_is_better(value: float, worst_value: float) -> float:
    if pd.isna(value) or not np.isfinite(value):
        return np.nan
    if value <= 0:
        return 100.0
    if value >= worst_value:
        return 0.0
    return clamp((1.0 - value / worst_value) * 100.0)


def to_bs_code(symbol: str) -> str:
    symbol = symbol.strip()
    if symbol.startswith(("sh.", "sz.")):
        return symbol
    if symbol.startswith(("6", "9")):
        return f"sh.{symbol}"
    return f"sz.{symbol}"


def baostock_result_to_df(rs) -> pd.DataFrame:
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


# =========================
# 4. Baostock 数据函数
# =========================

def init_baostock():
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"Baostock 登录失败: {lg.error_code}, {lg.error_msg}")


def logout_baostock():
    try:
        bs.logout()
    except Exception:
        pass


def fetch_price_and_valuation_history_bs(
    symbol: str,
    start_date: str,
    end_date: str,
    adjustflag: str = "2",
) -> pd.DataFrame:
    """
    一次取回:
    date, open, high, low, close, volume, amount, turn, pctChg, peTTM, pbMRQ
    """
    bs_code = to_bs_code(symbol)

    last_err = None
    for i in range(5):
        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,code,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag=adjustflag,
            )
            if rs.error_code != "0":
                raise RuntimeError(f"query_history_k_data_plus 失败: {rs.error_code}, {rs.error_msg}")

            df = baostock_result_to_df(rs)
            if df.empty:
                raise ValueError("Baostock 历史行情返回为空")

            for col in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg", "peTTM", "pbMRQ"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

            df = df.rename(columns={
                "date": "trade_date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "amount": "amount",
                "turn": "turn",
                "pctChg": "pct_chg",
                "peTTM": "pe_ttm",
                "pbMRQ": "pb",
            })
            return df

        except Exception as e:
            last_err = e
            print(f"[Baostock-价格估值] 第 {i+1} 次失败: {e}")
            time.sleep(2 + i * 2)

    raise RuntimeError(f"获取价格/估值数据失败: {last_err}")


def get_recent_completed_quarters(n: int = 4):
    """
    获取最近 n 个已完成季度
    例如当前是 2026-04-05，则最近已完成季度应从 2025Q4 开始
    """
    now = datetime.now()
    year = now.year
    month = now.month

    # 当前日期所属季度
    current_quarter = (month - 1) // 3 + 1

    # 最近“已完成”季度 = 当前季度的前一个季度
    completed_quarter = current_quarter - 1
    completed_year = year
    if completed_quarter == 0:
        completed_quarter = 4
        completed_year -= 1

    result = []
    y, q = completed_year, completed_quarter
    for _ in range(n):
        result.append((y, q))
        q -= 1
        if q == 0:
            q = 4
            y -= 1
    return result


def fetch_latest_roe_bs(symbol: str) -> Tuple[float, pd.DataFrame, str]:
    """
    只查最近4个“已完成季度”，找到最近一个有效 ROE 就返回
    优先 dupontROE，失败再回退 roeAvg
    返回百分数口径，例如 13.07
    """
    bs_code = to_bs_code(symbol)

    # 最近四个已完成季度
    year_quarters = get_recent_completed_quarters(6)
    print("最近四个已完成季度:", year_quarters)

    # 先查 dupontROE
    for year, quarter in tqdm(year_quarters, desc="获取 ROE(最近四个已完成季度)", ncols=100):
        try:
            rs = bs.query_dupont_data(code=bs_code, year=year, quarter=quarter)
            if rs.error_code != "0":
                continue

            df = baostock_result_to_df(rs)
            if df.empty or "dupontROE" not in df.columns:
                continue

            df["dupontROE"] = pd.to_numeric(df["dupontROE"], errors="coerce")
            stat_col = "statDate" if "statDate" in df.columns else "pubDate"
            df[stat_col] = pd.to_datetime(df[stat_col], errors="coerce")
            df = df.dropna(subset=[stat_col, "dupontROE"]).sort_values(stat_col)

            if not df.empty:
                latest_roe = float(df["dupontROE"].iloc[-1]) * 100
                hist = df[[stat_col, "dupontROE"]].rename(
                    columns={stat_col: "date", "dupontROE": "roe"}
                )
                hist["roe"] = pd.to_numeric(hist["roe"], errors="coerce") * 100
                return latest_roe, hist.reset_index(drop=True), "dupontROE"

        except Exception as e:
            print(f"\nquery_dupont_data {year}Q{quarter} 失败: {e}")

    # 再查 roeAvg
    for year, quarter in tqdm(year_quarters, desc="获取 ROE回退(roeAvg)", ncols=100):
        try:
            rs = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
            if rs.error_code != "0":
                continue

            df = baostock_result_to_df(rs)
            if df.empty or "roeAvg" not in df.columns:
                continue

            df["roeAvg"] = pd.to_numeric(df["roeAvg"], errors="coerce")
            stat_col = "statDate" if "statDate" in df.columns else "pubDate"
            df[stat_col] = pd.to_datetime(df[stat_col], errors="coerce")
            df = df.dropna(subset=[stat_col, "roeAvg"]).sort_values(stat_col)

            if not df.empty:
                latest_roe = float(df["roeAvg"].iloc[-1]) * 100
                hist = df[[stat_col, "roeAvg"]].rename(
                    columns={stat_col: "date", "roeAvg": "roe"}
                )
                hist["roe"] = pd.to_numeric(hist["roe"], errors="coerce") * 100
                return latest_roe, hist.reset_index(drop=True), "roeAvg"

        except Exception as e:
            print(f"\nquery_profit_data {year}Q{quarter} 失败: {e}")

    raise RuntimeError(f"Baostock 未能获取到最近四个已完成季度有效 ROE 数据: {year_quarters}")

# =========================
# 5. 股息率
# =========================

def fetch_current_dividend_yield(
    symbol: str,
    current_price: float,
    manual_dividend_yield: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    优先:
    1) 手工输入当前股息率
    2) AKShare 的 CNInfo 分红明细 -> 近12个月现金分红 / 当前价 * 100
    """
    if manual_dividend_yield is not None and np.isfinite(manual_dividend_yield):
        return float(manual_dividend_yield), {
            "dividend_method": "manual_input",
            "ttm_cash_dividend_per_share": np.nan,
        }

    if ak is None:
        raise RuntimeError("未安装 akshare，且未手工提供 MANUAL_DIVIDEND_YIELD")

    try:
        div_df = ak.stock_dividend_cninfo(symbol=symbol)
        if div_df is not None and not div_df.empty:
            date_candidates = ["除权日", "派息日", "股权登记日", "实施方案公告日期"]
            cash_candidates = ["派息比例", "税前分红率"]

            date_col = next((c for c in date_candidates if c in div_df.columns), None)
            cash_col = next((c for c in cash_candidates if c in div_df.columns), None)

            if date_col and cash_col:
                temp = div_df.copy()
                temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
                temp[cash_col] = pd.to_numeric(temp[cash_col], errors="coerce")
                temp = temp.dropna(subset=[date_col, cash_col])

                cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
                ttm_div_per_share = temp.loc[temp[date_col] >= cutoff, cash_col].sum() / 10.0

                if current_price > 0 and ttm_div_per_share >= 0:
                    dividend_yield = ttm_div_per_share / current_price * 100.0
                    return float(dividend_yield), {
                        "dividend_method": "akshare_stock_dividend_cninfo_ttm_estimated",
                        "ttm_cash_dividend_per_share": round(ttm_div_per_share, 6),
                    }
    except Exception as e:
        print(f"[股息率] stock_dividend_cninfo 失败: {e}")

    try:
        detail_df = ak.stock_history_dividend_detail(symbol=symbol, indicator="分红")
        if detail_df is not None and not detail_df.empty:
            date_candidates = ["除权除息日", "股权登记日", "实施公告日"]
            cash_candidates = ["派息", "每10股派息", "分红总额"]

            date_col = next((c for c in date_candidates if c in detail_df.columns), None)
            cash_col = next((c for c in cash_candidates if c in detail_df.columns), None)

            if date_col and cash_col:
                temp = detail_df.copy()
                temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
                temp[cash_col] = pd.to_numeric(temp[cash_col], errors="coerce")
                temp = temp.dropna(subset=[date_col, cash_col])

                cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
                ttm_raw = temp.loc[temp[date_col] >= cutoff, cash_col].sum()
                ttm_div_per_share = ttm_raw / 10.0
                if current_price > 0 and ttm_div_per_share >= 0:
                    dividend_yield = ttm_div_per_share / current_price * 100.0
                    return float(dividend_yield), {
                        "dividend_method": "akshare_stock_history_dividend_detail_ttm_estimated",
                        "ttm_cash_dividend_per_share": round(ttm_div_per_share, 6),
                    }
    except Exception as e:
        print(f"[股息率] stock_history_dividend_detail 失败: {e}")

    raise RuntimeError("无法自动获取股息率。建议手工设置 MANUAL_DIVIDEND_YIELD，例如 3.8 表示 3.8%")


# =========================
# 6. 各项得分
# =========================

def calc_valuation_score(hist_df: pd.DataFrame, method: str = "pb_pe_avg") -> Tuple[float, Dict[str, float]]:
    latest = hist_df.iloc[-1]
    end_date = hist_df["trade_date"].max()

    windows = {
        "1y": 365,
        "3y": 365 * 3,
        "5y": 365 * 5,
    }
    weights = {
        "1y": 0.30,
        "3y": 0.35,
        "5y": 0.35,
    }

    current_pb = float(latest["pb"]) if pd.notna(latest["pb"]) else np.nan
    current_pe = float(latest["pe_ttm"]) if pd.notna(latest["pe_ttm"]) else np.nan

    details = {
        "current_pb": round(current_pb, 4) if pd.notna(current_pb) else np.nan,
        "current_pe_ttm": round(current_pe, 4) if pd.notna(current_pe) else np.nan,
    }

    total = 0.0
    valid_weight_sum = 0.0

    for key, days in windows.items():
        start = end_date - pd.Timedelta(days=days)
        sub = hist_df.loc[hist_df["trade_date"] >= start].copy()

        pb_percentile = percentile_low_is_better(sub["pb"], current_pb)
        pe_percentile = percentile_low_is_better(sub["pe_ttm"], current_pe)

        pb_score = percentile_score_low_is_better(sub["pb"], current_pb)
        pe_score = percentile_score_low_is_better(sub["pe_ttm"], current_pe)

        if method == "pb_only":
            score = pb_score
        elif method == "pe_only":
            score = pe_score
        else:
            vals = [x for x in [pb_score, pe_score] if pd.notna(x)]
            score = float(np.mean(vals)) if vals else np.nan

        details[f"{key}_pb_percentile_pct"] = round(pb_percentile * 100, 4) if pd.notna(pb_percentile) else np.nan
        details[f"{key}_pb_score"] = round(pb_score, 4) if pd.notna(pb_score) else np.nan

        details[f"{key}_pe_percentile_pct"] = round(pe_percentile * 100, 4) if pd.notna(pe_percentile) else np.nan
        details[f"{key}_pe_score"] = round(pe_score, 4) if pd.notna(pe_score) else np.nan

        details[f"{key}_valuation_score"] = round(score, 4) if pd.notna(score) else np.nan

        if pd.notna(score):
            total += score * weights[key]
            valid_weight_sum += weights[key]

    if valid_weight_sum == 0:
        return np.nan, details

    total = total / valid_weight_sum
    return round(total, 4), details


def calc_dividend_spread_score(
    current_dividend_yield: float,
    ten_year_bond_yield: float,
    max_spread: float = 5.0
) -> Tuple[float, Dict[str, float]]:
    spread = current_dividend_yield - ten_year_bond_yield if pd.notna(current_dividend_yield) else np.nan
    score = linear_score_higher_is_better(spread, max_spread)

    return round(score, 4), {
        "dividend_yield": round(current_dividend_yield, 4) if pd.notna(current_dividend_yield) else np.nan,
        "ten_year_bond_yield": ten_year_bond_yield,
        "dividend_spread": round(spread, 4) if pd.notna(spread) else np.nan,
        "dividend_spread_score": round(score, 4) if pd.notna(score) else np.nan,
    }


def calc_price_position_score(price_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    latest_close = float(price_df.iloc[-1]["close"])

    period_weights = {
        30: 0.05,
        60: 0.10,
        90: 0.30,
        120: 0.30,
        180: 0.10,
        300: 0.15,
    }

    details = {}
    total = 0.0
    valid_weight_sum = 0.0

    for period, w in period_weights.items():
        sub = price_df.tail(period).copy()
        if len(sub) < 2:
            details[f"{period}d_price_score"] = np.nan
            continue

        low_ = float(sub["low"].min())
        high_ = float(sub["high"].max())

        if math.isclose(high_, low_):
            position = 0.5
            score = 50.0
        else:
            position = (latest_close - low_) / (high_ - low_)
            score = clamp((1.0 - position) * 100.0)

        details[f"{period}d_low"] = round(low_, 4)
        details[f"{period}d_high"] = round(high_, 4)
        details[f"{period}d_position_pct"] = round(position * 100, 4)
        details[f"{period}d_price_score"] = round(score, 4)

        total += score * w
        valid_weight_sum += w

    if valid_weight_sum == 0:
        return np.nan, details

    total = total / valid_weight_sum
    return round(total, 4), details


def calc_position_score(current_position_weight: float, position_max: float = 0.10) -> Tuple[float, Dict[str, float]]:
    score = linear_score_lower_is_better(current_position_weight, position_max)
    return round(score, 4), {
        "current_position_weight": current_position_weight,
        "position_max_for_zero_score": position_max,
        "position_score": round(score, 4),
    }


def calc_roe_score(roe_value: float, full_score_roe: float = 20.0) -> Tuple[float, Dict[str, float]]:
    score = linear_score_higher_is_better(roe_value, full_score_roe)
    return round(score, 4), {
        "roe": round(roe_value, 4),
        "roe_full_score_threshold": full_score_roe,
        "roe_score": round(score, 4),
    }


# =========================
# 7. 主流程
# =========================

def calculate_stock_score(
    symbol: str,
    ten_year_bond_yield: float,
    current_position_weight: float,
    valuation_method: str = "pb_pe_avg",
    roe_full_score: float = 20.0,
    max_dividend_spread: float = 5.0,
    position_max: float = 0.10,
    adjustflag: str = "2",
    manual_dividend_yield: Optional[float] = None,
) -> ScoreResult:
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365 * START_YEARS_BACK)).strftime("%Y-%m-%d")

    print("1/4 正在登录 Baostock...")
    init_baostock()

    try:
        print("2/4 正在获取价格、PB、PE 历史数据...")
        hist_df = fetch_price_and_valuation_history_bs(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjustflag=adjustflag,
        )
        print(f"价格/估值数据获取完成，共 {len(hist_df)} 条")

        print("3/4 正在获取 ROE 数据...")
        latest_roe, roe_hist_df, roe_source = fetch_latest_roe_bs(symbol=symbol)
        print(f"ROE 获取完成，来源: {roe_source}，最新值: {latest_roe:.4f}")

    finally:
        logout_baostock()
        print("Baostock 已退出")

    print("4/4 正在获取股息率...")
    current_price = float(hist_df.iloc[-1]["close"])
    current_dividend_yield, dividend_extra = fetch_current_dividend_yield(
        symbol=symbol,
        current_price=current_price,
        manual_dividend_yield=manual_dividend_yield,
    )
    print(f"股息率获取完成，当前股息率: {current_dividend_yield:.4f}%")

    valuation_score, valuation_details = calc_valuation_score(hist_df, method=valuation_method)
    dividend_score, dividend_details = calc_dividend_spread_score(
        current_dividend_yield=current_dividend_yield,
        ten_year_bond_yield=ten_year_bond_yield,
        max_spread=max_dividend_spread,
    )
    price_position_score, price_details = calc_price_position_score(hist_df)
    position_score, position_details = calc_position_score(
        current_position_weight=current_position_weight,
        position_max=position_max,
    )
    roe_score, roe_details = calc_roe_score(
        roe_value=latest_roe,
        full_score_roe=roe_full_score,
    )

    total_score = (
        valuation_score * 0.30
        + dividend_score * 0.30
        + price_position_score * 0.20
        + position_score * 0.10
        + roe_score * 0.10
    )

    details = {}

    # 五大指标总分
    details["score_valuation"] = round(valuation_score, 4)
    details["score_dividend_spread"] = round(dividend_score, 4)
    details["score_price_position"] = round(price_position_score, 4)
    details["score_position"] = round(position_score, 4)
    details["score_roe"] = round(roe_score, 4)

    # 五大指标加权贡献
    details["weighted_valuation"] = round(valuation_score * 0.30, 4)
    details["weighted_dividend_spread"] = round(dividend_score * 0.30, 4)
    details["weighted_price_position"] = round(price_position_score * 0.20, 4)
    details["weighted_position"] = round(position_score * 0.10, 4)
    details["weighted_roe"] = round(roe_score * 0.10, 4)

    details.update(valuation_details)
    details.update(dividend_details)
    details.update(dividend_extra)
    details.update(price_details)
    details.update(position_details)
    details.update(roe_details)

    details["roe_source"] = roe_source
    details["current_price"] = round(current_price, 4)

    return ScoreResult(
        stock=symbol,
        valuation_score=round(valuation_score, 4),
        dividend_spread_score=round(dividend_score, 4),
        price_position_score=round(price_position_score, 4),
        position_score=round(position_score, 4),
        roe_score=round(roe_score, 4),
        total_score=round(total_score, 4),
        details=details,
    )


# =========================
# 8. 执行
# =========================

if __name__ == "__main__":
    result = calculate_stock_score(
        symbol=SYMBOL,
        ten_year_bond_yield=TEN_YEAR_BOND_YIELD,
        current_position_weight=CURRENT_POSITION_WEIGHT,
        valuation_method=VALUATION_METHOD,
        roe_full_score=ROE_FULL_SCORE,
        max_dividend_spread=MAX_DIVIDEND_SPREAD,
        position_max=POSITION_MAX,
        adjustflag=BAOSTOCK_ADJUSTFLAG,
        manual_dividend_yield=MANUAL_DIVIDEND_YIELD,
    )

    print("=" * 70)
    print(f"股票: {result.stock}")
    print(f"总分: {result.total_score:.2f}")
    print("=" * 70)

    print("\n一、五大指标总分")
    print(f"1. 历史估值分位得分: {result.valuation_score:.2f}")
    print(f"2. 股息率-国债利差得分: {result.dividend_spread_score:.2f}")
    print(f"3. 多周期价格区位得分: {result.price_position_score:.2f}")
    print(f"4. 组合仓位偏离得分: {result.position_score:.2f}")
    print(f"5. ROE盈利能力得分: {result.roe_score:.2f}")

    print("\n二、五大指标加权贡献分")
    print(f"1. 历史估值分位贡献分: {result.details['weighted_valuation']:.2f}")
    print(f"2. 股息率-国债利差贡献分: {result.details['weighted_dividend_spread']:.2f}")
    print(f"3. 多周期价格区位贡献分: {result.details['weighted_price_position']:.2f}")
    print(f"4. 组合仓位偏离贡献分: {result.details['weighted_position']:.2f}")
    print(f"5. ROE盈利能力贡献分: {result.details['weighted_roe']:.2f}")

    print("\n三、指标一：历史估值分位明细")
    print(f"当前 PB: {result.details['current_pb']}")
    print(f"当前 PE(TTM): {result.details['current_pe_ttm']}")
    print(f"1年内 PB 分位: {result.details['1y_pb_percentile_pct']:.2f}% -> PB得分: {result.details['1y_pb_score']:.2f}")
    print(f"1年内 PE 分位: {result.details['1y_pe_percentile_pct']:.2f}% -> PE得分: {result.details['1y_pe_score']:.2f}")
    print(f"1年估值得分: {result.details['1y_valuation_score']:.2f}")
    print(f"3年内 PB 分位: {result.details['3y_pb_percentile_pct']:.2f}% -> PB得分: {result.details['3y_pb_score']:.2f}")
    print(f"3年内 PE 分位: {result.details['3y_pe_percentile_pct']:.2f}% -> PE得分: {result.details['3y_pe_score']:.2f}")
    print(f"3年估值得分: {result.details['3y_valuation_score']:.2f}")
    print(f"5年内 PB 分位: {result.details['5y_pb_percentile_pct']:.2f}% -> PB得分: {result.details['5y_pb_score']:.2f}")
    print(f"5年内 PE 分位: {result.details['5y_pe_percentile_pct']:.2f}% -> PE得分: {result.details['5y_pe_score']:.2f}")
    print(f"5年估值得分: {result.details['5y_valuation_score']:.2f}")
    print(f"指标一总分: {result.details['score_valuation']:.2f}")

    print("\n四、指标二：股息率-国债利差明细")
    print(f"当前股息率: {result.details['dividend_yield']:.4f}%")
    print(f"十年国债收益率: {result.details['ten_year_bond_yield']:.4f}%")
    print(f"股息率利差: {result.details['dividend_spread']:.4f}%")
    print(f"股息率来源: {result.details['dividend_method']}")
    print(f"TTM每股现金分红: {result.details['ttm_cash_dividend_per_share']}")
    print(f"指标二得分: {result.details['dividend_spread_score']:.2f}")

    print("\n五、指标三：多周期价格区位明细")
    for period in [30, 60, 90, 120, 180, 300]:
        print(
            f"{period}日区间: 低点={result.details[f'{period}d_low']}, "
            f"高点={result.details[f'{period}d_high']}, "
            f"当前区位={result.details[f'{period}d_position_pct']:.2f}% -> "
            f"得分={result.details[f'{period}d_price_score']:.2f}"
        )
    print(f"指标三总分: {result.details['score_price_position']:.2f}")

    print("\n六、指标四：组合仓位偏离明细")
    print(f"当前仓位占比: {result.details['current_position_weight']:.4f}")
    print(f"满惩罚仓位阈值: {result.details['position_max_for_zero_score']:.4f}")
    print(f"指标四得分: {result.details['position_score']:.2f}")

    print("\n七、指标五：ROE盈利能力明细")
    print(f"ROE来源: {result.details['roe_source']}")
    print(f"最近有效 ROE: {result.details['roe']:.4f}%")
    print(f"ROE满分阈值: {result.details['roe_full_score_threshold']:.2f}%")
    print(f"指标五得分: {result.details['roe_score']:.2f}")

    print("\n八、全部原始明细字典")
    for k, v in result.details.items():
        print(f"{k}: {v}")