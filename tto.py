# -*- coding: utf-8 -*-
"""
独立脚本: 计算 600795 的分红质量分 (dividend_quality_score)。
不依赖项目内业务模块，直接抓取公开财报/分红数据并计算。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import akshare as ak
import numpy as np
import pandas as pd

def _normalize_col_name(name: str) -> str:
    return re.sub(r"\s+", "", str(name))


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str:
    norm_map = {_normalize_col_name(c): c for c in df.columns}
    for cand in candidates:
        hit = norm_map.get(_normalize_col_name(cand))
        if hit is not None:
            return hit
    raise RuntimeError(f"无法匹配字段，候选={candidates}")


def _save_result_json(payload: dict, output_dir: str = "outputs") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"dividend_quality_600795_{ts}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(path)


def _extract_div_per_share_from_plan_text(plan_text: str) -> Optional[float]:
    if not plan_text:
        return None
    text = str(plan_text).strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*派\s*(\d+(?:\.\d+)?)\s*元", text)
    if not m:
        return None
    base = float(m.group(1))
    amount = float(m.group(2))
    if base <= 0:
        return None
    per_share = amount / base
    return per_share if per_share >= 0 else None


def fetch_annual_dividend_per_share(symbol: str) -> Dict[int, float]:
    """优先用分红方案提取每股分红；失败时用历史分红明细兜底。"""
    annual_div: Dict[int, float] = {}

    try:
        div_df = ak.stock_dividend_cninfo(symbol=symbol)
    except Exception:
        div_df = None

    if div_df is not None and not div_df.empty:
        text_col = "实施方案分红说明" if "实施方案分红说明" in div_df.columns else None
        report_col = "报告时间" if "报告时间" in div_df.columns else None
        if text_col and report_col:
            temp = div_df.copy()
            temp["div_per_share"] = temp[text_col].apply(_extract_div_per_share_from_plan_text)
            temp[report_col] = temp[report_col].astype(str).str.strip()
            temp = temp.dropna(subset=["div_per_share"])
            if not temp.empty:
                temp["year"] = temp[report_col].str.extract(r"(\d{4})", expand=False)
                temp = temp.dropna(subset=["year"])
                temp["year"] = temp["year"].astype(int)
                grouped = temp.groupby("year", as_index=False)["div_per_share"].sum()
                for row in grouped.itertuples(index=False):
                    y = int(row.year)
                    annual_div[y] = float(row.div_per_share)

    if not annual_div:
        try:
            detail_df = ak.stock_history_dividend_detail(symbol=symbol, indicator="分红")
        except Exception:
            detail_df = None

        if detail_df is not None and not detail_df.empty:
            date_col = "除权除息日" if "除权除息日" in detail_df.columns else None
            cash_col = "派息" if "派息" in detail_df.columns else None
            if date_col and cash_col:
                temp = detail_df.copy()
                temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
                temp[cash_col] = pd.to_numeric(temp[cash_col], errors="coerce")
                temp = temp.dropna(subset=[date_col, cash_col])
                if not temp.empty:
                    temp["year"] = temp[date_col].dt.year.astype(int)
                    temp["div_per_share"] = temp[cash_col] / 10.0
                    grouped = temp.groupby("year", as_index=False)["div_per_share"].sum()
                    for row in grouped.itertuples(index=False):
                        y = int(row.year)
                        annual_div[y] = float(row.div_per_share)

    return annual_div


def _to_stock_code_for_sina(symbol: str) -> str:
    return f"sh{symbol}" if symbol.startswith(("6", "9")) else f"sz{symbol}"


def fetch_annual_eps(symbol: str) -> Dict[int, float]:
    """从利润表提取每年的基本每股收益(EPS)。"""
    stock_code = _to_stock_code_for_sina(symbol)
    df = ak.stock_financial_report_sina(stock=stock_code, symbol="利润表")
    if df is None or df.empty:
        raise RuntimeError("利润表为空")

    date_col = _resolve_column(df, ["报告日"])
    eps_col = _resolve_column(df, ["基本每股收益", "基本每股收益(元/股)", "一、基本每股收益"])

    temp = df[[date_col, eps_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[eps_col] = pd.to_numeric(temp[eps_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, eps_col])
    temp["year"] = temp[date_col].dt.year.astype(int)
    temp["month"] = temp[date_col].dt.month.astype(int)

    # 优先使用年报(12月)，若缺失则退化为当年最后一条。
    annual_pref = temp[temp["month"] == 12].copy()
    if annual_pref.empty:
        annual_pref = temp.copy()
    annual_pref = annual_pref.sort_values(date_col).drop_duplicates(subset=["year"], keep="last")

    result: Dict[int, float] = {}
    for row in annual_pref.itertuples(index=False):
        year = int(getattr(row, "year"))
        eps = float(getattr(row, eps_col))
        result[year] = eps
    return result


def _calc_stability_score(div_series_10y: list[float]) -> tuple[float, float]:
    arr = np.array(div_series_10y, dtype=float)
    avg = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))

    if avg <= 0:
        return 0.0, np.inf

    cv = std / avg
    if cv <= 0.20:
        score = 100.0
    elif cv >= 1.00:
        score = 0.0
    else:
        score = (1.0 - (cv - 0.20) / (1.00 - 0.20)) * 100.0
    return float(max(0.0, min(100.0, score))), float(cv)


def _calc_growth_score(div_series: list[float]) -> tuple[float, float, float, float, int, int]:
    n = len(div_series)
    if n <= 1:
        avg_only = float(div_series[0]) if n == 1 else 0.0
        return 50.0, 0.0, avg_only, avg_only, n, n

    split = n // 2
    old = np.array(div_series[:split], dtype=float)
    new = np.array(div_series[split:], dtype=float)
    avg_old = float(np.mean(old))
    avg_new = float(np.mean(new))

    if avg_old <= 0:
        g = 1.0 if avg_new > 0 else 0.0
    else:
        g = (avg_new - avg_old) / avg_old

    if g <= -0.30:
        score = 0.0
    elif g >= 0.50:
        score = 100.0
    else:
        score = ((g + 0.30) / 0.80) * 100.0

    return float(max(0.0, min(100.0, score))), float(g), avg_old, avg_new, len(old), len(new)


def _calc_payout_safety_score(
    payout_ratios_pct: list[Optional[float]],
    missing_years_count: int,
) -> tuple[float, int]:
    n_over = 0
    for p in payout_ratios_pct:
        if p is not None and p > 95.0:
            n_over += 1

    # 缺失年份视为分红质量风险，计入超阈值年份。
    n_over += int(missing_years_count)

    if n_over == 0:
        score = 100.0
    elif n_over == 1:
        score = 70.0
    elif n_over == 2:
        score = 40.0
    else:
        score = 0.0

    return score, n_over


def main() -> None:
    symbol = "600795"
    div_per_share_map = fetch_annual_dividend_per_share(symbol)
    eps_map = fetch_annual_eps(symbol)

    if not div_per_share_map:
        raise RuntimeError("未拿到分红数据")

    all_div_years = sorted(div_per_share_map.keys())

    def _payout_ratio_for_year(y: int) -> Optional[float]:
        dps = float(div_per_share_map.get(y, 0.0))
        eps = eps_map.get(y)
        if eps is None or eps <= 0:
            return None
        return float(dps / eps * 100.0)

    years_with_valid_payout = [y for y in all_div_years if _payout_ratio_for_year(y) is not None]
    if years_with_valid_payout:
        latest_year = max(years_with_valid_payout)
    else:
        latest_year = max(all_div_years)

    earliest_year = min(all_div_years)
    start_year = max(earliest_year, latest_year - 9)
    years_used = list(range(start_year, latest_year + 1))
    if not years_used:
        raise RuntimeError("没有可用于计算的年度分红数据")

    missing_years = [y for y in years_used if y not in div_per_share_map]
    div_series: list[float] = [float(div_per_share_map.get(y, 0.0)) for y in years_used]
    payout_ratios_pct: list[Optional[float]] = [_payout_ratio_for_year(y) for y in years_used]

    stability_score, cv = _calc_stability_score(div_series)
    growth_score, g, avg_old, avg_new, old_n, new_n = _calc_growth_score(div_series)
    payout_safety_score, n_over = _calc_payout_safety_score(
        payout_ratios_pct,
        missing_years_count=len(missing_years),
    )

    dividend_quality_score = (
        stability_score * 0.40
        + growth_score * 0.30
        + payout_safety_score * 0.30
    )

    print(f"股票: {symbol}")
    print("-" * 70)
    print(f"用于计算的年份: {years_used[0]}-{years_used[-1]} (共{len(years_used)}年, 最多年取10年)")
    if missing_years:
        print(f"缺失年份(按0分红计入并扣分): {missing_years}")
    print("每股分红与分红支付率(缺失/<=0 EPS 记为 N/A):")
    for y, dps, payout in zip(years_used, div_series, payout_ratios_pct):
        payout_txt = "N/A" if payout is None else f"{payout:.2f}%"
        print(
            f"{y}: DPS={dps:.4f}, PayoutRatio={payout_txt}"
        )

    print("-" * 70)
    print(f"StabilityScore: {stability_score:.4f} (CV={cv if np.isfinite(cv) else float('inf'):.6f})")
    print(
        f"GrowthScore: {growth_score:.4f} "
        f"(g={g:.6f}, avg_old={avg_old:.6f}[{old_n}年], avg_new={avg_new:.6f}[{new_n}年])"
    )
    print(
        f"PayoutSafetyScore: {payout_safety_score:.4f} "
        f"(n_over_threshold_or_missing={n_over}, missing_years={len(missing_years)})"
    )
    print(f"DividendQualityScore: {dividend_quality_score:.4f}")

    payload = {
        "symbol": symbol,
        "latest_year": latest_year,
        "years_used": years_used,
        "years_used_count": len(years_used),
        "missing_years": missing_years,
        "missing_years_count": len(missing_years),
        "annual_dividend_per_share": {str(y): round(div_per_share_map.get(y, 0.0), 6) for y in years_used},
        "annual_eps": {str(y): (round(eps_map.get(y, 0.0), 6) if eps_map.get(y) is not None else None) for y in years_used},
        "annual_payout_ratio_pct": {
            str(y): (None if p is None else round(float(p), 6))
            for y, p in zip(years_used, payout_ratios_pct)
        },
        "sub_scores": {
            "stability_score": round(stability_score, 4),
            "growth_score": round(growth_score, 4),
            "payout_safety_score": round(payout_safety_score, 4),
        },
        "dividend_quality_score": round(dividend_quality_score, 4),
        "raw_metrics": {
            "cv": None if not np.isfinite(cv) else round(cv, 6),
            "g": round(g, 6),
            "avg_old": round(avg_old, 6),
            "avg_new": round(avg_new, 6),
            "old_window_years": old_n,
            "new_window_years": new_n,
            "n_over_threshold_or_missing": int(n_over),
        },
        "formula": {
            "stability_score": "CV<=0.20:100; CV>=1.00:0; else (1-(CV-0.20)/(1.00-0.20))*100",
            "growth_score": "g<=-0.30:0; g>=0.50:100; else (g+0.30)/0.80*100",
            "payout_safety_score": "n_over(>95%)+missing_years; n=0:100; n=1:70; n=2:40; n>=3:0",
            "dividend_quality_score": "stability*0.40 + growth*0.30 + payout_safety*0.30",
        },
    }

    saved = _save_result_json(payload)
    print(f"结果JSON已保存: {saved}")


if __name__ == "__main__":
    main()
