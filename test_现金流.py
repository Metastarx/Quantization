# -*- coding: utf-8 -*-
"""
独立脚本: 计算 600795 的自由现金流指数 (FCF Score)。
不依赖项目内业务模块，直接抓取公开行情/财报数据并计算。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import akshare as ak
import pandas as pd


DEBUG_PRINT_TABLE = True


def _clear_proxy_env() -> None:
    for key in [
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    ]:
        os.environ.pop(key, None)


def _http_get(url: str, timeout: int = 10) -> str:
    _clear_proxy_env()
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )

    last_err = None
    raw = b""
    for i in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            last_err = None
            break
        except Exception as exc:
            last_err = exc
            time.sleep(1 + i)
    if last_err is not None:
        raise last_err
    try:
        return raw.decode("gbk", errors="ignore")
    except Exception:
        return raw.decode("utf-8", errors="ignore")


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
    path = Path(output_dir) / f"fcf_score_600795_{ts}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(path)


def fetch_quote_and_market_cap(symbol: str) -> Tuple[str, str, float, float]:
    if symbol.startswith(("6", "9")):
        code = f"sh{symbol}"
    else:
        code = f"sz{symbol}"

    url = f"https://qt.gtimg.cn/q={code}"
    text = _http_get(url)
    payload = text.strip()

    # 示例: v_sh600795="1~国电电力~600795~4.76~...~848.98~848.98~...";
    m = re.search(r'="([^"]+)"', payload)
    if not m:
        raise RuntimeError(f"腾讯行情返回格式异常: {payload[:120]}")

    parts = m.group(1).split("~")
    if len(parts) < 46:
        raise RuntimeError(f"腾讯行情字段不足: count={len(parts)}")

    name = parts[1].strip() or symbol
    price = float(parts[3])
    # 腾讯返回的总市值通常是“亿元”。
    market_cap = float(parts[44]) * 100_000_000.0

    if price <= 0 or market_cap <= 0:
        raise RuntimeError(f"腾讯行情价格或总市值无效: price={price}, market_cap={market_cap}")

    return code, name, price, market_cap


def _score_linear(value: float, low_zero: float, high_full: float) -> float:
    if value <= low_zero:
        return 0.0
    if value >= high_full:
        return 100.0
    return (value - low_zero) / (high_full - low_zero) * 100.0


def _to_stock_code_for_sina(symbol: str) -> str:
    return f"sh{symbol}" if symbol.startswith(("6", "9")) else f"sz{symbol}"


def _annualize_with_previous_year_fill(
    yearly_quarter_values: Dict[int, Dict[int, float]],
) -> Dict[int, float]:
    """
    年度口径规则(避免对未年报年份过度外推):
    - 有Q4: 直接用Q4(全年)
    - 无Q4: 直接使用当年最新累计值(YTD)
    """
    annual_values: Dict[int, float] = {}

    for y in sorted(yearly_quarter_values.keys()):
        q_map = yearly_quarter_values.get(y, {})
        if not q_map:
            continue

        if 4 in q_map:
            annual_values[y] = float(q_map[4])
            continue

        latest_q = max(q_map.keys())
        current_partial = float(q_map[latest_q])
        annual_values[y] = current_partial

    return annual_values


def fetch_annual_cashflow_metrics(symbol: str) -> Dict[int, Dict[str, float]]:
    stock_code = _to_stock_code_for_sina(symbol)
    df = ak.stock_financial_report_sina(stock=stock_code, symbol="现金流量表")
    if df is None or df.empty:
        raise RuntimeError("现金流量表为空")

    if DEBUG_PRINT_TABLE:
        print(f"原始现金流量表数据行数: {len(df)}, 列数: {len(df.columns)}")
        print("完整列名清单:")
        for i, col in enumerate(df.columns.tolist(), start=1):
            print(f"{i:02d}. {col}")
        print("前5行(完整显示，不截断):")
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 5000,
            "display.max_colwidth", None,
        ):
            print(df.head(5).to_string(index=False))



    required = {
        "date": _resolve_column(df, ["报告日"]),
        "cfo": _resolve_column(df, ["经营活动产生的现金流量净额"]),
        "capex": _resolve_column(df, ["购建固定资产、无形资产和其他长期资产所支付的现金"]),
        # 这里用现金流量表口径近似“现金分红总额”。
        "cash_div": _resolve_column(df, ["分配股利、利润或偿付利息所支付的现金"]),
    }

    temp = df[[required["date"], required["cfo"], required["capex"], required["cash_div"]]].copy()
    temp[required["date"]] = pd.to_datetime(temp[required["date"]], errors="coerce")
    temp[required["cfo"]] = pd.to_numeric(temp[required["cfo"]], errors="coerce")
    temp[required["capex"]] = pd.to_numeric(temp[required["capex"]], errors="coerce")
    temp[required["cash_div"]] = pd.to_numeric(temp[required["cash_div"]], errors="coerce")
    temp = temp.dropna(subset=[required["date"]])
    temp = temp.sort_values(required["date"]).drop_duplicates(subset=[required["date"]], keep="last")

    temp["year"] = temp[required["date"]].dt.year.astype(int)
    temp["quarter"] = ((temp[required["date"]].dt.month - 1) // 3 + 1).astype(int)

    cfo_q: Dict[int, Dict[int, float]] = {}
    capex_q: Dict[int, Dict[int, float]] = {}
    div_q: Dict[int, Dict[int, float]] = {}

    for _, row in temp.iterrows():
        y = int(row["year"])
        q = int(row["quarter"])
        cfo_q.setdefault(y, {})[q] = float(row[required["cfo"]]) if pd.notna(row[required["cfo"]]) else 0.0
        capex_q.setdefault(y, {})[q] = float(row[required["capex"]]) if pd.notna(row[required["capex"]]) else 0.0
        div_q.setdefault(y, {})[q] = float(row[required["cash_div"]]) if pd.notna(row[required["cash_div"]]) else 0.0

    cfo_annual = _annualize_with_previous_year_fill(cfo_q)
    capex_annual = _annualize_with_previous_year_fill(capex_q)
    div_annual = _annualize_with_previous_year_fill(div_q)

    years = sorted(set(cfo_annual.keys()) | set(capex_annual.keys()) | set(div_annual.keys()))
    result: Dict[int, Dict[str, float]] = {}
    for y in years:
        cfo = float(cfo_annual.get(y, 0.0))
        capex = float(capex_annual.get(y, 0.0))
        cash_div = float(div_annual.get(y, 0.0))
        fcf = cfo - capex
        result[y] = {
            "cfo": cfo,
            "capex": capex,
            "cash_div": cash_div,
            "fcf": fcf,
        }
    return result


def main() -> None:
    symbol = "600612"
    _clear_proxy_env()

    secid, name, price, market_cap = fetch_quote_and_market_cap(symbol)
    annual = fetch_annual_cashflow_metrics(symbol)
    if not annual:
        raise RuntimeError("未拿到可用的年化现金流数据")

    years = sorted(annual.keys())
    latest_year = years[-1]
    latest = annual[latest_year]

    # Score1: FCF Yield
    fcf_yield_pct = latest["fcf"] / market_cap * 100.0 if market_cap > 0 else 0.0
    score1 = _score_linear(fcf_yield_pct, 0.0, 8.0)

    # Score2: FCF cover of cash dividend
    cash_div = latest["cash_div"]
    if cash_div > 0:
        cover = latest["fcf"] / cash_div
        score2 = _score_linear(cover, 0.0, 2.0)
    else:
        cover = float("inf") if latest["fcf"] > 0 else 0.0
        score2 = 100.0 if latest["fcf"] > 0 else 0.0

    # Score3: stability in recent 3 years
    last3_years = years[-3:]
    positive_count = sum(1 for y in last3_years if annual[y]["fcf"] > 0)
    score3 = positive_count / 3.0 * 100.0

    fcf_score = score1 * 0.40 + score2 * 0.40 + score3 * 0.20

    print(f"股票: {symbol} {name} ({secid})")
    print(f"当前价格: {price:.4f}")
    print(f"总市值: {market_cap:.2f}")
    print("-" * 70)
    print("近三年现金流(年报用全年，未年报用当年累计):")
    for y in last3_years:
        row = annual[y]
        print(
            f"{y}: CFO={row['cfo']:.2f}, CAPEX={row['capex']:.2f}, "
            f"FCF={row['fcf']:.2f}, CashDiv={row['cash_div']:.2f}"
        )

    print("-" * 70)
    print(f"Score1(FCF Yield): {score1:.4f} (FCF Yield={fcf_yield_pct:.4f}%)")
    if cash_div > 0:
        print(f"Score2(Cover): {score2:.4f} (Cover={cover:.4f})")
    else:
        print(f"Score2(Cover): {score2:.4f} (CashDiv<=0, cover按规则处理)")
    print(f"Score3(Stability): {score3:.4f} (近3年FCF为正年数={positive_count})")
    print(f"FCF_Score: {fcf_score:.4f}")

    payload = {
        "symbol": symbol,
        "name": name,
        "market": secid,
        "current_price": round(price, 6),
        "market_cap": round(market_cap, 2),
        "latest_year": latest_year,
        "latest_year_metrics": latest,
        "last3_years": last3_years,
        "annual_metrics": annual,
        "scores": {
            "score1_fcf_yield": round(score1, 4),
            "score2_cover": round(score2, 4),
            "score3_stability": round(score3, 4),
            "fcf_score": round(fcf_score, 4),
        },
        "raw_ratios": {
            "fcf_yield_pct": round(fcf_yield_pct, 6),
            "cover": None if cover == float("inf") else round(float(cover), 6),
            "positive_fcf_years_in_last3": positive_count,
        },
        "formula": {
            "score1": "FCFYield<=0:0, >=8:100, else FCFYield/8*100",
            "score2": "Cover<=0:0, >=2:100, else Cover/2*100",
            "score3": "近3年FCF为正年数/3*100",
            "fcf_score": "score1*0.4 + score2*0.4 + score3*0.2",
        },
    }
    saved = _save_result_json(payload)
    print(f"结果JSON已保存: {saved}")


if __name__ == "__main__":
    main()
