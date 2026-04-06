# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Dict, Optional, Tuple
import re
import urllib.request

import akshare as ak
import baostock as bs
import numpy as np
import pandas as pd


class StockData:
    """股票原始数据层: 仅负责从 baostock / akshare 拉取并做最小清洗。"""

    @staticmethod
    def _clear_proxy_env() -> None:
        for key in [
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy",
        ]:
            os.environ.pop(key, None)

    @staticmethod
    def _to_bs_code(symbol: str) -> str:
        code = symbol.strip()
        if code.startswith(("sh.", "sz.")):
            return code
        if code.startswith(("6", "9")):
            return f"sh.{code}"
        return f"sz.{code}"

    @staticmethod
    def _call_with_timeout(func, timeout_seconds: int, *args, **kwargs):
        """Run potentially blocking network calls with a hard timeout."""
        result_holder: Dict[str, object] = {}
        error_holder: Dict[str, BaseException] = {}

        def _target() -> None:
            try:
                result_holder["value"] = func(*args, **kwargs)
            except BaseException as exc:
                error_holder["error"] = exc

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout_seconds)

        if thread.is_alive():
            raise TimeoutError(f"调用超时: {func.__name__}, 超时={timeout_seconds}s")
        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder.get("value")

    @staticmethod
    def _baostock_result_to_df(rs) -> pd.DataFrame:
        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())
        return pd.DataFrame(rows, columns=rs.fields)

    @staticmethod
    def _init_baostock() -> None:
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"Baostock 登录失败: {lg.error_code}, {lg.error_msg}")

    @staticmethod
    def _logout_baostock() -> None:
        try:
            bs.logout()
        except Exception:
            pass

    @staticmethod
    def _fetch_price_and_valuation_history(
        symbol: str,
        start_date: str,
        end_date: str,
        adjustflag: str = "2",
    ) -> pd.DataFrame:
        bs_code = StockData._to_bs_code(symbol)
        last_err = None

        for i in range(5):
            try:
                rs = StockData._call_with_timeout(
                    bs.query_history_k_data_plus,
                    25,
                    bs_code,
                    "date,code,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag=adjustflag,
                )
                if rs.error_code != "0":
                    raise RuntimeError(
                        f"query_history_k_data_plus 失败: {rs.error_code}, {rs.error_msg}"
                    )

                df = StockData._baostock_result_to_df(rs)
                if df.empty:
                    raise ValueError("Baostock 历史行情返回为空")

                numeric_cols = [
                    "open", "high", "low", "close", "volume", "amount",
                    "turn", "pctChg", "peTTM", "pbMRQ",
                ]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

                return df.rename(
                    columns={
                        "date": "trade_date",
                        "pctChg": "pct_chg",
                        "peTTM": "pe_ttm",
                        "pbMRQ": "pb",
                    }
                )

            except Exception as exc:
                last_err = exc
                time.sleep(2 + i * 2)

        raise RuntimeError(f"获取价格/估值数据失败: {last_err}")

    @staticmethod
    def _get_recent_completed_quarters(n: int = 6) -> list[tuple[int, int]]:
        now = datetime.now()
        year = now.year
        month = now.month

        current_quarter = (month - 1) // 3 + 1
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

    @staticmethod
    def _fetch_latest_roe(symbol: str) -> Tuple[float, pd.DataFrame, str]:
        bs_code = StockData._to_bs_code(symbol)
        # Pull enough quarters to ensure at least 3 annual ROE points can be assembled.
        year_quarters = StockData._get_recent_completed_quarters(16)

        def _collect_annual_roe(query_func, roe_col: str) -> Optional[pd.DataFrame]:
            frames = []
            for year, quarter in year_quarters:
                try:
                    rs = StockData._call_with_timeout(
                        query_func,
                        20,
                        code=bs_code,
                        year=year,
                        quarter=quarter,
                    )
                    if rs.error_code != "0":
                        continue

                    df = StockData._baostock_result_to_df(rs)
                    if df.empty or roe_col not in df.columns:
                        continue

                    stat_col = "statDate" if "statDate" in df.columns else "pubDate"
                    if stat_col not in df.columns:
                        continue

                    temp = df[[stat_col, roe_col]].copy()
                    temp[stat_col] = pd.to_datetime(temp[stat_col], errors="coerce")
                    temp[roe_col] = pd.to_numeric(temp[roe_col], errors="coerce")
                    temp = temp.dropna(subset=[stat_col, roe_col])
                    if temp.empty:
                        continue

                    temp = temp.rename(columns={stat_col: "date", roe_col: "roe"})
                    temp["roe"] = temp["roe"] * 100
                    temp["year"] = temp["date"].dt.year
                    temp["quarter"] = ((temp["date"].dt.month - 1) // 3 + 1).astype(int)
                    frames.append(temp)
                except Exception:
                    continue

            if not frames:
                return None

            hist = pd.concat(frames, ignore_index=True)
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
            hist["roe"] = pd.to_numeric(hist["roe"], errors="coerce")
            hist["year"] = pd.to_numeric(hist.get("year"), errors="coerce")
            hist["quarter"] = pd.to_numeric(hist.get("quarter"), errors="coerce")
            hist = hist.dropna(subset=["date", "roe"])\
                .sort_values("date")\
                .drop_duplicates(subset=["date"], keep="last")
            if hist.empty:
                return None

            # Keep one latest record per (year, quarter) for annualization.
            q_hist = (
                hist.dropna(subset=["year", "quarter"])
                .assign(year=lambda x: x["year"].astype(int), quarter=lambda x: x["quarter"].astype(int))
                .sort_values("date")
                .drop_duplicates(subset=["year", "quarter"], keep="last")
            )
            if q_hist.empty:
                return None

            year_to_quarter_roe: Dict[int, Dict[int, float]] = {}
            for row in q_hist.itertuples(index=False):
                y = int(row.year)
                q = int(row.quarter)
                year_to_quarter_roe.setdefault(y, {})[q] = float(row.roe)

            annual_rows = []
            annual_by_year: Dict[int, float] = {}

            for y in sorted(year_to_quarter_roe.keys()):
                q_map = year_to_quarter_roe[y]
                if not q_map:
                    continue

                estimated = False
                if 4 in q_map:
                    annual_roe = float(q_map[4])
                else:
                    latest_q = max(q_map.keys())
                    current_partial = float(q_map[latest_q])

                    prev_full = annual_by_year.get(y - 1)
                    prev_same_q = year_to_quarter_roe.get(y - 1, {}).get(latest_q)

                    if prev_full is not None and prev_same_q is not None:
                        # Estimate missing quarters by borrowing last year's remainder.
                        annual_roe = current_partial + (float(prev_full) - float(prev_same_q))
                        estimated = True
                    elif prev_full is not None:
                        annual_roe = float(prev_full)
                        estimated = True
                    else:
                        annual_roe = current_partial

                annual_by_year[y] = annual_roe
                annual_rows.append(
                    {
                        "date": pd.Timestamp(year=y, month=12, day=31),
                        "roe": annual_roe,
                        "is_estimated": estimated,
                    }
                )

            if not annual_rows:
                return None

            annual_hist = pd.DataFrame(annual_rows).sort_values("date").tail(3)[["date", "roe", "is_estimated"]]
            if annual_hist.empty:
                return None
            return annual_hist.reset_index(drop=True)


        hist_dupont = _collect_annual_roe(bs.query_dupont_data, "dupontROE")
        if hist_dupont is not None and not hist_dupont.empty:
            latest_roe = float(hist_dupont["roe"].iloc[-1])
            return latest_roe, hist_dupont, "dupontROE"

        hist_profit = _collect_annual_roe(bs.query_profit_data, "roeAvg")
        if hist_profit is not None and not hist_profit.empty:
            latest_roe = float(hist_profit["roe"].iloc[-1])
            return latest_roe, hist_profit, "roeAvg"

        raise RuntimeError(f"Baostock 未能获取到最近有效 ROE 数据: {year_quarters}")

    @staticmethod
    def _normalize_col_name(name: str) -> str:
        return re.sub(r"\s+", "", str(name))

    @staticmethod
    def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str:
        norm_map = {StockData._normalize_col_name(c): c for c in df.columns}
        for cand in candidates:
            hit = norm_map.get(StockData._normalize_col_name(cand))
            if hit is not None:
                return hit
        raise RuntimeError(f"无法匹配字段，候选={candidates}")

    @staticmethod
    def _http_get(url: str, timeout: int = 10) -> str:
        StockData._clear_proxy_env()
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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        try:
            return raw.decode("gbk", errors="ignore")
        except Exception:
            return raw.decode("utf-8", errors="ignore")

    @staticmethod
    def _to_tencent_code(symbol: str) -> str:
        if symbol.startswith(("6", "9")):
            return f"sh{symbol}"
        return f"sz{symbol}"

    @staticmethod
    def _fetch_quote_and_market_cap(symbol: str) -> Tuple[str, float, float]:
        code = StockData._to_tencent_code(symbol)
        text = StockData._http_get(f"https://qt.gtimg.cn/q={code}")
        payload = text.strip()
        m = re.search(r'="([^"]+)"', payload)
        if not m:
            raise RuntimeError(f"腾讯行情返回格式异常: {payload[:120]}")
        parts = m.group(1).split("~")
        if len(parts) < 46:
            raise RuntimeError(f"腾讯行情字段不足: count={len(parts)}")
        price = float(parts[3])
        market_cap = float(parts[44]) * 100_000_000.0  # 亿元 -> 元
        if price <= 0 or market_cap <= 0:
            raise RuntimeError(f"腾讯行情价格或总市值无效: price={price}, market_cap={market_cap}")
        return code, price, market_cap

    @staticmethod
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
        return amount / base

    @staticmethod
    def _fetch_annual_dividend_per_share(symbol: str) -> Dict[int, float]:
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
                temp["div_per_share"] = temp[text_col].apply(StockData._extract_div_per_share_from_plan_text)
                temp[report_col] = temp[report_col].astype(str).str.strip()
                temp = temp.dropna(subset=["div_per_share"])
                if not temp.empty:
                    temp["year"] = temp[report_col].str.extract(r"(\d{4})", expand=False)
                    temp = temp.dropna(subset=["year"])
                    temp["year"] = temp["year"].astype(int)
                    grouped = temp.groupby("year", as_index=False)["div_per_share"].sum()
                    for row in grouped.itertuples(index=False):
                        annual_div[int(row.year)] = float(row.div_per_share)

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
                            annual_div[int(row.year)] = float(row.div_per_share)

        return annual_div

    @staticmethod
    def _fetch_annual_eps(symbol: str) -> Dict[int, float]:
        stock_code = StockData._to_tencent_code(symbol)
        df = ak.stock_financial_report_sina(stock=stock_code, symbol="利润表")
        if df is None or df.empty:
            raise RuntimeError("利润表为空")

        date_col = StockData._resolve_column(df, ["报告日"])
        eps_col = StockData._resolve_column(df, ["基本每股收益", "基本每股收益(元/股)", "一、基本每股收益"])

        temp = df[[date_col, eps_col]].copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp[eps_col] = pd.to_numeric(temp[eps_col], errors="coerce")
        temp = temp.dropna(subset=[date_col, eps_col])
        temp["year"] = temp[date_col].dt.year.astype(int)
        temp["month"] = temp[date_col].dt.month.astype(int)

        annual_pref = temp[temp["month"] == 12].copy()
        if annual_pref.empty:
            annual_pref = temp.copy()
        annual_pref = annual_pref.sort_values(date_col).drop_duplicates(subset=["year"], keep="last")

        eps_map: Dict[int, float] = {}
        for row in annual_pref.itertuples(index=False):
            eps_map[int(getattr(row, "year"))] = float(getattr(row, eps_col))
        return eps_map

    @staticmethod
    def _calc_dividend_quality_score(symbol: str) -> Tuple[float, Dict[str, object]]:
        div_map = StockData._fetch_annual_dividend_per_share(symbol)
        if not div_map:
            raise RuntimeError("未拿到分红数据")

        eps_map = StockData._fetch_annual_eps(symbol)
        all_div_years = sorted(div_map.keys())

        def _payout_ratio_for_year(y: int) -> Optional[float]:
            dps = float(div_map.get(y, 0.0))
            eps = eps_map.get(y)
            if eps is None or eps <= 0:
                return None
            return float(dps / eps * 100.0)

        years_with_valid_payout = [y for y in all_div_years if _payout_ratio_for_year(y) is not None]
        latest_year = max(years_with_valid_payout) if years_with_valid_payout else max(all_div_years)

        earliest_year = min(all_div_years)
        start_year = max(earliest_year, latest_year - 9)
        years_used = list(range(start_year, latest_year + 1))
        if not years_used:
            raise RuntimeError("没有可用于计算的年度分红数据")

        missing_years = [y for y in years_used if y not in div_map]
        div_series = np.array([float(div_map.get(y, 0.0)) for y in years_used], dtype=float)
        payout_ratios = [_payout_ratio_for_year(y) for y in years_used]

        avg = float(np.mean(div_series))
        std = float(np.std(div_series, ddof=0))
        if avg <= 0:
            stability_score = 0.0
            cv = np.inf
        else:
            cv = std / avg
            if cv <= 0.20:
                stability_score = 100.0
            elif cv >= 1.00:
                stability_score = 0.0
            else:
                stability_score = (1.0 - (cv - 0.20) / 0.80) * 100.0

        n = len(div_series)
        if n <= 1:
            growth_score = 50.0
            g = 0.0
            avg_old = float(div_series[0]) if n == 1 else 0.0
            avg_new = avg_old
            old_n = n
            new_n = n
        else:
            split = n // 2
            old_part = div_series[:split]
            new_part = div_series[split:]
            avg_old = float(np.mean(old_part))
            avg_new = float(np.mean(new_part))
            old_n = len(old_part)
            new_n = len(new_part)

            if avg_old <= 0:
                g = 1.0 if avg_new > 0 else 0.0
            else:
                g = (avg_new - avg_old) / avg_old

            if g <= -0.30:
                growth_score = 0.0
            elif g >= 0.50:
                growth_score = 100.0
            else:
                growth_score = ((g + 0.30) / 0.80) * 100.0

        n_over = 0
        for p in payout_ratios:
            if p is not None and p > 95.0:
                n_over += 1
        n_over += len(missing_years)
        if n_over == 0:
            payout_safety_score = 100.0
        elif n_over == 1:
            payout_safety_score = 70.0
        elif n_over == 2:
            payout_safety_score = 40.0
        else:
            payout_safety_score = 0.0

        dividend_quality_score = (
            stability_score * 0.40
            + growth_score * 0.30
            + payout_safety_score * 0.30
        )

        details: Dict[str, object] = {
            "dividend_quality_score": round(float(dividend_quality_score), 4),
            "dividend_quality_stability_score": round(float(stability_score), 4),
            "dividend_quality_growth_score": round(float(growth_score), 4),
            "dividend_quality_payout_safety_score": round(float(payout_safety_score), 4),
            "dividend_quality_cv": None if not np.isfinite(cv) else round(float(cv), 6),
            "dividend_quality_g": round(float(g), 6),
            "dividend_quality_avg_old": round(float(avg_old), 6),
            "dividend_quality_avg_new": round(float(avg_new), 6),
            "dividend_quality_old_window_years": int(old_n),
            "dividend_quality_new_window_years": int(new_n),
            "dividend_quality_n_over_threshold_or_missing": int(n_over),
            "dividend_quality_years_used": [int(y) for y in years_used],
            "dividend_quality_missing_years": [int(y) for y in missing_years],
        }
        return round(float(dividend_quality_score), 4), details

    @staticmethod
    def _fetch_annual_cash_dividend_total(symbol: str, total_shares_estimate: float) -> Dict[int, float]:
        annual_total: Dict[int, float] = {}

        try:
            div_df = ak.stock_dividend_cninfo(symbol=symbol)
        except Exception:
            div_df = None

        if div_df is not None and not div_df.empty:
            text_col = "实施方案分红说明" if "实施方案分红说明" in div_df.columns else None
            report_col = "报告时间" if "报告时间" in div_df.columns else None
            if text_col and report_col:
                temp = div_df.copy()
                temp["div_per_share"] = temp[text_col].apply(StockData._extract_div_per_share_from_plan_text)
                temp[report_col] = temp[report_col].astype(str).str.strip()
                temp = temp.dropna(subset=["div_per_share"])
                if not temp.empty:
                    temp["year"] = temp[report_col].str.extract(r"(\d{4})", expand=False)
                    temp = temp.dropna(subset=["year"])
                    temp["year"] = temp["year"].astype(int)
                    grouped = temp.groupby("year", as_index=False)["div_per_share"].sum()
                    for row in grouped.itertuples(index=False):
                        y = int(row.year)
                        annual_total[y] = float(row.div_per_share) * total_shares_estimate

        if not annual_total:
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
                            annual_total[y] = float(row.div_per_share) * total_shares_estimate

        return annual_total

    @staticmethod
    def _annualize_with_previous_year_fill(yearly_quarter_values: Dict[int, Dict[int, float]]) -> Dict[int, float]:
        """Annual metric rule for cashflow: Q4 uses full-year, otherwise use latest YTD.

        This avoids over-extrapolating unfinished fiscal years using prior-year remainder.
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

    @staticmethod
    def _fetch_cash_score(symbol: str) -> Tuple[float, Dict[str, float]]:
        _, quote_price, market_cap = StockData._fetch_quote_and_market_cap(symbol)
        total_shares_estimate = market_cap / quote_price if quote_price > 0 else 0.0

        stock_code = StockData._to_tencent_code(symbol)
        df = ak.stock_financial_report_sina(stock=stock_code, symbol="现金流量表")
        if df is None or df.empty:
            raise RuntimeError("现金流量表为空")

        required = {
            "date": StockData._resolve_column(df, ["报告日"]),
            "cfo": StockData._resolve_column(df, ["经营活动产生的现金流量净额"]),
            "capex": StockData._resolve_column(df, ["购建固定资产、无形资产和其他长期资产所支付的现金"]),
        }

        temp = df[[required["date"], required["cfo"], required["capex"]]].copy()
        temp[required["date"]] = pd.to_datetime(temp[required["date"]], errors="coerce")
        temp[required["cfo"]] = pd.to_numeric(temp[required["cfo"]], errors="coerce")
        temp[required["capex"]] = pd.to_numeric(temp[required["capex"]], errors="coerce")
        temp = temp.dropna(subset=[required["date"]])
        temp = temp.sort_values(required["date"]).drop_duplicates(subset=[required["date"]], keep="last")

        temp["year"] = temp[required["date"]].dt.year.astype(int)
        temp["quarter"] = ((temp[required["date"]].dt.month - 1) // 3 + 1).astype(int)

        cfo_q: Dict[int, Dict[int, float]] = {}
        capex_q: Dict[int, Dict[int, float]] = {}
        for _, row in temp.iterrows():
            y = int(row["year"])
            q = int(row["quarter"])
            cfo_q.setdefault(y, {})[q] = float(row[required["cfo"]]) if pd.notna(row[required["cfo"]]) else 0.0
            capex_q.setdefault(y, {})[q] = float(row[required["capex"]]) if pd.notna(row[required["capex"]]) else 0.0

        cfo_annual = StockData._annualize_with_previous_year_fill(cfo_q)
        capex_annual = StockData._annualize_with_previous_year_fill(capex_q)
        div_annual = StockData._fetch_annual_cash_dividend_total(symbol, total_shares_estimate)

        years = sorted(set(cfo_annual.keys()) | set(capex_annual.keys()))
        if not years:
            raise RuntimeError("未拿到可用年化现金流数据")

        latest_year = years[-1]
        latest_cfo = float(cfo_annual.get(latest_year, 0.0))
        latest_capex = float(capex_annual.get(latest_year, 0.0))
        latest_cash_div = float(div_annual.get(latest_year, 0.0))
        latest_fcf = latest_cfo - latest_capex

        fcf_yield_pct = latest_fcf / market_cap * 100.0 if market_cap > 0 else 0.0
        score1 = 0.0 if fcf_yield_pct <= 0 else (100.0 if fcf_yield_pct >= 8.0 else fcf_yield_pct / 8.0 * 100.0)

        if latest_cash_div > 0:
            cover = latest_fcf / latest_cash_div
            score2 = 0.0 if cover <= 0 else (100.0 if cover >= 2.0 else cover / 2.0 * 100.0)
        else:
            cover = np.inf if latest_fcf > 0 else 0.0
            score2 = 100.0 if latest_fcf > 0 else 0.0

        last3 = years[-3:]
        positive_count = sum(1 for y in last3 if (float(cfo_annual.get(y, 0.0)) - float(capex_annual.get(y, 0.0))) > 0)
        score3 = positive_count / 3.0 * 100.0

        cash_score = score1 * 0.40 + score2 * 0.40 + score3 * 0.20

        details = {
            "cash_score": round(cash_score, 4),
            "cash_score1_fcf_yield": round(score1, 4),
            "cash_score2_cover": round(score2, 4),
            "cash_score3_stability": round(score3, 4),
            "cash_fcf_yield_pct": round(fcf_yield_pct, 6),
            "cash_cover": None if cover == np.inf else round(float(cover), 6),
            "cash_positive_fcf_years_last3": int(positive_count),
            "cash_latest_year": int(latest_year),
            "cash_latest_cfo": round(latest_cfo, 2),
            "cash_latest_capex": round(latest_capex, 2),
            "cash_latest_fcf": round(latest_fcf, 2),
            "cash_latest_dividend_total": round(latest_cash_div, 2),
            "cash_total_shares_estimate": round(total_shares_estimate, 2),
            "cash_quote_price": round(quote_price, 4),
            "cash_market_cap": round(market_cap, 2),
        }
        return round(cash_score, 4), details







    @staticmethod
    def _fetch_current_dividend_yield(
        symbol: str,
        current_price: float,
        manual_dividend_yield: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        if manual_dividend_yield is not None and np.isfinite(manual_dividend_yield):
            return float(manual_dividend_yield), {
                "dividend_method": "manual_input",
                "ttm_cash_dividend_per_share": np.nan,
            }

        div_df = StockData._call_with_timeout(
            ak.stock_dividend_cninfo,
            25,
            symbol=symbol,
        )
    

        if div_df is not None and not div_df.empty:
            text_col = "实施方案分红说明" if "实施方案分红说明" in div_df.columns else None
            report_col = "报告时间" if "报告时间" in div_df.columns else None

            def extract_dividend_per_share(plan_text):
                if pd.isna(plan_text):
                    return None

                plan_text = str(plan_text).strip()
                match = re.search(r"(\d+(?:\.\d+)?)\s*派\s*(\d+(?:\.\d+)?)\s*元", plan_text)
                if not match:
                    return None

                base = float(match.group(1))
                amount = float(match.group(2))

                if base <= 0:
                    return None

                return amount / base

            if text_col and report_col:
                temp = div_df.copy()
                temp["每股派息"] = temp[text_col].apply(extract_dividend_per_share)
                temp[report_col] = temp[report_col].astype(str).str.strip()

                temp = temp.dropna(subset=["每股派息"])

                current_year = pd.Timestamp.today().year
                selected_year = None

                # 从当前年份开始往前找，找到第一个“存在年报”的年份
                for y in range(current_year, current_year - 10, -1):
                    if (temp[report_col] == f"{y}年报").any():
                        selected_year = y
                        break

                # print(temp)
                # print(temp[[report_col, text_col, "每股派息"]].head())
                # print("选中的统计年份:", selected_year)

                if selected_year is not None:
                    # 只要找到了 x年报，就把所有 x 开头的都算上
                    yearly_temp = temp[temp[report_col].str.startswith(str(selected_year))].copy()
                    total_div_per_share = yearly_temp["每股派息"].sum()

                    # print(yearly_temp[[report_col, text_col, "每股派息"]])
                    # print(f"按报告时间统计，选中年份 {selected_year} 的每股派息合计:", total_div_per_share)

                    if current_price > 0 and total_div_per_share >= 0:
                        dividend_yield = total_div_per_share / current_price * 100.0
                        return float(dividend_yield), {
                            "dividend_method": "akshare_stock_dividend_by_report_year",
                            "dividend_year": int(selected_year),
                            "cash_dividend_per_share": round(float(total_div_per_share), 6),
                        }
                        
        detail_df = StockData._call_with_timeout(
            ak.stock_history_dividend_detail,
            25,
            symbol=symbol,
            indicator="分红",
        )
        
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

        raise RuntimeError("无法自动获取股息率。建议设置 manual_dividend_yield，例如 3.8")

    @staticmethod
    def get_raw_data(
        symbol: str,
        manual_dividend_yield: Optional[float] = None,
        adjustflag: str = "2",
        start_years_back: int = 6,
        manage_baostock_session: bool = True,
        category: str = "",
    ) -> Dict[str, object]:
        """输入股票代码，返回策略计算所需的原始数据。"""
        StockData._clear_proxy_env()

        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (
            datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
            - pd.Timedelta(days=365 * start_years_back)
        ).strftime("%Y-%m-%d")

        if manage_baostock_session:
            StockData._init_baostock()

        try:
            fetch_err = None
            for attempt in range(2):
                try:
                    hist_df = StockData._fetch_price_and_valuation_history(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        adjustflag=adjustflag,
                    )
                    latest_roe, roe_hist_df, roe_source = StockData._fetch_latest_roe(symbol=symbol)
                    fetch_err = None
                    break
                except Exception as exc:
                    fetch_err = exc
                    msg = str(exc).lower()
                    is_transient = (
                        ("decompress" in msg)
                        or ("接收数据异常" in msg)
                        or (("utf-8" in msg) and ("decode" in msg))
                        or ("invalid start byte" in msg)
                    )
                    if attempt == 0 and is_transient:
                        try:
                            StockData._logout_baostock()
                        except Exception:
                            pass
                        time.sleep(1)
                        StockData._init_baostock()
                        continue
                    break

            if fetch_err is not None:
                raise fetch_err
        finally:
            if manage_baostock_session:
                StockData._logout_baostock()

        current_price = float(hist_df.iloc[-1]["close"])
        current_dividend_yield, dividend_extra = StockData._fetch_current_dividend_yield(
            symbol=symbol,
            current_price=current_price,
            manual_dividend_yield=manual_dividend_yield,
        )

        is_financial = str(category).strip() == "金融"
        cash_score = np.nan
        cash_details: Dict[str, object] = {"cash_score": np.nan, "cash_score_enabled": False}
        dividend_quality_score = 0.0
        dividend_quality_details: Dict[str, object] = {"dividend_quality_score_enabled": True}

        try:
            dividend_quality_score, dq_details = StockData._calc_dividend_quality_score(symbol)
            dividend_quality_details.update(dq_details)
        except Exception as exc:
            dividend_quality_score = 0.0
            dividend_quality_details.update(
                {
                    "dividend_quality_score": 0.0,
                    "dividend_quality_error": str(exc),
                }
            )

        if not is_financial:
            try:
                cash_score, calc_details = StockData._fetch_cash_score(symbol)
                cash_details = {"cash_score_enabled": True}
                cash_details.update(calc_details)
            except Exception as exc:
                cash_score = 0.0
                cash_details = {
                    "cash_score": 0.0,
                    "cash_score_enabled": True,
                    "cash_score_error": str(exc),
                }

        return {
            "symbol": symbol,
            "category": str(category).strip(),
            "is_financial": is_financial,
            "hist_df": hist_df,
            "latest_roe": latest_roe,
            "roe_hist_df": roe_hist_df,
            "roe_source": roe_source,
            "current_price": current_price,
            "current_dividend_yield": current_dividend_yield,
            "dividend_extra": dividend_extra,
            "dividend_quality_score": dividend_quality_score,
            "dividend_quality_details": dividend_quality_details,
            "cash_score": cash_score,
            "cash_details": cash_details,
        }
