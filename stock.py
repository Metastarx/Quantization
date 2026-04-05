# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

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
                rs = bs.query_history_k_data_plus(
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
        year_quarters = StockData._get_recent_completed_quarters(6)

        for year, quarter in year_quarters:
            try:
                rs = bs.query_dupont_data(code=bs_code, year=year, quarter=quarter)
                if rs.error_code != "0":
                    continue

                df = StockData._baostock_result_to_df(rs)
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
            except Exception:
                continue

        for year, quarter in year_quarters:
            try:
                rs = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
                if rs.error_code != "0":
                    continue

                df = StockData._baostock_result_to_df(rs)
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
            except Exception:
                continue

        raise RuntimeError(f"Baostock 未能获取到最近有效 ROE 数据: {year_quarters}")

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

        raise RuntimeError("无法自动获取股息率。建议设置 manual_dividend_yield，例如 3.8")

    @staticmethod
    def get_raw_data(
        symbol: str,
        manual_dividend_yield: Optional[float] = None,
        adjustflag: str = "2",
        start_years_back: int = 6,
    ) -> Dict[str, object]:
        """输入股票代码，返回策略计算所需的原始数据。"""
        StockData._clear_proxy_env()

        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (
            datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
            - pd.Timedelta(days=365 * start_years_back)
        ).strftime("%Y-%m-%d")

        StockData._init_baostock()
        try:
            hist_df = StockData._fetch_price_and_valuation_history(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjustflag=adjustflag,
            )
            latest_roe, roe_hist_df, roe_source = StockData._fetch_latest_roe(symbol=symbol)
        finally:
            StockData._logout_baostock()

        current_price = float(hist_df.iloc[-1]["close"])
        current_dividend_yield, dividend_extra = StockData._fetch_current_dividend_yield(
            symbol=symbol,
            current_price=current_price,
            manual_dividend_yield=manual_dividend_yield,
        )

        return {
            "symbol": symbol,
            "hist_df": hist_df,
            "latest_roe": latest_roe,
            "roe_hist_df": roe_hist_df,
            "roe_source": roe_source,
            "current_price": current_price,
            "current_dividend_yield": current_dividend_yield,
            "dividend_extra": dividend_extra,
        }
