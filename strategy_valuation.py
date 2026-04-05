# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from read_data import StockItem
from stock import StockData


@dataclass
class StrategyResult:
    stock: str
    valuation_score: float
    dividend_spread_score: float
    price_position_score: float
    position_score: float
    roe_score: float
    total_score: float
    details: Dict[str, float]


class StrategyValuation:
    """策略层: 调用 stock 数据并进行指标评分、加权聚合。"""

    TEN_YEAR_BOND_YIELD = 1.81
    CURRENT_POSITION_WEIGHT = 0.0
    VALUATION_METHOD = "pb_pe_avg"
    ROE_FULL_SCORE = 20.0
    MAX_DIVIDEND_SPREAD = 5.0
    POSITION_MAX = 0.10
    BAOSTOCK_ADJUSTFLAG = "2"
    START_YEARS_BACK = 6
    MANUAL_DIVIDEND_YIELD: Optional[float] = None

    @staticmethod
    def _clamp(x: float, low: float = 0.0, high: float = 100.0) -> float:
        return max(low, min(high, x))

    @staticmethod
    def _percentile_low_is_better(series: pd.Series, current_value: float) -> float:
        s = pd.to_numeric(series, errors="coerce").dropna()
        s = s[np.isfinite(s)]
        if s.empty or pd.isna(current_value) or not np.isfinite(current_value):
            return np.nan
        return float((s <= current_value).mean())

    @staticmethod
    def _percentile_score_low_is_better(series: pd.Series, current_value: float) -> float:
        p = StrategyValuation._percentile_low_is_better(series, current_value)
        if pd.isna(p):
            return np.nan
        return StrategyValuation._clamp((1.0 - p) * 100.0)

    @staticmethod
    def _linear_score_higher_is_better(value: float, full_score_value: float) -> float:
        if pd.isna(value) or not np.isfinite(value):
            return np.nan
        if value <= 0:
            return 0.0
        if value >= full_score_value:
            return 100.0
        return StrategyValuation._clamp(value / full_score_value * 100.0)

    @staticmethod
    def _linear_score_lower_is_better(value: float, worst_value: float) -> float:
        if pd.isna(value) or not np.isfinite(value):
            return np.nan
        if value <= 0:
            return 100.0
        if value >= worst_value:
            return 0.0
        return StrategyValuation._clamp((1.0 - value / worst_value) * 100.0)

    @staticmethod
    def _calc_valuation_score(hist_df: pd.DataFrame, method: str = "pb_pe_avg") -> Tuple[float, Dict[str, float]]:
        latest = hist_df.iloc[-1]
        end_date = hist_df["trade_date"].max()

        windows = {"1y": 365, "3y": 365 * 3, "5y": 365 * 5}
        weights = {"1y": 0.30, "3y": 0.35, "5y": 0.35}

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

            pb_percentile = StrategyValuation._percentile_low_is_better(sub["pb"], current_pb)
            pe_percentile = StrategyValuation._percentile_low_is_better(sub["pe_ttm"], current_pe)

            pb_score = StrategyValuation._percentile_score_low_is_better(sub["pb"], current_pb)
            pe_score = StrategyValuation._percentile_score_low_is_better(sub["pe_ttm"], current_pe)

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

    @staticmethod
    def _calc_dividend_spread_score(
        current_dividend_yield: float,
        ten_year_bond_yield: float,
        max_spread: float,
    ) -> Tuple[float, Dict[str, float]]:
        spread = current_dividend_yield - ten_year_bond_yield if pd.notna(current_dividend_yield) else np.nan
        score = StrategyValuation._linear_score_higher_is_better(spread, max_spread)

        return round(score, 4), {
            "dividend_yield": round(current_dividend_yield, 4) if pd.notna(current_dividend_yield) else np.nan,
            "ten_year_bond_yield": ten_year_bond_yield,
            "dividend_spread": round(spread, 4) if pd.notna(spread) else np.nan,
            "dividend_spread_score": round(score, 4) if pd.notna(score) else np.nan,
        }

    @staticmethod
    def _calc_price_position_score(price_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        latest_close = float(price_df.iloc[-1]["close"])

        period_weights = {30: 0.05, 60: 0.10, 90: 0.30, 120: 0.30, 180: 0.10, 300: 0.15}
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
                score = StrategyValuation._clamp((1.0 - position) * 100.0)

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

    @staticmethod
    def _calc_position_score(current_position_weight: float, position_max: float) -> Tuple[float, Dict[str, float]]:
        score = StrategyValuation._linear_score_lower_is_better(current_position_weight, position_max)
        return round(score, 4), {
            "current_position_weight": current_position_weight,
            "position_max_for_zero_score": position_max,
            "position_score": round(score, 4),
        }

    @staticmethod
    def _calc_roe_score(roe_value: float, full_score_roe: float) -> Tuple[float, Dict[str, float]]:
        score = StrategyValuation._linear_score_higher_is_better(roe_value, full_score_roe)
        return round(score, 4), {
            "roe": round(roe_value, 4),
            "roe_full_score_threshold": full_score_roe,
            "roe_score": round(score, 4),
        }

    @staticmethod
    def _calc_position_score_by_max_normalized(
        current_position_weight: float,
        max_position_weight: float,
    ) -> Tuple[float, Dict[str, float]]:
        if pd.isna(current_position_weight) or current_position_weight < 0:
            current_position_weight = 0.0
        if pd.isna(max_position_weight) or max_position_weight <= 0:
            score = 0.0
        else:
            score = StrategyValuation._clamp(current_position_weight / max_position_weight * 100.0)

        return round(score, 4), {
            "current_position_weight": round(float(current_position_weight), 6),
            "max_position_weight": round(float(max_position_weight), 6) if max_position_weight > 0 else 0.0,
            "position_score": round(score, 4),
            "position_score_method": "max_normalized_by_portfolio_weight",
        }

    @staticmethod
    def run(
        symbol: str,
        raw_data: Optional[Dict[str, object]] = None,
        current_position_weight: Optional[float] = None,
        max_position_weight: Optional[float] = None,
    ) -> StrategyResult:
        if raw_data is None:
            raw_data = StockData.get_raw_data(
                symbol=symbol,
                manual_dividend_yield=StrategyValuation.MANUAL_DIVIDEND_YIELD,
                adjustflag=StrategyValuation.BAOSTOCK_ADJUSTFLAG,
                start_years_back=StrategyValuation.START_YEARS_BACK,
            )

        hist_df = raw_data["hist_df"]
        latest_roe = float(raw_data["latest_roe"])
        current_dividend_yield = float(raw_data["current_dividend_yield"])

        valuation_score, valuation_details = StrategyValuation._calc_valuation_score(
            hist_df,
            method=StrategyValuation.VALUATION_METHOD,
        )
        dividend_score, dividend_details = StrategyValuation._calc_dividend_spread_score(
            current_dividend_yield=current_dividend_yield,
            ten_year_bond_yield=StrategyValuation.TEN_YEAR_BOND_YIELD,
            max_spread=StrategyValuation.MAX_DIVIDEND_SPREAD,
        )
        price_position_score, price_details = StrategyValuation._calc_price_position_score(hist_df)
        if current_position_weight is not None and max_position_weight is not None:
            position_score, position_details = StrategyValuation._calc_position_score_by_max_normalized(
                current_position_weight=current_position_weight,
                max_position_weight=max_position_weight,
            )
        else:
            position_score, position_details = StrategyValuation._calc_position_score(
                current_position_weight=StrategyValuation.CURRENT_POSITION_WEIGHT,
                position_max=StrategyValuation.POSITION_MAX,
            )
        roe_score, roe_details = StrategyValuation._calc_roe_score(
            roe_value=latest_roe,
            full_score_roe=StrategyValuation.ROE_FULL_SCORE,
        )

        total_score = (
            valuation_score * 0.30
            + dividend_score * 0.30
            + price_position_score * 0.20
            + position_score * 0.10
            + roe_score * 0.10
        )

        details: Dict[str, float] = {}
        details["score_valuation"] = round(valuation_score, 4)
        details["score_dividend_spread"] = round(dividend_score, 4)
        details["score_price_position"] = round(price_position_score, 4)
        details["score_position"] = round(position_score, 4)
        details["score_roe"] = round(roe_score, 4)

        details["weighted_valuation"] = round(valuation_score * 0.30, 4)
        details["weighted_dividend_spread"] = round(dividend_score * 0.30, 4)
        details["weighted_price_position"] = round(price_position_score * 0.20, 4)
        details["weighted_position"] = round(position_score * 0.10, 4)
        details["weighted_roe"] = round(roe_score * 0.10, 4)

        details.update(valuation_details)
        details.update(dividend_details)
        details.update(raw_data["dividend_extra"])
        details.update(price_details)
        details.update(position_details)
        details.update(roe_details)

        details["roe_source"] = raw_data["roe_source"]
        details["current_price"] = round(float(raw_data["current_price"]), 4)

        return StrategyResult(
            stock=symbol,
            valuation_score=round(valuation_score, 4),
            dividend_spread_score=round(dividend_score, 4),
            price_position_score=round(price_position_score, 4),
            position_score=round(position_score, 4),
            roe_score=round(roe_score, 4),
            total_score=round(total_score, 4),
            details=details,
        )

    @staticmethod
    def run_from_stock_items(items: List[StockItem]) -> List[Dict[str, object]]:
        """
        策略层组合入口: 输入 csv 原始持仓数据(代码、数量、名称)，
        先计算总持仓市值和每只仓位占比，再执行每只股票策略评分。
        """
        if not items:
            return []

        records: List[Dict[str, object]] = []
        total_market_value = 0.0

        # 先准备每个股票的原始数据和市值
        for item in items:
            quantity = max(int(item.quantity), 0)
            try:
                raw_data = StockData.get_raw_data(
                    symbol=item.symbol,
                    manual_dividend_yield=StrategyValuation.MANUAL_DIVIDEND_YIELD,
                    adjustflag=StrategyValuation.BAOSTOCK_ADJUSTFLAG,
                    start_years_back=StrategyValuation.START_YEARS_BACK,
                )
                current_price = float(raw_data["current_price"])
                market_value = current_price * quantity

                records.append(
                    {
                        "symbol": item.symbol,
                        "name": item.name,
                        "quantity": quantity,
                        "current_price": current_price,
                        "market_value": market_value,
                        "raw_data": raw_data,
                        "error": None,
                    }
                )
                total_market_value += market_value
            except Exception as exc:
                records.append(
                    {
                        "symbol": item.symbol,
                        "name": item.name,
                        "quantity": quantity,
                        "current_price": 0.0,
                        "market_value": 0.0,
                        "position_weight": 0.0,
                        "strategy_result": None,
                        "error": str(exc),
                    }
                )

        # 计算仓位占比和 max-normal 仓位得分
        weights = []
        for rec in records:
            if rec.get("raw_data") is None:
                continue
            if total_market_value > 0:
                w = rec["market_value"] / total_market_value
            else:
                w = 0.0
            rec["position_weight"] = w
            weights.append(w)

        max_weight = max(weights) if weights else 0.0

        for rec in records:
            if rec.get("raw_data") is None:
                continue
            result = StrategyValuation.run(
                symbol=rec["symbol"],
                raw_data=rec["raw_data"],
                current_position_weight=rec["position_weight"],
                max_position_weight=max_weight,
            )
            result.details["holding_name"] = rec["name"]
            result.details["holding_quantity"] = rec["quantity"]
            result.details["holding_market_value"] = round(rec["market_value"], 4)
            result.details["holding_position_weight_pct"] = round(rec["position_weight"] * 100, 4)
            result.details["portfolio_total_market_value"] = round(total_market_value, 4)

            rec["strategy_result"] = result
            rec.pop("raw_data", None)

        return records
