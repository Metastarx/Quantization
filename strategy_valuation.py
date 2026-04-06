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
    dividend_quality_score: float
    price_position_score: float
    position_score: float
    roe_score: float
    cash_score: float
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
        weights = {"1y": 0.25, "3y": 0.40, "5y": 0.35}

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

            # pb_pct_text = f"{pb_percentile * 100:.2f}%" if pd.notna(pb_percentile) else "nan"
            # pe_pct_text = f"{pe_percentile * 100:.2f}%" if pd.notna(pe_percentile) else "nan"
            # print(f"[估值分位] {key}: PB占比={pb_pct_text}, PE占比={pe_pct_text}")

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
    def _calc_roe_trend_score(delta: float) -> float:
        # Piecewise mapping with delta bounds at [-3, 3].
        if pd.isna(delta) or not np.isfinite(delta):
            return np.nan
        if delta <= -3:
            return 0.0
        if delta >= 3:
            return 100.0
        return StrategyValuation._clamp(((delta + 3.0) / 6.0) * 100.0)

    @staticmethod
    def _calc_roe_score(
        roe_value: float,
        full_score_roe: float,
        roe_hist_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, Dict[str, float]]:
        absolute_score = StrategyValuation._linear_score_higher_is_better(roe_value, full_score_roe)

        roe_base = np.nan
        delta = np.nan
        trend_score = np.nan

        if roe_hist_df is not None and not roe_hist_df.empty and "roe" in roe_hist_df.columns:
            hist = roe_hist_df.copy()
            if "date" in hist.columns:
                hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
                hist = hist.sort_values("date")
            hist["roe"] = pd.to_numeric(hist["roe"], errors="coerce")
            hist = hist.dropna(subset=["roe"])

            # Use latest three ROE points: roe1(current), roe2, roe3.
            if len(hist) >= 3:
                roe1 = float(hist["roe"].iloc[-1])
                roe2 = float(hist["roe"].iloc[-2])
                roe3 = float(hist["roe"].iloc[-3])
                roe_base = (roe2 + roe3) / 2.0
                delta = roe1 - roe_base
                trend_score = StrategyValuation._calc_roe_trend_score(delta)

        # If history is not enough, keep stable behavior by falling back to absolute score.
        if pd.isna(trend_score):
            trend_score = absolute_score

        score = absolute_score * 0.70 + trend_score * 0.30
        return round(score, 4), {
            "roe": round(roe_value, 4),
            "roe_full_score_threshold": full_score_roe,
            "roe_absolute_score": round(absolute_score, 4) if pd.notna(absolute_score) else np.nan,
            "roe_trend_base": round(roe_base, 4) if pd.notna(roe_base) else np.nan,
            "roe_delta": round(delta, 4) if pd.notna(delta) else np.nan,
            "roe_trend_score": round(trend_score, 4) if pd.notna(trend_score) else np.nan,
            "roe_score": round(score, 4) if pd.notna(score) else np.nan,
            "roe_score_formula": "absolute_70pct_plus_trend_30pct",
        }







    @staticmethod
    def _calc_position_score_by_max_normalized(
        current_position_weight: float,
        max_position_weight: float,
    ) -> Tuple[float, Dict[str, float]]:
        if pd.isna(current_position_weight) or current_position_weight < 0:
            current_position_weight = 0.0
        if pd.isna(max_position_weight) or max_position_weight <= 0:
            score = 100.0
        else:
            score = StrategyValuation._clamp((1.0 - current_position_weight / max_position_weight) * 100.0)

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
        is_financial = bool(raw_data.get("is_financial", False))
        category = str(raw_data.get("category", "")).strip()
        cash_score_raw = raw_data.get("cash_score", np.nan)
        cash_score = float(cash_score_raw) if pd.notna(cash_score_raw) else np.nan
        dividend_quality_raw = raw_data.get("dividend_quality_score", 0.0)
        dividend_quality_score = float(dividend_quality_raw) if pd.notna(dividend_quality_raw) else 0.0

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
            roe_hist_df=raw_data.get("roe_hist_df"),
        )

        

        if is_financial:
            # 金融: 旧五因子；非金融: 在原公式基础上增加 roe*0.5 + cash*0.5。
            base_total = (
            valuation_score * 0.30
            + price_position_score * 0.15
            + dividend_score * 0.20
            + dividend_quality_score * 0.10
            + position_score * 0.10
            + roe_score * 0.15
            )
            weighted_roe = roe_score * 0.10
            weighted_cash = np.nan
            weighted_dividend_quality = dividend_quality_score * 0.10
            total_score = base_total
            score_formula = "financial_dividend_split_formula"
        else:
            weighted_roe = roe_score * 0.1
            weighted_cash = (cash_score * 0.05) if pd.notna(cash_score) else 0.0
            weighted_dividend_quality = dividend_quality_score * 0.10
            total_score = (
                valuation_score * 0.30
                + price_position_score * 0.15
                + dividend_score * 0.20
                + weighted_dividend_quality
                + position_score * 0.10
                + weighted_roe
                + weighted_cash
            )

            score_formula = "non_financial_dividend_split_formula"

        details: Dict[str, float] = {}
        details["score_valuation"] = round(valuation_score, 4)
        details["score_dividend_spread"] = round(dividend_score, 4)
        details["score_dividend_quality"] = round(dividend_quality_score, 4)
        details["score_price_position"] = round(price_position_score, 4)
        details["score_position"] = round(position_score, 4)
        details["score_roe"] = round(roe_score, 4)
        details["score_cash"] = round(cash_score, 4) if pd.notna(cash_score) else np.nan

        details["weighted_valuation"] = round(valuation_score * 0.30, 4)
        details["weighted_dividend_spread"] = round(dividend_score * 0.20, 4)
        details["weighted_dividend_quality"] = round(weighted_dividend_quality, 4)
        details["weighted_price_position"] = round(price_position_score * 0.20, 4)
        details["weighted_position"] = round(position_score * 0.10, 4)
        details["weighted_roe"] = round(weighted_roe, 4)
        details["weighted_cash"] = round(weighted_cash, 4) if pd.notna(weighted_cash) else np.nan
        details["category"] = category
        details["is_financial"] = is_financial
        details["score_formula"] = score_formula

        details.update(valuation_details)
        details.update(dividend_details)
        details.update(raw_data["dividend_extra"])
        details.update(price_details)
        details.update(position_details)
        details.update(roe_details)
        details.update(raw_data.get("dividend_quality_details", {}))
        details.update(raw_data.get("cash_details", {}))

        details["roe_source"] = raw_data["roe_source"]
        details["current_price"] = round(float(raw_data["current_price"]), 4)

        return StrategyResult(
            stock=symbol,
            valuation_score=round(valuation_score, 4),
            dividend_spread_score=round(dividend_score, 4),
            dividend_quality_score=round(dividend_quality_score, 4),
            price_position_score=round(price_position_score, 4),
            position_score=round(position_score, 4),
            roe_score=round(roe_score, 4),
            cash_score=round(cash_score, 4) if pd.notna(cash_score) else np.nan,
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

        print("[组合预处理] 登录 Baostock（全组合一次）")
        StockData._init_baostock()
        try:
            # 先准备每个股票的原始数据和市值
            for item in items:
                quantity = max(int(item.quantity), 0)
                try:
                    print(f"[组合预处理] 开始拉取 {item.symbol} {item.name} 的原始数据")
                    raw_data = StockData.get_raw_data(
                        symbol=item.symbol,
                        manual_dividend_yield=StrategyValuation.MANUAL_DIVIDEND_YIELD,
                        adjustflag=StrategyValuation.BAOSTOCK_ADJUSTFLAG,
                        start_years_back=StrategyValuation.START_YEARS_BACK,
                        manage_baostock_session=False,
                        category=item.category,
                    )
                    print(f"[组合预处理] 完成拉取 {item.symbol} {item.name} 的原始数据")
                    current_price = float(raw_data["current_price"])
                    market_value = current_price * quantity

                    records.append(
                        {
                            "symbol": item.symbol,
                            "name": item.name,
                            "category": item.category,
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
                            "category": item.category,
                            "quantity": quantity,
                            "current_price": 0.0,
                            "market_value": 0.0,
                            "position_weight": 0.0,
                            "strategy_result": None,
                            "error": str(exc),
                        }
                    )
        finally:
            StockData._logout_baostock()
            print("[组合预处理] 已退出 Baostock（全组合结束）")

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

            score_items = {
                "valuation": {
                    "raw": {
                        "current_pb": result.details.get("current_pb"),
                        "current_pe_ttm": result.details.get("current_pe_ttm"),
                        "1y_pb_percentile_pct": result.details.get("1y_pb_percentile_pct"),
                        "1y_pe_percentile_pct": result.details.get("1y_pe_percentile_pct"),
                        "3y_pb_percentile_pct": result.details.get("3y_pb_percentile_pct"),
                        "3y_pe_percentile_pct": result.details.get("3y_pe_percentile_pct"),
                        "5y_pb_percentile_pct": result.details.get("5y_pb_percentile_pct"),
                        "5y_pe_percentile_pct": result.details.get("5y_pe_percentile_pct"),
                    },
                    "score": result.details.get("score_valuation"),
                    "weighted": result.details.get("weighted_valuation"),
                },
                "dividend_spread": {
                    "raw": {
                        "dividend_yield": result.details.get("dividend_yield"),
                        "ten_year_bond_yield": result.details.get("ten_year_bond_yield"),
                        "dividend_spread": result.details.get("dividend_spread"),
                        "dividend_method": result.details.get("dividend_method"),
                        "ttm_cash_dividend_per_share": result.details.get("ttm_cash_dividend_per_share"),
                    },
                    "score": result.details.get("score_dividend_spread"),
                    "weighted": result.details.get("weighted_dividend_spread"),
                },
                "dividend_quality": {
                    "raw": {
                        "dividend_quality_cv": result.details.get("dividend_quality_cv"),
                        "dividend_quality_g": result.details.get("dividend_quality_g"),
                        "dividend_quality_avg_old": result.details.get("dividend_quality_avg_old"),
                        "dividend_quality_avg_new": result.details.get("dividend_quality_avg_new"),
                        "dividend_quality_missing_years": result.details.get("dividend_quality_missing_years"),
                        "dividend_quality_error": result.details.get("dividend_quality_error"),
                    },
                    "score": result.details.get("score_dividend_quality"),
                    "weighted": result.details.get("weighted_dividend_quality"),
                },
                "price_position": {
                    "raw": {
                        "30d_position_pct": result.details.get("30d_position_pct"),
                        "60d_position_pct": result.details.get("60d_position_pct"),
                        "90d_position_pct": result.details.get("90d_position_pct"),
                        "120d_position_pct": result.details.get("120d_position_pct"),
                        "180d_position_pct": result.details.get("180d_position_pct"),
                        "300d_position_pct": result.details.get("300d_position_pct"),
                    },
                    "score": result.details.get("score_price_position"),
                    "weighted": result.details.get("weighted_price_position"),
                },
                "position": {
                    "raw": {
                        "holding_market_value": result.details.get("holding_market_value"),
                        "holding_position_weight_pct": result.details.get("holding_position_weight_pct"),
                        "max_position_weight": result.details.get("max_position_weight"),
                        "position_score_method": result.details.get("position_score_method"),
                    },
                    "score": result.details.get("score_position"),
                    "weighted": result.details.get("weighted_position"),
                },
                "roe": {
                    "raw": {
                        "roe": result.details.get("roe"),
                        "roe_source": result.details.get("roe_source"),
                        "roe_full_score_threshold": result.details.get("roe_full_score_threshold"),
                        "roe_prev_two_year_avg": result.details.get("roe_trend_base"),
                    },
                    "score": result.details.get("score_roe"),
                    "weighted": result.details.get("weighted_roe"),
                },
                "cash": {
                    "raw": {
                        "cash_score_enabled": result.details.get("cash_score_enabled"),
                        "cash_latest_year": result.details.get("cash_latest_year"),
                        "cash_latest_cfo": result.details.get("cash_latest_cfo"),
                        "cash_latest_capex": result.details.get("cash_latest_capex"),
                        "cash_latest_fcf": result.details.get("cash_latest_fcf"),
                        "cash_latest_dividend_total": result.details.get("cash_latest_dividend_total"),
                        "cash_fcf_yield_pct": result.details.get("cash_fcf_yield_pct"),
                        "cash_cover": result.details.get("cash_cover"),
                        "cash_positive_fcf_years_last3": result.details.get("cash_positive_fcf_years_last3"),
                        "cash_score_error": result.details.get("cash_score_error"),
                    },
                    "score": result.details.get("score_cash"),
                    "weighted": result.details.get("weighted_cash"),
                },
            }

            rec["strategy_result"] = result
            rec["total_score"] = result.total_score
            rec["score_summary"] = {
                "valuation_score": result.valuation_score,
                "dividend_spread_score": result.dividend_spread_score,
                "dividend_quality_score": result.dividend_quality_score,
                "price_position_score": result.price_position_score,
                "position_score": result.position_score,
                "roe_score": result.roe_score,
                "cash_score": result.cash_score,
                "category": result.details.get("category"),
                "is_financial": result.details.get("is_financial"),
                "score_formula": result.details.get("score_formula"),
                "total_score": result.total_score,
            }
            rec["score_items"] = score_items
            rec.pop("raw_data", None)

        return records
