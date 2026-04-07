# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

from read_data import ReadData
from stock import StockData


INPUT_CSV = "stock.csv"
OUTPUT_DIR = Path("outputs")


def _to_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _calc_dividend_amount(
    quantity: int,
    current_price: float,
    dividend_yield_pct: float,
    dividend_extra: Dict[str, object],
) -> float:
    # Prefer per-share cash dividend if stock.py already returned one.
    cash_per_share = _to_float(dividend_extra.get("cash_dividend_per_share"), default=-1.0)
    if cash_per_share >= 0:
        return quantity * cash_per_share

    ttm_cash_per_share = _to_float(dividend_extra.get("ttm_cash_dividend_per_share"), default=-1.0)
    if ttm_cash_per_share >= 0:
        return quantity * ttm_cash_per_share

    return quantity * current_price * dividend_yield_pct / 100.0


def generate_dividend_csv(input_csv: str = INPUT_CSV) -> Path:
    items = ReadData.read_stock_items_from_csv(input_csv)
    items = [item for item in items if item.quantity > 0]

    if not items:
        raise RuntimeError("没有可处理的持仓（数量都为 0 或文件为空）")

    rows: List[Dict[str, object]] = []
    total_dividend = 0.0

    for idx, item in enumerate(items, start=1):
        print(f"[{idx}/{len(items)}] 计算 {item.symbol} {item.name}...")
        raw = StockData.get_raw_data(symbol=item.symbol, category=item.category)

        dividend_yield_pct = _to_float(raw.get("current_dividend_yield"), default=0.0)
        current_price = _to_float(raw.get("current_price"), default=0.0)
        dividend_extra = raw.get("dividend_extra") or {}
        if not isinstance(dividend_extra, dict):
            dividend_extra = {}

        dividend_amount = _calc_dividend_amount(
            quantity=item.quantity,
            current_price=current_price,
            dividend_yield_pct=dividend_yield_pct,
            dividend_extra=dividend_extra,
        )
        total_dividend += dividend_amount

        rows.append(
            {
                "股票号": item.symbol,
                "股票数量": item.quantity,
                "股票名称": item.name,
                "股票分红率": round(dividend_yield_pct, 4),
                "分红金额": round(dividend_amount, 2),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"dividend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    import pandas as pd

    pd.DataFrame(rows, columns=["股票号", "股票数量", "股票名称", "股票分红率", "分红金额"]).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n已生成: {output_path}")
    print(f"总分红: {total_dividend:.2f}")
    return output_path


if __name__ == "__main__":
    generate_dividend_csv()
