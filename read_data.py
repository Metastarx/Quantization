# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class StockItem:
    symbol: str
    quantity: int
    name: str
    category: str = ""


class ReadData:
    """IO 读取层。"""

    @staticmethod
    def read_symbols_from_csv(csv_path: str) -> List[str]:
        path = Path(csv_path)
        if not path.exists():
            return []

        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
        if df.empty:
            return []

        first_col = df.columns[0]
        symbols = (
            df[first_col]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .str.zfill(6)
            .str.extract(r"(\d{6})", expand=False)
            .dropna()
            .tolist()
        )
        return symbols

    @staticmethod
    def read_stock_items_from_csv(csv_path: str) -> List[StockItem]:
        """读取股票清单: 第一列代码、第二列持仓数量、第三列名称。"""
        path = Path(csv_path)
        if not path.exists():
            return []

        # 文件没有表头，固定按四列读取（第四列可为空）
        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            header=None,
            names=["symbol", "quantity", "name", "category"],
            dtype={"symbol": str},
        )
        if df.empty:
            return []

        df["symbol"] = (
            df["symbol"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .str.zfill(6)
            .str.extract(r"(\d{6})", expand=False)
        )
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
        df["name"] = df["name"].fillna("").astype(str)
        df["category"] = df["category"].fillna("").astype(str).str.strip()
        df = df.dropna(subset=["symbol"])

        return [
            StockItem(
                symbol=row.symbol,
                quantity=int(row.quantity),
                name=row.name,
                category=row.category,
            )
            for row in df.itertuples(index=False)
        ]
