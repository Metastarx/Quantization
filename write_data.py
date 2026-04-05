# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class WriteData:
    """IO 写入层。当前先提供策略结果落盘能力。"""

    @staticmethod
    def save_json(data: Any, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if is_dataclass(data):
            payload = asdict(data)
        else:
            payload = data

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

        return str(path)

    @staticmethod
    def build_default_output_path(symbol: str, output_dir: str = "outputs") -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(Path(output_dir) / f"valuation_{symbol}_{ts}.json")
