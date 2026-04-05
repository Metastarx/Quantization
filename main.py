# -*- coding: utf-8 -*-
from __future__ import annotations

from read_data import ReadData
from strategy_valuation import StrategyValuation
from write_data import WriteData


STOCK_CSV = "stock.csv"


def main() -> None:
    items = ReadData.read_stock_items_from_csv(STOCK_CSV)
    if not items:
        print(f"未读取到股票清单: {STOCK_CSV}")
        return

    print(f"共读取到 {len(items)} 只股票，开始执行组合策略...")

    records = StrategyValuation.run_from_stock_items(items)
    if not records:
        print("策略层没有返回可保存结果")
        return

    success_count = 0
    fail_count = 0

    for idx, rec in enumerate(records, start=1):
        print("-" * 70)
        print(f"[{idx}/{len(records)}] {rec['symbol']} {rec['name']} 持仓={rec['quantity']}")

        try:
            if rec.get("strategy_result") is None:
                print(f"执行失败: {rec['symbol']} {rec['name']}, 错误: {rec.get('error', '未知错误')}")
                fail_count += 1
                continue

            payload = {
                "symbol": rec["symbol"],
                "name": rec["name"],
                "quantity": rec["quantity"],
                "current_price": rec["current_price"],
                "market_value": rec["market_value"],
                "position_weight": rec["position_weight"],
                "strategy_result": rec["strategy_result"],
            }

            output_path = WriteData.build_default_output_path(symbol=rec["symbol"])
            saved_path = WriteData.save_json(payload, output_path)

            print(f"仓位占比: {rec['position_weight'] * 100:.2f}%")
            print(f"总分: {rec['strategy_result'].total_score:.2f}")
            print(f"结果已保存: {saved_path}")
            success_count += 1
        except Exception as exc:
            print(f"执行失败: {rec['symbol']} {rec['name']}, 错误: {exc}")
            fail_count += 1

    print("=" * 70)
    print(f"执行完成: 成功 {success_count}，失败 {fail_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
