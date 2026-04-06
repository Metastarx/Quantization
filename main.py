# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime

from read_data import ReadData
from sender_qmsg import SenderLayerQmsg
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

    records = sorted(
        records,
        key=lambda rec: rec.get("total_score")
        if rec.get("total_score") is not None
        else (
            rec.get("strategy_result").total_score
            if rec.get("strategy_result") is not None
            else float("-inf")
        ),
        reverse=True,
    )

    success_count = 0
    fail_count = 0
    concise_records = []
    detailed_records = []

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    concise_output_path = f"outputs/valuation_{run_ts}_简略.json"
    detailed_output_path = f"outputs/valuation_{run_ts}_详细.json"

    for idx, rec in enumerate(records, start=1):
        try:
            header_text = "\n".join(
                [
                    f"[{idx}/{len(records)}] {rec['symbol']} {rec['name']} 持仓={rec['quantity']}"
                ]
            )

            if rec.get("strategy_result") is None:
                message_text = "\n".join(
                    [
                        header_text,
                        f"执行失败: {rec['symbol']} {rec['name']}, 错误: {rec.get('error', '未知错误')}",
                    ]
                )
                print(message_text)
                SenderLayerQmsg.sender_layer_enqueue(message_text)
                print(f"[入队] {rec['symbol']} -> 等待统一发送")

                concise_records.append(
                    {
                        "symbol": rec["symbol"],
                        "name": rec["name"],
                        "quantity": rec["quantity"],
                        "status": "failed",
                        "error": rec.get("error", "未知错误"),
                    }
                )
                detailed_records.append(
                    {
                        "symbol": rec["symbol"],
                        "name": rec["name"],
                        "quantity": rec["quantity"],
                        "status": "failed",
                        "error": rec.get("error", "未知错误"),
                    }
                )

                fail_count += 1
                continue

            score_summary = rec.get("score_summary", {})

            score_items = rec.get("score_items", {})
            valuation_raw = score_items.get("valuation", {}).get("raw", {})
            dividend_raw = score_items.get("dividend_spread", {}).get("raw", {})
            dividend_quality_raw = score_items.get("dividend_quality", {}).get("raw", {})
            position_raw = score_items.get("position", {}).get("raw", {})
            roe_raw = score_items.get("roe", {}).get("raw", {})
            cash_raw = score_items.get("cash", {}).get("raw", {})

            concise_item = {
                "symbol": rec["symbol"],
                "name": rec["name"],
                "quantity": rec["quantity"],
                "current_price": rec["current_price"],
                "market_value": rec["market_value"],
                "position_weight": rec["position_weight"],
                "total_score": rec.get("total_score", rec["strategy_result"].total_score),
                "score_summary": score_summary,
                "raw_values": {
                    "current_price": rec["current_price"],
                    "current_pb": valuation_raw.get("current_pb"),
                    "current_pe_ttm": valuation_raw.get("current_pe_ttm"),
                    "dividend_yield": dividend_raw.get("dividend_yield"),
                    "dividend_spread": dividend_raw.get("dividend_spread"),
                    "dividend_quality_score": score_summary.get("dividend_quality_score"),
                    "holding_position_weight_pct": position_raw.get("holding_position_weight_pct"),
                    "roe": roe_raw.get("roe"),
                    "cash_score": score_summary.get("cash_score"),
                    "cash_fcf_yield_pct": cash_raw.get("cash_fcf_yield_pct"),
                },
            }
            concise_records.append(concise_item)

            details = rec["strategy_result"].details
            detailed_item = {
                "symbol": rec["symbol"],
                "name": rec["name"],
                "quantity": rec["quantity"],
                "current_price": rec["current_price"],
                "market_value": rec["market_value"],
                "position_weight": rec["position_weight"],
                "total_score": rec.get("total_score", rec["strategy_result"].total_score),
                "score_weights": {
                    "formula": score_summary.get("score_formula"),
                    "category": score_summary.get("category"),
                    "is_financial": score_summary.get("is_financial"),
                    "valuation": 0.30,
                    "dividend_spread": 0.20,
                    "dividend_quality": 0.10,
                    "price_position": 0.20,
                    "position": 0.10,
                    "roe": 0.10 if score_summary.get("is_financial") else 0.50,
                    "cash": None if score_summary.get("is_financial") else 0.50,
                },
                "score_summary": score_summary,
                "score_items": score_items,
                "score_contributions": {
                    "weighted_valuation": details.get("weighted_valuation"),
                    "weighted_dividend_spread": details.get("weighted_dividend_spread"),
                    "weighted_dividend_quality": details.get("weighted_dividend_quality"),
                    "weighted_price_position": details.get("weighted_price_position"),
                    "weighted_position": details.get("weighted_position"),
                    "weighted_roe": details.get("weighted_roe"),
                    "weighted_cash": details.get("weighted_cash"),
                },
                "strategy_details": details,
            }
            detailed_records.append(detailed_item)

            result_text = "\n".join(
                [
                    header_text,
                    f"总分: {rec.get('total_score', rec['strategy_result'].total_score):.2f}",
                    (
                        "分项得分: \n"
                        f"PEPB分位估值={score_summary.get('valuation_score')}, "
                        f"价格区位={score_summary.get('price_position_score')}, "
                        f"股息利差={score_summary.get('dividend_spread_score')}, "
                        f"分红质量分={score_summary.get('dividend_quality_score')}, "
                        f"仓位={score_summary.get('position_score')}, "
                        f"ROE质量分={score_summary.get('roe_score')}, "
                        f"现金流分={score_summary.get('cash_score')}"
                    ),
                    (
                        "原始值: \n"
                        f"当前股价={rec['current_price']}, "
                        f"PB={valuation_raw.get('current_pb')}, "
                        f"PE(TTM)={valuation_raw.get('current_pe_ttm')}, "
                        f"股息率={dividend_raw.get('dividend_yield')}, "
                        f"利差={dividend_raw.get('dividend_spread')}, "
                        f"仓位占比%={position_raw.get('holding_position_weight_pct')}, "
                        f"ROE={roe_raw.get('roe')}, "
                        f"FCF收益率%={cash_raw.get('cash_fcf_yield_pct')}"
                    ),
                    (
                        "评分模式: "
                        f"category={score_summary.get('category')}, "
                        f"is_financial={score_summary.get('is_financial')}, "
                        f"formula={score_summary.get('score_formula')}"
                    ),
                ]
            )
            print(result_text)
            SenderLayerQmsg.sender_layer_enqueue(result_text)
            print(f"[入队] {rec['symbol']} -> 等待统一发送")
            success_count += 1
        except Exception as exc:
            error_text = "\n".join(
                [
                    "-" * 70,
                    f"[{idx}/{len(records)}] {rec['symbol']} {rec['name']} 持仓={rec['quantity']}",
                    f"执行失败: {rec['symbol']} {rec['name']}, 错误: {exc}",
                ]
            )
            print(error_text)
            SenderLayerQmsg.sender_layer_enqueue(error_text)
            print(f"[入队] {rec['symbol']} -> 等待统一发送")

            concise_records.append(
                {
                    "symbol": rec.get("symbol"),
                    "name": rec.get("name"),
                    "quantity": rec.get("quantity"),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            detailed_records.append(
                {
                    "symbol": rec.get("symbol"),
                    "name": rec.get("name"),
                    "quantity": rec.get("quantity"),
                    "status": "failed",
                    "error": str(exc),
                }
            )

            fail_count += 1

    print("=" * 70)
    print(f"执行完成: 成功 {success_count}，失败 {fail_count}")
    print("=" * 70)

    concise_payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_records": len(records),
        "success_count": success_count,
        "fail_count": fail_count,
        "records": concise_records,
    }
    detailed_payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_records": len(records),
        "success_count": success_count,
        "fail_count": fail_count,
        "scoring_formula": {
            "financial": {
                "valuation": "valuation_score * 0.30",
                    "dividend_spread": "dividend_score * 0.20",
                    "dividend_quality": "dividend_quality_score * 0.10",
                "price_position": "price_position_score * 0.20",
                "position": "position_score * 0.10",
                "roe": "roe_score * 0.10",
                "total": "sum(weighted five factors)",
            },
            "non_financial": {
                "valuation": "valuation_score * 0.30",
                    "dividend_spread": "dividend_score * 0.20",
                    "dividend_quality": "dividend_quality_score * 0.10",
                "price_position": "price_position_score * 0.20",
                "position": "position_score * 0.10",
                "roe": "roe_score * 0.50",
                "cash": "cash_score * 0.50",
                "total": "sum(base four factors + roe*0.5 + cash*0.5)",
            },
        },
        "records": detailed_records,
    }

    concise_saved_path = WriteData.save_json(concise_payload, concise_output_path)
    detailed_saved_path = WriteData.save_json(detailed_payload, detailed_output_path)
    print(f"简略结果已保存: {concise_saved_path}")
    print(f"详细结果已保存: {detailed_saved_path}")




    pending_count = len(SenderLayerQmsg.SENDER_LAYER_PENDING_MESSAGES)
    if pending_count == 0:
        print("没有待发送消息")
    else:
        print(f"开始统一发送，共 {pending_count} 条消息...")
        send_result = SenderLayerQmsg.sender_layer_flush()
        print(f"[已发送] 合并消息 -> {send_result}")


if __name__ == "__main__":
    main()
