#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按 License City 计算平均开票价格（基于 intention_order_analysis.parquet）

功能：
- 读取数据（默认路径：../formatted/intention_order_analysis.parquet）
- 筛选：车型分组 = CM2；Invoice_Upload_Time ∈ [2025-09-10, 2025-11-09]
- 统计：分 License City 的开票订单数与平均开票价格

使用：
python scripts/invoice_avg_price_by_city.py \
  --data-path ../formatted/intention_order_analysis.parquet \
  --group CM2 --start-date 2025-09-10 --end-date 2025-11-09 \
  --output ./processed/analysis_results/invoice_avg_price_by_city.csv

示例（全车型）：
python scripts/invoice_avg_price_by_city.py \
  --data-path ../formatted/intention_order_analysis.parquet \
  --group ALL --start-date 2025-09-10 --end-date 2025-11-09 \
  --output ./processed/analysis_results/invoice_avg_price_by_city_all_models.csv
"""

import argparse
import os
import re
from typing import Dict, List, Optional

import pandas as pd


DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "formatted", "intention_order_analysis.parquet")


def normalize_col(name: str) -> str:
    return re.sub(r"[\s_]+", "", str(name).strip().lower())


def resolve_column(df: pd.DataFrame, key: str, candidates_map: Dict[str, List[str]]) -> str:
    norm_to_actual = {normalize_col(c): c for c in df.columns}
    for cand in candidates_map.get(key, []):
        norm_cand = normalize_col(cand)
        if norm_cand in norm_to_actual:
            return norm_to_actual[norm_cand]
    raise KeyError(f"未在数据集中找到需要的字段: {key}. 可用字段: {list(df.columns)}")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    return pd.read_parquet(path)


def filter_by_group_and_invoice_time(
    df: pd.DataFrame,
    model_group_value: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """按车型分组与开票上传时间范围进行筛选（包含边界）。"""
    candidates = {
        "model_group": [
            "车型分组",
            "model_group",
            "car_model_group",
            "车型",
            "modelgroup",
        ],
        "invoice_time": [
            "Invoice_Upload_Time",
            "Invoice Upload Time",
            "invoice_upload_time",
            "开票时间",
            "开票上传时间",
        ],
    }

    # 当 group 指定为 ALL/全部/* 时，不进行车型分组过滤
    is_all = str(model_group_value).strip().lower() in {"all", "全部", "*"}
    col_model = None if is_all else resolve_column(df, "model_group", candidates)
    col_invoice = resolve_column(df, "invoice_time", candidates)

    if col_model is not None:
        df = df[df[col_model].astype(str) == str(model_group_value)].copy()

    s_invoice = pd.to_datetime(df[col_invoice], errors="coerce")
    if is_all:
        # 全车型：仅保留有开票上传时间的记录，不做日期范围过滤
        df = df[s_invoice.notna()].copy()
    else:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        df = df[(s_invoice >= start) & (s_invoice <= end)].copy()
    return df


def compute_city_invoice_stats(df: pd.DataFrame) -> pd.DataFrame:
    """分 License City 统计开票订单数与平均开票价格。"""
    candidates = {
        "license_city": [
            "License City",
            "license_city",
            "上牌城市",
            "上牌市",
        ],
        "invoice_price": [
            "开票价格",
            "Invoice Price",
            "invoice_price",
            "开票金额",
            "invoice_amount",
        ],
    }

    col_city = resolve_column(df, "license_city", candidates)
    col_price = resolve_column(df, "invoice_price", candidates)

    # 转数值，忽略无法解析的价格
    price = pd.to_numeric(df[col_price], errors="coerce")

    # 聚合：订单数（city内记录数），平均价格（忽略NaN）
    res = (
        pd.DataFrame({col_city: df[col_city], "price": price})
        .groupby(col_city, dropna=False)
        .agg(invoice_orders=("price", "size"), avg_invoice_price=("price", "mean"))
        .reset_index()
    )

    # 四舍五入到整数
    res["avg_invoice_price"] = res["avg_invoice_price"].round(0)
    # 排序：按订单数降序
    res = res.sort_values(["invoice_orders", "avg_invoice_price"], ascending=[False, False])
    return res


def ensure_output_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="分城市计算平均开票价格")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="数据文件路径（Parquet）")
    parser.add_argument("--group", default="CM2", help="车型分组，默认 CM2")
    parser.add_argument("--start-date", default="2025-09-10", help="开票上传开始日期（默认 2025-09-10）")
    parser.add_argument("--end-date", default="2025-11-09", help="开票上传结束日期（默认 2025-11-09）")
    parser.add_argument("--output", default="./processed/analysis_results/invoice_avg_price_by_city.csv", help="结果输出 CSV 路径")

    args = parser.parse_args(argv)

    df = load_data(args.data_path)
    df_filt = filter_by_group_and_invoice_time(df, args.group, args.start_date, args.end_date)

    if len(df_filt) == 0:
        print("⚠️ 筛选后数据为空，请检查车型分组与日期范围。")
        return 0

    res = compute_city_invoice_stats(df_filt)

    grp_label = args.group if str(args.group).strip().lower() not in {"all", "全部", "*"} else "ALL(全部车型)"
    print(f"=== 分城市的开票订单数与平均开票价格（车型分组：{grp_label}） ===")
    # 打印前 30 行示例（平均价格显示为整数）
    preview = res.head(30).copy()
    preview["avg_invoice_price"] = preview["avg_invoice_price"].astype("Int64")
    print(preview.rename(columns={
        preview.columns[0]: "License City",
        "invoice_orders": "开票订单数",
        "avg_invoice_price": "平均开票价格",
    }).to_string(index=False))

    out_path = ensure_output_path(args.output)
    if out_path:
        res.to_csv(out_path, index=False)
        print(f"✅ 已保存结果到: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())