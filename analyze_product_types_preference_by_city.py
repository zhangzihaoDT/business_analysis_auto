#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按 Store City 计算 Product_Types 偏好度（基于 intention_order_analysis.parquet）

功能：
- 读取数据（默认路径：../formatted/intention_order_analysis.parquet）
- 筛选：车型分组 = CM2；Lock_Time ∈ [2025-09-10, 2025-10-15]
- 统计：分 Store City 与 Product_Types 的订单数
- 偏好度系数：同一城市内，每个 Product_Types 的订单数 / 该城市订单数最多的 Product_Types 的订单数
  示例：上海：A=100, B=200 => A 的偏好系数 = 100/200 = 0.5

使用：
python scripts/product_types_preference_by_city.py \
  --data-path ../formatted/intention_order_analysis.parquet \
  --group CM2 --start-date 2025-09-10 --end-date 2025-10-15 \
  --output ../processed/analysis_results/product_types_preference_by_city.csv
"""

import argparse
import os
import re
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "formatted", "intention_order_analysis.parquet")


def normalize_col(name: str) -> str:
    return re.sub(r"[\s_]+", "", str(name).strip().lower())


def resolve_column(df: pd.DataFrame, key: str, candidates_map: Dict[str, List[str]]) -> str:
    """在 candidates_map 中为 key 提供候选列名，返回数据集中真实存在的列名。"""
    norm_to_actual = {normalize_col(c): c for c in df.columns}
    for cand in candidates_map.get(key, []):
        norm_cand = normalize_col(cand)
        if norm_cand in norm_to_actual:
            return norm_to_actual[norm_cand]
    raise KeyError(f"未在数据集中找到需要的字段: {key}. 可用字段: {list(df.columns)}")


def safe_fillna_text(series: pd.Series, unknown_label: str = "未知") -> pd.Series:
    """在可能为分类类型的列上安全填充文本缺失值。"""
    try:
        if pd.api.types.is_categorical_dtype(series):
            if unknown_label not in series.cat.categories:
                series = series.cat.add_categories([unknown_label])
            return series.fillna(unknown_label)
        else:
            return series.fillna(unknown_label)
    except Exception:
        s = series.astype(str)
        return s.replace({"nan": unknown_label})


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    return pd.read_parquet(path)


def get_product_type_from_name(product_name: str) -> str:
    """根据 Product Name 派生产品类型（增程/纯电），无法识别返回“未知”。"""
    try:
        if product_name is None:
            return "未知"
        # 处理 NA 与字符串
        s = str(product_name).strip()
        if len(s) == 0 or s.lower() in {"nan", "none", "null"}:
            return "未知"

        # 规则：含“52”或“66”视为增程，否则纯电
        # 有些命名包含“新一代”，但本质判断仍以电池容量数字为主
        if any(num in s for num in ["52", "66"]):
            return "增程"
        else:
            return "纯电"
    except Exception:
        return "未知"


def filter_by_group_and_lock_time(
    df: pd.DataFrame,
    model_group_value: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """按车型分组与 Lock_Time 时间范围进行筛选（包含边界）。"""
    candidates = {
        "model_group": [
            "车型分组",
            "model_group",
            "car_model_group",
            "车型",
            "modelgroup",
        ],
        "lock_time": [
            "Lock_Time",
            "Lock Time",
            "lock_time",
            "锁单时间",
            "锁定时间",
        ],
    }

    col_model = resolve_column(df, "model_group", candidates)
    col_lock = resolve_column(df, "lock_time", candidates)

    df = df[df[col_model].astype(str) == str(model_group_value)].copy()

    s_lock = pd.to_datetime(df[col_lock], errors="coerce")
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(s_lock >= start) & (s_lock <= end)].copy()
    return df


def filter_by_group_and_intention_time(
    df: pd.DataFrame,
    model_group_value: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """按车型分组与 Intention Payment Time 时间范围进行筛选（包含边界）。"""
    candidates = {
        "model_group": [
            "车型分组",
            "model_group",
            "car_model_group",
            "车型",
            "modelgroup",
        ],
        "intention_payment_time": [
            "Intention Payment Time",
            "意向支付时间",
            "intention_payment_time",
            "intent_payment_time",
            "intentpaytime",
        ],
    }

    col_model = resolve_column(df, "model_group", candidates)
    col_intent = resolve_column(df, "intention_payment_time", candidates)

    df = df[df[col_model].astype(str) == str(model_group_value)].copy()

    s_intent = pd.to_datetime(df[col_intent], errors="coerce")
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(s_intent >= start) & (s_intent <= end)].copy()
    return df


def compute_preference_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """分 Store City 统计 Product_Types 订单数与偏好度系数。"""
    candidates = {
        "store_city": [
            "Store City",
            "Store_City",
            "门店城市",
            "城市",
            "City",
        ],
        "product_types": [
            "Product_Types",
            "Product Types",
            "产品类型",
            "product_types",
            # 兼容数据集中不存在 Product_Types 时的备选字段
            "Product Name",
            "产品名称",
        ],
    }

    col_city = resolve_column(df, "store_city", candidates)
    # 先尝试解析产品类型列；若解析得到的是“Product Name”，则根据名称派生为“增程/纯电”类别
    col_pt = resolve_column(df, "product_types", candidates)

    s_city = safe_fillna_text(df[col_city], "未知")
    if normalize_col(col_pt) in {normalize_col("Product Name"), normalize_col("产品名称")}:
        type_series = pd.Series([get_product_type_from_name(x) for x in df[col_pt]], index=df.index)
        s_pt = safe_fillna_text(type_series, "未知")
        # 输出列名统一为“Product_Type”
        pt_col_name = "Product_Type"
    else:
        s_pt = safe_fillna_text(df[col_pt], "未知")
        pt_col_name = col_pt

    grp = (
        pd.DataFrame({col_city: s_city, pt_col_name: s_pt})
        .groupby([col_city, pt_col_name], dropna=False)
        .size()
        .reset_index(name="order_count")
    )

    # 每城总订单与最大订单数（用于计算偏好度系数与占比）
    grp["city_total"] = grp.groupby(col_city)["order_count"].transform("sum")
    grp["city_max"] = grp.groupby(col_city)["order_count"].transform("max")
    grp["preference_coef"] = (grp["order_count"] / grp["city_max"]).astype(float)
    grp["share_pct"] = (grp["order_count"] / grp["city_total"] * 100).round(2)

    # 排序：城市内按订单数降序
    grp = grp.sort_values([col_city, "order_count"], ascending=[True, False]).reset_index(drop=True)
    return grp[[col_city, pt_col_name, "order_count", "preference_coef", "share_pct"]]


def ensure_output_dir(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="分城市计算产品类型偏好度")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="数据文件路径（Parquet）")
    parser.add_argument("--group", default="CM2", help="车型分组，默认 CM2")
    parser.add_argument("--start-date", default="2025-09-10", help="Lock_Time 开始日期（默认 2025-09-10）")
    parser.add_argument("--end-date", default="2025-10-15", help="Lock_Time 结束日期（默认 2025-10-15）")
    parser.add_argument("--output", default=None, help="结果输出 CSV 路径（可选）")
    parser.add_argument(
        "--ratio-output",
        default="./processed/analysis_results/cm2_city_range_ev_counts.csv",
        help="城市增程/纯电数量与比值输出 CSV 路径（可选）",
    )
    parser.add_argument(
        "--combined-output",
        default="./processed/analysis_results/city_cm2_range_ev_ls9_orders.csv",
        help="合并表输出：License City, CM2增程订单数, CM2纯电订单数, 增程/纯电比值, LS9 小订数",
    )
    parser.add_argument(
        "--underperf-csv",
        default="./processed/analysis_results/ls9_underperformance_big_cities.csv",
        help="大盘城市LS9相对不足榜 CSV 输出",
    )
    parser.add_argument(
        "--underperf-html",
        default="./reports/LS9_大盘城市线性残差榜.html",
        help="大盘城市LS9相对不足榜 HTML 输出",
    )

    args = parser.parse_args(argv)

    df = load_data(args.data_path)
    df_filt = filter_by_group_and_lock_time(df, args.group, args.start_date, args.end_date)

    if len(df_filt) == 0:
        print("⚠️ 筛选后数据为空，请检查车型分组与日期范围。")
        return 0

    res = compute_preference_by_city(df_filt)

    # 打印前若干行示例
    print("=== 分城市的产品类型订单数与偏好度 ===")
    print(res.head(30).to_string(index=False))

    # 若存在“增程/纯电”类别，额外输出每个城市的“增程/纯电”比值
    type_col_candidates = [c for c in res.columns if normalize_col(c) in {normalize_col("Product_Type"), normalize_col("Product_Types"), normalize_col("产品类型")}]
    city_col_candidates = [c for c in res.columns if normalize_col(c) in {normalize_col("Store City"), normalize_col("store_city"), normalize_col("门店城市"), normalize_col("城市"), normalize_col("City")}]
    if type_col_candidates and city_col_candidates:
        tcol = type_col_candidates[0]
        ccol = city_col_candidates[0]
        pivot = res.pivot_table(index=ccol, columns=tcol, values="order_count", aggfunc="sum", fill_value=0)
        if any(k in pivot.columns for k in ["增程", "纯电"]):
            range_counts = pivot["增程"] if "增程" in pivot.columns else pd.Series(0, index=pivot.index)
            ev_counts = pivot["纯电"] if "纯电" in pivot.columns else pd.Series(0, index=pivot.index)
            # 转为浮点并处理分母为0的情况
            numer = pd.to_numeric(range_counts, errors="coerce").astype(float)
            denom = pd.to_numeric(ev_counts, errors="coerce").astype(float)
            denom = denom.replace(0.0, np.nan)
            ratio = (numer / denom).round(4)
            ratio_df = pd.DataFrame({
                ccol: pivot.index,
                "增程订单数": range_counts,
                "纯电订单数": ev_counts,
                "增程/纯电比值": ratio,
            })
            ratio_df = ratio_df.reset_index(drop=True)
            print("\n=== 每城市 增程/纯电 订单比值 ===")
            print(ratio_df.head(30).to_string(index=False))
            # 保存增程/纯电数量与比值
            if args.ratio_output:
                out_ratio = ensure_output_dir(args.ratio_output)
                if out_ratio:
                    ratio_to_save = ratio_df.copy()
                    # 统一城市列名
                    if ccol != "Store City":
                        ratio_to_save = ratio_to_save.rename(columns={ccol: "Store City"})
                    ratio_to_save.to_csv(out_ratio, index=False)
                    print(f"✅ 已保存增程/纯电数量与比值到: {out_ratio}")

    # 保存结果（可选）
    out_path = ensure_output_dir(args.output)
    if out_path:
        res.to_csv(out_path, index=False)
        print(f"✅ 已保存结果到: {out_path}")

    # 生成新数据集：筛选车型分组=LS9，意向支付时间范围，分 Store City 的订单数，并合并增程/纯电比值
    target_group = os.environ.get("TARGET_GROUP", "LS9")  # 允许通过环境变量覆盖，默认 LS9
    target_start = os.environ.get("INTENTION_START_DATE", "2025-11-04")
    target_end = os.environ.get("INTENTION_END_DATE", "2025-11-09")

    df_target = filter_by_group_and_intention_time(df, target_group, target_start, target_end)

    # 统计目标数据集的城市订单数
    city_candidates = {
        "store_city": [
            "Store City",
            "Store_City",
            "门店城市",
            "城市",
            "City",
        ],
    }
    city_col = resolve_column(df_target, "store_city", city_candidates)
    s_city_t = safe_fillna_text(df_target[city_col], "未知")
    city_counts = (
        pd.DataFrame({city_col: s_city_t})
        .groupby(city_col, dropna=False)
        .size()
        .reset_index(name="orders_count")
        .sort_values("orders_count", ascending=False)
    )

    # 与比值表合并
    # 从上面打印用的 ratio_df 复用；若未生成则重新计算
    try:
        ratio_df  # type: ignore[name-defined]
    except NameError:
        # 重新计算比值，沿用当前脚本的筛选（group/start/end）
        ratio_source = compute_preference_by_city(df_filt)
        tcol = [c for c in ratio_source.columns if normalize_col(c) in {normalize_col("Product_Type"), normalize_col("Product_Types"), normalize_col("产品类型")}][0]
        ccol = [c for c in ratio_source.columns if normalize_col(c) in {normalize_col("Store City"), normalize_col("store_city"), normalize_col("门店城市"), normalize_col("城市"), normalize_col("City")}][0]
        pivot = ratio_source.pivot_table(index=ccol, columns=tcol, values="order_count", aggfunc="sum", fill_value=0)
        range_counts = pivot["增程"] if "增程" in pivot.columns else pd.Series(0, index=pivot.index)
        ev_counts = pivot["纯电"] if "纯电" in pivot.columns else pd.Series(0, index=pivot.index)
        numer = pd.to_numeric(range_counts, errors="coerce").astype(float)
        denom = pd.to_numeric(ev_counts, errors="coerce").astype(float)
        denom = denom.replace(0.0, np.nan)
        ratio_vals = (numer / denom).round(4)
        ratio_df = pd.DataFrame({
            ccol: pivot.index,
            "增程/纯电比值": ratio_vals,
        }).reset_index(drop=True)

    merged = city_counts.merge(ratio_df, left_on=city_col, right_on=ratio_df.columns[0], how="left")
    merged = merged[[city_col, "orders_count", "增程/纯电比值"]]

    # 输出与保存
    print("\n=== 新数据集：LS9意向支付范围内城市订单数 + 比值 ===")
    print(merged.head(50).to_string(index=False))

    out_new = ensure_output_dir(os.environ.get("OUTPUT_NEW", "./processed/analysis_results/store_city_ls9_intent_counts_with_ratio.csv"))
    if out_new:
        merged.to_csv(out_new, index=False)
        print(f"✅ 已保存新数据集到: {out_new}")

    # 直接输出合并表：CM2增程/纯电数量与比值 + LS9小订数
    try:
        ratio_df  # type: ignore[name-defined]
        # 统一列名与选择需要的列
        cm2_df = ratio_df.copy()
        cm2_df = cm2_df.rename(columns={
            cm2_df.columns[0]: "Store City",
            "增程订单数": "CM2增程订单数",
            "纯电订单数": "CM2纯电订单数",
        })
        ls9_counts = city_counts.copy()
        ls9_counts = ls9_counts.rename(columns={city_col: "Store City", "orders_count": "LS9 小订数"})

        combined = cm2_df.merge(ls9_counts, on="Store City", how="inner")
        # 列顺序
        combined = combined[["Store City", "CM2增程订单数", "CM2纯电订单数", "增程/纯电比值", "LS9 小订数"]]

        print("\n=== 城市 CM2增程/纯电数量与比值 + LS9小订数 ===")
        print(combined.head(30).to_string(index=False))

        out_combined = ensure_output_dir(args.combined_output)
        if out_combined:
            combined.to_csv(out_combined, index=False)
            print(f"✅ 已保存合并表到: {out_combined}")
    except NameError:
        print("⚠️ 未能生成合并表：缺少增程/纯电数量与比值数据。请确认产品类型列可解析为增程/纯电。")

    # 生成大盘城市线性残差榜（统一字段）
    try:
        # 构建统一字段的全量城市数据
        full_df = cm2_df.merge(ls9_counts, on="Store City", how="inner").copy()
        full_df = full_df.rename(columns={
            "CM2增程订单数": "cm2_range",
            "CM2纯电订单数": "cm2_ev",
            "增程/纯电比值": "ratio",
            "LS9 小订数": "ls9_orders",
        })
        # 数值化
        for c in ["cm2_range", "cm2_ev", "ratio", "ls9_orders"]:
            full_df[c] = pd.to_numeric(full_df[c], errors="coerce")

        # 基线拟合：过滤 cm2_range>=30 且 ls9_orders 有值
        base = full_df[(full_df["cm2_range"] >= 30) & full_df["ls9_orders"].notna()].copy()
        x = base["cm2_range"].values
        y = base["ls9_orders"].values
        if len(base) >= 2:
            b, a = np.polyfit(x, y, 1)
        else:
            b, a = 0.0, float(y.mean()) if len(y) else 0.0

        full_df["ls9_to_range"] = (full_df["ls9_orders"] / full_df["cm2_range"]).replace([np.inf, -np.inf], np.nan)
        full_df["pred_ls9"] = a + b * full_df["cm2_range"]
        full_df["resid"] = full_df["ls9_orders"] - full_df["pred_ls9"]
        full_df["resid_pct"] = (full_df["ls9_orders"] / full_df["pred_ls9"]).replace([np.inf, -np.inf], np.nan)

        # 大盘筛选：cm2_range>=100，按残差升序（不足优先）
        big = full_df[full_df["cm2_range"] >= 100].copy().sort_values("resid")
        out_cols = [
            "Store City", "cm2_range", "cm2_ev", "ratio",
            "ls9_orders", "ls9_to_range", "pred_ls9", "resid", "resid_pct",
        ]

        # 输出CSV
        out_under_csv = ensure_output_dir(args.underperf_csv)
        if out_under_csv:
            big[out_cols].to_csv(out_under_csv, index=False)
            print(f"✅ 已保存大盘城市线性残差榜CSV到: {out_under_csv}")

        # 输出HTML
        # 构建简洁HTML表格，保留同字段
        html_preview = big[out_cols].head(50).copy()
        # 格式化数值
        for c in ["cm2_range", "cm2_ev", "ls9_orders", "pred_ls9", "resid"]:
            html_preview[c] = html_preview[c].map(lambda v: f"{float(v):.0f}" if pd.notna(v) else "")
        for c in ["ratio", "ls9_to_range", "resid_pct"]:
            html_preview[c] = html_preview[c].map(lambda v: f"{float(v):.3f}" if pd.notna(v) else "")

        head_html = (
            f"<h2>LS9 大盘城市线性残差榜</h2>"
            f"<p>线性拟合：LS9 = {a:.3f} + {b:.3f} × CM2增程（样本：{len(base)} 城市，门槛：CM2增程≥30）；"
            f"大盘筛选：CM2增程≥100。字段：{', '.join(out_cols)}。</p>"
        )
        html_table = html_preview.to_html(index=False, escape=False)

        out_under_html = args.underperf_html
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(out_under_html)), exist_ok=True)
        with open(out_under_html, "w", encoding="utf-8") as f:
            f.write(
                """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8" />
<title>LS9 大盘城市线性残差榜</title>
<style>
body { font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,'Noto Sans','Microsoft YaHei',sans-serif; margin: 24px; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ddd; padding: 8px; }
th { background: #f5f5f5; }
</style>
</head>
<body>
"""
                + head_html
                + html_table
                + "\n</body>\n</html>"
            )
        print(f"✅ 已保存大盘城市线性残差榜HTML到: {out_under_html}")
    except Exception as e:
        print(f"⚠️ 生成大盘城市线性残差榜时出错: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
