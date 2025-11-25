import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_INPUT = Path(
    "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"
)
OUT_DIR = Path(
    "/Users/zihao_/Documents/coding/dataset/processed/analysis_results"
)


def normalize(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace(" ", " ")


def resolve_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for cand in candidates:
        if cand in df.columns:
            return cand
    cand_norm = [normalize(c) for c in candidates]
    col_norm_map = {normalize(c): c for c in df.columns}
    for cn in cand_norm:
        if cn in col_norm_map:
            return col_norm_map[cn]
    raise KeyError(
        f"Unable to resolve column from candidates: {candidates}. Available columns: {list(df.columns)}"
    )


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(空表，无数据)"
    df_fmt = df.copy()
    return df_fmt.to_markdown(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="锁单数分车型分大区汇总"
    )
    parser.add_argument(
        "--start",
        required=True,
        help="开始日期，格式 YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="结束日期，格式 YYYY-MM-DD",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="输入 Parquet 文件路径（默认为 intention_order_analysis.parquet）",
    )
    parser.add_argument(
        "--models",
        help="逗号分隔的车型列表，仅显示这些车型列，例如: 'CM2,CM2 增程,LS9'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)

    lock_col = resolve_column(
        df,
        [
            "Lock_Time",
            "Lock Time",
            "lock_time",
            "lock time",
            "LockTime",
            "锁单时间",
            "LockDate",
        ],
    )
    region_col = resolve_column(
        df,
        [
            "Parent Region Name",
            "Parent_Region_Name",
            "Parent Region",
            "Region Name",
            "区域",
            "大区",
        ],
    )
    model_group_col = resolve_column(
        df,
        [
            "车型分组",
            "Model Group",
            "Vehicle Group",
            "Car Group",
            "车型",
        ],
    )
    product_name_col = resolve_column(
        df,
        [
            "productname",
            "ProductName",
            "Product Name",
            "产品名称",
            "商品名称",
        ],
    )
    order_no_col = resolve_column(
        df,
        [
            "Order Number",
            "Order_Number",
            "OrderNo",
            "Order No",
            "订单编号",
            "订单号",
            "意向单号",
            "Intention Order Number",
        ],
    )

    df[lock_col] = pd.to_datetime(df[lock_col], errors="coerce")

    mask = df[lock_col].notna() & (
        (df[lock_col].dt.date >= start_date) & (df[lock_col].dt.date <= end_date)
    )
    df_period = df.loc[mask, [lock_col, region_col, model_group_col, product_name_col, order_no_col]].copy()
    df_period["车型分类"] = df_period[model_group_col].astype(str)
    cm2_mask = df_period[model_group_col].astype(str).str.upper() == "CM2"
    is_range_ext = df_period[product_name_col].astype(str).str.contains(r"52|66", case=False, na=False)
    df_period.loc[cm2_mask & is_range_ext, "车型分类"] = "CM2 增程"
    df_period.loc[cm2_mask & ~is_range_ext, "车型分类"] = "CM2"

    grouped = (
        df_period.groupby([region_col, "车型分类"]).agg(订单数=(order_no_col, pd.Series.nunique)).reset_index()
    )
    grouped.rename(columns={region_col: "Parent Region Name"}, inplace=True)
    pivot_df = grouped.pivot_table(
        index="Parent Region Name",
        columns="车型分类",
        values="订单数",
        fill_value=0,
        aggfunc="sum",
    )
    pivot_df = pivot_df.sort_index().sort_index(axis=1)

    if args.models:
        wanted = [m.strip() for m in str(args.models).split(",") if m.strip()]
        cols = [c for c in wanted if c in pivot_df.columns]
        if cols:
            pivot_df = pivot_df[cols]

    percent_df = pivot_df.div(pivot_df.sum(axis=0).replace(0, pd.NA), axis=1).fillna(0) * 100
    percent_df = percent_df.round(2)

    md_lines = []
    md_lines.append("# 锁单分车型分大区汇总")
    md_lines.append("")
    md_lines.append(f"- 源文件: `{input_path}`")
    md_lines.append(f"- 时间区间: `{args.start}` ~ `{args.end}`")
    md_lines.append("")
    md_lines.append("## 区域 x 车型矩阵")
    md_lines.append(df_to_md(pivot_df.reset_index()))
    md_lines.append("")
    md_lines.append("## 分 region 占比（%）（按车型列归一化）")
    md_lines.append(df_to_md(percent_df.reset_index()))

    try:
        level_col = resolve_column(
            df,
            [
                "License City Level",
                "license_city_level",
                "License City Tier",
                "city_level",
                "City Level",
                "城市级别",
                "城市等级",
                "上牌城市级别",
                "上牌城市等级",
            ],
        )
        level_series = df.loc[mask, level_col].astype(str).fillna("未知")
        level_counts = level_series.value_counts()
        total_orders = int(level_series.size)
        level_share = (level_counts / max(total_orders, 1) * 100).round(2)
        level_df = (
            pd.DataFrame({"level": level_counts.index, "lock_orders": level_counts.values, "share_pct": level_share.values})
            .sort_values(["lock_orders", "share_pct"], ascending=[False, False])
        )
        md_lines.append("")
        md_lines.append("## 分 license_city_level 的锁单量与占比")
        md_lines.append(df_to_md(level_df))
    except KeyError as e:
        md_lines.append("")
        md_lines.append("## 分 license_city_level 的锁单量与占比")
        md_lines.append(f"字段缺失：{e}")

    # 追加：分 License Province 的锁单量与占比（Top 10）
    try:
        prov_col = resolve_column(
            df,
            [
                "License Province",
                "license_province",
                "License Province Name",
                "Province",
                "province",
                "上牌省份",
                "上牌省",
                "省份",
            ],
        )
        prov_series = df.loc[mask, prov_col].astype(str).fillna("未知")
        prov_counts = prov_series.value_counts()
        total_orders_p = int(prov_series.size)
        prov_share = (prov_counts / max(total_orders_p, 1) * 100).round(2)
        prov_df = (
            pd.DataFrame({"province": prov_counts.index, "lock_orders": prov_counts.values, "share_pct": prov_share.values})
            .sort_values(["lock_orders", "share_pct"], ascending=[False, False])
            .head(10)
        )
        md_lines.append("")
        md_lines.append("## 分 License Province 的锁单量与占比（Top 10）")
        md_lines.append(df_to_md(prov_df))
    except KeyError as e:
        md_lines.append("")
        md_lines.append("## 分 License Province 的锁单量与占比（Top 10）")
        md_lines.append(f"字段缺失：{e}")

    # 追加：分 License City 的锁单量与占比（Top 10）
    try:
        city_col = resolve_column(
            df,
            [
                "License City",
                "license_city",
                "License City Name",
                "Store City",
                "store_city",
                "城市",
                "上牌城市",
                "上牌市",
            ],
        )
        city_series = df.loc[mask, city_col].astype(str).fillna("未知")
        city_counts = city_series.value_counts()
        total_orders_c = int(city_series.size)
        city_share = (city_counts / max(total_orders_c, 1) * 100).round(2)
        city_df = (
            pd.DataFrame({"city": city_counts.index, "lock_orders": city_counts.values, "share_pct": city_share.values})
            .sort_values(["lock_orders", "share_pct"], ascending=[False, False])
            .head(10)
        )
        md_lines.append("")
        md_lines.append("## 分 License City 的锁单量与占比（Top 10）")
        md_lines.append(df_to_md(city_df))
    except KeyError as e:
        md_lines.append("")
        md_lines.append("## 分 License City 的锁单量与占比（Top 10）")
        md_lines.append(f"字段缺失：{e}")

    report_name = f"lock_summary_{args.start}_to_{args.end}.md"
    report_path = OUT_DIR / report_name
    report_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Report saved: {report_path}")
    print("\n预览：")
    print(df_to_md(pivot_df.reset_index()))
    print("\n占比预览：")
    print(df_to_md(percent_df.reset_index()))


if __name__ == "__main__":
    main()
