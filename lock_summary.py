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

    df["车型分类"] = df[model_group_col].astype(str)
    _cm2_all = df[model_group_col].astype(str).str.upper() == "CM2"
    _is_range_ext_all = df[product_name_col].astype(str).str.contains(r"52|66", case=False, na=False)
    df.loc[_cm2_all & _is_range_ext_all, "车型分类"] = "CM2 增程"
    df.loc[_cm2_all & ~_is_range_ext_all, "车型分类"] = "CM2"

    mask = df[lock_col].notna() & (
        (df[lock_col].dt.date >= start_date) & (df[lock_col].dt.date <= end_date)
    )
    wanted_models = [m.strip() for m in str(args.models).split(",") if m.strip()] if args.models else []
    model_filter = df["车型分类"].isin(wanted_models) if wanted_models else pd.Series(True, index=df.index)
    lock_total = int(df.loc[mask & model_filter, order_no_col].nunique())

    # 2) 大定相关统计（周期内、含大定支付时间）
    summary_df = None
    summary_missing_msg = None
    retained_by_date_total_df = None
    retained_by_date_pivot_df = None
    per_model_df = None
    per_model_pivot_df = None
    try:
        deposit_col = resolve_column(
            df,
            [
                "Deposit_Payment_Time",
                "Deposit Payment Time",
                "deposit_payment_time",
                "大定支付时间",
                "大定时间",
                "下定时间",
            ],
        )
        df[deposit_col] = pd.to_datetime(df[deposit_col], errors="coerce")
        dmask = df[deposit_col].notna() & (
            (df[deposit_col].dt.date >= start_date) & (df[deposit_col].dt.date <= end_date)
        )
        deposit_total = int(df.loc[dmask & model_filter, order_no_col].nunique())

        try:
            refund_col = resolve_column(
                df,
                [
                    "Deposit_Refund_Time",
                    "Deposit Refund Time",
                    "deposit_refund_time",
                    "大定退订时间",
                    "退订时间",
                    "退款时间",
                ],
            )
            df[refund_col] = pd.to_datetime(df[refund_col], errors="coerce")
            deposit_refund_total = int(
                df.loc[dmask & model_filter & df[refund_col].notna(), order_no_col].nunique()
            )
            deposit_retained_total = int(
                df.loc[dmask & model_filter & df[refund_col].isna() & df[lock_col].isna(), order_no_col].nunique()
            )
            _retained_mask = dmask & model_filter & df[refund_col].isna() & df[lock_col].isna()
            _retained_df = df.loc[_retained_mask, [deposit_col, order_no_col, "车型分类"]].copy()
            _retained_df["deposit_date"] = _retained_df[deposit_col].dt.date
            _retained_grp = (
                _retained_df.groupby(["deposit_date", "车型分类"]).agg(retained_orders=(order_no_col, pd.Series.nunique)).reset_index()
            )
            retained_by_date_pivot_df = (
                _retained_grp.pivot_table(index="deposit_date", columns="车型分类", values="retained_orders", fill_value=0, aggfunc="sum").sort_index()
            )
            if wanted_models:
                cols = [m for m in wanted_models if m in retained_by_date_pivot_df.columns]
                if cols:
                    retained_by_date_pivot_df = retained_by_date_pivot_df[cols]
            retained_by_date_total_df = (
                _retained_df.groupby(["deposit_date"]).agg(retained_orders=(order_no_col, pd.Series.nunique)).reset_index().sort_values("deposit_date")
            )

            # 分车型概览统计（仅在指定多个车型时输出）
            if wanted_models and len(wanted_models) >= 2:
                rows = []
                for m in wanted_models:
                    m_filter = df["车型分类"] == m
                    rows.append(
                        {
                            "车型分类": m,
                            "锁单总数": int(df.loc[mask & m_filter, order_no_col].nunique()),
                            "大定总数": int(df.loc[dmask & m_filter, order_no_col].nunique()),
                            "大定留存总数": int(df.loc[dmask & m_filter & df[refund_col].isna() & df[lock_col].isna(), order_no_col].nunique()),
                            "大定退订数": int(df.loc[dmask & m_filter & df[refund_col].notna(), order_no_col].nunique()),
                        }
                    )
                per_model_df = pd.DataFrame(rows)
        except KeyError as e:
            deposit_refund_total = 0
            deposit_retained_total = int(
                df.loc[dmask & model_filter & df[lock_col].isna(), order_no_col].nunique()
            )
            summary_missing_msg = f"字段缺失：{e}"
            _retained_mask = dmask & model_filter & df[lock_col].isna()
            _retained_df = df.loc[_retained_mask, [deposit_col, order_no_col, "车型分类"]].copy()
            _retained_df["deposit_date"] = _retained_df[deposit_col].dt.date
            _retained_grp = (
                _retained_df.groupby(["deposit_date", "车型分类"]).agg(retained_orders=(order_no_col, pd.Series.nunique)).reset_index()
            )
            retained_by_date_pivot_df = (
                _retained_grp.pivot_table(index="deposit_date", columns="车型分类", values="retained_orders", fill_value=0, aggfunc="sum").sort_index()
            )
            if wanted_models:
                cols = [m for m in wanted_models if m in retained_by_date_pivot_df.columns]
                if cols:
                    retained_by_date_pivot_df = retained_by_date_pivot_df[cols]
            retained_by_date_total_df = (
                _retained_df.groupby(["deposit_date"]).agg(retained_orders=(order_no_col, pd.Series.nunique)).reset_index().sort_values("deposit_date")
            )

            if wanted_models and len(wanted_models) >= 2:
                rows = []
                for m in wanted_models:
                    m_filter = df["车型分类"] == m
                    rows.append(
                        {
                            "车型分类": m,
                            "锁单总数": int(df.loc[mask & m_filter, order_no_col].nunique()),
                            "大定总数": int(df.loc[dmask & m_filter, order_no_col].nunique()),
                            "大定留存总数": int(df.loc[dmask & m_filter & df[lock_col].isna(), order_no_col].nunique()),
                            "大定退订数": 0,
                        }
                    )
                per_model_df = pd.DataFrame(rows)

        summary_df = pd.DataFrame(
            {
                "指标": ["锁单总数", "大定总数", "大定留存总数", "大定退订数"],
                "数量": [lock_total, deposit_total, deposit_retained_total, deposit_refund_total],
            }
        )
        if per_model_df is not None and not per_model_df.empty:
            _pivot = per_model_df.set_index("车型分类").T
            _pivot.index.name = "指标"
            per_model_pivot_df = _pivot.reset_index()
    except KeyError as e:
        # 无大定支付时间字段，仅输出锁单总数并提示缺失字段
        summary_missing_msg = f"字段缺失：{e}"
        summary_df = pd.DataFrame({"指标": ["锁单总数"], "数量": [lock_total]})
    df_period = df.loc[mask & model_filter, [lock_col, region_col, model_group_col, product_name_col, order_no_col]].copy()
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
    if per_model_pivot_df is not None:
        md_lines.append("## 概览统计（分车型）")
        md_lines.append(df_to_md(per_model_pivot_df))
        md_lines.append("")
    else:
        md_lines.append("## 概览统计")
        md_lines.append(df_to_md(summary_df))
        if summary_missing_msg:
            md_lines.append("")
            md_lines.append(f"缺失字段提示：{summary_missing_msg}")
        md_lines.append("")
    if wanted_models and len(wanted_models) >= 2:
        md_lines.append("## 大定留存的 Deposit_Payment_Time 分布（按日，分车型）")
        if retained_by_date_pivot_df is not None and not retained_by_date_pivot_df.empty:
            md_lines.append(df_to_md(retained_by_date_pivot_df.reset_index()))
        else:
            md_lines.append("(空表，无数据)")
    else:
        md_lines.append("## 大定留存的 Deposit_Payment_Time 分布（按日）")
        if retained_by_date_total_df is not None and not retained_by_date_total_df.empty:
            md_lines.append(df_to_md(retained_by_date_total_df))
        else:
            md_lines.append("(空表，无数据)")
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
        level_series = df.loc[mask & model_filter, level_col].astype(str).fillna("未知")
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
        prov_series = df.loc[mask & model_filter, prov_col].astype(str).fillna("未知")
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
        city_series = df.loc[mask & model_filter, city_col].astype(str).fillna("未知")
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
