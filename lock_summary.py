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


BUSINESS_GROUPS = {
    "LS6纯电": ["CM0", "CM1", "CM2"],
    "LS6增程": ["CM2 增程"],
    "L6": ["DM0", "DM1"],
}

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
    
    # 构建反向映射: 车型 -> 业务组名
    model_to_biz = {}
    for grp_name, models in BUSINESS_GROUPS.items():
        for m in models:
            model_to_biz[m] = grp_name
            
    df["业务定义"] = df["车型分类"].map(model_to_biz).fillna("其他")

    mask = df[lock_col].notna() & (
        (df[lock_col].dt.date >= start_date) & (df[lock_col].dt.date <= end_date)
    )
    wanted_tokens = [m.strip() for m in str(args.models).split(",") if m.strip()] if args.models else []
    requested_biz = [t for t in wanted_tokens if t in BUSINESS_GROUPS]
    requested_models = [t for t in wanted_tokens if t not in BUSINESS_GROUPS]
    expanded_models: List[str] = []
    for t in requested_models:
        expanded_models.append(t)
    for t in requested_biz:
        expanded_models.extend(BUSINESS_GROUPS[t])
    biz_mode = True if requested_biz and not requested_models else False
    model_filter_series = df["车型分类"].isin(expanded_models) if expanded_models else pd.Series(True, index=df.index)
    biz_filter_series = df["业务定义"].isin(requested_biz) if biz_mode else pd.Series(True, index=df.index)
    active_filter = model_filter_series & biz_filter_series
    lock_total = int(df.loc[mask & active_filter, order_no_col].nunique())

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
        deposit_total = int(df.loc[dmask & active_filter, order_no_col].nunique())

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
                df.loc[dmask & active_filter & df[refund_col].notna(), order_no_col].nunique()
            )
            deposit_retained_total = int(
                df.loc[dmask & active_filter & df[refund_col].isna() & df[lock_col].isna(), order_no_col].nunique()
            )
            _retained_mask = dmask & active_filter & df[refund_col].isna() & df[lock_col].isna()
            _retained_df = df.loc[_retained_mask, [deposit_col, order_no_col, "车型分类"]].copy()
            _retained_df["deposit_date"] = _retained_df[deposit_col].dt.date
            _retained_grp = (
                _retained_df.groupby(["deposit_date", "车型分类"]).agg(retained_orders=(order_no_col, pd.Series.nunique)).reset_index()
            )
            retained_by_date_pivot_df = (
                _retained_grp.pivot_table(index="deposit_date", columns="车型分类", values="retained_orders", fill_value=0, aggfunc="sum", observed=False).sort_index()
            )
            if expanded_models:
                cols = [m for m in expanded_models if m in retained_by_date_pivot_df.columns]
                if cols:
                    retained_by_date_pivot_df = retained_by_date_pivot_df[cols]
            retained_by_date_total_df = (
                _retained_df.groupby(["deposit_date"]).agg(retained_orders=(order_no_col, pd.Series.nunique)).reset_index().sort_values("deposit_date")
            )

            # 分车型概览统计（仅在指定多个车型时输出）
            if (not biz_mode) and (requested_models and len(requested_models) >= 2):
                rows = []
                for m in requested_models:
                    m_filter = df["车型分类"] == m
                    rows.append(
                        {
                            "车型分类": m,
                            "锁单总数": int(df.loc[mask & active_filter & m_filter, order_no_col].nunique()),
                            "大定总数": int(df.loc[dmask & active_filter & m_filter, order_no_col].nunique()),
                            "大定留存总数": int(df.loc[dmask & active_filter & m_filter & df[refund_col].isna() & df[lock_col].isna(), order_no_col].nunique()),
                            "大定退订数": int(df.loc[dmask & active_filter & m_filter & df[refund_col].notna(), order_no_col].nunique()),
                        }
                    )
                per_model_df = pd.DataFrame(rows)
        except KeyError as e:
            deposit_refund_total = 0
            deposit_retained_total = int(
                df.loc[dmask & active_filter & df[lock_col].isna(), order_no_col].nunique()
            )
            summary_missing_msg = f"字段缺失：{e}"
            _retained_mask = dmask & active_filter & df[lock_col].isna()
            _retained_df = df.loc[_retained_mask, [deposit_col, order_no_col, "车型分类"]].copy()
            _retained_df["deposit_date"] = _retained_df[deposit_col].dt.date
            _retained_grp = (
                _retained_df.groupby(["deposit_date", "车型分类"]).agg(retained_orders=(order_no_col, pd.Series.nunique)).reset_index()
            )
            retained_by_date_pivot_df = (
                _retained_grp.pivot_table(index="deposit_date", columns="车型分类", values="retained_orders", fill_value=0, aggfunc="sum").sort_index()
            )
            if expanded_models:
                cols = [m for m in expanded_models if m in retained_by_date_pivot_df.columns]
                if cols:
                    retained_by_date_pivot_df = retained_by_date_pivot_df[cols]
            retained_by_date_total_df = (
                _retained_df.groupby(["deposit_date"]).agg(retained_orders=(order_no_col, pd.Series.nunique)).reset_index().sort_values("deposit_date")
            )

            if (not biz_mode) and (requested_models and len(requested_models) >= 2):
                rows = []
                for m in requested_models:
                    m_filter = df["车型分类"] == m
                    rows.append(
                        {
                            "车型分类": m,
                            "锁单总数": int(df.loc[mask & active_filter & m_filter, order_no_col].nunique()),
                            "大定总数": int(df.loc[dmask & active_filter & m_filter, order_no_col].nunique()),
                            "大定留存总数": int(df.loc[dmask & active_filter & m_filter & df[lock_col].isna(), order_no_col].nunique()),
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
    df_period = df.loc[mask & active_filter, [lock_col, region_col, model_group_col, product_name_col, order_no_col]].copy()
    df_period["车型分类"] = df_period[model_group_col].astype(str)
    cm2_mask = df_period[model_group_col].astype(str).str.upper() == "CM2"
    is_range_ext = df_period[product_name_col].astype(str).str.contains(r"52|66", case=False, na=False)
    df_period.loc[cm2_mask & is_range_ext, "车型分类"] = "CM2 增程"
    df_period.loc[cm2_mask & ~is_range_ext, "车型分类"] = "CM2"
    df_period["业务定义"] = df_period["车型分类"].map(model_to_biz).fillna("其他")

    grouped = (
        df_period.groupby([region_col, "车型分类"], observed=False).agg(订单数=(order_no_col, pd.Series.nunique)).reset_index()
    )
    grouped.rename(columns={region_col: "Parent Region Name"}, inplace=True)
    pivot_df = grouped.pivot_table(
        index="Parent Region Name",
        columns="车型分类",
        values="订单数",
        fill_value=0,
        aggfunc="sum",
        observed=False,
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
    if (not biz_mode) and (requested_models and len(requested_models) >= 2):
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
    preview_pivot = None
    preview_percent = None
    if biz_mode:
        grouped_biz = (
            df_period.groupby([region_col, "业务定义"], observed=False).agg(订单数=(order_no_col, pd.Series.nunique)).reset_index()
        )
        grouped_biz.rename(columns={region_col: "Parent Region Name"}, inplace=True)
        pivot_biz = grouped_biz.pivot_table(
            index="Parent Region Name",
            columns="业务定义",
            values="订单数",
            fill_value=0,
            aggfunc="sum",
            observed=False,
        ).sort_index().sort_index(axis=1)
        if requested_biz:
            cols_biz = [c for c in requested_biz if c in pivot_biz.columns]
            if cols_biz:
                pivot_biz = pivot_biz[cols_biz]
        percent_biz = pivot_biz.div(pivot_biz.sum(axis=0).replace(0, pd.NA), axis=1).fillna(0) * 100
        percent_biz = percent_biz.round(2)
        md_lines.append("## 区域 x 业务定义矩阵")
        md_lines.append(df_to_md(pivot_biz.reset_index()))
        md_lines.append("")
        md_lines.append("## 分 region 占比（%）（按业务定义列归一化）")
        md_lines.append(df_to_md(percent_biz.reset_index()))
        preview_pivot = pivot_biz
        preview_percent = percent_biz
    else:
        md_lines.append("## 区域 x 车型矩阵")
        md_lines.append(df_to_md(pivot_df.reset_index()))
        md_lines.append("")
        md_lines.append("## 分 region 占比（%）（按车型列归一化）")
        md_lines.append(df_to_md(percent_df.reset_index()))
        preview_pivot = pivot_df
        preview_percent = percent_df
    grouped_biz = (
        df_period.groupby([region_col, "业务定义"], observed=False).agg(订单数=(order_no_col, pd.Series.nunique)).reset_index()
    )
    grouped_biz.rename(columns={region_col: "Parent Region Name"}, inplace=True)
    pivot_biz = grouped_biz.pivot_table(
        index="Parent Region Name",
        columns="业务定义",
        values="订单数",
        fill_value=0,
        aggfunc="sum",
        observed=False,
    )
    pivot_biz = pivot_biz.sort_index().sort_index(axis=1)
    percent_biz = pivot_biz.div(pivot_biz.sum(axis=0).replace(0, pd.NA), axis=1).fillna(0) * 100
    percent_biz = percent_biz.round(2)
    md_lines.append("")
    md_lines.append("## 区域 x 业务定义矩阵")
    md_lines.append(df_to_md(pivot_biz.reset_index()))
    md_lines.append("")
    md_lines.append("## 分 region 占比（%）（按业务定义列归一化）")
    md_lines.append(df_to_md(percent_biz.reset_index()))

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
        level_series = df.loc[mask & active_filter, level_col]
        
        # 统计剔除前的“未知”情况（NaN 或 "nan"）
        total_count_l = len(level_series)
        def is_unknown_level(val):
            if pd.isna(val):
                return True
            s = str(val).strip().lower()
            return s == "nan" or s == "null" or s == ""

        unknown_level_mask = level_series.apply(is_unknown_level).astype(bool)
        unknown_l_count = unknown_level_mask.sum()
        unknown_l_pct = (unknown_l_count / total_count_l * 100) if total_count_l > 0 else 0.0

        # 剔除无效值
        level_series_valid = level_series[~unknown_level_mask].astype(str).str.strip()

        level_counts = level_series_valid.value_counts()
        total_orders = len(level_series_valid)
        level_share = (level_counts / max(total_orders, 1) * 100).round(2)
        level_df = (
            pd.DataFrame({"level": level_counts.index, "lock_orders": level_counts.values, "share_pct": level_share.values})
            .sort_values(["lock_orders", "share_pct"], ascending=[False, False])
        )
        md_lines.append("")
        md_lines.append("## 分 license_city_level 的锁单量与占比")
        md_lines.append(f"> 注：已剔除无效/空值样本 {unknown_l_count} 个（占比 {unknown_l_pct:.2f}%），下表基于有效样本 {total_orders} 个统计。")
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
        prov_series = df.loc[mask & active_filter, prov_col].astype(str).fillna("未知")
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
        city_series = df.loc[mask & active_filter, city_col].astype(str).fillna("未知")
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

    # 追加：分年龄段的锁单量与占比
    try:
        age_col = resolve_column(
            df,
            [
                "owner_age",
                "Owner Age",
                "Owner_Age",
                "age",
                "Age",
                "车主年龄",
            ],
        )
        
        # 提取包含年龄、车型、订单号的子集
        age_sub_df = df.loc[mask & active_filter, [age_col, "车型分类", order_no_col, region_col]].copy()
        age_sub_df["age_raw"] = pd.to_numeric(age_sub_df[age_col], errors="coerce")
        
        # 1. 全局统计（均值/中位数等）
        age_series_raw = age_sub_df["age_raw"]
        total_count_age_raw = len(age_series_raw)
        
        out_of_range_mask = age_series_raw.notna() & ~((age_series_raw >= 16) & (age_series_raw <= 85))
        out_of_range_count = int(out_of_range_mask.sum())
        out_of_range_pct = (out_of_range_count / total_count_age_raw * 100) if total_count_age_raw > 0 else 0.0
        
        # 仅保留有效区间内的用于统计数值特征
        valid_ages_stats = age_series_raw.where((age_series_raw >= 16) & (age_series_raw <= 85)).dropna()
        
        if not valid_ages_stats.empty:
            age_mean = valid_ages_stats.mean()
            age_median = valid_ages_stats.median()
            age_std = valid_ages_stats.std()
            md_lines.append("")
            md_lines.append("## 车主年龄统计")
            md_lines.append(f"- 平均值: {age_mean:.2f}")
            md_lines.append(f"- 中位数: {age_median:.2f}")
            md_lines.append(f"- 标准差: {age_std:.2f}")
            md_lines.append(f"> 注：已根据区间过滤剔除不在 [16,85] 的样本 {out_of_range_count} 个（占比 {out_of_range_pct:.2f}%）。")
            if (not biz_mode) and (requested_models and len(requested_models) >= 2):
                age_valid_df = age_sub_df[
                    age_sub_df["age_raw"].notna()
                    & (age_sub_df["age_raw"] >= 16)
                    & (age_sub_df["age_raw"] <= 85)
                ].copy()
                if not age_valid_df.empty:
                    per_model_age = (
                        age_valid_df.groupby("车型分类")["age_raw"]
                        .agg(["mean", "median", "std"])
                        .round(2)
                    )
                    if args.models:
                        wanted = [m.strip() for m in str(args.models).split(",") if m.strip()]
                        cols = [m for m in wanted if m in per_model_age.index]
                        if cols:
                            per_model_age = per_model_age.loc[cols]
                    per_model_age = per_model_age.rename(
                        columns={"mean": "平均值", "median": "中位数", "std": "标准差"}
                    ).T
                    per_model_age.index.name = "指标"
                    md_lines.append("")
                    md_lines.append("## 车主年龄统计（分车型）")
                    md_lines.append(df_to_md(per_model_age.reset_index()))
        else:
            md_lines.append("")
            md_lines.append("## 车主年龄统计")
            md_lines.append("(无有效年龄数据)")
            md_lines.append(f"> 注：已根据区间过滤剔除不在 [16,85] 的样本 {out_of_range_count} 个（占比 {out_of_range_pct:.2f}%）。")

        # 修正：使用 start_date 的年份作为基准
        current_year = start_date.year

        def map_age_group(age):
            if pd.isna(age):
                return "未知"
            birth_year = current_year - age
            if birth_year >= 2000:
                return "00后"
            elif birth_year >= 1995:
                return "95后"
            elif birth_year >= 1990:
                return "90后"
            elif birth_year >= 1985:
                return "85后"
            elif birth_year >= 1980:
                return "80后"
            elif birth_year >= 1975:
                return "75后"
            elif birth_year >= 1970:
                return "70后"
            else:
                return "70前"

        # 2. 分组分布（分车型）
        # 先处理区间外的值：设为 NaN，以便 map_age_group 映射为 "未知"
        # 注意：之前的 valid_ages_stats 已经 dropna 了，这里我们要对 DataFrame 操作
        age_sub_df["age_clean"] = age_sub_df["age_raw"].where(
            (age_sub_df["age_raw"] >= 16) & (age_sub_df["age_raw"] <= 85), pd.NA
        )
        age_sub_df["age_group"] = age_sub_df["age_clean"].apply(map_age_group)
        
        # 统计剔除前的“未知”情况（包含原始NaN和区间外的）
        raw_na_count = age_sub_df["age_raw"].isna().sum()
        total_excluded = raw_na_count + out_of_range_count
        total_excluded_pct = (total_excluded / total_count_age_raw * 100) if total_count_age_raw > 0 else 0.0
        
        # 过滤掉 "未知"
        valid_group_df = age_sub_df[age_sub_df["age_group"] != "未知"].copy()
        valid_sample_count = len(valid_group_df)
        
        # --- A. 总体分布 (Backward Compatibility) ---
        group_counts = valid_group_df["age_group"].value_counts()
        total_orders_age = len(valid_group_df)
        age_share = (group_counts / max(total_orders_age, 1) * 100).round(2)

        age_df = pd.DataFrame({
            "age_group": group_counts.index,
            "lock_orders": group_counts.values,
            "share_pct": age_share.values
        })
        sort_order = ["00后", "95后", "90后", "85后", "80后", "75后", "70后", "70前"]
        age_df["age_group"] = pd.Categorical(age_df["age_group"], categories=sort_order, ordered=True)
        age_df = age_df.sort_values("age_group")

        md_lines.append("")
        md_lines.append("## 分年龄段的锁单量与占比")
        md_lines.append(f"> 注：已剔除年龄未知或不在[16,85]区间的样本共 {total_excluded} 个（占比 {total_excluded_pct:.2f}%），下表基于有效样本 {valid_sample_count} 个统计。")
        md_lines.append(df_to_md(age_df))

        # --- B. 分车型分布 (New Feature) ---
        if not biz_mode:
            age_pivot = valid_group_df.pivot_table(
                index="age_group",
                columns="车型分类",
                values=order_no_col,
                aggfunc="nunique",
                fill_value=0
            )
            age_pivot = age_pivot.reindex(sort_order)
            if args.models:
                wanted = [m.strip() for m in str(args.models).split(",") if m.strip()]
                cols = [c for c in wanted if c in age_pivot.columns]
                if cols:
                    age_pivot = age_pivot[cols]
            age_pct_df = age_pivot.div(age_pivot.sum(axis=0).replace(0, pd.NA), axis=1).fillna(0) * 100
            age_pct_df = age_pct_df.round(2)
            md_lines.append("")
            md_lines.append("## 分年龄段的锁单量与占比（分车型占比%）")
            md_lines.append(df_to_md(age_pct_df.reset_index()))
            age_region_df = valid_group_df[[ "age_group", "车型分类", region_col, order_no_col ]].copy()
            age_region_df = age_region_df[age_region_df[region_col].notna()]
            if not age_region_df.empty:
                age_region_counts = (
                    age_region_df.groupby(["age_group", region_col, "车型分类"])
                    .agg(订单数=(order_no_col, pd.Series.nunique))
                    .reset_index()
                )
                models_for_cross = []
                if requested_models:
                    models_for_cross = [m for m in requested_models if m in age_region_counts["车型分类"].unique()]
                else:
                    models_for_cross = list(age_region_counts["车型分类"].unique())
                for m in models_for_cross:
                    sub = age_region_counts[age_region_counts["车型分类"] == m]
                    if sub.empty:
                        continue
                    pivot_counts = sub.pivot_table(
                        index="age_group",
                        columns=region_col,
                        values="订单数",
                        aggfunc="sum",
                        fill_value=0,
                    )
                    pivot_counts = pivot_counts.reindex(sort_order)
                    row_sum = pivot_counts.sum(axis=1).replace(0, pd.NA)
                    pivot_pct = pivot_counts.div(row_sum, axis=0).fillna(0) * 100
                    pivot_pct = pivot_pct.round(2)
                    md_lines.append("")
                    md_lines.append(f"## 分年龄段的锁单在不同区域的占比（{m}，按年龄段行归一化）")
                    md_lines.append(df_to_md(pivot_pct.reset_index()))

    except KeyError as e:
        md_lines.append("")
        md_lines.append("## 分年龄段的锁单量与占比")
        md_lines.append(f"字段缺失：{e}")

    # 追加：分性别的锁单量与占比
    try:
        gender_col = resolve_column(
            df,
            [
                "owner_gender",
                "Owner Gender",
                "Owner_Gender",
                "gender",
                "Gender",
                "车主性别",
                "性别",
            ],
        )
        raw_gender_series = df.loc[mask & active_filter, gender_col]

        def is_unknown_gender(val):
            if pd.isna(val):
                return True
            s = str(val).strip()
            if not s:  # empty string
                return True
            s_lower = s.lower()
            if s_lower in ["默认未知", "none", "nan", "null", "未知"]:
                return True
            return False

        # 识别未知/无效值
        unknown_mask = raw_gender_series.apply(is_unknown_gender)
        
        # 统计
        total_count_g = len(raw_gender_series)
        unknown_g_count = unknown_mask.sum()
        unknown_g_pct = (unknown_g_count / total_count_g * 100) if total_count_g > 0 else 0.0

        # 提取有效值
        gender_series_valid = raw_gender_series[~unknown_mask].astype(str).str.strip()
        
        gender_counts = gender_series_valid.value_counts()
        total_orders_gender = len(gender_series_valid)
        gender_share = (gender_counts / max(total_orders_gender, 1) * 100).round(2)

        gender_df = (
            pd.DataFrame({"gender": gender_counts.index, "lock_orders": gender_counts.values, "share_pct": gender_share.values})
            .sort_values(["lock_orders", "share_pct"], ascending=[False, False])
        )

        md_lines.append("")
        md_lines.append("## 分性别的锁单量与占比")
        md_lines.append(f"> 注：已剔除性别为“默认未知/None/空”的样本 {unknown_g_count} 个（占比 {unknown_g_pct:.2f}%），下表基于有效样本 {total_orders_gender} 个统计。")
        md_lines.append(df_to_md(gender_df))

    except KeyError as e:
        md_lines.append("")
        md_lines.append("## 分性别的锁单量与占比")
        md_lines.append(f"字段缺失：{e}")

    report_name = f"lock_summary_{args.start}_to_{args.end}.md"
    report_path = OUT_DIR / report_name
    report_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Report saved: {report_path}")
    print("\n预览：")
    if preview_pivot is not None:
        print(df_to_md(preview_pivot.reset_index()))
    print("\n占比预览：")
    if preview_percent is not None:
        print(df_to_md(preview_percent.reset_index()))


if __name__ == "__main__":
    main()
