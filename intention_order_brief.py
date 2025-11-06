#!/usr/bin/env python3
import argparse
import os
import sys
import re
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
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


def safe_fillna_text(series: pd.Series, unknown_label: str = "未知") -> pd.Series:
    """Fill NA on possibly categorical series with a textual unknown label safely."""
    try:
        if pd.api.types.is_categorical_dtype(series):
            # add label when needed
            if unknown_label not in series.cat.categories:
                series = series.cat.add_categories([unknown_label])
            return series.fillna(unknown_label)
        else:
            return series.fillna(unknown_label)
    except Exception:
        # fallback: cast to string and replace 'nan'
        s = series.astype(str)
        return s.replace({"nan": unknown_label})


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    df = pd.read_parquet(path)
    return df


def filter_data(
    df: pd.DataFrame,
    model_group_value: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
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
    col_time = resolve_column(df, "intention_payment_time", candidates)

    # 车型分组筛选
    df = df[df[col_model].astype(str) == str(model_group_value)].copy()

    # 时间筛选（包含边界）
    df[col_time] = pd.to_datetime(df[col_time], errors="coerce")
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(df[col_time] >= start) & (df[col_time] <= end)].copy()

    return df


def compute_channel_share(df: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "channel": [
            "first_middle_channel_name",
            "first_main_channel_group",
            "First Middle Channel Name",
            "firstmiddlechannelname",
            "中间渠道",
            "首中渠道",
        ]
    }
    col_channel = resolve_column(df, "channel", candidates)
    counts = safe_fillna_text(df[col_channel], "未知").value_counts()
    total = int(counts.sum()) if len(df) == 0 else len(df)
    share = (counts / max(total, 1) * 100).round(2)
    res = pd.DataFrame({"count": counts, "share_pct": share})
    res.index.name = col_channel
    res = res.reset_index().sort_values("count", ascending=False)
    return res


def compute_channel_conversion(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Compute N-day conversion rate per channel.

    Converted if order has Lock_Time and (Lock_Time - Intention Payment Time) in [0, window_days] days.
    Denominator is total intention orders per channel in the filtered dataset.
    """
    # Resolve columns
    channel_candidates = {
        "channel": [
            "first_middle_channel_name",
            "first_main_channel_group",
            "First Middle Channel Name",
            "firstmiddlechannelname",
            "中间渠道",
            "首中渠道",
        ]
    }
    time_candidates = {
        "intention_payment_time": [
            "Intention Payment Time",
            "意向支付时间",
            "intention_payment_time",
            "intent_payment_time",
            "intentpaytime",
        ],
        "lock_time": [
            "Lock_Time",
            "Lock Time",
            "lock_time",
            "锁单时间",
            "锁定时间",
        ],
    }

    col_channel = resolve_column(df, "channel", channel_candidates)
    col_intent = resolve_column(df, "intention_payment_time", time_candidates)
    col_lock = resolve_column(df, "lock_time", time_candidates)

    # Prepare series
    s_channel = safe_fillna_text(df[col_channel], "未知")
    s_intent = pd.to_datetime(df[col_intent], errors="coerce")
    s_lock = pd.to_datetime(df[col_lock], errors="coerce")

    delta_days = (s_lock - s_intent).dt.days
    converted = (s_lock.notna()) & (delta_days >= 0) & (delta_days <= window_days)

    grp = pd.DataFrame({
        "channel": s_channel,
        "converted": converted,
    }).groupby("channel", dropna=False).agg(
        total=("converted", "size"),
        converted_count=("converted", "sum"),
    )
    grp["conversion_pct"] = (grp["converted_count"] / grp["total"]) * 100
    grp["conversion_pct"] = grp["conversion_pct"].round(2)
    return grp.reset_index().sort_values(["total", "conversion_pct"], ascending=[False, False])


def compute_gender_conversion(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Compute N-day conversion rate per gender."""
    candidates = {
        "gender": ["order_gender", "buyer_gender", "性别", "gender"],
    }
    time_candidates = {
        "intention_payment_time": [
            "Intention Payment Time",
            "意向支付时间",
            "intention_payment_time",
            "intent_payment_time",
            "intentpaytime",
        ],
        "lock_time": [
            "Lock_Time",
            "Lock Time",
            "lock_time",
            "锁单时间",
            "锁定时间",
        ],
    }

    col_gender = resolve_column(df, "gender", candidates)
    col_intent = resolve_column(df, "intention_payment_time", time_candidates)
    col_lock = resolve_column(df, "lock_time", time_candidates)

    s_gender = safe_fillna_text(df[col_gender], "未知")
    s_intent = pd.to_datetime(df[col_intent], errors="coerce")
    s_lock = pd.to_datetime(df[col_lock], errors="coerce")

    delta_days = (s_lock - s_intent).dt.days
    converted = (s_lock.notna()) & (delta_days >= 0) & (delta_days <= window_days)

    grp = pd.DataFrame({
        "gender": s_gender,
        "converted": converted,
    }).groupby("gender", dropna=False).agg(
        total=("converted", "size"),
        converted_count=("converted", "sum"),
    )
    grp["conversion_pct"] = (grp["converted_count"] / grp["total"]) * 100
    grp["conversion_pct"] = grp["conversion_pct"].round(2)
    return grp.reset_index().sort_values(["total", "conversion_pct"], ascending=[False, False])


def compute_age_bucket_conversion(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Compute N-day conversion rate per age bucket using the same binning as profile."""
    candidates = {
        "age": ["buyer_age", "年龄", "age", "order_buyer_age"],
    }
    time_candidates = {
        "intention_payment_time": [
            "Intention Payment Time",
            "意向支付时间",
            "intention_payment_time",
            "intent_payment_time",
            "intentpaytime",
        ],
        "lock_time": [
            "Lock_Time",
            "Lock Time",
            "lock_time",
            "锁单时间",
            "锁定时间",
        ],
    }

    col_age = resolve_column(df, "age", candidates)
    col_intent = resolve_column(df, "intention_payment_time", time_candidates)
    col_lock = resolve_column(df, "lock_time", time_candidates)

    age_series = pd.to_numeric(df[col_age], errors="coerce")
    bins = [0, 24, 34, 44, 54, 150]
    labels = ["<25", "25-34", "35-44", "45-54", "55+"]
    age_binned = pd.cut(age_series, bins=bins, labels=labels, right=True)
    age_binned = safe_fillna_text(age_binned, "未知")

    s_intent = pd.to_datetime(df[col_intent], errors="coerce")
    s_lock = pd.to_datetime(df[col_lock], errors="coerce")
    delta_days = (s_lock - s_intent).dt.days
    converted = (s_lock.notna()) & (delta_days >= 0) & (delta_days <= window_days)

    grp = pd.DataFrame({
        "age_bucket": age_binned,
        "converted": converted,
    }).groupby("age_bucket", dropna=False).agg(
        total=("converted", "size"),
        converted_count=("converted", "sum"),
    )
    grp["conversion_pct"] = (grp["converted_count"] / grp["total"]) * 100
    grp["conversion_pct"] = grp["conversion_pct"].round(2)
    # keep order similar to labels, put '未知' at end
    order = labels + (["未知"] if "未知" in grp.index.astype(str).tolist() else [])
    grp = grp.reindex(order).dropna(how="all")
    return grp.reset_index().sort_values(["total", "conversion_pct"], ascending=[False, False])


def compute_parent_region_conversion(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Compute N-day conversion rate per parent region."""
    candidates = {
        "parent_region": [
            "Parent Region Name",
            "父区域名",
            "parentregionname",
            "区域父级名称",
            "parent_region_name",
        ],
    }
    time_candidates = {
        "intention_payment_time": [
            "Intention Payment Time",
            "意向支付时间",
            "intention_payment_time",
            "intent_payment_time",
            "intentpaytime",
        ],
        "lock_time": [
            "Lock_Time",
            "Lock Time",
            "lock_time",
            "锁单时间",
            "锁定时间",
        ],
    }

    col_region = resolve_column(df, "parent_region", candidates)
    col_intent = resolve_column(df, "intention_payment_time", time_candidates)
    col_lock = resolve_column(df, "lock_time", time_candidates)

    s_region = safe_fillna_text(df[col_region], "未知")
    s_intent = pd.to_datetime(df[col_intent], errors="coerce")
    s_lock = pd.to_datetime(df[col_lock], errors="coerce")
    delta_days = (s_lock - s_intent).dt.days
    converted = (s_lock.notna()) & (delta_days >= 0) & (delta_days <= window_days)

    grp = pd.DataFrame({
        "parent_region": s_region,
        "converted": converted,
    }).groupby("parent_region", dropna=False).agg(
        total=("converted", "size"),
        converted_count=("converted", "sum"),
    )
    grp["conversion_pct"] = (grp["converted_count"] / grp["total"]) * 100
    grp["conversion_pct"] = grp["conversion_pct"].round(2)
    return grp.reset_index().sort_values(["total", "conversion_pct"], ascending=[False, False])


def compute_profiles(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    candidates = {
        "gender": ["order_gender", "buyer_gender", "性别", "gender"],
        "age": ["buyer_age", "年龄", "age", "order_buyer_age"],
        "parent_region": [
            "Parent Region Name",
            "父区域名",
            "parentregionname",
            "区域父级名称",
            "parent_region_name",
        ],
    }

    # 性别分布
    col_gender = resolve_column(df, "gender", candidates)
    gender_counts = safe_fillna_text(df[col_gender], "未知").value_counts()
    gender_share = (gender_counts / max(len(df), 1) * 100).round(2)
    gender_df = (
        pd.DataFrame({"count": gender_counts, "share_pct": gender_share})
        .reset_index()
        .rename(columns={"index": col_gender})
        .sort_values("count", ascending=False)
    )

    # 年龄分布（分箱 + 简单统计）
    col_age = resolve_column(df, "age", candidates)
    age_series = pd.to_numeric(df[col_age], errors="coerce")
    bins = [0, 24, 34, 44, 54, 150]
    labels = ["<25", "25-34", "35-44", "45-54", "55+"]
    age_binned = pd.cut(age_series, bins=bins, labels=labels, right=True)
    age_counts = age_binned.value_counts().reindex(labels).fillna(0).astype(int)
    age_share = (age_counts / max(len(df), 1) * 100).round(2)
    age_df = pd.DataFrame({"age_bucket": labels, "count": age_counts.values, "share_pct": age_share.values})
    age_stats = {
        "mean": float(np.nanmean(age_series)) if len(age_series) else np.nan,
        "median": float(np.nanmedian(age_series)) if len(age_series) else np.nan,
    }

    # 区域分布（父区域）
    col_region = resolve_column(df, "parent_region", candidates)
    region_counts = safe_fillna_text(df[col_region], "未知").value_counts()
    region_share = (region_counts / max(len(df), 1) * 100).round(2)
    region_df = (
        pd.DataFrame({"count": region_counts, "share_pct": region_share})
        .reset_index()
        .rename(columns={"index": col_region})
        .sort_values("count", ascending=False)
    )

    return {
        "gender": gender_df,
        "age": age_df,
        "age_stats": pd.DataFrame([age_stats]),
        "region": region_df,
    }


def format_section(title: str) -> str:
    return f"\n【{title}】\n"


def generate_report(
    df_filtered: pd.DataFrame,
    model_group: str,
    start_date: str,
    end_date: str,
    window_days: int,
) -> str:
    total = len(df_filtered)
    lines: List[str] = []
    lines.append(f"意向订单简报（车型分组={model_group}，时间范围={start_date} 至 {end_date}）")
    lines.append(f"样本量：{total}")

    # 渠道占比
    try:
        channel_df = compute_channel_share(df_filtered)
        lines.append(format_section("分渠道订单占比（first_middle_channel_name）"))
        for _, row in channel_df.iterrows():
            channel = str(row[channel_df.columns[0]])
            cnt = int(row["count"]) if not pd.isna(row["count"]) else 0
            pct = float(row["share_pct"]) if not pd.isna(row["share_pct"]) else 0.0
            lines.append(f"- {channel}: {cnt}（{pct:.2f}%）")
    except KeyError as e:
        lines.append(format_section("分渠道订单占比"))
        lines.append(f"字段缺失：{e}")

    # 分渠道的N日小订转化率
    try:
        conv_df = compute_channel_conversion(df_filtered, window_days)
        lines.append(format_section(f"分渠道的{window_days}日小订转化率（Lock_Time 与 Intention Payment ≤ {window_days} 天）"))
        for _, row in conv_df.iterrows():
            channel = str(row["channel"])
            total = int(row["total"]) if not pd.isna(row["total"]) else 0
            converted = int(row["converted_count"]) if not pd.isna(row["converted_count"]) else 0
            pct = float(row["conversion_pct"]) if not pd.isna(row["conversion_pct"]) else 0.0
            lines.append(f"- {channel}: {converted}/{total}（{pct:.2f}%）")
    except KeyError as e:
        lines.append(format_section(f"分渠道的{window_days}日小订转化率"))
        lines.append(f"字段缺失：{e}")

    # 画像结构
    lines.append(format_section("画像订单结构"))
    try:
        profiles = compute_profiles(df_filtered)
        gender_df = profiles["gender"]
        lines.append("性别分布：")
        for _, row in gender_df.iterrows():
            g = str(row[gender_df.columns[0]])
            cnt = int(row["count"]) if not pd.isna(row["count"]) else 0
            pct = float(row["share_pct"]) if not pd.isna(row["share_pct"]) else 0.0
            lines.append(f"- {g}: {cnt}（{pct:.2f}%）")

        # 性别的N日小订转化率
        try:
            gender_conv_df = compute_gender_conversion(df_filtered, window_days)
            lines.append(f"性别的{window_days}日小订转化率：")
            for _, row in gender_conv_df.iterrows():
                g = str(row["gender"]) 
                total = int(row["total"]) if not pd.isna(row["total"]) else 0
                converted = int(row["converted_count"]) if not pd.isna(row["converted_count"]) else 0
                pct = float(row["conversion_pct"]) if not pd.isna(row["conversion_pct"]) else 0.0
                lines.append(f"- {g}: {converted}/{total}（{pct:.2f}%）")
        except KeyError as e:
            lines.append(f"字段缺失（性别转化率）：{e}")

        age_df = profiles["age"]
        lines.append("年龄分布（分箱）：")
        for _, row in age_df.iterrows():
            bucket = str(row["age_bucket"])
            cnt = int(row["count"]) if not pd.isna(row["count"]) else 0
            pct = float(row["share_pct"]) if not pd.isna(row["share_pct"]) else 0.0
            lines.append(f"- {bucket}: {cnt}（{pct:.2f}%）")
        age_stats_df = profiles["age_stats"].iloc[0]
        mean_age = age_stats_df.get("mean", np.nan)
        median_age = age_stats_df.get("median", np.nan)
        if not np.isnan(mean_age):
            lines.append(f"年龄均值：{mean_age:.1f}；年龄中位数：{median_age:.1f}")

        # 年龄（分箱）的N日小订转化率
        try:
            age_conv_df = compute_age_bucket_conversion(df_filtered, window_days)
            lines.append(f"年龄（分箱）的{window_days}日小订转化率：")
            for _, row in age_conv_df.iterrows():
                bucket = str(row["age_bucket"]) 
                total = int(row["total"]) if not pd.isna(row["total"]) else 0
                converted = int(row["converted_count"]) if not pd.isna(row["converted_count"]) else 0
                pct = float(row["conversion_pct"]) if not pd.isna(row["conversion_pct"]) else 0.0
                lines.append(f"- {bucket}: {converted}/{total}（{pct:.2f}%）")
        except KeyError as e:
            lines.append(f"字段缺失（年龄转化率）：{e}")

        region_df = profiles["region"]
        lines.append("父区域分布（Top 15）：")
        top_n = region_df.head(15)
        for _, row in top_n.iterrows():
            region = str(row[region_df.columns[0]])
            cnt = int(row["count"]) if not pd.isna(row["count"]) else 0
            pct = float(row["share_pct"]) if not pd.isna(row["share_pct"]) else 0.0
            lines.append(f"- {region}: {cnt}（{pct:.2f}%）")

        # 父区域的N日小订转化率
        try:
            region_conv_df = compute_parent_region_conversion(df_filtered, window_days)
            lines.append(f"父区域的{window_days}日小订转化率（Top 15）：")
            for _, row in region_conv_df.head(15).iterrows():
                region = str(row["parent_region"]) 
                total = int(row["total"]) if not pd.isna(row["total"]) else 0
                converted = int(row["converted_count"]) if not pd.isna(row["converted_count"]) else 0
                pct = float(row["conversion_pct"]) if not pd.isna(row["conversion_pct"]) else 0.0
                lines.append(f"- {region}: {converted}/{total}（{pct:.2f}%）")
        except KeyError as e:
            lines.append(f"字段缺失（父区域转化率）：{e}")
    except KeyError as e:
        lines.append(f"字段缺失：{e}")

    return "\n".join(lines)


def ensure_reports_dir(root_dir: str) -> str:
    reports_dir = os.path.join(root_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="意向订单分析并生成简报")
    parser.add_argument("-m", "--model-group", required=True, help="车型分组，例如：LS9")
    parser.add_argument("--start-date", required=True, help="开始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="结束日期，格式 YYYY-MM-DD")
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="转化窗口天数（Lock_Time 与 Intention Payment 的最大间隔天数），默认 30",
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help=f"数据文件路径（默认：{DEFAULT_DATA_PATH}）",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出简报文件路径（默认写入到项目 reports 目录）",
    )

    args = parser.parse_args(argv)

    # 读取数据
    df = load_data(args.data_path)

    # 筛选
    df_filtered = filter_data(df, args.model_group, args.start_date, args.end_date)

    # 生成简报文本
    report_text = generate_report(df_filtered, args.model_group, args.start_date, args.end_date, args.window_days)

    # 输出路径
    script_dir = os.path.dirname(__file__)
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))
    reports_dir = ensure_reports_dir(root_dir)

    if args.output:
        out_path = args.output
    else:
        today_str = datetime.now().strftime("%Y-%m-%d")
        out_filename = f"意向订单简报_{args.model_group}_{args.start_date}_至_{args.end_date}_{today_str}.txt"
        out_path = os.path.join(reports_dir, out_filename)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print(f"简报已生成：{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())