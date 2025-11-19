#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from typing import Optional, List

import pandas as pd


BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_INPUT = os.path.join(BASE, "original", "leads_assign_city_store2_1.csv")
OUT_DIR = os.path.join(BASE, "processed", "analysis_results")


def load_csv(path: str) -> pd.DataFrame:
    encodings = ["utf-16", "utf-16-le", "utf-8-sig", "utf-8", "gb18030", "latin1"]
    seps = [",", "\t", ";", "|"]
    last_err = None
    for enc in encodings:
        sep_guess = ","
        try:
            with open(path, "r", encoding=enc) as f:
                first = f.readline()
            sep_guess = max(seps, key=lambda s: first.count(s))
        except Exception:
            pass

        try:
            df = pd.read_csv(path, encoding=enc, sep=sep_guess)
            if df.shape[1] > 1:
                return df
        except Exception:
            try:
                df = pd.read_csv(
                    path,
                    encoding=enc,
                    sep=sep_guess,
                    engine="python",
                    on_bad_lines="skip",
                    quotechar='"',
                    escapechar='\\',
                    doublequote=True,
                )
                if df.shape[1] > 1:
                    return df
            except Exception as e2:
                last_err = e2
                continue
    raise RuntimeError(f"无法读取CSV: {path}. 最后错误: {last_err}")


def resolve_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    def norm(s: str) -> str:
        return str(s).strip().lower().replace(" ", "").replace("_", "")

    index = {norm(c): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        nc = norm(c)
        if nc in index:
            return index[nc]
    return None


def build_date(df: pd.DataFrame, col_year: Optional[str], col_month: Optional[str], col_day: Optional[str], col_full: Optional[str]) -> pd.Series:
    # 1) Prefer full timestamp column
    if col_full and col_full in df.columns:
        dt = pd.to_datetime(df[col_full], errors="coerce")
        if dt.notna().any():
            return dt.dt.date

    # 2) Try parsing 'day' column directly if it contains full date strings
    if col_day and col_day in df.columns:
        s = df[col_day].astype(str)
        looks_like_date = s.str.contains("-|") | s.str.contains("年")
        if looks_like_date.any():
            dt_try = pd.to_datetime(s.str.replace("年", "-").str.replace("月", "-").str.replace("日", ""), errors="coerce")
            if dt_try.notna().any():
                return dt_try.dt.date

    # 3) Compose from year/month/day
    parts = {}
    if col_year and col_year in df.columns:
        parts["year"] = pd.to_numeric(df[col_year], errors="coerce")
    if col_month and col_month in df.columns:
        parts["month"] = pd.to_numeric(df[col_month], errors="coerce")
    if col_day and col_day in df.columns:
        parts["day"] = pd.to_numeric(df[col_day], errors="coerce")
    if {"year", "month", "day"}.issubset(parts.keys()):
        y = parts["year"].fillna(0).astype(int)
        m = parts["month"].fillna(1).astype(int)
        d = parts["day"].fillna(1).astype(int)
        dt = pd.to_datetime(pd.DataFrame({"y": y, "m": m, "d": d}), errors="coerce")
        if dt.notna().any():
            return dt.dt.date

    # 4) Fail: return all-NA series
    return pd.to_datetime(pd.Series([None] * len(df))).dt.date


def pivot_metrics(df: pd.DataFrame, col_city: str, col_region: str, col_date: str, col_metric_name: str, col_metric_value: str) -> pd.DataFrame:
    # Keep only relevant columns
    keep = [col_city, col_region, col_date, col_metric_name, col_metric_value]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必要列: {missing}")

    tmp = df[keep].copy()
    # Clean and coerce metric values to numeric
    if tmp[col_metric_value].dtype == object:
        s = tmp[col_metric_value].astype(str)
        s = s.str.replace(",", "", regex=False).str.strip()
        tmp[col_metric_value] = pd.to_numeric(s, errors="coerce")
    else:
        tmp[col_metric_value] = pd.to_numeric(tmp[col_metric_value], errors="coerce")

    # Aggregation rules:
    # - 下发门店数: use max per (city, region, date)
    # - 线索识别数: use sum per (city, region, date)
    idx_cols = [col_city, col_region, col_date]
    assign_df = (
        tmp[tmp[col_metric_name] == "下发门店数"]
        .groupby(idx_cols, as_index=False)[col_metric_value]
        .max()
        .rename(columns={col_metric_value: "下发门店数"})
    )
    identify_df = (
        tmp[tmp[col_metric_name] == "线索识别数"]
        .groupby(idx_cols, as_index=False)[col_metric_value]
        .sum()
        .rename(columns={col_metric_value: "线索识别数"})
    )

    wide = pd.merge(assign_df, identify_df, on=idx_cols, how="outer").fillna(0)

    # Reorder columns
    wide = wide[[col_city, col_region, col_date, "下发门店数", "线索识别数"]]
    # Ensure numeric types
    wide["下发门店数"] = pd.to_numeric(wide["下发门店数"], errors="coerce").fillna(0)
    wide["线索识别数"] = pd.to_numeric(wide["线索识别数"], errors="coerce").fillna(0)
    return wide


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="按指定区间汇总下发门店数(按日-城取max)与线索识别数(取sum)")
    parser.add_argument("--start", required=True, help="开始日期，格式YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="结束日期，格式YYYY-MM-DD")
    args = parser.parse_args(argv)

    input_path = DEFAULT_INPUT
    start = args.start
    end = args.end

    if not os.path.exists(input_path):
        print(f"❌ 输入文件不存在: {input_path}")
        return 1

    print(f"📥 使用固定源文件: {input_path}")
    df = load_csv(input_path)
    print(f"📊 原始维度: {df.shape[0]} 行 × {df.shape[1]} 列")

    # Resolve necessary columns
    col_city = resolve_col(df, ["lc_assign_1st2sales_city_name", "city", "城市"])
    col_region = resolve_col(df, ["lc_assign_1st2sales_region_name", "region", "区域"])
    col_day = resolve_col(df, ["日(lc_assign_time_min)", "lc_assign_time_min 日", "day(lc_assign_time_min)", "day"])
    col_year = resolve_col(df, ["lc_assign_time_min 年", "年(lc_assign_time_min)", "year(lc_assign_time_min)", "year"])
    col_month = resolve_col(df, ["lc_assign_time_min 月", "月(lc_assign_time_min)", "month(lc_assign_time_min)", "month"])
    col_full = resolve_col(df, ["lc_assign_time_min", "assign_time", "时间", "date"])
    col_metric_name = resolve_col(df, ["度量名称", "Measure Names", "measure_names", "metric_name"])
    col_metric_value = resolve_col(df, ["度量值", "Measure Values", "measure_values", "metric_value", "value"])

    if not (col_city and col_region and col_metric_name and col_metric_value):
        print("❌ 关键列未识别：", {
            "city": col_city,
            "region": col_region,
            "metric_name": col_metric_name,
            "metric_value": col_metric_value,
        })
        print("🧭 列名样例: ", ", ".join(map(str, list(df.columns)[:12])), "...")
        return 2

    # Build date
    date_series = build_date(df, col_year, col_month, col_day, col_full)
    df = df.copy()
    df["__date__"] = date_series
    if df["__date__"].isna().all():
        print("❌ 无法构建有效日期列（缺少月或完整时间列），请检查原始数据列。")
        return 3

    # Pivot to wide format
    wide = pivot_metrics(df, col_city, col_region, "__date__", col_metric_name, col_metric_value)
    print(f"🔁 转置后维度: {wide.shape[0]} 行 × {wide.shape[1]} 列")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Filter by date range
    start_d = pd.to_datetime(start).date()
    end_d = pd.to_datetime(end).date()
    mask = (wide["__date__"].notna()) & (wide["__date__"] >= start_d) & (wide["__date__"] <= end_d)
    sel = wide.loc[mask].copy()
    print(f"📆 过滤区间: [{start} ~ {end}]，匹配行数: {sel.shape[0]}")

    # Per-city summary: 下发门店数使用 max，线索识别数使用 sum
    city_sum = sel.groupby([col_city, col_region], as_index=False).agg({
        "下发门店数": "max",
        "线索识别数": "sum",
    })

    # 合并处理：将“上海市”同城不同区域的记录合并为单行
    # 合并逻辑：
    # - 线索识别数：对同城的区域行求和
    # - 下发门店数：对同城的区域行求和（保持与总计一致，总计按城市行相加）
    #   注：若需改为同城“max”可将下方 sum 改为 max，但这会改变总计口径
    def merge_city_rows(df: pd.DataFrame, city_name: str) -> pd.DataFrame:
        rows = df[df[col_city] == city_name]
        if len(rows) <= 1:
            return df
        merged_assign = int(pd.to_numeric(rows["下发门店数"], errors="coerce").fillna(0).sum())
        merged_identify = int(pd.to_numeric(rows["线索识别数"], errors="coerce").fillna(0).sum())
        # 选择首个非空区域作为展示（也可固定为“合并”或“上海区”）
        region_value = (
            rows[col_region].dropna().iloc[0] if not rows[col_region].dropna().empty else ""
        )
        new_row = {
            col_city: city_name,
            col_region: region_value,
            "下发门店数": merged_assign,
            "线索识别数": merged_identify,
        }
        df = df[df[col_city] != city_name]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        return df

    city_sum = merge_city_rows(city_sum, "上海市")
    # Compute totals: 下发门店数为各城市最大值之和；线索识别数为区间内总和
    total_assign = int(city_sum["下发门店数"].sum())
    total_identify = int(sel["线索识别数"].sum())
    print(f"✅ 总计（{start}~{end}）：下发门店数={total_assign}，线索识别数={total_identify}")

    # Daily leads summary within range: 每天线索识别数（全城市合计）
    daily_leads = sel.groupby("__date__", as_index=False)["线索识别数"].sum()
    # Show a preview of first 7 days
    preview_days = min(7, len(daily_leads))
    if preview_days > 0:
        print("📅 区间内每日线索识别数预览：")
        for _, row in daily_leads.sort_values("__date__").head(preview_days).iterrows():
            print(f"- {row['__date__']}: 线索识别数={int(row['线索识别数'])}")

    # Build a single Markdown report
    def df_to_md(df: pd.DataFrame, columns: list, headers: list) -> str:
        out = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
        for _, r in df[columns].iterrows():
            row = [str(r[c]) for c in columns]
            out.append("|" + "|".join(row) + "|")
        return "\n".join(out)

    report_name = f"leads_assign_summary_{start}_to_{end}.md"
    report_path = os.path.join(OUT_DIR, report_name)

    # Sort city summary by 线索识别数 desc
    city_sum_sorted = city_sum.sort_values("线索识别数", ascending=False).copy()
    # Ensure integer formatting
    city_sum_sorted["下发门店数"] = city_sum_sorted["下发门店数"].astype(int)
    city_sum_sorted["线索识别数"] = city_sum_sorted["线索识别数"].astype(int)

    daily_sorted = daily_leads.sort_values("__date__").copy()
    daily_sorted["线索识别数"] = daily_sorted["线索识别数"].astype(int)

    lines = []
    lines.append(f"# 线索与门店下发汇总报告\n")
    lines.append(f"- 源文件: `{input_path}`")
    lines.append(f"- 时间区间: `{start}` ~ `{end}`\n")
    lines.append("## 区间总计")
    lines.append(f"- 下发门店数（按城取区间内日max后相加）: `{total_assign}`")
    lines.append(f"- 线索识别数（区间内合计）: `{total_identify}`\n")

    lines.append("## 分城市汇总（按线索识别数降序）")
    lines.append(df_to_md(
        city_sum_sorted,
        [col_city, col_region, "下发门店数", "线索识别数"],
        ["城市", "区域", "下发门店数(max)", "线索识别数(sum)"]
    ))
    lines.append("")

    lines.append("## 每日线索识别数（全城市合计）")
    lines.append(df_to_md(
        daily_sorted,
        ["__date__", "线索识别数"],
        ["日期", "线索识别数(sum)"]
    ))
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📝 已生成Markdown报告: {report_path}")

    # Show top 10 cities by 线索识别数
    top10 = city_sum.sort_values("线索识别数", ascending=False).head(10)
    print("🏙️ Top10 城市（按线索识别数）：")
    for _, row in top10.iterrows():
        print(f"- {row[col_city]}（{row[col_region]}）：线索识别数={int(row['线索识别数'])}，下发门店数={int(row['下发门店数'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())