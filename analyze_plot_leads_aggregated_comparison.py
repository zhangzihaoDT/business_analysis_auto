#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


EXCLUDE_COLS = {"date", "主要渠道线索数比例", "线索数标准差", "线索识别数"}


def load_leads_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError("CSV缺少 'date' 列")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    return df


def filter_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    return df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()


def get_channel_columns(df: pd.DataFrame) -> List[str]:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    channels = [c for c in num_cols if c not in EXCLUDE_COLS]
    return channels


def aggregate_window(df: pd.DataFrame, channels: List[str]) -> pd.DataFrame:
    sums = df[channels].sum()
    out = pd.DataFrame({"channel": sums.index, "count": sums.values})
    denom = df["线索识别数"].sum() if "线索识别数" in df.columns else out["count"].sum()
    if denom and denom > 0:
        out["share_pct"] = (out["count"] / denom * 100).round(2)
    else:
        out["share_pct"] = 0.0
    return out


def make_aggregated_comparison_table(win1: pd.DataFrame, win2: pd.DataFrame, title_left: str, title_right: str) -> go.Figure:
    # Merge by channel to align rows
    merged = pd.merge(win1, win2, on="channel", how="outer", suffixes=("_w1", "_w2")).fillna(0)
    # Sort by count of window2 descending
    merged = merged.sort_values("count_w2", ascending=False)

    # Ensure percentage columns exist
    if "share_pct_w1" not in merged.columns:
        merged["share_pct_w1"] = 0.0
    if "share_pct_w2" not in merged.columns:
        merged["share_pct_w2"] = 0.0

    # Difference column (percentage difference right-left)
    merged["diff_pct"] = merged["share_pct_w2"] - merged["share_pct_w1"]

    # Compute cumulative metrics for Pareto: keep percentage for table, use counts for right Y axis
    merged["cum_pct_w1"] = merged["share_pct_w1"].cumsum().clip(upper=100.0)
    merged["cum_pct_w2"] = merged["share_pct_w2"].cumsum().clip(upper=100.0)
    merged["cum_count_w1"] = merged["count_w1"].cumsum()
    merged["cum_count_w2"] = merged["count_w2"].cumsum()

    # Totals for each window
    total_w1 = int(merged["count_w1"].sum())
    total_w2 = int(merged["count_w2"].sum())

    channels = merged["channel"].tolist()
    counts_w1 = merged["count_w1"].astype(int).tolist()
    counts_w2 = merged["count_w2"].astype(int).tolist()
    shares_w1 = merged["share_pct_w1"].tolist()
    shares_w2 = merged["share_pct_w2"].tolist()
    diffs_pct = merged["diff_pct"].tolist()
    cums_pct_w1 = merged["cum_pct_w1"].tolist()
    cums_pct_w2 = merged["cum_pct_w2"].tolist()
    cums_cnt_w1 = merged["cum_count_w1"].astype(int).tolist()
    cums_cnt_w2 = merged["cum_count_w2"].astype(int).tolist()

    # Build dashboard: grouped bars + cumulative percent lines + detail table
    color1 = "#27AD00"
    color2 = "#005783"
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.12,
        specs=[[{"secondary_y": True}], [{"type": "table"}]],
        subplot_titles=(f"渠道结构帕累托对比（{title_left} vs {title_right}）", "渠道结构明细（数量/占比/累计占比）"),
    )

    # Bars (percentage on primary Y-axis)
    fig.add_bar(
        x=channels, y=shares_w1, name=title_left, marker_color=color1,
        hovertemplate="渠道=%{x}<br>占比=%{y:.2f}%<br>数量=%{customdata}<extra></extra>",
        customdata=[[str(c)] for c in counts_w1],
        row=1, col=1,
    )
    fig.add_bar(
        x=channels, y=shares_w2, name=title_right, marker_color=color2,
        hovertemplate="渠道=%{x}<br>占比=%{y:.2f}%<br>数量=%{customdata}<extra></extra>",
        customdata=[[str(c)] for c in counts_w2],
        row=1, col=1,
    )
    fig.update_layout(barmode="group")

    # Pareto cumulative lines (absolute counts on secondary y-axis)
    fig.add_scatter(
        x=channels, y=cums_cnt_w1, name=f"累计数量（{title_left}）",
        mode="lines+markers", line=dict(color="#71D95B", width=2),
        hovertemplate="渠道=%{x}<br>累计数量=%{y}<extra></extra>",
        row=1, col=1, secondary_y=True,
    )
    fig.add_scatter(
        x=channels, y=cums_cnt_w2, name=f"累计数量（{title_right}）",
        mode="lines+markers", line=dict(color="#3A8BB7", width=2),
        hovertemplate="渠道=%{x}<br>累计数量=%{y}<extra></extra>",
        row=1, col=1, secondary_y=True,
    )
    # Axes titles: left as percentage (cap at 30%), right as cumulative counts
    fig.update_yaxes(title_text="占比%", range=[0, 30], tickformat=".2f", row=1, col=1, secondary_y=False)
    y_max_cnt = max(total_w1, total_w2)
    fig.update_yaxes(title_text="累计数量", range=[0, y_max_cnt], tickformat=".0f", row=1, col=1, secondary_y=True)

    # Table header and cells
    header_vals = [
        "渠道",
        f"数量（{title_left}）",
        f"占比%（{title_left}）",
        f"累计占比%（{title_left}）",
        f"数量（{title_right}）",
        f"占比%（{title_right}）",
        f"累计占比%（{title_right}）",
        "占比差异%（右-左）",
    ]
    header_colors = ["#f0f0f0", color1, color1, color1, color2, color2, color2, "#6f6f6f"]

    cells_channels = channels + ["总计"]
    cells_counts_w1 = counts_w1 + [total_w1]
    cells_shares_w1 = [f"{p:.2f}%" for p in shares_w1] + ["100.00%"]
    cells_cums_w1 = [f"{p:.2f}%" for p in cums_pct_w1] + ["100.00%"]
    cells_counts_w2 = counts_w2 + [total_w2]
    cells_shares_w2 = [f"{p:.2f}%" for p in shares_w2] + ["100.00%"]
    cells_cums_w2 = [f"{p:.2f}%" for p in cums_pct_w2] + ["100.00%"]
    cells_diff_pct = [f"{p:+.2f}%" for p in diffs_pct] + [f"{(100.0-100.0):+.2f}%"]

    fig.add_table(
        header=dict(values=header_vals, fill_color=header_colors, align="center", font=dict(color="white", size=12), height=32),
        cells=dict(values=[
            cells_channels,
            cells_counts_w1,
            cells_shares_w1,
            cells_cums_w1,
            cells_counts_w2,
            cells_shares_w2,
            cells_cums_w2,
            cells_diff_pct,
        ], align="center"),
        row=2, col=1,
    )

    fig.update_layout(title="线索渠道结构帕累托对比看板", margin=dict(l=20, r=20, t=70, b=20), width=1280, height=820)
    return fig


def find_latest_leads_csv(base_dir: Path) -> Optional[Path]:
    """在 processed/ 下查找匹配 leads_daily_*.csv 的最新文件（按修改时间）。"""
    processed_dir = base_dir / "processed"
    if not processed_dir.exists():
        return None
    candidates = list(processed_dir.glob("leads_daily_*.csv"))
    if not candidates:
        # 容错：有些导出可能命名为 leads_daily*.csv（无下划线）
        candidates = list(processed_dir.glob("leads_daily*.csv"))
    if not candidates:
        return None
    # 按修改时间排序，选择最新
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main():
    ap = argparse.ArgumentParser(description="从 leads_daily CSV 聚合两时间窗渠道结构并生成帕累托对比看板")
    ap.add_argument("--csv", required=False, help="CSV路径（默认自动选择 processed 下最新 leads_daily_*.csv）")
    ap.add_argument("--start1", required=True, help="时间窗1起始日期，如 2025-11-01")
    ap.add_argument("--end1", required=True, help="时间窗1结束日期，如 2025-11-07")
    ap.add_argument("--start2", required=True, help="时间窗2起始日期，如 2025-11-08")
    ap.add_argument("--end2", required=True, help="时间窗2结束日期，如 2025-11-14")
    ap.add_argument("--out", default=None, help="输出HTML路径，默认保存在reports目录")
    args = ap.parse_args()

    # 选择 CSV：优先使用 --csv，其次自动选择 processed 下最新 leads_daily_*.csv
    if args.csv:
        csv_path = Path(args.csv)
    else:
        base_dir = Path.cwd()
        latest = find_latest_leads_csv(base_dir)
        if not latest:
            raise FileNotFoundError(
                "未找到默认 CSV。请提供 --csv，或在 processed/ 放置命名为 'leads_daily_*.csv' 的文件。"
            )
        csv_path = latest
        print(f"使用默认 CSV: {csv_path}")

    df = load_leads_csv(csv_path)

    # 使用用户手动定义的时间窗
    start1, end1, start2, end2 = args.start1, args.end1, args.start2, args.end2

    df1 = filter_window(df, start1, end1)
    df2 = filter_window(df, start2, end2)
    channels = get_channel_columns(df)

    agg1 = aggregate_window(df1, channels).rename(columns={"count": "count_w1", "share_pct": "share_pct_w1"})
    agg2 = aggregate_window(df2, channels).rename(columns={"count": "count_w2", "share_pct": "share_pct_w2"})

    fig = make_aggregated_comparison_table(
        win1=agg1,
        win2=agg2,
        title_left=f"{start1} 至 {end1}",
        title_right=f"{start2} 至 {end2}",
    )

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("reports") / f"线索渠道结构帕累托对比_{start1}_至_{end1}_vs_{start2}_至_{end2}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved Pareto dashboard to: {out_path}")


if __name__ == "__main__":
    main()