#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_INPUT = Path("/Users/zihao_/Documents/coding/dataset/processed/Core_Metrics_transposed.csv")


def load_core_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    date_col = "日(日期)"
    if date_col not in df.columns:
        raise KeyError(f"未找到日期列: {date_col}")
    try:
        df["date"] = pd.to_datetime(df[date_col], format="%Y年%m月%d日", errors="coerce")
    except Exception:
        ser = (
            df[date_col]
            .astype(str)
            .str.replace("年", "-", regex=False)
            .str.replace("月", "-", regex=False)
            .str.replace("日", "", regex=False)
        )
        df["date"] = pd.to_datetime(ser, errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_conversion_rates(df: pd.DataFrame) -> pd.DataFrame:
    col_30 = "30 日锁单线索数"
    col_7 = "7 日内锁单线索数"
    col_eff = "有效线索数"
    for c in (col_30, col_7, col_eff):
        if c not in df.columns:
            raise KeyError(f"未找到必要列: {c}")
    denom = df[col_eff].astype(float).replace(0, pd.NA)
    df["30日线索锁单转化率"] = (df[col_30].astype(float) / denom) * 100.0
    df["7日线索锁单转化率"] = (df[col_7].astype(float) / denom) * 100.0
    return df


def align_series_by_day(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, series: pd.Series, ma_window: int = 7) -> pd.Series:
    mask = (df["date"] >= start) & (df["date"] <= end)
    sub = pd.DataFrame({"date": df.loc[mask, "date"], "val": series.loc[mask].astype(float)})
    sub = sub.dropna(subset=["date"]).copy()
    sub["val_ma"] = sub["val"].rolling(window=ma_window, min_periods=1).mean()
    sub["day_n"] = (sub["date"] - start).dt.days
    aligned = sub.set_index("day_n")["val_ma"].sort_index()
    full = pd.Index(range(0, 366), name="day_n")
    aligned = aligned.reindex(full)
    return aligned


def make_alignment_chart(df: pd.DataFrame) -> go.Figure:
    w2_start = pd.to_datetime("2023-12-01")
    w2_end = pd.to_datetime("2024-11-30")
    w1_start = pd.to_datetime("2024-12-01")
    w1_end = pd.to_datetime("2025-11-30")

    m_lock = "锁单数"
    m_eff = "有效线索数"
    m_td_eff = "有效试驾数"
    m_dis = "下发线索数"
    m_store = "下发线索门店数"
    m_owner = "下发线索主理数"

    store_avg = df[m_dis].astype(float) / df[m_store].replace(0, pd.NA).astype(float)
    owner_avg = df[m_dis].astype(float) / df[m_owner].replace(0, pd.NA).astype(float)

    metrics = [
        ("锁单数", df[m_lock].astype(float), "数", ":.0f"),
        ("有效线索数", df[m_eff].astype(float), "数", ":.0f"),
        ("7日线索转化率", df["7日线索锁单转化率"].astype(float), "百分比", ":.2f%"),
        ("30日线索转化率", df["30日线索锁单转化率"].astype(float), "百分比", ":.2f%"),
        ("有效试驾数", df[m_td_eff].astype(float), "数", ":.0f"),
        ("店均下发线索数", store_avg, "数", ":.2f"),
        ("主理人均下发线索数", owner_avg, "数", ":.2f"),
    ]

    titles = [
        "锁单数（按第N天对齐，MA7）",
        "有效线索数（按第N天对齐，MA7）",
        "7日线索转化率（按第N天对齐，MA7）",
        "30日线索转化率（按第N天对齐，MA7）",
        "有效试驾数（按第N天对齐，MA7）",
        "店均下发线索数（按第N天对齐，MA7）",
        "主理人均下发线索数（按第N天对齐，MA7）",
    ]

    fig = make_subplots(rows=len(metrics), cols=1, shared_xaxes=True, subplot_titles=titles, vertical_spacing=0.06)

    def add_local_legend(yaxis_name: str, entries: list):
        dom = getattr(fig.layout, yaxis_name).domain if hasattr(fig.layout, yaxis_name) else None
        if not dom:
            return
        suffix = "" if yaxis_name == "yaxis" else yaxis_name.replace("yaxis", "")
        xaxis_name = "xaxis" + suffix
        xdom = getattr(fig.layout, xaxis_name).domain if hasattr(fig.layout, xaxis_name) else [0.0, 1.0]
        y = dom[1] - (dom[1] - dom[0]) * 0.04
        x = xdom[0] + (xdom[1] - xdom[0]) * 0.02
        label = "  ".join([f"<span style='color:{c}'>■</span> {t}" for t, c in entries])
        fig.add_annotation(
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#7B848F",
            borderwidth=1,
            font=dict(size=12, color="#7B848F"),
            xanchor="left",
            yanchor="top",
        )

    x_vals = list(range(0, 366))
    for i, (name, series, unit, fmt) in enumerate(metrics, start=1):
        s_w1 = align_series_by_day(df, w1_start, w1_end, series, ma_window=7)
        s_w2 = align_series_by_day(df, w2_start, w2_end, series, ma_window=7)
        fig.add_trace(
            go.Scatter(
                name=f"{name} 2024-12-01 至 2025-11-30 (MA7)",
                x=x_vals,
                y=s_w1.values,
                mode="lines",
                line=dict(color="#27AD00", width=2),
                hovertemplate=f"第N天=%{{x}}<br>{name}(MA7)=%{{y{fmt}}}<extra></extra>",
                showlegend=False,
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name=f"{name} 2023-12-01 至 2024-11-30 (MA7)",
                x=x_vals,
                y=s_w2.values,
                mode="lines",
                line=dict(color="#005783", width=2),
                hovertemplate=f"第N天=%{{x}}<br>{name}(MA7)=%{{y{fmt}}}<extra></extra>",
                showlegend=False,
            ),
            row=i,
            col=1,
        )
        fig.update_yaxes(title_text=name, row=i, col=1)

        yaxis_name = "yaxis" + ("" if i == 1 else str(i))
        add_local_legend(yaxis_name, [("2024-12-01 至 2025-11-30 (MA7)", "#27AD00"), ("2023-12-01 至 2024-11-30 (MA7)", "#005783")])

    fig.update_layout(
        title="两个窗口按第N天对齐的指标对比",
        legend_title="窗口",
        height=2200,
        margin=dict(l=40, r=40, t=60, b=80),
        showlegend=False,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    fig.update_xaxes(title_text="第 N 天 (0-365)", row=len(metrics), col=1)
    fig.update_xaxes(gridcolor="#ebedf0", zerolinecolor="#ebedf0", linecolor="#7B848F", tickfont=dict(color="#7B848F"))
    fig.update_yaxes(gridcolor="#ebedf0", zerolinecolor="#ebedf0", linecolor="#7B848F", tickfont=dict(color="#7B848F"))
    return fig


def main():
    ap = argparse.ArgumentParser(description="按第N天对齐两个窗口，绘制核心指标对比折线")
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="输入CSV路径")
    ap.add_argument("--out", default=None, help="输出HTML路径，默认写入 reports/两个窗口_第N天对齐.html")
    args = ap.parse_args()

    df = load_core_metrics(Path(args.input))
    df = compute_conversion_rates(df)
    fig = make_alignment_chart(df)

    out_path = Path(args.out) if args.out else Path("reports") / "两个窗口_第N天对齐.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="inline")
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
