#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_INPUT = Path("/Users/zihao_/Documents/coding/dataset/processed/Core_Metrics_transposed.csv")


def load_core_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 日期解析：形如“2023年10月10日”
    date_col = "日(日期)"
    if date_col not in df.columns:
        raise KeyError(f"未找到日期列: {date_col}")
    try:
        df["date"] = pd.to_datetime(df[date_col], format="%Y年%m月%d日", errors="coerce")
    except Exception:
        # 兜底：替换中文年月日为连接符再解析
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


def compute_ma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()
    df[f"30日线索锁单转化率_MA{window}"] = (
        df["30日线索锁单转化率"].rolling(window=window, min_periods=1).mean()
    )
    df[f"7日线索锁单转化率_MA{window}"] = (
        df["7日线索锁单转化率"].rolling(window=window, min_periods=1).mean()
    )
    return df


def make_two_panel_chart(df: pd.DataFrame, window: int) -> go.Figure:
    fig = make_subplots(
        rows=20,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "日级原始转化率",
            f"移动平均转化率 (MA{window})",
            "",
            "线索量（日级）",
            f"线索量移动平均 (MA{window})",
            "",
            "在营门店数（30 日内有订单&当时已开业）",
            "",
            "试驾（日级）",
            "",
            "下发线索效率（日级）",
            "",
            "",
            "集中度分布（两个窗口）",
            "集中度（日级，Top10%门店占比，MA7）",
            "车型锁单（日级）",
            "车型上市/平销期指标",
            "上市期天数对比（两个窗口）",
            "年龄分布（锁单期分车型，bin=1）",
            "年龄统计对比（均值/中位数/标准差）",
        ),
        vertical_spacing=0.02,
        specs=[
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "table"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "table"}],
            [{"type": "xy"}],
            [{"type": "table"}],
            [{"type": "xy"}],
            [{"type": "table"}],
            [{"type": "xy"}],
            [{"type": "table"}],
            [{"type": "table"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "table"}],
            [{"type": "table"}],
            [{"type": "xy"}],
            [{"type": "table"}],
        ],
        row_heights=[0.19, 0.19, 0.14, 0.19, 0.19, 0.14, 0.18, 0.15, 0.18, 0.15, 0.18, 0.15, 0.14, 0.14, 0.14, 0.14, 0.20, 0.20, 0.18, 0.30],
    )

    fig.add_trace(
        go.Scatter(
            name="30日线索锁单转化率",
            x=df["date"],
            y=df["30日线索锁单转化率"],
            mode="lines",
            line=dict(color="#005783", width=2),
            hovertemplate="日期=%{x|%Y-%m-%d}<br>30日转化率=%{y:.2f}%<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name="7日线索锁单转化率",
            x=df["date"],
            y=df["7日线索锁单转化率"],
            mode="lines",
            line=dict(color="#27AD00", width=2),
            hovertemplate="日期=%{x|%Y-%m-%d}<br>7日转化率=%{y:.2f}%<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            name=f"30日转化率 MA{window}",
            x=df["date"],
            y=df[f"30日线索锁单转化率_MA{window}"],
            mode="lines",
            line=dict(color="#005783", width=3),
            hovertemplate="日期=%{x|%Y-%m-%d}<br>30日MA=%{y:.2f}%<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name=f"7日转化率 MA{window}",
            x=df["date"],
            y=df[f"7日线索锁单转化率_MA{window}"],
            mode="lines",
            line=dict(color="#27AD00", width=3),
            hovertemplate="日期=%{x|%Y-%m-%d}<br>7日MA=%{y:.2f}%<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="线索锁单转化率与线索量（原始与移动平均） + 在营门店数 + 试驾",
        legend_title="指标",
        margin=dict(l=40, r=40, t=60, b=200),
        height=5000,
    )
    fig.update_yaxes(title_text="转化率（%）", row=1, col=1)
    fig.update_yaxes(title_text="转化率（%）", row=2, col=1)
    fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
    fig.update_xaxes(rangeslider=dict(visible=False), row=2, col=1)
    return fig


def main():
    ap = argparse.ArgumentParser(description="读取日级核心指标数据，绘制两图：原始与移动平均折线（Plotly）")
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="输入CSV路径，默认使用processed/Core_Metrics_transposed.csv")
    ap.add_argument(
        "--out",
        default=None,
        help="输出HTML路径，默认写入 reports/线索锁单转化率_日级.html",
    )
    ap.add_argument("--ma-window", type=int, default=7, help="移动平均窗口大小，默认7")
    args = ap.parse_args()

    df = load_core_metrics(Path(args.input))
    df = compute_conversion_rates(df)
    df = compute_ma(df, args.ma_window)
    fig = make_two_panel_chart(df, args.ma_window)

    def _stats_for_window(d0: pd.Timestamp, d1: pd.Timestamp):
        m1 = "30日线索锁单转化率"
        m2 = "7日线索锁单转化率"
        dmask = (df["date"] >= d0) & (df["date"] <= d1)
        sub = df.loc[dmask, [m1, m2]].astype(float)
        def stats(series: pd.Series):
            s = series.dropna()
            return {
                "mean": float(s.mean()) if len(s) else float("nan"),
                "median": float(s.median()) if len(s) else float("nan"),
                "std": float(s.std()) if len(s) else float("nan"),
            }
        return stats(sub[m1]), stats(sub[m2])

    w2_start = pd.to_datetime("2023-12-01")
    w2_end = pd.to_datetime("2024-11-30")
    w1_start = pd.to_datetime("2024-12-01")
    w1_end = pd.to_datetime("2025-11-30")

    m1_w1, m2_w1 = _stats_for_window(w1_start, w1_end)
    m1_w2, m2_w2 = _stats_for_window(w2_start, w2_end)

    header_vals = [
        "指标",
        "7日 2024-12-01 至 2025-11-30",
        "7日 2023-12-01 至 2024-11-30",
        "7日 环比(%)",
        "30日 2024-12-01 至 2025-11-30",
        "30日 2023-12-01 至 2024-11-30",
        "30日 环比(%)",
    ]
    def fmtp(x: float) -> str:
        return "-" if pd.isna(x) else f"{x:.2f}%"
    def diffp(a: float, b: float) -> str:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return "-"
        return f"{((a - b) / b) * 100:+.2f}%"

    rows_metric = ["平均值(%)", "中位数(%)", "标准差(%)"]
    col_7_w1 = [fmtp(m2_w1["mean"]), fmtp(m2_w1["median"]), fmtp(m2_w1["std"])]
    col_7_w2 = [fmtp(m2_w2["mean"]), fmtp(m2_w2["median"]), fmtp(m2_w2["std"])]
    col_7_diff = [diffp(m2_w1["mean"], m2_w2["mean"]), diffp(m2_w1["median"], m2_w2["median"]), diffp(m2_w1["std"], m2_w2["std"])]
    col_30_w1 = [fmtp(m1_w1["mean"]), fmtp(m1_w1["median"]), fmtp(m1_w1["std"])]
    col_30_w2 = [fmtp(m1_w2["mean"]), fmtp(m1_w2["median"]), fmtp(m1_w2["std"])]
    col_30_diff = [diffp(m1_w1["mean"], m1_w2["mean"]), diffp(m1_w1["median"], m1_w2["median"]), diffp(m1_w1["std"], m1_w2["std"])]

    fig.add_table(
        header=dict(values=header_vals, fill_color="#f6f6f6", align="center", font=dict(color="#333", size=12), height=36),
        cells=dict(values=[rows_metric, col_7_w1, col_7_w2, col_7_diff, col_30_w1, col_30_w2, col_30_diff], align="center", height=32),
        row=3,
        col=1,
    )

    m_eff = "有效线索数"
    m_dis = "下发线索数"
    df[f"{m_eff}_MA{args.ma_window}"] = df[m_eff].astype(float).rolling(window=args.ma_window, min_periods=1).mean()
    df[f"{m_dis}_MA{args.ma_window}"] = df[m_dis].astype(float).rolling(window=args.ma_window, min_periods=1).mean()

    fig.add_trace(
        go.Scatter(
            name="有效线索数",
            x=df["date"],
            y=df[m_eff],
            mode="lines",
            line=dict(color="#6A5ACD", width=2),
            hovertemplate="日期=%{x|%Y-%m-%d}<br>有效线索数=%{y:.0f}<extra></extra>",
            showlegend=False,
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name="下发线索数",
            x=df["date"],
            y=df[m_dis],
            mode="lines",
            line=dict(color="#FF7F0E", width=2),
            hovertemplate="日期=%{x|%Y-%m-%d}<br>下发线索数=%{y:.0f}<extra></extra>",
            showlegend=False,
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name=f"有效线索数 MA{args.ma_window}",
            x=df["date"],
            y=df[f"{m_eff}_MA{args.ma_window}"],
            mode="lines",
            line=dict(color="#6A5ACD", width=3),
            hovertemplate="日期=%{x|%Y-%m-%d}<br>有效线索数MA=%{y:.0f}<extra></extra>",
            showlegend=False,
        ),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name=f"下发线索数 MA{args.ma_window}",
            x=df["date"],
            y=df[f"{m_dis}_MA{args.ma_window}"],
            mode="lines",
            line=dict(color="#FF7F0E", width=3),
            hovertemplate="日期=%{x|%Y-%m-%d}<br>下发线索数MA=%{y:.0f}<extra></extra>",
            showlegend=False,
        ),
        row=5,
        col=1,
    )

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
            bordercolor="#dddddd",
            borderwidth=1,
            font=dict(size=12, color="#333"),
            xanchor="left",
            yanchor="top",
        )

    add_local_legend("yaxis", [("30日线索锁单转化率", "#005783"), ("7日线索锁单转化率", "#27AD00")])
    add_local_legend("yaxis2", [(f"30日转化率 MA{args.ma_window}", "#005783"), (f"7日转化率 MA{args.ma_window}", "#27AD00")])
    add_local_legend("yaxis3", [("有效线索数", "#6A5ACD"), ("下发线索数", "#FF7F0E")])
    add_local_legend("yaxis4", [(f"有效线索数 MA{args.ma_window}", "#6A5ACD"), (f"下发线索数 MA{args.ma_window}", "#FF7F0E")])

    def stats_raw(d0: pd.Timestamp, d1: pd.Timestamp):
        dmask = (df["date"] >= d0) & (df["date"] <= d1)
        sub = df.loc[dmask, [m_eff, m_dis]].astype(float)
        def s(series: pd.Series):
            x = series.dropna()
            return {
                "mean": float(x.mean()) if len(x) else float("nan"),
                "median": float(x.median()) if len(x) else float("nan"),
                "std": float(x.std()) if len(x) else float("nan"),
            }
        return s(sub[m_eff]), s(sub[m_dis])

    eff_w1, dis_w1 = stats_raw(w1_start, w1_end)
    eff_w2, dis_w2 = stats_raw(w2_start, w2_end)

    hdr2 = [
        "指标",
        "有效 2024-12-01 至 2025-11-30",
        "有效 2023-12-01 至 2024-11-30",
        "有效 环比(%)",
        "下发 2024-12-01 至 2025-11-30",
        "下发 2023-12-01 至 2024-11-30",
        "下发 环比(%)",
    ]
    def diff_num(a: float, b: float) -> str:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return "-"
        return f"{((a - b) / b) * 100:+.2f}%"
    rmetr = ["平均值(数)", "中位数(数)", "标准差(数)"]
    col_eff_w1 = [f"{eff_w1['mean']:.0f}" if pd.notna(eff_w1['mean']) else "-", f"{eff_w1['median']:.0f}" if pd.notna(eff_w1['median']) else "-", f"{eff_w1['std']:.0f}" if pd.notna(eff_w1['std']) else "-"]
    col_eff_w2 = [f"{eff_w2['mean']:.0f}" if pd.notna(eff_w2['mean']) else "-", f"{eff_w2['median']:.0f}" if pd.notna(eff_w2['median']) else "-", f"{eff_w2['std']:.0f}" if pd.notna(eff_w2['std']) else "-"]
    col_dis_w1 = [f"{dis_w1['mean']:.0f}" if pd.notna(dis_w1['mean']) else "-", f"{dis_w1['median']:.0f}" if pd.notna(dis_w1['median']) else "-", f"{dis_w1['std']:.0f}" if pd.notna(dis_w1['std']) else "-"]
    col_dis_w2 = [f"{dis_w2['mean']:.0f}" if pd.notna(dis_w2['mean']) else "-", f"{dis_w2['median']:.0f}" if pd.notna(dis_w2['median']) else "-", f"{dis_w2['std']:.0f}" if pd.notna(dis_w2['std']) else "-"]

    col_eff_diff = [diff_num(eff_w1['mean'], eff_w2['mean']), diff_num(eff_w1['median'], eff_w2['median']), diff_num(eff_w1['std'], eff_w2['std'])]
    col_dis_diff = [diff_num(dis_w1['mean'], dis_w2['mean']), diff_num(dis_w1['median'], dis_w2['median']), diff_num(dis_w1['std'], dis_w2['std'])]

    fig.add_table(
        header=dict(values=hdr2, fill_color="#f6f6f6", align="center", font=dict(color="#333", size=12), height=36),
        cells=dict(values=[rmetr, col_eff_w1, col_eff_w2, col_eff_diff, col_dis_w1, col_dis_w2, col_dis_diff], align="center", height=32),
        row=6,
        col=1,
    )

    # 读取门店订单数据，计算在营门店数（日级）
    orders_path = Path("/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet")
    try:
        df_orders = pd.read_parquet(orders_path)
    except Exception:
        df_orders = pd.DataFrame()

    def resolve(df: pd.DataFrame, logical: str) -> str:
        cand = {
            "order_time": ["Order_Create_Time", "订单创建时间", "order_create_time", "下单时间", "Lock_Time", "锁单时间"],
            "store_create": ["store_create_date", "门店开业时间", "Store_Create_Date"],
            "store_name": ["Store Name", "门店名称", "store_name"],
        }
        for c in cand.get(logical, []):
            if c in df.columns:
                return c
        # 归一化匹配
        def norm(s: str) -> str:
            return s.strip().lower().replace("_", " ")
        need = [norm(x) for x in cand.get(logical, [])]
        col_map = {norm(x): x for x in df.columns}
        for t in need:
            if t in col_map:
                return col_map[t]
        raise KeyError(f"未找到列: {logical}; 备选={cand.get(logical)}")

    active_series = None
    if not df_orders.empty:
        col_time = None
        for c in ["Order_Create_Time", "Lock_Time"]:
            if c in df_orders.columns:
                col_time = c
                break
        if col_time is None:
            col_time = resolve(df_orders, "order_time")
        col_store = resolve(df_orders, "store_name")
        col_create = resolve(df_orders, "store_create")

        s_time = pd.to_datetime(df_orders[col_time], errors="coerce")
        s_day = s_time.dt.floor("D")
        s_store = df_orders[col_store].astype(str)
        s_open = pd.to_datetime(df_orders[col_create], errors="coerce").dt.floor("D")

        # 每店每日订单数
        daily = (
            pd.DataFrame({"store": s_store, "day": s_day})
            .dropna()
            .groupby(["store", "day"], dropna=False)
            .size()
            .reset_index(name="cnt")
        )
        # 构造日期范围（基于核心指标的范围）
        date_start = df["date"].min()
        date_end = df["date"].max()
        full_days = pd.date_range(date_start, date_end, freq="D")

        # 透视：日期 x 门店
        pivot = daily.pivot(index="day", columns="store", values="cnt").fillna(0)
        pivot = pivot.reindex(full_days, fill_value=0)
        roll = pivot.rolling(window=30, min_periods=1).sum()

        # 门店开业日序列
        open_map = (
            pd.DataFrame({"store": s_store, "open": s_open})
            .dropna()
            .groupby("store")
            .agg(open=("open", "min"))
        )
        open_series = open_map["open"] if "open" in open_map.columns else open_map.squeeze()

        active_counts = []
        for d in full_days:
            if roll.empty:
                active_counts.append(0)
                continue
            rs = roll.loc[d]
            # 开业判断：开业日期存在且不晚于当天
            open_mask = open_series.reindex(rs.index)
            open_ok = open_mask.notna() & (open_mask <= d)
            active = (rs > 0) & open_ok
            active_counts.append(int(active.sum()))

        active_series = pd.Series(active_counts, index=full_days, name="在营门店数")

        # 计算“下发线索门店数”日级序列（基于 first_assign_time 或订单时间）
        assign_col = None
        for c in ["first_assign_time", "Order_Create_Time", "Lock_Time"]:
            if c in df_orders.columns:
                assign_col = c
                break
        if assign_col is not None:
            assign_day = pd.to_datetime(df_orders[assign_col], errors="coerce").dt.floor("D")
            assign_store = df_orders[col_store].astype(str)
            assign_daily = (
                pd.DataFrame({"store": assign_store, "day": assign_day})
                .dropna()
                .groupby(["day"])
                .agg(store_cnt=("store", lambda s: len(pd.unique(s))))
                .reindex(full_days)
                .fillna(0)
            )
            assign_series = assign_daily["store_cnt"].rename("下发线索门店数")
        else:
            assign_series = pd.Series([pd.NA] * len(full_days), index=full_days, name="下发线索门店数")

    if active_series is not None:
        # 计算总门店数用于百分比统计
        total_stores = int(pd.Series(df_orders[col_store].astype(str)).nunique()) if not df_orders.empty else None

        fig.add_trace(
            go.Scatter(
                name="在营门店数（30 日内有订单&当时已开业）",
                x=active_series.index,
                y=active_series.values,
                mode="lines",
                line=dict(color="#008080", width=2),
                hovertemplate="日期=%{x|%Y-%m-%d}<br>在营门店数（30 日内有订单&当时已开业）=%{y:.0f}<extra></extra>",
                showlegend=False,
            ),
            row=7,
            col=1,
        )
        if 'assign_series' in locals():
            fig.add_trace(
                go.Scatter(
                    name="下发线索门店数",
                    x=assign_series.index,
                    y=assign_series.values,
                    mode="lines",
                    line=dict(color="#C93D3D", width=2),
                    hovertemplate="日期=%{x|%Y-%m-%d}<br>下发线索门店数=%{y:.0f}<extra></extra>",
                    showlegend=False,
                ),
                row=7,
                col=1,
            )

        add_local_legend("yaxis5", [("在营门店数（30 日内有订单&当时已开业）", "#008080"), ("下发线索门店数", "#C93D3D")])

        # 在营门店数 vs 下发线索门店数 窗口统计（仅环比为百分比，其他为数值）
        def stats_num(series: pd.Series, d0: pd.Timestamp, d1: pd.Timestamp):
            s = series.loc[(series.index >= d0) & (series.index <= d1)].dropna()
            return {
                'mean': float(s.mean()) if len(s) else float('nan'),
                'median': float(s.median()) if len(s) else float('nan'),
                'std': float(s.std()) if len(s) else float('nan'),
            }

        act_w1 = stats_num(active_series.astype(float), w1_start, w1_end)
        act_w2 = stats_num(active_series.astype(float), w2_start, w2_end)
        asn_w1 = stats_num(assign_series.astype(float), w1_start, w1_end) if 'assign_series' in locals() else {'mean': float('nan'), 'median': float('nan'), 'std': float('nan')}
        asn_w2 = stats_num(assign_series.astype(float), w2_start, w2_end) if 'assign_series' in locals() else {'mean': float('nan'), 'median': float('nan'), 'std': float('nan')}

        def fmtn(x: float) -> str:
            return '-' if pd.isna(x) else f"{x:.0f}"
        def diffp(a: float, b: float) -> str:
            if pd.isna(a) or pd.isna(b) or b == 0:
                return '-'
            return f"{((a - b) / b) * 100:+.2f}%"

        hdr3 = [
            '指标',
            '在营门店数（30 日内有订单&当时已开业） 2024-12-01 至 2025-11-30',
            '在营门店数（30 日内有订单&当时已开业） 2023-12-01 至 2024-11-30',
            '在营门店数（30 日内有订单&当时已开业） 环比(%)',
            '下发门店 2024-12-01 至 2025-11-30',
            '下发门店 2023-12-01 至 2024-11-30',
            '下发门店 环比(%)',
        ]
        r3 = ['平均值(数)', '中位数(数)', '标准差(数)']
        col_act_w1 = [fmtn(act_w1['mean']), fmtn(act_w1['median']), fmtn(act_w1['std'])]
        col_act_w2 = [fmtn(act_w2['mean']), fmtn(act_w2['median']), fmtn(act_w2['std'])]
        col_act_diff = [diffp(act_w1['mean'], act_w2['mean']), diffp(act_w1['median'], act_w2['median']), diffp(act_w1['std'], act_w2['std'])]
        col_asn_w1 = [fmtn(asn_w1['mean']), fmtn(asn_w1['median']), fmtn(asn_w1['std'])]
        col_asn_w2 = [fmtn(asn_w2['mean']), fmtn(asn_w2['median']), fmtn(asn_w2['std'])]
        col_asn_diff = [diffp(asn_w1['mean'], asn_w2['mean']), diffp(asn_w1['median'], asn_w2['median']), diffp(asn_w1['std'], asn_w2['std'])]

        fig.add_table(
            header=dict(values=hdr3, fill_color="#f6f6f6", align="center", font=dict(color="#333", size=12), height=36),
            cells=dict(values=[r3, col_act_w1, col_act_w2, col_act_diff, col_asn_w1, col_asn_w2, col_asn_diff], align="center", height=32),
            row=8,
            col=1,
        )

    # 试驾模块：试驾锁单数 vs 有效试驾数 折线对比 + 窗口表格
    m_td_lock = "试驾锁单数"
    m_td_eff = "有效试驾数"
    if m_td_lock in df.columns and m_td_eff in df.columns:
        fig.add_trace(
            go.Scatter(
                name=m_td_lock,
                x=df["date"],
                y=df[m_td_lock].astype(float),
                mode="lines",
                line=dict(color="#2E91E5", width=2),
                hovertemplate="日期=%{x|%Y-%m-%d}<br>试驾锁单数=%{y:.0f}<extra></extra>",
                showlegend=False,
            ),
            row=9,
            col=1,
        )
        df[f"{m_td_eff}_MA{args.ma_window}"] = df[m_td_eff].astype(float).rolling(window=args.ma_window, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                name=f"{m_td_eff} MA{args.ma_window}",
                x=df["date"],
                y=df[f"{m_td_eff}_MA{args.ma_window}"],
                mode="lines",
                line=dict(color="#E15F99", width=2),
                hovertemplate="日期=%{x|%Y-%m-%d}<br>有效试驾数MA=%{y:.0f}<extra></extra>",
                showlegend=False,
            ),
            row=9,
            col=1,
        )
        add_local_legend("yaxis6", [(m_td_lock, "#2E91E5"), (f"{m_td_eff} MA{args.ma_window}", "#E15F99")])

        # 派生比值（每日）：试驾锁单比值 = 试驾锁单数 / 有效试驾数
        ratio_daily = (df[m_td_lock].astype(float) / df[m_td_eff].replace(0, pd.NA).astype(float)) * 100.0

        def stats_series(series: pd.Series, d0: pd.Timestamp, d1: pd.Timestamp):
            s = series.loc[(df["date"] >= d0) & (df["date"] <= d1)].dropna()
            return {
                "mean": float(s.mean()) if len(s) else float("nan"),
                "median": float(s.median()) if len(s) else float("nan"),
                "std": float(s.std()) if len(s) else float("nan"),
            }

        td_lock_w1 = stats_series(df[m_td_lock].astype(float), w1_start, w1_end)
        td_lock_w2 = stats_series(df[m_td_lock].astype(float), w2_start, w2_end)
        td_eff_w1 = stats_series(df[m_td_eff].astype(float), w1_start, w1_end)
        td_eff_w2 = stats_series(df[m_td_eff].astype(float), w2_start, w2_end)
        ratio_w1 = stats_series(ratio_daily, w1_start, w1_end)
        ratio_w2 = stats_series(ratio_daily, w2_start, w2_end)

        def fmtn(x: float) -> str:
            return "-" if pd.isna(x) else f"{x:.0f}"
        def fmtp(x: float) -> str:
            return "-" if pd.isna(x) else f"{x:.2f}%"
        def diffp(a: float, b: float) -> str:
            if pd.isna(a) or pd.isna(b) or b == 0:
                return "-"
            return f"{((a - b) / b) * 100:+.2f}%"

        hdr_td = [
            "指标",
            "试驾锁单比值 2024-12-01 至 2025-11-30",
            "试驾锁单比值 2023-12-01 至 2024-11-30",
            "试驾锁单比值 环比(%)",
            "试驾锁单占比 2024-12-01 至 2025-11-30",
            "试驾锁单占比 2023-12-01 至 2024-11-30",
            "试驾锁单占比 环比(%)",
            "有效试驾数 2024-12-01 至 2025-11-30",
            "有效试驾数 2023-12-01 至 2024-11-30",
            "有效试驾数 环比(%)",
        ]
        r_td = ["平均值", "中位数", "标准差"]
        col_ratio_w1 = [fmtp(ratio_w1["mean"]), fmtp(ratio_w1["median"]), fmtp(ratio_w1["std"])]
        col_ratio_w2 = [fmtp(ratio_w2["mean"]), fmtp(ratio_w2["median"]), fmtp(ratio_w2["std"])]
        col_ratio_diff = [diffp(ratio_w1["mean"], ratio_w2["mean"]), diffp(ratio_w1["median"], ratio_w2["median"]), diffp(ratio_w1["std"], ratio_w2["std"])]

        m_lock_total = "锁单数"
        if m_lock_total in df.columns:
            share_daily = (df[m_td_lock].astype(float) / df[m_lock_total].replace(0, pd.NA).astype(float)) * 100.0
        else:
            share_daily = pd.Series(index=df.index, dtype=float)

        share_w1 = stats_series(share_daily, w1_start, w1_end)
        share_w2 = stats_series(share_daily, w2_start, w2_end)
        col_share_w1 = [fmtp(share_w1["mean"]), fmtp(share_w1["median"]), fmtp(share_w1["std"])]
        col_share_w2 = [fmtp(share_w2["mean"]), fmtp(share_w2["median"]), fmtp(share_w2["std"])]
        col_share_diff = [diffp(share_w1["mean"], share_w2["mean"]), diffp(share_w1["median"], share_w2["median"]), diffp(share_w1["std"], share_w2["std"])]
        col_eff_w1 = [fmtn(td_eff_w1["mean"]), fmtn(td_eff_w1["median"]), fmtn(td_eff_w1["std"])]
        col_eff_w2 = [fmtn(td_eff_w2["mean"]), fmtn(td_eff_w2["median"]), fmtn(td_eff_w2["std"])]
        col_eff_diff = [diffp(td_eff_w1["mean"], td_eff_w2["mean"]), diffp(td_eff_w1["median"], td_eff_w2["median"]), diffp(td_eff_w1["std"], td_eff_w2["std"])]

        fig.add_table(
            header=dict(values=hdr_td, fill_color="#f6f6f6", align="center", font=dict(color="#333", size=12), height=36),
            cells=dict(values=[r_td, col_ratio_w1, col_ratio_w2, col_ratio_diff, col_share_w1, col_share_w2, col_share_diff, col_eff_w1, col_eff_w2, col_eff_diff], align="center", height=32),
            row=10,
            col=1,
        )

    m_dis_eff = "下发线索数"
    m_store_cnt = "下发线索门店数"
    m_owner_cnt = "下发线索主理数"
    if m_dis_eff in df.columns and m_store_cnt in df.columns and m_owner_cnt in df.columns:
        store_avg_daily = df[m_dis_eff].astype(float) / df[m_store_cnt].replace(0, pd.NA).astype(float)
        owner_avg_daily = df[m_dis_eff].astype(float) / df[m_owner_cnt].replace(0, pd.NA).astype(float)

        fig.add_trace(
            go.Scatter(
                name="店均下发线索数",
                x=df["date"],
                y=store_avg_daily,
                mode="lines",
                line=dict(color="#9467BD", width=2),
                hovertemplate="日期=%{x|%Y-%m-%d}<br>店均下发线索数=%{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=11,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name="主理人均下发线索量",
                x=df["date"],
                y=owner_avg_daily,
                mode="lines",
                line=dict(color="#17BECF", width=2),
                hovertemplate="日期=%{x|%Y-%m-%d}<br>主理人均下发线索量=%{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=11,
            col=1,
        )
        add_local_legend("yaxis7", [("店均下发线索数", "#9467BD"), ("主理人均下发线索量", "#17BECF")])

        def stats_series(series: pd.Series, d0: pd.Timestamp, d1: pd.Timestamp):
            s = series.loc[(df["date"] >= d0) & (df["date"] <= d1)].dropna()
            return {
                "mean": float(s.mean()) if len(s) else float("nan"),
                "median": float(s.median()) if len(s) else float("nan"),
                "std": float(s.std()) if len(s) else float("nan"),
            }

        store_w1 = stats_series(store_avg_daily, w1_start, w1_end)
        store_w2 = stats_series(store_avg_daily, w2_start, w2_end)
        owner_w1 = stats_series(owner_avg_daily, w1_start, w1_end)
        owner_w2 = stats_series(owner_avg_daily, w2_start, w2_end)

        def fmtn2(x: float) -> str:
            return "-" if pd.isna(x) else f"{x:.2f}"
        def diffp(a: float, b: float) -> str:
            if pd.isna(a) or pd.isna(b) or b == 0:
                return "-"
            return f"{((a - b) / b) * 100:+.2f}%"

        hdr_eff = [
            "指标",
            "店均下发线索数 2024-12-01 至 2025-11-30",
            "店均下发线索数 2023-12-01 至 2024-11-30",
            "店均下发线索数 环比(%)",
            "主理人均下发线索量 2024-12-01 至 2025-11-30",
            "主理人均下发线索量 2023-12-01 至 2024-11-30",
            "主理人均下发线索量 环比(%)",
        ]
        r_eff = ["平均值(数)", "中位数(数)", "标准差(数)"]
        col_store_w1 = [fmtn2(store_w1["mean"]), fmtn2(store_w1["median"]), fmtn2(store_w1["std"])]
        col_store_w2 = [fmtn2(store_w2["mean"]), fmtn2(store_w2["median"]), fmtn2(store_w2["std"])]
        col_store_diff = [diffp(store_w1["mean"], store_w2["mean"]), diffp(store_w1["median"], store_w2["median"]), diffp(store_w1["std"], store_w2["std"])]
        col_owner_w1 = [fmtn2(owner_w1["mean"]), fmtn2(owner_w1["median"]), fmtn2(owner_w1["std"])]
        col_owner_w2 = [fmtn2(owner_w2["mean"]), fmtn2(owner_w2["median"]), fmtn2(owner_w2["std"])]
        col_owner_diff = [diffp(owner_w1["mean"], owner_w2["mean"]), diffp(owner_w1["median"], owner_w2["median"]), diffp(owner_w1["std"], owner_w2["std"])]

        fig.add_table(
            header=dict(values=hdr_eff, fill_color="#f6f6f6", align="center", font=dict(color="#333", size=12), height=36),
            cells=dict(values=[r_eff, col_store_w1, col_store_w2, col_store_diff, col_owner_w1, col_owner_w2, col_owner_diff], align="center", height=32),
            row=12,
            col=1,
        )

    # 锁单门店与集中度（窗口对比） - 使用 intention_order_analysis.parquet
    orders_path = Path("/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet")
    try:
        df_orders2 = pd.read_parquet(orders_path)
    except Exception:
        df_orders2 = pd.DataFrame()

    def resolve2(df: pd.DataFrame, logical: str) -> str:
        cand = {
            "lock_time": ["Lock_Time", "Lock Time", "锁单时间"],
            "store_name": ["Store Name", "门店名称", "store_name"],
        }
        for c in cand.get(logical, []):
            if c in df.columns:
                return c
        # 归一化匹配
        def norm(s: str) -> str:
            return s.strip().lower().replace("_", " ")
        need = [norm(x) for x in cand.get(logical, [])]
        col_map = {norm(x): x for x in df.columns}
        for t in need:
            if t in col_map:
                return col_map[t]
        return cand.get(logical, [""])[0]

    def window_lock_stats(d0: pd.Timestamp, d1: pd.Timestamp):
        if df_orders2.empty:
            return float("nan"), float("nan"), float("nan")
        col_lock = resolve2(df_orders2, "lock_time")
        col_store = resolve2(df_orders2, "store_name")
        s_lock = pd.to_datetime(df_orders2[col_lock], errors="coerce")
        mask = s_lock.between(d0, d1, inclusive="both")
        sub = df_orders2.loc[mask, [col_store]].dropna()
        if sub.empty:
            return float("nan"), float("nan"), float("nan")
        counts = sub.groupby(col_store).size()
        store_cnt = int(counts.index.nunique())
        total_locks = int(counts.sum())
        avg_per_store = (total_locks / store_cnt) if store_cnt > 0 else float("nan")
        if total_locks > 0 and store_cnt > 0:
            top_k = max(1, int((store_cnt * 0.10) + 0.9999))
            top_sum = counts.sort_values(ascending=False).head(top_k).sum()
            conc = float(top_sum) / float(total_locks)
        else:
            conc = float("nan")
        return float(store_cnt), float(avg_per_store), float(conc)

    s1_cnt, s1_avg, s1_hhi = window_lock_stats(w1_start, w1_end)
    s2_cnt, s2_avg, s2_hhi = window_lock_stats(w2_start, w2_end)

    def fmti(x: float) -> str:
        return "-" if pd.isna(x) else f"{x:.0f}"
    def fmtd(x: float) -> str:
        return "-" if pd.isna(x) else f"{x:.2f}"
    def fmt_conc(x: float) -> str:
        return "-" if pd.isna(x) else f"{x:.2f}"
    def diffp2(a: float, b: float) -> str:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return "-"
        return f"{((a - b) / b) * 100:+.2f}%"

    hdr_lock = [
        "指标",
        "锁单门店数 2024-12-01 至 2025-11-30",
        "锁单门店数 2023-12-01 至 2024-11-30",
        "锁单门店数 环比(%)",
        "店均锁单数 2024-12-01 至 2025-11-30",
        "店均锁单数 2023-12-01 至 2024-11-30",
        "店均锁单数 环比(%)",
        "集中度 2024-12-01 至 2025-11-30",
        "集中度 2023-12-01 至 2024-11-30",
        "集中度 环比(%)",
    ]
    r_lock = ["数值"]
    col_storecnt_w1 = [fmti(s1_cnt)]
    col_storecnt_w2 = [fmti(s2_cnt)]
    col_storecnt_diff = [diffp2(s1_cnt, s2_cnt)]
    col_avg_w1 = [fmtd(s1_avg)]
    col_avg_w2 = [fmtd(s2_avg)]
    col_avg_diff = [diffp2(s1_avg, s2_avg)]
    col_hhi_w1 = [fmt_conc(s1_hhi)]
    col_hhi_w2 = [fmt_conc(s2_hhi)]
    col_hhi_diff = [diffp2(s1_hhi, s2_hhi)]

    fig.add_table(
        header=dict(values=hdr_lock, fill_color="#f6f6f6", align="center", font=dict(color="#333", size=12), height=36),
        cells=dict(values=[r_lock, col_storecnt_w1, col_storecnt_w2, col_storecnt_diff, col_avg_w1, col_avg_w2, col_avg_diff, col_hhi_w1, col_hhi_w2, col_hhi_diff], align="center", height=32),
        row=13,
        col=1,
    )

    def decile_distribution(d0: pd.Timestamp, d1: pd.Timestamp):
        if df_orders2.empty:
            return [float("nan")] * 10, [float("nan")] * 10
        col_lock = resolve2(df_orders2, "lock_time")
        col_store = resolve2(df_orders2, "store_name")
        s_lock = pd.to_datetime(df_orders2[col_lock], errors="coerce")
        mask = s_lock.between(d0, d1, inclusive="both")
        sub = df_orders2.loc[mask, [col_store]].dropna()
        if sub.empty:
            return [float("nan")] * 10, [float("nan")] * 10
        counts = sub.groupby(col_store).size().sort_values(ascending=False)
        s = int(counts.index.nunique())
        t = float(counts.sum())
        if s == 0 or t == 0:
            return [float("nan")] * 10, [float("nan")] * 10
        chunk = max(1, int((s * 0.10) + 0.9999))
        parts = []
        for k in range(1, 11):
            start = (k - 1) * chunk
            end = k * chunk
            part = float(counts.iloc[start:end].sum())
            parts.append(part)
        cum = []
        acc = 0.0
        for v in parts:
            acc += v
            cum.append(acc / t * 100.0)
        return cum, parts

    bins = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    d1_vals, d1_counts = decile_distribution(w1_start, w1_end)
    d2_vals, d2_counts = decile_distribution(w2_start, w2_end)

    fig.add_trace(
        go.Scatter(
            name="集中度分布 2024-12-01 至 2025-11-30",
            x=bins,
            y=d1_vals,
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
            text=["-" if pd.isna(x) else f"{int(x)}" for x in d1_counts],
            hovertemplate="分位=%{x}<br>累计占比=%{y:.2f}%<br>锁单数=%{text}<extra></extra>",
            showlegend=False,
        ),
        row=14,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name="集中度分布 2023-12-01 至 2024-11-30",
            x=bins,
            y=d2_vals,
            mode="lines+markers",
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=6),
            text=["-" if pd.isna(x) else f"{int(x)}" for x in d2_counts],
            hovertemplate="分位=%{x}<br>累计占比=%{y:.2f}%<br>锁单数=%{text}<extra></extra>",
            showlegend=False,
        ),
        row=14,
        col=1,
    )
    add_local_legend("yaxis8", [("集中度分布 2024-12-01 至 2025-11-30", "#1f77b4"), ("集中度分布 2023-12-01 至 2024-11-30", "#ff7f0e")])

    def daily_top10_share():
        if df_orders2.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
        col_lock = resolve2(df_orders2, "lock_time")
        col_store = resolve2(df_orders2, "store_name")
        s_lock = pd.to_datetime(df_orders2[col_lock], errors="coerce")
        s_day = s_lock.dt.floor("D")
        s_store = df_orders2[col_store].astype(str)
        daily = (
            pd.DataFrame({"store": s_store, "day": s_day})
            .dropna()
            .groupby(["store", "day"], dropna=False)
            .size()
            .reset_index(name="cnt")
        )
        date_start = df["date"].min()
        date_end = df["date"].max()
        full_days = pd.date_range(date_start, date_end, freq="D")
        pivot = daily.pivot(index="day", columns="store", values="cnt").fillna(0)
        pivot = pivot.reindex(full_days, fill_value=0)
        shares = []
        for d in full_days:
            if pivot.empty:
                shares.append(float("nan"))
                continue
            rs = pivot.loc[d]
            tot = float(rs.sum())
            if tot <= 0:
                shares.append(float("nan"))
                continue
            s = int((rs > 0).sum())
            k = max(1, int((s * 0.10) + 0.9999))
            top_sum = float(rs.sort_values(ascending=False).head(k).sum())
            shares.append(top_sum / tot * 100.0)
        series = pd.Series(shares, index=full_days, name="Top10%门店锁单占比(日级)")
        s1 = series.loc[(series.index >= w1_start) & (series.index <= w1_end)]
        s2 = series.loc[(series.index >= w2_start) & (series.index <= w2_end)]
        return series, s1, s2

    s_all, s_w1, s_w2 = daily_top10_share()
    s_all_ma = s_all.rolling(window=args.ma_window, min_periods=1).mean()
    s_w1_ma = s_all_ma.loc[(s_all_ma.index >= w1_start) & (s_all_ma.index <= w1_end)]
    s_w2_ma = s_all_ma.loc[(s_all_ma.index >= w2_start) & (s_all_ma.index <= w2_end)]
    if len(s_w1_ma):
        fig.add_trace(
            go.Scatter(
                name=f"Top10%门店占比（日级） MA{args.ma_window} 2024-12-01 至 2025-11-30",
                x=s_w1_ma.index,
                y=s_w1_ma.values,
                mode="lines",
                line=dict(color="#1f77b4", width=2),
                hovertemplate="日期=%{x|%Y-%m-%d}<br>占比=%{y:.2f}%<extra></extra>",
                showlegend=False,
            ),
            row=15,
            col=1,
        )
    if len(s_w2_ma):
        fig.add_trace(
            go.Scatter(
                name=f"Top10%门店占比（日级） MA{args.ma_window} 2023-12-01 至 2024-11-30",
                x=s_w2_ma.index,
                y=s_w2_ma.values,
                mode="lines",
                line=dict(color="#ff7f0e", width=2),
                hovertemplate="日期=%{x|%Y-%m-%d}<br>占比=%{y:.2f}%<extra></extra>",
                showlegend=False,
            ),
            row=15,
            col=1,
        )
    add_local_legend("yaxis9", [(f"Top10%门店占比（日级） MA{args.ma_window} 2024-12-01 至 2025-11-30", "#1f77b4"), (f"Top10%门店占比（日级） MA{args.ma_window} 2023-12-01 至 2024-11-30", "#ff7f0e")])

    def resolve_model(df: pd.DataFrame) -> str:
        for c in ["车型分组", "model_group", "car_model_group", "车型"]:
            if c in df.columns:
                return c
        return "model_group"

    if not df_orders2.empty:
        col_lock = resolve2(df_orders2, "lock_time")
        col_model = resolve_model(df_orders2)
        s_lock = pd.to_datetime(df_orders2[col_lock], errors="coerce")
        s_day = s_lock.dt.floor("D")
        s_model = df_orders2[col_model].astype(str)
        def map_model(x: str) -> str:
            if x in {"CM0", "CM1", "CM2"}:
                return "LS6"
            if x in {"DM0", "DM1"}:
                return "L6"
            if x == "LS9":
                return "LS9"
            return ""
        m = s_model.map(map_model)
        daily = (
            pd.DataFrame({"day": s_day, "model": m})
            .dropna()
            .query("model != ''")
            .groupby(["day", "model"], dropna=False)
            .size()
            .reset_index(name="cnt")
        )
        date_start = df["date"].min()
        date_end = df["date"].max()
        full_days = pd.date_range(date_start, date_end, freq="D")
        pivot = daily.pivot(index="day", columns="model", values="cnt").fillna(0)
        pivot = pivot.reindex(full_days, fill_value=0)
        for name, color in [("LS6", "#2E91E5"), ("L6", "#E15F99"), ("LS9", "#7B848F")]:
            series = pivot[name] if name in pivot.columns else pd.Series([0]*len(full_days), index=full_days)
            fig.add_trace(
                go.Scatter(
                    name=name,
                    x=series.index,
                    y=series.values,
                    mode="lines",
                    line=dict(color=color, width=2),
                    hovertemplate="日期=%{x|%Y-%m-%d}<br>锁单数=%{y:.0f}<extra></extra>",
                    showlegend=False,
                ),
                row=16,
                col=1,
            )
        add_local_legend("yaxis10", [("LS6", "#2E91E5"), ("L6", "#E15F99"), ("LS9", "#7B848F")])

        periods = {
            "CM0": {"start": "2023-10-12", "end": "2023-11-12"},
            "DM0": {"start": "2024-05-13", "end": "2024-06-15"},
            "CM1": {"start": "2024-09-26", "end": "2024-10-20"},
            "CM2": {"start": "2025-09-10", "end": "2025-10-15"},
            "DM1": {"start": "2025-05-13", "end": "2025-06-15"},
            "LS9": {"start": "2025-11-12", "end": "2025-12-04"},
        }
        col_lock2 = resolve2(df_orders2, "lock_time")
        col_model2 = resolve_model(df_orders2)
        # 参考 lock_summary.py 的 CM2 拆分逻辑：使用产品名称识别增程与否
        def resolve_product(df: pd.DataFrame) -> str:
            for c in ["productname", "ProductName", "Product Name", "产品名称", "商品名称"]:
                if c in df.columns:
                    return c
            return "productname"
        col_product2 = resolve_product(df_orders2)
        s_lock2 = pd.to_datetime(df_orders2[col_lock2], errors="coerce")
        s_model2 = df_orders2[col_model2].astype(str)
        s_product2 = df_orders2[col_product2].astype(str)
        # 构造分类：非 CM2 保持原值；CM2 根据产品名称包含 52/66 拆分为 “CM2 增程”/“CM2”
        s_class2 = s_model2.copy()
        is_cm2 = s_model2.str.upper() == "CM2"
        is_range_ext = s_product2.str.contains(r"52|66", case=False, na=False)
        s_class2.loc[is_cm2 & is_range_ext] = "CM2 增程"
        s_class2.loc[is_cm2 & ~is_range_ext] = "CM2"
        def _mask_for(model_code: str):
            if model_code == "CM2":
                return s_class2.isin(["CM2", "CM2 增程"])
            if model_code == "CM2 增程":
                return s_class2 == "CM2 增程"
            return s_model2 == model_code
        def count_in_range(model_code: str, d0: pd.Timestamp, d1: pd.Timestamp) -> int:
            mask = _mask_for(model_code) & s_lock2.between(d0, d1, inclusive="both")
            return int(mask.sum())
        def count_in_next_30(model_code: str, d_end: pd.Timestamp) -> int:
            d0 = d_end
            d1 = d_end + pd.Timedelta(days=30)
            mask = _mask_for(model_code) & (s_lock2 >= d0) & (s_lock2 < d1)
            return int(mask.sum())
        def count_in_next_60(model_code: str, d_end: pd.Timestamp) -> tuple:
            d0 = d_end + pd.Timedelta(days=30)
            max_dt = pd.to_datetime(s_lock2.max()).floor('D')
            d1_cap = d_end + pd.Timedelta(days=60)
            # cap end to max_day (exclusive per half-open interval)
            d1 = min(d1_cap, max_dt)
            if pd.isna(d0) or pd.isna(d1) or d1 <= d0:
                return 0, 0
            days = (d1 - d0).days
            mask = _mask_for(model_code) & (s_lock2 >= d0) & (s_lock2 < d1)
            return int(mask.sum()), int(days)
        models = ["CM0", "DM0", "CM1", "CM2", "CM2 增程", "DM1", "LS9"]
        row_days = []
        row_list_sum = []
        row_list_avg = []
        row_flat_sum = []
        row_flat_avg = []
        row_flat60_sum = []
        row_flat60_avg = []
        for mcode in models:
            pcode = "CM2" if mcode == "CM2 增程" else mcode
            d0 = pd.to_datetime(periods[pcode]["start"]) if pcode in periods else pd.NaT
            d1 = pd.to_datetime(periods[pcode]["end"]) if pcode in periods else pd.NaT
            if pd.isna(d0) or pd.isna(d1) or d1 < d0:
                row_days.append("-")
                row_list_sum.append("-")
                row_list_avg.append("-")
                row_flat_sum.append("-")
                row_flat_avg.append("-")
                continue
            days = (d1 - d0).days + 1
            list_sum = count_in_range(mcode, d0, d1)
            list_avg = (list_sum / days) if days > 0 else float("nan")
            flat_sum = count_in_next_30(mcode, d1)
            flat_avg = (flat_sum / 30.0)
            row_days.append(f"{days:.0f}")
            row_list_sum.append(f"{list_sum:.0f}")
            row_list_avg.append("-" if pd.isna(list_avg) else f"{list_avg:.0f}")
            row_flat_sum.append(f"{flat_sum:.0f}")
            row_flat_avg.append("-" if pd.isna(flat_avg) else f"{flat_avg:.0f}")
            flat60_sum, flat60_days = count_in_next_60(mcode, d1)
            flat60_avg = (flat60_sum / flat60_days) if flat60_days > 0 else float('nan')
            row_flat60_sum.append(f"{flat60_sum:.0f}")
            row_flat60_avg.append("-" if pd.isna(flat60_avg) else f"{flat60_avg:.0f}")

        hdr_period = ["指标"] + models
        r_names = [
            "上市期天数",
            "上市期锁单数",
            "上市期锁单日均",
            "平销期30 日锁单数",
            "平销期30 日日均锁单数",
            "平销期60 日锁单数",
            "平销期60 日日均锁单数",
        ]
        cols = [r_names]
        for i, _ in enumerate(models):
            cols.append([
                row_days[i],
                row_list_sum[i],
                row_list_avg[i],
                row_flat_sum[i],
                row_flat_avg[i],
                row_flat60_sum[i] if i < len(row_flat60_sum) else "-",
                row_flat60_avg[i] if i < len(row_flat60_avg) else "-",
            ])
        # column color grouping: CM series, DM series, LS9
        cm_color = "#EAF2FF"
        dm_color = "#FFF2E6"
        ls9_color = "#F2F2F2"
        hdr_colors = ["#f6f6f6"] + [cm_color if m.startswith("CM") else (dm_color if m.startswith("DM") else ls9_color) for m in models]
        cell_colors = ["#FFFFFF"] + [cm_color if m.startswith("CM") else (dm_color if m.startswith("DM") else ls9_color) for m in models]
        fig.add_table(
            header=dict(values=hdr_period, fill_color=hdr_colors, align="center", font=dict(color="#333", size=12), height=36),
            cells=dict(values=cols, fill_color=cell_colors, align="center", height=32),
            row=17,
            col=1,
        )

        def days_overlap(d0: pd.Timestamp, d1: pd.Timestamp, w0: pd.Timestamp, w1: pd.Timestamp) -> int:
            a0 = max(d0, w0)
            a1 = min(d1, w1)
            if pd.isna(a0) or pd.isna(a1) or a1 < a0:
                return 0
            return int((a1 - a0).days) + 1

        models_period = list(periods.keys())
        next_map = {"CM0": "CM1", "CM1": "CM2", "DM0": "DM1"}
        max_dt = pd.to_datetime(s_lock2.max(), errors="coerce")
        max_dt = pd.to_datetime(max_dt).floor("D") if pd.notna(max_dt) else pd.NaT
        w1_list_days = []
        w1_non_list_days = []
        w2_list_days = []
        w2_non_list_days = []
        w1_list_orders = []
        w1_non_list_orders = []
        w2_list_orders = []
        w2_non_list_orders = []
        for mcode in models_period:
            d0 = pd.to_datetime(periods[mcode]["start"]) if mcode in periods else pd.NaT
            d1 = pd.to_datetime(periods[mcode]["end"]) if mcode in periods else pd.NaT
            ol1 = days_overlap(d0, d1, w1_start, w1_end)
            ol2 = days_overlap(d0, d1, w2_start, w2_end)
            # 非上市期：从上市期结束到改款车型的 start-1，与窗口交集；若无改款则为 [end, max_lock_time]
            nxt = next_map.get(mcode)
            if nxt and nxt in periods:
                nxt_start = pd.to_datetime(periods[nxt]["start"]) if periods[nxt].get("start") else pd.NaT
            else:
                nxt_start = pd.NaT
            if pd.notna(d1):
                if pd.notna(nxt_start) and (nxt_start > d1):
                    non_list_end = nxt_start - pd.Timedelta(days=1)
                elif pd.notna(max_dt) and (max_dt >= d1):
                    non_list_end = max_dt
                else:
                    non_list_end = pd.NaT
            else:
                non_list_end = pd.NaT
            if pd.notna(non_list_end):
                nl1 = days_overlap(d1, non_list_end, w1_start, w1_end)
                nl2 = days_overlap(d1, non_list_end, w2_start, w2_end)
            else:
                nl1 = 0
                nl2 = 0
            w1_list_days.append(f"{ol1:.0f}")
            w1_non_list_days.append(f"{nl1:.0f}")
            w2_list_days.append(f"{ol2:.0f}")
            w2_non_list_days.append(f"{nl2:.0f}")
            if ol1 > 0:
                ls1_start = max(d0, w1_start)
                ls1_end = min(d1, w1_end)
                mask_ls1 = _mask_for(mcode) & s_lock2.between(ls1_start, ls1_end, inclusive="both")
                w1_list_orders.append(f"{int(mask_ls1.sum()):.0f}")
            else:
                w1_list_orders.append("-")
            if nl1 > 0 and pd.notna(d1) and pd.notna(non_list_end):
                nls1_start = max(d1, w1_start)
                nls1_end = min(non_list_end, w1_end)
                mask_nls1 = _mask_for(mcode) & s_lock2.between(nls1_start, nls1_end, inclusive="both")
                w1_non_list_orders.append(f"{int(mask_nls1.sum()):.0f}")
            else:
                w1_non_list_orders.append("-")
            if ol2 > 0:
                ls2_start = max(d0, w2_start)
                ls2_end = min(d1, w2_end)
                mask_ls2 = _mask_for(mcode) & s_lock2.between(ls2_start, ls2_end, inclusive="both")
                w2_list_orders.append(f"{int(mask_ls2.sum()):.0f}")
            else:
                w2_list_orders.append("-")
            if nl2 > 0 and pd.notna(d1) and pd.notna(non_list_end):
                nls2_start = max(d1, w2_start)
                nls2_end = min(non_list_end, w2_end)
                mask_nls2 = _mask_for(mcode) & s_lock2.between(nls2_start, nls2_end, inclusive="both")
                w2_non_list_orders.append(f"{int(mask_nls2.sum()):.0f}")
            else:
                w2_non_list_orders.append("-")

        hdr_days = [
            "车型",
            "上市天数 2024-12-01 至 2025-11-30",
            "上市锁单数 2024-12-01 至 2025-11-30",
            "非上市天数 2024-12-01 至 2025-11-30",
            "非上市锁单数 2024-12-01 至 2025-11-30",
            "上市天数 2023-12-01 至 2024-11-30",
            "上市锁单数 2023-12-01 至 2024-11-30",
            "非上市天数 2023-12-01 至 2024-11-30",
            "非上市锁单数 2023-12-01 至 2024-11-30",
        ]
        fig.add_table(
            header=dict(values=hdr_days, fill_color="#f6f6f6", align="center", font=dict(color="#333", size=12), height=36),
            cells=dict(values=[models_period, w1_list_days, w1_list_orders, w1_non_list_days, w1_non_list_orders, w2_list_days, w2_list_orders, w2_non_list_days, w2_non_list_orders], align="center", height=32),
            row=18,
            col=1,
        )

        export_dir = Path("/Users/zihao_/Documents/coding/dataset/processed/analysis_results")
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = export_dir / "上市期天数对比_两个窗口.csv"
        df_export = pd.DataFrame({
            "车型": models_period,
            "上市天数 2024-12-01 至 2025-11-30": w1_list_days,
            "上市锁单数 2024-12-01 至 2025-11-30": w1_list_orders,
            "非上市天数 2024-12-01 至 2025-11-30": w1_non_list_days,
            "非上市锁单数 2024-12-01 至 2025-11-30": w1_non_list_orders,
            "上市天数 2023-12-01 至 2024-11-30": w2_list_days,
            "上市锁单数 2023-12-01 至 2024-11-30": w2_list_orders,
            "非上市天数 2023-12-01 至 2024-11-30": w2_non_list_days,
            "非上市锁单数 2023-12-01 至 2024-11-30": w2_non_list_orders,
        })
        df_export.to_csv(export_path, index=False, encoding="utf-8-sig")
        def resolve_owner_age(df: pd.DataFrame) -> str:
            cands = ["owner_age", "Owner Age", "Age", "age", "车主年龄", "客户年龄"]
            for c in cands:
                if c in df.columns:
                    return c
            def norm(s: str) -> str:
                return s.strip().lower().replace("_", " ")
            m = {norm(c): c for c in df.columns}
            for c in cands:
                cn = norm(c)
                if cn in m:
                    return m[cn]
            return cands[0]

        age_col = resolve_owner_age(df_orders2)
        ages = pd.to_numeric(df_orders2[age_col], errors="coerce")
        lock_times = s_lock2
        model_codes = s_model2.astype(str)
        def phase_range(model: str) -> tuple:
            d0 = pd.to_datetime(periods[model]["start"]) if model in periods else pd.NaT
            nxt = next_map.get(model)
            if nxt and nxt in periods:
                nxt_start = pd.to_datetime(periods[nxt]["start"]) if periods[nxt].get("start") else pd.NaT
                d1 = nxt_start - pd.Timedelta(days=1) if pd.notna(nxt_start) else pd.NaT
            else:
                d1 = max_dt
            return d0, d1
        bins = list(range(10, 81))
        x_vals = list(range(10, 80))
        color_map = {
            "CM0": "#2E91E5",
            "CM1": "#4378BF",
            "CM2": "#6AA2E5",
            "DM0": "#E15F99",
            "DM1": "#F3B3C5",
            "LS9": "#7B848F",
        }
        def hex_to_rgba(h: str, a: float) -> str:
            h = h.lstrip("#")
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            return f"rgba({r},{g},{b},{a})"
        series_legend = []
        for m in ["CM0", "CM1", "CM2", "DM0", "DM1", "LS9"]:
            d0, d1 = phase_range(m)
            if pd.isna(d0) or pd.isna(d1) or d1 < d0:
                continue
            mask = (model_codes == m) & lock_times.between(d0, d1, inclusive="both")
            ages_m = ages.loc[mask]
            ages_m = ages_m[(ages_m >= 10) & (ages_m <= 80)]
            if ages_m.empty:
                continue
            counts, _ = np.histogram(ages_m.values, bins=bins)
            total = counts.sum()
            perc = (counts / total * 100.0) if total > 0 else counts.astype(float)
            fig.add_trace(
                go.Scatter(
                    name=m,
                    x=x_vals,
                    y=perc,
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color=color_map.get(m, "#999999"), width=2),
                    fillcolor=hex_to_rgba(color_map.get(m, "#999999"), 0.1),
                    hovertemplate="年龄=%{x}<br>占比=%{y:.2f}%<extra></extra>",
                    showlegend=False,
                ),
                row=19,
                col=1,
            )
            series_legend.append((m, color_map.get(m, "#999999")))
        add_local_legend("yaxis11", series_legend)

        hdr_age_stats = ["车型", "均值", "中位数", "标准差"]
        rows_age_stats = []
        for m in ["CM0", "CM1", "CM2", "DM0", "DM1", "LS9"]:
            d0, d1 = phase_range(m)
            if pd.isna(d0) or pd.isna(d1) or d1 < d0:
                rows_age_stats.append([m, "-", "-", "-"])
                continue
            mask = (model_codes == m) & lock_times.between(d0, d1, inclusive="both")
            ages_m = ages.loc[mask]
            ages_m = ages_m[(ages_m >= 10) & (ages_m <= 80)]
            if ages_m.empty:
                rows_age_stats.append([m, "-", "-", "-"])
                continue
            mean_val = ages_m.mean()
            median_val = ages_m.median()
            std_val = ages_m.std()
            rows_age_stats.append([m, f"{mean_val:.2f}", f"{median_val:.2f}", f"{std_val:.2f}"])
        row_colors = [
            ("#EAF2FF" if m.startswith("CM") else ("#FFF2E6" if m.startswith("DM") else "#F2F2F2"))
            for m, _, _, _ in rows_age_stats
        ]
        fig.add_table(
            header=dict(values=hdr_age_stats, fill_color="#f6f6f6", align="center", font=dict(color="#333", size=12), height=36),
            cells=dict(values=list(zip(*rows_age_stats)), fill_color=[row_colors] * len(hdr_age_stats), align="center", height=32),
            row=20,
            col=1,
        )

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("reports") / "线索锁单转化率_日级.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="inline")
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
