#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
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
        rows=12,
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
        ),
        vertical_spacing=0.045,
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
        ],
        row_heights=[0.17, 0.17, 0.16, 0.17, 0.17, 0.16, 0.15, 0.16, 0.15, 0.16, 0.15, 0.16],
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
        height=3400,
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

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("reports") / "线索锁单转化率_日级.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="inline")
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
