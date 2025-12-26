#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm

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
    sub = pd.DataFrame(
        {
            "date": df.loc[mask, "date"],
            "val": pd.to_numeric(series.loc[mask], errors="coerce"),
        }
    )
    sub = sub.dropna(subset=["date"]).copy()
    sub["val_ma"] = sub["val"].rolling(window=ma_window, min_periods=1).mean()
    sub["day_n"] = (sub["date"] - start).dt.days
    aligned = sub.set_index("day_n")["val_ma"].sort_index()
    full = pd.Index(range(0, 366), name="day_n")
    aligned = aligned.reindex(full)
    return aligned


def resolve(df: pd.DataFrame, logical: str) -> str:
    cand = {
        "order_time": ["Order_Create_Time", "订单创建时间", "order_create_time", "下单时间", "Lock_Time", "锁单时间"],
        "store_create": ["store_create_date", "门店开业时间", "Store_Create_Date"],
        "store_name": ["Store Name", "门店名称", "store_name"],
        "age": ["owner_age", "Owner Age", "Owner_Age", "age", "Age", "车主年龄"],
        "model_group": ["车型分组", "Model Group", "Vehicle Group", "Car Group", "车型"],
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


def load_active_store_series(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    orders_path = Path("/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet")
    try:
        df_orders = pd.read_parquet(orders_path)
    except Exception:
        print(f"Warning: Could not read {orders_path}")
        return pd.Series(dtype=float)

    if df_orders.empty:
        return pd.Series(dtype=float)

    col_time = None
    for c in ["Order_Create_Time", "Lock_Time"]:
        if c in df_orders.columns:
            col_time = c
            break
    if col_time is None:
        try:
            col_time = resolve(df_orders, "order_time")
        except KeyError:
            return pd.Series(dtype=float)

    try:
        col_store = resolve(df_orders, "store_name")
        col_create = resolve(df_orders, "store_create")
    except KeyError:
        return pd.Series(dtype=float)

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
    
    full_days = pd.date_range(start_date, end_date, freq="D")

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
        # Check if d is in roll index, if not (should not happen due to reindex), handle gracefully
        if d not in roll.index:
             active_counts.append(0)
             continue
             
        rs = roll.loc[d]
        # 开业判断：开业日期存在且不晚于当天
        open_mask = open_series.reindex(rs.index)
        open_ok = open_mask.notna() & (open_mask <= d)
        active = (rs > 0) & open_ok
        active_counts.append(int(active.sum()))

    return pd.Series(active_counts, index=full_days, name="在营门店数")


def load_conversion_duration_series(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    orders_path = Path("/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet")
    try:
        df = pd.read_parquet(orders_path)
    except Exception:
        print(f"Warning: Could not read {orders_path}")
        return pd.Series(dtype=float)

    if df.empty:
        return pd.Series(dtype=float)

    # Columns: Lock_Time, first_assign_time
    col_lock = "Lock_Time"
    col_assign = "first_assign_time"
    
    if col_lock not in df.columns or col_assign not in df.columns:
        return pd.Series(dtype=float)
        
    df[col_lock] = pd.to_datetime(df[col_lock], errors="coerce")
    df[col_assign] = pd.to_datetime(df[col_assign], errors="coerce")
    
    # Filter valid lock times
    valid = df.dropna(subset=[col_lock, col_assign]).copy()
    
    # Calculate duration in days
    valid["duration_days"] = (valid[col_lock] - valid[col_assign]).dt.total_seconds() / 86400.0
    
    # Filter reasonable duration (e.g. >= 0)
    # Some data might have assign time > lock time due to data issues, we filter them or keep as negative?
    # Usually duration should be >= 0. Let's filter >= 0.
    valid = valid[valid["duration_days"] >= 0]
    
    valid["day"] = valid[col_lock].dt.floor("D")
    
    # Group by lock day and calc mean
    daily_avg = valid.groupby("day")["duration_days"].mean()
    
    # Reindex to full range
    full_days = pd.date_range(start_date, end_date, freq="D")
    daily_avg = daily_avg.reindex(full_days) # Don't fill 0, let it be NaN if no data
    
    daily_avg.name = "线索-锁单转化时长"
    return daily_avg


def load_daily_mean_age_series(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    orders_path = Path("/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet")
    try:
        df = pd.read_parquet(orders_path)
    except Exception:
        print(f"Warning: Could not read {orders_path}")
        return pd.Series(dtype=float)

    if df.empty:
        return pd.Series(dtype=float)

    try:
        col_lock = resolve(df, "order_time")
        col_model = resolve(df, "model_group")
        col_age = resolve(df, "age")
    except KeyError:
        return pd.Series(dtype=float)

    df[col_lock] = pd.to_datetime(df[col_lock], errors="coerce")
    df_sub = df[[col_lock, col_model, col_age]].copy()
    df_sub = df_sub[df_sub[col_lock].notna()]
    df_sub["day"] = pd.to_datetime(df_sub[col_lock], errors="coerce").dt.floor("D")
    df_sub = df_sub[(df_sub["day"] >= start_date) & (df_sub["day"] <= end_date)]
    df_sub["model_group"] = df_sub[col_model].astype(str)
    df_sub = df_sub[df_sub["model_group"].isin(["CM0", "CM1", "CM2"])]

    df_sub["age_raw"] = pd.to_numeric(df_sub[col_age], errors="coerce")
    df_sub["age_clean"] = df_sub["age_raw"].where(
        (df_sub["age_raw"] >= 16) & (df_sub["age_raw"] <= 85)
    )
    df_sub = df_sub.dropna(subset=["age_clean"])
    if df_sub.empty:
        return pd.DataFrame()

    grouped = df_sub.groupby("day")["age_clean"]
    daily_mean = grouped.mean()
    daily_cnt = grouped.count()
    full_days = pd.date_range(start_date, end_date, freq="D")
    daily_mean = daily_mean.reindex(full_days)
    daily_cnt = daily_cnt.reindex(full_days).fillna(0)
    out = pd.DataFrame(
        {
            "车主平均年龄": daily_mean,
            "车主平均年龄样本数": daily_cnt,
        },
        index=full_days,
    )
    return out


def make_alignment_chart(df: pd.DataFrame) -> go.Figure:
    # Merge Active Store Count
    if not df.empty:
        d_min, d_max = df["date"].min(), df["date"].max()
        active_s = load_active_store_series(d_min, d_max)
        if not active_s.empty:
            # Join on date
            temp = active_s.to_frame(name="在营门店数")
            temp.index.name = "date"
            df = df.merge(temp, on="date", how="left")
            df["在营门店数"] = df["在营门店数"].fillna(0)
    
    # Merge Conversion Duration
    if not df.empty:
        d_min, d_max = df["date"].min(), df["date"].max()
        dur_s = load_conversion_duration_series(d_min, d_max)
        if not dur_s.empty:
            temp = dur_s.to_frame(name="线索-锁单转化时长")
            temp.index.name = "date"
            df = df.merge(temp, on="date", how="left")
            # Do not fillna(0) for duration, keep as NaN

    if not df.empty:
        d_min, d_max = df["date"].min(), df["date"].max()
        age_df = load_daily_mean_age_series(d_min, d_max)
        if not age_df.empty:
            temp = age_df.copy()
            temp.index.name = "date"
            df = df.merge(temp, on="date", how="left")

    w3_start = pd.to_datetime("2023-01-01")
    w3_end = pd.to_datetime("2023-12-31")
    w2_start = pd.to_datetime("2024-01-01")
    w2_end = pd.to_datetime("2024-12-31")
    w1_start = pd.to_datetime("2025-01-01")
    w1_end = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    w1_end_str = w1_end.strftime("%Y-%m-%d")

    m_lock = "锁单数"
    m_eff = "有效线索数"
    m_td_eff = "有效试驾数"
    m_dis = "下发线索数"
    m_store = "下发线索门店数"
    m_owner = "下发线索主理数"
    m_active = "在营门店数"
    m_dur = "线索-锁单转化时长"

    dis_num = pd.to_numeric(df[m_dis], errors="coerce")
    store_den = pd.to_numeric(df[m_store], errors="coerce").replace(0, pd.NA)
    owner_den = pd.to_numeric(df[m_owner], errors="coerce").replace(0, pd.NA)
    store_avg = dis_num / store_den
    owner_avg = dis_num / owner_den
    
    # Check if m_active exists
    active_col = df[m_active].astype(float) if m_active in df.columns else pd.Series([0]*len(df))
    # Check if m_dur exists
    dur_col = df[m_dur].astype(float) if m_dur in df.columns else pd.Series([pd.NA]*len(df))
    age_col = pd.to_numeric(df["车主平均年龄"], errors="coerce") if "车主平均年龄" in df.columns else pd.Series([pd.NA]*len(df))

    metrics = [
        ("锁单数", df[m_lock].astype(float), "数", ":.0f"),
        ("有效线索数", df[m_eff].astype(float), "数", ":.0f"),
        ("7日线索转化率", df["7日线索锁单转化率"].astype(float), "百分比", ":.2f%"),
        ("30日线索转化率", df["30日线索锁单转化率"].astype(float), "百分比", ":.2f%"),
        ("有效试驾数", df[m_td_eff].astype(float), "数", ":.0f"),
        ("店均下发线索数", store_avg, "数", ":.2f"),
        ("主理人均下发线索数", owner_avg, "数", ":.2f"),
        ("在营门店数", active_col, "数", ":.0f"),
        ("线索-锁单转化时长", dur_col, "天", ":.2f"),
        ("车主平均年龄", age_col, "岁", ":.2f"),
    ]

    titles = [
        "锁单数（按第N天对齐，MA7）",
        "有效线索数（按第N天对齐，MA7）",
        "7日线索转化率（按第N天对齐，MA7）",
        "30日线索转化率（按第N天对齐，MA7）",
        "有效试驾数（按第N天对齐，MA7）",
        "店均下发线索数（按第N天对齐，MA7）",
        "主理人均下发线索数（按第N天对齐，MA7）",
        "在营门店数（按第N天对齐，MA7）",
        "线索-锁单转化时长（按第N天对齐，MA7）",
        "车主平均年龄（按第N天对齐，MA7，车型分组=CM0,CM1,CM2）",
        "锁单数 vs 线索-锁单转化时长 (散点图)",
    ]
    
    # Interleave titles with None for table rows
    subplot_titles = []
    for t in titles:
        subplot_titles.append(t)
        subplot_titles.append("")

    rows_count = (len(metrics) + 1) * 2
    row_heights = []
    # Assign relative heights: Plot gets more space than Table
    for _ in range(len(metrics) + 1):
        row_heights.append(0.25)  # Plot
        row_heights.append(0.15)  # Table
    
    # Normalize heights
    total_h = sum(row_heights)
    row_heights = [h / total_h for h in row_heights]

    specs = [[{"type": "xy"}], [{"type": "table"}]] * (len(metrics) + 1)

    fig = make_subplots(
        rows=rows_count, 
        cols=1, 
        shared_xaxes=False, 
        subplot_titles=subplot_titles, 
        vertical_spacing=0.015,
        row_heights=row_heights,
        specs=specs
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
            bordercolor="#7B848F",
            borderwidth=1,
            font=dict(size=12, color="#7B848F"),
            xanchor="left",
            yanchor="top",
        )
    
    summary_metrics = {
        "有效线索数",
        "有效试驾数",
        "7日线索转化率",
        "30日线索转化率",
        "锁单数",
        "在营门店数",
        "主理人均下发线索数",
    }
    summary_rate_metrics = {"7日线索转化率", "30日线索转化率"}
    summary_rows = []

    # Dynamic comparison ranges based on w1_end
    # Range 4: Current Month MTD
    curr_m_start = w1_end.replace(day=1)
    curr_m_end = w1_end
    
    # Range 3: Previous Month Same Days
    prev_m_end = w1_end - pd.DateOffset(months=1)
    prev_m_start = prev_m_end.replace(day=1)
    
    # Range 2: Last Year Same Month MTD
    last_y_end = w1_end - pd.DateOffset(years=1)
    last_y_start = last_y_end.replace(day=1)
    
    # Range 1: Two Years Ago Same Month MTD
    two_y_end = w1_end - pd.DateOffset(years=2)
    two_y_start = two_y_end.replace(day=1)
    
    comp_ranges = [
        (f"{two_y_start.strftime('%Y-%m-%d')}～{two_y_end.strftime('%Y-%m-%d')}", two_y_start, two_y_end),
        (f"{last_y_start.strftime('%Y-%m-%d')}～{last_y_end.strftime('%Y-%m-%d')}", last_y_start, last_y_end),
        (f"{prev_m_start.strftime('%Y-%m-%d')}～{prev_m_end.strftime('%Y-%m-%d')}", prev_m_start, prev_m_end),
        (f"{curr_m_start.strftime('%Y-%m-%d')}～{curr_m_end.strftime('%Y-%m-%d')}", curr_m_start, curr_m_end),
    ]

    x_vals = list(range(0, 366))
    for i, (name, series, unit, fmt) in enumerate(metrics, start=1):
        plot_row = i * 2 - 1
        table_row = i * 2
        
        s_w1 = align_series_by_day(df, w1_start, w1_end, series, ma_window=7)
        s_w2 = align_series_by_day(df, w2_start, w2_end, series, ma_window=7)
        s_w3 = align_series_by_day(df, w3_start, w3_end, series, ma_window=7)

        fig.add_trace(
            go.Scatter(
                name=f"{name} 2025-01-01 至 {w1_end_str} (MA7)",
                x=x_vals,
                y=s_w1.values,
                mode="lines",
                line=dict(color="#27AD00", width=2),
                hovertemplate=f"第N天=%{{x}}<br>{name}(MA7)=%{{y{fmt}}}<extra></extra>",
                showlegend=False,
            ),
            row=plot_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name=f"{name} 2024-01-01 至 2024-12-31 (MA7)",
                x=x_vals,
                y=s_w2.values,
                mode="lines",
                line=dict(color="#005783", width=2),
                hovertemplate=f"第N天=%{{x}}<br>{name}(MA7)=%{{y{fmt}}}<extra></extra>",
                showlegend=False,
            ),
            row=plot_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name=f"{name} 2023-01-01 至 2023-12-31 (MA7)",
                x=x_vals,
                y=s_w3.values,
                mode="lines",
                line=dict(color="#A0A0A0", width=2),
                hovertemplate=f"第N天=%{{x}}<br>{name}(MA7)=%{{y{fmt}}}<extra></extra>",
                showlegend=False,
            ),
            row=plot_row,
            col=1,
        )
        fig.update_yaxes(title_text=name, row=plot_row, col=1)

        yaxis_name = "yaxis" + ("" if i == 1 else str(i))
        add_local_legend(yaxis_name, [
            (f"2025-01-01 至 {w1_end_str} (MA7)", "#27AD00"), 
            ("2024-01-01 至 2024-12-31 (MA7)", "#005783"),
            ("2023-01-01 至 2023-12-31 (MA7)", "#A0A0A0")
        ])
        
        # Calculate comparison means
        comp_vals = []
        raw_means = []
        for _, start, end in comp_ranges:
            mask = (df["date"] >= start) & (df["date"] <= end)
            mean_val = pd.to_numeric(series[mask], errors="coerce").mean()
            raw_means.append(mean_val)
            if pd.isna(mean_val):
                comp_vals.append("-")
            else:
                clean_fmt = fmt.replace(':', '')
                suffix = ""
                if clean_fmt.endswith('%'):
                    clean_fmt = clean_fmt.rstrip('%')
                    suffix = "%"
                try:
                    comp_vals.append(f"{mean_val:{clean_fmt}}{suffix}")
                except ValueError:
                     comp_vals.append(f"{mean_val}")
        
        if name in summary_metrics:
            row = {"指标": name}
            for label, start, end in comp_ranges:
                mask = (df["date"] >= start) & (df["date"] <= end)
                if name in {"锁单数", "有效线索数", "有效试驾数", "在营门店数"}:
                    window_vals = pd.to_numeric(series[mask], errors="coerce")
                    total = window_vals.sum()
                    row[label] = f"{total:.0f}" if pd.notna(total) else "-"
                elif name == "7日线索转化率":
                    num = pd.to_numeric(
                        df.loc[mask, "7 日内锁单线索数"], errors="coerce"
                    ).sum()
                    den = pd.to_numeric(
                        df.loc[mask, "有效线索数"], errors="coerce"
                    ).sum()
                    if pd.notna(num) and den and den != 0:
                        val = num / den * 100.0
                        row[label] = f"{val:.2f}%"
                    else:
                        row[label] = "-"
                elif name == "30日线索转化率":
                    num = pd.to_numeric(
                        df.loc[mask, "30 日锁单线索数"], errors="coerce"
                    ).sum()
                    den = pd.to_numeric(
                        df.loc[mask, "有效线索数"], errors="coerce"
                    ).sum()
                    if pd.notna(num) and den and den != 0:
                        val = num / den * 100.0
                        row[label] = f"{val:.2f}%"
                    else:
                        row[label] = "-"
                elif name == "主理人均下发线索数":
                    num = pd.to_numeric(df.loc[mask, m_dis], errors="coerce").sum()
                    den = pd.to_numeric(df.loc[mask, m_owner], errors="coerce").sum()
                    if pd.notna(num) and den and den != 0:
                        val = num / den
                        row[label] = f"{val:.2f}"
                    else:
                        row[label] = "-"
                else:
                    row[label] = "-"
            summary_rows.append(row)

        # Calculate YoY
        # raw_means indices: 0 -> 2023-12, 1 -> 2024-12, 2 -> 2025-11, 3 -> 2025-12
        val_23_12 = raw_means[0]
        val_24_12 = raw_means[1]
        val_25_11 = raw_means[2]
        val_25_12 = raw_means[3]
        
        # YoY 24 vs 23
        yoy1_val = "-"
        if pd.notna(val_24_12) and pd.notna(val_23_12) and val_23_12 != 0:
            diff = (val_24_12 - val_23_12) / val_23_12
            sign = "+" if diff > 0 else ""
            yoy1_val = f"{sign}{diff:.2%}"

        # MoM 25-12 vs 25-11
        mom_val = "-"
        if pd.notna(val_25_12) and pd.notna(val_25_11) and val_25_11 != 0:
            diff = (val_25_12 - val_25_11) / val_25_11
            sign = "+" if diff > 0 else ""
            mom_val = f"{sign}{diff:.2%}"
            
        # YoY 25 vs 24
        yoy2_val = "-"
        if pd.notna(val_25_12) and pd.notna(val_24_12) and val_24_12 != 0:
            diff = (val_25_12 - val_24_12) / val_24_12
            sign = "+" if diff > 0 else ""
            yoy2_val = f"{sign}{diff:.2%}"
            
        comp_vals.append(yoy1_val)
        comp_vals.append(mom_val)
        comp_vals.append(yoy2_val)

        # Add Table
        header_vals = ["指标"] + [r[0] for r in comp_ranges] + ["同比 (24 vs 23)", "环比 (vs 11月)", "同比 (25 vs 24)"]
        rows = []
        rows.append(["日均值 (原始数据)"] + comp_vals)

        if name == "车主平均年龄" and "车主平均年龄样本数" in df.columns:
            count_vals = []
            for _, start, end in comp_ranges:
                mask = (df["date"] >= start) & (df["date"] <= end)
                cnt = pd.to_numeric(df.loc[mask, "车主平均年龄样本数"], errors="coerce").sum()
                if pd.notna(cnt) and cnt != 0:
                    count_vals.append(f"{int(cnt)}")
                else:
                    count_vals.append("-")
            count_row = ["统计所用的锁单数"] + count_vals + ["-", "-", "-"]
            rows.append(count_row)

        cols = list(zip(*rows))
        cell_values = [list(col) for col in cols]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=header_vals,
                    font=dict(size=11, color="white"),
                    align="center",
                    fill_color="#555555",
                    height=24
                ),
                cells=dict(
                    values=cell_values,
                    font=dict(size=11),
                    align="center",
                    fill_color="#F5F5F5",
                    height=24
                )
            ),
            row=table_row,
            col=1
        )

    # Scatter & Correlation Module
    idx_extra = len(metrics) + 1
    row_scat = idx_extra * 2 - 1
    row_tbl = idx_extra * 2
    
    if m_lock in df.columns and "线索-锁单转化时长" in df.columns:
        mask_25 = (df["date"] >= w1_start) & (df["date"] <= w1_end)
        mask_24 = (df["date"] >= w2_start) & (df["date"] <= w2_end)
        
        # Extract data and drop NaNs
        df_25 = df.loc[mask_25, [m_lock, "线索-锁单转化时长"]].dropna().astype(float)
        df_24 = df.loc[mask_24, [m_lock, "线索-锁单转化时长"]].dropna().astype(float)
        
        # Apply filtering for Scatter Plot and LOWESS: Duration <= 100 days, Lock Count <= 1000
        df_25_filtered = df_25[(df_25["线索-锁单转化时长"] <= 100) & (df_25[m_lock] <= 1000)].sort_values(by=m_lock)
        df_24_filtered = df_24[(df_24["线索-锁单转化时长"] <= 100) & (df_24[m_lock] <= 1000)].sort_values(by=m_lock)
        
        # Scatter Traces
        fig.add_trace(
            go.Scatter(
                x=df_25_filtered[m_lock],
                y=df_25_filtered["线索-锁单转化时长"],
                mode="markers",
                name=f"2025 (2025-01-01~{w1_end_str})",
                marker=dict(color="#27AD00", size=6, opacity=0.6),
                hovertemplate="锁单数: %{x}<br>转化时长: %{y:.2f}天<extra>2025</extra>"
            ),
            row=row_scat, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df_24_filtered[m_lock],
                y=df_24_filtered["线索-锁单转化时长"],
                mode="markers",
                name="2024 (2024-01-01~2024-12-31)",
                marker=dict(color="#005783", size=6, opacity=0.6),
                hovertemplate="锁单数: %{x}<br>转化时长: %{y:.2f}天<extra>2024</extra>"
            ),
            row=row_scat, col=1
        )
        
        # LOWESS Curve (Overall)
        # Combine filtered data
        df_all = pd.concat([df_25_filtered, df_24_filtered], axis=0).sort_values(by=m_lock)
        if not df_all.empty:
            lowess = sm.nonparametric.lowess
            # lowess returns (x, y) sorted by x
            z = lowess(df_all["线索-锁单转化时长"], df_all[m_lock], frac=0.3)
            
            fig.add_trace(
                go.Scatter(
                    x=z[:, 0],
                    y=z[:, 1],
                    mode="lines",
                    name="整体趋势 (LOWESS)",
                    line=dict(color="red", width=3),
                    hovertemplate="锁单数: %{x:.0f}<br>趋势预测: %{y:.2f}天<extra>LOWESS</extra>"
                ),
                row=row_scat, col=1
            )
        
        fig.update_xaxes(title_text="锁单数", row=row_scat, col=1)
        fig.update_yaxes(title_text="转化时长(天)", row=row_scat, col=1)

        yaxis_name_scat = "yaxis" + str(idx_extra)
        add_local_legend(yaxis_name_scat, [
            (f"2025 (2025-01-01~{w1_end_str})", "#27AD00"),
            ("2024 (2024-01-01~2024-12-31)", "#005783"),
            ("整体趋势 (LOWESS)", "red")
        ])
        
        # Correlation Calculation (Already filtered above)
        p_25 = df_25_filtered[m_lock].corr(df_25_filtered["线索-锁单转化时长"], method="pearson") if not df_25_filtered.empty else pd.NA
        s_25 = df_25_filtered[m_lock].corr(df_25_filtered["线索-锁单转化时长"], method="spearman") if not df_25_filtered.empty else pd.NA
        p_24 = df_24_filtered[m_lock].corr(df_24_filtered["线索-锁单转化时长"], method="pearson") if not df_24_filtered.empty else pd.NA
        s_24 = df_24_filtered[m_lock].corr(df_24_filtered["线索-锁单转化时长"], method="spearman") if not df_24_filtered.empty else pd.NA
        
        def fmt_corr(v):
            return f"{v:.4f}" if pd.notna(v) else "-"

        header = ["窗口", "Pearson系数 (剔除>1000锁单 & >100天时长)", "Spearman系数 (剔除>1000锁单 & >100天时长)"]
        cells = [
            [f"2025 (2025-01-01 ~ {w1_end_str})", "2024 (2024-01-01 ~ 2024-12-31)"],
            [fmt_corr(p_25), fmt_corr(p_24)],
            [fmt_corr(s_25), fmt_corr(s_24)]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=header, font=dict(size=11, color="white"), align="center", fill_color="#555555", height=24),
                cells=dict(values=cells, font=dict(size=11), align="center", fill_color="#F5F5F5", height=24)
            ),
            row=row_tbl, col=1
        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = Path("processed/analysis_results/core_metrics_alignment_summary_2023_2025.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig.update_layout(
        title=f"两个窗口按第N天对齐的指标对比 (观察时间: {w1_end_str}, 当年第 {w1_end.dayofyear} 天)",
        legend_title="窗口",
        height=5600,
        margin=dict(l=40, r=40, t=60, b=80),
        showlegend=False,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    fig.update_xaxes(
        title_text="第 N 天 (0-365)",
        showticklabels=True,
        gridcolor="#ebedf0",
        zerolinecolor="#ebedf0",
        linecolor="#7B848F",
        tickfont=dict(color="#7B848F"),
    )
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
