import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


PARQUET_FILE = Path("/Users/zihao_/Documents/coding/dataset/formatted/order_data.parquet")
BUSINESS_DEF_FILE = Path(
    "/Users/zihao_/Documents/github/26W06_Tool_calls/schema/business_definition.json"
)
SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT = SCRIPT_DIR / "reports" / "analyze_order.html"


def load_business_definition(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_sql_condition(df: pd.DataFrame, condition_str: str) -> pd.Series:
    def not_like_replacer(match):
        val = match.group(1)
        return f"~df['product_name'].str.contains('{val}', na=False, regex=False)"

    condition_str = re.sub(
        r"product_name\s+NOT\s+LIKE\s+'%([^%]+)%+'",
        not_like_replacer,
        condition_str,
    )

    def like_replacer(match):
        val = match.group(1)
        return f"df['product_name'].str.contains('{val}', na=False, regex=False)"

    condition_str = re.sub(
        r"product_name\s+LIKE\s+'%([^%]+)%+'",
        like_replacer,
        condition_str,
    )

    condition_str = condition_str.replace(" AND ", " & ").replace(" OR ", " | ")

    try:
        return eval(condition_str)
    except Exception as e:
        print(f"⚠️ 解析条件失败: {condition_str}, Error: {e}")
        return pd.Series([False] * len(df), index=df.index)


def load_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    print(f"📖 Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"✅ Loaded {len(df)} rows.")
    return df


def apply_series_group_logic(df: pd.DataFrame, business_def: dict) -> pd.DataFrame:
    logic: Dict[str, str] = business_def.get("series_group_logic", {})
    if "product_name" not in df.columns:
        df["series_group_logic"] = pd.NA
        return df

    group_col = pd.Series(pd.NA, index=df.index, dtype="string")
    default_group = "其他"
    for group, cond in logic.items():
        if str(cond).strip().upper() == "ELSE":
            default_group = group
            continue
        mask = parse_sql_condition(df, str(cond))
        if not isinstance(mask, pd.Series):
            continue
        mask = mask.fillna(False)
        assignable = group_col.isna() & mask
        if assignable.any():
            group_col = group_col.where(~assignable, group)

    df["series_group_logic"] = group_col.fillna(default_group).astype("string")
    return df


def build_hourly_intention_counts(
    df: pd.DataFrame, business_def: dict, target_groups: List[str]
) -> pd.DataFrame:
    if "intention_payment_time" not in df.columns:
        raise KeyError("数据缺少列: intention_payment_time")
    if "order_number" not in df.columns:
        raise KeyError("数据缺少列: order_number")
    if not pd.api.types.is_datetime64_any_dtype(df["intention_payment_time"]):
        df["intention_payment_time"] = pd.to_datetime(df["intention_payment_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    rows: List[Dict[str, object]] = []

    for g in target_groups:
        start_str = (time_periods.get(g, {}) or {}).get("start")
        if not start_str:
            continue
        start_day = pd.Timestamp(start_str)
        end_day = start_day + pd.Timedelta(days=1)

        m_group = df["series_group_logic"].eq(g)
        m_time = df["intention_payment_time"].notna()
        m_day = (df["intention_payment_time"] >= start_day) & (df["intention_payment_time"] < end_day)
        df_day = df.loc[m_group & m_time & m_day, ["order_number", "intention_payment_time"]].copy()
        if df_day.empty:
            for hour in range(24):
                rows.append(
                    {
                        "series_group_logic": g,
                        "start_date": start_day.date().isoformat(),
                        "hour": hour,
                        "intention_orders": 0,
                    }
                )
            continue

        df_day["hour"] = df_day["intention_payment_time"].dt.hour.astype("int64")
        hourly = df_day.groupby("hour")["order_number"].nunique()
        for hour in range(24):
            rows.append(
                {
                    "series_group_logic": g,
                    "start_date": start_day.date().isoformat(),
                    "hour": hour,
                    "intention_orders": int(hourly.get(hour, 0)),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(["__group_order", "hour"]).drop(columns="__group_order").reset_index(drop=True)
    return out


def build_presale_summary(
    df: pd.DataFrame, business_def: dict, target_groups: List[str]
) -> pd.DataFrame:
    if "intention_payment_time" not in df.columns:
        raise KeyError("数据缺少列: intention_payment_time")
    if "order_number" not in df.columns:
        raise KeyError("数据缺少列: order_number")
    if "series_group_logic" not in df.columns:
        raise KeyError("数据缺少列: series_group_logic")
    if not pd.api.types.is_datetime64_any_dtype(df["intention_payment_time"]):
        df["intention_payment_time"] = pd.to_datetime(df["intention_payment_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    rows: List[Dict[str, object]] = []

    base = df.loc[
        df["intention_payment_time"].notna(), ["order_number", "intention_payment_time", "series_group_logic"]
    ].copy()

    for g in target_groups:
        tp = time_periods.get(g, {}) or {}
        start_str = tp.get("start")
        end_str = tp.get("end")
        if not start_str:
            continue

        start_day = pd.Timestamp(start_str)
        end_day = pd.Timestamp(end_str) if end_str else start_day

        base_g = base.loc[base["series_group_logic"].eq(g)].copy()

        start_end_excl = start_day + pd.Timedelta(days=1)
        start_day_slice = base_g.loc[
            (base_g["intention_payment_time"] >= start_day)
            & (base_g["intention_payment_time"] < start_end_excl),
            ["order_number", "intention_payment_time"],
        ].copy()
        if start_day_slice.empty:
            hourly_full = pd.Series([0] * 24, index=pd.RangeIndex(0, 24), dtype="int64")
        else:
            start_day_slice["hour"] = start_day_slice["intention_payment_time"].dt.hour.astype("int64")
            hourly = start_day_slice.groupby("hour")["order_number"].nunique()
            hourly_full = hourly.reindex(range(24), fill_value=0).astype("int64")

        peak_hour = int(hourly_full.idxmax())
        peak_count = int(hourly_full.iloc[peak_hour])
        next_hour_count = int(hourly_full.iloc[peak_hour + 1]) if peak_hour < 23 else 0
        start_day_total = int(hourly_full.sum())

        end_limit_excl = end_day + pd.Timedelta(days=1)
        first_week_end_excl = min(start_day + pd.Timedelta(days=7), end_limit_excl)
        week_slice = base_g.loc[
            (base_g["intention_payment_time"] >= start_day)
            & (base_g["intention_payment_time"] < first_week_end_excl),
            ["order_number", "intention_payment_time"],
        ].copy()
        if week_slice.empty:
            first_week_total = 0
        else:
            week_slice["date"] = week_slice["intention_payment_time"].dt.floor("D")
            daily = week_slice.groupby("date")["order_number"].nunique()
            first_week_total = int(daily.sum())

        delta_to_sat = (5 - start_day.weekday()) % 7
        if delta_to_sat == 0:
            delta_to_sat = 7
        weekend_start = start_day + pd.Timedelta(days=int(delta_to_sat))
        weekend_days = [weekend_start, weekend_start + pd.Timedelta(days=1)]
        weekend_total = 0
        for d in weekend_days:
            if d > end_day:
                continue
            day_end_excl = d + pd.Timedelta(days=1)
            day_slice = base_g.loc[
                (base_g["intention_payment_time"] >= d) & (base_g["intention_payment_time"] < day_end_excl),
                "order_number",
            ]
            weekend_total += int(day_slice.nunique())

        rows.append(
            {
                "series_group_logic": g,
                "预售日期": start_day.date().isoformat(),
                "峰值小时小订数": peak_count,
                "峰值后第二小时小订数": next_hour_count,
                "预售当日累计小订数": start_day_total,
                "第一个周末小订数": weekend_total,
                "第一周累计小订数": first_week_total,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(["__group_order"]).drop(columns="__group_order").reset_index(drop=True)
    return out


def build_hourly_lock_counts(df: pd.DataFrame, business_def: dict, target_groups: List[str]) -> pd.DataFrame:
    if "lock_time" not in df.columns:
        raise KeyError("数据缺少列: lock_time")
    if "order_number" not in df.columns:
        raise KeyError("数据缺少列: order_number")
    if not pd.api.types.is_datetime64_any_dtype(df["lock_time"]):
        df["lock_time"] = pd.to_datetime(df["lock_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    rows: List[Dict[str, object]] = []

    for g in target_groups:
        end_str = (time_periods.get(g, {}) or {}).get("end")
        if not end_str:
            continue
        end_day = pd.Timestamp(end_str)
        if g == "CM0":
            end_day = end_day + pd.Timedelta(days=1)
        end_day_excl = end_day + pd.Timedelta(days=1)

        m_group = df["series_group_logic"].eq(g)
        m_time = df["lock_time"].notna()
        m_day = (df["lock_time"] >= end_day) & (df["lock_time"] < end_day_excl)
        df_day = df.loc[m_group & m_time & m_day, ["order_number", "lock_time"]].copy()
        if df_day.empty:
            for hour in range(24):
                rows.append(
                    {
                        "series_group_logic": g,
                        "end_date": end_day.date().isoformat(),
                        "hour": hour,
                        "lock_orders": 0,
                    }
                )
            continue

        df_day["hour"] = df_day["lock_time"].dt.hour.astype("int64")
        hourly = df_day.groupby("hour")["order_number"].nunique()
        for hour in range(24):
            rows.append(
                {
                    "series_group_logic": g,
                    "end_date": end_day.date().isoformat(),
                    "hour": hour,
                    "lock_orders": int(hourly.get(hour, 0)),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(["__group_order", "hour"]).drop(columns="__group_order").reset_index(drop=True)
    return out


def build_listing_summary(df: pd.DataFrame, business_def: dict, target_groups: List[str]) -> pd.DataFrame:
    if "lock_time" not in df.columns:
        raise KeyError("数据缺少列: lock_time")
    if "order_number" not in df.columns:
        raise KeyError("数据缺少列: order_number")
    if "series_group_logic" not in df.columns:
        raise KeyError("数据缺少列: series_group_logic")
    if not pd.api.types.is_datetime64_any_dtype(df["lock_time"]):
        df["lock_time"] = pd.to_datetime(df["lock_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    rows: List[Dict[str, object]] = []

    base = df.loc[df["lock_time"].notna(), ["order_number", "lock_time", "series_group_logic"]].copy()

    for g in target_groups:
        tp = time_periods.get(g, {}) or {}
        end_str = tp.get("end")
        finish_str = tp.get("finish")
        if not end_str:
            continue

        end_day = pd.Timestamp(end_str)
        if g == "CM0":
            end_day = end_day + pd.Timedelta(days=1)
        finish_day = pd.Timestamp(finish_str) if finish_str else end_day

        base_g = base.loc[base["series_group_logic"].eq(g)].copy()

        end_excl = end_day + pd.Timedelta(days=1)
        end_day_slice = base_g.loc[
            (base_g["lock_time"] >= end_day) & (base_g["lock_time"] < end_excl),
            ["order_number", "lock_time"],
        ].copy()
        if end_day_slice.empty:
            hourly_full = pd.Series([0] * 24, index=pd.RangeIndex(0, 24), dtype="int64")
        else:
            end_day_slice["hour"] = end_day_slice["lock_time"].dt.hour.astype("int64")
            hourly = end_day_slice.groupby("hour")["order_number"].nunique()
            hourly_full = hourly.reindex(range(24), fill_value=0).astype("int64")

        peak_hour = int(hourly_full.idxmax())
        peak_count = int(hourly_full.iloc[peak_hour])
        next_hour_count = int(hourly_full.iloc[peak_hour + 1]) if peak_hour < 23 else 0
        end_day_total = int(hourly_full.sum())

        finish_limit_excl = finish_day + pd.Timedelta(days=1)
        after_30d_end_excl = min(end_day + pd.Timedelta(days=31), finish_limit_excl)
        window_slice = base_g.loc[
            (base_g["lock_time"] >= end_day) & (base_g["lock_time"] < after_30d_end_excl),
            ["order_number", "lock_time"],
        ].copy()
        if window_slice.empty:
            after_30d_total = 0
        else:
            window_slice["date"] = window_slice["lock_time"].dt.floor("D")
            daily = window_slice.groupby("date")["order_number"].nunique()
            after_30d_total = int(daily.sum())

        delta_to_sat = (5 - end_day.weekday()) % 7
        if delta_to_sat == 0:
            delta_to_sat = 7
        weekend_start = end_day + pd.Timedelta(days=int(delta_to_sat))
        weekend_days = [weekend_start, weekend_start + pd.Timedelta(days=1)]
        weekend_total = 0
        for d in weekend_days:
            if d > finish_day:
                continue
            day_excl = d + pd.Timedelta(days=1)
            day_slice = base_g.loc[(base_g["lock_time"] >= d) & (base_g["lock_time"] < day_excl), "order_number"]
            weekend_total += int(day_slice.nunique())

        rows.append(
            {
                "series_group_logic": g,
                "上市日期": end_day.date().isoformat(),
                "峰值小时锁单数": peak_count,
                "峰值后第二小时锁单数": next_hour_count,
                "上市当日累计锁单数": end_day_total,
                "第一个周末锁单数": weekend_total,
                "上市后30日累计锁单数": after_30d_total,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(["__group_order"]).drop(columns="__group_order").reset_index(drop=True)
    return out


def _render_hourly_bar_figure(
    hourly_df: pd.DataFrame,
    target_groups: List[str],
    date_col: str,
    value_col: str,
    fig_title: str,
    y_title: str,
    subplot_date_label: str,
) -> go.Figure:
    summary = (
        hourly_df.groupby(["series_group_logic", date_col], as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "day_total"})
    )

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            f"{g} ({subplot_date_label}: {summary.loc[summary['series_group_logic'].eq(g), date_col].iloc[0] if (summary['series_group_logic'].eq(g).any()) else ''})"
            for g in target_groups
        ],
    )

    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    for g, (r, c) in zip(target_groups, positions):
        dfg = hourly_df[hourly_df["series_group_logic"].eq(g)]
        x = dfg["hour"].tolist() if not dfg.empty else list(range(24))
        y = dfg[value_col].tolist() if not dfg.empty else [0] * 24
        fig.add_trace(go.Bar(x=x, y=y, name=g, showlegend=False), row=r, col=c)
        if x and y:
            peak_i = int(np.argmax(y))
            peak_x = x[peak_i]
            peak_y = int(y[peak_i])
            fig.add_annotation(
                x=peak_x,
                y=peak_y,
                text=str(peak_y),
                showarrow=False,
                yshift=10,
                bgcolor="rgba(255,255,255,0.7)",
                row=r,
                col=c,
            )
        fig.update_xaxes(title_text="Hour", tickmode="linear", dtick=1, row=r, col=c)
        fig.update_yaxes(title_text=y_title, row=r, col=c)

    fig.update_layout(height=720, title=fig_title, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def render_report(
    presale_hourly_df: pd.DataFrame,
    presale_summary_df: pd.DataFrame,
    listing_hourly_df: pd.DataFrame,
    listing_summary_df: pd.DataFrame,
    target_groups: List[str],
) -> str:
    css = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; color: #333; }
        h1 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; border-left: 5px solid #3498db; padding-left: 10px; }
        h3 { color: #2980b9; margin-top: 25px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        th, td { text-align: left; padding: 10px; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: 600; color: #555; }
        tr:hover { background-color: #f5f5f5; }
        .timestamp { color: #888; font-size: 0.9em; margin-bottom: 20px; }
        .summary-box { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    </style>
    """

    html_content: List[str] = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>订单分析报告</title>",
        css,
        "</head>",
        "<body>",
        "<h1>订单分析报告 (Order Data)</h1>",
        f"<div class='timestamp'>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>",
        "<h2>1. 预售期每小时小订数（按 series_group_logic）</h2>",
        "<div class='summary-box'>",
        "<p>口径：每个 series_group_logic 使用业务定义中的预售期 start 日期；统计该日期内每小时小订数（intention_payment_time 非空的 order_number 去重计数）。</p>",
        "</div>",
    ]

    if presale_hourly_df.empty:
        html_content.append("<p>⚠️ 未生成任何统计结果（可能缺少 time_periods 或数据列）。</p>")
        html_content.append("</body></html>")
        return "\n".join(html_content)

    html_content.append("<h3>1.1 汇总</h3>")
    if presale_summary_df is None or presale_summary_df.empty:
        html_content.append("<p>⚠️ 汇总表为空（可能缺少 time_periods 的 start/end 或分组无数据）。</p>")
    else:
        html_content.append(
            presale_summary_df.to_html(
                index=False,
                classes="table",
                escape=False,
                float_format=lambda x: "{:,.0f}".format(x) if isinstance(x, (int, float)) else x,
            )
        )
        html_content.append(
            "<div class='summary-box'>"
            "<p><b>备注</b></p>"
            "<ul>"
            "<li>预售日期：业务定义 time_periods 中的 start_date</li>"
            "<li>峰值小时小订数：取预售当日每小时小订数的峰值小时</li>"
            "<li>峰值后第二小时小订数：取峰值小时后的第二个小时（peak_hour + 1）</li>"
            "<li>第一个周末小订数：取 startday 后第一个双休日（周六+周日）两日求和</li>"
            "<li>第一周累计小订数：startday ~ startday+6 的累计求和，且不超过 end day</li>"
            "</ul>"
            "</div>"
        )

    html_content.append("<h3>1.2 可视化</h3>")
    fig1 = _render_hourly_bar_figure(
        presale_hourly_df,
        target_groups,
        date_col="start_date",
        value_col="intention_orders",
        fig_title="预售期开始日每小时小订数（series_group_logic）",
        y_title="Intention Orders",
        subplot_date_label="预售期开始日",
    )
    html_content.append(pio.to_html(fig1, full_html=False, include_plotlyjs="cdn"))

    html_content.append("<h2>2. 上市期每小时锁单数（按 series_group_logic）</h2>")
    html_content.append("<div class='summary-box'>")
    html_content.append(
        "<p>口径：先用业务定义 series_group_logic 根据 product_name 对订单归类；然后对每个 series_group_logic 使用业务定义 time_periods 中的 end 日期（上市日期），统计该日期内每小时锁单数（lock_time 非空的 order_number 去重计数）。</p>"
    )
    html_content.append("</div>")

    html_content.append("<h3>2.1 汇总</h3>")
    if listing_summary_df is None or listing_summary_df.empty:
        html_content.append("<p>⚠️ 汇总表为空（可能缺少 time_periods 的 end/finish 或分组无数据）。</p>")
    else:
        html_content.append(
            listing_summary_df.to_html(
                index=False,
                classes="table",
                escape=False,
                float_format=lambda x: "{:,.0f}".format(x) if isinstance(x, (int, float)) else x,
            )
        )
        html_content.append(
            "<div class='summary-box'>"
            "<p><b>备注</b></p>"
            "<ul>"
            "<li>上市日期：业务定义 time_periods 中的 end_date；其中 CM0 特殊处理：上市日期取 end_date + 1（事故）</li>"
            "<li>峰值小时锁单数：取上市当日每小时锁单数的峰值小时</li>"
            "<li>峰值后第二小时锁单数：取峰值小时后的第二个小时（peak_hour + 1）</li>"
            "<li>第一个周末锁单数：取 endday 后第一个双休日（周六+周日）两日求和（不超过 finish day）</li>"
            "<li>上市后30日累计锁单数：endday ~ endday+30 的累计求和，且不超过 finish day</li>"
            "</ul>"
            "</div>"
        )

    html_content.append("<h3>2.2 可视化</h3>")
    fig2 = _render_hourly_bar_figure(
        listing_hourly_df,
        target_groups,
        date_col="end_date",
        value_col="lock_orders",
        fig_title="上市日期每小时锁单数（series_group_logic）",
        y_title="Lock Orders",
        subplot_date_label="上市日期",
    )
    html_content.append(pio.to_html(fig2, full_html=False, include_plotlyjs="cdn"))
    html_content.append("</body></html>")
    return "\n".join(html_content)



def main(output_path: Path = DEFAULT_OUTPUT) -> int:
    business_def = load_business_definition(BUSINESS_DEF_FILE)
    df = load_data(PARQUET_FILE)
    df = apply_series_group_logic(df, business_def)

    target_groups = ["CM0","DM0","CM1","DM1","CM2","LS9"]
    presale_hourly_df = build_hourly_intention_counts(df, business_def, target_groups)
    presale_summary_df = build_presale_summary(df, business_def, target_groups)
    listing_hourly_df = build_hourly_lock_counts(df, business_def, target_groups)
    listing_summary_df = build_listing_summary(df, business_def, target_groups)

    html = render_report(presale_hourly_df, presale_summary_df, listing_hourly_df, listing_summary_df, target_groups)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"✅ 已生成报告: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
