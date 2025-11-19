#!/usr/bin/env python3
"""
绘制 2025 年锁单量「累计」与「累计环比(按日)」对比折线图。

- 数据源：/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet
- 计算口径：
  1) 每天含有 Lock_Time 的订单数（按日统计）
  2) 累计锁单量：按日累积求和
  3) 累计环比：累计锁单量相较前一日的环比变化（pct_change，单位百分比）

- 可选参数：
  --input 输入 parquet 路径（默认如上）
  --out 输出 HTML 折线图路径（默认 reports/lock_cumulative_ratio_2025.html）
"""

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_INPUT = Path(
    "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"
)
DEFAULT_OUT = Path(
    "/Users/zihao_/Documents/coding/dataset/reports/lock_cumulative_ratio_2025.html"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="绘制 2025 年锁单量累计与累计环比（按日）"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="输入 parquet 文件路径 (包含 Lock_Time 列)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT),
        help="输出 HTML 折线图路径",
    )
    return parser.parse_args()


def compute_daily_lock_cumulative_and_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """计算 2025 全年的每日锁单量、累计锁单量，并生成相对 2024 同日累计的同比(%)。

    同比口径：累计值 YoY% = (累计2025 / 累计2024_same_date - 1) * 100
    其中 2024 累计为同月同日的累计（忽略闰年 2/29 对 2025 的不存在影响）。
    当 2024 累计为 0 时，YoY% 记为 NaN。
    """
    if "Lock_Time" not in df.columns:
        raise KeyError("缺少 Lock_Time 列，无法统计锁单量")

    # 确保为 datetime
    df = df.copy()
    df["Lock_Time"] = pd.to_datetime(df["Lock_Time"], errors="coerce")
    if "Invoice_Upload_Time" not in df.columns:
        raise KeyError("缺少 Invoice_Upload_Time 列，无法统计交付")
    df["Invoice_Upload_Time"] = pd.to_datetime(df["Invoice_Upload_Time"], errors="coerce")

    # 过滤 2024/2025 年，且 Lock_Time 非空
    mask_valid = df["Lock_Time"].notna()
    df_valid = df.loc[mask_valid, ["Lock_Time"]].copy()
    df_valid["lock_date"] = df_valid["Lock_Time"].dt.date

    df_2025 = df_valid[df_valid["Lock_Time"].dt.year == 2025]
    df_2024 = df_valid[df_valid["Lock_Time"].dt.year == 2024]

    # 交付数据：订单同时具备 Lock_Time 与 Invoice_Upload_Time，以开票时间作为“交付日期”
    df_delivered = df[(df["Lock_Time"].notna()) & (df["Invoice_Upload_Time"].notna())].copy()
    df_delivered_2025 = df_delivered[df_delivered["Invoice_Upload_Time"].dt.year == 2025].copy()
    df_delivered_2025["deliver_date"] = df_delivered_2025["Invoice_Upload_Time"].dt.date
    daily_delivered_2025 = df_delivered_2025.groupby("deliver_date").size().sort_index()

    # 按日统计锁单数
    daily_2025 = df_2025.groupby("lock_date").size().sort_index()
    daily_2024 = df_2024.groupby("lock_date").size().sort_index()

    # 构建 2025 完整日期序列并填充缺失日为 0
    start_2025 = date(2025, 1, 1)
    end_2025 = date(2025, 12, 31)
    full_2025 = pd.date_range(start=start_2025, end=end_2025, freq="D").date
    daily_2025 = daily_2025.reindex(full_2025, fill_value=0)
    daily_delivered_2025 = daily_delivered_2025.reindex(full_2025, fill_value=0)

    # 为每个 2025 日期映射一个 2024 同月同日日期索引（2025 无 2/29，不会触发无效日期）
    baseline_2024_index = [date(2024, d.month, d.day) for d in full_2025]
    daily_2024_mapped = daily_2024.reindex(baseline_2024_index, fill_value=0)
    # 对齐索引到 2025 日期轴，保证后续拼装长度一致
    daily_2024_mapped.index = pd.Index(full_2025)

    # 累计锁单量（两年）
    cumulative_2025 = daily_2025.cumsum()
    cumulative_2024 = daily_2024_mapped.cumsum()
    cumulative_delivered_2025 = daily_delivered_2025.cumsum()

    # 累计同比（百分比）：(2025 / 2024 - 1) * 100，当 2024 为 0 时设为 NaN
    yoy_pct = (cumulative_2025 / cumulative_2024 - 1.0) * 100.0
    yoy_pct = yoy_pct.where(cumulative_2024 != 0, other=pd.NA)

    # today() 用于区分已发生与未来（预测）
    today = date.today()
    # 将 today 限制在 2025 范围内
    if today < start_2025:
        today = start_2025
    if today > end_2025:
        today = end_2025

    # 找到 today 在 2025 日期轴中的位置
    # full_2025 是 numpy 数组，转为列表以获取索引
    idx_today = list(full_2025).index(today)

    # 计算当前时点的同比
    cum_2025_today = cumulative_2025.iloc[idx_today]
    cum_2024_today = cumulative_2024.iloc[idx_today]
    if cum_2024_today and cum_2024_today > 0:
        yoy_today = (cum_2025_today / cum_2024_today) - 1.0
    else:
        yoy_today = 0.0

    # 使用当前同比值模拟未来每日锁单量（基于 2024 同日的每日数）
    future_dates = full_2025[idx_today + 1 :]
    daily_2024_future = daily_2024_mapped.loc[future_dates]
    daily_2025_future_pred = daily_2024_future * (1.0 + yoy_today)

    # 预测累计：从当前累计起点继续累加预测的未来每日值
    cumulative_2025_future_pred = cum_2025_today + daily_2025_future_pred.cumsum()

    # 2024 全年总锁单量
    total_2024 = int(daily_2024.sum()) if len(daily_2024) > 0 else 0
    forecast_total_2025 = int(round((1.0 + yoy_today) * total_2024))

    # 组装结果（包含预测字段）
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(full_2025)),
            "daily_lock_count": daily_2025.values,
            "cumulative_lock_count": cumulative_2025.values,
            "daily_delivery_count": daily_delivered_2025.values,
            "cumulative_delivery_count": cumulative_delivered_2025.values,
            "cumulative_2024_baseline": cumulative_2024.values,
            "cumulative_yoy_pct": yoy_pct.values,
        }
    )

    # 仅展示 today() 之前的实际曲线；today() 之后的实际部分置为 NaN
    mask_future = result["date"].dt.date > today
    result["cumulative_lock_count_actual"] = result["cumulative_lock_count"].astype(float)
    result.loc[mask_future, "cumulative_lock_count_actual"] = pd.NA
    result["cumulative_delivery_count_actual"] = result["cumulative_delivery_count"].astype(float)
    result.loc[mask_future, "cumulative_delivery_count_actual"] = pd.NA
    result["cumulative_yoy_pct_actual"] = result["cumulative_yoy_pct"].astype(float)
    result.loc[mask_future, "cumulative_yoy_pct_actual"] = pd.NA

    # 追加预测列（对未发生部分填充预测累计，对已发生部分填充 NaN）
    result["cumulative_2025_pred"] = pd.NA
    if len(future_dates) > 0:
        result.loc[result["date"].dt.date.isin(future_dates), "cumulative_2025_pred"] = (
            cumulative_2025_future_pred.values
        )

    # 同比预测：未来部分用 yoy_today 常数（百分比）
    result["cumulative_yoy_pred_pct"] = pd.NA
    if len(future_dates) > 0:
        result.loc[result["date"].dt.date.isin(future_dates), "cumulative_yoy_pred_pct"] = (
            (yoy_today * 100.0)
        )

    # 附加元信息
    result.attrs["today"] = today
    result.attrs["forecast_total_2025"] = forecast_total_2025
    result.attrs["yoy_today_pct"] = yoy_today * 100.0

    return result


def build_figure(df: pd.DataFrame) -> go.Figure:
    """构建双轴折线图：左轴累计锁单量，右轴累计环比(%)。"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 累计锁单量（左轴）实际，颜色 #27AD00（仅到 today）
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cumulative_lock_count_actual"],
            name="累计锁单量(实际)",
            mode="lines",
            line=dict(color="#27AD00", width=2),
            hovertemplate="日期:%{x|%Y-%m-%d}<br>累计:%{y}<extra></extra>",
        ),
        secondary_y=False,
    )

    # 累计交付数（左轴）实际，颜色橙色（仅到 today）
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cumulative_delivery_count_actual"],
            name="累计交付数(实际)",
            mode="lines",
            line=dict(color="#FF7F0E", width=2),
            hovertemplate="日期:%{x|%Y-%m-%d}<br>累计交付:%{y}<extra></extra>",
        ),
        secondary_y=False,
    )

    # 累计锁单量（左轴）预测，颜色同色虚线
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cumulative_2025_pred"],
            name="累计锁单量(预测)",
            mode="lines",
            line=dict(color="#27AD00", width=2, dash="dash"),
            hovertemplate="日期:%{x|%Y-%m-%d}<br>预测累计:%{y:.0f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # 累计同比（右轴，百分比）实际，颜色 #005783（仅到 today）
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cumulative_yoy_pct_actual"],
            name="累计同比(%)",
            mode="lines",
            line=dict(color="#005783", width=2),
            hovertemplate="日期:%{x|%Y-%m-%d}<br>同比:%{y:.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    # 累计同比（右轴，百分比）预测，虚线
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cumulative_yoy_pred_pct"],
            name="累计同比(预测%)",
            mode="lines",
            line=dict(color="#005783", width=2, dash="dash"),
            hovertemplate="日期:%{x|%Y-%m-%d}<br>预测同比:%{y:.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    # 标题与注释
    fig.update_layout(
        title="2025 年锁单量累计与累计同比（相对 2024 同日，含预测）",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=60, b=60),
    )

    fig.update_xaxes(
        title_text="日期",
        dtick="M1",
        showgrid=True,
        gridcolor="#eee",
    )

    fig.update_yaxes(title_text="累计锁单量", secondary_y=False, showgrid=True, gridcolor="#f5f5f5")
    fig.update_yaxes(title_text="累计同比(%)", secondary_y=True)

    # 标注今天竖线（使用 shape + annotation 避免 Timestamp 求均值错误）
    if "today" in df.attrs:
        today_dt = pd.to_datetime(df.attrs["today"])
        fig.add_shape(
            type="line",
            x0=today_dt,
            x1=today_dt,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="#999", dash="dot"),
        )
        fig.add_annotation(
            x=today_dt,
            y=1.02,
            xref="x",
            yref="paper",
            showarrow=False,
            text="today",
        )

    # 在年末处标注预测总锁单量
    if "forecast_total_2025" in df.attrs:
        year_end = pd.to_datetime(date(2025, 12, 31))
        fig.add_annotation(
            x=year_end,
            y=df["cumulative_lock_count"].max(),
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            text=f"预测总锁单量: {df.attrs['forecast_total_2025']}",
        )

    return fig


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 读取数据
    df = pd.read_parquet(input_path)

    # 计算指标
    result = compute_daily_lock_cumulative_and_ratio(df)

    # 构建图形
    fig = build_figure(result)

    # 确保输出目录存在
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")

    # 控制台输出简要信息
    total_locks = int(result["cumulative_lock_count"].iloc[-1])
    forecast_total = result.attrs.get("forecast_total_2025")
    yoy_today_pct = result.attrs.get("yoy_today_pct")
    print(f"📈 2025 累计锁单量(截至 today): {total_locks}")
    if forecast_total is not None:
        print(f"🔮 预测 2025 年总锁单量: {forecast_total} (基于当前同比 {yoy_today_pct:.2f}%)")
    print(f"✅ 已生成图表: {out_path}")


if __name__ == "__main__":
    main()