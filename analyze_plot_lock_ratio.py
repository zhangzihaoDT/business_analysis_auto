#!/usr/bin/env python3
"""
绘制锁单累计同比分析图 (2025 vs 2024 和 2026 vs 2025)：
1. 仅展示累计同比曲线 (Cumulative YoY %)
2. X 轴以 2025 年日期为基准 (0~365天对齐)
3. 包含两条曲线：
   - 2025 累计同比 (相对于 2024)
   - 2026 累计同比 (相对于 2025)
4. Tooltip 显示：
   - 日期 (真实日期)
   - 当日锁单数 (Daily Count)
   - 累计锁单数 (Cumulative Count)
   - 累计同比 (Cumulative YoY)
5. 末端显示数值 Text 标记

样式遵循 skill/visualization-style 规范。
"""

import argparse
from datetime import date, datetime
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Visualization Style Constants ---
COLOR_MAIN = "#3498DB"      # Blue (用于 2025 - 基准/完整年)
COLOR_CONTRAST = "#E67E22"  # Orange (用于 2026 - 当前/观察年)
COLOR_DARK = "#373f4a"
COLOR_GRID = "#ebedf0"
COLOR_TEXT = "#7B848F"
COLOR_BG = "#FFFFFF"

DEFAULT_INPUT = Path(
    "/Users/zihao_/Documents/coding/dataset/formatted/order_full_data.parquet"
)
DEFAULT_OUT = Path(
    "/Users/zihao_/Documents/coding/dataset/reports/lock_ratio_analysis.html"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制锁单累计同比分析图")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="输入 parquet 文件路径")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="输出 HTML 文件路径")
    return parser.parse_args()

def get_daily_counts(df: pd.DataFrame, year: int) -> pd.Series:
    """获取指定年份的日锁单数 Series (index=date, value=count)。"""
    # 筛选年份
    df_year = df[df["lock_time"].dt.year == year].copy()
    if df_year.empty:
        return pd.Series(dtype=int)
    
    # 统计每日锁单数 (order_number 去重)
    if "order_number" in df_year.columns:
        daily = df_year.groupby(df_year["lock_time"].dt.date)["order_number"].nunique()
    else:
        daily = df_year.groupby(df_year["lock_time"].dt.date).size()
    
    return daily.sort_index()

def align_to_2025_axis(daily_series: pd.Series, target_year: int) -> pd.DataFrame:
    """
    将指定年份的日数据对齐到 2025 年的日期轴 (MM-DD 对齐)。
    如果是闰年 (2024)，去掉 02-29。
    返回 DataFrame，index 为 2025 日期，包含 'raw_date', 'count'。
    """
    # 2025 全年日期序列 (365天)
    start_2025 = date(2025, 1, 1)
    end_2025 = date(2025, 12, 31)
    idx_2025 = pd.date_range(start_2025, end_2025, freq="D").date
    
    # 构建结果容器
    aligned_data = []
    
    for d_2025 in idx_2025:
        # 构造目标年份的同月同日
        try:
            d_target = date(target_year, d_2025.month, d_2025.day)
            # 查找该日的数据
            val = daily_series.get(d_target, 0)
            real_date = d_target
        except ValueError:
            # 只有当 target_year 非闰年但 d_2025 是 02-29 时才会异常
            # 但 2025 本身是平年，不会产生 02-29，所以这里几乎不会触发
            # 除非 d_2025 来源变了
            val = 0
            real_date = None 
            
        aligned_data.append({
            "axis_date": d_2025,
            "real_date": real_date,
            "count": val
        })
        
    return pd.DataFrame(aligned_data).set_index("axis_date")

def compute_yoy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """计算核心绘图数据。"""
    if "lock_time" not in df.columns:
        raise KeyError("缺少 lock_time 列")

    # 1. 预处理
    df = df.copy()
    df["lock_time"] = pd.to_datetime(df["lock_time"], errors="coerce")
    df = df[df["lock_time"].notna()]
    
    # 2. 获取各年原始数据
    daily_2024 = get_daily_counts(df, 2024)
    daily_2025 = get_daily_counts(df, 2025)
    daily_2026 = get_daily_counts(df, 2026)
    
    # 3. 对齐数据到 2025 轴
    # 2025 本身 (作为 2025 vs 2024 的分子，2026 vs 2025 的分母)
    df_2025_aligned = align_to_2025_axis(daily_2025, 2025)
    
    # 2024 (作为 2025 vs 2024 的分母) - 闰年会自动跳过 02-29 (因为 axis_date 只有 02-28 和 03-01)
    df_2024_aligned = align_to_2025_axis(daily_2024, 2024)
    
    # 2026 (作为 2026 vs 2025 的分子)
    df_2026_aligned = align_to_2025_axis(daily_2026, 2026)
    
    # 4. 计算累计值和同比
    
    # --- Series 1: 2025 累计同比 (2025 vs 2024) ---
    cum_2025 = df_2025_aligned["count"].cumsum()
    cum_2024 = df_2024_aligned["count"].cumsum()
    
    yoy_2025 = (cum_2025 / cum_2024 - 1.0) * 100.0
    yoy_2025 = yoy_2025.replace([np.inf, -np.inf], np.nan)
    
    # --- Series 2: 2026 累计同比 (2026 vs 2025) ---
    cum_2026 = df_2026_aligned["count"].cumsum()
    # 注意：2026 是未来，需要截断到今天
    today = date.today()
    # 找到 2026 对应的 axis_date (即 2025-MM-DD)
    # 如果 today 是 2026-01-27，对应 axis_date 是 2025-01-27
    if today.year == 2026:
        cutoff_date = date(2025, today.month, today.day)
        mask_future = df_2026_aligned.index > cutoff_date
        yoy_2026 = (cum_2026 / cum_2025 - 1.0) * 100.0
        yoy_2026 = yoy_2026.replace([np.inf, -np.inf], np.nan)
        yoy_2026[mask_future] = np.nan
        # 对应的 count 和 cum 也设为 nan 以便 tooltip 不显示未来数据
        df_2026_aligned.loc[mask_future, "count"] = np.nan
        cum_2026[mask_future] = np.nan
    else:
        # 如果不是 2026 年 (比如回测)，全量计算或全量 NaN
        # 假设当前就在 2026 年初，这里简化处理
        yoy_2026 = (cum_2026 / cum_2025 - 1.0) * 100.0
        
    # 5. 整合结果
    result = pd.DataFrame({
        "axis_date": df_2025_aligned.index,
        
        # 2025 曲线数据
        "date_2025": df_2025_aligned["real_date"],
        "daily_2025": df_2025_aligned["count"],
        "cum_2025": cum_2025,
        "yoy_2025": yoy_2025,
        
        # 2026 曲线数据
        "date_2026": df_2026_aligned["real_date"],
        "daily_2026": df_2026_aligned["count"],
        "cum_2026": cum_2026,
        "yoy_2026": yoy_2026
    })
    
    return result

def build_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    
    # --- Helper: Create end label text array ---
    def create_end_label(series, color):
        text_list = [""] * len(series)
        last_valid_idx = series.last_valid_index()
        if last_valid_idx is not None:
            val = series[last_valid_idx]
            # Get integer location
            loc = series.index.get_loc(last_valid_idx)
            # Format: <b>+15.3%</b>
            text_list[loc] = f"<b>{val:+.1f}%</b>"
        return text_list

    # --- Trace 1: 2025 累计同比 (基准) ---
    # Customdata: [real_date, daily, cum, yoy]
    custom_data_2025 = np.stack((
        df["date_2025"].astype(str),
        df["daily_2025"].fillna(0),
        df["cum_2025"].fillna(0),
        df["yoy_2025"].fillna(0)
    ), axis=-1)
    
    text_2025 = create_end_label(df["yoy_2025"], COLOR_MAIN)
    
    fig.add_trace(go.Scatter(
        x=df["axis_date"],
        y=df["yoy_2025"],
        name="2025 累计同比 (vs 2024)",
        mode="lines+text",  # Enable text
        text=text_2025,
        textposition="middle right",
        textfont=dict(color=COLOR_MAIN, size=12),
        cliponaxis=False,   # Allow text to overflow axis
        line=dict(color=COLOR_MAIN, width=2),
        customdata=custom_data_2025,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>" +
            "累计同比: %{y:.1f}%<br>" +
            "当日锁单: %{customdata[1]:.0f}<br>" +
            "累计锁单: %{customdata[2]:.0f}" +
            "<extra>2025</extra>"
        )
    ))
    
    # --- Trace 2: 2026 累计同比 (当前) ---
    custom_data_2026 = np.stack((
        df["date_2026"].apply(lambda x: str(x) if pd.notnull(x) else ""),
        df["daily_2026"].fillna(0),
        df["cum_2026"].fillna(0),
        df["yoy_2026"].fillna(0)
    ), axis=-1)
    
    text_2026 = create_end_label(df["yoy_2026"], COLOR_CONTRAST)
    
    fig.add_trace(go.Scatter(
        x=df["axis_date"],
        y=df["yoy_2026"],
        name="2026 累计同比 (vs 2025)",
        mode="lines+text",  # Enable text
        text=text_2026,
        textposition="middle right",
        textfont=dict(color=COLOR_CONTRAST, size=13),
        cliponaxis=False,   # Allow text to overflow axis
        line=dict(color=COLOR_CONTRAST, width=3), # 加粗以突出
        customdata=custom_data_2026,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>" +
            "累计同比: %{y:.1f}%<br>" +
            "当日锁单: %{customdata[1]:.0f}<br>" +
            "累计锁单: %{customdata[2]:.0f}" +
            "<extra>2026</extra>"
        )
    ))
    
    # --- Layout (Style Applied) ---
    fig.update_layout(
        title="锁单累计同比趋势对比 (2025 vs 2026)",
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
        xaxis=dict(
            title="日期 (对齐到 2025 年)",
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID,
            tickfont=dict(color=COLOR_TEXT),
            title_font=dict(color=COLOR_TEXT),
            showline=True,
            linecolor=COLOR_GRID,
            dtick="M1",
            tickformat="%m-%d" # 只显示月日
        ),
        yaxis=dict(
            title="累计同比 (%)",
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID,
            tickfont=dict(color=COLOR_TEXT),
            title_font=dict(color=COLOR_TEXT),
            showline=True,
            linecolor=COLOR_GRID
        ),
        legend=dict(
            bordercolor=COLOR_TEXT,
            borderwidth=1,
            font=dict(color=COLOR_TEXT),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        margin=dict(l=60, r=80, t=80, b=60), # Increased right margin for text
        height=600
    )
    
    # 添加 0% 参考线
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_TEXT, opacity=0.5)
    
    return fig

def main():
    args = parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)
    
    if not input_path.exists():
        print(f"❌ 错误: 输入文件不存在 {input_path}")
        sys.exit(1)
        
    print(f"🔄 读取数据: {input_path}")
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"❌ 读取 Parquet 失败: {e}")
        sys.exit(1)
        
    print("🔄 计算指标...")
    try:
        metrics_df = compute_yoy_metrics(df)
    except Exception as e:
        print(f"❌ 计算指标失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("🎨 绘制图表...")
    fig = build_figure(metrics_df)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 保存报告: {out_path}")
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print("✅ 完成!")

if __name__ == "__main__":
    main()
