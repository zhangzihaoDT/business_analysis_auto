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

新增模块：LS6 车型增程占比趋势 (2025 vs 2026)
- 仅筛选 series='LS6' 且 lock_time >= 2025-09-10
- 计算日增程占比 (Daily REEV Ratio, MA7 Smoothed)
- 添加 2025 和 2026 的年均值虚线 (Weighted Average)
- 2025 vs 2026 对比

样式遵循 skill/visualization-style 规范。
"""

import argparse
from datetime import date, datetime
import json
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
BUSINESS_DEF_PATH = Path(
    "/Users/zihao_/Documents/github/W52_reasoning/world/business_definition.json"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制锁单累计同比分析图")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="输入 parquet 文件路径")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="输出 HTML 文件路径")
    return parser.parse_args()

def load_business_def() -> dict:
    """加载业务定义文件。"""
    if BUSINESS_DEF_PATH.exists():
        try:
            with open(BUSINESS_DEF_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 加载 business_definition.json 失败: {e}")
    else:
        print(f"⚠️ business_definition.json 不存在: {BUSINESS_DEF_PATH}")
    return {}

def get_launch_date_for_product(product_name: str, biz_def: dict) -> date:
    """根据 product_name 和业务定义获取上市日期 (end_day)。"""
    if not biz_def:
        return None
        
    name = str(product_name).strip()
    series_group = None
    
    # 简化的 series_group 匹配逻辑 (参考 business_definition.json)
    if "LS9" in name:
        series_group = "LS9"
    elif "LS6" in name:
        if "新一代" in name:
            series_group = "CM2"
        elif "全新" in name:
            series_group = "CM1"
        else:
            series_group = "CM0"
    elif "L6" in name:
        if "全新" in name:
            series_group = "DM1"
        else:
            series_group = "DM0"
    
    if not series_group:
        return None
        
    try:
        date_str = biz_def.get("time_periods", {}).get(series_group, {}).get("end")
        if date_str:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        pass
        
    return None

def get_product_type_from_name(product_name: str) -> str:
    """根据 Product Name 派生产品类型（增程/纯电），无法识别返回“未知”。"""
    try:
        if product_name is None:
            return "未知"
        # 处理 NA 与字符串
        s = str(product_name).strip()
        if len(s) == 0 or s.lower() in {"nan", "none", "null"}:
            return "未知"

        # 规则：含“52”或“66”视为增程，否则纯电
        # 参考 analyze_product_types_preference_by_city.py 逻辑
        if any(num in s for num in ["52", "66"]):
            return "增程"
        else:
            return "纯电"
    except Exception:
        return "未知"

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
    """计算核心绘图数据 (累计同比)。"""
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

def compute_ls6_reev_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """计算 LS6 增程占比数据 (2025 vs 2026) - 日销量占比 MA7。
    注意：仅统计 lock_time >= 2025-09-10 的数据。
    """
    if "lock_time" not in df.columns or "series" not in df.columns:
        raise KeyError("缺少 lock_time 或 series 列")
        
    # 1. 预处理：筛选 LS6 并派生 product_type
    df_ls6 = df[df["series"] == "LS6"].copy()
    df_ls6["lock_time"] = pd.to_datetime(df_ls6["lock_time"], errors="coerce")
    df_ls6 = df_ls6[df_ls6["lock_time"].notna()]
    
    # 筛选 2025-09-10 之后的数据
    start_date_filter = pd.Timestamp("2025-09-10")
    df_ls6 = df_ls6[df_ls6["lock_time"] >= start_date_filter]
    
    if df_ls6.empty:
        print("⚠️ 警告: 未找到 LS6 数据 (>= 2025-09-10)")
        return pd.DataFrame()
        
    df_ls6["product_type"] = df_ls6["product_name"].apply(get_product_type_from_name)
    df_ls6["is_reev"] = (df_ls6["product_type"] == "增程").astype(int)
    
    # 2. 计算 2025 和 2026 的每日数据
    results = {}
    years = [2025, 2026]
    today = date.today()
    
    for year in years:
        df_year = df_ls6[df_ls6["lock_time"].dt.year == year]
        
        # 构造全年日期索引以正确计算 MA7
        start_date = date(year, 1, 1)
        if year < today.year:
            end_date = date(year, 12, 31)
        elif year == today.year:
            end_date = today
        else:
            end_date = date(year, 12, 31)

        # 每日聚合
        daily_total = df_year.groupby(df_year["lock_time"].dt.date).size()
        daily_reev = df_year.groupby(df_year["lock_time"].dt.date)["is_reev"].sum()
        
        # Reindex to continuous range
        full_idx = pd.date_range(start_date, end_date, freq='D').date
        s_total = daily_total.reindex(full_idx, fill_value=0)
        s_reev = daily_reev.reindex(full_idx, fill_value=0)
        
        # 计算日占比
        with np.errstate(divide='ignore', invalid='ignore'):
            s_ratio = (s_reev / s_total) * 100.0
        s_ratio = s_ratio.replace([np.inf, -np.inf], np.nan)
        
        # 计算 MA7
        s_ma7 = s_ratio.rolling(window=7, min_periods=1).mean()
        
        # 对齐到 2025 轴
        aligned_total = align_to_2025_axis(s_total, year)
        aligned_reev = align_to_2025_axis(s_reev, year)
        aligned_ma7 = align_to_2025_axis(s_ma7, year)
        
        # 处理未来数据
        if year == today.year:
            cutoff_date_2025 = date(2025, today.month, today.day)
            mask_future = aligned_ma7.index > cutoff_date_2025
            aligned_ma7.loc[mask_future, "count"] = np.nan
            aligned_total.loc[mask_future, "count"] = np.nan
            aligned_reev.loc[mask_future, "count"] = np.nan

        # 处理缺失数据 (无销量日)
        mask_missing = aligned_ma7["real_date"].isna()
        aligned_ma7.loc[mask_missing, "count"] = np.nan
        
        # 计算该年整体均值 (Weighted Average)
        total_count = len(df_year)
        reev_count = df_year["is_reev"].sum()
        avg_ratio = (reev_count / total_count * 100.0) if total_count > 0 else 0.0

        results[year] = {
            "real_date": aligned_total["real_date"],
            "daily_total": aligned_total["count"],
            "daily_reev": aligned_reev["count"],
            "ma7_ratio": aligned_ma7["count"],
            "avg_ratio": avg_ratio
        }
        
    # 3. 整合结果
    result = pd.DataFrame({
        "axis_date": results[2025]["real_date"].index,
        
        "date_2025": results[2025]["real_date"],
        "total_2025": results[2025]["daily_total"],
        "reev_2025": results[2025]["daily_reev"],
        "ratio_ma7_2025": results[2025]["ma7_ratio"],
        "avg_2025": results[2025]["avg_ratio"],
        
        "date_2026": results[2026]["real_date"],
        "total_2026": results[2026]["daily_total"],
        "reev_2026": results[2026]["daily_reev"],
        "ratio_ma7_2026": results[2026]["ma7_ratio"],
        "avg_2026": results[2026]["avg_ratio"]
    })
    
    return result

def compute_reev_product_breakdown(df: pd.DataFrame) -> dict:
    """计算所有增程车型各配置内部占比 (2025 vs 2026) - 日销量占比 MA7。
    分母：当日所有增程车型总销量。
    分子：各增程 Product Name 当日销量。
    返回: {product_name: DataFrame(axis_date, ratio_2025, ratio_2026, ...)}
    """
    if "lock_time" not in df.columns:
        raise KeyError("缺少 lock_time 列")
        
    # 1. 预处理：
    df = df.copy()
    df["lock_time"] = pd.to_datetime(df["lock_time"], errors="coerce")
    df = df[df["lock_time"].notna()]
    
    # 不再限制 LS6 和 2025-09-10
    
    if df.empty:
        return {}
        
    df["product_type"] = df["product_name"].apply(get_product_type_from_name)
    
    # 仅保留增程车型
    df_reev = df[df["product_type"] == "增程"].copy()
    
    if df_reev.empty:
        print("⚠️ 警告: 未找到任何增程车型数据")
        return {}

    reev_products = df_reev["product_name"].unique()
    reev_products = [p for p in reev_products if pd.notnull(p)]
    
    # 加载业务定义
    biz_def = load_business_def()
    
    results = {}
    years = [2025, 2026]
    today = date.today()
    
    # 预计算每年的增程总销量 (分母)
    daily_reev_totals = {}
    for year in years:
        df_year = df_reev[df_reev["lock_time"].dt.year == year]
        daily_reev_totals[year] = df_year.groupby(df_year["lock_time"].dt.date).size()

    # 对每个增程配置计算指标
    for product in reev_products:
        product_data = {}
        
        # 获取该车型的上市日期 (end_day)
        launch_date = get_launch_date_for_product(product, biz_def)
        
        for year in years:
            df_year = df_reev[
                (df_reev["lock_time"].dt.year == year) & 
                (df_reev["product_name"] == product)
            ]
            
            # 构造全年日期索引
            start_date = date(year, 1, 1)
            if year < today.year:
                end_date = date(year, 12, 31)
            elif year == today.year:
                end_date = today
            else:
                end_date = date(year, 12, 31)

            # 每日聚合 (分子)
            daily_prod = df_year.groupby(df_year["lock_time"].dt.date).size()
            
            # 获取分母 (增程总销量)
            s_total_year = daily_reev_totals.get(year, pd.Series(dtype=int))
            
            # Reindex to continuous range
            full_idx = pd.date_range(start_date, end_date, freq='D').date
            s_prod = daily_prod.reindex(full_idx, fill_value=0)
            s_total = s_total_year.reindex(full_idx, fill_value=0)
            
            # 计算日占比 (分母为当日所有增程总量)
            with np.errstate(divide='ignore', invalid='ignore'):
                s_ratio = (s_prod / s_total) * 100.0
            s_ratio = s_ratio.replace([np.inf, -np.inf], np.nan)
            
            # 计算 MA7
            s_ma7 = s_ratio.rolling(window=7, min_periods=1).mean()
            
            # 对齐到 2025 轴
            aligned_prod = align_to_2025_axis(s_prod, year)
            aligned_ma7 = align_to_2025_axis(s_ma7, year)
            
            # --- Filter: Pre-launch data ---
            if launch_date:
                # real_date 可能是 None (比如 02-29 或未来日期在对齐时), 需要处理
                # 只比较非 None 的 real_date
                # 逻辑: if real_date < launch_date, set to NaN
                
                # 创建 mask
                mask_pre_launch = aligned_ma7["real_date"].apply(
                    lambda d: d < launch_date if isinstance(d, date) else False
                )
                
                # 还有一种情况: real_date 是 None (e.g. 02-29 in 2024 aligned to 2025)
                # 这种本身就是 NaN, 不需要额外处理
                
                # 应用 mask
                aligned_ma7.loc[mask_pre_launch, "count"] = np.nan
                aligned_prod.loc[mask_pre_launch, "count"] = np.nan
            
            # 处理未来数据 (仅针对当前年份)
            if year == today.year:
                cutoff_date_2025 = date(2025, today.month, today.day)
                mask_future = aligned_ma7.index > cutoff_date_2025
                aligned_ma7.loc[mask_future, "count"] = np.nan
                aligned_prod.loc[mask_future, "count"] = np.nan

            # 处理缺失数据
            mask_missing = aligned_ma7["real_date"].isna()
            aligned_ma7.loc[mask_missing, "count"] = np.nan

            product_data[year] = {
                "real_date": aligned_ma7["real_date"],
                "daily_count": aligned_prod["count"],
                "ma7_ratio": aligned_ma7["count"]
            }
            
        # 整合该 product 的结果
        # 使用 2025 的轴作为基准
        axis_date = product_data[2025]["real_date"].index
        
        results[product] = pd.DataFrame({
            "axis_date": axis_date,
            "date_2025": product_data[2025]["real_date"],
            "count_2025": product_data[2025]["daily_count"],
            "ratio_2025": product_data[2025]["ma7_ratio"],
            
            "date_2026": product_data[2026]["real_date"],
            "count_2026": product_data[2026]["daily_count"],
            "ratio_2026": product_data[2026]["ma7_ratio"]
        })
        
    return results

def build_figure(df: pd.DataFrame) -> go.Figure:
    """绘制累计同比图表。"""
    fig = go.Figure()
    
    # --- Helper: Create end label text array ---
    def create_end_label(series, color):
        text_list = [""] * len(series)
        last_valid_idx = series.last_valid_index()
        if last_valid_idx is not None:
            val = series[last_valid_idx]
            loc = series.index.get_loc(last_valid_idx)
            text_list[loc] = f"<b>{val:+.1f}%</b>"
        return text_list

    # --- Trace 1: 2025 累计同比 (基准) ---
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
        mode="lines+text",
        text=text_2025,
        textposition="middle right",
        textfont=dict(color=COLOR_MAIN, size=12),
        cliponaxis=False,
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
        mode="lines+text",
        text=text_2026,
        textposition="middle right",
        textfont=dict(color=COLOR_CONTRAST, size=13),
        cliponaxis=False,
        line=dict(color=COLOR_CONTRAST, width=3),
        customdata=custom_data_2026,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>" +
            "累计同比: %{y:.1f}%<br>" +
            "当日锁单: %{customdata[1]:.0f}<br>" +
            "累计锁单: %{customdata[2]:.0f}" +
            "<extra>2026</extra>"
        )
    ))
    
    # --- Layout ---
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
            tickformat="%m-%d"
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
            orientation="v",       # 垂直排列
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02                 # 放置在图表右侧
        ),
        hovermode="x unified",
        margin=dict(l=60, r=80, t=80, b=60),
        height=600
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_TEXT, opacity=0.5)
    return fig

def build_ls6_reev_figure(df: pd.DataFrame) -> go.Figure:
    """绘制 LS6 增程占比图表 (日销量 MA7)。"""
    fig = go.Figure()
    
    if df.empty:
        return fig

    # --- Helper: Create end label ---
    def create_end_label(series, color):
        text_list = [""] * len(series)
        last_valid_idx = series.last_valid_index()
        if last_valid_idx is not None:
            val = series[last_valid_idx]
            loc = series.index.get_loc(last_valid_idx)
            text_list[loc] = f"<b>{val:.1f}%</b>"
        return text_list

    # --- Trace 1: 2025 占比 (MA7) ---
    custom_data_2025 = np.stack((
        df["date_2025"].astype(str),
        df["total_2025"].fillna(0),
        df["reev_2025"].fillna(0),
        df["ratio_ma7_2025"].fillna(0)
    ), axis=-1)
    
    text_2025 = create_end_label(df["ratio_ma7_2025"], COLOR_MAIN)
    
    fig.add_trace(go.Scatter(
        x=df["axis_date"],
        y=df["ratio_ma7_2025"],
        name="2025 日增程占比 (MA7)",
        mode="lines+text",
        text=text_2025,
        textposition="middle right",
        textfont=dict(color=COLOR_MAIN, size=12),
        cliponaxis=False,
        line=dict(color=COLOR_MAIN, width=2),
        customdata=custom_data_2025,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>" +
            "MA7 占比: %{y:.1f}%<br>" +
            "当日总量: %{customdata[1]:.0f}<br>" +
            "当日增程: %{customdata[2]:.0f}" +
            "<extra>2025</extra>"
        )
    ))
    
    # --- Trace 2: 2026 占比 (MA7) ---
    custom_data_2026 = np.stack((
        df["date_2026"].apply(lambda x: str(x) if pd.notnull(x) else ""),
        df["total_2026"].fillna(0),
        df["reev_2026"].fillna(0),
        df["ratio_ma7_2026"].fillna(0)
    ), axis=-1)
    
    text_2026 = create_end_label(df["ratio_ma7_2026"], COLOR_CONTRAST)
    
    fig.add_trace(go.Scatter(
        x=df["axis_date"],
        y=df["ratio_ma7_2026"],
        name="2026 日增程占比 (MA7)",
        mode="lines+text",
        text=text_2026,
        textposition="middle right",
        textfont=dict(color=COLOR_CONTRAST, size=13),
        cliponaxis=False,
        line=dict(color=COLOR_CONTRAST, width=3),
        customdata=custom_data_2026,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>" +
            "MA7 占比: %{y:.1f}%<br>" +
            "当日总量: %{customdata[1]:.0f}<br>" +
            "当日增程: %{customdata[2]:.0f}" +
            "<extra>2026</extra>"
        )
    ))
    
    # --- Trace 3 & 4: Average Lines ---
    # Get averages
    avg_2025 = df["avg_2025"].iloc[0] if "avg_2025" in df.columns else 0
    avg_2026 = df["avg_2026"].iloc[0] if "avg_2026" in df.columns else 0
    
    # 2025 Average
    fig.add_trace(go.Scatter(
        x=[df["axis_date"].min(), df["axis_date"].max()],
        y=[avg_2025, avg_2025],
        name=f"2025 均值 ({avg_2025:.1f}%)",
        mode="lines",
        line=dict(color=COLOR_MAIN, width=1.5, dash="dash"),
        opacity=0.5,
        hoverinfo="skip"
    ))
    
    # 2026 Average
    fig.add_trace(go.Scatter(
        x=[df["axis_date"].min(), df["axis_date"].max()],
        y=[avg_2026, avg_2026],
        name=f"2026 均值 ({avg_2026:.1f}%)",
        mode="lines",
        line=dict(color=COLOR_CONTRAST, width=1.5, dash="dash"),
        opacity=0.8,
        hoverinfo="skip"
    ))
    
    # --- Layout ---
    fig.update_layout(
        title="LS6 车型日增程占比趋势 (MA7 Smoothed, 2025 vs 2026)",
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
            tickformat="%m-%d"
        ),
        yaxis=dict(
            title="日增程占比 (MA7, %)",
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID,
            tickfont=dict(color=COLOR_TEXT),
            title_font=dict(color=COLOR_TEXT),
            showline=True,
            linecolor=COLOR_GRID,
            range=[0, 105]
        ),
        legend=dict(
            bordercolor=COLOR_TEXT,
            borderwidth=1,
            font=dict(color=COLOR_TEXT),
            orientation="v",       # 垂直排列
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02                 # 放置在图表右侧
        ),
        hovermode="x unified",
        margin=dict(l=60, r=80, t=80, b=60),
        height=600
    )
    
    return fig

def build_reev_product_breakdown_figure(metrics_dict: dict) -> go.Figure:
    """绘制所有增程车型各配置内部占比趋势图 (MA7 Smoothed, 2025 vs 2026)。"""
    fig = go.Figure()
    
    if not metrics_dict:
        return fig

    # 颜色列表
    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", 
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
    ]
    
    # --- Helper: Create end label ---
    def create_end_label(series, color, opacity):
        text_list = [""] * len(series)
        last_valid_idx = series.last_valid_index()
        if last_valid_idx is not None:
            val = series[last_valid_idx]
            loc = series.index.get_loc(last_valid_idx)
            text_list[loc] = f"<b>{val:.1f}%</b>"
        return text_list

    # 1. 计算当前 (2026) 占比并排序 Top 2
    current_ratios = []
    for product_name, df in metrics_dict.items():
        s = df["ratio_2026"].dropna()
        if not s.empty:
            last_val = s.iloc[-1]
        else:
            last_val = 0.0
        current_ratios.append((product_name, last_val))
    
    # 降序排序，取 Top 2
    current_ratios.sort(key=lambda x: x[1], reverse=True)
    top2_products = {p[0] for p in current_ratios[:2]}
    
    # 获取 Top 1 和 Top 2 的具体名称，用于分配固定颜色
    top1_name = current_ratios[0][0] if len(current_ratios) >= 1 else None
    top2_name = current_ratios[1][0] if len(current_ratios) >= 2 else None
    
    print(f"🔝 Top 2 Products (Based on 2026 Latest Ratio): {top2_products}")

    for i, (product_name, df) in enumerate(metrics_dict.items()):
        # 颜色分配逻辑：
        # Top 1 -> COLOR_CONTRAST (Orange, 对应 2026 重点色)
        # Top 2 -> COLOR_MAIN (Blue, 对应 2025 基准色)
        # Others -> Cycle through colors list
        if product_name == top1_name:
            color = COLOR_CONTRAST
        elif product_name == top2_name:
            color = COLOR_MAIN
        else:
            color = colors[i % len(colors)]
        
        # 判断是否为 Top 2
        is_top2 = product_name in top2_products
        opacity = 1.0 if is_top2 else 0.2
        width = 3.0 if is_top2 else 1.5
        
        # --- Trace 2025 (Solid, Reduced Opacity) ---
        custom_data_2025 = np.stack((
            df["date_2025"].astype(str),
            df["count_2025"].fillna(0),
            df["ratio_2025"].fillna(0)
        ), axis=-1)
        
        fig.add_trace(go.Scatter(
            x=df["axis_date"],
            y=df["ratio_2025"],
            name=f"{product_name} (2025)",
            mode="lines",
            line=dict(color=color, width=width),  
            opacity=opacity * 0.5,  # 2025 降低透明度以突出 2026
            showlegend=False,  # 不显示 2025 图例
            legendgroup=product_name, # 属于同一图例组
            customdata=custom_data_2025,
            hovertemplate=(
                f"<b>{product_name} (2025)</b><br>" +
                "日期: %{customdata[0]}<br>" +
                "MA7 占比: %{y:.1f}%<br>" +
                "当日销量: %{customdata[1]:.0f}" +
                "<extra></extra>"
            )
        ))
        
        # --- Trace 2026 (Solid, Same Opacity) ---
        custom_data_2026 = np.stack((
            df["date_2026"].apply(lambda x: str(x) if pd.notnull(x) else ""),
            df["count_2026"].fillna(0),
            df["ratio_2026"].fillna(0)
        ), axis=-1)
        
        text_2026 = create_end_label(df["ratio_2026"], color, opacity)
        
        # 非 Top 2 的 text 也可以隐藏或者淡化
        mode_2026 = "lines+text" if is_top2 else "lines"
        
        fig.add_trace(go.Scatter(
            x=df["axis_date"],
            y=df["ratio_2026"],
            name=product_name,  # 图例只显示车型名称，不带年份
            mode=mode_2026,
            text=text_2026,
            textposition="middle right",
            textfont=dict(color=color, size=12),
            cliponaxis=False,
            line=dict(color=color, width=width),
            opacity=opacity,
            legendgroup=product_name, # 属于同一图例组
            customdata=custom_data_2026,
            hovertemplate=(
                f"<b>{product_name} (2026)</b><br>" +
                "日期: %{customdata[0]}<br>" +
                "MA7 占比: %{y:.1f}%<br>" +
                "当日销量: %{customdata[1]:.0f}" +
                "<extra></extra>"
            )
        ))

    # --- Layout ---
    fig.update_layout(
        title="增程车型各配置内部占比趋势 (分母=增程总销量, MA7, Top2 Highlighted)",
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
            tickformat="%m-%d"
        ),
        yaxis=dict(
            title="日销量占比 (MA7, %)",
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID,
            tickfont=dict(color=COLOR_TEXT),
            title_font=dict(color=COLOR_TEXT),
            showline=True,
            linecolor=COLOR_GRID,
            range=[0, None]  # 自适应上限
        ),
        legend=dict(
            bordercolor=COLOR_TEXT,
            borderwidth=1,
            font=dict(color=COLOR_TEXT),
            orientation="v",       # 垂直排列
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02                 # 放置在图表右侧
        ),
        hovermode="x unified",
        margin=dict(l=60, r=80, t=80, b=60),
        height=600
    )
    
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
        
    print("🔄 计算指标 1 (累计同比)...")
    try:
        metrics_df = compute_yoy_metrics(df)
        fig1 = build_figure(metrics_df)
    except Exception as e:
        print(f"❌ 计算累计同比失败: {e}")
        import traceback
        traceback.print_exc()
        fig1 = None

    print("🔄 计算指标 2 (LS6 增程占比)...")
    try:
        ls6_metrics = compute_ls6_reev_metrics(df)
        fig2 = build_ls6_reev_figure(ls6_metrics)
    except Exception as e:
        print(f"❌ 计算 LS6 增程占比失败: {e}")
        import traceback
        traceback.print_exc()
        fig2 = None
        
    print("🔄 计算指标 3 (所有增程配置占比)...")
    try:
        reev_prod_metrics = compute_reev_product_breakdown(df)
        fig3 = build_reev_product_breakdown_figure(reev_prod_metrics)
    except Exception as e:
        print(f"❌ 计算增程配置占比失败: {e}")
        import traceback
        traceback.print_exc()
        fig3 = None
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 保存报告: {out_path}")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("<html><head><meta charset='utf-8'><title>锁单分析报告</title></head><body>")
        if fig1:
            f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        if fig2:
            f.write("<br><hr><br>")
            f.write(fig2.to_html(full_html=False, include_plotlyjs=False))
        if fig3:
            f.write("<br><hr><br>")
            f.write(fig3.to_html(full_html=False, include_plotlyjs=False))
        f.write("</body></html>")
        
    print("✅ 完成!")

if __name__ == "__main__":
    main()
