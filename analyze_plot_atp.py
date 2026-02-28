#!/usr/bin/env python3
"""
Analyze Plot ATP (Average Transaction Price)
基于开票价格 (invoice_amount) 的分析脚本。

参考: /Users/zihao_/Documents/coding/dataset/scripts/analyze_plot_lock_ratio.py
业务定义: /Users/zihao_/Documents/github/W52_reasoning/world/business_definition.json

Output:
- Table: Series, Series Group, Product Type, Product Name, Order Count, Average Price
- Time Filter: invoice_upload_time in 2025 (2025-01-01 ~ 2025-12-31)
"""

import pandas as pd
import json
import re
import argparse
from pathlib import Path
import plotly.graph_objects as go

# --- Constants ---
PARQUET_FILE = Path("/Users/zihao_/Documents/coding/dataset/formatted/order_full_data.parquet")
BUSINESS_DEF_FILE = Path("/Users/zihao_/Documents/github/W52_reasoning/world/business_definition.json")
OUTPUT_HTML = Path("/Users/zihao_/Documents/coding/dataset/reports/atp_analysis.html")

# --- Visualization Style Constants ---
COLOR_MAIN = "#3498DB"      # Blue (用于 2025 - 基准/完整年)
COLOR_CONTRAST = "#E67E22"  # Orange (用于 2026 - 当前/观察年)
COLOR_DARK = "#373f4a"
COLOR_GRID = "#ebedf0"
COLOR_TEXT = "#7B848F"
COLOR_BG = "#FFFFFF"

def load_business_definition(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_sql_condition(df: pd.DataFrame, condition_str: str) -> pd.Series:
    """
    解析简单的 SQL-like 条件并应用到 DataFrame
    """
    # 1. Replace NOT LIKE
    def not_like_replacer(match):
        val = match.group(1)
        return f"~df['product_name'].str.contains('{val}', na=False, regex=False)"
    
    condition_str = re.sub(r"product_name\s+NOT\s+LIKE\s+'%([^%]+)%+'", not_like_replacer, condition_str)
    
    # 2. Replace LIKE
    def like_replacer(match):
        val = match.group(1)
        return f"df['product_name'].str.contains('{val}', na=False, regex=False)"
        
    condition_str = re.sub(r"product_name\s+LIKE\s+'%([^%]+)%+'", like_replacer, condition_str)
    
    # 3. Replace AND / OR / ELSE
    condition_str = condition_str.replace(" AND ", " & ").replace(" OR ", " | ")
    
    if condition_str.strip() == "ELSE":
        return pd.Series([True] * len(df), index=df.index)

    # 4. Eval
    try:
        return eval(condition_str)
    except Exception as e:
        print(f"⚠️ Failed to parse condition: {condition_str}, Error: {e}")
        return pd.Series([False] * len(df), index=df.index)

def apply_business_logic(df: pd.DataFrame, business_def: dict) -> pd.DataFrame:
    """
    Apply Series Group and Product Type logic.
    """
    df = df.copy()
    
    # 1. Series Group Logic
    series_group_logic = business_def.get("series_group_logic", {})
    df["series_group"] = "其他" # Default
    
    # Iterate through logic. Note: The order in JSON matters if overlapping, but here keys are likely exclusive or priority based.
    # Usually specific to general. "ELSE" should be last.
    # We'll assume the dictionary order is somewhat preserved or we process "其他" last.
    
    # We need to process non-ELSE first
    for group_name, condition in series_group_logic.items():
        if condition == "ELSE":
            continue
        mask = parse_sql_condition(df, condition)
        df.loc[mask, "series_group"] = group_name
        
    # 2. Product Type Logic
    product_type_logic = business_def.get("product_type_logic", {})
    df["product_type"] = "未知"
    for p_type, condition in product_type_logic.items():
        mask = parse_sql_condition(df, condition)
        df.loc[mask, "product_type"] = p_type
        
    # 3. Series Mapping (LS6, L6 etc.) from series_group
    model_series_mapping = business_def.get("model_series_mapping", {})
    # Reverse mapping for easier lookup: CM0 -> LS6
    group_to_series = {}
    for series, groups in model_series_mapping.items():
        for g in groups:
            group_to_series[g] = series
            
    df["series_derived"] = df["series_group"].map(group_to_series)
    # Fill NaN (those not in mapping, e.g. LS7, L7, 其他) with series_group or handle separately
    # If series_group is LS7, and not in mapping, maybe we should keep it as LS7?
    # Checking series_group_logic: "LS7": "product_name LIKE '%LS7%'".
    # So if series_group is LS7, series_derived is NaN because LS7 is not in model_series_mapping keys (only LS6, L6 there).
    # We should fillna with series_group if it looks like a series, or keep it.
    # For now, let's fill with series_group if derived is null, but maybe strip numbers?
    # Actually, user wants "series (车型)". LS7 is a series.
    # So if not mapped, use series_group (which seems to include LS7, L7).
    df["series_derived"] = df["series_derived"].fillna(df["series_group"])
    
    return df

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze ATP (Average Transaction Price)")
    parser.add_argument("--order-types", nargs="+", default=["用户车"], help="Filter by order_type (default: 用户车)")
    return parser.parse_args()

def build_summary_figure(summary_df: pd.DataFrame, year_label: str, color: str = COLOR_MAIN) -> go.Figure:
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(summary_df.columns),
            fill_color=color,
            font=dict(color='white', size=12),
            align='left',
            line_color=COLOR_GRID
        ),
        cells=dict(
            values=[summary_df[k].tolist() for k in summary_df.columns],
            fill_color=COLOR_BG,
            font=dict(color=COLOR_DARK, size=11),
            align='left',
            line_color=COLOR_GRID,
            height=25
        )
    )])
    
    fig.update_layout(
        title=dict(
            text=f"Order Type Distribution ({year_label})<br><sup>Data Source: Invoice {year_label} & Lock Time Not Null</sup>",
            font=dict(color=COLOR_DARK, size=16),
            x=0.01,
            xanchor='left'
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=300 if len(summary_df) < 10 else 500
    )
    return fig

def build_analysis_figure(agg_df: pd.DataFrame, filter_info: str, year_label: str, color: str = COLOR_MAIN) -> go.Figure:
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(agg_df.columns),
            fill_color=color,
            font=dict(color='white', size=12),
            align='left',
            line_color=COLOR_GRID
        ),
        cells=dict(
            values=[agg_df[k].tolist() for k in agg_df.columns],
            fill_color=COLOR_BG,
            font=dict(color=COLOR_DARK, size=11),
            align='left',
            line_color=COLOR_GRID,
            height=25
        )
    )])
    
    fig.update_layout(
        title=dict(
            text=f"Invoice Price (ATP) Analysis {year_label}<br><sup>{filter_info}</sup>",
            font=dict(color=COLOR_DARK, size=16),
            x=0.01,
            xanchor='left'
        ),
        margin=dict(l=20, r=20, t=80, b=20),
        height=800
    )
    return fig

def generate_order_type_summary(df_subset: pd.DataFrame) -> pd.DataFrame:
    """
    Helper to generate order type summary table data
    """
    if "order_type" not in df_subset.columns:
        return pd.DataFrame()

    df_summary = df_subset.copy()
    df_summary["order_type"] = df_summary["order_type"].fillna("Unknown").astype(str)
    
    def get_category(ot):
        if ot == "用户车": return "用户车"
        if ot in ["大客户", "员工"]: return "员工和大客户"
        if ot == "集团员工": return "集团员工"
        if ot == "经销商员工": return "经销商员工"
        if ot == "试驾车": return "试驾车"
        return "其他"
        
    df_summary["category"] = df_summary["order_type"].apply(get_category)
    
    # Aggregate
    grp = df_summary.groupby("category")["order_number"].count()
    
    # Prepare rows
    rows = []
    fixed_cats = ["用户车", "员工和大客户", "集团员工", "经销商员工", "试驾车"]
    for cat in fixed_cats:
        count = grp.get(cat, 0)
        rows.append({"Category": cat, "Order Count": count})
        
    # Handle "其他"
    other_count = grp.get("其他", 0)
    other_details = df_summary[df_summary["category"] == "其他"].groupby("order_type")["order_number"].count()
    details_list = []
    for t, c in other_details.items():
        t_label = "未知" if t == "Unknown" else t
        details_list.append(f"{t_label}: {c}")
    other_details_str = "、".join(details_list)
    other_label = f"其他（{other_details_str}）" if other_details_str else "其他"
    
    if other_count > 0 or "其他" in grp.index:
         rows.append({"Category": other_label, "Order Count": other_count})
         
    # Add Total Row
    total_count = sum(r["Order Count"] for r in rows)
    rows.append({"Category": "总计", "Order Count": total_count})
    
    return pd.DataFrame(rows)

def generate_analysis_aggregation(df_processed: pd.DataFrame) -> pd.DataFrame:
    """
    Helper to generate ATP analysis aggregation table data
    """
    df_processed["product_name"] = df_processed["product_name"].fillna("Unknown")
    
    group_cols = ["series_derived", "series_group", "product_type", "product_name"]
    agg_df = df_processed.groupby(group_cols).agg(
        order_count=("order_number", "count"),
        avg_price=("invoice_amount", "mean")
    ).reset_index()
    
    agg_df.columns = ["Series", "Series Group", "Product Type", "Product Name", "Order Count", "Avg Price"]
    agg_df = agg_df.sort_values(by=["Series", "Series Group", "Product Name"])
    
    # Add Total Row
    if not df_processed.empty:
        total_count = len(df_processed)
        total_avg_price = df_processed["invoice_amount"].mean()
        total_row = pd.DataFrame([{
            "Series": "总计", "Series Group": "", "Product Type": "", "Product Name": "",
            "Order Count": total_count, "Avg Price": total_avg_price
        }])
        agg_df = pd.concat([agg_df, total_row], ignore_index=True)

    agg_df["Avg Price"] = agg_df["Avg Price"].map(lambda x: f"{x:,.0f}")
    return agg_df

def build_metric_figure(df: pd.DataFrame, year_label: str, color: str = COLOR_DARK) -> go.Figure:
    """
    Build Indicator figure for Overall Weighted Average Invoice Price.
    Formula: Total Invoice Amount / Total Order Count
    """
    total_amount = df["invoice_amount"].sum()
    total_count = len(df)
    avg_price = df["invoice_amount"].mean() if total_count > 0 else 0
    
    fig = go.Figure(go.Indicator(
        mode = "number",
        value = avg_price,
        number = {'prefix': "¥", "valueformat": ",.0f", "font": {"color": color}},
        title = {"text": f"Global Average Invoice Price (ATP) {year_label} - N={total_count}"},
        domain = {'row': 0, 'column': 0}
    ))
    
    fig.update_layout(
        annotations=[
            dict(
                x=0.5,
                y=-0.25,
                text=(
                    f"<b>Calculation Logic:</b> Based on Raw Order Data (All Series/Products Mixed)<br>"
                    f"Formula: Total Invoice Amount (¥{total_amount:,.0f}) / Total Order Count ({total_count})<br>"
                    f"<i>*This value is inherently volume-weighted by Order Count across all dimensions (Series, Group, Product Type, Name).</i>"
                ),
                showarrow=False,
                xref="paper",
                yref="paper",
                font=dict(size=14, color="gray")
            )
        ],
        height=350,
        margin=dict(l=20, r=20, t=50, b=80)
    )
    return fig

def build_period_comparison_figure(df_full: pd.DataFrame, filter_info: str) -> go.Figure:
    """
    Build Comparison Table (Plot 4): Weighted Average Invoice Price by Period & Segment.
    Expanded to show Series breakdown, Share, Count, and Price.
    Periods:
      - 2025 Full Year
      - 2025 Dec
      - 2026 Jan
    Segments:
      - All Series
      - Sedan (L6+L7)
      - SUV (LS6+LS7+LS9)
    """
    
    # Define Periods
    # (Label, Filter Function, Color Logic handled later)
    periods = [
        ("2025 Full Year", lambda df: df[df["invoice_upload_time"].dt.year == 2025]),
        ("2025 Dec", lambda df: df[(df["invoice_upload_time"].dt.year == 2025) & (df["invoice_upload_time"].dt.month == 12)]),
        ("2026 Jan", lambda df: df[(df["invoice_upload_time"].dt.year == 2026) & (df["invoice_upload_time"].dt.month == 1)])
    ]
    
    # Define Segments
    segments = {
        "All Series": lambda df: df,
        "Sedan (L6+L7)": lambda df: df[df["series_derived"].isin(["L6", "L7"])],
        "SUV (LS6+LS7+LS9)": lambda df: df[df["series_derived"].isin(["LS6", "LS7", "LS9"])]
    }
    
    rows = []
    
    for seg_name, seg_func in segments.items():
        # Get Segment Data (Global)
        seg_df_global = seg_func(df_full)
        
        # Identify Series in this Segment (sorted)
        # Use series_derived
        series_list = sorted(seg_df_global["series_derived"].dropna().unique())
        
        # Prepare data structure for this segment block
        # We need to calculate metrics for each series and the total segment for each period
        
        # 1. Calculate Segment Totals per Period (for Share calculation)
        period_totals = {}
        for p_name, p_func in periods:
            p_df = p_func(seg_df_global)
            period_totals[p_name] = len(p_df)
            
        # 2. Build Rows for each Series
        for series in series_list:
            row_data = {"Segment": seg_name, "Series": series}
            for p_name, p_func in periods:
                p_df_seg = p_func(seg_df_global)
                p_df_series = p_df_seg[p_df_seg["series_derived"] == series]
                
                count = len(p_df_series)
                amount = p_df_series["invoice_amount"].sum()
                avg_price = amount / count if count > 0 else 0
                
                total_seg_count = period_totals[p_name]
                share = count / total_seg_count if total_seg_count > 0 else 0
                
                row_data[f"{p_name}_Count"] = count
                row_data[f"{p_name}_Share"] = f"{share:.1%}"
                row_data[f"{p_name}_Price"] = f"¥{avg_price:,.0f}"
            rows.append(row_data)
            
        # 3. Build Summary Row for Segment
        summary_row = {"Segment": seg_name, "Series": "<b>Total</b>"}
        for p_name, p_func in periods:
            p_df_seg = p_func(seg_df_global)
            
            count = len(p_df_seg)
            amount = p_df_seg["invoice_amount"].sum()
            avg_price = amount / count if count > 0 else 0
            
            summary_row[f"{p_name}_Count"] = f"<b>{count}</b>"
            summary_row[f"{p_name}_Share"] = "<b>100.0%</b>"
            summary_row[f"{p_name}_Price"] = f"<b>¥{avg_price:,.0f}</b>"
        rows.append(summary_row)
        
    res_df = pd.DataFrame(rows)
    
    # Define Columns and Colors
    final_cols = ["Segment", "Series"]
    header_colors = [COLOR_DARK, COLOR_DARK]
    
    for p_name, _ in periods:
        # Order: Count, Share, Price
        final_cols.extend([f"{p_name}_Count", f"{p_name}_Share", f"{p_name}_Price"])
        
        # Color
        c = COLOR_CONTRAST if "2026" in p_name else COLOR_MAIN
        header_colors.extend([c, c, c])
        
    res_df = res_df[final_cols]
    
    # Rename columns for display (remove prefix for cleaner look? Or keep to distinguish?)
    # Keeping full names is safer, but maybe "Count", "Share", "Avg Price" with Period Header?
    # Plotly Table headers are single level. We will use "Period Count", "Period Share", "Period Price"
    
    display_cols = ["Segment", "Series"]
    for p_name, _ in periods:
        display_cols.append(f"{p_name}<br>Count")
        display_cols.append(f"{p_name}<br>Share")
        display_cols.append(f"{p_name}<br>Avg Price")
        
    # Build Table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=display_cols,
            fill_color=header_colors,
            font=dict(color='white', size=11),
            align='center',
            line_color=COLOR_GRID,
            height=40
        ),
        cells=dict(
            values=[res_df[k].tolist() for k in res_df.columns],
            fill_color=COLOR_BG,
            font=dict(color='black', size=11),
            align='center',
            line_color=COLOR_GRID,
            height=30
        )
    )])
    
    fig.update_layout(
        title=dict(
            text="Weighted Average Invoice Price by Period & Segment (Detailed)",
            x=0.01,
            xanchor='left'
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=len(res_df) * 30 + 150
    )
    
    return fig

def build_ls6_launch_price_figure(df: pd.DataFrame, business_def: dict) -> go.Figure:
    """
    Plot 5: LS6 Invoice Price vs Days Since Launch (End Day) by Series Group.
    """
    # 1. Filter LS6
    if "series_derived" not in df.columns:
        return None
        
    df_ls6 = df[df["series_derived"] == "LS6"].copy()
    if df_ls6.empty:
        return None
        
    # 2. Get Launch Dates
    time_periods = business_def.get("time_periods", {})
    
    # 3. Calculate Days Since Launch
    def get_launch_date(group):
        period_info = time_periods.get(group)
        if period_info and "end" in period_info:
             return pd.Timestamp(period_info["end"])
        return pd.NaT
        
    unique_groups = df_ls6["series_group"].unique()
    group_launch_map = {g: get_launch_date(g) for g in unique_groups}
    
    # Remove groups without launch date
    valid_groups = {g: d for g, d in group_launch_map.items() if pd.notnull(d)}
    df_ls6 = df_ls6[df_ls6["series_group"].isin(valid_groups)].copy()
    
    if df_ls6.empty:
        return None

    df_ls6["launch_date"] = df_ls6["series_group"].map(valid_groups)
    df_ls6["days_since_launch"] = (df_ls6["invoice_upload_time"] - df_ls6["launch_date"]).dt.days
    
    # Filter range: -60 to +365 days to focus on launch period
    df_ls6 = df_ls6[(df_ls6["days_since_launch"] >= -60) & (df_ls6["days_since_launch"] <= 365)]
    
    # 4. Aggregate (Daily Mean & Count)
    agg_df = df_ls6.groupby(["series_group", "days_since_launch"]).agg(
        invoice_amount=("invoice_amount", "mean"),
        order_count=("invoice_amount", "count")
    ).reset_index()
    
    # 5. Plot
    fig = go.Figure()
    
    # Color mapping for LS6 generations
    # CM2 (Newest) -> Contrast (Orange)
    # CM1 (Previous) -> Main (Blue)
    # CM0 (Oldest) -> Text Color (Grey/Blueish Grey)
    color_map = {
        "CM2": COLOR_CONTRAST,
        "CM1": COLOR_MAIN,
        "CM0": COLOR_TEXT 
    }
    
    # Sort groups by launch date
    sorted_groups = sorted(valid_groups.keys(), key=lambda g: valid_groups[g])
    
    for g in sorted_groups:
        sub = agg_df[agg_df["series_group"] == g].sort_values("days_since_launch")
        if sub.empty:
            continue
            
        # Smoothing (7-day rolling)
        sub["smooth_price"] = sub["invoice_amount"].rolling(window=7, min_periods=1).mean()
        sub["smooth_count"] = sub["order_count"].rolling(window=7, min_periods=1).mean()
        
        launch_date_str = valid_groups[g].strftime("%Y-%m-%d")
        
        # Determine color
        line_color = color_map.get(g, COLOR_DARK) # Default to dark if unknown
        
        fig.add_trace(go.Scatter(
            x=sub["days_since_launch"],
            y=sub["smooth_price"],
            mode='lines',
            name=f"{g} (Launch: {launch_date_str})",
            line=dict(color=line_color, width=3),
            customdata=sub[["smooth_count"]],
            hovertemplate=(
                "<b>%{y:,.0f}</b><br>" +
                "Day: %{x}<br>" +
                "Count (7d Avg): %{customdata[0]:.1f}<br>" +
                "<extra>%{fullData.name}</extra>"
            )
        ))

        # Add annotation at the end of the line
        last_point = sub.iloc[-1]
        fig.add_annotation(
            x=last_point["days_since_launch"],
            y=last_point["smooth_price"],
            text=f"¥{last_point['smooth_price']:,.0f}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=11, color=line_color, family="Arial, sans-serif"),
            xshift=8
        )
        
    fig.update_layout(
        title=dict(
            text="LS6 Invoice Price Trends vs Launch Time (Aligned by End Day)",
            font=dict(color=COLOR_DARK, size=16),
            x=0.01,
            xanchor='left'
        ),
        xaxis=dict(
            title="Days Since Launch (0 = Period End Date)",
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID,
            tickfont=dict(color=COLOR_TEXT),
            title_font=dict(color=COLOR_TEXT),
            showline=True,
            linecolor=COLOR_GRID
        ),
        yaxis=dict(
            title="Average Invoice Price (7-day Smoothed)",
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
            font=dict(color=COLOR_TEXT)
        ),
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
        hovermode="x unified",
        height=600,
        margin=dict(l=20, r=60, t=60, b=20) # Increase right margin for labels
    )
    
    return fig

def build_reev_price_trend_figure(df: pd.DataFrame) -> go.Figure:
    """
    Plot 6: REEV Main Selling Models Price Trend.
    Models: '智己LS9 66 Ultra', '智己LS9 52 Ultra', '新一代智己LS6 52 Max', '新一代智己LS6 66 Max'
    """
    target_models = ['智己LS9 66 Ultra', '智己LS9 52 Ultra', '新一代智己LS6 52 Max', '新一代智己LS6 66 Max']
    
    # Filter for target models
    df_filtered = df[df["product_name"].isin(target_models)].copy()
    
    if df_filtered.empty:
        return None

    # Aggregate by Date
    if "invoice_upload_time" not in df_filtered.columns:
        return None

    df_filtered["date"] = df_filtered["invoice_upload_time"].dt.date
    agg_df = df_filtered.groupby(["product_name", "date"]).agg(
        avg_price=("invoice_amount", "mean"),
        count=("invoice_amount", "count")
    ).reset_index()
    
    agg_df["date"] = pd.to_datetime(agg_df["date"])
    
    fig = go.Figure()
    
    # Use consistent colors from the file
    colors = {
        '新一代智己LS6 52 Max': COLOR_CONTRAST,  # Orange (Key Model)
        '新一代智己LS6 66 Max': COLOR_TEXT,      # Grey (Secondary LS6)
        '智己LS9 52 Ultra': COLOR_MAIN,        # Blue (Key LS9)
        '智己LS9 66 Ultra': COLOR_DARK         # Dark Grey (Secondary LS9)
    }
    
    # Sort dates
    agg_df = agg_df.sort_values("date")
    
    for model in target_models:
        sub = agg_df[agg_df["product_name"] == model].copy()
        if sub.empty:
            continue
            
        # Reindex to full date range for better smoothing
        sub = sub.set_index("date").sort_index()
        idx = pd.date_range(sub.index.min(), sub.index.max())
        sub = sub.reindex(idx)
        
        # Smoothing (7-day rolling)
        sub["smooth_price"] = sub["avg_price"].rolling(window=7, min_periods=1).mean()
        sub["smooth_count"] = sub["count"].rolling(window=7, min_periods=1).mean()
        
        # Reset index for plotting and drop NaNs
        sub = sub.reset_index().rename(columns={"index": "date"})
        sub = sub.dropna(subset=["smooth_price"])
        
        if sub.empty:
            continue
            
        line_color = colors.get(model, COLOR_DARK)
        
        fig.add_trace(go.Scatter(
            x=sub["date"],
            y=sub["smooth_price"],
            mode='lines',
            name=model,
            line=dict(color=line_color, width=3),
            customdata=sub[["smooth_count"]],
            hovertemplate=(
                "<b>%{y:,.0f}</b><br>" +
                "Date: %{x|%Y-%m-%d}<br>" +
                "Count (7d Avg): %{customdata[0]:.1f}<br>" +
                "<extra>%{fullData.name}</extra>"
            )
        ))

        # Add annotation at the end of the line
        last_point = sub.iloc[-1]
        fig.add_annotation(
            x=last_point["date"],
            y=last_point["smooth_price"],
            text=f"¥{last_point['smooth_price']:,.0f}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=11, color=line_color, family="Arial, sans-serif"),
            xshift=8
        )
        
    fig.update_layout(
        title=dict(
            text="REEV Main Selling Models Price Trend (7-day Smoothed)",
            font=dict(color=COLOR_DARK, size=16),
            x=0.01,
            xanchor='left'
        ),
        xaxis=dict(
            title="Date",
            gridcolor=COLOR_GRID,
            showline=True,
            linecolor=COLOR_GRID
        ),
        yaxis=dict(
            title="Average Invoice Price",
            gridcolor=COLOR_GRID,
            showline=True,
            linecolor=COLOR_GRID
        ),
        legend=dict(
            bordercolor=COLOR_TEXT,
            borderwidth=1,
            font=dict(color=COLOR_TEXT)
        ),
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
        hovermode="x unified",
        height=600,
        margin=dict(l=20, r=60, t=60, b=20)
    )
    
    return fig

def main():
    args = parse_args()

    # 1. Load Data
    print(f"Loading data from {PARQUET_FILE}...")
    df = pd.read_parquet(PARQUET_FILE)
    print(f"Loaded {len(df)} rows.")
    
    # 2. Basic Cleaning
    if "invoice_upload_time" not in df.columns or "lock_time" not in df.columns:
        print("Error: 'invoice_upload_time' or 'lock_time' column missing.")
        return

    # Convert to datetime
    df["invoice_upload_time"] = pd.to_datetime(df["invoice_upload_time"], errors='coerce')

    # Load Business Def
    print(f"Loading business definitions from {BUSINESS_DEF_FILE}...")
    business_def = load_business_definition(BUSINESS_DEF_FILE)
    
    # --- Branch for Plot 5 (Full History) ---
    # Filter: Lock Time Not Null (but keep historical dates for Launch Analysis)
    df_history = df[df["lock_time"].notnull()].copy()
    
    # Apply Order Type Filter (Global arg) to history
    target_types = args.order_types
    if "order_type" in df_history.columns:
        df_history["order_type"] = df_history["order_type"].fillna("Unknown").astype(str)
        if target_types:
            df_history = df_history[df_history["order_type"].isin(target_types)].copy()
            
    # Apply Business Logic to history
    df_history_processed = apply_business_logic(df_history, business_def)
    
    # Generate Plot 5
    fig5 = build_ls6_launch_price_figure(df_history_processed, business_def)
    
    # Generate Plot 6
    fig6 = build_reev_price_trend_figure(df_history_processed)
    # ----------------------------------------
    
    # 3. Global Filter: Lock Time Not Null & Invoice Time >= 2025-01-01 (Standard Logic)
    start_global = pd.Timestamp("2025-01-01")
    df_clean = df[
        (df["lock_time"].notnull()) & 
        (df["invoice_upload_time"] >= start_global)
    ].copy()
    print(f"Data with Lock Time not null & Invoice Time >= 2025-01-01: {len(df_clean)} rows.")
    
    # 4. Prepare Data for Summary (2025 & 2026)
    # 2025
    start_2025 = pd.Timestamp("2025-01-01")
    end_2025 = pd.Timestamp("2025-12-31 23:59:59")
    mask_2025 = (df_clean["invoice_upload_time"] >= start_2025) & (df_clean["invoice_upload_time"] <= end_2025)
    df_2025 = df_clean[mask_2025].copy()
    print(f"Data for 2025 Analysis: {len(df_2025)} rows.")

    # 2026
    start_2026 = pd.Timestamp("2026-01-01")
    end_2026 = pd.Timestamp("2026-12-31 23:59:59")
    mask_2026 = (df_clean["invoice_upload_time"] >= start_2026) & (df_clean["invoice_upload_time"] <= end_2026)
    df_2026 = df_clean[mask_2026].copy()
    print(f"Data for 2026 Analysis: {len(df_2026)} rows.")

    # --- Plot 1: Order Type Summary (2025 & 2026) ---
    summary_df_2025 = generate_order_type_summary(df_2025)
    print("\n--- Order Type Summary (2025) ---")
    if not summary_df_2025.empty:
        print(summary_df_2025.to_string(index=False))

    summary_df_2026 = generate_order_type_summary(df_2026)
    print("\n--- Order Type Summary (2026) ---")
    if not summary_df_2026.empty:
        print(summary_df_2026.to_string(index=False))

    # 5. Apply Order Type Filter (Global for Analysis Plots)
    df_filtered_full = df_clean.copy()
    
    if "order_type" in df_filtered_full.columns:
        df_filtered_full["order_type"] = df_filtered_full["order_type"].fillna("Unknown").astype(str)
        if target_types:
            print(f"Filtering by order_type: {target_types}")
            df_filtered_full = df_filtered_full[df_filtered_full["order_type"].isin(target_types)].copy()
            print(f"Data after order_type filter (Full Time Range 2025+): {len(df_filtered_full)} rows.")
    
    # 6. Apply Business Logic (Full Time Range 2025+)
    # Note: business_def already loaded
    df_processed_full = apply_business_logic(df_filtered_full, business_def)
    
    # 7. Prepare Processed Data for Plots 2 & 3 (2025 & 2026)
    # 2025 Processed
    mask_2025_processed = (df_processed_full["invoice_upload_time"] >= start_2025) & (df_processed_full["invoice_upload_time"] <= end_2025)
    df_2025_processed = df_processed_full[mask_2025_processed].copy()
    
    # 2026 Processed
    mask_2026_processed = (df_processed_full["invoice_upload_time"] >= start_2026) & (df_processed_full["invoice_upload_time"] <= end_2026)
    df_2026_processed = df_processed_full[mask_2026_processed].copy()
    
    # --- Plot 2: Detailed Analysis Table (2025 & 2026) ---
    agg_df_2025 = generate_analysis_aggregation(df_2025_processed)
    print("\n--- Analysis Result (2025 Invoice Data) ---")
    print(agg_df_2025.to_string(index=False))

    agg_df_2026 = generate_analysis_aggregation(df_2026_processed)
    print("\n--- Analysis Result (2026 Invoice Data) ---")
    print(agg_df_2026.to_string(index=False))
    
    # --- Figures Construction ---
    filter_info = f"Filter: Order Type = {args.order_types}"
    
    # Fig 1: Summaries
    fig1_2025 = None
    if not summary_df_2025.empty:
        fig1_2025 = build_summary_figure(summary_df_2025, "2025 Full Year", COLOR_MAIN)

    fig1_2026 = None
    if not summary_df_2026.empty:
        fig1_2026 = build_summary_figure(summary_df_2026, "2026 Full Year", COLOR_CONTRAST)
        
    # Fig 2: Analysis
    fig2_2025 = build_analysis_figure(agg_df_2025, filter_info, "2025", COLOR_MAIN)
    fig2_2026 = build_analysis_figure(agg_df_2026, filter_info, "2026", COLOR_CONTRAST)
    
    # Fig 3: Indicator
    fig3_2025 = build_metric_figure(df_2025_processed, "2025", COLOR_MAIN)
    fig3_2026 = build_metric_figure(df_2026_processed, "2026", COLOR_CONTRAST)
    
    # Fig 4: Comparison
    fig4 = build_period_comparison_figure(df_processed_full, filter_info)
    
    # 8. Save to HTML
    print(f"Saving HTML report to {OUTPUT_HTML}...")
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write("<html><head><meta charset='utf-8'><title>Invoice Price Analysis Report</title></head><body>")
        
        # Order Type Summaries
        if fig1_2025:
             f.write(fig1_2025.to_html(full_html=False, include_plotlyjs='cdn'))
             f.write("<br>")
        if fig1_2026:
             include_js = False if fig1_2025 else 'cdn'
             f.write(fig1_2026.to_html(full_html=False, include_plotlyjs=include_js))
             f.write("<br><hr><br>")
             
        # Detailed Analysis Tables
        if fig2_2025:
             include_js = False if (fig1_2025 or fig1_2026) else 'cdn'
             f.write(fig2_2025.to_html(full_html=False, include_plotlyjs=include_js))
             f.write("<br>")
 
        if fig2_2026:
             include_js = False if (fig1_2025 or fig1_2026 or fig2_2025) else 'cdn'
             f.write(fig2_2026.to_html(full_html=False, include_plotlyjs=include_js))
             f.write("<br><hr><br>")

        # Indicator
        if fig3_2025:
             include_js = False if (fig1_2025 or fig1_2026 or fig2_2025 or fig2_2026) else 'cdn'
             f.write(fig3_2025.to_html(full_html=False, include_plotlyjs=include_js))
             f.write("<br>")
             
        if fig3_2026:
             include_js = False if (fig1_2025 or fig1_2026 or fig2_2025 or fig2_2026 or fig3_2025) else 'cdn'
             f.write(fig3_2026.to_html(full_html=False, include_plotlyjs=include_js))
             f.write("<br><hr><br>")

        # Comparison
        if fig4:
             include_js = False if (fig1_2025 or fig1_2026 or fig2_2025 or fig2_2026 or fig3_2025 or fig3_2026) else 'cdn'
             f.write(fig4.to_html(full_html=False, include_plotlyjs=include_js))
             f.write("<br><hr><br>")
             
        # Plot 5: LS6 Launch Trends
        if fig5:
             include_js = False if (fig1_2025 or fig1_2026 or fig2_2025 or fig2_2026 or fig3_2025 or fig3_2026 or fig4) else 'cdn'
             f.write(fig5.to_html(full_html=False, include_plotlyjs=include_js))
             f.write("<br><hr><br>")
             
        # Plot 6: REEV Price Trend
        if fig6:
             include_js = False if (fig1_2025 or fig1_2026 or fig2_2025 or fig2_2026 or fig3_2025 or fig3_2026 or fig4 or fig5) else 'cdn'
             f.write(fig6.to_html(full_html=False, include_plotlyjs=include_js))
             
        f.write("</body></html>")

if __name__ == "__main__":
    main()
