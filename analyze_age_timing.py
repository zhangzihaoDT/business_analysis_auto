import pandas as pd
from pathlib import Path
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import re

# Constants
PARQUET_FILE = Path("/Users/zihao_/Documents/coding/dataset/formatted/order_full_data.parquet")
BUSINESS_DEF_FILE = Path("/Users/zihao_/Documents/github/W52_reasoning/world/business_definition.json")
OUTPUT_HTML = Path("reports/age_timing_heatmap.html")

def load_business_definition(file_path: Path) -> dict:
    if not file_path.exists():
        # Try local path if absolute path fails
        local_path = Path("world/business_definition.json")
        if local_path.exists():
            return json.load(open(local_path, 'r', encoding='utf-8'))
        raise FileNotFoundError(f"Business definition file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_sql_condition(df: pd.DataFrame, condition_str: str, product_col: str) -> pd.Series:
    """
    Parses simple SQL-like conditions for product filtering.
    """
    if condition_str == "ELSE":
        return pd.Series([False] * len(df), index=df.index)

    # Replace "Product Name" variants with "TARGET_COL"
    s = re.sub(r"(?i)product\s+name", "TARGET_COL", condition_str)
    s = re.sub(r"(?i)product_name", "TARGET_COL", s)
    
    # NOT LIKE
    def not_like_replacer(match):
        val = match.group(1)
        return f"~df['{product_col}'].astype(str).str.contains('{val}', case=False, na=False)"
    s = re.sub(r"TARGET_COL\s+NOT\s+LIKE\s+'%([^%]+)%+'", not_like_replacer, s)

    # LIKE
    def like_replacer(match):
        val = match.group(1)
        return f"df['{product_col}'].astype(str).str.contains('{val}', case=False, na=False)"
    s = re.sub(r"TARGET_COL\s+LIKE\s+'%([^%]+)%+'", like_replacer, s)
    
    # AND / OR
    s = s.replace(" AND ", " & ").replace(" OR ", " | ")
    
    try:
        return eval(s)
    except Exception as e:
        print(f"Warning: Failed to evaluate condition '{condition_str}': {e}")
        return pd.Series([False] * len(df), index=df.index)

def map_age_group(age, current_year):
    """
    Based on logic from scripts/lock_summary.py
    """
    if pd.isna(age):
        return "未知"
    
    birth_year = current_year - age
    
    if birth_year >= 2000:
        return "00后"
    elif birth_year >= 1995:
        return "95后"
    elif birth_year >= 1990:
        return "90后"
    elif birth_year >= 1985:
        return "85后"
    elif birth_year >= 1980:
        return "80后"
    elif birth_year >= 1975:
        return "75后"
    elif birth_year >= 1970:
        return "70后"
    else:
        return "70前"

def main():
    if not PARQUET_FILE.exists():
        print(f"Error: {PARQUET_FILE} not found.")
        return

    print(f"Loading data from {PARQUET_FILE}...")
    df = pd.read_parquet(PARQUET_FILE)
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df['lock_time']):
        df['lock_time'] = pd.to_datetime(df['lock_time'], errors='coerce')
        
    # Apply Retained Order Filter: lock_time exists AND approve_refund_time is NULL
    print("Applying Retained Order Filter (approve_refund_time is NULL)...")
    initial_count = len(df)
    df = df[df['approve_refund_time'].isna()]
    print(f"Filtered out {initial_count - len(df)} refunded orders. Remaining: {len(df)}")
    
    # ---------------------------------------------------------
    # Apply Sub-model Classification & Time Period Filtering
    # ---------------------------------------------------------
    print("Loading Business Definitions...")
    biz_def = load_business_definition(BUSINESS_DEF_FILE)
    series_group_logic = biz_def.get('series_group_logic', {})
    time_periods = biz_def.get('time_periods', {})
    model_series_mapping = biz_def.get('model_series_mapping', {})
    
    # Determine product column
    product_col = 'product_name'
    if product_col not in df.columns:
        # Fallback if product_name not found
        for c in ['model_name', 'product_name_cn']:
            if c in df.columns:
                product_col = c
                break
    print(f"Using product column: {product_col}")

    # Classify Sub-models
    print("Classifying Sub-models...")
    df['sub_model'] = '其他'
    
    # Iterate logic. 
    for sub_model, condition in series_group_logic.items():
        if condition == "ELSE":
            continue
        mask = parse_sql_condition(df, condition, product_col)
        df.loc[mask, 'sub_model'] = sub_model
        
    # Filter by Time Periods (Exclude orders <= end_date)
    print("Filtering by Time Periods (Removing orders <= end_date)...")
    initial_filtered_count = len(df)
    
    keep_mask = pd.Series([True] * len(df), index=df.index)
    
    for sub_model, period in time_periods.items():
        if 'end' in period:
            # Parse end date
            end_date_str = period['end']
            # We want to exclude orders BEFORE or ON the end date.
            # Usually 'end' is inclusive day.
            # So we exclude lock_time < (end_date + 1 day)
            end_date_ts = pd.Timestamp(end_date_str) + pd.Timedelta(days=1)
            
            # Mask for this sub_model
            sub_mask = df['sub_model'] == sub_model
            
            # Identify rows to DROP: sub_model match AND lock_time < end_date_ts
            drop_rows = sub_mask & (df['lock_time'] < end_date_ts)
            
            keep_mask = keep_mask & (~drop_rows)
            
            count_dropped = drop_rows.sum()
            if count_dropped > 0:
                print(f"  {sub_model}: Dropping {count_dropped} orders locked before {end_date_ts}")

    df = df[keep_mask]
    print(f"Filtered out {initial_filtered_count - len(df)} orders based on time periods. Remaining: {len(df)}")
    
    # Map Sub-models to Series (LS6, L6, LS9)
    print("Mapping Sub-models to Series...")
    
    sub_to_series = {}
    for series, subs in model_series_mapping.items():
        for sub in subs:
            sub_to_series[sub] = series
            
    def get_series(row):
        sub = row['sub_model']
        if sub in sub_to_series:
            return sub_to_series[sub]
        # Fallback for known series names if they match sub_model or are unmapped
        if sub in ["LS9", "LS7", "L7"]:
            return sub
        return "Other"

    df['series'] = df.apply(get_series, axis=1)
    
    # Define periods
    # 2024: 2024-01-01 ~ 2024-12-31
    mask_2024 = (df['lock_time'] >= '2024-01-01') & (df['lock_time'] <= '2024-12-31')
    
    # 2025: >= 2025-01-01
    mask_2025 = (df['lock_time'] >= '2025-01-01')
    
    df_2024 = df[mask_2024].copy()
    df_2025 = df[mask_2025].copy()
    
    print(f"2024 Total Records: {len(df_2024)}")
    print(f"2025 Total Records: {len(df_2025)}")
    
    # Apply age mapping
    print("Classifying age groups...")
    df_2024['age_group'] = df_2024['age'].apply(lambda x: map_age_group(x, 2024))
    df_2025['age_group'] = df_2025['age'].apply(lambda x: map_age_group(x, 2025))
    
    # Pre-calculate pivot tables for all (year, series) combinations to determine ranges
    series_list = ["LS6", "L6", "LS9"]
    age_order = ["00后", "95后", "90后", "85后", "80后", "75后", "70后", "70前"]
    months = list(range(1, 13))
    
    pivots_store = {} # Key: (year, series_name) -> pivot_price_wan
    pivots_count_store = {} # Key: (year, series_name) -> pivot_count

    for year, sub_df in [(2024, df_2024), (2025, df_2025)]:
        # Ensure numeric
        if 'invoice_amount' in sub_df.columns:
            sub_df['invoice_amount_numeric'] = pd.to_numeric(sub_df['invoice_amount'], errors='coerce')
        else:
            sub_df['invoice_amount_numeric'] = pd.Series([float('nan')] * len(sub_df))
            
        sub_df['month'] = sub_df['lock_time'].dt.month
        
        for series_name in series_list:
            if 'series' not in sub_df.columns:
                series_df = pd.DataFrame()
            else:
                series_df = sub_df[sub_df['series'] == series_name]
                
            if series_df.empty:
                pivot_price_wan = pd.DataFrame(index=age_order, columns=months)
                pivot_count = pd.DataFrame(index=age_order, columns=months, data=0)
            else:
                pivot_price = series_df.groupby(['age_group', 'month'])['invoice_amount_numeric'].mean().unstack()
                pivot_price = pivot_price.reindex(index=age_order, columns=months)
                pivot_price_wan = pivot_price / 10000
                
                pivot_count = series_df.groupby(['age_group', 'month']).size().unstack(fill_value=0)
                pivot_count = pivot_count.reindex(index=age_order, columns=months, fill_value=0)
            
            pivots_store[(year, series_name)] = pivot_price_wan
            pivots_count_store[(year, series_name)] = pivot_count

    # Determine Min/Max per Series
    series_ranges = {}
    for s in series_list:
        vals = []
        for y in [2024, 2025]:
            v = pivots_store[(y, s)].values.flatten()
            vals.extend(v[~pd.isna(v)])
        
        if vals:
            series_ranges[s] = (min(vals), max(vals))
        else:
            series_ranges[s] = (0, 1) # Default

    titles = ["2024 销量占比", "2025 销量占比"]
    for s in series_list:
        titles.extend([f"2024 {s} 均价", f"2025 {s} 均价"])

    fig = make_subplots(
        rows=4, cols=2, 
        subplot_titles=titles,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        vertical_spacing=0.08
    )

    for i, (year, sub_df) in enumerate([(2024, df_2024), (2025, df_2025)]):
        # --- Part 1: Timing Distribution ---
        sub_df['month'] = sub_df['lock_time'].dt.month
        pivot = sub_df.groupby(['age_group', 'month']).size().unstack(fill_value=0)
        pivot = pivot.reindex(index=age_order, columns=months, fill_value=0)
        col_sums = pivot.sum(axis=0)
        pivot_pct = pivot.div(col_sums, axis=1).replace(float('nan'), 0) * 100
        
        text_matrix = []
        label_matrix = []
        for age in age_order:
            row_text = []
            row_labels = []
            for m in months:
                count = pivot.loc[age, m]
                pct = pivot_pct.loc[age, m]
                row_text.append(f"年份: {year}<br>年龄段: {age}<br>月份: {m}月<br>留存锁单数: {count}<br>月内占比: {pct:.1f}%")
                # Add labels for age "00后" OR month 12
                if age == "00后" or m == 12:
                    row_labels.append(f"{pct:.0f}%")
                else:
                    row_labels.append("")
            text_matrix.append(row_text)
            label_matrix.append(row_labels)
            
        colorscale_timing = [
            [0.0, 'lightgrey'], [0.125, 'white'], [1.0, '#3498DB']
        ]
            
        fig.add_trace(
            go.Heatmap(
                z=pivot_pct.values,
                x=[f"{m}月" for m in months],
                y=age_order,
                hovertext=text_matrix,
                text=label_matrix,
                texttemplate="%{text}",
                hoverinfo="text",
                coloraxis="coloraxis",
                zmin=0, zmax=40,
            ),
            row=1, col=i+1
        )
        
        # --- Part 2: Price by Series ---
        for s_idx, series_name in enumerate(series_list):
            pivot_price_wan = pivots_store[(year, series_name)]
            pivot_count = pivots_count_store[(year, series_name)]
            
            # Prepare tooltip and labels
            text_matrix_price = []
            label_matrix_price = []
            
            # Get Anchor Value (00后, 12月)
            try:
                anchor_val = pivot_price_wan.loc["00后", 12]
            except KeyError:
                anchor_val = float('nan')
                
            for age in age_order:
                row_text = []
                row_labels = []
                for m in months:
                    val = pivot_price_wan.loc[age, m]
                    count = pivot_count.loc[age, m]
                    if pd.isna(val):
                        row_text.append(f"年份: {year}<br>车系: {series_name}<br>年龄段: {age}<br>月份: {m}月<br>留存锁单数: {count}<br>无开票数据")
                        row_labels.append("")
                    else:
                        row_text.append(f"年份: {year}<br>车系: {series_name}<br>年龄段: {age}<br>月份: {m}月<br>留存锁单数: {count}<br>开票均价: {val:.2f}万元")
                        # Add labels for age "00后" OR month 12
                        if age == "00后" and m == 12:
                            # Anchor itself: Absolute value
                            row_labels.append(f"{val:.1f}")
                        elif age == "00后" or m == 12:
                            # Other target cells: Difference from anchor
                            if pd.notna(anchor_val):
                                diff = val - anchor_val
                                row_labels.append(f"{diff:+.1f}")
                            else:
                                # Fallback if anchor is missing: Absolute value
                                row_labels.append(f"{val:.1f}")
                        else:
                            row_labels.append("")
                text_matrix_price.append(row_text)
                label_matrix_price.append(row_labels)

            c_axis = f"coloraxis{s_idx+2}"

            fig.add_trace(
                go.Heatmap(
                    z=pivot_price_wan.values,
                    x=[f"{m}月" for m in months],
                    y=age_order,
                    hovertext=text_matrix_price,
                    text=label_matrix_price,
                    texttemplate="%{text}",
                    hoverinfo="text",
                    coloraxis=c_axis,
                ),
                row=2+s_idx, col=i+1
            )

    # Update Layout with Multiple Color Axes
    layout_update = {
        "title_text": "2024 vs 2025 年龄段购车洞察：时间偏好(Row 1) 与 各车系开票均价(Rows 2-4)",
        "height": 1800,
        "width": 1300, 
        "template": "plotly_white",
        "coloraxis": dict(
            colorscale=colorscale_timing,
            cmin=0, cmax=40,
            colorbar=dict(title="销量占比 (%)", orientation='v', len=0.2, y=0.905, x=1.02)
        )
    }
    
    # Custom Orange Scale
    custom_oranges = [[0, 'white'], [1, '#E67E22']]

    # LS6 (Row 2)
    s_min, s_max = series_ranges[series_list[0]]
    layout_update["coloraxis2"] = dict(
        colorscale=custom_oranges,
        cmin=s_min, cmax=s_max,
        colorbar=dict(title=f"{series_list[0]} 均价", len=0.2, y=0.635, x=1.02)
    )
    
    # L6 (Row 3)
    s_min, s_max = series_ranges[series_list[1]]
    layout_update["coloraxis3"] = dict(
        colorscale=custom_oranges,
        cmin=s_min, cmax=s_max,
        colorbar=dict(title=f"{series_list[1]} 均价", len=0.2, y=0.365, x=1.02)
    )
    
    # LS9 (Row 4)
    s_min, s_max = series_ranges[series_list[2]]
    layout_update["coloraxis4"] = dict(
        colorscale=custom_oranges,
        cmin=s_min, cmax=s_max,
        colorbar=dict(title=f"{series_list[2]} 均价", len=0.2, y=0.095, x=1.02)
    )
    
    fig.update_layout(**layout_update)


    
    # Ensure output directory exists
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUTPUT_HTML)
    print(f"Heatmap saved to {OUTPUT_HTML}")

    # ---------------------------------------------------------
    # Text Analysis for Console
    # ---------------------------------------------------------
    target_groups = ["95后", "00后"]
    
    for group in target_groups:
        print(f"\n{'='*30}")
        print(f"Analyzing Group: {group}")
        print(f"{'='*30}")
        
        for year, sub_df in [(2024, df_2024), (2025, df_2025)]:
            group_data = sub_df[sub_df['age_group'] == group].copy()
            count = len(group_data)
            
            print(f"\n--- {year} ---")
            print(f"Total Orders: {count}")
            
            if count == 0:
                print("  No data.")
                continue
                
            # Extract month
            group_data['month'] = group_data['lock_time'].dt.month
            
            # Count by month
            monthly_counts = group_data['month'].value_counts().sort_index()
            
            # Calculate percentage
            monthly_pct = (monthly_counts / count * 100).round(2)
            
            print(f"Monthly Distribution:")
            print(f"  {'Month':<5} | {'Count':<6} | {'Pct(%)':<6}")
            print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*8}")
            
            for m in monthly_counts.index:
                print(f"  {m:<5} | {monthly_counts[m]:<6} | {monthly_pct[m]:<6}")
            
            # Find peak month
            if not monthly_counts.empty:
                peak_month = monthly_counts.idxmax()
                print(f"\n  Peak Month: {peak_month} (Count: {monthly_counts[peak_month]}, {monthly_pct[peak_month]}%)")
                
                # Check top 3 months
                top3 = monthly_counts.sort_values(ascending=False).head(3)
                top3_str = ", ".join([f"{m}月({c}单)" for m, c in top3.items()])
                print(f"  Top 3 Months: {top3_str}")

if __name__ == "__main__":
    main()
