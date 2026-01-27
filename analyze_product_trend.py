import pandas as pd
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
import statsmodels.api as sm
import numpy as np

# Paths
PARQUET_FILE = Path("/Users/zihao_/Documents/coding/dataset/formatted/order_full_data.parquet")
BUSINESS_DEF_FILE = Path("/Users/zihao_/Documents/github/W52_reasoning/world/business_definition.json")
REPORT_DIR = Path("/Users/zihao_/Documents/coding/dataset/scripts/reports")
REPORT_DIR.mkdir(exist_ok=True, parents=True)

OUTPUT_HTML = REPORT_DIR / "analyze_product_trend.html"
OUTPUT_CSV = REPORT_DIR / "analyze_product_trend.csv"

# Colors
COLORS = {
    "新一代智己LS6 Max+": "#3498DB",  # Blue
    "新一代智己LS6 Max": "#E67E22",   # Orange
    "新一代智己LS6 52 Max": "#3498DB", # Blue
    "新一代智己LS6 66 Max": "#E67E22", # Orange
    "其他": "#E67E22",              # Orange
    "52kWh": "#3498DB",             # Blue
    "66kWh": "#E67E22",             # Orange
    "Other": "#95A5A6"              # Gray
}

def load_business_definition(file_path: Path) -> dict:
    """加载业务定义文件"""
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_data(file_path: Path) -> pd.DataFrame:
    """加载 Parquet 数据"""
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    print(f"📖 Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"✅ Loaded {len(df)} rows.")
    return df

def get_common_layout(title: str, xaxis_title: str = None, yaxis_title: str = None, yaxis_format: str = None):
    """获取统一的 Plotly Layout 配置"""
    layout = dict(
        title=title,
        template="plotly_white",
        plot_bgcolor='#FFFFFF',
        hovermode="x unified",
        xaxis=dict(
            title=xaxis_title,
            gridcolor='#ebedf0',
            zerolinecolor='#ebedf0',
            tickfont=dict(color='#7B848F'),
            title_font=dict(color='#7B848F'),
            showgrid=True
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor='#ebedf0',
            zerolinecolor='#ebedf0',
            tickfont=dict(color='#7B848F'),
            title_font=dict(color='#7B848F'),
            showgrid=True,
            tickformat=yaxis_format
        ),
        legend=dict(
            bordercolor='#7B848F',
            font=dict(color='#7B848F')
        )
    )
    return layout

def main():
    # 1. Load Configuration
    business_def = load_business_definition(BUSINESS_DEF_FILE)
    
    # Get CM2 End Date
    cm2_info = business_def.get('time_periods', {}).get('CM2')
    if not cm2_info:
        raise ValueError("CM2 definition not found in business_definition.json")
    
    cm2_end_str = cm2_info['end']
    cm2_end_date = pd.Timestamp(cm2_end_str)
    print(f"Start Date (CM2 End Date): {cm2_end_date}")

    # Get LS9 End Date
    ls9_info = business_def.get('time_periods', {}).get('LS9')
    if not ls9_info:
        print("Warning: LS9 definition not found in business_definition.json")
        ls9_end_date = cm2_end_date # Fallback
    else:
        ls9_end_str = ls9_info['end']
        ls9_end_date = pd.Timestamp(ls9_end_str)
        print(f"LS9 Start Date (LS9 End Date): {ls9_end_date}")

    # 2. Load Data
    df = load_data(PARQUET_FILE)
    
    # Ensure lock_time is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['lock_time']):
        df['lock_time'] = pd.to_datetime(df['lock_time'], errors='coerce')

    # 3. Filter Data
    # Time Filter: Since CM2 end day
    time_mask = df['lock_time'] >= cm2_end_date
    # Time Filter for LS9
    ls9_time_mask = df['lock_time'] >= ls9_end_date
    
    # --- Analysis 1: Pure Electric 76kWh ---
    print("\n--- Processing Analysis 1: Pure Electric 76kWh ---")
    
    # Product Filter
    # target_products_pe = business_def.get('battery_capacity', {}).get('76kwh', [])
    target_products_pe = ["新一代智己LS6 Max+", "新一代智己LS6 Max"]
    
    # Grouping Logic
    def group_pe(name):
        return name

    figs1, df_csv1 = run_analysis(
        df, 
        time_mask & df['product_name'].isin(target_products_pe), 
        group_pe, 
        "CM2 纯电 76kWh (Max+ vs Max)", 
        "PE_76kWh"
    )

    # --- Analysis 2: EREV 52/66kWh (CM2 Only) ---
    print("\n--- Processing Analysis 2: EREV 52/66kWh (CM2) ---")
    
    # Product Filter
    products_52 = business_def.get('battery_capacity', {}).get('52kwh', [])
    products_66 = business_def.get('battery_capacity', {}).get('66kwh', [])
    target_products_erev_all = products_52 + products_66
    
    # Filter for CM2 only ("新一代" and "LS6")
    # target_products_cm2_erev = [p for p in target_products_erev_all if "新一代" in p and "LS6" in p]
    target_products_cm2_erev = ["新一代智己LS6 52 Max", "新一代智己LS6 66 Max"]
    
    if not target_products_cm2_erev:
         print("Warning: No CM2 EREV products found.")

    # Grouping Logic
    def group_erev_cm2(name):
        return name
        
    # Keep original group_erev for Analysis 3
    def group_erev(name):
        if '52' in name:
            return '52kWh'
        elif '66' in name:
            return '66kWh'
        else:
            return 'Other'

    figs2, df_csv2 = run_analysis(
        df, 
        time_mask & df['product_name'].isin(target_products_cm2_erev), 
        group_erev_cm2, 
        "CM2 增程 (52 Max vs 66 Max)", 
        "EREV_CM2"
    )

    # --- Analysis 3: LS9 EREV 52/66kWh ---
    print("\n--- Processing Analysis 3: LS9 EREV 52/66kWh ---")
    
    # Filter for LS9 only ("LS9")
    target_products_ls9 = [p for p in target_products_erev_all if "LS9" in p]
    
    if not target_products_ls9:
        print("Warning: No LS9 products found in 52/66kwh lists.")
        
    figs3, df_csv3 = run_analysis(
        df,
        ls9_time_mask & df['product_name'].isin(target_products_ls9),
        group_erev, # Reuse same grouping logic (52 vs 66)
        "LS9 增程 52/66kWh",
        "LS9"
    )

    # 11. Save Outputs
    
    # Merge CSVs
    combined_csv = pd.concat([df_csv1, df_csv2, df_csv3], axis=1)
    combined_csv.to_csv(OUTPUT_CSV)
    print(f"✅ Saved Combined CSV to {OUTPUT_CSV}")
    
    # Generate HTML
    generate_html(OUTPUT_HTML, OUTPUT_CSV.name, cm2_end_date, [figs1, figs2, figs3])

def run_analysis(df, mask, group_func, title_prefix, col_prefix):
    filtered_df = df[mask].copy()
    print(f"✅ Filtered {len(filtered_df)} rows for {title_prefix}")
    
    filtered_df['group'] = filtered_df['product_name'].apply(group_func)
    
    # Aggregation: Daily Lock Count
    filtered_df['date'] = filtered_df['lock_time'].dt.floor('D')
    daily_counts = filtered_df.groupby(['date', 'group']).size().reset_index(name='count')
    pivot_df = daily_counts.pivot(index='date', columns='group', values='count').fillna(0).sort_index()
    
    # Aggregation: Retained
    if 'approve_refund_time' not in filtered_df.columns:
        filtered_df['approve_refund_time'] = pd.NaT
    if not pd.api.types.is_datetime64_any_dtype(filtered_df['approve_refund_time']):
        filtered_df['approve_refund_time'] = pd.to_datetime(filtered_df['approve_refund_time'], errors='coerce')
        
    retained_mask = filtered_df['approve_refund_time'].isna()
    retained_df = filtered_df[retained_mask].copy()
    daily_retained = retained_df.groupby(['date', 'group']).size().reset_index(name='count')
    pivot_retained = daily_retained.pivot(index='date', columns='group', values='count').fillna(0).reindex(pivot_df.index, fill_value=0)
    
    # Share
    pivot_df['total'] = pivot_df.sum(axis=1)
    total_series = pivot_df['total'].copy() # Keep total series for anomaly detection
    share_df = pivot_df.copy()
    for col in pivot_df.columns:
        if col != 'total':
            share_df[col] = pivot_df[col] / pivot_df['total']
    share_df = share_df.drop(columns=['total'])
    pivot_df = pivot_df.drop(columns=['total'])
    
    # CSV Prep
    pivot_export = pivot_df.add_prefix(f'{col_prefix}_Total_')
    retained_export = pivot_retained.add_prefix(f'{col_prefix}_Retained_')
    share_export = share_df.add_prefix(f'{col_prefix}_Share_')
    df_csv = pd.concat([pivot_export, retained_export, share_export], axis=1)
    
    # Charts
    # Chart 1: Volume
    fig1 = go.Figure()
    for col in pivot_df.columns:
        fig1.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df[col], mode='lines+markers', name=col, line=dict(color=COLORS.get(col, '#333333'))))
    fig1.update_layout(get_common_layout(f"{title_prefix} 锁单趋势 (Total)", "日期", "日锁单量"))
    
    # Chart 2: Cancelled (Refunded) Volume
    # Calculation: Total Locks - Retained Locks
    fig2 = go.Figure()
    
    # Ensure indices match for subtraction
    # pivot_retained is already reindexed to match pivot_df in run_analysis
    pivot_cancelled = pivot_df - pivot_retained
    
    for col in pivot_cancelled.columns:
        fig2.add_trace(go.Scatter(
            x=pivot_cancelled.index, 
            y=pivot_cancelled[col], 
            mode='lines+markers', 
            name=col,
            line=dict(color=COLORS.get(col, '#333333'))
        ))
    fig2.update_layout(get_common_layout(f"{title_prefix} 锁单退订趋势 (Cancelled)", "日期", "日退订锁单量"))
    
    # Chart 3: Share
    fig3 = go.Figure()
    for col in share_df.columns:
        fig3.add_trace(go.Scatter(x=share_df.index, y=share_df[col], mode='lines+markers', name=f"{col} (实际)", line=dict(color=COLORS.get(col, '#333333'), width=1, dash='dot'), opacity=0.5))
        # Lowess
        series = share_df[col].dropna()
        if len(series) > 5:
            y, x = series.values, series.index
            x_num = x.map(pd.Timestamp.timestamp).to_numpy()
            lowess = sm.nonparametric.lowess(y, x_num, frac=0.3)
            fig3.add_trace(go.Scatter(x=series.index, y=lowess[:, 1], mode='lines', name=f"{col} (Trend)", line=dict(color=COLORS.get(col, '#333333'), width=3)))
            
        # Anomaly Detection: Fake Anomaly (Scale Effect)
        # Logic: Share changed significantly, Total changed significantly, but Group Volume did NOT change significantly.
        # This implies the share change is driven by the denominator (Total) rather than the numerator (Group).
        
        # Calculate dynamic threshold based on rolling std dev (7 days)
        # We use a minimum threshold of 0.1 to avoid noise when variance is extremely low
        
        delta_share = share_df[col].pct_change()
        delta_vol = pivot_df[col].pct_change()
        delta_total = total_series.pct_change()
        
        # Calculate rolling volatility (std of pct_change) for each series
        # We look at the past 7 days to establish "normal" volatility
        vol_share = delta_share.rolling(window=7, min_periods=3).std().fillna(0.1)
        vol_total = delta_total.rolling(window=7, min_periods=3).std().fillna(0.1)
        vol_group = delta_vol.rolling(window=7, min_periods=3).std().fillna(0.1)
        
        # Dynamic Thresholds: 2 * Sigma (Standard Deviations)
        # If current change is > 2 * recent volatility, it's significant.
        # We clamp the threshold between 0.1 and 0.5 to prevent it from being too sensitive or too insensitive
        # UPDATE: Relaxed constraint to 1.0 sigma to catch more potential anomalies
        thresh_share = (1.0 * vol_share).clip(lower=0.1, upper=0.5)
        thresh_total = (1.0 * vol_total).clip(lower=0.1, upper=0.5)
        thresh_group = (1.0 * vol_group).clip(lower=0.1, upper=0.5)
        
        # Align indices
        valid_indices = delta_share.dropna().index.intersection(delta_vol.dropna().index).intersection(delta_total.dropna().index)
        
        # Check conditions
        anomaly_mask = (
            (delta_share.loc[valid_indices].abs() >= thresh_share.loc[valid_indices]) & 
            (delta_total.loc[valid_indices].abs() >= thresh_total.loc[valid_indices]) & 
            (delta_vol.loc[valid_indices].abs() < thresh_group.loc[valid_indices])
        )
        
        anomaly_dates = valid_indices[anomaly_mask]
        if not anomaly_dates.empty:
            anomaly_values = share_df.loc[anomaly_dates, col]
            
            # Create hover text with details
            hover_text = []
            for date in anomaly_dates:
                val_share = delta_share.loc[date]
                thr_share = thresh_share.loc[date]
                val_total = delta_total.loc[date]
                thr_total = thresh_total.loc[date]
                val_group = delta_vol.loc[date]
                thr_group = thresh_group.loc[date]
                
                txt = (
                    f"<b>Scale Effect Anomaly</b><br>"
                    f"Date: {date.strftime('%Y-%m-%d')}<br>"
                    f"Share: {anomaly_values[date]:.1%}<br>"
                    f"Reason: Total Vol Shift<br>"
                    f"<br><b>Details (Val vs 1σ):</b><br>"
                    f"ΔShare: {val_share:.1%} (Thresh: {thr_share:.1%})<br>"
                    f"ΔTotal: {val_total:.1%} (Thresh: {thr_total:.1%})<br>"
                    f"ΔGroup: {val_group:.1%} (Thresh: {thr_group:.1%})"
                )
                hover_text.append(txt)

            fig3.add_trace(go.Scatter(
                x=anomaly_dates, 
                y=anomaly_values, 
                mode='markers', 
                name=f"{col} (Scale Effect)", 
                marker=dict(symbol='x', size=10, color='red'),
                hoverinfo='text',
                hovertext=hover_text
            ))

    fig3.update_layout(get_common_layout(f"{title_prefix} 占比趋势 (Share)", "日期", "占比", ".1%"))
    
    return [fig1, fig2, fig3], df_csv

def generate_html(output_path, csv_name, start_date, fig_groups):
    css = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; color: #333; }
        h1, h2 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        .download-link { margin: 20px 0; }
        .download-link a { background-color: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }
        .download-link a:hover { background-color: #2980b9; }
        .chart-container { margin-bottom: 40px; padding: 20px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 5px; }
        .section { margin-bottom: 60px; }
    </style>
    """
    
    sections_html = ""
    titles = ["1. CM2 纯电 76kWh (Max+ vs Max)", "2. CM2 增程 (52 Max vs 66 Max)", "3. LS9 增程 52/66kWh"]
    
    for title, figs in zip(titles, fig_groups):
        charts_html = "".join([f'<div class="chart-container">{pio.to_html(f, full_html=False, include_plotlyjs="cdn")}</div>' for f in figs])
        sections_html += f'<div class="section"><h2>{title}</h2>{charts_html}</div>'

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset='utf-8'><title>Product Trend Analysis</title>{css}</head>
    <body>
        <h1>CM2 车型趋势分析报告</h1>
        <div>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        <div class="download-link"><a href="{csv_name}" download>📥 下载完整数据 (CSV)</a></div>
        
        <div style="background-color: #fdf2f2; border-left: 5px solid #e74c3c; padding: 15px; margin: 20px 0; border-radius: 4px;">
            <h3 style="margin-top: 0; color: #c0392b; font-size: 16px;">ℹ️ 关于图表中“红色 x”标记的说明</h3>
            <p style="margin: 5px 0; color: #555; line-height: 1.5;">
                在 <strong>占比趋势 (Share)</strong> 图表中，<strong style="color: #c0392b;">红色 'x' 标记</strong> 表示 <strong>“比例假异常 (Scale Effect)”</strong>。
            </p>
            <ul style="margin: 5px 0; padding-left: 20px; color: #555;">
                <li><strong>含义：</strong>该车型的<strong>自身销量</strong>（分子）并未发生剧烈变化，但由于<strong>整体大盘</strong>（分母）发生了剧烈波动，导致其<strong>占比</strong>被动出现了大幅变化。</li>
                <li><strong>通俗理解：</strong>“我销量没变，是世界（大盘）变了，导致我的份额被动变了。”</li>
                <li><strong>判定逻辑：</strong>基于 <strong>7天滚动标准差 (1σ)</strong> 的动态阈值：
                    <ul>
                        <li>占比变化 > 动态阈值 (1σ, min 10%, max 50%)</li>
                        <li>且 总量变化 > 动态阈值 (1σ, min 10%, max 50%)</li>
                        <li>但 自身销量变化 < 动态阈值 (1σ, min 10%, max 50%)</li>
                    </ul>
                </li>
            </ul>
        </div>

        {sections_html}
    </body>
    </html>
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✅ Generated Report: {output_path}")


if __name__ == "__main__":
    main()
