import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
import os
from pathlib import Path

file_path = '/Users/zihao_/Documents/coding/dataset/original/业务数据记录_竞争PK（LS6）.csv'

def parse_pk_count(count_str):
    if pd.isna(count_str):
        return 0
    if isinstance(count_str, (int, float)):
        return count_str
    return int(str(count_str).replace(',', ''))

def parse_week_info(week_str):
    # Normalize tilde
    week_str = week_str.replace('～', '~')
    parts = week_str.split('~')
    start_str = parts[0].strip()
    
    try:
        dt = datetime.strptime(start_str, '%Y-%m-%d')
        # Use ISO calendar to handle week numbers correctly
        iso_year, iso_week, _ = dt.isocalendar()
        return dt, iso_year, iso_week
    except ValueError:
        return None, None, None

def get_attribution_analysis(df, series_name='LS6'):
    """
    Generate attribution analysis for the specific series.
    Returns an HTML string with summary and tables.
    """
    # Filter data for the series
    sdf = df[df['series'] == series_name].copy()
    if sdf.empty:
        return ""
        
    # Get unique weeks and sort them
    unique_weeks = np.sort(sdf['Start_Date'].unique())
    
    if len(unique_weeks) < 8:
        return f"<p>数据不足，无法进行归因分析 (需至少8周数据，当前: {len(unique_weeks)}周)</p>"

    # Define periods: Recent 4 weeks vs Previous 5-8 weeks
    recent_weeks = unique_weeks[-4:]
    previous_weeks = unique_weeks[-8:-4]
    
    recent_str = f"{pd.to_datetime(recent_weeks[0]).strftime('%Y-%m-%d')} ~ {pd.to_datetime(recent_weeks[-1]).strftime('%Y-%m-%d')}"
    prev_str = f"{pd.to_datetime(previous_weeks[0]).strftime('%Y-%m-%d')} ~ {pd.to_datetime(previous_weeks[-1]).strftime('%Y-%m-%d')}"

    def calculate_metrics(weeks):
        data = sdf[sdf['Start_Date'].isin(weeks)]
        
        # Total PK per week average
        # Group by Start_Date to get weekly totals
        weekly_totals = data.groupby('Start_Date')['PK_Count'].sum()
        avg_total = weekly_totals.mean()
        
        # Top 3 concentration per week average
        weekly_stats = []
        competitor_stats = {} # Avg PK count per competitor
        
        for w in weeks:
            w_data = data[data['Start_Date'] == w]
            
            # Keep only Top 10 for comparable analysis
            w_sorted = w_data.sort_values('PK_Count', ascending=False).head(10)
            
            w_total = w_sorted['PK_Count'].sum()
            top3 = w_sorted.head(3)
            top3_sum = top3['PK_Count'].sum()
            
            conc = top3_sum / w_total if w_total > 0 else 0
            weekly_stats.append({
                'total': w_total,
                'top3_sum': top3_sum,
                'conc': conc
            })
            
            for _, row in w_sorted.iterrows():
                comp = row['车系']
                count = row['PK_Count']
                if comp not in competitor_stats:
                    competitor_stats[comp] = []
                competitor_stats[comp].append(count)
        
        avg_top3 = np.mean([x['top3_sum'] for x in weekly_stats])
        avg_conc = np.mean([x['conc'] for x in weekly_stats])
        
        # Average competitor counts
        avg_comp_counts = {k: np.mean(v) for k, v in competitor_stats.items()}
        
        return {
            'avg_total': avg_total,
            'avg_top3': avg_top3,
            'avg_conc': avg_conc,
            'avg_comp_counts': avg_comp_counts
        }

    stats_recent = calculate_metrics(recent_weeks)
    stats_prev = calculate_metrics(previous_weeks)
    
    # 1. Summary HTML
    summary_html = f"""
    <h4>归因分析摘要 (近期 vs 前期)</h4>
    <p><strong>近期周期:</strong> {recent_str} <br> <strong>对比周期:</strong> {prev_str}</p>
    <table class="table table-striped">
        <thead>
            <tr>
                <th>指标</th>
                <th>前5-8周均值</th>
                <th>近4周均值</th>
                <th>变化</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>平均周度总PK (Top10)</td>
                <td>{stats_prev['avg_total']:.1f}</td>
                <td>{stats_recent['avg_total']:.1f}</td>
                <td style="color: {'#E74C3C' if stats_recent['avg_total'] < stats_prev['avg_total'] else '#2ECC71'}">{stats_recent['avg_total'] - stats_prev['avg_total']:.1f}</td>
            </tr>
            <tr>
                <td>平均TOP3总和</td>
                <td>{stats_prev['avg_top3']:.1f}</td>
                <td>{stats_recent['avg_top3']:.1f}</td>
                <td style="color: {'#E74C3C' if stats_recent['avg_top3'] < stats_prev['avg_top3'] else '#2ECC71'}">{stats_recent['avg_top3'] - stats_prev['avg_top3']:.1f}</td>
            </tr>
            <tr>
                <td>平均集中度 (Top3%)</td>
                <td>{stats_prev['avg_conc']:.1%}</td>
                <td>{stats_recent['avg_conc']:.1%}</td>
                <td style="color: {'#E74C3C' if stats_recent['avg_conc'] < stats_prev['avg_conc'] else '#2ECC71'}">{(stats_recent['avg_conc'] - stats_prev['avg_conc'])*100:.1f} pct</td>
            </tr>
        </tbody>
    </table>
    """
    
    # 2. Detail Analysis
    all_comps = set(stats_recent['avg_comp_counts'].keys()) | set(stats_prev['avg_comp_counts'].keys())
    comp_changes = []
    
    for comp in all_comps:
        prev = stats_prev['avg_comp_counts'].get(comp, 0)
        curr = stats_recent['avg_comp_counts'].get(comp, 0)
        diff = curr - prev
        
        prev_share = prev / stats_prev['avg_total'] if stats_prev['avg_total'] > 0 else 0
        curr_share = curr / stats_recent['avg_total'] if stats_recent['avg_total'] > 0 else 0
        share_diff = curr_share - prev_share
        
        comp_changes.append({
            'competitor': comp,
            'prev': prev,
            'curr': curr,
            'diff': diff,
            'share_diff': share_diff
        })
        
    # Sort by Abs Drop (The 'Drop')
    comp_changes.sort(key=lambda x: x['diff'])
    top_drops = comp_changes[:5]
    
    # Sort by Share Gain (The 'Dispersers')
    comp_changes.sort(key=lambda x: x['share_diff'], reverse=True)
    top_gainers = comp_changes[:5]
    
    def generate_comp_table(items, title, is_share=False):
        rows = ""
        for item in items:
            val_fmt = "{:+.1%}".format(item['share_diff']) if is_share else "{:+.1f}".format(item['diff'])
            # 颜色逻辑：根据展示的指标（份额或绝对值）的正负来决定颜色
            check_val = item['share_diff'] if is_share else item['diff']
            color = "#2ECC71" if check_val > 0 else "#E74C3C"
            rows += f"""
            <tr>
                <td>{item['competitor']}</td>
                <td>{item['prev']:.1f}</td>
                <td>{item['curr']:.1f}</td>
                <td style="color: {color}">{val_fmt}</td>
            </tr>
            """
        return f"""
        <div style="flex: 1; min-width: 300px; margin-right: 20px;">
            <h5>{title}</h5>
            <table class="table">
                <thead>
                    <tr>
                        <th>竞品</th>
                        <th>前期均值</th>
                        <th>近期均值</th>
                        <th>{'份额变化' if is_share else '绝对值变化'}</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """

    details_html = f"""
    <div style="display: flex; flex-wrap: wrap;">
        {generate_comp_table(top_drops, "绝对值下降 TOP5 (拖累项)")}
        {generate_comp_table(top_gainers, "份额提升 TOP5 (分散项)", is_share=True)}
    </div>
    """
    
    return summary_html + details_html

def get_common_layout(title: str, xaxis_title: str = None, yaxis_title: str = None):
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
            showgrid=True
        ),
        legend=dict(
            x=0.99,
            y=0.99,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#ebedf0',
            borderwidth=1
        ),
        margin=dict(t=80, b=40, l=60, r=40)
    )
    return layout

def analyze_pk_trend():
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    
    # Preprocessing
    df['PK_Count'] = df['PK次数'].apply(parse_pk_count)
    
    # Parse Dates
    week_data = df['Week'].apply(parse_week_info)
    df['Start_Date'] = [x[0] for x in week_data]
    df['Year'] = [x[1] for x in week_data]
    df['Week_Num'] = [x[2] for x in week_data]
    
    # Drop rows where date parsing failed
    df = df.dropna(subset=['Start_Date'])
    
    # Prepare Comp Tables Data
    comp_year_summary = df.groupby(['series', 'Year', '车系'])['PK_Count'].sum().reset_index()
    comp_year_totals = df.groupby(['series', 'Year'])['PK_Count'].sum()

    series_list = ['LS6', 'L6', 'LS9']
    
    # Generate HTML content
    html_content = []
    # Header and Styles
    html_content.append("""<html>
<head>
<meta charset="utf-8" />
<script charset="utf-8" src="https://cdn.plot.ly/plotly-3.1.1.min.js"></script>
<style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; padding: 20px; color: #2c3e50; max-width: 1200px; margin: 0 auto; }
    
    /* 模仿 analyze_2025.py 的标题风格 */
    h1 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-bottom: 30px; }
    h2 { color: #34495e; margin-top: 40px; border-left: 5px solid #3498db; padding-left: 15px; font-size: 24px; }
    h3 { color: #2980b9; margin-top: 20px; margin-bottom: 10px; font-size: 18px; }
    
    /* 容器风格 */
    .series-section { margin-bottom: 60px; }
    .chart-wrapper { margin-bottom: 30px; border: 1px solid #ebedf0; border-radius: 8px; padding: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.02); background: #fff; }
    
    /* 表格风格 */
    .table-container { margin-top: 20px; overflow-x: auto; }
    .table { width: 100%; border-collapse: collapse; font-size: 14px; margin-bottom: 10px; }
    .table thead th { background: #f8f9fa; color: #555; font-weight: 600; padding: 12px; border-bottom: 2px solid #ddd; text-align: center; }
    .table tbody td { padding: 12px; border-top: 1px solid #eee; text-align: center; border-bottom: 1px solid #eee; }
    .table-striped tbody tr:nth-child(odd) { background: #fbfcfd; }
    .table-striped tbody tr:hover { background: #f5f5f5; }
    
    .timestamp { color: #888; font-size: 0.9em; margin-bottom: 30px; }
</style>
</head>
<body>
<h1>周度竞品PK分析报告 (2024 vs 2025 vs 2026)</h1>
<div class='timestamp'>生成时间: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</div>
""")

    for idx, series_name in enumerate(series_list):
        # 容器开始
        html_content.append(f"<div class='series-section'>")
        html_content.append(f"<h2>{series_name} 系列分析</h2>")
        
        # --- Data Preparation ---
        sdf = df[df['series'] == series_name].copy()
        
        if not sdf.empty:
            agg = sdf.groupby('Start_Date')[['PK_Count']].apply(lambda g: pd.Series({
                'Top3_Sum': g['PK_Count'].sort_values(ascending=False).head(3).sum(),
                'Total_Sum': g['PK_Count'].sum()
            })).reset_index()
            agg['Concentration'] = agg.apply(lambda r: (r['Top3_Sum'] / r['Total_Sum']) if (pd.notna(r['Total_Sum']) and r['Total_Sum'] > 0) else 0, axis=1)
            agg['Year'] = agg['Start_Date'].apply(lambda d: d.isocalendar()[0])
            agg['Week_Num'] = agg['Start_Date'].apply(lambda d: d.isocalendar()[1])
            agg_2024 = agg[agg['Year'] == 2024].sort_values('Week_Num')
            agg_2025 = agg[agg['Year'] == 2025].sort_values('Week_Num')
            agg_2026 = agg[agg['Year'] == 2026].sort_values('Week_Num')

            # --- Chart 1: Top3 Sum ---
            fig_sum = go.Figure()
            fig_sum.add_trace(go.Scatter(
                x=agg_2026['Week_Num'], y=agg_2026['Top3_Sum'],
                mode='lines+markers', name='2026',
                line=dict(color='#2ECC71', width=3), marker=dict(size=8)
            ))
            fig_sum.add_trace(go.Scatter(
                x=agg_2025['Week_Num'], y=agg_2025['Top3_Sum'],
                mode='lines+markers', name='2025',
                line=dict(color='#E67E22', width=3), marker=dict(size=8)
            ))
            fig_sum.add_trace(go.Scatter(
                x=agg_2024['Week_Num'], y=agg_2024['Top3_Sum'],
                mode='lines+markers', name='2024',
                line=dict(color='#3498DB', width=3), marker=dict(size=8)
            ))
            
            layout_sum = get_common_layout(
                title=f"{series_name} TOP3 PK次数总和趋势",
                xaxis_title='周数',
                yaxis_title='PK次数'
            )
            fig_sum.update_layout(layout_sum)
            
            # 设置高度为 450px，与 analyze_2025.py 保持一致（虽然 analyze_2025.py 没有显式设置 height，但 Plotly 默认或 CSS 控制通常在 450px 左右，这里显式设置以确保一致性）
            # 注意：analyze_2025.py 实际上使用的是 Plotly 默认高度 (450px)。
            # 为了完全对齐，我们在 HTML div 中设置高度，而不是在 layout 中写死（如果 analyze_2025.py 是这样做的）。
            # 检查发现 analyze_2025.py 使用 pio.to_html(..., full_html=False) 生成的 div 通常自带高度或自适应。
            # 但用户要求“设置每张图的高度...一样”，我们可以在 get_common_layout 中添加默认高度。
            
            html_content.append(f"<h3>{series_name} TOP3 PK次数总和趋势</h3>")
            html_content.append(f'<div class="chart-wrapper">{pio.to_html(fig_sum, full_html=False, include_plotlyjs=False, default_height="450px")}</div>')

            # --- Chart 2: Concentration ---
            fig_conc = go.Figure()
            fig_conc.add_trace(go.Scatter(
                x=agg_2026['Week_Num'], y=agg_2026['Concentration'],
                mode='lines+markers', name='2026',
                line=dict(color='#2ECC71', width=3, dash='dot'), marker=dict(size=8, symbol='diamond')
            ))
            fig_conc.add_trace(go.Scatter(
                x=agg_2025['Week_Num'], y=agg_2025['Concentration'],
                mode='lines+markers', name='2025',
                line=dict(color='#E67E22', width=3, dash='dot'), marker=dict(size=8, symbol='diamond')
            ))
            fig_conc.add_trace(go.Scatter(
                x=agg_2024['Week_Num'], y=agg_2024['Concentration'],
                mode='lines+markers', name='2024',
                line=dict(color='#3498DB', width=3, dash='dot'), marker=dict(size=8, symbol='diamond')
            ))
            
            layout_conc = get_common_layout(
                title=f"{series_name} TOP3 集中度趋势 (%)",
                xaxis_title='周数',
                yaxis_title='集中度 (%)'
            )
            layout_conc['yaxis']['tickformat'] = '.1%'
            fig_conc.update_layout(layout_conc)
            
            html_content.append(f"<h3>{series_name} TOP3 集中度趋势 (%)</h3>")
            html_content.append(f'<div class="chart-wrapper">{pio.to_html(fig_conc, full_html=False, include_plotlyjs=False, default_height="450px")}</div>')
            
            # --- Table: Competitor Comparison --- 3. Table Generation ---
        # Pre-calculate ranks for all years
        series_ranks = {}
        for yr in [2024, 2025, 2026]:
            df_yr = comp_year_summary[(comp_year_summary['series'] == series_name) & (comp_year_summary['Year'] == yr)]
            df_yr = df_yr.sort_values('PK_Count', ascending=False).reset_index(drop=True)
            series_ranks[yr] = {row['车系']: idx for idx, row in df_yr.iterrows()}

        names_data = {}
        for yr in [2024, 2025, 2026]:
            scomp = comp_year_summary[(comp_year_summary['series'] == series_name) & (comp_year_summary['Year'] == yr)]
            scomp = scomp.sort_values('PK_Count', ascending=False).head(10)
            
            try:
                total_pk = comp_year_totals.loc[(series_name, yr)]
            except KeyError:
                total_pk = 0
            
            current_names = []
            # scomp is already sorted by PK_Count desc, so idx corresponds to rank (0-based)
            # Reset index to ensure enumerate gives 0, 1, 2...
            scomp = scomp.reset_index(drop=True)
            
            for rank_idx, row in scomp.iterrows():
                series_name_item = row['车系']
                if total_pk > 0:
                    share = row['PK_Count'] / total_pk
                    text = f"{series_name_item}（{share:.0%}）"
                else:
                    text = f"{series_name_item}"
                
                # Determine color based on rank change vs previous year
                color = "black"
                prev_yr = yr - 1
                if prev_yr in series_ranks:
                    # If not in previous year's list, treat rank as very low (9999)
                    prev_rank = series_ranks[prev_yr].get(series_name_item, 9999)
                    current_rank = rank_idx
                    
                    if current_rank < prev_rank:
                        color = "#2ECC71" # Green (Improved)
                    elif current_rank > prev_rank:
                        color = "#E74C3C" # Red (Declined)
                
                if color != "black":
                    current_names.append(f'<span style="color: {color};">{text}</span>')
                else:
                    current_names.append(text)
            
            while len(current_names) < 10:
                current_names.append('')
            
            names_data[yr] = current_names
                
        ranking_labels = [f'第{i+1}' for i in range(10)]
        rows = []
        for i, label in enumerate(ranking_labels):
            rows.append({
                '排名': label,
                '2024': names_data[2024][i],
                '2025': names_data[2025][i],
                '2026': names_data[2026][i]
            })
        
        df_single = pd.DataFrame(rows, columns=['排名', '2024', '2025', '2026'])
        table_html = df_single.to_html(index=False, escape=False, classes='table table-striped', border=0)
        
        html_content.append(f"<h3>{series_name} TOP10 竞争对手对比</h3>")
        html_content.append(f"<div class='table-container'>{table_html}</div>")

        # --- Attribution Analysis ---
        attribution_html = get_attribution_analysis(df, series_name)
        if attribution_html:
            html_content.append(f"<h3>{series_name} 归因分析 (近期趋势)</h3>")
            html_content.append(f"<div class='chart-wrapper'>{attribution_html}</div>")
        
        # 容器结束
        html_content.append("</div>")

    html_content.append("</body></html>")
    
    project_root = Path(__file__).resolve().parents[1]
    output_file = project_root / "reports" / "pk_trend_analysis.html"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(html_content), encoding='utf-8')
    print(f"Report generated at: {output_file}")

if __name__ == "__main__":
    analyze_pk_trend()
