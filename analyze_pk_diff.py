import pandas as pd
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
<h1>周度竞品PK分析报告 (2024 vs 2025)</h1>
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

            # --- Chart 1: Top3 Sum ---
            fig_sum = go.Figure()
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
        names_2024 = []
        names_2025 = []
        for yr in [2024, 2025]:
            scomp = comp_year_summary[(comp_year_summary['series'] == series_name) & (comp_year_summary['Year'] == yr)]
            scomp = scomp.sort_values('PK_Count', ascending=False).head(3)
            try:
                total_pk = comp_year_totals.loc[(series_name, yr)]
            except KeyError:
                total_pk = 0
            
            def fmt_name(r):
                if total_pk and total_pk > 0:
                    share = r['PK_Count'] / total_pk
                    return f"{r['车系']}（{share:.0%}）"
                else:
                    return f"{r['车系']}"
            
            names = [fmt_name(r) for _, r in scomp.iterrows()]
            while len(names) < 3:
                names.append('')
            
            if yr == 2024:
                names_2024 = names
            else:
                names_2025 = names
                
        ranking_labels = ['第一', '第二', '第三']
        rows = []
        for i, label in enumerate(ranking_labels):
            rows.append({
                '排名': label,
                '2024': names_2024[i] if i < len(names_2024) else '',
                '2025': names_2025[i] if i < len(names_2025) else ''
            })
        
        df_single = pd.DataFrame(rows, columns=['排名', '2024', '2025'])
        table_html = df_single.to_html(index=False, escape=False, classes='table table-striped', border=0)
        
        html_content.append(f"<h3>{series_name} TOP3 竞争对手对比</h3>")
        html_content.append(f"<div class='table-container'>{table_html}</div>")
        
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
