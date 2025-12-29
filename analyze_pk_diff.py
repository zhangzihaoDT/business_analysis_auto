import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    
    # Calculate Weekly Top 3 Sum
    # Group by Year, Week_Num (and Start_Date to keep it)
    # We need to process each week separately
    
    results = []
    
    # Group by unique week identifier (Start_Date)
    for start_date, group in df.groupby('Start_Date'):
        # Get Year and Week from the first row of the group
        year = group['Year'].iloc[0]
        week_num = group['Week_Num'].iloc[0]
        
        # Sort by PK Count descending and take top 3
        sorted_group = group.sort_values('PK_Count', ascending=False)
        top3_sum = sorted_group.head(3)['PK_Count'].sum()
        total_sum = sorted_group['PK_Count'].sum()
        
        # Calculate Concentration (Share)
        # Avoid division by zero
        concentration = (top3_sum / total_sum) if total_sum > 0 else 0
        
        results.append({
            'Start_Date': start_date,
            'Year': year,
            'Week_Num': week_num,
            'Top3_Sum': top3_sum,
            'Concentration': concentration
        })
        
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(['Year', 'Week_Num'])
    
    # Separate into 2024 and 2025
    df_2024 = res_df[res_df['Year'] == 2024].copy()
    df_2025 = res_df[res_df['Year'] == 2025].copy()
    
    print("2024 Data Points:", len(df_2024))
    print("2025 Data Points:", len(df_2025))
    
    # 指定需要绘制的车系
    series_list = ['LS6', 'L6', 'LS9']
    total_rows = 2 * len(series_list)
    subplot_titles = []
    for s in series_list:
        subplot_titles.extend([f'{s} TOP3 PK次数总和', f'{s} TOP3 集中度 (%)'])
    # 创建多子图
    fig = make_subplots(
        rows=total_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=tuple(subplot_titles)
    )
    
    current_row = 1

    # --- 每个车系：次数与占比 ---
    for series_name in series_list:
        sdf = df[df['series'] == series_name].copy()
        if sdf.empty:
            current_row += 2
            continue
        agg = sdf.groupby('Start_Date').apply(lambda g: pd.Series({
            'Top3_Sum': g['PK_Count'].sort_values(ascending=False).head(3).sum(),
            'Total_Sum': g['PK_Count'].sum()
        })).reset_index()
        agg['Concentration'] = agg.apply(lambda r: (r['Top3_Sum'] / r['Total_Sum']) if (pd.notna(r['Total_Sum']) and r['Total_Sum'] > 0) else 0, axis=1)
        agg['Year'] = agg['Start_Date'].apply(lambda d: d.isocalendar()[0])
        agg['Week_Num'] = agg['Start_Date'].apply(lambda d: d.isocalendar()[1])
        agg_2024 = agg[agg['Year'] == 2024].sort_values('Week_Num')
        agg_2025 = agg[agg['Year'] == 2025].sort_values('Week_Num')

        # 次数（TOP3 总和）
        fig.add_trace(go.Scatter(
            x=agg_2025['Week_Num'],
            y=agg_2025['Top3_Sum'],
            mode='lines+markers',
            name=f'{series_name} 2025 (TOP3 总和)',
            line=dict(color='#E67E22', width=3),
            marker=dict(size=8),
            legendgroup=f'{series_name}-2025'
        ), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=agg_2024['Week_Num'],
            y=agg_2024['Top3_Sum'],
            mode='lines+markers',
            name=f'{series_name} 2024 (TOP3 总和)',
            line=dict(color='#3498DB', width=3),
            marker=dict(size=8),
            legendgroup=f'{series_name}-2024'
        ), row=current_row, col=1)

        # 集中度（TOP3 占比）
        fig.add_trace(go.Scatter(
            x=agg_2025['Week_Num'],
            y=agg_2025['Concentration'],
            mode='lines+markers',
            name=f'{series_name} 2025 (TOP3 集中度)',
            line=dict(color='#E67E22', width=3, dash='dot'),
            marker=dict(size=8, symbol='diamond'),
            legendgroup=f'{series_name}-2025',
            showlegend=True
        ), row=current_row + 1, col=1)
        fig.add_trace(go.Scatter(
            x=agg_2024['Week_Num'],
            y=agg_2024['Concentration'],
            mode='lines+markers',
            name=f'{series_name} 2024 (TOP3 集中度)',
            line=dict(color='#3498DB', width=3, dash='dot'),
            marker=dict(size=8, symbol='diamond'),
            legendgroup=f'{series_name}-2024',
            showlegend=True
        ), row=current_row + 1, col=1)
        current_row += 2
    
    
    # Layout Styling
    fig.update_layout(
        title='周度竞品PK分析：TOP3总和与集中度 (2024 vs 2025)',
        plot_bgcolor='#FFFFFF',
        height=1800,
        legend=dict(
            bordercolor='#7B848F',
            borderwidth=1,
            font=dict(color='#7B848F')
        )
    )
    
    share_rows = {i for i in range(1, total_rows + 1) if i % 2 == 0}
    for row in range(1, total_rows + 1):
        fig.update_xaxes(
            title_text='周数' if row in share_rows else None,
            gridcolor='#ebedf0',
            zerolinecolor='#ebedf0',
            tickfont=dict(color='#7B848F'),
            title_font=dict(color='#7B848F'),
            row=row, col=1
        )
        fig.update_yaxes(
            title_text=('TOP3 PK次数总和' if row % 2 == 1 else 'TOP3 集中度 (%)'),
            gridcolor='#ebedf0',
            zerolinecolor='#ebedf0',
            tickfont=dict(color='#7B848F'),
            title_font=dict(color='#7B848F'),
            tickformat='.1%' if row in share_rows or row % 2 == 0 else None,
            row=row, col=1
        )
    
    output_file = Path("reports/pk_trend_analysis.html")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 生成系列-年份维度的TOP3竞争对手表格
    comp_year_summary = df.groupby(['series', 'Year', '车系'])['PK_Count'].sum().reset_index()
    comp_year_totals = df.groupby(['series', 'Year'])['PK_Count'].sum()
    series_list = ['LS6', 'L6', 'LS9']
    # 为每个 series 生成独立表格（行：第一/第二/第三；列：2024、2025）
    series_tables_html_parts = []
    for s in series_list:
        names_2024 = []
        names_2025 = []
        for yr in [2024, 2025]:
            scomp = comp_year_summary[(comp_year_summary['series'] == s) & (comp_year_summary['Year'] == yr)]
            scomp = scomp.sort_values('PK_Count', ascending=False).head(3)
            try:
                total_pk = comp_year_totals.loc[(s, yr)]
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
            rows.append({'排名': label, '2024': names_2024[i] if i < len(names_2024) else '', '2025': names_2025[i] if i < len(names_2025) else ''})
        df_single = pd.DataFrame(rows, columns=['排名', '2024', '2025'])
        table_html = df_single.to_html(index=False, escape=False, classes='table table-striped', border=0)
        series_tables_html_parts.append(f"<h3 style=\"color:#7B848F;\">{s}</h3>\n{table_html}")
    comp_table_html = "\n".join(series_tables_html_parts)
    
    # 合并图表和表格到一个HTML
    html_body = fig.to_html(full_html=True, include_plotlyjs='cdn')
    inject_html = f"""
    <style>
      .table {{ width: 100%; border-collapse: collapse; font-size: 14px; color: #2c3e50; }}
      .table thead th {{ background: #f5f7fa; color: #7B848F; font-weight: 600; padding: 8px; border-bottom: 1px solid #eaecef; text-align: center; }}
      .table tbody td {{ padding: 8px; border-top: 1px solid #f0f2f5; text-align: center; }}
      .table-striped tbody tr:nth-child(odd) {{ background: #fbfcfd; }}
      .table-striped tbody tr:hover {{ background: #f6f8fa; }}
      h3 {{ margin-top: 8px; margin-bottom: 8px; }}
    </style>
    <div style="padding:16px">
      <h2 style="color:#7B848F;">系列对比：TOP3 竞争对手（2024 vs 2025）</h2>
      {comp_table_html}
    </div>
    """
    # 更稳健地注入：优先替换 </body>，若未命中则追加
    if "</body>" in html_body:
        final_html = html_body.replace("</body>", inject_html + "</body>")
    else:
        final_html = html_body + inject_html + "</body></html>"
    
    output_file.write_text(final_html, encoding='utf-8')
    print(f"Chart saved to {output_file}")
    
    print("\nSummary Data (Head):")
    print(res_df.head())

if __name__ == "__main__":
    analyze_pk_trend()
