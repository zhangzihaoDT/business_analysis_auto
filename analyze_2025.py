import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

PARQUET_FILE = Path("/Users/zihao_/Documents/coding/dataset/formatted/order_full_data.parquet")
BUSINESS_DEF_FILE = Path("/Users/zihao_/Documents/github/W52_reasoning/world/business_definition.json")
DEFAULT_OUTPUT = Path("reports/review_2025.html")

def load_data(file_path: Path) -> pd.DataFrame:
    """加载 Parquet 数据"""
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    print(f"📖 Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"✅ Loaded {len(df)} rows.")
    return df

def get_period_mask(df: pd.DataFrame, date_col: str, year: int) -> pd.Series:
    """
    生成指定年份的时间过滤掩码
    year=2024: 2024-01-01 ~ 2024-12-31
    year=2025: 2025-01-01 ~ max (即 >= 2025-01-01)
    """
    if date_col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
        
    # 确保是 datetime 类型
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
         df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    start_date = pd.Timestamp(f"{year}-01-01")
    
    if year == 2024:
        end_date = pd.Timestamp(f"{year}-12-31 23:59:59")
        return (df[date_col] >= start_date) & (df[date_col] <= end_date)
    else:
        # For 2025 and beyond, just take everything from start_date
        return df[date_col] >= start_date

def calculate_metrics(df: pd.DataFrame) -> dict:
    """计算核心指标"""
    metrics = {}
    
    # 定义指标列表
    metric_names = ["锁单数", "开票数", "锁单退订数"]
    
    def get_metric_mask(metric_name: str, year: int) -> pd.Series:
        """
        获取指定指标在指定年份的过滤掩码
        """
        if metric_name == "锁单数":
            return get_period_mask(df, "lock_time", year)
            
        elif metric_name == "开票数":
            return get_period_mask(df, "invoice_upload_time", year)
            
        elif metric_name == "锁单退订数":
            # approve_refund_time在这两个周期
            # 且 lock_time 不为空 (not null)
            # (已移除 apply_refund_time 校验条件)
            
            # 1. approve_refund_time 在周期内
            time_mask = get_period_mask(df, "approve_refund_time", year)
            
            # 2. lock_time 不为空
            if "lock_time" not in df.columns:
                 lock_mask = pd.Series([False] * len(df), index=df.index)
            else:
                 lock_mask = df["lock_time"].notna()
                 
            return time_mask & lock_mask
            
        return pd.Series([False] * len(df), index=df.index)
    
    # 1. 总体概览
    overall_stats = []
    
    for metric_name in metric_names:
        # 2024 数据
        mask_2024 = get_metric_mask(metric_name, 2024)
        count_2024 = df[mask_2024]['order_number'].nunique()
        
        # 2025 数据
        mask_2025 = get_metric_mask(metric_name, 2025)
        count_2025 = df[mask_2025]['order_number'].nunique()
        
        # 同比 (YoY) - 注意：2025可能不完整，这里仅计算简单增长率供参考，或者留空
        yoy = ((count_2025 - count_2024) / count_2024) if count_2024 > 0 else 0.0
        
        ratio_str = f"{yoy:.1%}"
        if yoy < 0:
            ratio_str = f"<span style='color: red'>{ratio_str}</span>"
            
        overall_stats.append({
            "指标": metric_name,
            "2024 全年": count_2024,
            "2025 (至今)": count_2025,
            "Diff": count_2025 - count_2024,
            "Ratio": ratio_str
        })
        
    metrics['overall'] = pd.DataFrame(overall_stats)
    
    # 2. 分 Series 对比 (拆分为三个独立的 DataFrame)
    series_details = {}
    
    # 明确指定需要展示的 series 顺序，并添加 Total
    target_series = ["L6", "LS6", "LS9"]
    
    for metric_name in metric_names:
        rows = []
        
        # 1. 计算各车型数据
        for s in target_series:
            # Filter by series
            series_mask = df['series'] == s
            
            # 2024
            mask_2024 = series_mask & get_metric_mask(metric_name, 2024)
            val_2024 = df[mask_2024]['order_number'].nunique()
            
            # 2025
            mask_2025 = series_mask & get_metric_mask(metric_name, 2025)
            val_2025 = df[mask_2025]['order_number'].nunique()
            
            # Diff & Ratio
            diff = val_2025 - val_2024
            ratio = (diff / val_2024) if val_2024 > 0 else 0.0
            
            ratio_str = f"{ratio:.1%}"
            if ratio < 0:
                ratio_str = f"<span style='color: red'>{ratio_str}</span>"
            
            rows.append({
                "Series": s,
                "2024 全年": val_2024,
                "2025 (至今)": val_2025,
                "Diff": diff,
                "Ratio": ratio_str
            })
            
        # 2. 计算 Total (仅包含 target_series 的总和)
        total_2024 = sum(row["2024 全年"] for row in rows)
        total_2025 = sum(row["2025 (至今)"] for row in rows)
        total_diff = total_2025 - total_2024
        total_ratio = (total_diff / total_2024) if total_2024 > 0 else 0.0
        
        total_ratio_str = f"{total_ratio:.1%}"
        if total_ratio < 0:
            total_ratio_str = f"<span style='color: red'>{total_ratio_str}</span>"
            
        rows.append({
            "Series": "总计",
            "2024 全年": total_2024,
            "2025 (至今)": total_2025,
            "Diff": total_diff,
            "Ratio": total_ratio_str
        })
            
        series_details[metric_name] = pd.DataFrame(rows)
        
    metrics['series_details'] = series_details
    
    return metrics

def generate_html(metrics: dict, output_file: Path):
    """生成 HTML 报告"""
    
    # CSS 样式
    css = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; color: #333; }
        h1 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; border-left: 5px solid #3498db; padding-left: 10px; }
        h3 { color: #2980b9; margin-top: 25px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        th, td { text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: 600; color: #555; }
        tr:hover { background-color: #f5f5f5; }
        .timestamp { color: #888; font-size: 0.9em; margin-bottom: 20px; }
        .summary-box { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    </style>
    """
    
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>2024 vs 2025 业务指标对比分析</title>",
        css,
        "</head>",
        "<body>",
        "<h1>2024 vs 2025 业务指标对比分析</h1>",
        f"<div class='timestamp'>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>"
    ]
    
    # 1. 指标概览
    html_content.append("<h2>1. 指标概览 (Overview)</h2>")
    html_content.append("<div class='summary-box'>")
    html_content.append("<p>统计周期说明：</p>")
    html_content.append("<ul>")
    html_content.append("<li>2024: 2024-01-01 ~ 2024-12-31</li>")
    html_content.append("<li>2025: 2025-01-01 ~ 至今 (Max Date)</li>")
    html_content.append("</ul>")
    html_content.append("</div>")
    
    df_overall = metrics['overall']
    html_content.append(df_overall.to_html(index=False, classes='table', escape=False, float_format=lambda x: '{:,.0f}'.format(x) if isinstance(x, (int, float)) else x))
    
    # 1.1 指标概览（分车型）
    html_content.append("<h2>1.1 指标概览 - 分车型 (By Series)</h2>")
    
    series_details = metrics['series_details']
    # 按特定顺序展示：锁单 -> 开票 -> 退订
    display_order = ["锁单数", "开票数", "锁单退订数"]
    
    for metric_name in display_order:
        if metric_name in series_details:
            df_table = series_details[metric_name]
            html_content.append(f"<h3>{metric_name}</h3>")
            html_content.append(df_table.to_html(index=False, classes='table', escape=False, float_format=lambda x: '{:,.0f}'.format(x) if isinstance(x, (int, float)) else x))
    
    html_content.append("</body></html>")
    
    # 保存
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_content))
    
    print(f"✅ Report generated at: {output_file}")

def main():
    try:
        df = load_data(PARQUET_FILE)
        metrics = calculate_metrics(df)
        generate_html(metrics, DEFAULT_OUTPUT)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
