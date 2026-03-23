#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置情况汇总脚本

功能：
- 读取转置后的配置详情数据 (CSV)
- 模块一：输出数据概览（锁单总数、交付数），含员工单维度
- 模块二：输出激光雷达 (OP-LASER) 的配置分布情况，含员工单维度
- 生成 Markdown 格式的分析报告

用法：
  python configure_summary.py --model CM2
  python configure_summary.py --model LS9
"""

import os
import sys
import argparse
import json
import pandas as pd
from datetime import datetime

# 项目根目录 (假设脚本在 scripts/ 目录下，向上两级或一级找到 processed 目录)
# 这里假设脚本在 /Users/zihao_/Documents/coding/dataset/scripts/ 目录下
# 数据在 /Users/zihao_/Documents/coding/dataset/processed/ 目录下
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_DIR, 'processed')
ANALYSIS_RESULTS_DIR = os.path.join(PROCESSED_DIR, 'analysis_results')
BUSINESS_DEFINITION_PATH = '/Users/zihao_/Documents/github/26W06_Tool_calls/schema/business_definition.json'

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_time_periods(json_path):
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        time_periods = data.get('time_periods', {})
        return time_periods if isinstance(time_periods, dict) else {}
    except Exception:
        return {}

def analyze_configuration(model):
    """分析指定车型的配置数据"""
    
    file_name = f'{model}_Configuration_Details_transposed.csv'
    file_path = os.path.join(PROCESSED_DIR, file_name)
    
    print(f"正在读取数据文件: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件不存在 -> {file_path}")
        print("请先运行 configuration_workflow.py 导出并转置数据。")
        sys.exit(1)
        
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        sys.exit(1)
        
    # 检查必要的列是否存在
    required_columns = ['order_number', 'lock_time', 'invoice_time', 'OP-LASER', 'Product_Types', 'Product Name']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"⚠️ 警告: 数据文件中缺少以下列，可能会影响分析结果: {missing_columns}")
        if 'lock_time' in missing_columns or 'order_number' in missing_columns:
            print("❌ 缺少核心列 (lock_time 或 order_number)，无法进行分析。")
            sys.exit(1)

    # ---------------------------------------------------------
    # 数据预处理
    # ---------------------------------------------------------
    # 过滤锁单数据
    locked_df = df[df['lock_time'].notna()].copy()
    
    if locked_df.empty:
        print("\n⚠️ 无锁单数据，无法生成报告。")
        return

    # 转换时间列以获取日期范围
    try:
        locked_df['lock_time_dt'] = pd.to_datetime(locked_df['lock_time'], errors='coerce')
        min_date = locked_df['lock_time_dt'].min().strftime('%Y-%m-%d')
        max_date = locked_df['lock_time_dt'].max().strftime('%Y-%m-%d')
    except Exception as e:
        print(f"⚠️ 时间转换警告: {e}")
        min_date = "Unknown"
        max_date = "Unknown"

    # 准备报告内容
    report_lines = []
    report_title = f"# {model} 配置情况分析报告"
    report_lines.append(report_title)
    report_lines.append("")
    # 移除源文件路径显示
    report_lines.append(f"- 数据时间范围 (Lock Time): `{min_date}` ~ `{max_date}`")
    report_lines.append(f"- 生成时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    report_lines.append("")

    # ---------------------------------------------------------
    # 模块一：数据概览
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(f"模块一：{model} 数据概览")
    print("="*50)
    
    total_locked = locked_df['order_number'].nunique()
    
    # 交付数据
    if 'invoice_time' in df.columns:
        delivered_df = locked_df[locked_df['invoice_time'].notna()]
        total_delivered = delivered_df['order_number'].nunique()
    else:
        total_delivered = 0
        delivered_df = pd.DataFrame()
        
    # 员工单统计 (Is Staff)
    is_staff_col = 'Is Staff'
    has_staff_info = is_staff_col in df.columns
    
    if has_staff_info:
        # 锁单拆解
        staff_locked = locked_df[locked_df[is_staff_col] == 'Y']['order_number'].nunique()
        non_staff_locked = locked_df[locked_df[is_staff_col] != 'Y']['order_number'].nunique() # 假设非Y即非员工
        
        # 交付拆解
        if not delivered_df.empty:
            staff_delivered = delivered_df[delivered_df[is_staff_col] == 'Y']['order_number'].nunique()
            non_staff_delivered = delivered_df[delivered_df[is_staff_col] != 'Y']['order_number'].nunique()
        else:
            staff_delivered = 0
            non_staff_delivered = 0
    else:
        print("⚠️ 数据中缺少 'Is Staff' 列，无法拆分员工单。")

    # 添加到报告
    report_lines.append("## 数据概览")
    
    if has_staff_info:
        report_lines.append("| 用户类型 | 锁单数 | 交付数 |")
        report_lines.append("| :--- | ---: | ---: |")
        report_lines.append(f"| 全部 | {total_locked} | {total_delivered} |")
        report_lines.append(f"| 员工单 (Is Staff=Y) | {staff_locked} | {staff_delivered} |")
        report_lines.append(f"| 非员工单 | {non_staff_locked} | {non_staff_delivered} |")
    else:
        report_lines.append("| 指标 | 数量 |")
        report_lines.append("| :--- | ---: |")
        report_lines.append(f"| 锁单总数 | {total_locked} |")
        report_lines.append(f"| 交付总数 | {total_delivered} |")
    
    report_lines.append("")

    print(f"🔒 锁单总数: {total_locked}")
    if has_staff_info:
        print(f"   - 员工单: {staff_locked}")
        print(f"   - 非员工: {non_staff_locked}")
    print(f"🚚 交付总数: {total_delivered}")

    # ---------------------------------------------------------
    # 数据完整度检查
    # ---------------------------------------------------------
    print("\n--- 配置数据完整度检查 (基于锁单) ---")
    
    # 定义关注的配置列
    # 排除非配置列和系统列
    exclude_cols = ['order_number', 'lock_time', 'invoice_time', 'Is Staff', 'lock_time_dt', '开票价格', 'Product Name', 'Product_Types']
    # 动态获取潜在配置列
    potential_config_cols = [c for c in locked_df.columns if c not in exclude_cols]
    
    # 优先展示常见配置
    priority_cols = ['EXCOLOR', 'INCOLOR', 'WHEEL', 'OP-LASER']
    # 剩余的列
    other_cols = [c for c in potential_config_cols if c not in priority_cols]
    # 合并列表
    target_cols = [c for c in priority_cols if c in locked_df.columns] + sorted(other_cols)

    completeness_data = []
    for col in target_cols:
        non_null_count = locked_df[col].count()
        completeness_rate = (non_null_count / total_locked) * 100 if total_locked > 0 else 0
        completeness_data.append({
            '配置项': col,
            '有效数据量': non_null_count,
            '完整度': f"{completeness_rate:.1f}%"
        })
    
    df_completeness = pd.DataFrame(completeness_data)
    if not df_completeness.empty:
        print(df_completeness.to_string(index=False))
        
        report_lines.append("### 配置数据完整度")
        report_lines.append(df_completeness.to_markdown(index=False))
        report_lines.append("")
    else:
        print("未检测到配置列。")

    
    # ---------------------------------------------------------
    # 模块二：激光雷达 (OP-LASER) 配置情况
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(f"模块二：{model} 激光雷达 (OP-LASER) 配置情况")
    print("="*50)
    
    if 'OP-LASER' in df.columns:
        # 1. 整体分布
        print(f"\n--- {model} 整体 OP-LASER 分布 (基于锁单) ---")
        laser_counts = locked_df['OP-LASER'].value_counts(dropna=False)
        laser_percentages = locked_df['OP-LASER'].value_counts(normalize=True, dropna=False) * 100
        
        df_laser_summary = pd.DataFrame({
            'OP-LASER': laser_counts.index,
            'Count': laser_counts.values,
            'Percentage': laser_percentages.values
        })
        # 格式化百分比
        df_laser_summary['Percentage'] = df_laser_summary['Percentage'].apply(lambda x: f"{x:.1f}%")
        
        print(df_laser_summary.to_string(index=False))
        
        # 添加到报告
        report_lines.append("## 激光雷达 (OP-LASER) 整体分布")
        report_lines.append(df_laser_summary.to_markdown(index=False))
        report_lines.append("")

        # 2. 分 Is Staff 的 OP-LASER 分布
        if has_staff_info:
            print("\n--- 分 [Is Staff] 的 OP-LASER 分布 ---")
            
            # 使用 pivot table 展示
            staff_pivot = pd.pivot_table(
                locked_df, 
                index=['OP-LASER'], 
                columns='Is Staff', 
                values='order_number', 
                aggfunc='count', 
                fill_value=0,
                margins=True,
                margins_name='Total'
            )
            print(staff_pivot)
            
            # 添加到报告
            report_lines.append("## 分员工单 (Is Staff) 激光雷达分布")
            # 重置索引以便在 markdown 中显示 OP-LASER 列
            staff_pivot_md = staff_pivot.reset_index()
            report_lines.append(staff_pivot_md.to_markdown(index=False))
            report_lines.append("")

        # 3. 分 Product_Types 和 Product Name (针对 高阶+Thor)
        if 'Product_Types' in df.columns and 'Product Name' in df.columns:
            target_laser = '高阶+Thor'
            print(f"\n--- 分 [Product_Types]、[Product Name] 的 {target_laser} 分布 ---")
            
            # 1. 计算每个车型的总锁单数
            model_counts = locked_df.groupby(['Product_Types', 'Product Name']).size().reset_index(name='Total Orders')
            
            # 2. 计算每个车型中 OP-LASER == '高阶+Thor' 的数量
            target_df = locked_df[locked_df['OP-LASER'] == target_laser]
            target_counts = target_df.groupby(['Product_Types', 'Product Name']).size().reset_index(name='Target Orders')
            
            # 3. 合并数据
            merged_df = pd.merge(model_counts, target_counts, on=['Product_Types', 'Product Name'], how='left')
            merged_df['Target Orders'] = merged_df['Target Orders'].fillna(0).astype(int)
            
            # 4. 计算渗透率
            merged_df['Take Rate'] = (merged_df['Target Orders'] / merged_df['Total Orders'] * 100).map('{:.1f}%'.format)
            
            # 5. 排序 (按目标数量降序)
            merged_df = merged_df.sort_values('Target Orders', ascending=False)
            
            # 6. 重命名列以显示在报告中
            display_df = merged_df.rename(columns={
                'Target Orders': f'{target_laser} 锁单数',
                'Total Orders': '车型总锁单数',
                'Take Rate': '渗透率'
            })
            
            print(display_df.to_string(index=False))

            # 添加到报告
            report_lines.append(f"## 分车型 (Product Name) {target_laser} 分布")
            report_lines.append(display_df.to_markdown(index=False))
            report_lines.append("")
            
    else:
        print("⚠️ 数据中缺少 'OP-LASER' 列，无法分析激光雷达配置。")
        report_lines.append("⚠️ 数据中缺少 'OP-LASER' 列，无法分析激光雷达配置。")

    # ---------------------------------------------------------
    # 模块三：轮毂 (WHEEL) 配置情况
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(f"模块三：{model} 轮毂 (WHEEL) 配置情况")
    print("="*50)

    if 'WHEEL' in df.columns:
        # 1. 整体分布
        print(f"\n--- {model} 整体 WHEEL 分布 (基于锁单) ---")
        wheel_counts = locked_df['WHEEL'].value_counts(dropna=False)
        wheel_percentages = locked_df['WHEEL'].value_counts(normalize=True, dropna=False) * 100
        
        df_wheel_summary = pd.DataFrame({
            'WHEEL': wheel_counts.index,
            'Count': wheel_counts.values,
            'Percentage': wheel_percentages.values
        })
        # 格式化百分比
        df_wheel_summary['Percentage'] = df_wheel_summary['Percentage'].apply(lambda x: f"{x:.1f}%")
        
        print(df_wheel_summary.to_string(index=False))
        
        # 添加到报告
        report_lines.append("## 轮毂 (WHEEL) 整体分布")
        report_lines.append(df_wheel_summary.to_markdown(index=False))
        report_lines.append("")

        # 2. 分 Is Staff 的 WHEEL 分布
        if has_staff_info:
            print("\n--- 分 [Is Staff] 的 WHEEL 分布 ---")
            
            # 使用 pivot table 展示
            wheel_staff_pivot = pd.pivot_table(
                locked_df, 
                index=['WHEEL'], 
                columns='Is Staff', 
                values='order_number', 
                aggfunc='count', 
                fill_value=0,
                margins=True,
                margins_name='Total'
            )
            print(wheel_staff_pivot)
            
            # 添加到报告
            report_lines.append("## 分员工单 (Is Staff) 轮毂分布")
            # 重置索引以便在 markdown 中显示 WHEEL 列
            wheel_staff_pivot_md = wheel_staff_pivot.reset_index()
            report_lines.append(wheel_staff_pivot_md.to_markdown(index=False))
            report_lines.append("")

        # 3. 分 Product Name 的 WHEEL 分布
        if 'Product Name' in df.columns:
            print("\n--- 分 [Product Name] 的 WHEEL 分布 ---")
            
            wheel_product_pivot = pd.pivot_table(
                locked_df,
                index=['WHEEL'],
                columns='Product Name',
                values='order_number',
                aggfunc='count',
                fill_value=0,
                margins=True,
                margins_name='Total'
            )
            # 计算百分比显示可能比较复杂，这里先只展示数量，或者计算行百分比
            print(wheel_product_pivot)

            # 添加到报告
            report_lines.append("## 分车型 (Product Name) 轮毂分布")
            wheel_product_pivot_md = wheel_product_pivot.reset_index()
            report_lines.append(wheel_product_pivot_md.to_markdown(index=False))
            report_lines.append("")

    else:
        print("⚠️ 数据中缺少 'WHEEL' 列，无法分析轮毂配置。")
        report_lines.append("⚠️ 数据中缺少 'WHEEL' 列，无法分析轮毂配置。")

    # ---------------------------------------------------------
    # 模块四：拖钩 (OP-Hitch) 配置情况
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(f"模块四：{model} 拖钩 (OP-Hitch) 配置情况")
    print("="*50)

    if 'OP-Hitch' in df.columns:
        # 1. 整体分布
        print(f"\n--- {model} 整体 OP-Hitch 分布 (基于锁单) ---")
        hitch_counts = locked_df['OP-Hitch'].value_counts(dropna=False)
        hitch_percentages = locked_df['OP-Hitch'].value_counts(normalize=True, dropna=False) * 100
        
        df_hitch_summary = pd.DataFrame({
            'OP-Hitch': hitch_counts.index,
            'Count': hitch_counts.values,
            'Percentage': hitch_percentages.values
        })
        # 格式化百分比
        df_hitch_summary['Percentage'] = df_hitch_summary['Percentage'].apply(lambda x: f"{x:.1f}%")
        
        print(df_hitch_summary.to_string(index=False))
        
        # 添加到报告
        report_lines.append("## 拖钩 (OP-Hitch) 整体分布")
        report_lines.append(df_hitch_summary.to_markdown(index=False))
        report_lines.append("")

        # 2. 分 Is Staff 的 OP-Hitch 分布
        if has_staff_info:
            print("\n--- 分 [Is Staff] 的 OP-Hitch 分布 ---")
            
            # 使用 pivot table 展示
            hitch_staff_pivot = pd.pivot_table(
                locked_df, 
                index=['OP-Hitch'], 
                columns='Is Staff', 
                values='order_number', 
                aggfunc='count', 
                fill_value=0,
                margins=True,
                margins_name='Total'
            )
            print(hitch_staff_pivot)
            
            # 添加到报告
            report_lines.append("## 分员工单 (Is Staff) 拖钩分布")
            # 重置索引以便在 markdown 中显示 OP-Hitch 列
            hitch_staff_pivot_md = hitch_staff_pivot.reset_index()
            report_lines.append(hitch_staff_pivot_md.to_markdown(index=False))
            report_lines.append("")

        # 3. 分 Product Name 的 OP-Hitch 分布
        if 'Product Name' in df.columns:
            print("\n--- 分 [Product Name] 的 OP-Hitch 分布 ---")
            
            hitch_product_pivot = pd.pivot_table(
                locked_df,
                index=['OP-Hitch'],
                columns='Product Name',
                values='order_number',
                aggfunc='count',
                fill_value=0,
                margins=True,
                margins_name='Total'
            )
            print(hitch_product_pivot)

            # 添加到报告
            report_lines.append("## 分车型 (Product Name) 拖钩分布")
            hitch_product_pivot_md = hitch_product_pivot.reset_index()
            report_lines.append(hitch_product_pivot_md.to_markdown(index=False))
            report_lines.append("")

    else:
        print("⚠️ 数据中缺少 'OP-Hitch' 列，无法分析拖钩配置。")
        # report_lines.append("⚠️ 数据中缺少 'OP-Hitch' 列，无法分析拖钩配置。") # Optional: decide if we want this in report if missing. 
        # Actually, for consistency with WHEEL block, let's keep it but maybe not clutter report if column is totally absent for models that don't have it.
        # But user asked for it, so let's include the message if it's missing so they know why.
        report_lines.append("⚠️ 数据中缺少 'OP-Hitch' 列，无法分析拖钩配置。")

    print("\n" + "="*50)
    print(f"模块五：{model} 冰箱 (OP-FRIDGE) 月度占比（分 Product_Types）")
    print("="*50)

    if 'OP-FRIDGE' in df.columns and 'Product_Types' in df.columns:
        time_periods = load_time_periods(BUSINESS_DEFINITION_PATH)
        listing_start = pd.to_datetime(time_periods.get(model, {}).get('end'), errors='coerce')
        if pd.isna(listing_start):
            listing_start = locked_df['lock_time_dt'].min()

        analysis_df = locked_df[locked_df['lock_time_dt'].notna()].copy()
        if not pd.isna(listing_start):
            analysis_df = analysis_df[analysis_df['lock_time_dt'] >= listing_start]

        if analysis_df.empty:
            print("⚠️ 上市时间过滤后无数据，跳过 OP-FRIDGE 月度分析。")
            report_lines.append("## 冰箱 (OP-FRIDGE) 月度占比（分 Product_Types，自上市以来）")
            report_lines.append("⚠️ 上市时间过滤后无数据，跳过 OP-FRIDGE 月度分析。")
            report_lines.append("")
        else:
            analysis_df['lock_month'] = analysis_df['lock_time_dt'].dt.to_period('M').astype(str)

            print(f"- 上市时间(End Day): {listing_start.strftime('%Y-%m-%d') if not pd.isna(listing_start) else 'Unknown'}")
            report_lines.append("## 冰箱 (OP-FRIDGE) 月度占比（分 Product_Types，自上市以来）")
            report_lines.append(f"- 上市时间(End Day): `{listing_start.strftime('%Y-%m-%d') if not pd.isna(listing_start) else 'Unknown'}`")
            report_lines.append("")

            analysis_df['Product_Types_str'] = analysis_df['Product_Types'].astype(str)
            product_types = sorted(analysis_df['Product_Types_str'].dropna().unique())
            for product_type in product_types:
                pt_df = analysis_df[analysis_df['Product_Types_str'] == product_type].copy()
                if pt_df.empty:
                    continue

                fridge_v = pt_df['OP-FRIDGE'].astype(str).str.strip().str.lower()
                yes_mask = fridge_v.isin({'是', 'y', 'yes', 'true', '1', '1.0'})

                total_by_month = pt_df.groupby('lock_month')['order_number'].nunique().rename('Total Orders')
                yes_by_month = pt_df[yes_mask].groupby('lock_month')['order_number'].nunique().rename('OP-FRIDGE=是 Orders')

                summary_df = pd.concat([total_by_month, yes_by_month], axis=1).fillna(0).reset_index()
                summary_df = summary_df.rename(columns={'lock_month': 'Month'})
                summary_df['OP-FRIDGE=是 Orders'] = summary_df['OP-FRIDGE=是 Orders'].astype(int)
                summary_df['Take Rate'] = (summary_df['OP-FRIDGE=是 Orders'] / summary_df['Total Orders'] * 100).round(1).map(lambda x: f"{x:.1f}%")
                summary_df = summary_df.sort_values('Month', ascending=True)

                print(f"\n--- Product_Types={product_type} ---")
                print(summary_df.to_string(index=False))

                report_lines.append(f"### {product_type}")
                report_lines.append(summary_df.to_markdown(index=False))
                report_lines.append("")

    else:
        print("⚠️ 数据中缺少 'OP-FRIDGE' 或 'Product_Types' 列，无法分析冰箱配置。")
        report_lines.append("## 冰箱 (OP-FRIDGE) 月度占比（分 Product_Types，自上市以来）")
        report_lines.append("⚠️ 数据中缺少 'OP-FRIDGE' 或 'Product_Types' 列，无法分析冰箱配置。")
        report_lines.append("")

    # ---------------------------------------------------------
    # 保存报告
    # ---------------------------------------------------------
    ensure_dir(ANALYSIS_RESULTS_DIR)
    output_filename = f"configure_summary_{model}_{min_date}_to_{max_date}.md"
    output_path = os.path.join(ANALYSIS_RESULTS_DIR, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
        
    print("\n" + "="*50)
    print(f"✅ 报告已生成: {output_path}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="配置情况汇总分析脚本")
    
    parser.add_argument('--CM2', action='store_true', help='分析 CM2 车型')
    parser.add_argument('--LS9', action='store_true', help='分析 LS9 车型')
    parser.add_argument('--model', type=str, help='指定车型 (例如 CM2, LS9)')
    
    args = parser.parse_args()
    
    model = None
    if args.CM2:
        model = 'CM2'
    elif args.LS9:
        model = 'LS9'
    elif args.model:
        model = args.model
        
    if not model:
        print("请指定车型: 使用 --CM2, --LS9 或 --model <ModelName>")
        sys.exit(1)
        
    analyze_configuration(model)

if __name__ == '__main__':
    main()
