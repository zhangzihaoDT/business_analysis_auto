#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Order 完整数据处理脚本

该脚本用于处理 Order_完整数据_data.csv 和 Order_完整数据_data_2024.csv 文件
将其合并、去重并转换为优化的 Parquet 格式

输入文件: 
- original/Order_完整数据_data.csv
- original/Order_完整数据_data_2024.csv
输出文件: formatted/order_full_data.parquet
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime

# 设置基础路径
BASE_DIR = Path("/Users/zihao_/Documents/coding/dataset")
ORIGINAL_DIR = BASE_DIR / "original"
FORMATTED_DIR = BASE_DIR / "formatted"
OUTPUT_FILE = FORMATTED_DIR / "order_full_data.parquet"

def read_csv_smart(file_path: Path) -> pd.DataFrame:
    """
    智能读取 CSV 文件，尝试多种编码和分隔符
    """
    if not file_path.exists():
        print(f"⚠️ 文件不存在: {file_path}")
        return pd.DataFrame()

    print(f"📖 正在读取: {file_path.name} ...")
    
    # 常见编码和分隔符组合
    encodings = ["utf-16", "utf-8", "utf-8-sig", "gb18030", "gbk"]
    separators = ["\t", ","]
    
    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(file_path, encoding=enc, sep=sep)
                
                # 简单验证读取是否成功：如果列数只有1且包含分隔符，说明分隔符不对
                if df.shape[1] == 1 and sep in str(df.columns[0]):
                    continue
                
                # 如果列数大于1，通常说明读取正确
                if df.shape[1] > 1:
                    print(f"✅ 读取成功 (编码: {enc}, 分隔符: '{sep if sep != '\t' else '\\t'}') - 形状: {df.shape}")
                    return df
            except Exception:
                continue
                
    # 如果都失败了，尝试默认读取
    try:
        print("⚠️ 尝试默认参数读取...")
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return pd.DataFrame()

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理列名：去除空白字符，统一命名风格
    """
    # 去除前后空格
    df.columns = df.columns.str.strip()
    
    # 预处理：统一将 'xxx_年/月/日' 格式转换为 'xxx 年/月/日'，以匹配下方的映射表
    # 这样可以兼容下划线和空格两种分隔符
    df.columns = df.columns.str.replace('_年/月/日', ' 年/月/日', regex=False)

    # 重命名映射表（根据之前的分析报告）
    rename_map = {
        'first_touch_time 年/月/日': 'first_touch_time',
        'delivery_date 年/月/日': 'delivery_date',
        'deposit_payment_time 年/月/日': 'deposit_payment_time',
        'deposit_refund_time 年/月/日': 'deposit_refund_time',
        'first_test_drive_time 年/月/日': 'first_test_drive_time',
        'intention_payment_time 年/月/日': 'intention_payment_time',
        'intention_refund_time 年/月/日': 'intention_refund_time',
        'invoice_upload_time 年/月/日': 'invoice_upload_time',
        'lock_time 年/月/日': 'lock_time',
        'order_create_time 年/月/日': 'order_create_date', # 区分 order_create_time
        'store_create_date 年/月/日': 'store_create_date',
        'approve_refund_time 年/月/日': 'approve_refund_time',
        'apply_refund_time 年/月/日': 'apply_refund_time',
        'Td CountD': 'td_countd',
        'Drive Series Cn': 'drive_series_cn',
        'Main Lead Id': 'main_lead_id',
    }
    
    # 应用重命名
    df = df.rename(columns=rename_map)
    
    # 将剩余列名转换为下划线风格（如果已经是英文）
    # 这里简单处理，只替换空格
    df.columns = df.columns.str.replace(' ', '_')
    
    return df

def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    转换数据类型
    """
    print("🔄 开始类型转换...")
    
    # 1. 日期列转换
    date_cols = [
        'first_touch_time', 'delivery_date', 'deposit_payment_time', 
        'deposit_refund_time', 'first_test_drive_time', 'intention_payment_time', 
        'intention_refund_time', 'invoice_upload_time', 'lock_time', 
        'order_create_date', 'store_create_date', 'order_create_time',
        'approve_refund_time', 'apply_refund_time'
    ]
    
    for col in date_cols:
        if col in df.columns:
            # 处理中文日期格式 (YYYY年MM月DD日)
            # 先将 series 转为 string
            s = df[col].astype(str)
            # 替换年月日
            s = s.str.replace('年', '-', regex=False).str.replace('月', '-', regex=False).str.replace('日', '', regex=False)
            # 处理可能的 'nan' 字符串
            s = s.replace({'nan': None, 'None': None, '': None})
            
            df[col] = pd.to_datetime(s, errors='coerce')
            print(f"   - 日期列转换: {col}")

    # 2. 数值列转换
    numeric_cols = ['age', 'invoice_amount', 'td_countd']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"   - 数值列转换: {col}")

    # 3. 分类列转换 (优化存储)
    cat_cols = [
        'product_name', 'final_payment_way', 'finance_product', 
        'first_middle_channel_name', 'gender', 'is_hold', 'is_staff',
        'license_city', 'license_city_level', 'license_province',
        'order_type', 'series', 'store_city', 'belong_intent_series',
        'drive_series_cn'
    ]
    
    for col in cat_cols:
        if col in df.columns:
            # 如果唯一值数量较少，转为 category
            if df[col].nunique() < df.shape[0] * 0.5:
                df[col] = df[col].astype('category')
                print(f"   - 分类列转换: {col}")
            else:
                df[col] = df[col].astype('string')

    # order_number 应该是字符串
    if 'order_number' in df.columns:
        df['order_number'] = df['order_number'].astype('string')

    return df

def main():
    # 1. 查找最新的数据文件
    csv_files = sorted(list(ORIGINAL_DIR.glob("Order_完整数据*.csv")), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not csv_files:
        print(f"❌ 未在 {ORIGINAL_DIR} 找到任何 'Order_完整数据*.csv' 文件")
        return

    latest_file = csv_files[0]
    print(f"🔍 发现最新数据文件: {latest_file.name}")
    
    # 2. 读取新数据
    df_new = read_csv_smart(latest_file)
    if df_new.empty:
        print("❌ 读取新数据失败，退出。")
        return
        
    # 3. 清理列名和转换类型（在新数据上进行）
    df_new = clean_column_names(df_new)
    df_new = convert_types(df_new)
    
    print(f"✅ 新数据处理完成: {df_new.shape[0]} 行")

    # 4. 增量更新逻辑
    if OUTPUT_FILE.exists():
        print(f"📚 发现现有 Parquet 文件: {OUTPUT_FILE}")
        try:
            df_existing = pd.read_parquet(OUTPUT_FILE)
            print(f"   现有数据: {df_existing.shape[0]} 行")

            # 修复：重命名旧数据中的未清洗列名 (防止 duplicate columns)
            legacy_map = {
                'approve_refund_time_年/月/日': 'approve_refund_time',
                'apply_refund_time_年/月/日': 'apply_refund_time',
                'approve_refund_time 年/月/日': 'approve_refund_time', # 增加空格版本以防万一
                'apply_refund_time 年/月/日': 'apply_refund_time'
            }
            
            for old_col, new_col in legacy_map.items():
                if old_col in df_existing.columns:
                    print(f"   🧹 处理旧数据列: {old_col} -> {new_col}")
                    
                    # 1. 先转换类型 (如果是字符串)
                    if df_existing[old_col].dtype == 'object':
                         try:
                             s = df_existing[old_col].astype(str)
                             s = s.str.replace('年', '-', regex=False).str.replace('月', '-', regex=False).str.replace('日', '', regex=False)
                             s = s.replace({'nan': None, 'None': None, '': None})
                             df_existing[old_col] = pd.to_datetime(s, errors='coerce')
                         except Exception as e:
                             print(f"      ⚠️ 转换失败: {e}")

                    # 2. 合并或重命名
                    if new_col in df_existing.columns:
                        # 如果目标列已存在，合并 (优先保留目标列的值，填充 NaNs)
                        df_existing[new_col] = df_existing[new_col].combine_first(df_existing[old_col])
                        df_existing = df_existing.drop(columns=[old_col])
                    else:
                        # 直接重命名
                        df_existing = df_existing.rename(columns={old_col: new_col})

            # 修复：移除冗余的 order_create_time 列（如果存在，且与 order_create_date 重复或新数据无此列）
            # if 'order_create_time' in df_existing.columns:
            #    print("   🧹 清理冗余列 'order_create_time' 以保持结构一致...")
            #    df_existing = df_existing.drop(columns=['order_create_time'])
            
            # 确保列结构一致
            common_cols = list(set(df_existing.columns) & set(df_new.columns))
            new_only = set(df_new.columns) - set(df_existing.columns)
            existing_only = set(df_existing.columns) - set(df_new.columns)
            
            if new_only or existing_only:
                print(f"⚠️ 列结构不一致:")
                if new_only: print(f"   新数据独有: {new_only}")
                if existing_only: print(f"   旧数据独有: {existing_only}")
                
                # 对齐列
                all_cols = list(set(df_existing.columns) | set(df_new.columns))
                df_existing = df_existing.reindex(columns=all_cols)
                df_new = df_new.reindex(columns=all_cols)
            
            # 智能合并
            if 'order_number' in df_new.columns and 'order_number' in df_existing.columns:
                print(f"🔄 执行智能增量合并...")
                
                # 转换为集合进行快速查找
                existing_orders = set(df_existing['order_number'].dropna())
                new_orders = set(df_new['order_number'].dropna())
                
                truly_new_orders = new_orders - existing_orders
                updated_orders = new_orders & existing_orders
                
                print(f"   新增订单: {len(truly_new_orders)}")
                print(f"   更新订单: {len(updated_orders)}")
                
                # 1. 保留旧数据中不在新数据里的（未更新的历史数据）
                # 注意：这里假设新文件只是增量或部分快照。如果是全量快照，逻辑可能不同。
                # 但根据用户描述，似乎是希望保留历史累积。
                # 如果新文件包含已有的订单，通常我们认为新文件的数据更新。
                
                # 移除旧数据中将被更新的订单
                df_final = df_existing[~df_existing['order_number'].isin(updated_orders)].copy()
                
                # 添加新数据（包含真正的新增和更新的订单）
                # 这里假设新文件里的记录就是最新的状态
                df_final = pd.concat([df_final, df_new], ignore_index=True)
                
            else:
                print("⚠️ 未找到 order_number 列，执行追加合并...")
                df_final = pd.concat([df_existing, df_new], ignore_index=True)
                
        except Exception as e:
            print(f"❌ 读取现有 Parquet 文件失败: {e}")
            print("   将仅使用新数据。")
            df_final = df_new
    else:
        print("📝 未发现现有 Parquet 文件，创建新文件...")
        df_final = df_new

    # 5. 最终去重（以防万一）
    if 'order_number' in df_final.columns:
        print(f"✂️  执行最终去重...")
        before_count = len(df_final)
        # keep='last' 确保保留最后加入的记录（即最新的）
        df_final = df_final.drop_duplicates(subset=['order_number'], keep='last')
        after_count = len(df_final)
        print(f"   去重前: {before_count}, 去重后: {after_count}, 移除: {before_count - after_count}")

    # 6. 保存
    if not FORMATTED_DIR.exists():
        FORMATTED_DIR.mkdir(parents=True)
    
    print(f"💾 保存到: {OUTPUT_FILE} ...")
    df_final.to_parquet(OUTPUT_FILE, index=False)
    
    # 验证
    if OUTPUT_FILE.exists():
        size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        print(f"✅ 保存成功! 文件大小: {size_mb:.2f} MB")
        print(f"   最终行数: {df_final.shape[0]}")
    else:
        print("❌ 保存失败")

if __name__ == "__main__":
    main()
