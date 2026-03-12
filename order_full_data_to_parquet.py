#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Order 完整数据处理脚本

该脚本用于处理 Order_完整数据_data.csv、Order_完整数据_data_2024.csv，以及 original 目录下最新的年度文件（如 Order_完整数据_data_2025*.csv）
将其合并、去重并转换为优化的 Parquet 格式

输入文件: 
- original/Order_完整数据_data.csv
- original/Order_完整数据_data_2024.csv
- original/Order_完整数据_data_2025*.csv（选最新的一个）
输出文件: formatted/order_full_data.parquet
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime

# 设置基础路径
BASE_DIR = Path("/Users/zihao_/Documents/coding/dataset")
ORIGINAL_DIR = BASE_DIR / "original"
FORMATTED_DIR = BASE_DIR / "formatted"
OUTPUT_FILE = FORMATTED_DIR / "order_full_data.parquet"

def normalize_owner_cell_phone(series: pd.Series) -> pd.Series:
    s_raw = series.astype("string").str.strip()
    num = pd.to_numeric(s_raw, errors="coerce").round(0)
    num_int = num.astype("Int64")
    s_num = num_int.astype("string")
    s = s_raw.where(num.isna(), s_num)
    s = s.str.strip().str.lower()
    s = s.replace(
        {
            "nan": pd.NA,
            "none": pd.NA,
            "null": pd.NA,
            "": pd.NA,
            "-": pd.NA,
            "无": pd.NA,
        }
    )
    digits = s.str.replace(r"\D", "", regex=True)
    digits = digits.str.replace(r"^0086", "", regex=True)
    digits = digits.str.replace(r"^86", "", regex=True)
    valid = digits.str.match(r"^1\d{10}$", na=False)
    return digits.where(valid, pd.NA).astype("string")

def read_csv_smart(file_path: Path) -> pd.DataFrame:
    """
    智能读取 CSV 文件，尝试多种编码和分隔符
    """
    if not file_path.exists():
        print(f"⚠️ 文件不存在: {file_path}")
        return pd.DataFrame()

    try:
        if file_path.stat().st_size == 0:
            print(f"❌ 文件为空(0字节): {file_path}")
            return pd.DataFrame()
    except Exception:
        pass

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
        'first_assign_time 年/月/日': 'first_assign_time',
        'lead_assign_time_max 年/月/日': 'lead_assign_time_max',
        'Td CountD': 'td_countd',
        'Drive Series Cn': 'drive_series_cn',
        'Main Lead Id': 'main_lead_id',
        'Parent Region Name': 'parent_region_name',
        'Parent_Region_Name': 'parent_region_name',
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

    # 2. 数值列转换
    numeric_cols = ['age', 'invoice_amount', 'td_countd']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. 分类列转换 (优化存储)
    cat_cols = [
        'product_name', 'final_payment_way', 'finance_product', 
        'first_middle_channel_name', 'gender', 'is_hold', 'is_staff',
        'license_city', 'license_city_level', 'license_province',
        'order_type', 'series', 'store_city', 'belong_intent_series',
        'drive_series_cn', 'parent_region_name'
    ]
    
    for col in cat_cols:
        if col in df.columns:
            # 如果唯一值数量较少，转为 category
            if df[col].nunique() < df.shape[0] * 0.5:
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype('string')

    # order_number 应该是字符串
    if 'order_number' in df.columns:
        df['order_number'] = df['order_number'].astype('string')

    if "owner_cell_phone" in df.columns:
        df["owner_cell_phone"] = normalize_owner_cell_phone(df["owner_cell_phone"])

    return df

def main():
    # 1. 按要求收集输入文件：基础文件 + 2024年度 + 2024~当年各年度文件（每年取最新）
    csv_files = []
    
    base_files = [
        ORIGINAL_DIR / "Order_完整数据_data.csv",
        ORIGINAL_DIR / "Order_完整数据_data_2024.csv",
    ]
    for bf in base_files:
        if bf.exists():
            csv_files.append(bf)
    
    current_year = datetime.now().strftime("%Y")
    try:
        current_year_int = int(current_year)
    except Exception:
        current_year_int = datetime.now().year

    year_files = list(ORIGINAL_DIR.glob("Order_完整数据_data_20*.csv"))
    latest_by_year: dict[int, Path] = {}
    for p in year_files:
        name = p.name
        year_part = name.replace("Order_完整数据_data_", "").split(".csv")[0]
        year_str = year_part.split("_")[0]
        if not year_str.isdigit():
            continue
        y = int(year_str)
        if y < 2024 or y > current_year_int:
            continue
        prev = latest_by_year.get(y)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            latest_by_year[y] = p

    for y in sorted(latest_by_year.keys()):
        p = latest_by_year[y]
        if p not in csv_files:
            csv_files.append(p)
    
    if not csv_files:
        print(f"❌ 未在 {ORIGINAL_DIR} 找到所需的输入文件（基础、2024或当年最新）")
        return

    print(f"🔍 找到 {len(csv_files)} 个数据文件，将按以下顺序处理:")
    for f in csv_files:
        print(f"   - {f.name}")
    
    # 2. 读取并合并所有新数据
    dfs = []
    failed_files = []
    for file_path in csv_files:
        df = read_csv_smart(file_path)
        if not df.empty:
            # 清理列名和转换类型
            # 注意：必须在合并前清理列名，以确保列名一致
            df = clean_column_names(df)
            df = convert_types(df)
            dfs.append(df)
        else:
            failed_files.append(file_path.name)
        
    if not dfs:
        print("❌ 没有成功读取到任何数据，退出。")
        return 1
    
    if failed_files:
        print("❌ 以下文件读取失败或为空，已终止更新以避免静默缺数:")
        for n in failed_files:
            print(f"   - {n}")
        return 1
        
    df_new = pd.concat(dfs, ignore_index=True)
    print(f"✅ 所有新数据合并完成: {df_new.shape[0]} 行")

    # 4. 增量更新逻辑
    if OUTPUT_FILE.exists():
        print(f"📚 发现现有 Parquet 文件: {OUTPUT_FILE}")
        try:
            df_existing = pd.read_parquet(OUTPUT_FILE)
            print(f"   现有数据: {df_existing.shape[0]} 行")
            legacy_map = {
                'approve_refund_time_年/月/日': 'approve_refund_time',
                'apply_refund_time_年/月/日': 'apply_refund_time',
                'approve_refund_time 年/月/日': 'approve_refund_time',
                'apply_refund_time 年/月/日': 'apply_refund_time',
                'first_assign_time_年/月/日': 'first_assign_time',
                'lead_assign_time_max_年/月/日': 'lead_assign_time_max',
                'first_assign_time 年/月/日': 'first_assign_time',
                'lead_assign_time_max 年/月/日': 'lead_assign_time_max',
                'Parent Region Name': 'parent_region_name',
                'Parent_Region_Name': 'parent_region_name'
            }
            for old_col, new_col in legacy_map.items():
                if old_col in df_existing.columns:
                    if df_existing[old_col].dtype == 'object':
                        try:
                            s = df_existing[old_col].astype(str)
                            s = s.str.replace('年', '-', regex=False).str.replace('月', '-', regex=False).str.replace('日', '', regex=False)
                            s = s.replace({'nan': None, 'None': None, '': None})
                            df_existing[old_col] = pd.to_datetime(s, errors='coerce')
                        except Exception as e:
                            print(f"      ⚠️ 转换失败: {e}")
                    if new_col in df_existing.columns:
                        df_existing[new_col] = df_existing[new_col].combine_first(df_existing[old_col])
                        df_existing = df_existing.drop(columns=[old_col])
                    else:
                        df_existing = df_existing.rename(columns={old_col: new_col})
            common_cols = list(set(df_existing.columns) & set(df_new.columns))
            new_only = set(df_new.columns) - set(df_existing.columns)
            existing_only = set(df_existing.columns) - set(df_new.columns)
            if new_only or existing_only:
                all_cols = list(set(df_existing.columns) | set(df_new.columns))
                df_existing = df_existing.reindex(columns=all_cols)
                df_new = df_new.reindex(columns=all_cols)
            if 'order_number' in df_new.columns and 'order_number' in df_existing.columns:
                existing_orders = set(df_existing['order_number'].dropna())
                new_orders = set(df_new['order_number'].dropna())
                truly_new_orders = new_orders - existing_orders
                updated_orders = new_orders & existing_orders
                df_final = df_existing[~df_existing['order_number'].isin(updated_orders)].copy()
                df_final = pd.concat([df_final, df_new], ignore_index=True)
            else:
                df_final = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception as e:
            print(f"❌ 读取现有 Parquet 文件失败: {e}")
            print("   将仅使用新数据。")
            df_final = df_new
    else:
        print("📝 未发现现有 Parquet 文件，创建新文件...")
        df_final = df_new

    if 'parent_region_name' in df_final.columns:
        if df_final['parent_region_name'].nunique() < df_final.shape[0] * 0.5:
            df_final['parent_region_name'] = df_final['parent_region_name'].astype('category')
        else:
            df_final['parent_region_name'] = df_final['parent_region_name'].astype('string')

    # 5. 最终去重（以防万一）
    if 'order_number' in df_final.columns:
        print(f"✂️  执行最终去重...")
        before_count = len(df_final)
        # keep='last' 确保保留最后加入的记录（即最新的）
        df_final = df_final.drop_duplicates(subset=['order_number'], keep='last')
        after_count = len(df_final)
        print(f"   去重前: {before_count}, 去重后: {after_count}, 移除: {before_count - after_count}")

    if "owner_cell_phone" in df_final.columns:
        s0 = df_final["owner_cell_phone"].astype("string")
        s0_norm = s0.str.strip().str.lower()
        present_before = int(
            (
                s0_norm.notna()
                & s0_norm.ne("")
                & ~s0_norm.isin(["nan", "none", "null", "-", "无"])
            ).sum()
        )
        cleaned = normalize_owner_cell_phone(s0)
        present_after = int(cleaned.notna().sum())
        removed = present_before - present_after if present_before >= present_after else 0
        df_final["owner_cell_phone"] = cleaned
        print(
            f"📱 owner_cell_phone 清洗汇总: 原始非空 {present_before}, 标准化有效 {present_after}, 剔除 {removed}"
        )
        invalid_mask = cleaned.notna() & cleaned.str.len().ne(11)
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            examples = cleaned[invalid_mask].dropna().value_counts().head(5)
            print(f"⚠️ owner_cell_phone 非11位: {invalid_count}")
            for v, c in examples.items():
                print(f"   - {v}: {int(c)}")
        else:
            print("✅ owner_cell_phone 均为11位或为空")

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
        return 0
    else:
        print("❌ 保存失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
