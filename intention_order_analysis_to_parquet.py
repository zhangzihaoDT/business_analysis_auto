#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意向订单分析数据处理脚本

该脚本用于处理 Intention_Order_Analysis_(Series_6)_data.csv 文件
将其转换为优化的Parquet格式

输入文件: original/Intention_Order_Analysis_(Series_6)_data.csv
输出文件: formatted/intention_order_analysis.parquet
"""

import pandas as pd
import numpy as np
import os
import chardet
import json
from datetime import datetime
from pathlib import Path

def load_processing_metadata(metadata_path):
    """
    加载处理元数据
    """
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  读取元数据文件失败: {e}，将创建新的元数据")
    
    # 返回默认元数据
    return {
        "last_processed_timestamp": None,
        "last_csv_modification_time": None,
        "last_processing_time": None,
        "total_records_processed": 0,
        "processing_history": [],
        "data_version": "1.0.0",
        "schema_version": "1.0.0",
        "incremental_mode": True,
        "notes": "Metadata file for tracking incremental updates of intention_order_analysis data"
    }

def save_processing_metadata(metadata_path, metadata):
    """
    保存处理元数据
    """
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        return True
    except Exception as e:
        print(f"❌ 保存元数据文件失败: {e}")
        return False

def check_csv_modification(csv_path, last_modification_time):
    """
    检查CSV文件是否有修改
    """
    if not os.path.exists(csv_path):
        return False, None
    
    current_mtime = os.path.getmtime(csv_path)
    current_mtime_str = datetime.fromtimestamp(current_mtime).isoformat()
    
    if last_modification_time is None:
        return True, current_mtime_str  # 首次处理
    
    return current_mtime_str != last_modification_time, current_mtime_str

def detect_encoding(file_path):
    """
    检测文件编码
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # 读取前10000字节进行检测
        result = chardet.detect(raw_data)
        return result['encoding'], result['confidence']

def detect_separator(file_path, encoding):
    """
    检测CSV文件的分隔符
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            first_line = f.readline()
            
        # 检测常见分隔符
        separators = [',', '\t', ';', '|']
        separator_counts = {}
        
        for sep in separators:
            count = first_line.count(sep)
            if count > 0:
                separator_counts[sep] = count
        
        if separator_counts:
            # 选择出现次数最多的分隔符
            best_separator = max(separator_counts, key=separator_counts.get)
            print(f"检测到分隔符: '{best_separator}' (出现 {separator_counts[best_separator]} 次)")
            return best_separator
        else:
            print("未检测到明显的分隔符，使用默认逗号")
            return ','
            
    except Exception as e:
        print(f"分隔符检测失败: {e}，使用默认逗号")
        return ','

def read_csv_with_encoding(file_path):
    """
    使用多种编码尝试读取CSV文件，并自动检测分隔符
    """
    # 首先检测文件编码
    encoding, confidence = detect_encoding(file_path)
    print(f"检测到文件编码: {encoding}，置信度: {confidence:.2f}")
    
    # 检测分隔符
    separator = detect_separator(file_path, encoding)
    
    # 尝试使用检测到的编码和分隔符读取
    try:
        df_data = pd.read_csv(file_path, encoding=encoding, sep=separator)
        print(f"使用 {encoding} 编码和 '{separator}' 分隔符成功读取文件")
        print(f"读取到 {df_data.shape[0]} 行 x {df_data.shape[1]} 列")
        return df_data, encoding
    except Exception as e:
        print(f"使用检测到的编码 {encoding} 读取失败，尝试其他编码...")
        
        # 尝试常见编码和分隔符组合
        encodings_to_try = ['utf-16', 'utf-8', 'latin1', 'gbk', 'gb2312', 'gb18030']
        separators_to_try = [separator, ',', '\t', ';', '|']
        
        for enc in encodings_to_try:
            for sep in separators_to_try:
                try:
                    df_data = pd.read_csv(file_path, encoding=enc, sep=sep)
                    if df_data.shape[1] > 1:  # 确保读取到多列
                        print(f"使用 {enc} 编码和 '{sep}' 分隔符成功读取文件")
                        print(f"读取到 {df_data.shape[0]} 行 x {df_data.shape[1]} 列")
                        return df_data, enc
                except:
                    continue
        
        raise Exception("尝试了多种编码和分隔符组合但都失败了")

def analyze_data_structure(df):
    """
    分析数据结构，打印基本信息
    """
    print(f"📊 数据维度: {df.shape[0]} 行 x {df.shape[1]} 列")
    
    # 只显示有大量空值的字段（空值比例>30%）
    high_null_cols = []
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_percentage = (null_count / len(df)) * 100
        if null_percentage > 30:
            high_null_cols.append(f"{col}: {null_count} ({null_percentage:.0f}%)")
    
    if high_null_cols:
        print(f"\n各列空值数量:")
        for col_info in high_null_cols:
            print(col_info)

def clean_and_convert_data(df):
    """
    清理和转换数据类型
    """
    print("\n" + "="*60)
    print(" 开始数据清洗和类型转换 ")
    print("="*60)
    
    df_cleaned = df.copy()
    
    # 0. 重命名列名（简化复杂的列名）
    column_rename_mapping = {
        'DATE(DATETRUNC(\'day\', [Order Create Time]))': 'Order_Create_Time',
        'DATE(DATETRUNC(\'day\', [Intention Payment Time]))': 'Intention_Payment_Time',
        'DATE(DATETRUNC(\'day\', [Deposit Payment Time]))': 'Deposit_Payment_Time',
        'DATE(DATETRUNC(\'day\', [intention_refund_time]))': 'intention_refund_time',
        'DATE(DATETRUNC(\'day\', [deposit_refund_time]))': 'deposit_refund_time',
        'DATE(DATETRUNC(\'day\', [Lock Time]))': 'Lock_Time',
        'DATE(DATETRUNC(\'day\', [first_touch_time]))': 'first_touch_time',
        'DATE([first_assign_time])': 'first_assign_time',
        'DATE(DATETRUNC(\'day\', DATE([Invoice Upload Time])))': 'Invoice_Upload_Time',
        'DATE(DATETRUNC(\'day\', DATE([store_create_date])))': 'store_create_date',
        'NOT ISNULL([Intention Payment Time])': 'Has_Intention_Payment'
    }
    
    # 执行重命名
    df_cleaned.rename(columns=column_rename_mapping, inplace=True)
    print("✅ 已重命名以下列名:")
    for old_name, new_name in column_rename_mapping.items():
        if old_name in df.columns:
            print(f"   {old_name} -> {new_name}")
    
    # 1. 处理日期列（使用重命名后的列名）
    date_columns = [
        'Order_Create_Time',
        'Intention_Payment_Time',
        'Deposit_Payment_Time',
        'intention_refund_time',
        'deposit_refund_time',
        'Lock_Time', 
        'first_touch_time',
        'first_assign_time',
        'Invoice_Upload_Time',
        'store_create_date'
    ]
    
    for col in date_columns:
        if col in df_cleaned.columns:
            try:
                # 尝试转换为日期类型
                df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                print(f"✅ 成功将 {col} 转换为日期类型")
            except Exception as e:
                print(f"❌ 转换 {col} 时出错: {e}")

    # 保留原始四个日期列，不进行合并或删除：
    # Intention_Payment_Time、Deposit_Payment_Time、intention_refund_time、deposit_refund_time
    # 以上列已在日期转换阶段进行类型转换，无需进一步处理
    
    # 2. 处理数值列（Order Number是字符串ID，不应转换为数值）
    numeric_columns = ['buyer_age', 'Order Number 不同计数']
    for col in numeric_columns:
        if col in df_cleaned.columns:
            try:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                print(f"✅ 成功将 {col} 转换为数值类型")
            except Exception as e:
                print(f"❌ 转换 {col} 时出错: {e}")
    
    # 2.1 特殊处理开票价格字段（清理逗号分隔符）
    if '开票价格' in df_cleaned.columns:
        try:
            # 清理逗号分隔符并转换为数值
            df_cleaned['开票价格'] = df_cleaned['开票价格'].astype(str).str.replace(',', '').replace('nan', np.nan)
            df_cleaned['开票价格'] = pd.to_numeric(df_cleaned['开票价格'], errors='coerce')
            print(f"✅ 成功将 开票价格 转换为数值类型（已清理逗号分隔符）")
        except Exception as e:
            print(f"❌ 转换 开票价格 时出错: {e}")
    
    # 3. 处理分类变量（使用重命名后的列名）
    category_columns = [
        '车型分组', 'Order Number', 'order_gender', 'first_main_channel_group',
        'Parent Region Name', 'License Province', 'license_city_level', 
        'License City', 'Has_Intention_Payment'
    ]
    
    for col in category_columns:
        if col in df_cleaned.columns:
            try:
                # 检查唯一值比例，如果小于50%则转换为category
                unique_ratio = df_cleaned[col].nunique() / len(df_cleaned)
                if unique_ratio < 0.5:
                    df_cleaned[col] = df_cleaned[col].astype('category')
                    print(f"✅ 已将 {col} 转换为category类型 (唯一值比例: {unique_ratio:.2%})")
                else:
                    print(f"⚠️  {col} 唯一值比例过高 ({unique_ratio:.2%})，保持为object类型")
            except Exception as e:
                print(f"❌ 转换 {col} 时出错: {e}")
    
    # 4. 强制文本类型的列（避免保存Parquet时被错误推断为数值）
    text_like_patterns = ['Phone', '电话', '手机号', '手机', '电话号']
    explicit_text_columns = ['Store Agent Phone', 'Order Number']
    for col in df_cleaned.columns:
        if col in explicit_text_columns or any(pat in str(col) for pat in text_like_patterns):
            try:
                df_cleaned[col] = df_cleaned[col].astype('string')
                print(f"✅ 已将 {col} 设置为string类型")
            except Exception as e:
                print(f"❌ 设置 {col} 为string类型时出错: {e}")

    return df_cleaned

def optimize_data_types(df):
    """
    优化数据类型以减少内存使用
    """
    df_optimized = df.copy()
    optimized_count = 0
    
    # 对于整数列，尝试使用更小的数据类型（Order Number保持为字符串）
    int_columns = ['buyer_age', 'Order Number 不同计数']
    for col in int_columns:
        if col in df_optimized.columns and df_optimized[col].dtype in ['float64', 'int64']:
            # 检查是否有非空值
            if df_optimized[col].notna().any():
                non_null_data = df_optimized[col].dropna()
                if len(non_null_data) == 0:
                    continue
                    
                min_val = non_null_data.min()
                max_val = non_null_data.max()
                
                # 根据数据范围选择合适的数据类型（使用可空整数类型）
                if min_val >= 0 and max_val <= 255:
                    df_optimized[col] = df_optimized[col].astype('UInt8')
                elif min_val >= -128 and max_val <= 127:
                    df_optimized[col] = df_optimized[col].astype('Int8')
                elif min_val >= 0 and max_val <= 65535:
                    df_optimized[col] = df_optimized[col].astype('UInt16')
                elif min_val >= -32768 and max_val <= 32767:
                    df_optimized[col] = df_optimized[col].astype('Int16')
                elif min_val >= 0 and max_val <= 4294967295:
                    df_optimized[col] = df_optimized[col].astype('UInt32')
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    df_optimized[col] = df_optimized[col].astype('Int32')
                else:
                    df_optimized[col] = df_optimized[col].astype('Int64')
                
                optimized_count += 1
    
    if optimized_count > 0:
        print(f"🔧 已优化 {optimized_count} 个字段的数据类型")
    
    return df_optimized

def process_intention_order_analysis_to_parquet():
    """
    处理意向订单分析数据并转换为Parquet格式（真正的增量更新）
    """
    # 定义文件路径
    input_file_path = "/Users/zihao_/Documents/coding/dataset/original/Intention_Order_Analysis_(Series_6)_data.csv"
    output_dir = "/Users/zihao_/Documents/coding/dataset/formatted/"
    output_file_path = os.path.join(output_dir, "intention_order_analysis.parquet")
    metadata_file_path = os.path.join(output_dir, "processing_metadata.json")
    
    try:
        print("🚀 开始处理意向订单分析数据...")
        
        # 0. 加载处理元数据
        metadata = load_processing_metadata(metadata_file_path)
        
        # 1. 检查CSV文件是否有修改
        has_changes, current_mtime = check_csv_modification(
            input_file_path, 
            metadata.get('last_csv_modification_time')
        )
        
        if not has_changes and metadata.get('incremental_mode', False):
            if os.path.exists(output_file_path):
                df_existing = pd.read_parquet(output_file_path)
                print(f"✅ 源文件未变化，返回现有数据: {df_existing.shape[0]} 行")
                return df_existing, output_file_path
        
        # 2. 检查是否存在历史Parquet文件
        df_existing = None
        processing_mode = "full"  # full: 全量处理, incremental: 增量处理
        
        if os.path.exists(output_file_path):
            try:
                df_existing = pd.read_parquet(output_file_path)
                print(f"📚 历史数据: {df_existing.shape[0]} 行")
                processing_mode = "incremental"
            except Exception as e:
                print(f"❌ 读取历史数据失败: {e}")
                df_existing = None
                processing_mode = "full"
        
        # 2. 读取新的CSV文件
        df_raw, encoding = read_csv_with_encoding(input_file_path)
        print(f"📖 新数据: {df_raw.shape[0]} 行")
        
        # 3. 分析数据结构（仅对新数据）
        analyze_data_structure(df_raw)
        
        # 4. 清理和转换数据类型
        df_cleaned = clean_and_convert_data(df_raw)
        
        # 5. 优化数据类型
        df_new = optimize_data_types(df_cleaned)
        
        # 6. 智能数据处理和合并
        print("\n" + "="*60)
        print(" 智能数据处理和合并 ")
        print("="*60)
        
        if processing_mode == "incremental" and df_existing is not None:
            print(f"📊 增量模式: 智能合并历史数据和新数据...")
            print(f"历史数据: {df_existing.shape[0]} 行")
            print(f"新数据: {df_new.shape[0]} 行")
            
            # 确保两个数据框有相同的列结构
            if set(df_existing.columns) != set(df_new.columns):
                print("⚠️  警告: 历史数据和新数据的列结构不完全一致")
                print(f"历史数据列数: {len(df_existing.columns)}")
                print(f"新数据列数: {len(df_new.columns)}")
                
                # 获取共同列和差异列
                common_cols = list(set(df_existing.columns) & set(df_new.columns))
                existing_only = set(df_existing.columns) - set(df_new.columns)
                new_only = set(df_new.columns) - set(df_existing.columns)
                
                print(f"共同列: {len(common_cols)} 个")
                if existing_only:
                    print(f"历史数据独有列: {list(existing_only)}")
                if new_only:
                    print(f"新数据独有列: {list(new_only)}")
                
                # 智能列对齐：保留所有列，缺失的用NaN填充
                all_cols = list(set(df_existing.columns) | set(df_new.columns))
                
                # 为历史数据添加缺失列
                for col in new_only:
                    df_existing[col] = pd.NA
                
                # 为新数据添加缺失列
                for col in existing_only:
                    df_new[col] = pd.NA
                
                # 重新排序列以保持一致性
                df_existing = df_existing[all_cols]
                df_new = df_new[all_cols]
                
                print(f"✅ 列结构已对齐: {len(all_cols)} 列")
            
            # 智能合并：处理新增和更新的订单记录
            if 'Order Number' in df_new.columns and 'Order Number' in df_existing.columns:
                print(f"🔍 分析订单数据变化...")
                existing_orders = set(df_existing['Order Number'].dropna())
                new_orders = set(df_new['Order Number'].dropna())
                
                truly_new_orders = new_orders - existing_orders
                updated_orders = new_orders & existing_orders
                removed_orders = existing_orders - new_orders
                
                print(f"现有订单数: {len(existing_orders)}")
                print(f"新文件订单数: {len(new_orders)}")
                print(f"真正新增订单: {len(truly_new_orders)}")
                print(f"可能更新的订单: {len(updated_orders)}")
                print(f"可能移除的订单: {len(removed_orders)}")
                
                # 修正的增量更新逻辑：只有在明确检测到是全量导出时才替换
                # 判断标准：如果移除的订单数量超过总订单数的50%，且新增订单很少，可能是全量快照
                total_existing = len(existing_orders)
                removal_ratio = len(removed_orders) / total_existing if total_existing > 0 else 0
                new_ratio = len(truly_new_orders) / len(new_orders) if len(new_orders) > 0 else 0
                
                # 更严格的判断条件：只有在移除比例很高且新增比例很低时才认为是完整快照
                # 同时要求新文件的订单数量显著小于历史数据，这通常表明是数据筛选或时间范围变化
                size_reduction_ratio = len(new_orders) / total_existing if total_existing > 0 else 1
                
                # 更保守的完整快照判断：只有在极端情况下才认为是完整快照
                # 1. 移除比例超过98%（非常高的移除比例）
                # 2. 新增订单比例低于1%（几乎没有新订单）
                # 3. 新文件大小相比历史数据显著减少（小于20%）
                # 4. 新文件订单数量超过5000（确保是大规模数据）
                # 5. 移除的订单数量超过新增订单数量的100倍（避免正常的数据更新被误判）
                is_full_snapshot = (
                    removal_ratio > 0.98 and 
                    new_ratio < 0.01 and 
                    size_reduction_ratio < 0.2 and 
                    len(new_orders) > 5000 and
                    len(removed_orders) > len(truly_new_orders) * 100
                )
                
                if is_full_snapshot:
                    print(f"🔄 检测到完整快照，采用完整替换策略")
                    
                    # 安全检查：如果数据减少过多，要求用户确认
                    data_loss_ratio = 1 - size_reduction_ratio
                    if data_loss_ratio > 0.5:  # 如果数据减少超过50%
                        print(f"⚠️  警告: 新数据相比历史数据减少了 {data_loss_ratio:.1%}")
                    
                    df_final = df_new.copy()
                    print(f"✅ 使用新数据完全替换: {df_final.shape[0]} 行")
                else:
                     print(f"📈 采用增量更新策略 (移除:{removal_ratio:.1%}, 新增:{new_ratio:.1%})")
                     
                     # 开始构建最终数据集
                     df_final = df_existing.copy()
                     
                     if len(truly_new_orders) > 0:
                         # 添加真正的新订单
                         df_new_records = df_new[df_new['Order Number'].isin(truly_new_orders)]
                         print(f"📈 添加 {len(df_new_records)} 条新记录")
                         
                         # 合并数据
                         df_final = pd.concat([df_final, df_new_records], ignore_index=True)
                         print(f"合并后总数据: {df_final.shape[0]} 行")
                     else:
                         print("✅ 未发现新订单")
                
                     # 处理可能的更新记录：基于时间戳和关键字段的智能更新策略（向量化优化）
                     if len(updated_orders) > 0:
                         print(f"🔄 检查 {len(updated_orders)} 个订单的更新...")
                         
                         # 定义关键时间字段
                         time_columns = ['Order_Create_Time', 'Intention_Payment_Time', 'intention_refund_time', 'Lock_Time', 'Invoice_Upload_Time','store_create_date']
                         available_time_cols = [col for col in time_columns if col in df_new.columns and col in df_existing.columns]
                         
                         # 定义关键业务字段（非时间字段）
                         business_columns = ['Product Name', '车型分组', '开票价格', 'buyer_age', 'order_gender', 'License Province', 'License City']
                         available_business_cols = [col for col in business_columns if col in df_new.columns and col in df_existing.columns]
                         
                         # 获取需要更新的订单数据（向量化操作）
                         df_updated_records = df_new[df_new['Order Number'].isin(updated_orders)].copy()
                         df_existing_updated = df_existing[df_existing['Order Number'].isin(updated_orders)].copy()
                         
                         # 设置索引以便快速合并比较
                         df_updated_records = df_updated_records.set_index('Order Number')
                         df_existing_updated = df_existing_updated.set_index('Order Number')
                         
                         # 向量化比较字段
                         orders_to_update = set()
                         update_stats = {}
                         
                         # 检查时间字段更新
                         if available_time_cols:
                             for time_col in available_time_cols:
                                 # 向量化比较：找出有更新的订单
                                 # 确保两个Series有相同的索引
                                 common_orders = df_updated_records.index.intersection(df_existing_updated.index)
                                 
                                 if len(common_orders) > 0:
                                     new_times = df_updated_records.loc[common_orders, time_col]
                                     existing_times = df_existing_updated.loc[common_orders, time_col]
                                     
                                     # 找出新数据不为空且与现有数据不同的订单
                                     has_new_data = pd.notna(new_times)
                                     is_different = (pd.isna(existing_times) | (new_times != existing_times))
                                     needs_update = has_new_data & is_different
                                     
                                     updated_orders_for_col = needs_update[needs_update].index.tolist()
                                     
                                     if updated_orders_for_col:
                                         orders_to_update.update(updated_orders_for_col)
                                         update_stats[time_col] = len(updated_orders_for_col)
                         
                         # 检查业务字段更新
                         if available_business_cols:
                             for business_col in available_business_cols:
                                 # 向量化比较：找出有更新的订单
                                 common_orders = df_updated_records.index.intersection(df_existing_updated.index)
                                 
                                 if len(common_orders) > 0:
                                     new_values = df_updated_records.loc[common_orders, business_col].astype(str)
                                     existing_values = df_existing_updated.loc[common_orders, business_col].astype(str)
                                     
                                     # 找出值不同的订单（包括从空值到有值的情况）
                                     is_different = (new_values != existing_values)
                                     
                                     # 排除两边都是NaN的情况
                                     both_nan = (new_values == 'nan') & (existing_values == 'nan')
                                     needs_update = is_different & ~both_nan
                                     
                                     updated_orders_for_col = needs_update[needs_update].index.tolist()
                                     
                                     if updated_orders_for_col:
                                         orders_to_update.update(updated_orders_for_col)
                                         update_stats[business_col] = len(updated_orders_for_col)
                         
                         if orders_to_update:
                             # 汇总显示更新统计
                             update_summary = ", ".join([f"{col}:{count}个" for col, count in update_stats.items()])
                             print(f"📈 发现 {len(orders_to_update)} 个订单需要更新 ({update_summary})")
                             
                             # 移除旧记录
                             df_final = df_final[~df_final['Order Number'].isin(orders_to_update)]
                             
                             # 添加更新后的记录
                             df_updated_final = df_new[df_new['Order Number'].isin(orders_to_update)]
                             df_final = pd.concat([df_final, df_updated_final], ignore_index=True)
                             
                             print(f"✅ 已更新 {len(orders_to_update)} 个订单的记录")
                         else:
                             print(f"✅ 重复订单无字段更新，保持现有数据")
                         
                         print(f"最终数据: {df_final.shape[0]} 行")
            else:
                print("⚠️  未找到 'Order Number' 列，执行简单合并")
                df_final = pd.concat([df_existing, df_new], ignore_index=True)
                
        else:
            print(f"📝 全量模式: 直接使用新数据")
            df_final = df_new
        
        # 7. 数据质量检查前的类型统一（避免Parquet保存时的混合类型问题）
        safe_string_cols = [
            'Order Number', 'Store Agent Phone', 'Buyer Cell Phone',
            'Store Agent Id', 'Buyer Identity No', 'Store Code'
        ]
        for col in safe_string_cols:
            if col in df_final.columns:
                try:
                    df_final[col] = df_final[col].astype('string')
                    # 打印一次即可，避免刷屏
                    print(f"🔒 已统一 {col} 为string类型以确保写入安全")
                except Exception as e:
                    print(f"⚠️  将 {col} 统一为string类型失败: {e}")

        # 7. 数据质量检查
        print("\n" + "="*60)
        print(" 最终数据质量报告 ")
        print("="*60)
        print(f"最终数据维度: {df_final.shape}")
        
        if 'Order_Create_Time' in df_final.columns:
            print(f"📅 数据时间范围: {df_final['Order_Create_Time'].min()} 到 {df_final['Order_Create_Time'].max()}")
        
        print(f"\n各列数据类型:")
        print(df_final.dtypes)
        
        print(f"\n各列空值数量:")
        null_counts = df_final.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                percentage = (count / len(df_final)) * 100
                print(f"{col}: {count} ({percentage:.2f}%)")
        
        # 8. 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")
        
        # 9. 保存为Parquet文件
        print(f"\n💾 正在保存最终数据...")
        df_final.to_parquet(output_file_path, index=False, engine='pyarrow', compression='snappy')
        
        # 10. 更新处理元数据
        print(f"\n📋 更新处理元数据...")
        current_time = datetime.now().isoformat()
        
        # 更新元数据
        metadata.update({
            'last_processed_timestamp': current_time,
            'last_csv_modification_time': current_mtime,
            'last_processing_time': current_time,
            'total_records_processed': len(df_final),
            'data_version': f"{metadata.get('data_version', '1.0.0')}",
            'processing_mode': processing_mode
        })
        
        # 添加处理历史记录
        processing_record = {
            'timestamp': current_time,
            'mode': processing_mode,
            'input_file_mtime': current_mtime,
            'records_before': len(df_existing) if df_existing is not None else 0,
            'records_after': len(df_final),
            'new_records_added': len(df_final) - (len(df_existing) if df_existing is not None else 0)
        }
        
        if 'processing_history' not in metadata:
            metadata['processing_history'] = []
        metadata['processing_history'].append(processing_record)
        
        # 保持历史记录不超过50条
        if len(metadata['processing_history']) > 50:
            metadata['processing_history'] = metadata['processing_history'][-50:]
        
        # 保存元数据
        if save_processing_metadata(metadata_file_path, metadata):
            print(f"✅ 元数据已更新")
        else:
            print(f"⚠️  元数据更新失败")
        
        # 11. 计算文件大小
        file_size = os.path.getsize(output_file_path) / (1024 * 1024)  # MB
        
        print(f"\n✅ 数据处理完成！")
        print(f"📁 输出文件: {output_file_path}")
        print(f"📊 文件大小: {file_size:.2f} MB")
        print(f"📈 最终数据维度: {df_final.shape[0]} 行 x {df_final.shape[1]} 列")
        print(f"🔧 处理模式: {processing_mode}")
        
        # 12. 显示最终数据样本
        print(f"\n最终数据样本（前5行）:")
        print(df_final.head())
        
        return df_final, output_file_path
        
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")
        raise e

if __name__ == "__main__":
    # 执行数据处理
    try:
        df, output_path = process_intention_order_analysis_to_parquet()
        print("\n🎉 意向订单分析数据处理完成！")
    except Exception as e:
        print(f"\n💥 处理失败: {e}")
