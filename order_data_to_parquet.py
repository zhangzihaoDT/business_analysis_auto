#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
订单观察数据处理脚本

该脚本用于处理 order_observation_data.csv 文件
将其转换为优化的Parquet格式

输入文件: original/order_observation_data.csv
输出文件: formatted/order_observation_data.parquet
"""

import pandas as pd
import numpy as np
import chardet
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path

def detect_encoding(file_path):
    """
    检测文件编码
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # 读取前10000字节进行检测
        result = chardet.detect(raw_data)
        return result['encoding'], result['confidence']

def read_csv_with_encoding(file_path):
    """
    使用多种编码尝试读取CSV文件
    """
    # 首先检测文件编码
    encoding, confidence = detect_encoding(file_path)
    print(f"检测到文件编码: {encoding}，置信度: {confidence:.2f}")
    
    # 尝试使用检测到的编码读取
    try:
        df_data = pd.read_csv(file_path, encoding=encoding, sep='\t')
        print(f"使用 {encoding} 编码成功读取文件")
        return df_data, encoding
    except Exception as e:
        print(f"使用检测到的编码 {encoding} 读取失败，尝试其他编码...")
        
        # 尝试常见编码
        encodings_to_try = ['utf-16', 'utf-8', 'latin1', 'gbk', 'gb2312', 'gb18030']
        for enc in encodings_to_try:
            try:
                df_data = pd.read_csv(file_path, encoding=enc, sep='\t')
                print(f"使用 {enc} 编码成功读取文件")
                return df_data, enc
            except:
                continue
        
        raise Exception("尝试了多种编码但都失败了")

def analyze_data_structure(df):
    """
    分析数据结构，打印字段名称和格式信息
    """
    print("\n" + "="*80)
    print(" 数据结构分析 ")
    print("="*80)
    
    print(f"\n📊 数据维度: {df.shape[0]} 行 x {df.shape[1]} 列")
    
    print(f"\n📋 字段名称列表:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    print(f"\n📈 字段详细信息:")
    print("-" * 80)
    print(f"{'序号':<4} {'字段名':<35} {'数据类型':<15} {'非空数量':<10} {'空值数量':<10} {'空值比例':<10}")
    print("-" * 80)
    
    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        non_null_count = df[col].count()
        null_count = df[col].isnull().sum()
        null_percentage = (null_count / len(df)) * 100
        
        print(f"{i:<4} {col[:34]:<35} {dtype:<15} {non_null_count:<10} {null_count:<10} {null_percentage:<10.2f}%")
    
    print("\n📝 数据样本（前5行）:")
    print("-" * 120)
    # 显示前5行，但限制列宽以便查看
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    print(df.head())
    
    # 重置pandas显示选项
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
    
    print("\n🔍 数值型字段统计:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    else:
        print("未发现数值型字段")
    
    print("\n📊 分类字段唯一值统计:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols[:10]:  # 只显示前10个分类字段
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} 个唯一值")
        if unique_count <= 10:  # 如果唯一值少于等于10个，显示所有值
            print(f"  值: {df[col].unique().tolist()}")
        else:
            print(f"  前5个值: {df[col].value_counts().head().index.tolist()}")

def split_merged_columns(df_raw):
    """
    分割合并的列（处理制表符分隔的数据）
    """
    if len(df_raw.columns) == 1:
        # 获取第一列的名称
        first_col_name = df_raw.columns[0]
        
        # 分割列名
        column_names = first_col_name.split('\t')
        print(f"分割后的列名: {column_names}")
        
        # 分割数据
        df_split = df_raw[first_col_name].str.split('\t', expand=True)
        
        # 设置列名
        df_split.columns = column_names
        
        return df_split
    else:
        return df_raw

def clean_and_convert_data(df):
    """
    清理和转换数据类型
    """
    print("\n" + "="*60)
    print(" 开始数据清洗和类型转换 ")
    print("="*60)
    
    df_cleaned = df.copy()
    
    # 1. 处理日期列
    date_columns = [
        '日(Intention Payment Time)',
        '日(Order Create Time)',
        '日(Lock Time)', 
        '日(intention_refund_time)',
        '日(Actual Refund Time)',
        'DATE([Invoice Upload Time])',
        'DATE([first_assign_time])'
    ]
    
    for col in date_columns:
        if col in df_cleaned.columns:
            try:
                # 使用新的中文日期解析函数
                df_cleaned[col] = df_cleaned[col].apply(parse_chinese_date)
                
                # 统计转换结果
                valid_dates = df_cleaned[col].notna().sum()
                total_rows = len(df_cleaned)
                success_rate = (valid_dates / total_rows) * 100
                
                print(f"✅ 成功将 {col} 转换为日期类型 (有效日期: {valid_dates}/{total_rows}, {success_rate:.2f}%)")
            except Exception as e:
                print(f"❌ 转换 {col} 时出错: {e}")
    
    # 2. 处理数值列
    numeric_columns = ['buyer_age', '平均值 Origin Amount', '平均值 开票价格', '平均值 折扣率', 'Order Number 不同计数']
    for col in numeric_columns:
        if col in df_cleaned.columns:
            try:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                print(f"✅ 成功将 {col} 转换为数值类型")
            except Exception as e:
                print(f"❌ 转换 {col} 时出错: {e}")
    
    # 3. 处理分类变量
    category_columns = [
        '车型分组', 'pre_vehicle_model_type', 'Product Name', 
        'sales_loyalty_type', 'order_gender', 'Province Name', 
        'License City', 'license_city_level', 'Parent Region Name',
        'first_middle_channel_name'
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
    
    return df_cleaned

def standardize_columns(df):
    """
    标准化列名和数据结构
    """
    print("\n" + "="*60)
    print(" 开始列名标准化 ")
    print("="*60)
    
    df_standardized = df.copy()
    
    # 打印当前列名
    print(f"当前数据列数: {len(df_standardized.columns)}")
    print("当前列名:")
    for i, col in enumerate(df_standardized.columns, 1):
        print(f"{i:2d}. {col}")
    
    return df_standardized

def optimize_data_types(df):
    """
    优化数据类型以减少内存使用
    """
    print("\n" + "="*60)
    print(" 开始数据类型优化 ")
    print("="*60)
    
    df_optimized = df.copy()
    
    # 对于整数列，尝试使用更小的数据类型
    int_columns = ['buyer_age', '度量值']
    for col in int_columns:
        if col in df_optimized.columns and df_optimized[col].dtype in ['float64', 'int64']:
            try:
                # 检查是否有非空值
                if df_optimized[col].notna().any():
                    non_null_data = df_optimized[col].dropna()
                    if len(non_null_data) == 0:
                        continue
                    
                    # 检查是否包含小数部分
                    has_decimals = (non_null_data % 1 != 0).any()
                    if has_decimals:
                        print(f"⚠️  {col} 包含小数值，保持为float64类型")
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
                    
                    print(f"✅ 已优化 {col} 的数据类型为: {df_optimized[col].dtype}")
            except Exception as e:
                print(f"❌ 优化 {col} 数据类型时出错: {e}，保持原类型")
    
    return df_optimized

def parse_chinese_date(date_str):
    """解析中文日期格式，如'2025年8月25日'"""
    if pd.isna(date_str) or date_str == 'nan':
        return pd.NaT
    try:
        # 处理中文日期格式
        if '年' in str(date_str) and '月' in str(date_str) and '日' in str(date_str):
            date_str = str(date_str).replace('年', '-').replace('月', '-').replace('日', '')
            return pd.to_datetime(date_str)
        else:
            return pd.to_datetime(date_str, errors='coerce')
    except:
        return pd.NaT

def get_file_hash(file_path):
    """
    计算文件的MD5哈希值
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"❌ 计算文件哈希值失败: {e}")
        return None

def load_file_tracking_info(tracking_file_path):
    """
    加载文件跟踪信息
    """
    if os.path.exists(tracking_file_path):
        try:
            with open(tracking_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  加载文件跟踪信息失败: {e}，将创建新的跟踪文件")
    return {}

def save_file_tracking_info(tracking_file_path, tracking_info):
    """
    保存文件跟踪信息
    """
    try:
        with open(tracking_file_path, 'w', encoding='utf-8') as f:
            json.dump(tracking_info, f, ensure_ascii=False, indent=2)
        print(f"✅ 文件跟踪信息已保存到: {tracking_file_path}")
    except Exception as e:
        print(f"❌ 保存文件跟踪信息失败: {e}")

def check_file_updates(input_dir, tracking_file_path):
    """
    检查文件更新，返回需要处理的文件列表
    """
    print("\n" + "="*60)
    print(" 检查文件更新 ")
    print("="*60)
    
    # 加载现有的跟踪信息
    tracking_info = load_file_tracking_info(tracking_file_path)
    
    # 获取所有CSV文件
    csv_files = list(Path(input_dir).glob("*.csv"))
    files_to_process = []
    
    for csv_file in csv_files:
        file_path = str(csv_file)
        file_name = csv_file.name
        
        # 计算当前文件哈希值
        current_hash = get_file_hash(file_path)
        if current_hash is None:
            continue
            
        # 检查是否需要处理
        if file_name not in tracking_info:
            print(f"🆕 发现新文件: {file_name}")
            files_to_process.append(file_path)
        elif tracking_info[file_name].get('hash') != current_hash:
            print(f"🔄 文件已更新: {file_name}")
            files_to_process.append(file_path)
        else:
            print(f"✅ 文件无变化: {file_name}")
    
    if not files_to_process:
        print("📋 所有文件都是最新的，无需处理")
    else:
        print(f"📊 需要处理 {len(files_to_process)} 个文件")
    
    return files_to_process, tracking_info

def process_single_file(file_path, output_dir):
    """
    处理单个CSV文件并转换为Parquet格式
    """
    file_name = Path(file_path).stem
    print(f"\n🔄 开始处理文件: {file_name}")
    
    try:
        # 1. 读取CSV文件
        df_raw, encoding = read_csv_with_encoding(file_path)
        print(f"原始数据维度: {df_raw.shape}")
        
        # 2. 分割合并的列
        df_data = split_merged_columns(df_raw)
        print(f"分割后数据维度: {df_data.shape}")
        
        # 3. 数据透视处理（如果需要）
        if '度量名称' in df_data.columns and '度量值' in df_data.columns:
            df_pivoted = pivot_metrics_data(df_data)
        else:
            print("⚠️  未发现度量名称和度量值列，跳过透视处理")
            df_pivoted = df_data
        
        # 4. 标准化列名
        df_standardized = standardize_columns(df_pivoted)
        
        # 5. 清理和转换数据类型
        df_cleaned = clean_and_convert_data(df_standardized)
        
        # 6. 优化数据类型
        df_optimized = optimize_data_types(df_cleaned)
        
        # 7. 保存为Parquet文件
        output_file_path = os.path.join(output_dir, f"{file_name}.parquet")
        df_optimized.to_parquet(output_file_path, index=False, engine='pyarrow', compression='snappy')
        
        # 8. 计算文件大小
        file_size = os.path.getsize(output_file_path) / (1024 * 1024)  # MB
        
        print(f"✅ 文件处理完成: {file_name}")
        print(f"📁 输出文件: {output_file_path}")
        print(f"📊 文件大小: {file_size:.2f} MB")
        print(f"📈 数据维度: {df_optimized.shape[0]} 行 x {df_optimized.shape[1]} 列")
        
        return df_optimized, output_file_path
        
    except Exception as e:
        print(f"❌ 处理文件 {file_name} 时发生错误: {e}")
        return None, None

def merge_parquet_files(parquet_files, output_file_path):
    """
    合并多个Parquet文件为一个完整的文件
    """
    print("\n" + "="*60)
    print(" 合并Parquet文件 ")
    print("="*60)
    
    if not parquet_files:
        print("⚠️  没有Parquet文件需要合并")
        return None
    
    try:
        # 读取所有Parquet文件
        dataframes = []
        total_rows = 0
        
        for parquet_file in parquet_files:
            if os.path.exists(parquet_file):
                df = pd.read_parquet(parquet_file)
                dataframes.append(df)
                total_rows += len(df)
                print(f"📁 已加载: {Path(parquet_file).name} ({len(df)} 行)")
            else:
                print(f"⚠️  文件不存在: {parquet_file}")
        
        if not dataframes:
            print("❌ 没有有效的Parquet文件可以合并")
            return None
        
        # 合并所有数据框
        print(f"🔄 正在合并 {len(dataframes)} 个数据文件...")
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        # 去重处理（基于Order Number）
        if 'Order Number' in merged_df.columns:
            original_count = len(merged_df)
            merged_df = merged_df.drop_duplicates(subset=['Order Number'], keep='last')
            deduplicated_count = len(merged_df)
            removed_count = original_count - deduplicated_count
            
            if removed_count > 0:
                print(f"🔄 已去除 {removed_count} 个重复订单")
            print(f"📊 合并后数据维度: {merged_df.shape[0]} 行 x {merged_df.shape[1]} 列")
        
        # 保存合并后的文件
        merged_df.to_parquet(output_file_path, index=False, engine='pyarrow', compression='snappy')
        
        # 计算文件大小
        file_size = os.path.getsize(output_file_path) / (1024 * 1024)  # MB
        
        print(f"✅ 文件合并完成！")
        print(f"📁 输出文件: {output_file_path}")
        print(f"📊 文件大小: {file_size:.2f} MB")
        print(f"📈 最终数据维度: {merged_df.shape[0]} 行 x {merged_df.shape[1]} 列")
        
        return merged_df
        
    except Exception as e:
        print(f"❌ 合并文件时发生错误: {e}")
        return None

def process_order_data_to_parquet(input_dir=None, output_dir=None):
    """
    批量处理订单观察数据并转换为Parquet格式
    
    Args:
        input_dir (str): 输入目录路径，包含多个CSV文件
        output_dir (str): 输出目录路径
    
    Returns:
        tuple: (合并后的DataFrame, 最终输出文件路径)
    """
    # 默认路径设置
    if input_dir is None:
        input_dir = "/Users/zihao_/Documents/coding/dataset/original/order_observation_data/"
    
    if output_dir is None:
        output_dir = "/Users/zihao_/Documents/coding/dataset/formatted/"
    
    # 文件跟踪信息路径
    tracking_file_path = os.path.join(output_dir, "file_tracking.json")
    
    # 最终合并文件路径
    final_output_file_path = os.path.join(output_dir, "order_observation_data_merged.parquet")
    
    try:
        print("🚀 开始批量处理订单观察数据...")
        print(f"📁 输入目录: {input_dir}")
        print(f"📁 输出目录: {output_dir}")
        
        # 1. 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")
        
        # 2. 检查文件更新
        files_to_process, tracking_info = check_file_updates(input_dir, tracking_file_path)
        
        if not files_to_process:
            print("\n📋 所有文件都是最新的，无需处理")
            # 检查是否存在最终合并文件
            if os.path.exists(final_output_file_path):
                print(f"✅ 最终合并文件已存在: {final_output_file_path}")
                return pd.read_parquet(final_output_file_path), final_output_file_path
            else:
                print("⚠️  最终合并文件不存在，将重新合并现有的Parquet文件")
        
        # 3. 处理需要更新的文件
        processed_files = []
        for file_path in files_to_process:
            df_processed, output_file_path = process_single_file(file_path, output_dir)
            if df_processed is not None and output_file_path is not None:
                processed_files.append(output_file_path)
                
                # 更新跟踪信息
                file_name = Path(file_path).name
                file_hash = get_file_hash(file_path)
                tracking_info[file_name] = {
                    'hash': file_hash,
                    'processed_time': datetime.now().isoformat(),
                    'output_file': output_file_path
                }
        
        # 4. 保存跟踪信息
        if processed_files:
            save_file_tracking_info(tracking_file_path, tracking_info)
        
        # 5. 获取所有现有的Parquet文件进行合并
        all_parquet_files = []
        for file_info in tracking_info.values():
            output_file = file_info.get('output_file')
            if output_file and os.path.exists(output_file):
                all_parquet_files.append(output_file)
        
        # 6. 合并所有Parquet文件
        if all_parquet_files:
            merged_df = merge_parquet_files(all_parquet_files, final_output_file_path)
            
            if merged_df is not None:
                # 7. 数据质量检查
                print("\n" + "="*60)
                print(" 最终数据质量报告 ")
                print("="*60)
                print(f"最终数据维度: {merged_df.shape}")
                print(f"\n各列数据类型:")
                print(merged_df.dtypes)
                
                print(f"\n各列空值数量:")
                null_counts = merged_df.isnull().sum()
                for col, count in null_counts.items():
                    if count > 0:
                        percentage = (count / len(merged_df)) * 100
                        print(f"{col}: {count} ({percentage:.2f}%)")
                
                # 8. 显示最终数据样本
                print(f"\n最终数据样本（前5行）:")
                print(merged_df.head())
                
                return merged_df, final_output_file_path
            else:
                print("❌ 文件合并失败")
                return None, None
        else:
            print("❌ 没有找到可合并的Parquet文件")
            return None, None
        
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")
        raise e

if __name__ == "__main__":
    # 执行数据处理
    try:
        df, output_path = process_order_data_to_parquet()
        print("\n🎉 订单观察数据处理完成！")
    except Exception as e:
        print(f"\n💥 处理失败: {e}")