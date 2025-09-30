#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
线索结构分析数据处理脚本

该脚本用于处理 leads_structure_analysis.xlsx 文件
将其转换为优化的Parquet格式

输入文件: original/leads_structure_analysis.xlsx
输出文件: formatted/leads_structure_analysis.parquet
"""

import pandas as pd
import numpy as np
import os
import chardet
from datetime import datetime

def detect_encoding(file_path):
    """
    检测文件编码
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # 读取前10000字节进行检测
        result = chardet.detect(raw_data)
        return result['encoding'], result['confidence']

def read_excel_file(file_path):
    """
    读取Excel文件
    处理leads_structure_analysis.xlsx文件
    """
    try:
        # 读取Excel文件，尝试第一个工作表
        df_data = pd.read_excel(file_path, sheet_name=0)
        print(f"成功读取Excel文件")
        print(f"列名: {list(df_data.columns)}")
        return df_data
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        
        # 尝试读取所有工作表，看看有哪些可用
        try:
            excel_file = pd.ExcelFile(file_path)
            print(f"Excel文件包含的工作表: {excel_file.sheet_names}")
            
            # 尝试读取第一个工作表
            if len(excel_file.sheet_names) > 0:
                df_data = pd.read_excel(file_path, sheet_name=excel_file.sheet_names[0])
                print(f"成功读取工作表: {excel_file.sheet_names[0]}")
                return df_data
        except Exception as e2:
            print(f"尝试读取工作表也失败: {e2}")
        
        raise Exception("无法读取Excel文件")

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

def clean_and_convert_data(df):
    """
    清理和转换数据类型
    """
    print("\n" + "="*60)
    print(" 开始数据清洗和类型转换 ")
    print("="*60)
    
    df_cleaned = df.copy()
    
    # 1. 处理日期列
    date_column = '日(lc_create_time)'
    if date_column in df_cleaned.columns:
        try:
            # 处理中文日期格式（如：2023年8月24日）
            # 先将中文日期转换为标准格式
            def convert_chinese_date(date_str):
                if pd.isna(date_str) or date_str == '':
                    return None
                try:
                    # 移除'年'、'月'、'日'字符，并替换为标准分隔符
                    date_str = str(date_str).replace('年', '-').replace('月', '-').replace('日', '')
                    return pd.to_datetime(date_str, format='%Y-%m-%d')
                except:
                    return None
            
            df_cleaned[date_column] = df_cleaned[date_column].apply(convert_chinese_date)
            print(f"✅ 成功将 {date_column} 转换为日期类型（处理中文格式）")
            
            # 显示转换后的日期样本
            valid_dates = df_cleaned[date_column].dropna()
            if len(valid_dates) > 0:
                print(f"   转换后的日期样本: {valid_dates.head().tolist()}")
            else:
                print(f"   ⚠️ 警告: 没有成功转换的日期")
                
        except Exception as e:
            print(f"❌ 转换 {date_column} 时出错: {e}")
    
    # 2. 处理数值列（除了日期列外的所有列都应该是数值型）
    numeric_columns = [col for col in df_cleaned.columns if col != date_column]
    
    for col in numeric_columns:
        if col in df_cleaned.columns:
            try:
                # 处理包含逗号的数字（如 "4,399"）
                if df_cleaned[col].dtype == 'object':
                    # 移除逗号并转换为数值
                    df_cleaned[col] = df_cleaned[col].astype(str).str.replace(',', '').replace('', '0')
                
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                print(f"✅ 成功将 {col} 转换为数值类型")
            except Exception as e:
                print(f"❌ 转换 {col} 时出错: {e}")
    
    return df_cleaned

def optimize_data_types(df):
    """
    优化数据类型以减少内存使用
    """
    print("\n" + "="*60)
    print(" 开始数据类型优化 ")
    print("="*60)
    
    df_optimized = df.copy()
    
    # 对于数值列，尝试使用更小的数据类型
    for col in df_optimized.columns:
        if col != '日(lc_create_time)' and df_optimized[col].dtype in ['float64', 'int64']:
            # 检查是否有非空值
            if df_optimized[col].notna().any():
                non_null_data = df_optimized[col].dropna()
                if len(non_null_data) == 0:
                    continue
                    
                min_val = non_null_data.min()
                max_val = non_null_data.max()
                
                # 检查是否包含小数（比例字段）
                has_decimals = (non_null_data % 1 != 0).any()
                
                if has_decimals or '比例' in col or '率' in col:
                    # 对于比例或包含小数的字段，使用float32
                    df_optimized[col] = df_optimized[col].astype('float32')
                    print(f"✅ 已优化 {col} 的数据类型为: {df_optimized[col].dtype} (保持浮点数)")
                else:
                    # 对于整数字段，根据数据范围选择合适的数据类型（使用可空整数类型）
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
    
    return df_optimized

def process_leads_structure_analysis_to_parquet():
    """
    处理线索结构分析数据并转换为Parquet格式
    """
    # 定义文件路径
    input_file_path = "/Users/zihao_/Documents/coding/dataset/original/leads_structure_analysis.xlsx"
    output_dir = "/Users/zihao_/Documents/coding/dataset/formatted/"
    output_file_path = os.path.join(output_dir, "leads_structure_analysis.parquet")
    
    try:
        print("🚀 开始处理线索结构分析数据...")
        print(f"📁 输入文件: {input_file_path}")
        print(f"📁 输出目录: {output_dir}")
        
        # 1. 读取Excel文件
        df_raw = read_excel_file(input_file_path)
        print(f"\n原始数据维度: {df_raw.shape}")
        
        # 2. 分析数据结构
        analyze_data_structure(df_raw)
        
        # 3. 清理和转换数据类型
        df_cleaned = clean_and_convert_data(df_raw)
        
        # 4. 优化数据类型
        df_optimized = optimize_data_types(df_cleaned)
        
        # 5. 数据质量检查
        print("\n" + "="*60)
        print(" 数据质量报告 ")
        print("="*60)
        print(f"最终数据维度: {df_optimized.shape}")
        print(f"\n各列数据类型:")
        print(df_optimized.dtypes)
        
        print(f"\n各列空值数量:")
        null_counts = df_optimized.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                percentage = (count / len(df_optimized)) * 100
                print(f"{col}: {count} ({percentage:.2f}%)")
        
        # 6. 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")
        
        # 7. 保存为Parquet文件
        df_optimized.to_parquet(output_file_path, index=False, engine='pyarrow', compression='snappy')
        
        # 8. 计算文件大小
        file_size = os.path.getsize(output_file_path) / (1024 * 1024)  # MB
        
        print(f"\n✅ 数据处理完成！")
        print(f"📁 输出文件: {output_file_path}")
        print(f"📊 文件大小: {file_size:.2f} MB")
        print(f"📈 数据维度: {df_optimized.shape[0]} 行 x {df_optimized.shape[1]} 列")
        
        # 9. 显示最终数据样本
        print(f"\n最终数据样本（前5行）:")
        print(df_optimized.head())
        
        return df_optimized, output_file_path
        
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")
        raise e

if __name__ == "__main__":
    # 执行数据处理
    try:
        df, output_path = process_leads_structure_analysis_to_parquet()
        print("\n🎉 线索结构分析数据处理完成！")
    except Exception as e:
        print(f"\n💥 处理失败: {e}")