#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证订单观察数据 Parquet 文件

该脚本用于验证生成的 order_observation_data.parquet 文件
并提供详细的数据描述信息
"""

import pandas as pd
import numpy as np
import os

def verify_parquet_file():
    """
    验证 Parquet 文件并提供数据描述
    """
    parquet_file_path = "/Users/zihao_/Documents/coding/dataset/formatted/order_observation_data.parquet"
    
    try:
        print("🔍 开始验证 Parquet 文件...")
        print(f"📁 文件路径: {parquet_file_path}")
        
        # 1. 检查文件是否存在
        if not os.path.exists(parquet_file_path):
            print("❌ Parquet 文件不存在！")
            return
        
        # 2. 读取 Parquet 文件
        df = pd.read_parquet(parquet_file_path)
        
        # 3. 基本信息
        print("\n" + "="*80)
        print(" 基本数据信息 ")
        print("="*80)
        print(f"📊 数据维度: {df.shape[0]} 行 x {df.shape[1]} 列")
        
        # 4. 文件大小
        file_size = os.path.getsize(parquet_file_path) / (1024 * 1024)  # MB
        print(f"📁 文件大小: {file_size:.2f} MB")
        
        # 5. 数据类型信息
        print("\n" + "="*80)
        print(" 数据类型信息 ")
        print("="*80)
        print(df.dtypes)
        
        # 6. 内存使用情况
        print("\n" + "="*80)
        print(" 内存使用情况 ")
        print("="*80)
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum() / (1024 * 1024)  # MB
        print(f"总内存使用: {total_memory:.2f} MB")
        print("\n各列内存使用:")
        for col, usage in memory_usage.items():
            if col != 'Index':
                usage_mb = usage / (1024 * 1024)
                print(f"{col}: {usage_mb:.3f} MB")
        
        # 7. 空值统计
        print("\n" + "="*80)
        print(" 空值统计 ")
        print("="*80)
        null_counts = df.isnull().sum()
        null_stats = []
        for col, count in null_counts.items():
            percentage = (count / len(df)) * 100
            null_stats.append({
                '字段名': col,
                '空值数量': count,
                '空值比例': f"{percentage:.2f}%"
            })
        
        null_df = pd.DataFrame(null_stats)
        print(null_df.to_string(index=False))
        
        # 8. 数值型字段统计
        print("\n" + "="*80)
        print(" 数值型字段统计 ")
        print("="*80)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe())
        else:
            print("未发现数值型字段")
        
        # 9. 分类字段统计
        print("\n" + "="*80)
        print(" 分类字段统计 ")
        print("="*80)
        categorical_cols = df.select_dtypes(include=['category']).columns
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"\n{col}:")
            print(f"  唯一值数量: {unique_count}")
            if unique_count <= 20:  # 如果唯一值少于等于20个，显示所有值
                value_counts = df[col].value_counts()
                print(f"  值分布:")
                for value, count in value_counts.items():
                    percentage = (count / len(df)) * 100
                    print(f"    {value}: {count} ({percentage:.2f}%)")
            else:
                print(f"  前10个值:")
                value_counts = df[col].value_counts().head(10)
                for value, count in value_counts.items():
                    percentage = (count / len(df)) * 100
                    print(f"    {value}: {count} ({percentage:.2f}%)")
        
        # 10. 日期字段统计
        print("\n" + "="*80)
        print(" 日期字段统计 ")
        print("="*80)
        date_cols = df.select_dtypes(include=['datetime64[ns]']).columns
        for col in date_cols:
            non_null_dates = df[col].dropna()
            if len(non_null_dates) > 0:
                print(f"\n{col}:")
                print(f"  非空记录数: {len(non_null_dates)}")
                print(f"  最早日期: {non_null_dates.min()}")
                print(f"  最晚日期: {non_null_dates.max()}")
                print(f"  日期范围: {(non_null_dates.max() - non_null_dates.min()).days} 天")
            else:
                print(f"\n{col}: 无有效日期数据")
        
        # 11. 数据样本
        print("\n" + "="*80)
        print(" 数据样本 ")
        print("="*80)
        print("前5行数据:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 30)
        print(df.head())
        
        # 重置pandas显示选项
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
        
        print("\n✅ Parquet 文件验证完成！")
        
        return df
        
    except Exception as e:
        print(f"❌ 验证过程中发生错误: {e}")
        raise e

if __name__ == "__main__":
    # 执行验证
    try:
        df = verify_parquet_file()
        print("\n🎉 验证成功完成！")
    except Exception as e:
        print(f"\n💥 验证失败: {e}")