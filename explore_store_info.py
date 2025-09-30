#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
店铺信息数据探索分析

该脚本用于加载和分析 store_info_data.csv 数据，进行基本的描述性统计分析
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# 设置pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)
warnings.filterwarnings('ignore')

def load_data():
    """加载店铺信息数据"""
    data_path = Path("/Users/zihao_/Documents/coding/dataset/original/store_info_data.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    print("📁 正在加载店铺信息数据...")
    
    # 尝试不同的编码方式和分隔符
    encodings = ['utf-16', 'utf-16le', 'utf-8', 'gbk', 'gb2312']
    separators = ['\t', ',', ';', '|']
    
    for encoding in encodings:
        for sep in separators:
            try:
                print(f"   尝试使用 {encoding} 编码，分隔符: '{sep}'...")
                df = pd.read_csv(data_path, encoding=encoding, sep=sep)
                
                # 检查是否成功解析（列数应该大于1）
                if df.shape[1] > 1:
                    print(f"✅ 数据加载成功！使用编码: {encoding}，分隔符: '{sep}'")
                    print(f"✅ 数据形状: {df.shape}")
                    return df
                    
            except UnicodeDecodeError:
                continue
            except Exception as e:
                continue
    
    # 如果所有方法都失败，尝试最后一种方法
    try:
        print("   尝试最后的方法：utf-16编码，制表符分隔...")
        df = pd.read_csv(data_path, encoding='utf-16', sep='\t', on_bad_lines='skip')
        print(f"✅ 数据加载成功！数据形状: {df.shape}")
        return df
    except Exception as e:
        raise ValueError(f"无法使用任何方式读取文件: {e}")

def basic_info_analysis(df):
    """基本信息分析"""
    print("\n" + "="*60)
    print("📊 基本信息分析")
    print("="*60)
    
    print(f"📏 数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"💾 内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n🏷️ 列名和数据类型:")
    print("-" * 40)
    for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
        print(f"{i:2d}. {col:<30} {str(dtype)}")
    
    print(f"\n📋 数据预览 (前5行):")
    print("-" * 40)
    print(df.head())
    
    return df

def missing_values_analysis(df):
    """缺失值分析"""
    print("\n" + "="*60)
    print("🔍 缺失值分析")
    print("="*60)
    
    missing_stats = pd.DataFrame({
        '缺失数量': df.isnull().sum(),
        '缺失比例(%)': (df.isnull().sum() / len(df) * 100).round(2)
    })
    
    missing_stats = missing_stats[missing_stats['缺失数量'] > 0].sort_values('缺失数量', ascending=False)
    
    if len(missing_stats) > 0:
        print("存在缺失值的列:")
        print(missing_stats)
    else:
        print("✅ 数据中没有缺失值")

def numerical_analysis(df):
    """数值型变量分析"""
    print("\n" + "="*60)
    print("📈 数值型变量分析")
    print("="*60)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) == 0:
        print("❌ 没有发现数值型变量")
        return
    
    print(f"📊 发现 {len(numerical_cols)} 个数值型变量:")
    for i, col in enumerate(numerical_cols, 1):
        print(f"{i:2d}. {col}")
    
    print(f"\n📋 数值型变量描述性统计:")
    print("-" * 40)
    desc_stats = df[numerical_cols].describe()
    print(desc_stats)

def categorical_analysis(df):
    """分类变量分析"""
    print("\n" + "="*60)
    print("🏷️ 分类变量分析")
    print("="*60)
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(categorical_cols) == 0:
        print("❌ 没有发现分类变量")
        return
    
    print(f"📊 发现 {len(categorical_cols)} 个分类变量:")
    for i, col in enumerate(categorical_cols, 1):
        print(f"{i:2d}. {col}")
    
    # 分析每个分类变量
    for col in categorical_cols[:10]:  # 最多分析前10个分类变量
        print(f"\n📋 {col} 的取值分布:")
        print("-" * 40)
        
        value_counts = df[col].value_counts()
        print(f"唯一值数量: {df[col].nunique()}")
        print(f"最频繁的值: {value_counts.index[0]} (出现 {value_counts.iloc[0]} 次)")
        
        # 显示前10个最频繁的值
        print("\n前10个最频繁的值:")
        top_values = value_counts.head(10)
        for value, count in top_values.items():
            percentage = (count / len(df)) * 100
            print(f"  {value}: {count} ({percentage:.1f}%)")

def data_quality_check(df):
    """数据质量检查"""
    print("\n" + "="*60)
    print("🔍 数据质量检查")
    print("="*60)
    
    # 检查重复行
    duplicate_rows = df.duplicated().sum()
    print(f"🔄 重复行数量: {duplicate_rows}")
    if duplicate_rows > 0:
        print(f"   重复比例: {duplicate_rows/len(df)*100:.2f}%")
    
    # 检查每列的唯一值数量
    print(f"\n📊 各列唯一值统计:")
    print("-" * 40)
    for col in df.columns:
        unique_count = df[col].nunique()
        unique_ratio = unique_count / len(df) * 100
        print(f"{col:<30} {unique_count:>8} ({unique_ratio:>5.1f}%)")

def generate_summary_report(df):
    """生成总结报告"""
    print("\n" + "="*60)
    print("📋 数据总结报告")
    print("="*60)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"📊 数据概览:")
    print(f"  • 总行数: {len(df):,}")
    print(f"  • 总列数: {len(df.columns)}")
    print(f"  • 数值型变量: {len(numerical_cols)} 个")
    print(f"  • 分类变量: {len(categorical_cols)} 个")
    print(f"  • 缺失值总数: {df.isnull().sum().sum():,}")
    print(f"  • 重复行数: {df.duplicated().sum():,}")
    print(f"  • 内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n🎯 主要发现:")
    
    # 缺失值最多的列
    missing_stats = df.isnull().sum().sort_values(ascending=False)
    if missing_stats.iloc[0] > 0:
        print(f"  • 缺失值最多的列: {missing_stats.index[0]} ({missing_stats.iloc[0]} 个)")
    
    # 唯一值最多的列
    unique_stats = df.nunique().sort_values(ascending=False)
    print(f"  • 唯一值最多的列: {unique_stats.index[0]} ({unique_stats.iloc[0]} 个)")

def main():
    """主函数"""
    try:
        print("🚀 开始店铺信息数据探索分析...")
        
        # 1. 加载数据
        df = load_data()
        
        # 2. 基本信息分析
        df = basic_info_analysis(df)
        
        # 3. 缺失值分析
        missing_values_analysis(df)
        
        # 4. 数值型变量分析
        numerical_analysis(df)
        
        # 5. 分类变量分析
        categorical_analysis(df)
        
        # 6. 数据质量检查
        data_quality_check(df)
        
        # 7. 生成总结报告
        generate_summary_report(df)
        
        print(f"\n🎉 店铺信息数据探索分析完成！")
        
        return df
        
    except Exception as e:
        print(f"❌ 分析过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    df = main()