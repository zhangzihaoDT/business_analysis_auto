#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
销售信息数据探索性分析脚本
用于分析 sales_info_data.csv 文件并转换为 JSON 格式
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

def explore_sales_info_data():
    """
    探索性分析销售信息数据
    """
    # 定义文件路径
    input_file = "/Users/zihao_/Documents/coding/dataset/original/sales_info_data.csv"
    output_dir = "/Users/zihao_/Documents/coding/dataset/formatted"
    output_file = os.path.join(output_dir, "sales_info_data.json")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("销售信息数据探索性分析")
    print("=" * 60)
    
    try:
        # 读取CSV文件，尝试不同编码
        print(f"\n正在读取文件: {input_file}")
        
        # 尝试不同的编码方式
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'latin-1']
        df = None
        
        for encoding in encodings:
             try:
                 print(f"尝试使用编码: {encoding}")
                 # 尝试不同的分隔符
                 separators = [',', '\t', ';', '|']
                 for sep in separators:
                     try:
                         df_temp = pd.read_csv(input_file, encoding=encoding, sep=sep)
                         # 检查是否正确解析（列数大于1或者列名不包含分隔符）
                         if df_temp.shape[1] > 1 or not any(s in df_temp.columns[0] for s in ['\t', ',', ';', '|']):
                             df = df_temp
                             print(f"✅ 成功使用编码: {encoding}, 分隔符: '{sep}'")
                             break
                     except:
                         continue
                 if df is not None:
                     break
             except UnicodeDecodeError:
                 print(f"❌ 编码 {encoding} 失败")
                 continue
             except Exception as e:
                 print(f"❌ 使用编码 {encoding} 时发生其他错误: {str(e)}")
                 continue
        
        if df is None:
            raise Exception("无法使用任何编码方式读取文件")
        
        # 基本信息
        print(f"\n📊 数据基本信息:")
        print(f"数据形状: {df.shape}")
        print(f"行数: {df.shape[0]:,}")
        print(f"列数: {df.shape[1]}")
        
        # 列信息
        print(f"\n📋 列信息:")
        print(df.info())
        
        # 数据类型统计
        print(f"\n🔢 数据类型分布:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"{dtype}: {count} 列")
        
        # 缺失值统计
        print(f"\n❌ 缺失值统计:")
        missing_stats = df.isnull().sum()
        missing_percent = (missing_stats / len(df)) * 100
        missing_df = pd.DataFrame({
            '缺失数量': missing_stats,
            '缺失百分比': missing_percent.round(2)
        })
        missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失数量', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df)
        else:
            print("✅ 无缺失值")
        
        # 数值列统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n📈 数值列描述性统计:")
            print(df[numeric_cols].describe())
        
        # 文本列统计
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            print(f"\n📝 文本列信息:")
            for col in text_cols[:10]:  # 只显示前10列
                unique_count = df[col].nunique()
                print(f"{col}: {unique_count} 个唯一值")
                if unique_count <= 20:  # 如果唯一值较少，显示前几个
                    sample_values = df[col].value_counts().head(5)
                    print(f"  前5个值: {list(sample_values.index)}")
        
        # 前几行数据预览
        print(f"\n👀 数据预览 (前5行):")
        print(df.head())
        
        # 描述性分析
        print(f"\n📊 描述性分析:")
        print("=" * 50)
        
        # 1. 统计不同Dealer_type的数量
        print(f"\n🏪 经销商类型(Dealer_type)分布:")
        dealer_type_counts = df['Dealer_type'].value_counts()
        for dealer_type, count in dealer_type_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {dealer_type}: {count:,} 个 ({percentage:.1f}%)")
        
        # 2. 按Dealer Name Fc分组统计
        print(f"\n🏢 按经销商名称(Dealer Name Fc)分组统计:")
        dealer_stats = df.groupby('Dealer Name Fc').agg({
            'Dealer Name Fc': 'count',  # 记录数
            'Member Name': 'nunique',   # 唯一成员姓名数
            'Member Code': 'nunique'    # 唯一成员代码数
        }).rename(columns={
            'Dealer Name Fc': '记录数',
            'Member Name': '成员姓名数',
            'Member Code': '成员代码数'
        }).sort_values('记录数', ascending=False)
        
        print(f"\n📈 经销商统计汇总 (前20名):")
        print(dealer_stats.head(20))
        
        # 3. 经销商统计概览
        print(f"\n📋 经销商统计概览:")
        print(f"  总经销商数量: {len(dealer_stats):,} 个")
        print(f"  平均每个经销商记录数: {dealer_stats['记录数'].mean():.1f}")
        print(f"  平均每个经销商成员数: {dealer_stats['成员姓名数'].mean():.1f}")
        print(f"  最多记录的经销商: {dealer_stats.index[0]} ({dealer_stats.iloc[0]['记录数']} 条记录)")
        print(f"  最少记录的经销商记录数: {dealer_stats['记录数'].min()}")
        
        # 4. 成员统计
        print(f"\n👥 成员统计:")
        total_unique_members = df['Member Name'].nunique()
        total_unique_codes = df['Member Code'].nunique()
        print(f"  总唯一成员姓名数: {total_unique_members:,}")
        print(f"  总唯一成员代码数: {total_unique_codes:,}")
        
        # 检查是否有重复的成员姓名但不同代码
        name_code_mapping = df.groupby('Member Name')['Member Code'].nunique()
        multiple_codes = name_code_mapping[name_code_mapping > 1]
        if len(multiple_codes) > 0:
            print(f"  有多个代码的成员姓名数: {len(multiple_codes)}")
            print(f"  示例: {list(multiple_codes.head(3).index)}")
        
        # 检查是否有重复的成员代码但不同姓名
        code_name_mapping = df.groupby('Member Code')['Member Name'].nunique()
        multiple_names = code_name_mapping[code_name_mapping > 1]
        if len(multiple_names) > 0:
            print(f"  有多个姓名的成员代码数: {len(multiple_names)}")
            print(f"  示例: {list(multiple_names.head(3).index)}")
        
        # 转换为JSON格式
        print(f"\n💾 正在转换为JSON格式...")
        
        # 处理NaN值，转换为None以便JSON序列化
        df_json = df.where(pd.notnull(df), None)
        
        # 转换为字典格式
        data_dict = {
            'metadata': {
                'total_rows': int(df.shape[0]),
                'total_columns': int(df.shape[1]),
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'missing_values': {col: int(count) for col, count in missing_stats.items() if count > 0}
            },
            'data': df_json.to_dict('records')
        }
        
        # 保存为JSON文件
        print(f"正在保存到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 分析完成！")
        print(f"JSON文件已保存到: {output_file}")
        print(f"文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
        
        return data_dict
        
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {input_file}")
        return None
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {str(e)}")
        return None

if __name__ == "__main__":
    result = explore_sales_info_data()
    if result:
        print("\n🎉 脚本执行成功！")
    else:
        print("\n💥 脚本执行失败！")