#!/usr/bin/env python3
"""
CM2配置数据转置处理脚本

该脚本读取export_cm2_configuration_data.py输出的原始数据，
进行数据清洗、类型转换，然后对Attribute Name/Code进行转置处理，
形成Attribute Name X Value Display Name的数据结构。

主要功能：
1. 数据清洗和类型转换：
   - 日期字段：将DATE([Order Lock Time])重命名为lock_time并转换为datetime格式
   - 数值字段：将智能冷暖双用冰箱(OP-FRIDGE)的"是"/"否"转换为1/0
   - 类别字段：将相关字段转换为category类型以优化内存使用
2. 数据转置：将属性名转换为列，属性值作为单元格值
3. 兼容新旧数据格式：支持Attribute Name和Attribute Code两种字段名

作者: AI Assistant
创建时间: 2024-10-23
更新时间: 2024-10-23 (添加数据清洗功能)
"""

import pandas as pd
import argparse
import os
import sys
from datetime import datetime
import logging
import numpy as np

def setup_logging(log_level='INFO'):
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def validate_input_file(file_path):
    """验证输入文件是否存在且可读"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"输入文件不存在: {file_path}")
    
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"无法读取输入文件: {file_path}")
    
    # 检查文件大小
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise ValueError(f"输入文件为空: {file_path}")
    
    logging.info(f"输入文件验证通过: {file_path} (大小: {file_size:,} 字节)")

def load_data(file_path):
    """加载CSV数据"""
    try:
        logging.info(f"正在加载数据: {file_path}")
        df = pd.read_csv(file_path)
        logging.info(f"数据加载成功，共 {len(df):,} 行，{len(df.columns)} 列")
        
        # 显示数据基本信息
        logging.info(f"列名: {list(df.columns)}")
        logging.info(f"数据形状: {df.shape}")
        
        # ---------------------------------------------------------------------
        # 1. 字段标准化映射（修复 Tableau 导出字段名不一致问题）
        # ---------------------------------------------------------------------
        # 映射规则：{原始可能出现的列名: 统一后的标准列名}
        column_mapping = {
            # 属性名
            'Attribute Code': 'Attribute Name',
            # 属性值（修复拼写错误 Value Dispaly Name -> Value Display Name）
            'Value Dispaly Name': 'Value Display Name',
            # 订单号
            'Order Number': 'order_number',
            # 锁单时间
            'DATE([Order Lock Time])': 'lock_time',
            # 发票上传时间
            'DATE([invoice_upload_time])': 'invoice_time'
        }
        
        # 执行重命名
        df = df.rename(columns=column_mapping)
        logging.info(f"已执行字段标准化映射，当前列名: {list(df.columns)}")

        # ---------------------------------------------------------------------
        # 2. 检查必要的列是否存在
        # ---------------------------------------------------------------------
        # 核心列：属性名、属性值、订单号
        # 注意：lock_time 虽然重要，但在某些未锁单数据中可能缺失，这里先作为可选或在清洗阶段处理
        # 但脚本核心逻辑依赖 order_number 作为主键，Attribute Name/Value 作为转置对象
        
        required_columns = ['Attribute Name', 'Value Display Name', 'order_number']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"当前数据列: {list(df.columns)}")
            raise ValueError(f"缺少必要的列: {missing_columns}")

        return df
    except Exception as e:
        logging.error(f"加载数据失败: {e}")
        raise

def analyze_data_structure(df):
    """分析数据结构"""
    logging.info("=== 数据结构分析 ===")
    
    # 分析Attribute Name的分布
    attribute_counts = df['Attribute Name'].value_counts()
    logging.info(f"不同的Attribute Name数量: {len(attribute_counts)}")
    logging.info("Attribute Name分布:")
    for attr, count in attribute_counts.items():
        logging.info(f"  {attr}: {count:,} 条记录")
    
    # 分析Value Display Name的分布
    value_counts = df['Value Display Name'].value_counts()
    logging.info(f"不同的Value Display Name数量: {len(value_counts)}")
    logging.info("Value Display Name分布:")
    for value, count in value_counts.head(10).items():
        logging.info(f"  {value}: {count:,} 条记录")
    
    # 分析订单数量
    order_count = df['order_number'].nunique()
    logging.info(f"不同订单数量: {order_count:,}")
    
    return attribute_counts, value_counts

def clean_and_convert_data(df):
    """
    数据清洗和类型转换
    """
    logging.info("开始数据清洗和类型转换...")
    
    # 创建数据副本避免修改原始数据
    df_cleaned = df.copy()
    
    # 1. 日期字段转换
    logging.info("转换日期字段...")
    
    # 处理DATE([Order Lock Time])字段
    if 'DATE([Order Lock Time])' in df_cleaned.columns:
        df_cleaned = df_cleaned.rename(columns={'DATE([Order Lock Time])': 'lock_time'})
        
        # 转换为datetime格式
        try:
            df_cleaned['lock_time'] = pd.to_datetime(df_cleaned['lock_time'], format='%Y/%m/%d')
            logging.info(f"lock_time字段转换成功，格式: {df_cleaned['lock_time'].dtype}")
        except Exception as e:
            logging.warning(f"lock_time转换失败，保持原格式: {e}")
    
    # 处理DATE([invoice_upload_time])字段
    if 'DATE([invoice_upload_time])' in df_cleaned.columns:
        df_cleaned = df_cleaned.rename(columns={'DATE([invoice_upload_time])': 'invoice_time'})
        
        # 转换为datetime格式
        try:
            df_cleaned['invoice_time'] = pd.to_datetime(df_cleaned['invoice_time'], format='%Y/%m/%d')
            logging.info(f"invoice_time字段转换成功，格式: {df_cleaned['invoice_time'].dtype}")
            logging.info(f"invoice_time非空值数量: {df_cleaned['invoice_time'].notna().sum()}")
        except Exception as e:
            logging.warning(f"invoice_time转换失败，保持原格式: {e}")
    
    # 2. 开票价格字段转换
    if '开票价格' in df_cleaned.columns:
        logging.info("转换开票价格字段...")
        try:
            # 移除千位分隔符并转换为数值
            df_cleaned['开票价格'] = df_cleaned['开票价格'].astype(str).str.replace(',', '').astype(float)
            logging.info(f"开票价格字段转换成功，格式: {df_cleaned['开票价格'].dtype}")
            logging.info(f"开票价格范围: {df_cleaned['开票价格'].min()} - {df_cleaned['开票价格'].max()}")
        except Exception as e:
            logging.warning(f"开票价格转换失败，保持原格式: {e}")
    
    # 3. 数值字段转换：将特定属性的"是"/"否"转换为1/0
    logging.info("转换数值字段...")
    
    # 定义需要转换为数值的属性映射
    numeric_attributes = {
        'OP-FRIDGE': '智能冷暖双用冰箱',  # 新数据格式
        '智能冷暖双用冰箱': '智能冷暖双用冰箱',  # 旧数据格式兼容
        '副驾屏幕及零重力座椅': '副驾屏幕及零重力座椅'  # 旧数据格式
    }
    
    # 转换"是"/"否"为1/0
    yes_no_mapping = {'是': 1, '否': 0}
    
    for attr_code, attr_name in numeric_attributes.items():
        mask = df_cleaned['Attribute Name'] == attr_code
        if mask.any():
            logging.info(f"转换属性 '{attr_code}' ({attr_name}) 的是/否值为1/0")
            df_cleaned.loc[mask, 'Value Display Name'] = df_cleaned.loc[mask, 'Value Display Name'].map(yes_no_mapping).fillna(df_cleaned.loc[mask, 'Value Display Name'])
            
            # 统计转换结果
            converted_count = mask.sum()
            logging.info(f"  转换了 {converted_count} 条记录")
    
    # 4. 类别字段转换
    logging.info("转换类别字段...")
    
    # 将Product_Types转换为category类型
    if 'Product_Types' in df_cleaned.columns:
        df_cleaned['Product_Types'] = df_cleaned['Product_Types'].astype('category')
        logging.info(f"Product_Types转换为category类型，类别: {df_cleaned['Product_Types'].cat.categories.tolist()}")
    
    # 将Product Name转换为category类型
    if 'Product Name' in df_cleaned.columns:
        df_cleaned['Product Name'] = df_cleaned['Product Name'].astype('category')
        logging.info(f"Product Name转换为category类型，共 {len(df_cleaned['Product Name'].cat.categories)} 个类别")
    
    # 将Attribute Name转换为category类型
    df_cleaned['Attribute Name'] = df_cleaned['Attribute Name'].astype('category')
    logging.info(f"Attribute Name转换为category类型，类别: {df_cleaned['Attribute Name'].cat.categories.tolist()}")
    
    logging.info("数据清洗和类型转换完成")
    return df_cleaned

def select_order_level_columns(df, candidate_columns):
    """自动检测订单级恒定列（每个order_number只有一个唯一值的列）。

    仅保留在同一订单内不随属性行变化的元数据列，避免将属性行级字段错误并入订单级表。
    """
    try:
        # 计算每个订单、每列的唯一值数量
        nunique_per_order = df.groupby('order_number')[candidate_columns].nunique(dropna=True)
        # 选择在所有订单中最大唯一值<=1的列（即每个订单至多一个值）
        const_cols_mask = (nunique_per_order.max(axis=0) <= 1)
        const_cols = [col for col, is_const in const_cols_mask.items() if bool(is_const)]
        return const_cols
    except Exception:
        # 兜底：若检测失败，返回候选列原样
        return candidate_columns


def transpose_data(df, include_all_meta=False):
    """
    对数据进行转置处理
    将Attribute Name转置为列，Value Display Name作为值
    
    修复版本：使用order_number作为主键，避免因多字段组合导致的数据丢失
    """
    logging.info("开始数据转置处理...")
    
    # 兜底重命名（即使用户使用 --skip-cleaning，也确保日期字段统一）
    try:
        # 将发票上传日期重命名为 invoice_time（如果存在且尚未统一）
        if 'DATE([invoice_upload_time])' in df.columns and 'invoice_time' not in df.columns:
            df = df.rename(columns={'DATE([invoice_upload_time])': 'invoice_time'})
            logging.info("已将 'DATE([invoice_upload_time])' 重命名为 'invoice_time'（兜底处理）")
        # 将锁单日期重命名为 lock_time（如果存在且尚未统一）
        if 'DATE([Order Lock Time])' in df.columns and 'lock_time' not in df.columns:
            df = df.rename(columns={'DATE([Order Lock Time])': 'lock_time'})
            logging.info("已将 'DATE([Order Lock Time])' 重命名为 'lock_time'（兜底处理）")
    except Exception as e:
        logging.warning(f"日期字段兜底重命名时发生警告: {e}")
    
    # 首先，只使用order_number作为主键进行转置，避免数据丢失
    logging.info(f"准备转置的数据形状: {df.shape}")
    logging.info(f"唯一订单数量: {df['order_number'].nunique()}")
    
    # 创建透视表，只使用order_number作为索引
    try:
        # 使用pivot_table处理可能的重复值，添加observed=True以避免FutureWarning
        pivot_df = df.pivot_table(
            index='order_number',
            columns='Attribute Name',
            values='Value Display Name',
            aggfunc='first',  # 如果有重复值，取第一个
            fill_value=None,
            observed=True  # 修复FutureWarning
        )
        
        # 重置索引，将order_number转为普通列
        pivot_df = pivot_df.reset_index()
        
        # 清理列名（移除多级列名的层级）
        pivot_df.columns.name = None
        
        logging.info(f"转置完成，新数据形状: {pivot_df.shape}")
        logging.info(f"转置后的列名: {list(pivot_df.columns)}")
        
        # 现在单独处理其他字段的合并（尽可能保留原始数据集的订单级元数据）
        base_exclude = {'order_number', 'Attribute Name', 'Value Display Name'}
        # 若原始数据尚有未统一的字段名，也排除（以免与属性行级内容混淆）
        if 'Attribute Code' in df.columns:
            base_exclude.add('Attribute Code')

        candidate_cols = [c for c in df.columns if c not in base_exclude]
        if include_all_meta:
            other_info_columns = candidate_cols
            logging.info("使用include_all_meta: 合并所有非属性列到订单级数据")
        else:
            other_info_columns = select_order_level_columns(df, candidate_cols)
            excluded = sorted(set(candidate_cols) - set(other_info_columns))
            logging.info(f"自动检测订单级恒定列，共 {len(other_info_columns)} 个；排除变化列 {len(excluded)} 个")
            if excluded:
                logging.debug(f"被排除的变化列: {excluded}")

        if other_info_columns:
            logging.info(f"合并其他信息列: {other_info_columns}")
            # 为每个订单获取其他信息（按订单聚合取第一个值）
            other_info = (
                df[['order_number'] + other_info_columns]
                .groupby('order_number', as_index=False)
                .first()
            )
            # 合并到转置后的数据
            pivot_df = pivot_df.merge(other_info, on='order_number', how='left')
            logging.info(f"合并后数据形状: {pivot_df.shape}")

        return pivot_df
        
    except Exception as e:
        logging.error(f"数据转置失败: {e}")
        raise


def post_transpose_cleaning(df):
    """
    对转置后的数据进行清洗和类型转换
    """
    logging.info("开始转置后数据清洗...")
    
    df_cleaned = df.copy()
    
    # 处理OP-LuxGift字段：将"是"转换为1，空值(NaN)转换为0
    if 'OP-LuxGift' in df_cleaned.columns:
        logging.info("处理OP-LuxGift字段：'是'->1, 空值->0")
        
        # 统计转换前的分布
        before_counts = df_cleaned['OP-LuxGift'].value_counts(dropna=False)
        logging.info(f"转换前分布: {dict(before_counts)}")
        
        # 转换逻辑
        df_cleaned['OP-LuxGift'] = df_cleaned['OP-LuxGift'].map({'是': 1}).fillna(0).astype(int)
        
        # 统计转换后的分布
        after_counts = df_cleaned['OP-LuxGift'].value_counts()
        logging.info(f"转换后分布: {dict(after_counts)}")
    
    # 可以在这里添加其他转置后的数据清洗逻辑
    
    logging.info("转置后数据清洗完成")
    return df_cleaned


def save_transposed_data(df, output_path):
    """保存转置后的数据"""
    try:
        logging.info(f"正在保存转置数据到: {output_path}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"创建输出目录: {output_dir}")
        
        # 保存为CSV
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        # 验证保存结果
        file_size = os.path.getsize(output_path)
        logging.info(f"数据保存成功: {output_path}")
        logging.info(f"输出文件大小: {file_size:,} 字节")
        
        return output_path
        
    except Exception as e:
        logging.error(f"保存数据失败: {e}")
        raise

def generate_output_filename(input_path, output_dir=None):
    """生成输出文件名"""
    # 获取输入文件的基本信息
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    input_name, input_ext = os.path.splitext(input_filename)
    
    # 如果没有指定输出目录，使用processed子目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), 'processed')
    
    # 生成带时间戳的输出文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"{input_name}_transposed_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    return output_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='CM2配置数据转置处理脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python transpose_cm2_data.py -i /path/to/CM2_Configuration_Details.csv
  python transpose_cm2_data.py -i /path/to/input.csv -o /path/to/output.csv
  python transpose_cm2_data.py -i /path/to/input.csv --output-dir /path/to/output/
  python transpose_cm2_data.py -i /path/to/input.csv --analyze-only
  python transpose_cm2_data.py -i /path/to/input.csv --skip-cleaning
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='输入的CM2配置数据CSV文件路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='输出文件路径（可选，默认自动生成）'
    )
    
    parser.add_argument(
        '--output-dir',
        help='输出目录（当未指定具体输出文件时使用）'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别（默认: INFO）'
    )
    
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='仅分析数据结构，不进行转置处理'
    )
    
    parser.add_argument(
        '--skip-cleaning',
        action='store_true',
        help='跳过数据清洗和类型转换，直接进行转置'
    )

    parser.add_argument(
        '--include-all-meta',
        action='store_true',
        help='包含所有非属性列作为订单级元数据（默认自动检测仅合并订单级恒定列）'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    try:
        # 验证输入文件
        validate_input_file(args.input)
        
        # 加载数据
        df = load_data(args.input)
        
        # 分析数据结构
        analyze_data_structure(df)
        
        if args.analyze_only:
            logging.info("仅分析模式，跳过数据清洗和转置处理")
            return
        
        # 根据参数决定是否进行数据清洗和类型转换
        if args.skip_cleaning:
            logging.info("跳过数据清洗和类型转换")
            processed_df = df
        else:
            processed_df = clean_and_convert_data(df)
        
        # 进行数据转置
        transposed_df = transpose_data(processed_df, include_all_meta=args.include_all_meta)
        
        # 转置后数据清洗（处理OP-LuxGift等字段）
        if not args.skip_cleaning:
            transposed_df = post_transpose_cleaning(transposed_df)
        
        # 确定输出路径
        if args.output:
            output_path = args.output
        else:
            output_path = generate_output_filename(args.input, args.output_dir)
        
        # 保存转置后的数据
        save_transposed_data(transposed_df, output_path)
        
        logging.info("=== 处理完成 ===")
        logging.info(f"输入文件: {args.input}")
        logging.info(f"输出文件: {output_path}")
        logging.info(f"原始数据: {df.shape[0]:,} 行 x {df.shape[1]} 列")
        logging.info(f"转置数据: {transposed_df.shape[0]:,} 行 x {transposed_df.shape[1]} 列")
        
    except Exception as e:
        logging.error(f"处理失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()