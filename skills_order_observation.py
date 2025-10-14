#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
锁单数据观察脚本
用于分析intention_order_analysis.parquet数据并生成简报
计算指标包括：日锁单数、CM2车型锁单数、CM2锁单周环比、CM1同期对比、累计锁单数、
2025年累计锁单数、2024年累计锁单数对比、CM2小订累计退订率、退订率增幅日环比、CM2存量小订数
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import argparse
import requests
from openai import OpenAI

# 忽略警告
warnings.filterwarnings('ignore')

# 设置pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

# 业务参数配置
TIME_PERIODS = {
    "CM0": { "start": "2023-08-25", "end": "2023-10-12" },
    "DM0": { "start": "2024-04-08", "end": "2024-05-13" },
    "CM1": { "start": "2024-08-30", "end": "2024-09-26" },
    "CM2": { "start": "2025-08-15", "end": "2025-09-10" },
    "DM1": { "start": "2025-04-18", "end": "2025-05-13" }
}


def load_data(file_path):
    """
    加载parquet数据文件
    
    Args:
        file_path: parquet文件路径
        
    Returns:
        DataFrame: 加载的数据
    """
    try:
        print(f"正在加载数据: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        df = pd.read_parquet(file_path)
        print(f"数据加载成功，共 {len(df)} 行")
        return df
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        raise


def calculate_daily_orders(df, target_date=None):
    """
    计算日锁单数
    
    Args:
        df: 数据DataFrame
        target_date: 目标日期，默认为昨天
        
    Returns:
        int: 日锁单数
        date: 目标日期
    """
    if target_date is None:
        target_date = datetime.now().date() - timedelta(days=1)
    else:
        target_date = pd.to_datetime(target_date).date()
    
    print(f"计算 {target_date} 的日锁单数")
    
    # 确保Lock_Time列为日期类型
    if 'Lock_Time' in df.columns:
        df['Lock_Time'] = pd.to_datetime(df['Lock_Time']).dt.date
        
        # 筛选目标日期的数据
        daily_df = df[df['Lock_Time'] == target_date]
        
        # 计算唯一订单数
        if 'Order Number' in df.columns:
            daily_orders = daily_df['Order Number'].nunique()
        else:
            print("警告: 未找到'Order Number'列，使用行数作为订单数")
            daily_orders = len(daily_df)
            
        print(f"{target_date} 的日锁单数: {daily_orders}")
        return daily_orders, target_date
    else:
        print("错误: 未找到'Lock_Time'列")
        return 0, target_date


def calculate_cm2_orders(df, target_date=None):
    """
    计算CM2车型锁单数
    
    Args:
        df: 数据DataFrame
        target_date: 目标日期，默认为昨天
        
    Returns:
        int: CM2车型锁单数
    """
    if target_date is None:
        target_date = datetime.now().date() - timedelta(days=1)
    else:
        target_date = pd.to_datetime(target_date).date()
    
    print(f"计算 {target_date} 的CM2车型锁单数")
    
    # 确保Lock_Time列为日期类型
    if 'Lock_Time' in df.columns:
        df['Lock_Time'] = pd.to_datetime(df['Lock_Time']).dt.date
        
        # 筛选目标日期的数据
        daily_df = df[df['Lock_Time'] == target_date]
        
        # 筛选CM2车型
        if '车型分组' in daily_df.columns:
            cm2_df = daily_df[daily_df['车型分组'] == 'CM2']
            
            # 计算唯一订单数
            if 'Order Number' in cm2_df.columns:
                cm2_orders = cm2_df['Order Number'].nunique()
            else:
                print("警告: 未找到'Order Number'列，使用行数作为订单数")
                cm2_orders = len(cm2_df)
                
            print(f"{target_date} 的CM2车型锁单数: {cm2_orders}")
            return cm2_orders
        else:
            print("错误: 未找到'车型分组'列")
            return 0
    else:
        print("错误: 未找到'Lock_Time'列")
        return 0


def calculate_cm2_weekly_change(df, target_date=None):
    """
    计算CM2锁单周环比
    
    Args:
        df: 数据DataFrame
        target_date: 目标日期，默认为昨天
        
    Returns:
        float: CM2锁单周环比(百分比)
    """
    if target_date is None:
        target_date = datetime.now().date() - timedelta(days=1)
    else:
        target_date = pd.to_datetime(target_date).date()
    
    # 计算7天前的日期
    previous_date = target_date - timedelta(days=7)
    
    print(f"计算 {target_date} 与 {previous_date} 的CM2锁单周环比")
    
    # 计算当前CM2锁单数
    current_cm2_orders = calculate_cm2_orders(df, target_date)
    
    # 计算7天前的CM2锁单数
    previous_cm2_orders = calculate_cm2_orders(df, previous_date)
    
    # 计算周环比
    if previous_cm2_orders > 0:
        weekly_change = (current_cm2_orders - previous_cm2_orders) / previous_cm2_orders * 100
        print(f"CM2锁单周环比: {weekly_change:.2f}%")
        return weekly_change
    else:
        print("警告: 前一周期CM2锁单数为0，无法计算环比")
        return None


def calculate_cm2_refund_rate(df, date):
    """
    计算CM2小订累计退订率
    
    参数:
    df (DataFrame): 包含订单数据的DataFrame
    date (datetime.date): 计算退订率的日期
    
    返回:
    dict: 包含退订率、退订订单数和总订单数的字典
    """
    # 获取CM2时间范围
    cm2_start = pd.to_datetime(TIME_PERIODS["CM2"]["start"]).date()
    cm2_end = pd.to_datetime(TIME_PERIODS["CM2"]["end"]).date()
    
    # 确保日期列为日期类型
    if 'Intention_Payment_Time' in df.columns and 'intention_refund_time' in df.columns:
        df['Intention_Payment_Time'] = pd.to_datetime(df['Intention_Payment_Time']).dt.date
        df['intention_refund_time'] = pd.to_datetime(df['intention_refund_time']).dt.date
        
        # 筛选CM2时间范围内的订单
        cm2_orders = df[(df['Intention_Payment_Time'] >= cm2_start) & (df['Intention_Payment_Time'] <= cm2_end)]
        
        # 计算总订单数
        if 'Order Number' in cm2_orders.columns:
            total_orders = cm2_orders['Order Number'].nunique()
        else:
            print("警告: 未找到'Order Number'列，使用行数作为订单数")
            total_orders = len(cm2_orders)
        
        # 计算截至指定日期的退订订单数
        refunded_orders = cm2_orders[
            (cm2_orders['intention_refund_time'].notna()) & 
            (cm2_orders['intention_refund_time'] <= date)
        ]
        
        if 'Order Number' in refunded_orders.columns:
            refunded_count = refunded_orders['Order Number'].nunique()
        else:
            print("警告: 未找到'Order Number'列，使用行数作为退订订单数")
            refunded_count = len(refunded_orders)
        
        # 计算退订率
        refund_rate = (refunded_count / total_orders) * 100 if total_orders > 0 else 0
        
        return {
            'refund_rate': refund_rate,
            'refunded_count': refunded_count,
            'total_orders': total_orders
        }
    else:
        print("错误: 未找到必要的列('Intention_Payment_Time'或'intention_refund_time')")
        return {
            'refund_rate': 0,
            'refunded_count': 0,
            'total_orders': 0
        }

def calculate_refund_rate_daily_change(df, target_date):
    """
    计算退订率增幅日环比
    
    参数:
    df (DataFrame): 包含订单数据的DataFrame
    target_date (datetime.date): 目标日期
    
    返回:
    float: 退订率增幅日环比（百分比）
    """
    # 获取当日、前一日和前两日的日期
    previous_date = target_date - timedelta(days=1)
    two_days_ago = target_date - timedelta(days=2)
    three_days_ago = target_date - timedelta(days=3)
    
    # 使用calculate_cm2_refund_rate函数获取各日期的退订率
    current_data = calculate_cm2_refund_rate(df, target_date)
    previous_data = calculate_cm2_refund_rate(df, previous_date)
    two_days_ago_data = calculate_cm2_refund_rate(df, two_days_ago)
    three_days_ago_data = calculate_cm2_refund_rate(df, three_days_ago)
    
    # 计算当日退订率变化（当日 - 前一日）
    current_change = current_data['refund_rate'] - previous_data['refund_rate']
    
    # 计算前一日退订率变化（前一日 - 前两日）
    previous_change = previous_data['refund_rate'] - two_days_ago_data['refund_rate']
    
    # 计算幅度日环比
    if abs(previous_change) > 0.001:  # 避免除以接近零的值
        # 修改计算方法：在现有基础上-1
        change_rate = (current_change / abs(previous_change)) * 100 - 100
    else:
        # 如果前一日变化接近0，则检查更早的变化
        earlier_change = two_days_ago_data['refund_rate'] - three_days_ago_data['refund_rate']
        if abs(earlier_change) > 0.001:
            # 修改计算方法：在现有基础上-1
            change_rate = (current_change / abs(earlier_change)) * 100 - 100
        else:
            # 如果历史变化都接近0，则无法计算环比，返回0
            change_rate = 0
    
    # 打印详细信息用于调试
    print("\n===== 退订率增幅日环比计算详情 =====")
    print(f"当日退订率: {current_data['refund_rate']:.2f}%")
    print(f"前一日退订率: {previous_data['refund_rate']:.2f}%")
    print(f"前两日退订率: {two_days_ago_data['refund_rate']:.2f}%")
    print(f"前三日退订率: {three_days_ago_data['refund_rate']:.2f}%")
    print(f"当日变化: {current_change:.2f}%")
    print(f"前一日变化: {previous_change:.2f}%")
    print(f"退订率增幅日环比: {change_rate:.2f}%")
    print("===================================\n")
    
    return change_rate


def calculate_cm2_delivery_count(df, target_date=None):
    """
    计算CM2交付数
    计算方式：在观察时间戳，CM2车型，且Invoice_Upload_Time不为空的订单数
    
    Args:
        df: 数据DataFrame
        target_date: 目标日期，默认为昨天
        
    Returns:
        int: CM2交付数
    """
    if target_date is None:
        target_date = datetime.now().date() - timedelta(days=1)
    else:
        target_date = pd.to_datetime(target_date).date()
    
    print(f"计算 {target_date} 的CM2交付数")
    
    # 确保Invoice_Upload_Time列为日期类型
    if 'Invoice_Upload_Time' in df.columns:
        # 转换为日期类型，保留原始数据
        df_copy = df.copy()
        df_copy['Invoice_Upload_Time'] = pd.to_datetime(df_copy['Invoice_Upload_Time']).dt.date
        
        # 筛选目标日期的数据（交付日期等于目标日期）
        daily_df = df_copy[df_copy['Invoice_Upload_Time'] == target_date]
        
        # 筛选CM2车型
        if '车型分组' in daily_df.columns:
            cm2_df = daily_df[daily_df['车型分组'] == 'CM2']
            
            # 计算唯一订单数
            if 'Order Number' in cm2_df.columns:
                delivery_count = cm2_df['Order Number'].nunique()
            else:
                print("警告: 未找到'Order Number'列，使用行数作为交付订单数")
                delivery_count = len(cm2_df)
                
            print(f"{target_date} 的CM2交付数: {delivery_count}")
            return delivery_count
        else:
            print("错误: 未找到'车型分组'列")
            return 0
    else:
        print("错误: 未找到'Invoice_Upload_Time'列")
        return 0


def calculate_rolling_average(df, target_date, days=7, value_type='delivery'):
    """
    计算滚动平均值
    
    Args:
        df: 数据DataFrame
        target_date: 目标日期
        days: 滚动天数，默认7天
        value_type: 计算类型，'delivery'表示交付数，'invoice_price'表示开票价格（基于交付时间）
        
    Returns:
        float: 滚动平均值
    """
    target_date = pd.to_datetime(target_date).date()
    
    # 计算开始日期（包含目标日期在内的前N天）
    start_date = target_date - timedelta(days=days-1)
    
    print(f"计算从 {start_date} 到 {target_date} 的{days}日滚动平均（{value_type}）")
    
    # 初始化值列表
    values = []
    
    # 根据类型计算每天的值
    for i in range(days):
        current_date = start_date + timedelta(days=i)
        
        if value_type == 'delivery':
            # 计算交付数（基于Invoice_Upload_Time）
            daily_value = calculate_cm2_delivery_count(df, current_date)
            values.append(daily_value)
        elif value_type == 'invoice_price':
            # 计算开票价格平均值（基于Invoice_Upload_Time交付时间）
            daily_value = calculate_daily_invoice_price(df, current_date)
            values.append(daily_value)
    
    # 计算平均值
    if values:
        avg_value = sum(values) / len(values)
        print(f"{days}日滚动平均值（{value_type}）: {avg_value:.2f}")
        return avg_value
    else:
        print(f"警告: 无法计算{days}日滚动平均值（{value_type}），返回0")
        return 0


def calculate_daily_invoice_price(df, target_date):
    """
    计算指定日期的CM2开票价格平均值（基于交付时间）
    
    Args:
        df: 数据DataFrame
        target_date: 目标日期
        
    Returns:
        float: 开票价格平均值
    """
    target_date = pd.to_datetime(target_date).date()
    
    print(f"计算 {target_date} 的CM2开票价格平均值（基于交付时间）")
    
    # 确保Invoice_Upload_Time列为日期类型
    if 'Invoice_Upload_Time' in df.columns:
        df['Invoice_Upload_Time'] = pd.to_datetime(df['Invoice_Upload_Time']).dt.date
        
        # 筛选目标日期的数据（基于交付时间）
        daily_df = df[df['Invoice_Upload_Time'] == target_date]
        
        # 筛选CM2车型
        if '车型分组' in daily_df.columns:
            cm2_df = daily_df[daily_df['车型分组'] == 'CM2']
            
            # 计算开票价格平均值
            if '开票价格' in cm2_df.columns:
                # 排除空值和0值
                valid_prices = cm2_df[
                    (cm2_df['开票价格'].notna()) & 
                    (cm2_df['开票价格'] > 0)
                ]['开票价格']
                
                if len(valid_prices) > 0:
                    avg_price = valid_prices.mean()
                    print(f"{target_date} 的CM2开票价格平均值（交付日期）: {avg_price:.2f}")
                    return avg_price
                else:
                    print(f"警告: {target_date} 没有有效的开票价格数据（交付日期）")
                    return 0
            else:
                print("警告: 未找到'开票价格'列")
                return 0
        else:
            print("错误: 未找到'车型分组'列")
            return 0
    else:
        print("错误: 未找到'Invoice_Upload_Time'列")
        return 0


def calculate_cm2_active_orders(df, target_date=None):
    """
    计算CM2存量小订数
    计算方式：总订单数 - 退订数 - 已转化小订数
    已转化判定：订单同时具有有效的锁单时间（Lock_Time）与意向支付时间（Intention_Payment_Time），且两者均不晚于目标日期

    Args:
        df: 数据DataFrame
        target_date: 目标日期，默认为昨天

    Returns:
        int: CM2存量小订数
    """
    if target_date is None:
        target_date = datetime.now().date() - timedelta(days=1)
    else:
        target_date = pd.to_datetime(target_date).date()

    # 获取CM2时间范围
    cm2_start = pd.to_datetime(TIME_PERIODS["CM2"]["start"]).date()
    cm2_end = pd.to_datetime(TIME_PERIODS["CM2"]["end"]).date()

    print(f"计算CM2存量小订数 (时间范围: {cm2_start} 到 {cm2_end}, 观察日: {target_date})")

    # 辅助函数：获取唯一订单ID集合
    def _unique_ids(df_slice):
        if 'Order Number' in df_slice.columns:
            return set(df_slice['Order Number'].dropna().astype(str).unique())
        else:
            return set(df_slice.index.tolist())

    # 确保必要日期列存在并转为日期类型
    if 'Intention_Payment_Time' in df.columns:
        df_copy = df.copy()
        df_copy['Intention_Payment_Time'] = pd.to_datetime(df_copy['Intention_Payment_Time'], errors='coerce').dt.date
        if 'intention_refund_time' in df_copy.columns:
            df_copy['intention_refund_time'] = pd.to_datetime(df_copy['intention_refund_time'], errors='coerce').dt.date
        if 'Lock_Time' in df_copy.columns:
            df_copy['Lock_Time'] = pd.to_datetime(df_copy['Lock_Time'], errors='coerce').dt.date

        # 筛选CM2时间范围内的小订订单
        cm2_orders = df_copy[(df_copy['Intention_Payment_Time'] >= cm2_start) & (df_copy['Intention_Payment_Time'] <= cm2_end)]

        # 总订单集合与数量
        total_ids = _unique_ids(cm2_orders)
        total_orders = len(total_ids)

        # 退订订单集合（截至目标日期）
        refunded_ids = set()
        if 'intention_refund_time' in cm2_orders.columns:
            refund_mask = cm2_orders['intention_refund_time'].notna() & (cm2_orders['intention_refund_time'] <= target_date)
            refunded_ids = _unique_ids(cm2_orders[refund_mask])
        else:
            print("警告: 未找到'intention_refund_time'列，退订订单数计为0")

        # 已转化订单集合（截至目标日期，需同时满足 Lock_Time 与 Intention_Payment_Time）
        converted_ids = set()
        if 'Lock_Time' in cm2_orders.columns:
            lock_mask = cm2_orders['Lock_Time'].notna() & (cm2_orders['Lock_Time'] <= target_date)
            pay_mask = cm2_orders['Intention_Payment_Time'].notna() & (cm2_orders['Intention_Payment_Time'] <= target_date)
            converted_mask = lock_mask & pay_mask
            converted_ids = _unique_ids(cm2_orders[converted_mask])
        else:
            print("警告: 未找到'Lock_Time'列，已转化订单数计为0")

        # 去重合并的非存量集合（退订 ∪ 已转化）
        non_active_ids = refunded_ids | converted_ids

        # 计算存量订单集合与数量
        active_ids = total_ids - non_active_ids
        active_count = len(active_ids)

        print(
            f"CM2存量小订数: {active_count} "
            f"(总订单数: {total_orders}, 退订订单数: {len(refunded_ids)}, 已转化订单数: {len(converted_ids)})"
        )
        return active_count
    else:
        print("错误: 未找到'Intention_Payment_Time'列")
        return 0


def calculate_yearly_cumulative_orders(df, year, target_date=None):
    """
    计算指定年份从1月1日到目标日期的累计锁单数
    
    Args:
        df: 数据DataFrame
        year: 年份，如2025
        target_date: 目标日期，默认为昨天
        
    Returns:
        int: 年度累计锁单数
    """
    if target_date is None:
        target_date = datetime.now().date() - timedelta(days=1)
    else:
        target_date = pd.to_datetime(target_date).date()
    
    # 计算年初日期
    start_date = datetime(year, 1, 1).date()
    
    # 确保目标日期不超过当前日期
    if target_date > datetime.now().date():
        target_date = datetime.now().date() - timedelta(days=1)
    
    print(f"计算 {year}年 从 {start_date} 到 {target_date} 的累计锁单数")
    
    # 确保Lock_Time列为日期类型
    if 'Lock_Time' in df.columns:
        df['Lock_Time'] = pd.to_datetime(df['Lock_Time']).dt.date
        
        # 筛选日期范围内的数据
        yearly_df = df[(df['Lock_Time'] >= start_date) & (df['Lock_Time'] <= target_date)]
        
        # 计算唯一订单数
        if 'Order Number' in yearly_df.columns:
            yearly_orders = yearly_df['Order Number'].nunique()
        else:
            print(f"警告: 未找到'Order Number'列，使用行数作为{year}年累计订单数")
            yearly_orders = len(yearly_df)
            
        print(f"{year}年累计锁单数: {yearly_orders}")
        return yearly_orders
    else:
        print("错误: 未找到'Lock_Time'列")
        return 0


def calculate_cm1_comparison(df, target_date=None):
    """
    计算CM1同期对比
    基于CM2的观察日期与CM2结束日期的差值，计算CM1对应时间点的锁单数
    
    Args:
        df: 数据DataFrame
        target_date: 目标日期，默认为昨天
        
    Returns:
        dict: 包含CM1同期锁单数和对比信息
    """
    if target_date is None:
        target_date = datetime.now().date() - timedelta(days=1)
    else:
        target_date = pd.to_datetime(target_date).date()
    
    # 获取CM2和CM1的结束日期
    cm2_end_date = pd.to_datetime(TIME_PERIODS["CM2"]["end"]).date()
    cm1_end_date = pd.to_datetime(TIME_PERIODS["CM1"]["end"]).date()
    
    # 计算目标日期与CM2结束日期的差值天数
    days_diff = (target_date - cm2_end_date).days
    
    # 计算CM1对应的同期日期
    cm1_comparable_date = cm1_end_date + timedelta(days=days_diff)
    
    print(f"计算CM1同期对比: CM2日期 {target_date} 对应CM1日期 {cm1_comparable_date}")
    
    # 确保Lock_Time列为日期类型
    if 'Lock_Time' in df.columns:
        df['Lock_Time'] = pd.to_datetime(df['Lock_Time']).dt.date
        
        # 筛选CM1对应日期的数据
        cm1_daily_df = df[df['Lock_Time'] == cm1_comparable_date]
        
        # 筛选CM1车型
        if '车型分组' in cm1_daily_df.columns:
            cm1_df = cm1_daily_df[cm1_daily_df['车型分组'] == 'CM1']
            
            # 计算唯一订单数
            if 'Order Number' in cm1_df.columns:
                cm1_orders = cm1_df['Order Number'].nunique()
            else:
                print("警告: 未找到'Order Number'列，使用行数作为订单数")
                cm1_orders = len(cm1_df)
            
            # 获取当前CM2锁单数进行对比
            cm2_orders = calculate_cm2_orders(df, target_date)
            
            # 计算同比变化
            if cm1_orders > 0:
                change_rate = (cm2_orders - cm1_orders) / cm1_orders * 100
                change_direction = "增长" if change_rate > 0 else "下降"
                print(f"CM1同期锁单数: {cm1_orders}, CM2当前锁单数: {cm2_orders}")
                print(f"同比变化: {abs(change_rate):.2f}% ({change_direction})")
                
                return {
                    "cm1_date": cm1_comparable_date,
                    "cm1_orders": cm1_orders,
                    "cm2_orders": cm2_orders,
                    "change_rate": change_rate,
                    "change_direction": change_direction
                }
            else:
                print("警告: CM1同期锁单数为0，无法计算同比变化")
                return {
                    "cm1_date": cm1_comparable_date,
                    "cm1_orders": cm1_orders,
                    "cm2_orders": cm2_orders,
                    "change_rate": None,
                    "change_direction": None
                }
        else:
            print("错误: 未找到'车型分组'列")
            return None
    else:
        print("错误: 未找到'Lock_Time'列")
        return None


def calculate_cumulative_orders(df, model_type, start_date, end_date):
    """
    计算指定车型在给定日期范围内的累计锁单数
    
    Args:
        df: 数据DataFrame
        model_type: 车型分组，如'CM1'或'CM2'
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        int: 累计锁单数
    """
    # 确保日期格式正确
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    
    print(f"计算{model_type}从{start_date}到{end_date}的累计锁单数")
    
    # 确保Lock_Time列为日期类型
    if 'Lock_Time' in df.columns:
        df['Lock_Time'] = pd.to_datetime(df['Lock_Time']).dt.date
        
        # 筛选日期范围内的数据
        date_range_df = df[(df['Lock_Time'] >= start_date) & (df['Lock_Time'] <= end_date)]
        
        # 筛选指定车型
        if '车型分组' in date_range_df.columns:
            model_df = date_range_df[date_range_df['车型分组'] == model_type]
            
            # 计算唯一订单数
            if 'Order Number' in model_df.columns:
                cumulative_orders = model_df['Order Number'].nunique()
            else:
                print(f"警告: 未找到'Order Number'列，使用行数作为{model_type}累计订单数")
                cumulative_orders = len(model_df)
                
            print(f"{model_type}从{start_date}到{end_date}的累计锁单数: {cumulative_orders}")
            return cumulative_orders
        else:
            print("错误: 未找到'车型分组'列")
            return 0
    else:
        print("错误: 未找到'Lock_Time'列")
        return 0


def send_to_flomo(content, flomo_api_url="https://flomoapp.com/iwh/NDIwOTAx/c62bd115ef72eb46a2289296744fe0dc/"):
    """
    将内容发送到 flomo API
    
    Args:
        content: 要发送的内容
        flomo_api_url: flomo API 的 URL
        
    Returns:
        bool: 是否发送成功
    """
    try:
        headers = {'Content-Type': 'application/json'}
        data = {'content': content}
        response = requests.post(flomo_api_url, headers=headers, json=data)
        
        if response.status_code == 200:
            print(f"成功同步到 flomo: {response.json().get('message', '未知响应')}")
            return True
        else:
            print(f"同步到 flomo 失败: HTTP {response.status_code}, {response.text}")
            return False
    except Exception as e:
        print(f"同步到 flomo 时发生错误: {str(e)}")
        return False

def generate_report(daily_orders, cm2_orders, cm2_weekly_change, target_date, cm1_comparison=None, cm2_cumulative=None, cm1_cumulative=None, year_2025_orders=None, year_2024_orders=None, cm2_refund_data=None, refund_rate_change=None, cm2_active_orders=None, cm2_delivery_count=None, delivery_rolling_avg=None, delivery_rolling_avg_prev=None, invoice_price_rolling_avg=None, invoice_price_rolling_avg_prev=None, sync_to_flomo=False):
    """
    生成简报
    
    Args:
        daily_orders: 日锁单数
        cm2_orders: CM2车型锁单数
        cm2_weekly_change: CM2锁单周环比
        target_date: 目标日期
        cm1_comparison: CM1同期对比数据
        cm2_cumulative: CM2累计锁单数
        cm1_cumulative: CM1累计锁单数
        year_2025_orders: 2025年累计锁单数
        year_2024_orders: 2024年累计锁单数
        cm2_refund_data: CM2小订累计退订率数据
        refund_rate_change: 退订率日环比
        cm2_active_orders: CM2存量小订数
        cm2_delivery_count: CM2交付数
        delivery_rolling_avg: 交付7日滚动平均
        delivery_rolling_avg_prev: 交付7日滚动平均前值
        invoice_price_rolling_avg: 开票价格7日滚动平均
        invoice_price_rolling_avg_prev: 开票价格7日滚动平均前值
        sync_to_flomo: 是否同步到 flomo
        
    Returns:
        str: 简报内容
    """
    report = []
    report.append("=" * 30)
    report.append(f"锁单数据观察简报 - {target_date}")
    report.append("=" * 30)
    report.append("")
    
    # 锁单数据部分
    report.append("一、锁单")
    
    # 日锁单数和CM2车型锁单数
    report.append(f"📊 日锁单数: {daily_orders}, CM2车型锁单数: {cm2_orders}")
    
    # 周环比
    if cm2_weekly_change is not None:
        trend = "上升" if cm2_weekly_change > 0 else "下降"
        report.append(f"   - 周环比: {abs(cm2_weekly_change):.2f}% ({trend})")
    else:
        report.append("   - 周环比: N/A (前一周期数据不足)")
    
    # CM1同期对比
    if cm1_comparison is not None and cm1_comparison.get('change_rate') is not None:
        direction = cm1_comparison.get('change_direction', '增长' if cm1_comparison['change_rate'] > 0 else '下降')
        report.append(f"   - CM1同期对比: {abs(cm1_comparison['change_rate']):.2f}% ({direction})")
        report.append(f"     CM1同期锁单数({cm1_comparison['cm1_date']}): {cm1_comparison['cm1_orders']}")
    else:
        report.append("   - CM1同期对比: N/A (数据不足)")
    
    # CM2累计锁单数
    if cm2_cumulative is not None:
        report.append(f"📈 CM2车型累计锁单数: {cm2_cumulative}")
        
        # 对比同期CM1累计锁单数
        if cm1_cumulative is not None:
            cm_diff = cm2_cumulative - cm1_cumulative
            cm_diff_direction = "高于" if cm_diff > 0 else "低于"
            report.append(f"   - 对比同期CM1累计锁单数: {cm1_cumulative} ({cm_diff_direction} {abs(cm_diff)})")
        else:
            report.append("   - 对比同期CM1累计锁单数: N/A (数据不足)")
    else:
        report.append("📈 CM2车型累计锁单数: N/A (数据不足)")
    
    # 年累计锁单数对比
    if year_2025_orders is not None:
        report.append(f"📆 2025年累计锁单数: {year_2025_orders}")
        
        if year_2024_orders is not None and year_2024_orders > 0:
            growth_rate = (year_2025_orders - year_2024_orders) / year_2024_orders * 100
            direction = "增长" if growth_rate > 0 else "下降"
            report.append(f"   - 对比2024年累计锁单数: {year_2024_orders} ({abs(growth_rate):.2f}% {direction})")
        else:
            report.append("   - 对比2024年累计锁单数: N/A (2024年数据不足)")
    
    report.append("")
    
    # 转化数据部分
    report.append("二、转化")
    
    # CM2小订累计退订率
    if cm2_refund_data is not None:
        report.append(f"🔄 CM2小订累计退订率: {cm2_refund_data['refund_rate']:.2f}%")
        report.append(f"   - 退订订单数: {cm2_refund_data['refunded_count']}")
        report.append(f"   - 总订单数: {cm2_refund_data['total_orders']}")
        
        # CM2存量小订数
        if cm2_active_orders is not None:
            report.append(f"📦 CM2存量小订数: {cm2_active_orders}")
        else:
            report.append("📦 CM2存量小订数: N/A (数据不足)")

        # 退订率增幅日环比
        if refund_rate_change is not None:
            direction = "上升" if refund_rate_change > 0 else "下降"
            report.append(f"📉 退订率增幅日环比: {refund_rate_change:.2f}% ({direction})")
        else:
            report.append("📉 退订率增幅日环比: N/A (数据不足)")
            
    else:
        report.append("🔄 CM2小订累计退订率: N/A (数据不足)")
        report.append("📉 退订率日环比: N/A (数据不足)")
        report.append("📦 CM2存量小订数: N/A (数据不足)")
    
    report.append("")
    
    # 交付数据部分
    report.append("三、交付")
    
    # CM2交付数
    if cm2_delivery_count is not None:
        report.append(f"🚗 CM2交付数: {cm2_delivery_count}")
    else:
        report.append("🚗 CM2交付数: N/A (数据不足)")
    
    # 交付7日滚动平均
    if delivery_rolling_avg is not None:
        report.append(f"📈 交付7日滚动平均: {delivery_rolling_avg:.2f}")
        
        if delivery_rolling_avg_prev is not None:
            change = delivery_rolling_avg - delivery_rolling_avg_prev
            change_rate = (change / delivery_rolling_avg_prev * 100) if delivery_rolling_avg_prev > 0 else 0
            direction = "上升" if change > 0 else "下降"
            report.append(f"   - 前值: {delivery_rolling_avg_prev:.2f}")
            report.append(f"   - 变化: {abs(change):.2f} ({abs(change_rate):.2f}% {direction})")
        else:
            report.append("   - 前值: N/A (数据不足)")
    else:
        report.append("1. 交付7日滚动平均: N/A (数据不足)")
    
    # 交付价格滚动平均
    if invoice_price_rolling_avg is not None:
        report.append(f"💰 交付价格滚动平均: {invoice_price_rolling_avg:.2f}")
        
        if invoice_price_rolling_avg_prev is not None:
            change = invoice_price_rolling_avg - invoice_price_rolling_avg_prev
            change_rate = (change / invoice_price_rolling_avg_prev * 100) if invoice_price_rolling_avg_prev > 0 else 0
            direction = "上升" if change > 0 else "下降"
            report.append(f"   - 前值: {invoice_price_rolling_avg_prev:.2f}")
            report.append(f"   - 变化: {abs(change):.2f} ({abs(change_rate):.2f}% {direction})")
        else:
            report.append("   - 前值: N/A (数据不足)")
    else:
        report.append("💰 交付价格滚动平均: N/A (数据不足)")
    
    report.append("")
    
    # 使用DeepSeek API生成结论与建议
    report.append("四、结论与建议")
    
    # 创建DeepSeek API客户端
    client = OpenAI(api_key="sk-8145b27fa56640ed8df695e9bd49ed8c", base_url="https://api.deepseek.com")
    
    # 获取当前日期是否为工作日
    target_date_weekday = pd.to_datetime(target_date).weekday()
    is_weekend = target_date_weekday >= 5  # 5和6分别代表周六和周日
    
    # 计算交付7日滚动平均的变化率
    delivery_rolling_change_rate = None
    if delivery_rolling_avg is not None and delivery_rolling_avg_prev is not None and delivery_rolling_avg_prev > 0:
        delivery_rolling_change_rate = (delivery_rolling_avg - delivery_rolling_avg_prev) / delivery_rolling_avg_prev * 100
    
    # 构建提示词，包含所有指标数据和关键判断标准
    prompt = f"""
    请根据以下汽车销售数据指标，生成专业的结论与建议（不超过4条）：
    
    一、锁单数据（重点关注CM2车型）:
    1. 日锁单数: {daily_orders}
    2. CM2车型锁单数: {cm2_orders}
    3. 今日是否工作日: {"否（周末）" if is_weekend else "是（工作日）"}
    4. CM2锁单周环比: {cm2_weekly_change}%
    5. CM2工作日平均锁单标准: 326台
    6. CM2周末锁单标准: 工作日的1.5-2倍（约489-652台）
    7. CM2锁单是否达标: {"未达标" if (not is_weekend and cm2_orders < 326) or (is_weekend and cm2_orders < 489) else "达标"}
    """
    
    if cm1_comparison is not None and cm1_comparison['change_rate'] is not None:
        prompt += f"8. CM2相比CM1同期变化率: {cm1_comparison['change_rate']}%\n"
    
    if cm2_cumulative is not None:
        prompt += f"9. CM2累计锁单数: {cm2_cumulative}\n"
    
    if year_2025_orders is not None and year_2024_orders is not None:
        growth_rate = (year_2025_orders - year_2024_orders) / year_2024_orders * 100 if year_2024_orders > 0 else None
        prompt += f"10. 2025年累计锁单数: {year_2025_orders}\n"
        prompt += f"11. 2024年同期累计锁单数: {year_2024_orders}\n"
        if growth_rate is not None:
            prompt += f"12. 年度累计锁单数同比增长率: {growth_rate:.2f}%\n"
    
    prompt += f"""
    二、转化数据（重点关注退订率增幅日环比）:
    """
    
    if cm2_refund_data is not None:
        prompt += f"13. CM2小订累计退订率: {cm2_refund_data['refund_rate']}%\n"
        prompt += f"14. 退订率增幅日环比: {refund_rate_change}%\n"
        prompt += f"15. 退订率增幅日环比正常范围: -50%至50%\n"
        prompt += f"16. 退订率增幅日环比是否异常: {"异常（需预警）" if abs(refund_rate_change) > 50 else "正常"}\n"
    
    if cm2_active_orders is not None:
        prompt += f"17. CM2存量小订数: {cm2_active_orders}\n"
    
    prompt += f"""
    三、交付数据（重点关注交付7日滚动平均提升幅度）:
    """
    
    if cm2_delivery_count is not None:
        prompt += f"18. CM2交付数: {cm2_delivery_count}\n"
    
    if delivery_rolling_avg is not None:
        prompt += f"19. 交付7日滚动平均: {delivery_rolling_avg}\n"
        if delivery_rolling_avg_prev is not None:
            prompt += f"20. 前一日交付7日滚动平均: {delivery_rolling_avg_prev}\n"
            if delivery_rolling_change_rate is not None:
                prompt += f"21. 交付7日滚动平均变化率: {delivery_rolling_change_rate:.2f}%\n"
                prompt += f"22. 交付提升是否显著: {"是（提升幅度>10%）" if delivery_rolling_change_rate > 10 else "否"}\n"
    
    if invoice_price_rolling_avg is not None:
        prompt += f"23. 交付价格滚动平均: {invoice_price_rolling_avg}\n"
    
    prompt += """
    请根据以上数据，重点分析以下三个方面：
    1. CM2车型锁单数是否达到工作日/周末的标准，如未达标需提出预警和改进建议
    2. 退订率增幅日环比是否在正常范围内（-50%至50%），如超过范围需提出预警和干预措施
    3. 交付7日滚动平均提升幅度是否超过10%，如是则需要指出并分析原因
    
    回复格式要求：
    - 直接以"- "开头列出每条建议
    - 每条建议应包含数据分析和简短的建议
    - 不要有任何开头语和结尾语
    - 如有预警情况，请在建议开头标注"【预警】"
    """
    
    try:
        # 调用DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位专业的汽车销售数据分析师，擅长从销售数据中提炼洞见并给出实用的业务建议。"},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        
        # 获取API返回的结论与建议
        ai_suggestions = response.choices[0].message.content.strip().split('\n')
        
        # 将AI生成的建议添加到报告中
        for suggestion in ai_suggestions:
            if suggestion.strip():
                report.append(suggestion.strip())
    except Exception as e:
        # 如果API调用失败，使用原有的固定建议
        print(f"DeepSeek API调用失败: {e}")
        
        if cm2_weekly_change is not None:
            if cm2_weekly_change > 10:
                report.append("- CM2车型锁单数显著增长，建议关注增长原因并加强相关营销策略")
            elif cm2_weekly_change < -10:
                report.append("- CM2车型锁单数明显下降，建议分析下降原因并采取相应措施")
            else:
                report.append("- CM2车型锁单数相对稳定，建议持续监控市场变化")
        
        if daily_orders > 0 and cm2_orders / daily_orders < 0.3:
            report.append("- CM2车型占比较低，建议加强CM2车型的推广力度")
        
        if cm1_comparison is not None and cm1_comparison['change_rate'] is not None:
            if cm1_comparison['change_rate'] > 20:
                report.append("- CM2相比CM1同期表现显著提升，建议分析成功因素并复制到其他车型")
            elif cm1_comparison['change_rate'] < -20:
                report.append("- CM2相比CM1同期表现明显下降，建议分析原因并制定改进策略")
        
        # 年度累计锁单数对比建议
        if year_2025_orders is not None and year_2024_orders is not None and year_2024_orders > 0:
            growth_rate = (year_2025_orders - year_2024_orders) / year_2024_orders * 100
            if growth_rate > 15:
                report.append("- 2025年累计锁单数同比大幅增长，年度销售目标完成情况良好")
            elif growth_rate < 0:
                report.append("- 2025年累计锁单数同比下降，需加强销售力度以达成年度目标")
        
        # 退订率和存量小订数建议
        if cm2_refund_data is not None:
            if cm2_refund_data['refund_rate'] > 15:
                report.append("- CM2小订累计退订率较高，建议分析退订原因，优化产品体验和售后服务")
            elif cm2_refund_data['refund_rate'] < 5:
                report.append("- CM2小订累计退订率较低，客户稳定性好，建议分析成功经验并推广")
                
            if refund_rate_change is not None:
                if refund_rate_change > 5:
                    report.append("- 退订率日环比上升明显，建议立即排查原因并采取干预措施")
                elif refund_rate_change < -5:
                    report.append("- 退订率日环比下降明显，建议分析成功因素并持续优化")
                    
            if cm2_active_orders is not None:
                report.append(f"- 当前CM2存量小订数为{cm2_active_orders}，建议针对这些客户制定专项维护计划，提高转化率")
    
    report.append("")
    report.append("=" * 30)
    report.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 30)
    
    report_text = "\n".join(report)
    
    # 如果启用了Flomo同步，则发送报告内容到Flomo
    if sync_to_flomo:
        print("正在同步报告到Flomo...")
        send_to_flomo(report_text)
    
    return report_text


def save_report(report, output_path):
    """
    保存简报到文件
    
    Args:
        report: 简报内容
        output_path: 输出文件路径
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"简报已保存至: {output_path}")
    except Exception as e:
        print(f"保存简报失败: {str(e)}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='锁单数据观察脚本')
    parser.add_argument('--date', type=str, help='指定观察日期，格式为YYYY-MM-DD，默认为昨天')
    parser.add_argument('--flomo', action='store_true', help='是否同步到 flomo')
    args = parser.parse_args()
    
    try:
        # 数据文件路径
        data_path = "../formatted/intention_order_analysis.parquet"
        
        # 加载数据
        df = load_data(data_path)
        
        # 获取目标日期
        if args.date:
            try:
                target_date = pd.to_datetime(args.date).date()
                print(f"使用指定日期: {target_date}")
            except ValueError:
                print(f"无效的日期格式: {args.date}，使用默认日期（昨天）")
                target_date = datetime.now().date() - timedelta(days=1)
        else:
            # 默认使用昨天的日期
            target_date = datetime.now().date() - timedelta(days=1)
            print(f"使用默认日期（昨天）: {target_date}")
        
        # 计算基本指标
        daily_orders, target_date = calculate_daily_orders(df, target_date)
        cm2_orders = calculate_cm2_orders(df, target_date)
        cm2_weekly_change = calculate_cm2_weekly_change(df, target_date)
        
        # 计算CM1同期对比
        cm1_comparison = calculate_cm1_comparison(df, target_date)
        
        # 计算CM2累计锁单数
        cm2_end_date = pd.to_datetime(TIME_PERIODS["CM2"]["end"]).date()
        cm2_cumulative = calculate_cumulative_orders(df, "CM2", cm2_end_date, target_date)
        
        # 计算CM1累计锁单数
        # 计算CM1对应的时间段
        days_diff = (target_date - cm2_end_date).days
        cm1_end_date = pd.to_datetime(TIME_PERIODS["CM1"]["end"]).date()
        cm1_comparable_end_date = cm1_end_date + timedelta(days=days_diff)
        cm1_cumulative = calculate_cumulative_orders(df, "CM1", cm1_end_date, cm1_comparable_end_date)
        
        # 计算2025年累计锁单数
        year_2025_orders = calculate_yearly_cumulative_orders(df, 2025, target_date)
        
        # 计算2024年累计锁单数（同期对比）
        # 获取2025年的日期，然后计算2024年同一天的数据
        date_2024 = datetime(2024, target_date.month, target_date.day).date()
        if date_2024.month == 2 and date_2024.day == 29 and not (2024 % 4 == 0 and (2024 % 100 != 0 or 2024 % 400 == 0)):
            # 处理闰年问题
            date_2024 = datetime(2024, 2, 28).date()
        year_2024_orders = calculate_yearly_cumulative_orders(df, 2024, date_2024)
        
        # 计算CM2小订累计退订率
        cm2_refund_data = calculate_cm2_refund_rate(df, target_date)
        refund_rate = cm2_refund_data['refund_rate']
        refund_count = cm2_refund_data['refunded_count']
        total_orders = cm2_refund_data['total_orders']
        
        # 计算退订率日环比
        refund_rate_change = calculate_refund_rate_daily_change(df, target_date)
        
        # 计算CM2存量小订数
        cm2_active_orders = calculate_cm2_active_orders(df, target_date)
        
        # 计算CM2交付数
        cm2_delivery_count = calculate_cm2_delivery_count(df, target_date)
        
        # 计算交付7日滚动平均
        delivery_rolling_avg = calculate_rolling_average(df, target_date, days=7, value_type='delivery')
        
        # 计算交付7日滚动平均前值（前一天的值）
        prev_date = pd.to_datetime(target_date).date() - timedelta(days=1)
        delivery_rolling_avg_prev = calculate_rolling_average(df, prev_date, days=7, value_type='delivery')
        
        # 计算开票价格7日滚动平均
        invoice_price_rolling_avg = calculate_rolling_average(df, target_date, days=7, value_type='invoice_price')
        
        # 计算开票价格7日滚动平均前值
        invoice_price_rolling_avg_prev = calculate_rolling_average(df, prev_date, days=7, value_type='invoice_price')
        
        # 输出退订率增幅日环比的两个值
        print("\n===== 退订率增幅日环比计算值 =====")
        print(f"当日退订率: {cm2_refund_data['refund_rate']:.2f}%")
        print(f"前一日退订率: {cm2_refund_data['refund_rate'] - refund_rate_change * cm2_refund_data['refund_rate'] / 100:.2f}%")
        print(f"退订率增幅日环比: {refund_rate_change:.2f}%")
        print("=============================\n")
        
        # 生成简报
        report = generate_report(
            daily_orders, 
            cm2_orders, 
            cm2_weekly_change, 
            target_date,
            cm1_comparison,
            cm2_cumulative,
            cm1_cumulative,
            year_2025_orders,
            year_2024_orders,
            cm2_refund_data,
            refund_rate_change,
            cm2_active_orders,
            cm2_delivery_count,
            delivery_rolling_avg,
            delivery_rolling_avg_prev,
            invoice_price_rolling_avg,
            invoice_price_rolling_avg_prev,
            sync_to_flomo=args.flomo
        )
        print("\n" + report)
        
        # 保存简报
        output_path = f"../reports/锁单数据简报_{target_date}.txt"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_report(report, output_path)
        
        print("脚本执行完成")
        return 0
    except Exception as e:
        print(f"脚本执行失败: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)