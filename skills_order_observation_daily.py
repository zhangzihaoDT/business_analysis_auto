#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
每日锁单数据观察脚本
功能：
1. 读取 order_full_data.parquet 数据
2. 计算昨日（T-1）的锁单数
3. 统计指定车型（CM2, DM1, LS9）的锁单情况
4. 发送飞书通知
"""

import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置常量
PARQUET_FILE = "/Users/zihao_/Documents/coding/dataset/formatted/order_full_data.parquet"
BUSINESS_DEF_FILE = "/Users/zihao_/Documents/github/W52_reasoning/world/business_definition.json"
# 适配新数据集的 series 值：CM2->LS6, DM1->L6
TARGET_MODELS = ["LS6", "L6", "LS9"]
WEBHOOK_URL = os.getenv("FS_WEBHOOK_URL")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='每日锁单数据观察脚本')
    parser.add_argument('--start', type=str, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='结束日期 (YYYY-MM-DD)')
    
    # 预处理 sys.argv 以支持 -N 这种非标准参数
    days_back = 1  # 默认昨天
    
    # 检查是否有负数参数 (如 -1, -2, -7)
    args_to_remove = []
    for arg in sys.argv[1:]:
        if arg.startswith('-') and len(arg) > 1 and arg[1:].isdigit():
            days_back = int(arg[1:])
            args_to_remove.append(arg)
    
    # 从 sys.argv 中移除这些参数，以免 argparse 报错
    for arg in args_to_remove:
        sys.argv.remove(arg)
        
    args = parser.parse_args()
    
    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date
    
    if args.start and args.end:
        try:
            start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
            end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
        except ValueError:
            print("❌ 日期格式错误，请使用 YYYY-MM-DD")
            sys.exit(1)
    elif args_to_remove:
        # 如果使用了 -N 参数
        start_date = datetime.now().date() - timedelta(days=days_back)
        end_date = datetime.now().date() - timedelta(days=1)
    
    return start_date, end_date

def load_business_definition(file_path):
    """加载业务定义文件"""
    if not os.path.exists(file_path):
        print(f"❌ 错误: 业务定义文件不存在 - {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载业务定义失败: {e}")
        return None

def load_data(file_path):
    """加载 Parquet 数据"""
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件不存在 - {file_path}")
        return None
    
    try:
        print(f"正在加载数据: {file_path}")
        df = pd.read_parquet(file_path)
        print(f"✅ 数据加载成功，共 {len(df)} 行")
        return df
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def analyze_daily_lock_orders(df, start_date, end_date):
    """
    分析锁单数据 (支持时间范围)
    """
    print(f"正在分析 {start_date} 至 {end_date} 的锁单数据...")
    
    # 加载业务定义以获取电池容量映射
    business_def = load_business_definition(BUSINESS_DEF_FILE)
    product_to_capacity = {}
    if business_def and "battery_capacity" in business_def:
        for capacity, products in business_def["battery_capacity"].items():
            for product in products:
                product_to_capacity[product] = capacity

    # 确保必要的列存在
    # 更新为新数据集的列名
    required_columns = ['lock_time', 'order_number', 'series', 'product_name']
    for col in required_columns:
        if col not in df.columns:
            print(f"❌ 错误: 数据缺失列 {col}")
            return None

    # 数据预处理
    df_copy = df.copy()
    df_copy['lock_time'] = pd.to_datetime(df_copy['lock_time'], errors='coerce').dt.date
    
    # 筛选目标日期范围的锁单数据
    daily_orders = df_copy[
        (df_copy['lock_time'] >= start_date) & 
        (df_copy['lock_time'] <= end_date)
    ]
    
    # 1. 计算总锁单数 (基于 order_number 去重)
    total_lock_count = daily_orders['order_number'].nunique()
    
    # 2. 分车型统计
    model_stats = {}
    for model in TARGET_MODELS:
        model_df = daily_orders[daily_orders['series'] == model]
        count = model_df['order_number'].nunique()
        
        stats = {"count": count}
        
        # 对 LS6 (原CM2) 和 LS9 进行电池容量细分
        if model in ["LS6", "LS9"]:
            capacity_counts = {"52kwh": 0, "66kwh": 0}
            # 只有当 product_to_capacity 存在时才进行细分
            if product_to_capacity:
                # 获取去重后的订单号及其对应的 product_name
                unique_orders = model_df[['order_number', 'product_name']].drop_duplicates('order_number')
                
                for _, row in unique_orders.iterrows():
                    p_name = row['product_name']
                    cap = product_to_capacity.get(p_name)
                    if cap in ["52kwh", "66kwh"]:
                        capacity_counts[cap] += 1
            
            stats["details"] = capacity_counts
            
        model_stats[model] = stats
        
    return {
        "start_date": start_date,
        "end_date": end_date,
        "total": total_lock_count,
        "models": model_stats
    }

def analyze_daily_invoice_orders(df, start_date, end_date):
    """
    分析开票数据 (基于 Invoice_Upload_Time)
    定义：有 Invoice_Upload_Time 且有 Lock_Time 的 Order Number 数
    """
    print(f"正在分析 {start_date} 至 {end_date} 的开票数据...")
    
    # 确保必要的列存在
    # 更新为新数据集的列名
    required_columns = ['invoice_upload_time', 'lock_time', 'order_number', 'series', 'invoice_amount']
    for col in required_columns:
        if col not in df.columns:
            print(f"❌ 错误: 数据缺失列 {col}")
            return None

    # 数据预处理
    df_copy = df.copy()
    df_copy['invoice_upload_time'] = pd.to_datetime(df_copy['invoice_upload_time'], errors='coerce').dt.date
    
    # 筛选条件：
    # 1. invoice_upload_time 在目标日期范围内
    # 2. lock_time 不为空 (题目要求：有 invoice_upload_time 且有 lock_time)
    invoice_orders = df_copy[
        (df_copy['invoice_upload_time'] >= start_date) & 
        (df_copy['invoice_upload_time'] <= end_date) &
        (df_copy['lock_time'].notna())
    ]
    
    # 1. 计算总开票数 (基于 order_number 去重)
    total_invoice_count = invoice_orders['order_number'].nunique()
    
    # 2. 分车型统计
    model_invoice_stats = {}
    for model in TARGET_MODELS:
        model_df = invoice_orders[invoice_orders['series'] == model]
        count = model_df['order_number'].nunique()
        
        # 计算该车型的平均开票价格
        model_valid_prices = model_df[
            (model_df['invoice_amount'].notna()) & 
            (model_df['invoice_amount'] > 0)
        ]['invoice_amount']
        avg_price = model_valid_prices.mean() if not model_valid_prices.empty else 0
        
        model_invoice_stats[model] = {
            "count": count,
            "avg_price": avg_price
        }
        
    return {
        "start_date": start_date,
        "end_date": end_date,
        "total": total_invoice_count,
        "models": model_invoice_stats
    }

def send_feishu_notification(lock_stats, invoice_stats):
    """发送飞书通知"""
    if not WEBHOOK_URL:
        print("❌ 错误: 未设置 FS_WEBHOOK_URL 环境变量，跳过发送消息")
        return

    # 构建标题日期字符串
    start_date = lock_stats['start_date']
    end_date = lock_stats['end_date']
    if start_date == end_date:
        date_str = str(start_date)
        title_prefix = "每日"
        lock_label = "昨日锁单数"
        invoice_label = "昨日开票数"
    else:
        date_str = f"{start_date} ~ {end_date}"
        title_prefix = "阶段性"
        lock_label = "期间锁单数"
        invoice_label = "期间开票数"

    # 构建锁单明细文本
    lock_model_details = []
    for model, stats in lock_stats['models'].items():
        count = stats["count"]
        detail_str = ""
        if "details" in stats:
            d = stats["details"]
            detail_parts = []
            if "52kwh" in d:
                detail_parts.append(f"52kw：{d['52kwh']}")
            if "66kwh" in d:
                detail_parts.append(f"66kw：{d['66kwh']}")
            if detail_parts:
                detail_str = "｜" + "，".join(detail_parts)
        lock_model_details.append(f"- {model}: {count} 单{detail_str}")
    lock_model_text = "\n".join(lock_model_details)

    # 构建开票明细文本
    invoice_model_details = []
    for model, info in invoice_stats['models'].items():
        price_str = f"{info['avg_price']/10000:.1f}w" if info['avg_price'] > 0 else "N/A"
        invoice_model_details.append(f"- {model}: {info['count']} 台｜平均开票价格：{price_str}")
    invoice_model_text = "\n".join(invoice_model_details)

    # 构建卡片内容
    card_content = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": f"📊 {title_prefix}业务数据观察 ({date_str})"
                },
                "template": "blue"
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**{lock_label}：** {lock_stats['total']}\n{lock_model_text}"
                    }
                },
                {
                    "tag": "hr"
                },
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**{invoice_label}：** {invoice_stats['total']} 台\n{invoice_model_text}"
                    }
                },
                {
                    "tag": "hr"
                },
                {
                    "tag": "note",
                    "elements": [
                        {
                            "tag": "plain_text",
                            "content": f"统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        }
                    ]
                }
            ]
        }
    }

    try:
        response = requests.post(WEBHOOK_URL, json=card_content)
        response.raise_for_status()
        result = response.json()
        if result.get("StatusCode") == 0:
            print("✅ 飞书消息发送成功")
        else:
            print(f"❌ 飞书消息发送异常: {result}")
    except Exception as e:
        print(f"❌ 发送飞书消息失败: {e}")

def main():
    # 0. 解析参数
    start_date, end_date = parse_arguments()
    
    # 1. 加载数据
    df = load_data(PARQUET_FILE)
    if df is None:
        return

    # 2. 分析数据
    lock_stats = analyze_daily_lock_orders(df, start_date, end_date)
    invoice_stats = analyze_daily_invoice_orders(df, start_date, end_date)
    
    if lock_stats and invoice_stats:
        # 打印结果到控制台
        print("\n" + "="*30)
        if start_date == end_date:
            print(f"📅 日期: {start_date}")
        else:
            print(f"📅 日期范围: {start_date} ~ {end_date}")
            
        print(f" 总锁单数: {lock_stats['total']}")
        print("   车型分布:")
        for model, stats in lock_stats['models'].items():
            count = stats["count"]
            detail_str = ""
            if "details" in stats:
                d = stats["details"]
                detail_parts = []
                if "52kwh" in d:
                    detail_parts.append(f"52kw：{d['52kwh']}")
                if "66kwh" in d:
                    detail_parts.append(f"66kw：{d['66kwh']}")
                if detail_parts:
                    detail_str = "｜" + "，".join(detail_parts)
            print(f"   - {model}: {count}{detail_str}")
            
        print("-" * 30)
        
        print(f"🚚 总开票数: {invoice_stats['total']} 台")
        print("   车型分布 (开票):")
        for model, info in invoice_stats['models'].items():
            price_display = f"{info['avg_price']/10000:.1f}w" if info['avg_price'] > 0 else "N/A"
            print(f"   - {model}: {info['count']} 台｜平均开票价格：{price_display}")
        print("="*30 + "\n")

        # 3. 发送飞书通知
        send_feishu_notification(lock_stats, invoice_stats)

if __name__ == "__main__":
    main()
