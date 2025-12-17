#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
每日锁单数据观察脚本
功能：
1. 读取 intention_order_analysis.parquet 数据
2. 计算昨日（T-1）的锁单数
3. 统计指定车型（CM2, DM1, LS9）的锁单情况
4. 发送飞书通知
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置常量
PARQUET_FILE = "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"
TARGET_MODELS = ["CM2", "DM1", "LS9"]
WEBHOOK_URL = os.getenv("FS_WEBHOOK_URL")

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

def analyze_daily_lock_orders(df, target_date=None):
    """
    分析日锁单数据
    """
    if target_date is None:
        target_date = datetime.now().date() - timedelta(days=1)
    
    print(f"正在分析 {target_date} 的锁单数据...")
    
    # 确保必要的列存在
    required_columns = ['Lock_Time', 'Order Number', '车型分组']
    for col in required_columns:
        if col not in df.columns:
            print(f"❌ 错误: 数据缺失列 {col}")
            return None

    # 数据预处理
    df_copy = df.copy()
    df_copy['Lock_Time'] = pd.to_datetime(df_copy['Lock_Time'], errors='coerce').dt.date
    
    # 筛选目标日期的锁单数据
    daily_orders = df_copy[df_copy['Lock_Time'] == target_date]
    
    # 1. 计算总锁单数 (基于 Order Number 去重)
    total_lock_count = daily_orders['Order Number'].nunique()
    
    # 2. 分车型统计
    model_stats = {}
    for model in TARGET_MODELS:
        model_df = daily_orders[daily_orders['车型分组'] == model]
        count = model_df['Order Number'].nunique()
        model_stats[model] = count
        
    return {
        "date": target_date,
        "total": total_lock_count,
        "models": model_stats
    }

def analyze_daily_delivery_orders(df, target_date=None):
    """
    分析日交付数据 (基于 Invoice_Upload_Time)
    定义：有 Invoice_Upload_Time 且有 Lock_Time 的 Order Number 数
    """
    if target_date is None:
        target_date = datetime.now().date() - timedelta(days=1)
    
    print(f"正在分析 {target_date} 的交付数据...")
    
    # 确保必要的列存在
    required_columns = ['Invoice_Upload_Time', 'Lock_Time', 'Order Number', '车型分组']
    for col in required_columns:
        if col not in df.columns:
            print(f"❌ 错误: 数据缺失列 {col}")
            return None

    # 数据预处理
    df_copy = df.copy()
    df_copy['Invoice_Upload_Time'] = pd.to_datetime(df_copy['Invoice_Upload_Time'], errors='coerce').dt.date
    
    # 筛选条件：
    # 1. Invoice_Upload_Time 为目标日期
    # 2. Lock_Time 不为空 (题目要求：有 Invoice_Upload_Time 且有 Lock_Time)
    # 注意：这里我们假设 Lock_Time 只要非空即可，不限制必须在目标日期之前（虽然业务上通常如此）
    delivery_orders = df_copy[
        (df_copy['Invoice_Upload_Time'] == target_date) & 
        (df_copy['Lock_Time'].notna())
    ]
    
    # 1. 计算总交付数 (基于 Order Number 去重)
    total_delivery_count = delivery_orders['Order Number'].nunique()
    
    # 2. 分车型统计
    model_stats = {}
    for model in TARGET_MODELS:
        model_df = delivery_orders[delivery_orders['车型分组'] == model]
        count = model_df['Order Number'].nunique()
        
        # 计算该车型的平均开票价格
        model_valid_prices = model_df[
            (model_df['开票价格'].notna()) & 
            (model_df['开票价格'] > 0)
        ]['开票价格']
        avg_price = model_valid_prices.mean() if not model_valid_prices.empty else 0
        
        model_stats[model] = {
            "count": count,
            "avg_price": avg_price
        }
        
    return {
        "date": target_date,
        "total": total_delivery_count,
        "models": model_stats
    }

def send_feishu_notification(lock_stats, delivery_stats):
    """发送飞书通知"""
    if not WEBHOOK_URL:
        print("❌ 错误: 未设置 FS_WEBHOOK_URL 环境变量，跳过发送消息")
        return

    # 构建锁单明细文本
    lock_model_details = []
    for model, count in lock_stats['models'].items():
        lock_model_details.append(f"- {model}: {count} 单")
    lock_model_text = "\n".join(lock_model_details)

    # 构建交付明细文本
    delivery_model_details = []
    for model, info in delivery_stats['models'].items():
        price_str = f"{info['avg_price']/10000:.1f}w" if info['avg_price'] > 0 else "N/A"
        delivery_model_details.append(f"- {model}: {info['count']} 台｜平均开票价格：{price_str}")
    delivery_model_text = "\n".join(delivery_model_details)

    # 构建卡片内容
    card_content = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": f"📊 每日业务数据观察 ({lock_stats['date']})"
                },
                "template": "blue"
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**昨日锁单数：** {lock_stats['total']}\n{lock_model_text}"
                    }
                },
                {
                    "tag": "hr"
                },
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**昨日交付数：** {delivery_stats['total']} 台\n{delivery_model_text}"
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
    # 1. 加载数据
    df = load_data(PARQUET_FILE)
    if df is None:
        return

    # 2. 分析数据
    # 默认分析昨天，也可以通过参数指定（这里先简单实现默认逻辑）
    lock_stats = analyze_daily_lock_orders(df)
    delivery_stats = analyze_daily_delivery_orders(df)
    
    if lock_stats and delivery_stats:
        # 打印结果到控制台
        print("\n" + "="*30)
        print(f"📅 日期: {lock_stats['date']}")
        print(f"� 总锁单数: {lock_stats['total']}")
        print("   车型分布:")
        for model, count in lock_stats['models'].items():
            print(f"   - {model}: {count}")
            
        print("-" * 30)
        
        print(f"🚚 总交付数: {delivery_stats['total']} 台")
        print("   车型分布:")
        for model, info in delivery_stats['models'].items():
            price_display = f"{info['avg_price']/10000:.1f}w" if info['avg_price'] > 0 else "N/A"
            print(f"   - {model}: {info['count']} 台｜平均开票价格：{price_display}")
        print("="*30 + "\n")

        # 3. 发送飞书通知
        send_feishu_notification(lock_stats, delivery_stats)

if __name__ == "__main__":
    main()
