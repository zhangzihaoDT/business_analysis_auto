#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
线索表转置：生成“一天一行”的数据集

输入：原始导出的线索 CSV（默认自动查找 original/ 或 formatted/ 下最新的
      leads_structure_expert(.csv|_*.csv) 或 线索表_*.csv / 线索表_导出.csv）
输出：processed/ 下按天聚合后的 CSV（默认 leads_daily_时间戳.csv）

策略：
1) 优先识别显式日期列（如：日期、Date、创建日期、线索时间等），按天聚合：
   - 数值列求和；
   - 增加 leads_count（当日记录数）。
2) 若无日期列，则尝试宽表转长表：列名若匹配日期模式（YYYY-MM-DD/YYYY/MM/DD 等），melt 为 (date,value) 再按天求和。
"""

import argparse
import os
import sys
import glob
import re
from datetime import datetime
import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("leads_transform.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


DATE_NAME_CANDIDATES = [
    '日期', 'date', 'Date', 'DAY', 'day', 'dt', '创建日期', '线索时间', '线索日期', '线索创建时间',
    'created_at', 'createdAt', 'create_time', '时间', 'lc_create_time'
]

DATE_HEADER_REGEX = re.compile(r"^(\d{4}[-/.]\d{2}[-/.]\d{2})$")


def load_csv_with_encoding(path):
    for enc in ('utf-8', 'utf-8-sig', 'gbk'):
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info(f"使用编码 {enc} 读取成功: {path}")
            return df
        except Exception as e:
            logger.debug(f"编码 {enc} 读取失败: {e}")
    # 最后一次尝试默认
    df = pd.read_csv(path)
    logger.info(f"使用默认编码读取成功: {path}")
    return df


def find_latest_leads_csv():
    candidates = []
    # original 优先（新命名）
    candidates.extend(glob.glob(os.path.join('original', 'leads_structure_expert.csv')))
    candidates.extend(glob.glob(os.path.join('original', 'leads_structure_expert_*.csv')))
    # legacy 命名
    candidates.extend(glob.glob(os.path.join('original', '线索表_*.csv')))
    candidates.extend(glob.glob(os.path.join('original', '线索表_导出.csv')))
    # 回退到 formatted（兼容历史）
    candidates.extend(glob.glob(os.path.join('formatted', 'leads_structure_expert.csv')))
    candidates.extend(glob.glob(os.path.join('formatted', 'leads_structure_expert_*.csv')))
    candidates.extend(glob.glob(os.path.join('formatted', '线索表_*.csv')))
    candidates.extend(glob.glob(os.path.join('formatted', '线索表_导出.csv')))

    if not candidates:
        return None

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    latest = candidates[0]
    logger.info(f"自动选择最新文件: {latest}")
    return latest


def _can_parse_datetime(series, sample_size=50, threshold=0.7):
    try:
        vals = series.dropna().astype(str).head(sample_size)
        if vals.empty:
            return False
        parsed = pd.to_datetime(vals, errors='coerce', infer_datetime_format=True)
        rate = parsed.notna().mean()
        return rate >= threshold
    except Exception:
        return False


def detect_date_column(df, user_col=None):
    if user_col and user_col in df.columns and _can_parse_datetime(df[user_col]):
        return user_col

    # 名称优先
    for name in DATE_NAME_CANDIDATES:
        if name in df.columns and _can_parse_datetime(df[name]):
            return name

    # 任意可解析列
    for col in df.columns:
        if _can_parse_datetime(df[col]):
            return col

    # 针对中文日期字符串列（如：2023年8月24日）
    for col in df.columns:
        s = df[col].astype(str).dropna().head(50)
        if s.str.contains(r"\d{4}年\d{1,2}月\d{1,2}日").any():
            return col

    # 列名包含“日(”或“日期(”的情况（如：日(lc_create_time)）
    for col in df.columns:
        if isinstance(col, str) and ('日(' in col or '日期(' in col):
            return col

    return None


def _parse_date_series(series):
    s = series.astype(str).str.strip()
    # 中文日期格式：YYYY年M月D日
    mask = s.str.contains(r"\d{4}年\d{1,2}月\d{1,2}日")
    if mask.any():
        parts = s.str.extract(r"(?P<y>\d{4})年(?P<m>\d{1,2})月(?P<d>\d{1,2})日")
        date_str = parts['y'] + '-' + parts['m'].str.zfill(2) + '-' + parts['d'].str.zfill(2)
        return pd.to_datetime(date_str, errors='coerce')
    # 其他常见分隔符
    s2 = s.str.replace('.', '-', regex=False).str.replace('/', '-', regex=False)
    return pd.to_datetime(s2, errors='coerce')


def pivot_metrics_by_date(df, date_col):
    df = df.copy()
    dt = _parse_date_series(df[date_col])
    df['date'] = dt.dt.date

    # 标准化列名以适配常见导出结构
    metric_name_col = None
    metric_value_col = None
    # 优先中文
    if '度量名称' in df.columns:
        metric_name_col = '度量名称'
    elif 'metric_name' in df.columns:
        metric_name_col = 'metric_name'

    if '度量值' in df.columns:
        metric_value_col = '度量值'
    elif 'metric_value' in df.columns:
        metric_value_col = 'metric_value'

    # 若没有标准结构，则退化为按天计数
    if not (metric_name_col and metric_value_col):
        grouped = df.groupby('date', dropna=True).size().to_frame('leads_count').reset_index()
        return grouped.sort_values('date')

    # 处理千分位等字符
    df[metric_value_col] = (
        df[metric_value_col]
        .astype(str)
        .str.replace(',', '', regex=False)
    )
    df[metric_value_col] = pd.to_numeric(df[metric_value_col], errors='coerce')

    # 透视到“一天一行，指标为列”
    pv = (
        df.pivot_table(
            index='date',
            columns=metric_name_col,
            values=metric_value_col,
            aggfunc='sum'
        )
        .reset_index()
    )

    # 列排序：date 在最前，其它按字典序
    cols = ['date'] + [c for c in pv.columns if c != 'date']
    pv = pv[cols]
    return pv.sort_values('date')


def melt_wide_dates(df):
    # 识别列名为日期的列
    date_cols = [c for c in df.columns if isinstance(c, str) and DATE_HEADER_REGEX.match(c)]
    if not date_cols:
        return None

    long_df = df.melt(value_vars=date_cols, var_name='date', value_name='value')
    long_df['date'] = pd.to_datetime(long_df['date'].str.replace('.', '-', regex=False).str.replace('/', '-', regex=False), errors='coerce')
    long_df['value'] = pd.to_numeric(long_df['value'], errors='coerce')
    agg = long_df.groupby(long_df['date'].dt.date)['value'].sum(min_count=1).reset_index()
    agg = agg.rename(columns={'value': 'value_total', 'date': 'date'})
    agg = agg.sort_values('date')
    return agg


def main():
    parser = argparse.ArgumentParser(description='线索表转置为按天一行的数据集')
    parser.add_argument('--input', help='输入 CSV 路径（默认自动查找 original/ 或 formatted/ 最新的线索表）')
    parser.add_argument('--output', help='输出 CSV 路径（默认 processed/leads_daily_时间戳.csv）')
    parser.add_argument('--date-column', help='指定日期列名（可选）')

    args = parser.parse_args()

    input_path = args.input or find_latest_leads_csv()
    if not input_path or not os.path.exists(input_path):
        logger.error('未找到可用的线索表 CSV，请先导出或指定 --input')
        return 1

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = args.output or os.path.join('processed', f'leads_daily_{ts}.csv')

    # 确保输出目录
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger.info(f'读取: {input_path}')
    df = load_csv_with_encoding(input_path)
    logger.info(f'数据形状: {df.shape}, 列: {list(df.columns)[:10]}...')

    date_col = detect_date_column(df, user_col=args.date_column)
    if date_col:
        logger.info(f'识别到日期列: {date_col}，转置为一天一行（指标为列）')
        out_df = pivot_metrics_by_date(df, date_col)
    else:
        logger.warning('未识别到日期列，尝试宽表转长表基于列名（日期格式）')
        out_df = melt_wide_dates(df)
        if out_df is None:
            logger.error('无法识别可用的日期信息，转置失败。请手动指定 --date-column 或检查数据结构。')
            return 1

    out_df.to_csv(output_path, index=False)
    logger.info(f'已写出: {output_path}，行数: {len(out_df)}')
    print(f'✅ 转置完成 -> {output_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())