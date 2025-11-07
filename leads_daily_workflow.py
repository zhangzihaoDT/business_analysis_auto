#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leads 一键工作流脚本（导出 + 日级转置）

功能：
- 步骤 1：调用 leads_table_export.py 导出线索表（默认 CSV）
- 步骤 2：调用 transform_leads_daily.py 生成“一天一行”的日级数据集

用法示例：
- 默认一键导出并转置（使用统一时间戳命名）：
  python scripts/leads_daily_workflow.py --verbose

- 指定导出文件与转置输出文件：
  python scripts/leads_daily_workflow.py \
    --export-output /Users/zihao_/Documents/coding/dataset/original/leads_structure_expert_20250101_120000.csv \
    --daily-output /Users/zihao_/Documents/coding/dataset/processed/leads_daily_20250101_120000.csv

- 使用个人访问令牌（PAT）：
  python scripts/leads_daily_workflow.py --token-name <NAME> --token-value <VALUE>

备注：
- 导出脚本：scripts/leads_table_export.py
- 转置脚本：scripts/transform_leads_daily.py
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def run_subprocess(command, cwd=None):
    """运行子进程并实时打印输出，返回退出码和输出。"""
    print(f"🚀 执行命令: {' '.join(command)}")
    if cwd:
        print(f"📁 工作目录: {cwd}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
        )
        output_lines = []
        while True:
            line = process.stdout.readline()
            if line == '' and process.poll() is not None:
                break
            if line:
                print(line.rstrip())
                output_lines.append(line)
        return process.returncode, ''.join(output_lines)
    except Exception as e:
        print(f"❌ 子进程执行失败: {e}")
        return -1, ""


def default_paths(timestamp: str):
    """生成默认的导出和日级输出路径（使用统一时间戳）。"""
    export_path = os.path.join(PROJECT_DIR, 'original', f'leads_structure_expert_{timestamp}.csv')
    daily_path = os.path.join(PROJECT_DIR, 'processed', f'leads_daily_{timestamp}.csv')
    return export_path, daily_path


def main():
    parser = argparse.ArgumentParser(description='Leads 导出 + 日级转置 一键工作流')

    # 输出路径配置
    parser.add_argument('--export-output', help='导出 CSV 文件路径（默认 original/leads_structure_expert_时间戳.csv）')
    parser.add_argument('--daily-output', help='日级转置后 CSV 文件路径（默认 processed/leads_daily_时间戳.csv）')

    # 导出相关参数透传
    parser.add_argument('--server', default='http://tableau.immotors.com', help='Tableau 服务器 URL')
    parser.add_argument('--username', default='analysis', help='Tableau 用户名（PAT 时忽略）')
    parser.add_argument('--password', default='analysis888', help='Tableau 密码（PAT 时忽略）')
    parser.add_argument('--token-name', help='个人访问令牌名称')
    parser.add_argument('--token-value', help='个人访问令牌值')
    parser.add_argument('--view', default='http://tableau.immotors.com/#/views/165/leads_structure_analysis?:iid=2', help='导出的视图路径或完整 URL')
    parser.add_argument('--timeout', type=int, default=600, help='导出操作超时时间（秒），默认 600')

    # 转置相关参数
    parser.add_argument('--date-column', help='指定日期列名（可选，默认自动识别）')

    # 展示
    parser.add_argument('--verbose', action='store_true', help='显示更详细的进度信息')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_output, daily_output = default_paths(timestamp)

    if args.export_output:
        export_output = args.export_output
    if args.daily_output:
        daily_output = args.daily_output

    # 步骤 1：导出
    print("\n" + "="*60)
    print("步骤 1/2：导出 Leads 线索表数据")
    print("="*60)
    print(f"📁 导出文件：{export_output}")
    print(f"⏱️ 超时设置：{args.timeout} 秒")

    export_cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, 'leads_table_export.py'),
        '--server', args.server,
        '--view', args.view,
        '--output', export_output,
        '--format', 'csv',
        '--timeout', str(args.timeout),
    ]
    # 选择凭证方式
    if args.token_name and args.token_value:
        export_cmd.extend(['--token-name', args.token_name, '--token-value', args.token_value])
    else:
        export_cmd.extend(['--username', args.username, '--password', args.password])
    # 详细日志
    if args.verbose:
        export_cmd.append('--verbose')

    rc, _out = run_subprocess(export_cmd, cwd=PROJECT_DIR)
    if rc != 0:
        print("💥 导出阶段失败，工作流终止。")
        sys.exit(1)

    # 步骤 2：转置
    print("\n" + "="*60)
    print("步骤 2/2：转置为日级数据并保存")
    print("="*60)
    print(f"📥 转置输入：{export_output}")
    print(f"📤 转置输出：{daily_output}")

    transform_cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, 'transform_leads_daily.py'),
        '--input', export_output,
        '--output', daily_output,
    ]
    if args.date_column:
        transform_cmd.extend(['--date-column', args.date_column])

    rc, _out = run_subprocess(transform_cmd, cwd=PROJECT_DIR)
    if rc != 0:
        print("💥 转置阶段失败，工作流终止。")
        sys.exit(1)

    # 完成
    print("\n" + "="*60)
    print("✅ Leads 工作流完成")
    print(f"📁 导出文件：{export_output}")
    print(f"📁 日级文件：{daily_output}")
    print("="*60)


if __name__ == '__main__':
    main()