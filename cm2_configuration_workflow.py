#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CM2 配置数据一键工作流脚本

功能：
- 第一步：调用 Tableau 导出 CM2 配置详情数据（configuration_details_CM2）
- 第二步：调用转置处理脚本，对导出数据进行转置和清洗

用法示例：
- 默认一键导出并转置（使用时间戳命名）：
  python cm2_configuration_workflow.py

- 指定导出文件与转置输出文件：
  python cm2_configuration_workflow.py \
    --export-output /Users/zihao_/Documents/coding/dataset/original/CM2_Configuration_Details_20250101_120000.csv \
    --transpose-output /Users/zihao_/Documents/coding/dataset/processed/CM2_Configuration_Details_transposed_20250101_120000.csv

- 跳过转置清洗步骤：
  python cm2_configuration_workflow.py --skip-cleaning

备注：
- 导出的视图为 17/configuration_details_CM2（由 export_cm2_configuration_data.py 调用 tableau_export.py 完成）
- 转置处理由 transpose_cm2_data.py 完成，支持 --skip-cleaning、--log-level 等参数
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

# 工作目录固定为脚本目录，方便相对导入和子进程执行
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# 允许导入同目录下的导出模块
sys.path.append(SCRIPT_DIR)

try:
    from export_cm2_configuration_data import export_cm2_configuration_data
except Exception as e:
    print(f"⚠️ 无法导入导出模块 export_cm2_configuration_data: {e}")
    export_cm2_configuration_data = None


def run_subprocess(command, cwd=None):
    """运行子进程并实时打印输出，返回退出码。"""
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
    """生成默认的导出和转置输出路径（复用同一时间戳便于对应）。"""
    export_path = os.path.join(
        PROJECT_DIR,
        'original',
        f'CM2_Configuration_Details_{timestamp}.csv'
    )
    transpose_path = os.path.join(
        PROJECT_DIR,
        'processed',
        f'CM2_Configuration_Details_transposed_{timestamp}.csv'
    )
    return export_path, transpose_path


def main():
    parser = argparse.ArgumentParser(
        description='CM2 配置数据导出 + 转置 一键工作流'
    )

    parser.add_argument(
        '--export-output',
        help='导出 CSV 文件路径（默认按时间戳生成到 original/）'
    )
    parser.add_argument(
        '--transpose-output',
        help='转置后 CSV 文件路径（默认按时间戳生成到 processed/）'
    )
    parser.add_argument(
        '--timeout', type=int, default=500,
        help='导出操作超时时间（秒），默认 500'
    )
    parser.add_argument(
        '--skip-cleaning', action='store_true',
        help='跳过转置后的清洗步骤'
    )
    parser.add_argument(
        '--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
        help='转置脚本日志级别（默认 INFO）'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='显示更详细的进度信息'
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_output, transpose_output = default_paths(timestamp)

    if args.export_output:
        export_output = args.export_output
    if args.transpose_output:
        transpose_output = args.transpose_output

    # 第一步：导出
    print("\n" + "="*60)
    print("步骤 1/2：导出 CM2 配置详情数据")
    print("="*60)
    print(f"📁 导出文件：{export_output}")
    print(f"⏱️ 超时设置：{args.timeout} 秒")

    if export_cm2_configuration_data is None:
        print("⚠️ 未能导入导出模块，改用子进程调用现有脚本。")
        rc, _ = run_subprocess(
            [
                sys.executable, os.path.join(SCRIPT_DIR, 'export_cm2_configuration_data.py'),
                '--output', export_output,
                '--timeout', str(args.timeout)
            ],
            cwd=SCRIPT_DIR
        )
        if rc != 0:
            print("💥 导出阶段失败，工作流终止。")
            sys.exit(1)
    else:
        ok = export_cm2_configuration_data(output_file=export_output, timeout=args.timeout)
        if not ok:
            print("💥 导出阶段失败，工作流终止。")
            sys.exit(1)

    # 第二步：转置
    print("\n" + "="*60)
    print("步骤 2/2：转置并保存处理结果")
    print("="*60)
    print(f"📥 转置输入：{export_output}")
    print(f"📤 转置输出：{transpose_output}")

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, 'transpose_cm2_data.py'),
        '-i', export_output,
        '-o', transpose_output,
        '--log-level', args.log_level
    ]
    if args.skip_cleaning:
        cmd.append('--skip-cleaning')

    rc, _ = run_subprocess(cmd, cwd=SCRIPT_DIR)
    if rc != 0:
        print("💥 转置阶段失败，工作流终止。")
        sys.exit(1)

    # 完成
    print("\n" + "="*60)
    print("✅ 工作流完成")
    print(f"📁 导出文件：{export_output}")
    print(f"📁 转置文件：{transpose_output}")
    print("="*60)


if __name__ == '__main__':
    main()