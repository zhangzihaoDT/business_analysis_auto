#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
线索表数据获取脚本（城市视图增量版）

功能增强：
- 针对视图 `leads_assign_city_store2_1` 做按日增量刷新。
- 每次运行仅覆盖刷新：现有数据的 max(日(lc_assign_time_min)) - N 天（默认 N=2）之后的数据。
- 保留更早的数据不变，合并新导出并去重。

使用方式示例：
python3 scripts/leads_table_export_city.py \
  --view "http://tableau.immotors.com/#/views/165/leads_assign_city_store2_1?:iid=1" \
  --output "original/leads_assign_city_store2_1.csv" \
  --days-offset 2 --verbose
"""

import argparse
import os
import subprocess
import logging
import sys
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

# 配置日志（单独日志文件，避免与基础版混淆）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("leads_export_city.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


DATE_COL = '日(lc_assign_time_min)'


def run_command(command, timeout=300, show_progress=True):
    """执行命令并返回结果，支持超时和进度显示"""
    logger.debug(f"执行命令: {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        start_time = datetime.now()
        progress_shown = False
        stdout_data = []
        stderr_data = []

        import select
        import time
        import sys as _sys

        for pipe in [process.stdout, process.stderr]:
            if pipe:
                os.set_blocking(pipe.fileno(), False)

        while process.poll() is None:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if timeout and elapsed_time > timeout:
                process.terminate()
                logger.warning(f"命令执行超时 ({timeout}秒)，已终止")
                return -1, "".join(stdout_data), f"命令执行超时 ({timeout}秒)，已终止"

            readable, _, _ = select.select([process.stdout, process.stderr], [], [], 0.5)
            for pipe in readable:
                line = pipe.readline()
                if line:
                    if pipe == process.stdout:
                        stdout_data.append(line)
                    else:
                        stderr_data.append(line)

            if show_progress and elapsed_time > 2:
                if not progress_shown or int(elapsed_time) % 10 == 0:
                    progress_shown = True
                    elapsed = int(elapsed_time)
                    _sys.stdout.write(f"\r正在执行命令... 已用时 {elapsed} 秒")
                    _sys.stdout.flush()

            time.sleep(0.1)

        stdout, stderr = process.communicate()
        stdout_data.append(stdout)
        stderr_data.append(stderr)

        if show_progress and progress_shown:
            _sys.stdout.write("\r" + " " * 50 + "\r")
            _sys.stdout.flush()

        stdout_result = "".join(stdout_data)
        stderr_result = "".join(stderr_data)

        if process.returncode != 0:
            logger.error(f"命令执行失败: {stderr_result}")
        else:
            logger.debug(f"命令执行成功: {stdout_result}")

        return process.returncode, stdout_result, stderr_result
    except Exception as e:
        logger.error(f"执行命令时发生错误: {str(e)}")
        return -1, "", str(e)


def login_tableau(server, username: Optional[str] = None, password: Optional[str] = None,
                  token_name: Optional[str] = None, token_value: Optional[str] = None) -> bool:
    """登录Tableau服务器"""
    logger.info(f"正在登录Tableau服务器: {server}")
    server = server.split('#')[0] if '#' in server else server

    if token_name and token_value:
        logger.info("使用个人访问令牌(PAT)登录")
        command = ["tabcmd", "login", "-s", server, "--token-name", token_name, "--token-value", token_value]
    else:
        logger.info(f"使用用户名密码登录: {username}")
        command = ["tabcmd", "login", "-s", server, "-u", username or "", "-p", password or ""]

    returncode, stdout, stderr = run_command(command)
    if returncode == 0:
        logger.info("登录成功")
        return True
    else:
        logger.error(f"登录失败: {stderr}")
        return False


def export_view(view_path, output_file, format="csv", timeout=600, show_progress=True,
                server=None, username=None, password=None, token_name=None, token_value=None) -> bool:
    """导出Tableau视图（复用基础版逻辑）"""
    logger.info(f"正在导出视图: {view_path} 到 {output_file}")

    if show_progress:
        print(f"开始导出 Tableau 视图: {view_path}")
        print(f"导出格式: {format}, 输出文件: {output_file}")
        print(f"超时设置: {timeout} 秒")
        print("导出过程可能需要几分钟，请耐心等待...")

    if view_path.startswith('http'):
        try:
            from urllib.parse import urlparse, unquote
            parsed_url = urlparse(view_path)
            path_parts = parsed_url.fragment.strip('/').split('/')

            if len(path_parts) >= 2 and path_parts[0] == 'views':
                sheet_part = path_parts[2] if len(path_parts) > 2 else path_parts[1]
                sheet_part = unquote(sheet_part.split('?', 1)[0])
                workbook_part = unquote(path_parts[1])
                tableau_path = f"{workbook_part}/{sheet_part}" if len(path_parts) > 2 else sheet_part
                logger.info(f"从URL提取的路径: {tableau_path}")
                if show_progress:
                    print(f"从URL提取的视图路径: {tableau_path}")
            else:
                tableau_path = view_path
                logger.info(f"使用完整URL: {tableau_path}")
        except Exception as e:
            logger.warning(f"从URL提取路径失败: {str(e)}，将使用原始URL")
            tableau_path = view_path
    else:
        paths_to_try = [view_path, view_path.replace(" ", ""), view_path.replace(" ", "%20")]
        parts = view_path.split('/')
        if len(parts) == 2:
            workbook, sheet = parts
            paths_to_try.append(f"{workbook.replace(' ', '')}/{sheet.replace(' ', '')}")
        logger.info(f"将尝试以下路径: {paths_to_try}")
        if show_progress:
            print(f"将尝试以下路径: {', '.join(paths_to_try)}")
        tableau_path = paths_to_try[0].split('?', 1)[0]

    if show_progress:
        print(f"正在尝试导出视图: {tableau_path}")

    command = ["tabcmd", "export", tableau_path, f"--{format}", "-f", output_file]
    if server:
        command.extend(["-s", server])
    if token_name and token_value:
        command.extend(["--token-name", token_name, "--token-value", token_value])
    elif username and password:
        command.extend(["-u", username, "-p", password])

    returncode, stdout, stderr = run_command(command, timeout=timeout, show_progress=show_progress)

    if returncode == 0:
        logger.info(f"导出成功: {output_file}")
        if show_progress:
            print(f"\n✅ 导出成功: {output_file}")
        return True
    else:
        error_msg = stderr if stderr else "未知错误"
        logger.error(f"导出失败: {error_msg}")
        if show_progress:
            print(f"\n❌ 导出失败: {error_msg}")
        return False


def logout_tableau():
    """登出Tableau服务器"""
    logger.info("正在登出Tableau服务器")
    command = ["tabcmd", "logout"]
    returncode, stdout, stderr = run_command(command)
    if returncode == 0:
        logger.info("登出成功")
        return True
    else:
        logger.error(f"登出失败: {stderr}")
        return False


def parse_date_series(series: pd.Series) -> pd.Series:
    """将日期列解析为 datetime（容错，兼容中文日期如“2023年1月2日”）"""
    s = series.astype(str)
    # 先尝试直接解析
    dt = pd.to_datetime(s, errors='coerce')
    if dt.notna().any():
        return dt

    # 针对中文日期：YYYY年M月D日 → YYYY-M-D
    s_cn = (
        s.str.replace('年', '-', regex=False)
         .str.replace('月', '-', regex=False)
         .str.replace('日', '', regex=False)
         .str.strip()
    )
    # 再尝试一次（不指定format，允许非零填充）
    dt_cn = pd.to_datetime(s_cn, errors='coerce')
    if dt_cn.notna().any():
        return dt_cn

    # 其他常见格式回退
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        dt_try = pd.to_datetime(s_cn, errors='coerce', format=fmt)
        if dt_try.notna().any():
            return dt_try

    # 全部失败
    return dt_cn


def incremental_refresh(view: str, output: str, format_: str, days_offset: int,
                        server: str, username: Optional[str], password: Optional[str],
                        token_name: Optional[str], token_value: Optional[str],
                        timeout: int, show_progress: bool) -> bool:
    """
    增量刷新逻辑：
    - 如果输出文件不存在：直接全量导出保存。
    - 如果存在：计算现有 CSV 的 max(DATE_COL) 作为基准；阈值=基准-N天；
      先保留旧数据中日期 < 阈值的部分，再导出最新数据并筛选日期 >= 阈值，合并后去重写回。
    """
    # 确保输出目录存在
    out_dir = os.path.dirname(output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(output):
        logger.info("未发现历史文件，执行全量导出（临时文件写入后覆盖）")
        out_dir = os.path.dirname(output) or "."
        tmp_full = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(output))[0]}.full_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
        ok = export_view(view, tmp_full, format_, timeout, show_progress,
                         server, username, password, token_name, token_value)
        if ok:
            os.replace(tmp_full, output)
            return True
        else:
            try:
                os.remove(tmp_full)
            except Exception:
                pass
            return False

    # 读取历史数据
    try:
        df_old = pd.read_csv(output)
    except Exception as e:
        logger.warning(f"读取历史文件失败，回退到全量导出（临时文件写入后覆盖）: {e}")
        tmp_full = os.path.join(os.path.dirname(output) or '.', f"{os.path.splitext(os.path.basename(output))[0]}.full_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
        ok = export_view(view, tmp_full, format_, timeout, show_progress,
                         server, username, password, token_name, token_value)
        if ok:
            os.replace(tmp_full, output)
            return True
        else:
            try:
                os.remove(tmp_full)
            except Exception:
                pass
            return False

    if DATE_COL not in df_old.columns:
        logger.warning(f"历史文件缺少日期列 {DATE_COL}，回退到全量导出（临时文件写入后覆盖）")
        tmp_full = os.path.join(os.path.dirname(output) or '.', f"{os.path.splitext(os.path.basename(output))[0]}.full_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
        ok = export_view(view, tmp_full, format_, timeout, show_progress,
                         server, username, password, token_name, token_value)
        if ok:
            os.replace(tmp_full, output)
            return True
        else:
            try:
                os.remove(tmp_full)
            except Exception:
                pass
            return False

    # 解析日期并计算阈值
    df_old[DATE_COL] = parse_date_series(df_old[DATE_COL])
    max_date = df_old[DATE_COL].max()
    if pd.isna(max_date):
        logger.warning("历史文件日期列解析为空，回退到全量导出（临时文件写入后覆盖）")
        tmp_full = os.path.join(os.path.dirname(output) or '.', f"{os.path.splitext(os.path.basename(output))[0]}.full_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
        ok = export_view(view, tmp_full, format_, timeout, show_progress,
                         server, username, password, token_name, token_value)
        if ok:
            os.replace(tmp_full, output)
            return True
        else:
            try:
                os.remove(tmp_full)
            except Exception:
                pass
            return False

    threshold = max_date - timedelta(days=days_offset)
    if show_progress:
        print(f"历史最大日期: {max_date.date()}，增量覆盖阈值: {threshold.date()}")
    logger.info(f"历史最大日期: {max_date}, 覆盖阈值(含当日): {threshold}")

    # 保留旧数据中阈值前的数据
    df_keep = df_old[df_old[DATE_COL] < threshold]

    # 导出最新数据到临时文件
    tmp_name = os.path.splitext(os.path.basename(output))[0]
    tmp_path = os.path.join(out_dir, f"{tmp_name}.tmp_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
    ok = export_view(view, tmp_path, format_, timeout, show_progress,
                     server, username, password, token_name, token_value)
    if not ok:
        # 导出失败不改动历史文件
        return False

    # 读取新数据并筛选阈值后的部分
    try:
        df_new = pd.read_csv(tmp_path)
    except Exception as e:
        logger.error(f"读取临时导出文件失败: {e}")
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return False

    if DATE_COL not in df_new.columns:
        logger.error(f"导出文件缺少日期列 {DATE_COL}")
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return False

    df_new[DATE_COL] = parse_date_series(df_new[DATE_COL])
    df_new_recent = df_new[df_new[DATE_COL] >= threshold]

    # 合并并去重
    df_merged = pd.concat([df_keep, df_new_recent], ignore_index=True)
    df_merged = df_merged.drop_duplicates()

    # 原子写入：先写到临时文件，再覆盖
    safe_tmp_out = os.path.join(out_dir, f"{tmp_name}.merged_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
    df_merged.to_csv(safe_tmp_out, index=False)

    # 覆盖输出
    os.replace(safe_tmp_out, output)

    # 清理临时导出
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    logger.info(f"增量刷新完成，写入: {output}，行数: {len(df_merged)}")
    if show_progress:
        print(f"\n✅ 增量刷新完成，合并后行数: {len(df_merged)}")
    return True


def main():
    parser = argparse.ArgumentParser(description="线索表数据获取（城市视图增量版）")
    parser.add_argument("--server", default="http://tableau.immotors.com", help="Tableau服务器URL")
    parser.add_argument("--username", default="analysis", help="Tableau用户名")
    parser.add_argument("--password", default="analysis888", help="Tableau密码")
    parser.add_argument("--token-name", help="个人访问令牌名称")
    parser.add_argument("--token-value", help="个人访问令牌值")
    parser.add_argument(
        "--view",
        default="http://tableau.immotors.com/#/views/165/leads_assign_city_store2_1?:iid=1",
        help="要导出的视图路径 (Workbook/Sheet) 或完整URL"
    )
    parser.add_argument(
        "--output",
        default=os.path.join("original", "leads_assign_city_store2_1.csv"),
        help="输出文件路径"
    )
    parser.add_argument("--format", default="csv", choices=["csv"], help="输出格式（仅csv，便于合并）")
    parser.add_argument("--timeout", type=int, default=900, help="导出操作超时时间（秒）")
    parser.add_argument("--no-progress", action="store_true", help="不显示进度提示")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    parser.add_argument("--days-offset", type=int, default=2, help="覆盖刷新阈值的天数偏移（默认2）")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    show_progress = not args.no_progress

    try:
        start_time = datetime.now()
        if show_progress:
            print(f"=== 城市视图增量刷新（Tableau） ===")
            print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"服务器: {args.server}")
            print(f"视图: {args.view}")
            print(f"输出文件: {args.output}")
            print(f"覆盖阈值: max({DATE_COL}) - {args.days_offset} 天")
            print("=" * 30)

        # 登录（优先PAT）
        if args.token_name and args.token_value:
            if show_progress:
                print("正在使用个人访问令牌(PAT)登录...")
            login_success = login_tableau(args.server, token_name=args.token_name, token_value=args.token_value)
        else:
            if show_progress:
                print(f"正在使用用户名 {args.username} 登录...")
            login_success = login_tableau(args.server, args.username, args.password)

        if not login_success:
            if show_progress:
                print("❌ 登录失败，无法继续")
            return 1

        if show_progress:
            print("✅ 登录成功")

        # 增量刷新
        ok = incremental_refresh(
            view=args.view,
            output=args.output,
            format_=args.format,
            days_offset=args.days_offset,
            server=args.server,
            username=(None if (args.token_name and args.token_value) else args.username),
            password=(None if (args.token_name and args.token_value) else args.password),
            token_name=args.token_name,
            token_value=args.token_value,
            timeout=args.timeout,
            show_progress=show_progress,
        )

        # 登出
        if show_progress:
            print("正在登出...")
        logout_tableau()

        end_time = datetime.now()
        elapsed = int((end_time - start_time).total_seconds())

        if ok:
            logger.info(f"增量刷新完成: {args.output}")
            if show_progress:
                print(f"\n✅ 增量刷新完成: {args.output}")
                print(f"总耗时: {elapsed} 秒")
            return 0
        else:
            if show_progress:
                print(f"\n❌ 增量刷新失败")
                print(f"总耗时: {elapsed} 秒")
            return 1

    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}")
        if show_progress:
            print(f"\n❌ 执行过程中发生错误: {str(e)}")
        try:
            logout_tableau()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())