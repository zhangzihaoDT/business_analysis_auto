#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tableau数据导出工具

使用Tabcmd命令行工具从Tableau服务器导出数据
"""

import argparse
import os
import subprocess
import logging
import sys
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tableau_export.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_command(command, timeout=300, show_progress=True):
    """
    执行命令并返回结果，支持超时和进度显示
    
    Args:
        command: 要执行的命令
        timeout: 超时时间（秒），默认5分钟
        show_progress: 是否显示进度提示
        
    Returns:
        tuple: (returncode, stdout, stderr)
    """
    logger.debug(f"执行命令: {' '.join(command)}")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 进度显示相关变量
        start_time = datetime.now()
        progress_shown = False
        stdout_data = []
        stderr_data = []
        
        # 非阻塞读取输出
        import select
        import time
        import sys
        
        # 设置非阻塞模式
        for pipe in [process.stdout, process.stderr]:
            if pipe:
                os.set_blocking(pipe.fileno(), False)
        
        # 循环检查进程状态和超时
        while process.poll() is None:
            # 检查超时
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if timeout and elapsed_time > timeout:
                process.terminate()
                logger.warning(f"命令执行超时 ({timeout}秒)，已终止")
                return -1, "".join(stdout_data), f"命令执行超时 ({timeout}秒)，已终止"
            
            # 读取输出
            readable, _, _ = select.select([process.stdout, process.stderr], [], [], 0.5)
            for pipe in readable:
                line = pipe.readline()
                if line:
                    if pipe == process.stdout:
                        stdout_data.append(line)
                    else:
                        stderr_data.append(line)
            
            # 显示进度
            if show_progress and elapsed_time > 2:  # 2秒后开始显示进度
                if not progress_shown or int(elapsed_time) % 10 == 0:  # 每10秒更新一次
                    progress_shown = True
                    elapsed = int(elapsed_time)
                    sys.stdout.write(f"\r正在执行命令... 已用时 {elapsed} 秒")
                    sys.stdout.flush()
            
            time.sleep(0.1)
        
        # 读取剩余输出
        stdout, stderr = process.communicate()
        stdout_data.append(stdout)
        stderr_data.append(stderr)
        
        # 清除进度显示
        if show_progress and progress_shown:
            sys.stdout.write("\r" + " " * 50 + "\r")
            sys.stdout.flush()
        
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

def login_tableau(server, username=None, password=None, token_name=None, token_value=None):
    """
    登录Tableau服务器
    
    Args:
        server: Tableau服务器URL
        username: 用户名
        password: 密码
        token_name: 个人访问令牌名称
        token_value: 个人访问令牌值
        
    Returns:
        bool: 是否登录成功
    """
    logger.info(f"正在登录Tableau服务器: {server}")
    
    # 移除URL中的#/home部分，因为tabcmd不需要这部分
    server = server.split('#')[0] if '#' in server else server
    
    # 使用个人访问令牌登录
    if token_name and token_value:
        logger.info("使用个人访问令牌(PAT)登录")
        command = ["tabcmd", "login", "-s", server, "--token-name", token_name, "--token-value", token_value]
    # 使用用户名密码登录
    else:
        logger.info(f"使用用户名密码登录: {username}")
        command = ["tabcmd", "login", "-s", server, "-u", username, "-p", password]
    
    returncode, stdout, stderr = run_command(command)
    
    if returncode == 0:
        logger.info("登录成功")
        return True
    else:
        logger.error(f"登录失败: {stderr}")
        return False

def export_view(view_path, output_file, format="csv", timeout=600, show_progress=True):
    """
    导出Tableau视图
    
    Args:
        view_path: 视图路径 (Workbook/Sheet 或 完整URL)
        output_file: 输出文件路径
        format: 输出格式 (csv, pdf, png, etc.)
        timeout: 导出操作超时时间（秒），默认10分钟
        show_progress: 是否显示进度提示
        
    Returns:
        bool: 是否导出成功
    """
    logger.info(f"正在导出视图: {view_path} 到 {output_file}")
    
    if show_progress:
        print(f"开始导出 Tableau 视图: {view_path}")
        print(f"导出格式: {format}, 输出文件: {output_file}")
        print(f"超时设置: {timeout} 秒")
        print("导出过程可能需要几分钟，请耐心等待...")
    
    # 检查是否是完整URL
    if view_path.startswith('http'):
        # 如果是完整URL，尝试提取工作簿和视图名称
        try:
            # 尝试从URL中提取视图路径
            from urllib.parse import urlparse
            parsed_url = urlparse(view_path)
            path_parts = parsed_url.fragment.strip('/').split('/')
            
            if len(path_parts) >= 2 and path_parts[0] == 'views':
                # URL格式可能是 #/views/workbook/sheet
                tableau_path = f"{path_parts[1]}/{path_parts[2]}" if len(path_parts) > 2 else path_parts[1]
                logger.info(f"从URL提取的路径: {tableau_path}")
                if show_progress:
                    print(f"从URL提取的视图路径: {tableau_path}")
            else:
                # 直接使用URL
                tableau_path = view_path
                logger.info(f"使用完整URL: {tableau_path}")
        except Exception as e:
            logger.warning(f"从URL提取路径失败: {str(e)}，将使用原始URL")
            tableau_path = view_path
    else:
        # 处理常规视图路径
        # 尝试多种格式化方式
        # 1. 原始路径
        paths_to_try = [view_path]
        
        # 2. 移除空格
        paths_to_try.append(view_path.replace(" ", ""))
        
        # 3. 使用URL编码
        paths_to_try.append(view_path.replace(" ", "%20"))
        
        # 4. 如果是workbook/sheet格式，尝试单独处理
        parts = view_path.split('/')
        if len(parts) == 2:
            workbook, sheet = parts
            # 移除所有空格
            paths_to_try.append(f"{workbook.replace(' ', '')}/{sheet.replace(' ', '')}")
        
        # 记录所有尝试的路径
        logger.info(f"将尝试以下路径: {paths_to_try}")
        if show_progress:
            print(f"将尝试以下路径: {', '.join(paths_to_try)}")
        
        # 默认使用第一种格式
        # 注意：不再自动去除查询参数，允许用户传递带参数的视图路径（如 view/sheet?param=value）
        tableau_path = paths_to_try[0]
    
    # 首先尝试原始路径
    if show_progress:
        print(f"正在尝试导出视图: {tableau_path}")
    
    command = ["tabcmd", "export", tableau_path, f"--{format}", "-f", output_file]
    returncode, stdout, stderr = run_command(command, timeout=timeout, show_progress=show_progress)
    
    # 如果失败且有多种路径可尝试，则逐一尝试其他路径
    if returncode != 0 and 'paths_to_try' in locals() and len(paths_to_try) > 1:
        for i, path in enumerate(paths_to_try[1:], 1):
            if show_progress:
                print(f"\n尝试替代路径 {i}: {path}")
            logger.info(f"尝试替代路径 {i}: {path}")
            command = ["tabcmd", "export", path, f"--{format}", "-f", output_file]
            returncode, stdout, stderr = run_command(command, timeout=timeout, show_progress=show_progress)
            if returncode == 0:
                logger.info(f"使用替代路径 {i} 成功")
                if show_progress:
                    print(f"使用替代路径 {i} 成功导出")
                break
    
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
    """
    登出Tableau服务器
    
    Returns:
        bool: 是否登出成功
    """
    logger.info("正在登出Tableau服务器")
    
    command = ["tabcmd", "logout"]
    returncode, stdout, stderr = run_command(command)
    
    if returncode == 0:
        logger.info("登出成功")
        return True
    else:
        logger.error(f"登出失败: {stderr}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Tableau数据导出工具")
    
    parser.add_argument("--server", default="https://tableau-hs.immotors.com", 
                        help="Tableau服务器URL")
    parser.add_argument("--mobile", action="store_true", 
                        help="使用移动端/非办公网络服务器地址 (https://mobile-tableau-hs.immotors.com/)")
    parser.add_argument("--username", default="analysis", 
                        help="Tableau用户名")
    parser.add_argument("--password", default="analysis888", 
                        help="Tableau密码")
    parser.add_argument("--token-name", 
                        help="个人访问令牌名称")
    parser.add_argument("--token-value", 
                        help="个人访问令牌值")
    parser.add_argument("--view", required=True, 
                        help="要导出的视图路径 (Workbook/Sheet) 或完整URL")
    parser.add_argument("--output", 
                        help="输出文件路径 (默认为当前目录下的视图名称)")
    parser.add_argument("--format", default="csv", choices=["csv", "pdf", "png"], 
                        help="输出格式 (默认为csv)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="导出操作超时时间（秒），默认10分钟")
    parser.add_argument("--no-progress", action="store_true",
                        help="不显示进度提示")
    parser.add_argument("--verbose", action="store_true", 
                        help="显示详细日志")
    
    args = parser.parse_args()
    
    # 如果指定了mobile参数，覆盖server地址
    if args.mobile:
        args.server = "https://mobile-tableau-hs.immotors.com/"
        if not args.no_progress:
            print("📱 使用移动端/非办公网络服务器地址")
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 如果未指定输出文件，则使用视图名称
    if not args.output:
        # 尝试从视图路径中提取名称
        if args.view.startswith('http'):
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(args.view)
                path_parts = parsed_url.fragment.strip('/').split('/')
                if len(path_parts) >= 2 and path_parts[0] == 'views':
                    view_name = path_parts[-1]  # 使用最后一部分作为视图名称
                else:
                    view_name = "tableau_export"
            except:
                view_name = "tableau_export"
        else:
            view_name = args.view.split("/")[-1]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"{view_name}_{timestamp}.{args.format}"
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 显示进度标志
    show_progress = not args.no_progress
    
    # 执行导出流程
    try:
        start_time = datetime.now()
        
        if show_progress:
            print(f"=== Tableau 数据导出工具 ===")
            print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"服务器: {args.server}")
            print(f"视图: {args.view}")
            print(f"输出文件: {args.output}")
            print("="*30)
        
        # 登录 - 优先使用个人访问令牌
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
        
        # 导出
        export_success = export_view(
            args.view, 
            args.output, 
            args.format, 
            timeout=args.timeout, 
            show_progress=show_progress
        )
        
        # 登出
        if show_progress:
            print("正在登出...")
        logout_tableau()
        
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        
        if export_success:
            logger.info(f"导出完成: {args.output}")
            if show_progress:
                print(f"\n✅ 导出完成: {args.output}")
                print(f"总耗时: {int(elapsed_time)} 秒")
            return 0
        else:
            if show_progress:
                print(f"\n❌ 导出失败")
                print(f"总耗时: {int(elapsed_time)} 秒")
            return 1
    
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}")
        if show_progress:
            print(f"\n❌ 执行过程中发生错误: {str(e)}")
        
        # 确保登出
        try:
            logout_tableau()
        except:
            pass
        return 1

if __name__ == "__main__":
    sys.exit(main())