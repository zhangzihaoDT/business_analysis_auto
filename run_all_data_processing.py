#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行所有数据处理脚本

该脚本会依次执行：
1. leads_structure_analysis_to_parquet.py - 处理线索结构分析数据
2. business_data_to_parquet.py - 处理业务数据
3. intention_order_analysis_to_parquet.py - 处理意向订单分析数据

使用方法:
    python run_all_data_processing.py
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_separator(title=""):
    """打印分隔线"""
    print("\n" + "="*60)
    if title:
        print(f" {title} ")
        print("="*60)
    else:
        print("="*60)

def run_script(script_name, description):
    """
    运行指定的Python脚本
    
    Args:
        script_name (str): 脚本文件名
        description (str): 脚本描述
    
    Returns:
        bool: 是否成功执行
    """
    print_separator(f"开始执行: {description}")
    print(f"📄 脚本: {script_name}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 获取脚本的完整路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, script_name)
    
    # 检查脚本是否存在
    if not os.path.exists(script_path):
        print(f"❌ 错误: 脚本文件不存在 - {script_path}")
        return False
    
    try:
        # 执行脚本
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=script_dir,
            capture_output=False,  # 直接显示输出
            text=True
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"\n✅ {description} 执行成功!")
            print(f"⏱️  执行时间: {execution_time:.2f} 秒")
            return True
        else:
            print(f"\n❌ {description} 执行失败!")
            print(f"💥 退出代码: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n💥 执行 {description} 时发生异常: {e}")
        return False

def main():
    """
    主函数 - 依次执行所有数据处理脚本
    """
    print_separator("数据处理脚本批量执行器")
    print(f"🚀 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 工作目录: {os.getcwd()}")
    
    # 定义要执行的脚本列表
    scripts_to_run = [
        {
            "script": "leads_structure_analysis_to_parquet.py",
            "description": "线索结构分析数据处理"
        },
        {
            "script": "business_data_to_parquet.py", 
            "description": "业务数据处理"
        },
        {
            "script": "intention_order_analysis_to_parquet.py",
            "description": "意向订单分析数据处理"
        }
    ]
    
    # 执行结果统计
    total_scripts = len(scripts_to_run)
    successful_scripts = 0
    failed_scripts = []
    
    overall_start_time = time.time()
    
    # 依次执行每个脚本
    for i, script_info in enumerate(scripts_to_run, 1):
        print(f"\n📋 进度: {i}/{total_scripts}")
        
        success = run_script(
            script_info["script"], 
            script_info["description"]
        )
        
        if success:
            successful_scripts += 1
        else:
            failed_scripts.append(script_info["description"])
        
        # 如果不是最后一个脚本，添加间隔
        if i < total_scripts:
            print("\n⏳ 等待 2 秒后继续...")
            time.sleep(2)
    
    # 总结执行结果
    overall_end_time = time.time()
    total_execution_time = overall_end_time - overall_start_time
    
    print_separator("执行总结")
    print(f"📊 总脚本数: {total_scripts}")
    print(f"✅ 成功执行: {successful_scripts}")
    print(f"❌ 执行失败: {len(failed_scripts)}")
    print(f"⏱️  总执行时间: {total_execution_time:.2f} 秒")
    print(f"🏁 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed_scripts:
        print(f"\n💥 失败的脚本:")
        for failed_script in failed_scripts:
            print(f"   - {failed_script}")
        print(f"\n⚠️  请检查失败的脚本并重新运行")
        sys.exit(1)
    else:
        print(f"\n🎉 所有数据处理脚本执行完成!")
        print(f"📁 请检查 formatted/ 目录下的输出文件")
        sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⚠️  用户中断执行")
        print(f"🛑 程序已停止")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 程序执行过程中发生未预期的错误: {e}")
        sys.exit(1)