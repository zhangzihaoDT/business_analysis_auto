import pandas as pd
from tableauhyperapi import TableName, HyperProcess, Connection, Telemetry
import os
import numpy as np
from datetime import datetime
import pantab as pt # 导入 pantab 库

def convert_hyper_to_csv(hyper_file_path, csv_output_path):
    """
    将.hyper文件转换为CSV文件
    
    Args:
        hyper_file_path: 输入的.hyper文件路径
        csv_output_path: 输出的CSV文件路径
    """
    print(f"🔄 正在处理文件: {hyper_file_path}")
    
    # 步骤1：从.hyper文件读取数据
    try:
        # 首先使用 tableauhyperapi 查看文件中的实际表名
        print("🔍 正在检查 Hyper 文件中的表结构...")
        table_name = None
        
        with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
            with Connection(endpoint=hyper.endpoint, database=hyper_file_path) as connection:
                # 获取所有 schema
                schemas = connection.catalog.get_schema_names()
                print(f"📋 发现的 schema: {[str(schema) for schema in schemas]}")
                
                for schema in schemas:
                    try:
                        # 获取该 schema 下的所有表名
                        tables = connection.catalog.get_table_names(schema)
                        print(f"📋 Schema '{schema}' 中的表: {[str(table) for table in tables]}")
                        
                        if tables:
                            # 使用第一个表，清理引号
                            raw_table_name = str(tables[0])
                            # 移除多余的引号
                            table_name = raw_table_name.replace('"', '')
                            print(f"🎯 找到表名: {raw_table_name} -> 清理后: {table_name}")
                            break
                    except Exception as schema_error:
                        print(f"⚠️ 检查 schema '{schema}' 时出错: {schema_error}")
                        continue
        
        # 尝试让 pantab 自动检测表
        try:
            print("📖 尝试让 pantab 自动检测表...")
            df = pt.frame_from_hyper(hyper_file_path)
            print(f"📊 成功从 Hyper 文件读取 {len(df)} 行数据到 DataFrame（自动检测）。")
        except Exception as auto_error:
            print(f"❌ 自动检测失败: {auto_error}")
            
            # 如果自动检测失败，使用原始的 tableauhyperapi 方法
            print("🔄 使用 tableauhyperapi 直接读取数据...")
            with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
                with Connection(endpoint=hyper.endpoint, database=hyper_file_path) as connection:
                    # 使用原始表名构造查询
                    raw_table_name = '"Extract"."Extract"'
                    query = f"SELECT * FROM {raw_table_name}"
                    print(f"📖 执行查询: {query}")
                    
                    result = connection.execute_query(query)
                    
                    # 获取列名
                    column_names = [col.name.unescaped for col in result.schema.columns]
                    print(f"📋 列名: {column_names}")
                    
                    # 分块读取并直接写入CSV文件以避免内存问题
                    chunk_size = 50000  # 增大块大小提高效率
                    rows = []
                    row_count = 0
                    chunk_count = 0
                    temp_files = []
                    
                    print(f"📊 开始分块读取并写入临时文件（每块 {chunk_size} 行）...")
                    
                    try:
                        for row in result:
                            # 处理日期类型转换
                            processed_row = []
                            for i, value in enumerate(row):
                                if hasattr(value, '__class__') and 'Date' in str(type(value)):
                                    # 将Tableau的Date对象转换为字符串
                                    processed_row.append(str(value))
                                else:
                                    processed_row.append(value)
                            
                            rows.append(processed_row)
                            row_count += 1
                            
                            # 每读取一定数量的行就写入临时文件
                            if len(rows) >= chunk_size:
                                chunk_df = pd.DataFrame(rows, columns=column_names)
                                
                                temp_file = f"{csv_output_path}.temp_{chunk_count}.csv"
                                chunk_df.to_csv(temp_file, index=False)
                                temp_files.append(temp_file)
                                
                                print(f"📈 已处理 {row_count} 行，写入临时文件 {chunk_count + 1}...")
                                
                                rows = []
                                chunk_count += 1
                        
                        # 处理最后一块数据
                        if rows:
                            chunk_df = pd.DataFrame(rows, columns=column_names)
                            
                            temp_file = f"{csv_output_path}.temp_{chunk_count}.csv"
                            chunk_df.to_csv(temp_file, index=False)
                            temp_files.append(temp_file)
                            chunk_count += 1
                    finally:
                        # 确保关闭result
                        if 'result' in locals():
                            try:
                                result.close()
                            except:
                                pass
                    
                    print(f"📊 总共读取 {row_count} 行数据，分为 {chunk_count} 个临时文件")
                    
                    # 合并所有临时文件
                    print("🔄 正在合并临时文件...")
                    all_chunks = []
                    for temp_file in temp_files:
                        chunk_df = pd.read_csv(temp_file)
                        all_chunks.append(chunk_df)
                    
                    df = pd.concat(all_chunks, ignore_index=True)
                    
                    # 清理临时文件
                    for temp_file in temp_files:
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                    
                    print(f"✅ 成功合并所有数据，总共 {len(df)} 行。")

    except Exception as e:
        print(f"❌ 读取Hyper文件失败: {e}")
        return False
                
    
    # 步骤2：保存为CSV文件
    try:
        output_dir = os.path.dirname(csv_output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 正在保存CSV文件: {csv_output_path}")
        df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
        
        # 验证保存结果
        file_size = os.path.getsize(csv_output_path) / (1024 * 1024)  # MB
        print(f"✅ 转换完成！文件大小: {file_size:.2f} MB")
        
        # 数据质量报告
        print_data_quality_report(df)
        
        return True
        
    except Exception as e:
        print(f"❌ 保存CSV文件失败: {str(e)}")
        return False

# 数据类型优化功能已移至 csv_to_parquet.py 脚本中处理

def print_data_quality_report(df):
    """
    打印数据质量报告
    """
    print("\n📊 数据质量报告:")
    print(f"  📏 数据维度: {df.shape[0]} 行 x {df.shape[1]} 列")
    
    # 缺失值统计
    missing_stats = df.isnull().sum()
    if missing_stats.sum() > 0:
        print("\n⚠️  缺失值统计:")
        for col, missing_count in missing_stats[missing_stats > 0].items():
            missing_pct = (missing_count / len(df)) * 100
            print(f"    {col}: {missing_count} ({missing_pct:.1f}%)")
    else:
        print("\n✅ 无缺失值")
    
    # 数据类型统计
    type_counts = df.dtypes.value_counts()
    print("\n📋 数据类型分布:")
    for dtype, count in type_counts.items():
        print(f"    {dtype}: {count} 列")
    
    # 内存使用情况
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"\n💾 内存使用: {memory_usage:.2f} MB")

if __name__ == "__main__":
    # 文件路径配置
    hyper_file = "/Users/zihao_/Documents/coding/dataset/original/乘用车上险量_0723.hyper"
    csv_file = "/Users/zihao_/Documents/coding/dataset/formatted/乘用车上险量_0723.csv"
    
    # 检查依赖库
    try:
        from tableauhyperapi import TableName
        print("✅ tableauhyperapi 已安装")
    except ImportError:
        print("❌ 错误: 'tableauhyperapi' 库未安装。请运行: pip install tableauhyperapi")
        exit(1)
    
    # 检查输入文件是否存在
    if not os.path.exists(hyper_file):
        print(f"❌ 错误: 输入文件不存在: {hyper_file}")
        exit(1)
    
    print("🚀 开始Hyper到CSV转换...")
    print("=" * 50)
    
    # 执行转换
    success = convert_hyper_to_csv(
        hyper_file_path=hyper_file,
        csv_output_path=csv_file
    )
    
    if success:
        print("\n🎉 转换成功完成！")
        print(f"📁 输出文件: {csv_file}")
        print("\n💡 提示: 如需转换为Parquet格式，请运行 csv_to_parquet.py")
    else:
        print("\n❌ 转换失败")
        exit(1)
