import pandas as pd
import os
from collections import defaultdict

def analyze_columns_globally(csv_file_path, sample_size=1000000):
    """
    全局分析列的特征，决定最佳数据类型
    """
    print(f"🔍 分析数据集特征: {csv_file_path}")
    
    # 读取样本数据进行分析
    chunk_size = 100000
    column_stats = defaultdict(lambda: {'unique_values': set(), 'total_count': 0, 'numeric_count': 0})
    
    sample_count = 0
    for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
        for col in chunk.columns:
            if chunk[col].dtype == 'object':
                # 收集唯一值（限制数量避免内存问题）
                unique_vals = chunk[col].dropna().unique()
                if len(column_stats[col]['unique_values']) < 10000:
                    column_stats[col]['unique_values'].update(unique_vals[:1000])
                
                column_stats[col]['total_count'] += len(chunk[col])
                
                # 检查数值转换可能性
                numeric_converted = pd.to_numeric(chunk[col], errors='coerce')
                column_stats[col]['numeric_count'] += numeric_converted.notna().sum()
        
        sample_count += len(chunk)
        if sample_count >= sample_size:
            break
    
    # 决定每列的最佳类型
    column_types = {}
    for col, stats in column_stats.items():
        unique_count = len(stats['unique_values'])
        total_count = stats['total_count']
        numeric_ratio = stats['numeric_count'] / total_count if total_count > 0 else 0
        
        if numeric_ratio > 0.8:
            column_types[col] = 'numeric'
        elif unique_count <= 1000 or (unique_count / total_count) < 0.5:
            column_types[col] = 'category'
        else:
            column_types[col] = 'object'
        
        print(f"  {col}: {unique_count} 唯一值, 数值比例 {numeric_ratio:.2%} -> {column_types[col]}")
    
    return column_types

def csv_to_parquet_with_global_optimization(csv_file_path, parquet_output_path):
    """
    基于全局分析优化CSV到Parquet转换
    """
    print(f"🔄 正在读取CSV文件: {csv_file_path}")
    
    # 第一步：全局分析
    column_types = analyze_columns_globally(csv_file_path)
    
    # 第二步：分块处理并应用优化
    chunk_size = 100000
    chunks = []
    
    try:
        for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
            # 优化日期列
            if '日期' in chunk.columns:
                chunk['日期'] = pd.to_datetime(chunk['日期'], errors='coerce')
            
            # 根据全局分析结果优化列类型
            for col in chunk.columns:
                if col in column_types:
                    if column_types[col] == 'numeric':
                        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                    elif column_types[col] == 'category':
                        chunk[col] = chunk[col].astype('category')
                    # object类型保持不变
            
            chunks.append(chunk)
            print(f"📈 已处理 {len(chunks) * chunk_size} 行...")
    
        # 合并所有块
        print("🔄 正在合并数据...")
        df = pd.concat(chunks, ignore_index=True)
        
        # 重新应用category类型（concat可能会重置类型）
        print("🔄 重新应用数据类型优化...")
        for col, target_type in column_types.items():
            if col in df.columns:
                if target_type == 'category':
                    df[col] = df[col].astype('category')
                elif target_type == 'numeric':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"📊 数据维度: {df.shape[0]} 行 x {df.shape[1]} 列")
        
        # 显示最终数据类型
        print("\n📋 优化后数据类型:")
        category_count = 0
        object_count = 0
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
            if str(dtype) == 'category':
                category_count += 1
            elif str(dtype) == 'object':
                object_count += 1
        
        print(f"\n📊 类型统计: {category_count} 个category列, {object_count} 个object列")
        
        # 保存为Parquet文件
        output_dir = os.path.dirname(parquet_output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 正在保存Parquet文件: {parquet_output_path}")
        df.to_parquet(parquet_output_path, index=False, engine='pyarrow', compression='snappy')
        
        # 验证保存结果
        file_size = os.path.getsize(parquet_output_path) / (1024 * 1024)  # MB
        print(f"✅ 转换完成！文件大小: {file_size:.2f} MB")
        
        # 内存使用情况
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"💾 内存使用: {memory_usage:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {str(e)}")
        return False

if __name__ == "__main__":
    csv_file = "/Users/zihao_/Documents/coding/dataset/formatted/月度_上险量_0723.csv"
    parquet_file = "/Users/zihao_/Documents/coding/dataset/formatted/月度_上险量_0723_optimized.parquet"
    
    # 检查CSV文件是否存在
    if not os.path.exists(csv_file):
        print(f"❌ 错误: CSV文件不存在: {csv_file}")
        exit(1)
    
    print("🚀 开始全局优化CSV到Parquet转换...")
    print("=" * 50)
    
    success = csv_to_parquet_with_global_optimization(csv_file, parquet_file)
    
    if success:
        print("\n🎉 转换成功完成！")
        print(f"📁 输出文件: {parquet_file}")
    else:
        print("\n❌ 转换失败")
        exit(1)