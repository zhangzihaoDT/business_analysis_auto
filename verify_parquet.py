import pandas as pd
import pyarrow.parquet as pq
import os

def verify_parquet_file(parquet_path):
    """
    验证Parquet文件的内容和结构
    """
    print(f"🔍 验证Parquet文件: {parquet_path}")
    print("=" * 50)
    
    try:
        # 检查文件是否存在
        if not os.path.exists(parquet_path):
            print(f"❌ 文件不存在: {parquet_path}")
            return False
        
        # 获取文件大小
        file_size = os.path.getsize(parquet_path) / (1024 * 1024)  # MB
        print(f"📁 文件大小: {file_size:.2f} MB")
        
        # 使用pyarrow读取元数据
        parquet_file = pq.ParquetFile(parquet_path)
        print(f"📊 行数: {parquet_file.metadata.num_rows:,}")
        print(f"📋 列数: {parquet_file.metadata.num_columns}")
        
        # 显示schema
        print("\n📋 数据结构:")
        schema = parquet_file.schema.to_arrow_schema()
        for i, field in enumerate(schema):
            print(f"  {i+1:2d}. {field.name}: {field.type}")
        
        # 读取前几行数据进行验证（只读取前10000行以节省内存）
        print("\n🔍 数据预览 (前5行):")
        # 使用pyarrow读取部分数据
        table = pq.read_table(parquet_path, columns=None)
        df_sample = table.slice(0, 10000).to_pandas()  # 只读取前10000行
        print(df_sample.head(5).to_string())
        
        # 检查数据类型
        print("\n📊 数据类型:")
        for col, dtype in df_sample.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # 检查是否有缺失值（检查前1000行）
        print("\n🔍 缺失值检查 (前1000行):")
        df_subset = df_sample.head(1000)
        missing_counts = df_subset.isnull().sum()
        for col, missing in missing_counts.items():
            if missing > 0:
                print(f"  {col}: {missing} 个缺失值")
        
        if missing_counts.sum() == 0:
            print("  ✅ 前1000行无缺失值")
        
        # 显示数据统计信息
        print("\n📈 数值列统计信息:")
        numeric_cols = df_sample.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            print(df_sample[numeric_cols].describe().to_string())
        else:
            print("  无数值列")
        
        print("\n✅ Parquet文件验证完成！")
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {str(e)}")
        return False

if __name__ == "__main__":
    parquet_file = "/Users/zihao_/Documents/coding/dataset/formatted/月度_上险量_0723_optimized.parquet"
    
    print("🚀 开始验证Parquet文件...")
    success = verify_parquet_file(parquet_file)
    
    if success:
        print("\n🎉 验证成功！Parquet文件格式正确。")
    else:
        print("\n❌ 验证失败！")
        exit(1)