import pandas as pd
import os

def process_business_data_to_parquet():
    """
    处理业务数据并输出为Parquet格式
    """
    # --- 1. 读取数据 ---
    # 定义文件路径
    csv_file_path = '/Users/zihao_/Documents/coding/dataset/original/业务数据记录_with表_表格.csv'
    parquet_dir = '/Users/zihao_/Documents/coding/dataset/formatted/'
    parquet_path = os.path.join(parquet_dir, 'business_daily_metrics.parquet')
    
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    
    # --- 2. 数据清洗和预处理 ---
    # 将'日期'列转换为datetime对象
    df['日期'] = pd.to_datetime(df['日期'])
    
    # 将列名'日期'重命名为'date'以便于后续处理
    df.rename(columns={'日期': 'date'}, inplace=True)
    
    # --- 3. 创建辅助指标 ---
    # 防止除以零错误，使用 .replace(0, 1) 不是最佳实践，更好的方法是使用 np.divide 或检查分母
    # 这里我们先按照您的要求实现，但请注意在实际生产中这可能隐藏数据问题
    df["试驾锁单占比"] = df["试驾锁单数"] / df["锁单数"].replace(0, 1)
    df["小订留存占比"] = df["小订留存锁单数"] / df["锁单数"].replace(0, 1)
    df["线索转化率"] = df["锁单数"] / df["有效线索数"].replace(0, 1)
    df["抖音线索占比"] = df["抖音战队线索数"] / df["有效线索数"].replace(0, 1)
    
    # --- 4. 计算7日滚动平均值（平滑处理） ---
    # 按时间排序，确保滚动计算的正确性
    df = df.sort_values("date")
    
    # 计算各项指标的7日滚动平均值
    df["锁单数_7日均值"] = df["锁单数"].rolling(window=7).mean()
    df["有效试驾数_7日均值"] = df["有效试驾数"].rolling(window=7).mean()
    df["试驾锁单占比_7日均值"] = df["试驾锁单占比"].rolling(window=7).mean()
    df["小订留存占比_7日均值"] = df["小订留存占比"].rolling(window=7).mean()
    df["线索转化率_7日均值"] = df["线索转化率"].rolling(window=7).mean()
    df["抖音线索占比_7日均值"] = df["抖音线索占比"].rolling(window=7).mean()
    
    # --- 5. 数据类型优化 ---
    # 优化数据类型以减少文件大小
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'date':
            # 检查是否可以转换为category
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:  # 如果唯一值比例小于50%，转换为category
                df[col] = df[col].astype('category')
    
    # --- 6. 保存到 Parquet ---
    # 确保目标目录存在
    if not os.path.exists(parquet_dir):
        os.makedirs(parquet_dir)
    
    # 保存为Parquet文件
    df.to_parquet(parquet_path, index=False, engine='pyarrow', compression='snappy')
    
    # 计算文件大小
    file_size = os.path.getsize(parquet_path) / (1024 * 1024)  # MB
    
    print(f"数据已成功处理并保存到 Parquet 文件: {parquet_path}")
    print(f"文件大小: {file_size:.2f} MB")
    print(f"数据维度: {df.shape[0]} 行 x {df.shape[1]} 列")
    
    # 显示处理后数据的前几行以供查阅
    print("\n数据概览:")
    print(df.describe())
    
    return df, parquet_path

if __name__ == "__main__":
    # 执行数据处理
    try:
        df, output_path = process_business_data_to_parquet()
        print("\n✅ 数据处理完成！")
        print(f"📁 输出文件: {output_path}")
    except FileNotFoundError as e:
        print(f"❌ 错误: 文件未找到 - {e}")
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")