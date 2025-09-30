import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from datetime import datetime
import psutil
import json
from tableauhyperapi import HyperProcess, Telemetry, Connection, TableName

def get_memory_usage():
    """获取当前内存使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_column_mapping():
    """
    获取中英文字段映射
    """
    column_mapping = {
        '日期': 'date',
        '年月': 'year_month',
        'SUB_MODEL_ID': 'sub_model_id',
        '子车型': 'sub_model_name',
        '品牌': 'brand',
        '厂商': 'manufacturer',
        '是否核心品牌': 'is_core_brand',
        '品牌属性': 'brand_attribute',
        '销售状态': 'sales_status',
        '型号编码': 'model_code',
        '型号简称': 'model_short_name',
        '年型': 'model_year',
        '上市时间': 'launch_date',
        '车型': 'model_name',
        '细分市场': 'segment',
        '细分市场_上汽': 'segment_saic',
        '细分市场-车身形式': 'segment_body_style',
        '百公里电耗(kWh)': 'power_consumption_per_100km',
        '续航里程(km)': 'driving_range_km',
        '排量': 'displacement',
        '充电桩费用': 'charging_cost',
        '变速箱': 'transmission',
        '座位数': 'seat_count',
        '车身形式': 'body_style',
        '轴距(mm)': 'wheelbase_mm',
        '轴距(Mm)': 'wheelbase_mm',
        '长(mm)': 'length_mm',
        '长(Mm)': 'length_mm',
        '宽(mm)': 'width_mm',
        '宽(Mm)': 'width_mm',
        '高(mm)': 'height_mm',
        '高(Mm)': 'height_mm',
        'Msrp': 'msrp',
        'MSRP': 'msrp',
        'TP': 'tp_avg',
        'Tp Avg': 'tp_avg',
        'TP重心': 'tp_center',
        '燃料种类': 'fuel_type',
        '燃料类型': 'fuel_type',
        '燃料类型 (组)': 'fuel_type_group',
        '是否豪华': 'is_luxury',
        '是否豪华品牌': 'is_luxury_brand',
        '是否新势力品牌': 'is_new_energy_brand',
        '整备质量(kg)': 'curb_weight_kg',
        '销量': 'sales_volume',
        '上险数': 'insurance_volume',
        '省': 'province',
        '市': 'city',
        '城市级别': 'city_tier',
        '限购/限行/双非限': 'purchase_restriction',
        '成交价格': 'transaction_price',
        '指导价': 'msrp_price',
        '层级': 'tier',
        '层级 (组)': 'tier_group',
        'TP 1万1档': 'tp_1w_tier',
        'TP 5万1档': 'tp_5w_tier',
        'TP 10万1档': 'tp_10w_tier'
    }
    return column_mapping

def get_hyper_table_info(hyper_file_path):
    """
    获取Hyper文件的表信息和行数
    """
    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        with Connection(endpoint=hyper.endpoint, database=hyper_file_path) as connection:
            # 获取所有schema
            schemas = connection.catalog.get_schema_names()
            print(f"📋 发现 {len(schemas)} 个schema: {schemas}")
            
            tables = []
            
            # 遍历所有schema查找表
            for schema in schemas:
                schema_tables = connection.catalog.get_table_names(schema)
                if schema_tables:
                    tables.extend([(schema, table) for table in schema_tables])
                    print(f"📊 在schema '{schema}' 中发现 {len(schema_tables)} 个表: {[str(table) for table in schema_tables]}")
            
            if not tables:
                raise ValueError("Hyper文件中没有找到任何表")
            
            # 使用第一个表
            schema_name, table_name = tables[0]
            print(f"📊 分析表: {schema_name}.{table_name}")
            
            # 获取表的行数
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            total_rows = connection.execute_scalar_query(count_query)
            
            # 获取列名并清理双引号
            columns_info = connection.catalog.get_table_definition(table_name).columns
            column_names = [str(col.name).strip('"') for col in columns_info]
            
            print(f"📊 表信息: {total_rows} 行 x {len(column_names)} 列")
            
            return table_name, column_names, total_rows

def read_hyper_file_batch(hyper_file_path, table_name, offset=0, limit=100000):
    """
    分批读取Hyper文件并返回DataFrame
    """
    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        with Connection(endpoint=hyper.endpoint, database=hyper_file_path) as connection:
            # 执行分页查询
            query = f"SELECT * FROM {table_name} LIMIT {limit} OFFSET {offset}"
            result = connection.execute_list_query(query)
            
            # 获取列名并清理双引号
            columns_info = connection.catalog.get_table_definition(table_name).columns
            column_names = [str(col.name).strip('"') for col in columns_info]
            
            # 创建DataFrame
            df = pd.DataFrame(result, columns=column_names)
            
            return df

def analyze_column_types(df, sample_size=1000000):
    """分析DataFrame的列特征，决定最佳数据类型"""
    print(f"🔍 分析数据集特征...")
    
    # 如果数据量太大，只分析样本
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size)
    else:
        sample_df = df
    column_types = {}
    
    for col in sample_df.columns:
        col_str = str(col)  # 确保列名是字符串
        if sample_df[col_str].dtype == 'object':
            unique_count = sample_df[col_str].nunique()
            total_count = len(sample_df[col_str])
            
            # 尝试转换为数值
            numeric_series = pd.to_numeric(sample_df[col_str], errors='coerce')
            numeric_ratio = numeric_series.notna().sum() / total_count
            
            if numeric_ratio > 0.8:  # 80%以上可以转为数值
                column_types[col_str] = 'numeric'
                print(f"  {col_str}: {unique_count} 唯一值, 数值比例 {numeric_ratio:.2%} -> numeric")
            else:
                # 转换为category的条件
                column_types[col_str] = 'category'
                print(f"  {col_str}: {unique_count} 唯一值, 数值比例 {numeric_ratio:.2%} -> category")
        else:
            column_types[col_str] = 'keep'  # 保持原类型
    
    return column_types

def hyper_to_parquet_optimized(hyper_file_path, parquet_output_path, batch_size=100000):
    """
    将Hyper文件分批转换为优化的Parquet文件
    """
    try:
        print(f"🔄 正在分析Hyper文件: {hyper_file_path}")
        
        # 获取表信息
        table_name, column_names, total_rows = get_hyper_table_info(hyper_file_path)
        
        print(f"📊 数据总量: {total_rows:,} 行，将分 {(total_rows + batch_size - 1) // batch_size} 批处理")
        print(f"📦 每批处理: {batch_size:,} 行")
        
        # 读取第一批数据用于分析列类型和设置schema
        print("\n🔍 读取样本数据分析列类型...")
        sample_df = read_hyper_file_batch(hyper_file_path, table_name, 0, min(batch_size, total_rows))
        
        # 分析列类型
        column_types = analyze_column_types(sample_df)
        
        # 应用中英文字段映射
        column_mapping = get_column_mapping()
        original_columns = sample_df.columns.tolist()
        
        print("\n🔄 应用中英文字段映射...")
        # 创建映射后的列名
        new_columns = []
        column_name_mapping = {}  # 保存原始列名到英文列名的映射
        
        for col in sample_df.columns:
            if col in column_mapping:
                new_col = column_mapping[col]
                column_name_mapping[col] = new_col
                new_columns.append(new_col)
                print(f"  {col} -> {new_col}")
            else:
                # 如果没有映射，保持原名但转为小写并替换特殊字符
                new_col = col.lower().replace('(', '_').replace(')', '').replace(' ', '_').replace('-', '_')
                column_name_mapping[col] = new_col
                new_columns.append(new_col)
                print(f"  {col} -> {new_col} (自动转换)")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(parquet_output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理第一批数据以确定schema
        print(f"\n🚀 开始分批处理数据...")
        
        # 应用数据类型优化到样本数据（跳过category类型避免编码冲突）
        for col, target_type in column_types.items():
            col_str = str(col)
            if col_str in sample_df.columns:
                if target_type == 'numeric':
                    sample_df[col_str] = pd.to_numeric(sample_df[col_str], errors='coerce')
                # 跳过category类型，保持为object以避免不同批次间的编码冲突
        
        # 优化日期列
        for col in sample_df.columns:
            col_str = str(col)
            if '日期' in col_str or 'date' in col_str.lower():
                try:
                    sample_df[col_str] = pd.to_datetime(sample_df[col_str], errors='coerce')
                except:
                    pass
        
        # 重命名列
        sample_df.columns = new_columns
        
        # 获取统一的schema
        unified_schema = pa.Table.from_pandas(sample_df).schema
        print(f"📋 确定统一schema: {len(unified_schema)} 个字段")
        
        # 创建Parquet writer
        parquet_writer = pq.ParquetWriter(parquet_output_path, unified_schema, compression='snappy')
        
        # 写入第一批数据
        first_table = pa.Table.from_pandas(sample_df, schema=unified_schema)
        parquet_writer.write_table(first_table)
        processed_rows = len(sample_df)
        
        print(f"\n📦 处理第 1/{(total_rows + batch_size - 1) // batch_size} 批 (行 1 - {len(sample_df):,})")
        memory_usage = get_memory_usage()
        progress = (processed_rows / total_rows) * 100
        print(f"✅ 已处理 {processed_rows:,}/{total_rows:,} 行 ({progress:.1f}%) | 内存: {memory_usage:.1f} MB")
        
        # 清理第一批数据
        del first_table
        
        # 处理剩余批次
        for batch_num in range(batch_size, total_rows, batch_size):
            current_batch = (batch_num // batch_size) + 1
            total_batches = (total_rows + batch_size - 1) // batch_size
            
            print(f"\n📦 处理第 {current_batch}/{total_batches} 批 (行 {batch_num+1:,} - {min(batch_num + batch_size, total_rows):,})")
            
            # 读取当前批次数据
            df_batch = read_hyper_file_batch(hyper_file_path, table_name, batch_num, batch_size)
            
            if len(df_batch) == 0:
                print("⚠️ 当前批次无数据，跳过")
                continue
            
            # 应用数据类型优化（跳过category类型避免编码冲突）
            for col, target_type in column_types.items():
                col_str = str(col)
                if col_str in df_batch.columns:
                    if target_type == 'numeric':
                        df_batch[col_str] = pd.to_numeric(df_batch[col_str], errors='coerce')
                    # 跳过category类型，保持为object以避免不同批次间的编码冲突
            
            # 优化日期列
            for col in df_batch.columns:
                col_str = str(col)
                if '日期' in col_str or 'date' in col_str.lower():
                    try:
                        df_batch[col_str] = pd.to_datetime(df_batch[col_str], errors='coerce')
                    except:
                        pass
            
            # 重命名列
            df_batch.columns = new_columns
            
            # 转换为PyArrow表，使用统一schema
            table = pa.Table.from_pandas(df_batch, schema=unified_schema)
            
            # 写入Parquet文件
            parquet_writer.write_table(table)
            processed_rows += len(df_batch)
            
            # 显示进度和内存使用
            memory_usage = get_memory_usage()
            progress = (processed_rows / total_rows) * 100
            print(f"✅ 已处理 {processed_rows:,}/{total_rows:,} 行 ({progress:.1f}%) | 内存: {memory_usage:.1f} MB")
            
            # 清理内存
            del df_batch, table
        
        # 关闭writer
        if parquet_writer:
            parquet_writer.close()
        
        # 验证保存结果
        file_size = os.path.getsize(parquet_output_path) / (1024 * 1024)  # MB
        memory_usage = get_memory_usage()
        print(f"\n✅ 转换完成！")
        print(f"📁 输出文件: {parquet_output_path}")
        print(f"📊 文件大小: {file_size:.2f} MB")
        print(f"💾 最终内存使用: {memory_usage:.2f} MB")
        
        # 生成schema.json文件（使用样本数据）
        # 注意：sample_df的列名已经在前面被修改为new_columns了
        generate_schema_json(sample_df, parquet_output_path, original_columns, column_name_mapping)
        
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def generate_schema_json(df, parquet_output_path, original_columns, column_name_mapping):
    """
    生成数据schema的JSON文件，包含中英文字段映射
    """
    # 基础文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(parquet_output_path))[0]
    schema_path = os.path.join(os.path.dirname(parquet_output_path), f"{base_name}_schema.json")
    
    # 字段类型映射
    dtype_mapping = {
        'object': '文本',
        'category': '分类',
        'int64': '整数',
        'int32': '整数',
        'int16': '整数',
        'int8': '整数',
        'uint64': '无符号整数',
        'uint32': '无符号整数',
        'uint16': '无符号整数',
        'uint8': '无符号整数',
        'float64': '浮点数',
        'float32': '浮点数',
        'datetime64[ns]': '日期时间',
        'bool': '布尔值'
    }
    
    # 字段注释映射（基于常见字段名）
    field_comments = {
        '日期': '数据记录日期',
        '年月': '年月信息',
        '品牌': '汽车品牌名称',
        '品牌（新）': '更新后的品牌分类',
        '厂商': '汽车制造厂商',
        '车系': '车型系列',
        '车型': '具体车型名称',
        '子车型': '细分车型',
        '车身形式': '车身类型（如SUV、轿车等）',
        '燃料种类': '燃料类型（如汽油、电动等）',
        '层级': '车型层级分类',
        '层级 (组)': '车型层级分组',
        '省': '省份',
        '市': '城市',
        '城市级别': '城市等级分类',
        '限购/限行/双非限': '城市限购限行政策',
        '上险数': '车辆上保险数量',
        '销量': '销售数量',
        '成交价格': '实际成交价格',
        'TP': '成交价格',
        '指导价': '厂商指导价格',
        '长(mm)': '车身长度（毫米）',
        '宽(mm)': '车身宽度（毫米）',
        '高(mm)': '车身高度（毫米）',
        '轴距(mm)': '轴距长度（毫米）'
    }
    
    # 构建schema结构
    schema = {
        base_name: {
            'description': '车辆上险量数据',
            'columns': list(df.columns),  # 英文列名
            'original_columns': original_columns,  # 原始中文列名
            'column_mapping': column_name_mapping,  # 中英文映射关系
            'column_explanations': {},
            'column_types': {},
            'value_mappings': {},
            'primary_metrics': [],
            'date_column': '',
            'metadata': {
                'time_granularity': 'monthly',
                'geo_dimension': 'province_city',
                'brand_dimension': 'brand',
                'notes': '车辆上险量月度统计数据，包含品牌、车型、地区等维度信息',
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'file_size_mb': round(os.path.getsize(parquet_output_path) / (1024 * 1024), 2)
            }
        }
    }
    
    # 填充字段信息
    for col in df.columns:
        # 数据类型
        dtype_str = str(df[col].dtype)
        schema[base_name]['column_types'][col] = dtype_mapping.get(dtype_str, dtype_str)
        
        # 字段注释 - 基于原始中文列名获取注释
        original_col = None
        for orig_col, mapped_col in column_name_mapping.items():
            if mapped_col == col:
                original_col = orig_col
                break
        
        if original_col:
            schema[base_name]['column_explanations'][col] = field_comments.get(original_col, f'{original_col}字段')
        else:
            schema[base_name]['column_explanations'][col] = f'{col}字段'
        
        # 识别日期列
        if 'date' in col.lower() or 'year_month' in col.lower():
            schema[base_name]['date_column'] = col
        
        # 识别主要指标
        if any(keyword in col for keyword in ['insurance_volume', 'sales_volume', 'volume']):
            schema[base_name]['primary_metrics'].append(col)
    
    # 如果没有找到主要指标，添加默认的
    if not schema[base_name]['primary_metrics']:
        numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        if numeric_cols:
            schema[base_name]['primary_metrics'] = numeric_cols[:3]  # 取前3个数值列
    
    # 保存schema文件
    try:
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
        print(f"📋 Schema文件已生成: {schema_path}")
    except Exception as e:
        print(f"⚠️ Schema文件生成失败: {e}")

if __name__ == "__main__":
    hyper_file = "/Users/zihao_/Documents/coding/dataset/original/乘用车上险量_0826.hyper"
    parquet_file = "/Users/zihao_/Documents/coding/dataset/formatted/乘用车上险量_0826.parquet"
    
    # 批处理大小配置（可根据内存情况调整）
    batch_size = 100000  # 每批处理10万行
    
    # 检查Hyper文件是否存在
    if not os.path.exists(hyper_file):
        print(f"❌ 错误: Hyper文件不存在: {hyper_file}")
        exit(1)
    
    print("🚀 开始大数据集Hyper到Parquet分批转换...")
    print("=" * 60)
    print(f"📦 批处理配置: 每批 {batch_size:,} 行")
    print(f"💾 内存优化: 分批处理避免内存溢出")
    print("=" * 60)
    
    success = hyper_to_parquet_optimized(hyper_file, parquet_file, batch_size)
    
    if success:
        print("\n🎉 大数据集转换成功完成！")
        print(f"📁 输出文件: {parquet_file}")
        print(f"🔧 优化效果: 内存使用控制在合理范围内")
    else:
        print("\n❌ 转换失败")
        exit(1)