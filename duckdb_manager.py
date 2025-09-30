import duckdb
import os
from datetime import datetime
import hashlib

# 数据库文件路径
DB_PATH = "/Users/zihao_/Documents/coding/Langchain_chatwithdata/database/central_analytics.duckdb"
# Parquet文件列表
PARQUET_FILES_TO_IMPORT = [
    "/Users/zihao_/Documents/coding/dataset/价格配置_data_transposed.parquet",
    "/Users/zihao_/Documents/coding/dataset/用户分层画像_data_all_clean.parquet",
    "/Users/zihao_/Documents/coding/dataset/上险数_03_data_截止 202504_clean.parquet",
    "/Users/zihao_/Documents/coding/dataset/订单观察_聚合结果.parquet"
]

def create_connection():
    """创建与DuckDB数据库的连接"""
    try:
        # Ensure the directory for the database file exists
        db_directory = os.path.dirname(DB_PATH)
        if db_directory and not os.path.exists(db_directory):
            os.makedirs(db_directory)
            print(f"数据库目录已创建: {db_directory}")
        conn = duckdb.connect(DB_PATH)
        return conn
    except Exception as e:
        print(f"数据库连接错误: {e}")
        return None

def get_file_hash(file_path):
    """计算文件的MD5哈希值，用于检测文件是否变化"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def create_metadata_table(conn):
    """创建元数据表，用于跟踪文件信息"""
    # 检查表是否存在
    result = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='file_metadata'").fetchone()
    
    if result:
        # 检查表结构，看是否有encoding列
        columns_df = conn.execute("PRAGMA table_info(file_metadata)").fetchdf()
        column_names = columns_df['name'].tolist()
        
        if 'encoding' not in column_names:
            # 添加encoding列
            print("正在向file_metadata表添加encoding列...")
            conn.execute("ALTER TABLE file_metadata ADD COLUMN encoding VARCHAR")
    else:
        # 创建新表
        conn.execute('''
        CREATE TABLE file_metadata (
            file_path VARCHAR PRIMARY KEY,
            table_name VARCHAR,
            last_hash TEXT,
            last_modified TIMESTAMP,
            last_import TIMESTAMP,
            record_count INTEGER,
            encoding VARCHAR
        )
        ''')
        print("已创建file_metadata表")

def register_parquet_file(conn, parquet_path, table_name=None):
    """将Parquet文件的数据导入到DuckDB指定的表中"""
    if not os.path.exists(parquet_path):
        print(f"Parquet文件不存在: {parquet_path}")
        return

    # 1. 准备表名
    if table_name is None:
        base_name = os.path.splitext(os.path.basename(parquet_path))[0]
        table_name = base_name # 使用原始文件名（不含扩展名）作为表名

    # 2. 获取文件信息
    last_modified = datetime.fromtimestamp(os.path.getmtime(parquet_path))
    file_hash = get_file_hash(parquet_path)

    # 3. 检查文件是否已注册且未更改，同时检查表是否存在
    meta_result = conn.execute(
        "SELECT last_hash FROM file_metadata WHERE file_path = ?",
        [parquet_path]
    ).fetchone()

    table_exists = False
    try:
        conn.execute(f'SELECT 1 FROM "{table_name}" LIMIT 1') # 使用双引号包围表名以处理特殊字符
        table_exists = True
    except duckdb.CatalogException: # 表不存在
        table_exists = False
    except Exception as e: # 其他错误
        print(f"检查表 '{table_name}' 是否存在时出错: {e}")
        # 决定是否继续，这里假设如果检查出错则认为表可能不存在或有问题，尝试重建
        table_exists = False 


    if meta_result and meta_result[0] == file_hash and table_exists:
        print(f"Parquet文件 {parquet_path} 未更改且表 '{table_name}' 已存在，跳过导入。")
        return
    elif meta_result and meta_result[0] == file_hash and not table_exists:
        print(f"Parquet文件 {parquet_path} 未更改但表 '{table_name}' 不存在，将重新创建表。")
    elif not meta_result:
        print(f"Parquet文件 {parquet_path} 首次注册，将创建表 '{table_name}'。")
    else: # File hash changed
        print(f"Parquet文件 {parquet_path} 已更改，将重新创建表 '{table_name}'。")

    # 4. 创建或替换表
    try:
        print(f"正在从 {parquet_path} 创建/替换表 '{table_name}'...")
        # 使用双引号包围表名，单引号包围路径
        conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM read_parquet(\'{parquet_path}\')')
        print(f"表 '{table_name}' 已成功从 {parquet_path} 创建/替换。")

        # 5. 获取记录数
        record_count_result = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()
        record_count = record_count_result[0] if record_count_result else 0


        # 6. 更新元数据表
        conn.execute(
            """
            INSERT OR REPLACE INTO file_metadata 
            (file_path, table_name, last_hash, last_modified, last_import, record_count, encoding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [parquet_path, table_name, file_hash, last_modified, datetime.now(), record_count, None] # Encoding is None for Parquet
        )
        print(f"元数据已为 {parquet_path} 更新，表 '{table_name}' 包含 {record_count} 条记录。")

    except Exception as e:
        print(f"处理Parquet文件 {parquet_path} 到表 '{table_name}' 时出错: {e}")


if __name__ == "__main__":
    conn = create_connection()
    if conn:
        create_metadata_table(conn) # 确保元数据表存在

        for parquet_file_path in PARQUET_FILES_TO_IMPORT:
            # 从文件名派生表名 (例如: "价格配置_data_transposed")
            table_name_derived = os.path.splitext(os.path.basename(parquet_file_path))[0]
            register_parquet_file(conn, parquet_file_path, table_name=table_name_derived)
        
        print("\n数据库中的所有表:")
        try:
            all_tables = conn.execute("SHOW TABLES;").fetchall()
            if all_tables:
                for tbl in all_tables:
                    print(f"- {tbl[0]}")
            else:
                print("数据库中没有表。")
        except Exception as e:
            print(f"显示表时出错: {e}")

        conn.close()
        print(f"\n所有指定的Parquet文件处理完成。数据库已保存到: {DB_PATH}")
    else:
        print("未能连接到数据库，操作中止。")