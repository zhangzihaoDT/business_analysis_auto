import duckdb
import yaml

import duckdb
import yaml
import json
import os

def load_field_mapping_from_dir(directory: str):
    """
    从指定目录中加载所有 *_mapping.json 文件，并合并为一个 dict。
    """
    field_mapping = {}
    for filename in os.listdir(directory):
        if filename.endswith("_mapping.json"):
            path = os.path.join(directory, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    mapping = json.load(f)
                    field_mapping.update(mapping)
            except Exception as e:
                print(f"⚠️ 读取映射文件失败: {path}, 错误: {e}")
    return field_mapping

def generate_schema_context_from_duckdb(db_path: str, output_yaml_path: str, formatted_dir: str = None):
    conn = duckdb.connect(db_path)

    # 加载中英字段映射（可选）
    field_mapping = {}
    if formatted_dir and os.path.exists(formatted_dir):
        field_mapping = load_field_mapping_from_dir(formatted_dir)
        print(f"🧩 读取到字段映射数: {len(field_mapping)}")

    # 获取所有表名
    tables = conn.execute("SHOW TABLES").fetchall()
    schema_context = {}

    for (table_name,) in tables:
        # 获取字段信息
        columns_info = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        columns = [col[1] for col in columns_info]

        # 中英字段解释（优先使用映射）
        column_explanations = {
            col: field_mapping.get(col, "") for col in columns
        }

        schema_context[table_name] = {
            "description": "",
            "columns": columns,
            "column_explanations": column_explanations,
            "primary_metrics": [],
            "date_column": "",
            "metadata": {
                "time_granularity": "",
                "geo_dimension": "",
                "brand_dimension": "",
                "notes": ""
            }
        }

    # 输出为 YAML
    with open(output_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump({"table_explanations": schema_context}, f, allow_unicode=True, sort_keys=False)

    print(f"✅ schema_context YAML 已生成: {output_yaml_path}")


# 用法示例
generate_schema_context_from_duckdb(
    db_path="data/insurance_sales_en.duckdb",
    output_yaml_path="data/profile_automotive.yaml",
    formatted_dir="/Users/zihao_/Documents/coding/dataset/formatted/"  # 自动加载 *_mapping.json
)