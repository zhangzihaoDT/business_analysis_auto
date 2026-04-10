import pandas as pd
import numpy as np
import json
import re
import subprocess
import os
import sys
from pathlib import Path

def parse_sql_condition(df, condition_str):
    def not_like_replacer(match):
        val = match.group(1)
        return f"~df['product_name'].str.contains('{val}', na=False, regex=False)"

    condition_str = re.sub(r"product_name\s+NOT\s+LIKE\s+'%([^%]+)%+'", not_like_replacer, condition_str)

    def like_replacer(match):
        val = match.group(1)
        return f"df['product_name'].str.contains('{val}', na=False, regex=False)"

    condition_str = re.sub(r"product_name\s+LIKE\s+'%([^%]+)%+'", like_replacer, condition_str)

    condition_str = condition_str.replace(" AND ", " & ").replace(" OR ", " | ")

    try:
        return eval(condition_str)
    except Exception as e:
        print(f"⚠️ 解析条件失败: {condition_str}, Error: {e}")
        return pd.Series([False] * len(df), index=df.index)

def apply_series_group_logic(df, business_def):
    logic = business_def.get("series_group_logic", {})
    if "product_name" not in df.columns:
        df["series_group_logic"] = pd.NA
        return df

    group_col = pd.Series(pd.NA, index=df.index, dtype="string")
    default_group = "其他"
    for group, cond in logic.items():
        if str(cond).strip().upper() == "ELSE":
            default_group = group
            continue
        mask = parse_sql_condition(df, str(cond))
        if not isinstance(mask, pd.Series):
            continue
        mask = mask.fillna(False)
        assignable = group_col.isna() & mask
        if assignable.any():
            group_col = group_col.where(~assignable, group)

    df["series_group_logic"] = group_col.fillna(default_group).astype("string")
    return df

def fetch_tableau_data(url, output_csv):
    print(f"正在从 Tableau 导出数据: {url}")
    
    # 尝试加载 .env 文件中的变量
    env_path = Path(os.path.dirname(__file__)) / ".env"
    token_name = ""
    token_value = ""
    if env_path.exists():
        try:
            for raw in env_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                if k == "TABLEAU_TOKEN_NAME":
                    token_name = v
                elif k == "TABLEAU_TOKEN_VALUE":
                    token_value = v
        except Exception as e:
            print(f"⚠️ 读取 .env 文件失败: {e}")

    # 如果环境变量里已经有则优先使用环境变量的
    token_name = os.environ.get("TABLEAU_TOKEN_NAME", token_name)
    token_value = os.environ.get("TABLEAU_TOKEN_VALUE", token_value)

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "tableau_export.py"),
        "--view", url,
        "--output", output_csv,
        "--format", "csv",
        "--timeout", "600"
    ]
    if token_name and token_value:
        cmd.extend(["--token-name", token_name, "--token-value", token_value])
        print(f"使用 Token 登录: {token_name}")
    else:
        print("未检测到 TABLEAU_TOKEN_NAME 或 TABLEAU_TOKEN_VALUE，将尝试使用默认方式登录。")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️ 导出数据失败:\n{result.stderr}\n{result.stdout}")
        return False
    print("✅ 导出成功")
    return True

def read_tableau_csv(csv_file, min_columns=2):
    encodings_to_try = ["utf-8-sig", "utf-16", "utf-8", "gb18030", "gbk"]
    for enc in encodings_to_try:
        try:
            df_tmp = pd.read_csv(csv_file, encoding=enc, sep="\t", low_memory=False)
            if len(df_tmp.columns) <= 1:
                df_tmp = pd.read_csv(csv_file, encoding=enc, sep=",", low_memory=False)
            if len(df_tmp.columns) >= min_columns:
                return df_tmp
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    return None

def main():
    tableau_url = "https://tableau-hs.immotors.com/#/views/17/config_attribute"
    wheel_url = "https://tableau-hs.immotors.com/#/views/17/wheel"
    
    # 定义基础路径
    base_dir = Path("/Users/zihao_/Documents/coding/dataset")
    original_dir = base_dir / "original"
    formatted_dir = base_dir / "formatted"
    
    # 确保目录存在
    original_dir.mkdir(parents=True, exist_ok=True)
    formatted_dir.mkdir(parents=True, exist_ok=True)
    
    config_csv_path_2026 = original_dir / "config_attribute_data_2026.csv"
    wheel_csv_path_2026 = original_dir / "wheel_data_2026.csv"
    config_parquet_path = formatted_dir / "config_attribute.parquet"
    order_data_path = formatted_dir / "order_data.parquet"
    business_def_path = Path("/Users/zihao_/Documents/github/26W06_Tool_calls/schema/business_definition.json")

    # 1. Fetch raw data for 2026
    if not config_csv_path_2026.exists():
        success = fetch_tableau_data(tableau_url, str(config_csv_path_2026))
        if not success:
            print("❌ 无法获取 Tableau 数据，请检查权限或网络。")
            return

    if not wheel_csv_path_2026.exists():
        success = fetch_tableau_data(wheel_url, str(wheel_csv_path_2026))
        if not success:
            print("⚠️ 无法获取 wheel 数据，将跳过轮毂补全。")

    # 2. Merge CSV files and convert config data to Parquet for performance
    print("正在加载并合并所有年份的配置文件...")
    years = ["2023", "2024", "2025", "2026"]
    df_list = []
    
    for year in years:
        csv_file = original_dir / f"config_attribute_data_{year}.csv"
        # 兼容处理用户拼写错误的情况，例如 config_attribute_data_20234csv
        if year == "2024":
            alt_csv_file = original_dir / "config_attribute_data_20234csv"
            if not csv_file.exists() and alt_csv_file.exists():
                csv_file = alt_csv_file

        if csv_file.exists():
            print(f" - 读取: {csv_file.name}")
            df_year = read_tableau_csv(csv_file, min_columns=2)
                     
            if df_year is not None:
                df_list.append(df_year)
            else:
                print(f"❌ 无法使用任何支持的编码读取文件: {csv_file.name}")
        else:
            print(f"⚠️ 找不到文件: {csv_file.name}")
            
    if not df_list:
        print("❌ 没有找到任何 config_attribute_data_*.csv 文件。")
        return
        
    config_df = pd.concat(df_list, ignore_index=True)
    print(f"✅ 合并完成，总行数: {len(config_df)}")

    # Standardize column name for Order Number
    order_col_name = None
    for col in config_df.columns:
        if col.lower().replace(" ", "_") in ["order_number", "ordernumber", "order_no"]:
            order_col_name = col
            break
            
    attr_col_name = None
    for col in config_df.columns:
        if "attribute" in col.lower():
            attr_col_name = col
            break

    if order_col_name and attr_col_name:
        print("正在解析 Attribute 列...")
        # 仅保留订单号和属性列并去除空值
        parsed_df = config_df[[order_col_name, attr_col_name]].dropna(subset=[attr_col_name, order_col_name]).copy()
        
        # 定义一个函数，使用正则精确提取键值对
        # 匹配模式：(非冒号逗号的字符) : (非逗号的字符)
        # 这样能有效避免由于异常的逗号或冒号导致的拆分错误
        import re
        pattern = re.compile(r'([^:,]+):([^,]+)')
        
        def extract_kv(text):
            if not isinstance(text, str):
                return []
            return pattern.findall(text)
            
        # 提取出所有的键值对元组列表
        parsed_df['kv_pairs'] = parsed_df[attr_col_name].apply(extract_kv)
        
        # 将列表炸开 (Explode)
        parsed_df = parsed_df.explode('kv_pairs').dropna(subset=['kv_pairs'])
        
        # 从元组中分离出 Attribute 和 value
        parsed_df['parsed_Attribute'] = parsed_df['kv_pairs'].apply(lambda x: x[0].strip())
        parsed_df['value'] = parsed_df['kv_pairs'].apply(lambda x: x[1].strip())
        
        # 删除原始的属性列和临时列
        if attr_col_name in parsed_df.columns:
            parsed_df = parsed_df.drop(columns=[attr_col_name])
        parsed_df = parsed_df.drop(columns=['kv_pairs'])
            
        parsed_df = parsed_df.rename(columns={order_col_name: 'Order Number', 'parsed_Attribute': 'Attribute'})
        config_df = parsed_df
        order_col_name = 'Order Number'
        print(f"✅ 解析完成，长表格式总行数: {len(config_df)}")

        wheel_df = read_tableau_csv(wheel_csv_path_2026, min_columns=1) if wheel_csv_path_2026.exists() else None
        if wheel_df is not None and {"Attribute", "value"}.issubset(set(config_df.columns)):
            # 找到 wheel 和 series 列
            wheel_cols = [c for c in wheel_df.columns if any(k in c.lower() for k in ["wheel", "rim", "轮毂", "轮辋"])]
            series_cols = [c for c in wheel_df.columns if "series" in c.lower()]
            
            wheel_col = wheel_cols[0] if wheel_cols else None
            series_col = series_cols[0] if series_cols else None

            if wheel_col:
                # 构建基于 Series 的轮毂候选集字典，和全局候选集
                wheel_candidates_by_series = {}
                if series_col:
                    for series_name, group in wheel_df.groupby(series_col):
                        wheel_candidates_by_series[series_name] = group[wheel_col].dropna().astype(str).str.strip().tolist()
                
                all_wheels = wheel_df[wheel_col].dropna().astype(str).str.strip().tolist()
                
                # 提前加载订单数据，用于获取每辆车的 series
                print("正在加载业务定义和订单数据用于轮毂补全...")
                with open(business_def_path, 'r', encoding='utf-8') as f:
                    business_def = json.load(f)
                
                # 为了不影响后续逻辑，这里只取需要的列并提取 series
                order_df_temp = pd.read_parquet(order_data_path)
                order_df_temp = apply_series_group_logic(order_df_temp, business_def)
                
                # 建立 series_group_logic 到 wheel Series (例如 CM0 -> LS6, DM0 -> L6) 的反向映射
                logic_to_series = {}
                for series_name, logics in business_def.get("model_series_mapping", {}).items():
                    for logic in logics:
                        logic_to_series[logic] = series_name
                # 未明确映射的，假设同名 (例如 LS9 -> LS9)
                for logic in ["LS8", "LS9", "LS7", "L7"]:
                    logic_to_series[logic] = logic
                    
                order_col_temp = "order_number" if "order_number" in order_df_temp.columns else order_df_temp.columns[0]
                order_logic_map = order_df_temp.set_index(order_col_temp)["series_group_logic"].to_dict()

                m_wheel = (config_df["Attribute"] == "轮毂") & config_df["value"].notna()
                
                # 建立映射列
                config_df["_car_logic"] = config_df["Order Number"].map(order_logic_map)
                config_df["_car_series"] = config_df["_car_logic"].map(logic_to_series)
                
                def match_wheel(row):
                    v = str(row["value"]).strip()
                    s = row["_car_series"]
                    
                    # 确定当前订单适用的候选集
                    candidates = wheel_candidates_by_series.get(s, all_wheels)
                    
                    if v in candidates:
                        return v
                        
                    # 尝试前缀匹配
                    matched = [x for x in candidates if x.startswith(v)]
                    if len(matched) == 1:
                        return matched[0]
                    elif len(matched) > 1:
                        return max(matched, key=len)
                        
                    # 如果没有找到匹配，回退到全局匹配
                    if s in wheel_candidates_by_series:
                        matched_global = [x for x in all_wheels if x.startswith(v)]
                        if len(matched_global) == 1:
                            return matched_global[0]
                        elif len(matched_global) > 1:
                            return max(matched_global, key=len)
                            
                    return v

                print("正在应用更精确的基于 Series 的轮毂补全策略...")
                config_df.loc[m_wheel, "value"] = config_df[m_wheel].apply(match_wheel, axis=1)
                config_df = config_df.drop(columns=["_car_logic", "_car_series"])
                
    else:
        print("⚠️ 未能找到订单号或 Attribute 列，跳过长表解析。")

    config_df.to_parquet(config_parquet_path, index=False)
    print(f"✅ 合并解析后的配置文件已保存至 {config_parquet_path}")

    if not order_col_name:
        print("⚠️ 未能在 Tableau 数据中找到订单号列（Order Number），将无法计算选配数！")
        configured_orders = set()
    else:
        configured_orders = set(config_df[order_col_name].dropna().astype(str))

    # 3. Load Business Definition & Order Data
    print("正在准备最终统计...")
    # 由于前面已经读取过一次业务定义和订单数据，这里可以直接使用 order_df_temp
    # 或者重新获取全量列以保证无副作用
    if 'business_def' not in locals():
        with open(business_def_path, 'r', encoding='utf-8') as f:
            business_def = json.load(f)
    
    if 'order_df_temp' in locals():
        order_df = order_df_temp
    else:
        order_df = pd.read_parquet(order_data_path)
        order_df = apply_series_group_logic(order_df, business_def)

    if not pd.api.types.is_datetime64_any_dtype(order_df["intention_payment_time"]):
        order_df["intention_payment_time"] = pd.to_datetime(order_df["intention_payment_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(order_df["intention_refund_time"]):
        order_df["intention_refund_time"] = pd.to_datetime(order_df["intention_refund_time"], errors="coerce")

    target_models = ["CM0", "DM0", "CM1", "DM1", "CM2", "LS9", "LS8"]
    
    print("\n" + "="*75)
    print(f"{'车型':<8} | {'订单数':<10} | {'选配数':<10} | {'小订数':<10} | {'留存小订数':<12} | {'小订留存选配数':<12}")
    print("-" * 75)

    time_periods = business_def.get("time_periods", {})

    for model in target_models:
        df_model = order_df[order_df["series_group_logic"] == model]
        
        # 订单数 (Total orders for the model)
        total_orders = df_model["order_number"].nunique()
        
        # 预售期判断相关变量
        tp = time_periods.get(model, {})
        start_str = tp.get("start")
        end_str = tp.get("end")
        
        # 小订数 (Intention payment time is not null)
        intention_mask = df_model["intention_payment_time"].notna()
        intention_orders = df_model[intention_mask]
        small_orders_count = intention_orders["order_number"].nunique()
        
        # 留存小订数计算（对齐 analyze_order.py 口径）：
        # 1. 在预售期间内小订 (start_day <= intention_payment_time < start_day + N_days)
        # 2. 未退订或在窗口外退订 (intention_refund_time is na OR intention_refund_time > start_day + N_days)
        if start_str and end_str:
            start_day = pd.Timestamp(start_str)
            presale_end_day = pd.Timestamp(end_str)
            n_days = int((presale_end_day.normalize() - start_day.normalize()).days + 1)
            n_days = max(1, n_days)
            
            presale_end_excl = presale_end_day + pd.Timedelta(days=1)
            window_end_excl = start_day + pd.Timedelta(days=int(n_days))
            window_end_excl = min(window_end_excl, presale_end_excl)
            
            m_retention = (
                intention_orders["intention_payment_time"] >= start_day
            ) & (
                intention_orders["intention_payment_time"] < window_end_excl
            ) & (
                intention_orders["intention_refund_time"].isna() | (intention_orders["intention_refund_time"] > window_end_excl)
            )
            retention_orders = intention_orders[m_retention]
        else:
            # Fallback 如果没有配置预售期，沿用旧逻辑（仅判断 intention_refund_time 为空）
            retention_mask = intention_orders["intention_refund_time"].isna()
            retention_orders = intention_orders[retention_mask]

        retained_small_orders_count = retention_orders["order_number"].nunique()
        
        # 选配数 (总选配数) & 小订留存选配数
        if configured_orders:
            configured_count = df_model.loc[df_model["order_number"].astype(str).isin(configured_orders), "order_number"].nunique()
            retained_configured_count = retention_orders.loc[retention_orders["order_number"].astype(str).isin(configured_orders), "order_number"].nunique()
        else:
            configured_count = 0
            retained_configured_count = 0
            
        print(f"{model:<10} | {total_orders:<13} | {configured_count:<10} | {small_orders_count:<13} | {retained_small_orders_count:<15} | {retained_configured_count:<15}")
    
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
