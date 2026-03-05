import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import json
import re
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm

PARQUET_FILE = Path("/Users/zihao_/Documents/coding/dataset/formatted/order_full_data.parquet")
BUSINESS_DEF_FILE = Path("/Users/zihao_/Documents/github/W52_reasoning/world/business_definition.json")
# Use script directory to determine output path
SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT = SCRIPT_DIR / "reports/review_2025.html"

def load_business_definition(file_path: Path) -> dict:
    """加载业务定义文件"""
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_sql_condition(df: pd.DataFrame, condition_str: str) -> pd.Series:
    """
    解析简单的 SQL-like 条件并应用到 DataFrame
    支持: LIKE, NOT LIKE, AND, OR
    例如: "product_name LIKE '%52%' OR product_name LIKE '%66%'"
    """
    # 1. 替换 NOT LIKE
    # pattern: product_name NOT LIKE '%value%'
    # replacement: ~df['product_name'].str.contains('value', na=False, regex=False)
    
    def not_like_replacer(match):
        val = match.group(1)
        return f"~df['product_name'].str.contains('{val}', na=False, regex=False)"
    
    condition_str = re.sub(r"product_name\s+NOT\s+LIKE\s+'%([^%]+)%+'", not_like_replacer, condition_str)
    
    # 2. 替换 LIKE
    # pattern: product_name LIKE '%value%'
    # replacement: df['product_name'].str.contains('value', na=False, regex=False)
    
    def like_replacer(match):
        val = match.group(1)
        return f"df['product_name'].str.contains('{val}', na=False, regex=False)"
        
    condition_str = re.sub(r"product_name\s+LIKE\s+'%([^%]+)%+'", like_replacer, condition_str)
    
    # 3. 替换 AND / OR
    condition_str = condition_str.replace(" AND ", " & ").replace(" OR ", " | ")
    
    # 4. Eval
    try:
        return eval(condition_str)
    except Exception as e:
        print(f"⚠️ 解析条件失败: {condition_str}, Error: {e}")
        return pd.Series([False] * len(df), index=df.index)

def load_data(file_path: Path) -> pd.DataFrame:
    """加载 Parquet 数据"""
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    print(f"📖 Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"✅ Loaded {len(df)} rows.")
    return df

def get_period_mask(df: pd.DataFrame, date_col: str, year: int) -> pd.Series:
    """
    生成指定年份的时间过滤掩码
    year=2024: 2024-01-01 ~ 2024-12-31
    year=2025: 2025-01-01 ~ max (即 >= 2025-01-01)
    """
    if date_col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
        
    # 确保是 datetime 类型
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
         df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    start_date = pd.Timestamp(f"{year}-01-01")
    end_date = pd.Timestamp(f"{year}-12-31 23:59:59")
    return (df[date_col] >= start_date) & (df[date_col] <= end_date)

def calculate_store_tenure_metrics(df):
    """
    Module 4.3.2: Store Tenure Analysis
    Calculates store tenure (Max Lock Time - Store Create Date) for Retained Lock Orders.
    Returns bin statistics for 2024 and 2025.
    """
    results = {}
    
    # Define bins
    # 0-30, 31-60, ...
    bins = [-1, 30, 90, 180, 360, 720, 1080, 100000]
    labels = ['0-30', '31-90', '91-180', '181-360', '361-720', '721-1080', '>1080']
    
    # Use internal helper to get mask if possible, but get_metric_mask is inside calculate_metrics.
    # We can duplicate the logic or pass the mask.
    # To keep it simple, I'll implement the filtering logic here or move get_metric_mask out.
    # Moving get_metric_mask is risky due to scope.
    # I'll re-implement the specific mask logic here for "Retained Lock Orders".
    
    for year in [2024, 2025]:
        # Filter for Retained Lock Orders: lock_time in year AND approve_refund_time is NULL
        start_date = pd.Timestamp(f"{year}-01-01")
        end_date = pd.Timestamp(f"{year}-12-31 23:59:59")
        lock_mask = (df['lock_time'] >= start_date) & (df['lock_time'] <= end_date)
            
        # Refund mask: approve_refund_time is NaT
        # Ensure approve_refund_time is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['approve_refund_time']):
             # We assume it's already converted in calculate_metrics or main, but to be safe:
             # Actually, we shouldn't modify df here if it's a view.
             # We'll assume the caller (calculate_metrics) has cleaned the data or we check safely.
             pass 
             
        refund_mask = df['approve_refund_time'].isna()
        
        mask = lock_mask & refund_mask
        df_year = df[mask].copy()
        
        if df_year.empty:
            results[year] = pd.DataFrame()
            continue
            
        # Retained orders per store (for locks & last lock time)
        store_stats = df_year.groupby('store_name').agg({
            'lock_time': 'max',
            'store_create_date': 'first',
            'order_number': 'count'
        }).rename(columns={'order_number': 'retained_locks'})
        
        store_stats['lock_time'] = pd.to_datetime(store_stats['lock_time'])
        store_stats['store_create_date'] = pd.to_datetime(store_stats['store_create_date'])
        
        period_end = pd.Timestamp(f"{year}-12-31 23:59:59")
        period_end_per_store = np.minimum(period_end.value, store_stats['lock_time'].astype('int64'))
        period_end_per_store = pd.to_datetime(period_end_per_store)
        store_stats['tenure_days'] = (period_end_per_store - store_stats['store_create_date']).dt.days
        # Clip negative to 0
        store_stats['tenure_days'] = store_stats['tenure_days'].clip(lower=0)
        
        # Active days within observation year (strict, aligned with 4.2)
        # Build daily counts for all locks in the year (regardless of refund)
        df_year_all = df[lock_mask].copy()
        df_year_all['date'] = pd.to_datetime(df_year_all['lock_time']).dt.floor('D')
        df_year_all = df_year_all.dropna(subset=['store_name', 'date'])
        
        # Full days of the year
        full_days = pd.date_range(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"), freq='D')
        
        if not df_year_all.empty:
            # Opening date per store (min)
            open_map = df_year_all.groupby('store_name')['store_create_date'].min()
            open_map = pd.to_datetime(open_map)
            open_map = open_map.reindex(store_stats.index)
            # Fill missing open dates with start of year
            open_map = open_map.fillna(pd.Timestamp(f"{year}-01-01"))
            
            # Daily counts pivot
            daily_counts = df_year_all.groupby(['date', 'store_name']).size().unstack(fill_value=0)
            # Ensure columns cover stores we care about
            daily_counts = daily_counts.reindex(full_days, fill_value=0)
            missing_cols = [s for s in store_stats.index if s not in daily_counts.columns]
            if missing_cols:
                for s in missing_cols:
                    daily_counts[s] = 0
            # Ensure column order aligns
            daily_counts = daily_counts[store_stats.index]
            
            # Rolling 30-day activity
            rolling_activity = daily_counts.rolling(window=30, min_periods=1).sum()
            
            # Compute active days per store: activity>0 and day>=open_date
            active_days = {}
            for s in store_stats.index:
                open_date = open_map.loc[s]
                day_mask = (rolling_activity.index >= open_date)
                act_mask = rolling_activity[s].values > 0
                active_days[s] = int(np.sum(day_mask & act_mask))
            store_stats['active_days'] = pd.Series(active_days)
        else:
            store_stats['active_days'] = 0
        
        # Binning
        store_stats['tenure_bin'] = pd.cut(store_stats['tenure_days'], bins=bins, labels=labels)
        
        # Aggregation by Bin
        bin_stats = store_stats.groupby('tenure_bin', observed=False).agg({
            'retained_locks': ['count', 'sum'],
            'active_days': ['sum']
        })
        
        # Flatten columns
        bin_stats.columns = ['store_count', 'total_locks', 'total_active_days']
        # Calculate Avg Locks per Store
        bin_stats['avg_locks'] = bin_stats['total_locks'] / bin_stats['store_count']
        bin_stats['avg_locks'] = bin_stats['avg_locks'].fillna(0)
        # Calculate Avg Daily Locks per Store (store-day)
        with np.errstate(divide='ignore', invalid='ignore'):
            bin_stats['avg_daily_locks'] = bin_stats['total_locks'] / bin_stats['total_active_days']
            bin_stats['avg_daily_locks'] = bin_stats['avg_daily_locks'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        results[year] = bin_stats
        
    return results

def calculate_metrics(df: pd.DataFrame) -> dict:
    """计算核心指标"""
    metrics = {}
    
    # 定义指标列表
    metric_names = ["锁单数", "开票数", "锁单退订数", "留存锁单数"]
    
    def get_metric_mask(metric_name: str, year: int) -> pd.Series:
        """
        获取指定指标在指定年份的过滤掩码
        """
        if metric_name == "锁单数":
            return get_period_mask(df, "lock_time", year)
            
        elif metric_name == "开票数":
            return get_period_mask(df, "invoice_upload_time", year)
            
        elif metric_name == "锁单退订数":
            # approve_refund_time 在这两个周期
            # 且 lock_time 不为空 (not null)
            
            # 1. approve_refund_time 在周期内
            time_mask = get_period_mask(df, "approve_refund_time", year)
            
            # 2. lock_time 不为空
            if "lock_time" not in df.columns:
                 lock_mask = pd.Series([False] * len(df), index=df.index)
            else:
                 lock_mask = df["lock_time"].notna()
                 
            return time_mask & lock_mask

        elif metric_name == "留存锁单数":
            # lock_time 在周期内 且 (approve_refund_time 为空 或 approve_refund_time > 周期结束)
            
            # 1. lock_time 在周期内
            lock_mask = get_period_mask(df, "lock_time", year)
            
            # 2. approve_refund_time 为空 或 > 周期结束
            if "approve_refund_time" not in df.columns:
                 refund_mask = pd.Series([True] * len(df), index=df.index)
            else:
                 # Ensure datetime
                 if not pd.api.types.is_datetime64_any_dtype(df["approve_refund_time"]):
                      df["approve_refund_time"] = pd.to_datetime(df["approve_refund_time"], errors='coerce')
                 
                 # Strict definition: Never refunded (as requested by user)
                 refund_mask = df["approve_refund_time"].isna()
            
            return lock_mask & refund_mask
            
        return pd.Series([False] * len(df), index=df.index)
    
    # 1. 总体概览
    overall_stats = []
    
    for metric_name in metric_names:
        # 2024 数据
        mask_2024 = get_metric_mask(metric_name, 2024)
        count_2024 = df[mask_2024]['order_number'].nunique()
        
        # 2025 数据
        mask_2025 = get_metric_mask(metric_name, 2025)
        count_2025 = df[mask_2025]['order_number'].nunique()
        
        # 同比 (YoY) - 注意：2025可能不完整，这里仅计算简单增长率供参考，或者留空
        yoy = ((count_2025 - count_2024) / count_2024) if count_2024 > 0 else 0.0
        
        ratio_str = f"{yoy:.1%}"
        if yoy < 0:
            ratio_str = f"<span style='color: red'>{ratio_str}</span>"
            
        overall_stats.append({
            "指标": metric_name,
            "2024 全年": count_2024,
            "2025 (至今)": count_2025,
            "Diff": count_2025 - count_2024,
            "Ratio": ratio_str
        })
        
    metrics['overall'] = pd.DataFrame(overall_stats)
    
    # 2. 分 Series 对比 (拆分为三个独立的 DataFrame)
    series_details = {}
    
    # 明确指定需要展示的 series 顺序，并添加 Total
    target_series = ["L6", "LS6", "LS9"]
    
    for metric_name in metric_names:
        rows = []
        
        # 1. 计算各车型数据
        for s in target_series:
            # Filter by series
            series_mask = df['series'] == s
            
            # 2024
            mask_2024 = series_mask & get_metric_mask(metric_name, 2024)
            val_2024 = df[mask_2024]['order_number'].nunique()
            
            # 2025
            mask_2025 = series_mask & get_metric_mask(metric_name, 2025)
            val_2025 = df[mask_2025]['order_number'].nunique()
            
            # Diff & Ratio
            diff = val_2025 - val_2024
            ratio = (diff / val_2024) if val_2024 > 0 else 0.0
            
            ratio_str = f"{ratio:.1%}"
            if ratio < 0:
                ratio_str = f"<span style='color: red'>{ratio_str}</span>"
            
            rows.append({
                "Series": s,
                "2024 全年": val_2024,
                "2025 (至今)": val_2025,
                "Diff": diff,
                "Ratio": ratio_str
            })
            
        # 2. 计算 Total (仅包含 target_series 的总和)
        total_2024 = sum(row["2024 全年"] for row in rows)
        total_2025 = sum(row["2025 (至今)"] for row in rows)
        total_diff = total_2025 - total_2024
        total_ratio = (total_diff / total_2024) if total_2024 > 0 else 0.0
        
        total_ratio_str = f"{total_ratio:.1%}"
        if total_ratio < 0:
            total_ratio_str = f"<span style='color: red'>{total_ratio_str}</span>"
            
        rows.append({
            "Series": "总计",
            "2024 全年": total_2024,
            "2025 (至今)": total_2025,
            "Diff": total_diff,
            "Ratio": total_ratio_str
        })
            
        series_details[metric_name] = pd.DataFrame(rows)
        
    metrics['series_details'] = series_details
    
    # 3. 分能源形式对比 (锁单数 & 留存锁单数)
    # 加载业务定义
    try:
        business_def = load_business_definition(BUSINESS_DEF_FILE)
        product_type_logic = business_def.get("product_type_logic", {})
    except Exception as e:
        print(f"⚠️ 加载业务定义失败: {e}")
        product_type_logic = {}
        
    energy_details = {}
    
    # 明确指定需要展示的 series 顺序
    target_series = ["L6", "LS6", "LS9"]
    target_energy_metrics = ["锁单数", "留存锁单数"]
    
    # 结构: energy_details[series][metric] = DataFrame
    
    for s in target_series:
        energy_details[s] = {}
        series_mask = (df['series'] == s)
        
        for metric_name in target_energy_metrics:
            rows = []
            
            for energy_type, condition_str in product_type_logic.items():
                # 获取该能源形式的 Mask
                energy_mask = parse_sql_condition(df, condition_str)
                
                # Series + Energy Mask
                combined_mask = series_mask & energy_mask
                
                # 2024
                mask_2024 = combined_mask & get_metric_mask(metric_name, 2024)
                val_2024 = df[mask_2024]['order_number'].nunique()
                
                # 2025
                mask_2025 = combined_mask & get_metric_mask(metric_name, 2025)
                val_2025 = df[mask_2025]['order_number'].nunique()
                
                # Diff & Ratio
                diff = val_2025 - val_2024
                ratio = (diff / val_2024) if val_2024 > 0 else 0.0
                
                ratio_str = f"{ratio:.1%}"
                if ratio < 0:
                    ratio_str = f"<span style='color: red'>{ratio_str}</span>"
                
                rows.append({
                    "能源形式": energy_type,
                    "2024 全年": val_2024,
                    "2025 (至今)": val_2025,
                    "Diff": diff,
                    "Ratio": ratio_str
                })
                
            # 2. 计算 Total (该 Series 下所有能源形式的总和)
            total_2024 = sum(row["2024 全年"] for row in rows)
            total_2025 = sum(row["2025 (至今)"] for row in rows)
            total_diff = total_2025 - total_2024
            total_ratio = (total_diff / total_2024) if total_2024 > 0 else 0.0
            
            total_ratio_str = f"{total_ratio:.1%}"
            if total_ratio < 0:
                total_ratio_str = f"<span style='color: red'>{total_ratio_str}</span>"
                
            rows.append({
                "能源形式": "总计",
                "2024 全年": total_2024,
                "2025 (至今)": total_2025,
                "Diff": total_diff,
                "Ratio": total_ratio_str
            })
            
            energy_details[s][metric_name] = pd.DataFrame(rows)
        
    metrics['energy_details'] = energy_details
    
    # 4. 退订分析 (Refund Analysis)
    # 确保时间列格式正确
    for col in ['approve_refund_time', 'lock_time']:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    # 需要计算的退订分类
    refund_categories = ["锁单退订总数", "前一年锁单退订", "当年锁单退订"]
    refund_stats = []
    
    # 预计算每年的数据
    year_data = {}
    years = [2024, 2025]
    
    for year in years:
        # 1. 筛选本周期产生退订申请的订单
        refund_mask = get_period_mask(df, "approve_refund_time", year)
        
        # 且必须是锁单 (lock_time 存在)
        if "lock_time" in df.columns:
            has_lock_time = df["lock_time"].notna()
        else:
            has_lock_time = pd.Series([False] * len(df), index=df.index)
            
        target_orders = df[refund_mask & has_lock_time].copy()
        
        total_refunds = target_orders['order_number'].nunique()
        
        # 2. 分类
        start_date = pd.Timestamp(f"{year}-01-01")
        
        # 前一年锁单退订: lock_time < start_date
        prior_mask = target_orders['lock_time'] < start_date
        prior_count = target_orders[prior_mask]['order_number'].nunique()
        
        # 当年锁单退订: lock_time >= start_date
        current_mask = target_orders['lock_time'] >= start_date
        current_count = target_orders[current_mask]['order_number'].nunique()
        
        year_data[year] = {
            "锁单退订总数": total_refunds,
            "前一年锁单退订": prior_count,
            "当年锁单退订": current_count
        }
    
    # 转置表格：行=分类，列=2024, 2025, Diff, Ratio
    for category in refund_categories:
        val_2024 = year_data[2024].get(category, 0)
        val_2025 = year_data[2025].get(category, 0)
        
        diff = val_2025 - val_2024
        ratio = (diff / val_2024) if val_2024 > 0 else 0.0
        
        ratio_str = f"{ratio:.1%}"
        if ratio < 0:
            ratio_str = f"<span style='color: red'>{ratio_str}</span>"
            
        refund_stats.append({
            "退订类型": category,
            "2024 全年": val_2024,
            "2025 (至今)": val_2025,
            "Diff": diff,
            "Ratio": ratio_str
        })
        
    metrics['refund_analysis'] = pd.DataFrame(refund_stats)
    
    # 5. 锁单-退订周期分布 (Refund Duration Distribution)
    duration_stats = []
    # Bins: 0, 7, 14, ..., 98, inf
    # range(0, 105, 7) -> 0, 7, ..., 98
    bins = list(range(0, 105, 7)) + [float('inf')]
    
    # Labels: 0-7, 7-14, ..., 98+
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-2)] + ["98+"]
    
    dist_data = []
    
    for year in [2024, 2025]:
        # 使用 approve_refund_time
        mask = get_period_mask(df, "approve_refund_time", year)
        
        if "lock_time" in df.columns:
             valid_lock = df["lock_time"].notna()
             subset = df[mask & valid_lock].copy()
             
             # Calculate duration
             subset['duration'] = (subset['approve_refund_time'] - subset['lock_time']).dt.days
             
             # Filter non-negative
             subset = subset[subset['duration'] >= 0]
             
             # Binning
             subset['bin'] = pd.cut(subset['duration'], bins=bins, labels=labels, right=False)
             
             counts = subset['bin'].value_counts().sort_index()
             
             # Convert to dict for easier usage
             # counts.index are strings (labels)
             dist_data.append({
                 "year": year,
                 "counts": counts
             })
             
    metrics['refund_duration_dist'] = dist_data
    
    # 6. 锁单未交付未退订分布 (Pending Delivery Distribution - 2025)
    # 过滤: lock_time >= 2025-01-01 (using 2025 logic), approve_refund_time is null, invoice_upload_time is null
    mask_2025_lock = get_period_mask(df, "lock_time", 2025)
    
    # Check for columns existence
    has_refund_col = "approve_refund_time" in df.columns
    has_invoice_col = "invoice_upload_time" in df.columns
    
    if has_refund_col and has_invoice_col:
        mask_no_refund = df["approve_refund_time"].isna()
        mask_no_invoice = df["invoice_upload_time"].isna()
        
        pending_mask = mask_2025_lock & mask_no_refund & mask_no_invoice
        pending_df = df[pending_mask].copy()
        
        # Calculate duration from lock_time to NOW
        now = pd.Timestamp.now()
        pending_df['duration'] = (now - pending_df['lock_time']).dt.days
        
        # Filter non-negative (sanity check)
        pending_df = pending_df[pending_df['duration'] >= 0]
        
        # Use same bins as refund analysis
        # Bins: 0, 7, 14, ..., 98, inf
        # bins and labels are already defined above
        
        pending_df['bin'] = pd.cut(pending_df['duration'], bins=bins, labels=labels, right=False)
        pending_counts = pending_df['bin'].value_counts().sort_index()
        
        metrics['pending_delivery_dist'] = {
            "total_count": pending_df['order_number'].nunique(),
            "counts": pending_counts,
            "data_timestamp": now
        }
        
    # 7. 锁单交付周期 (Lock-to-Delivery Cycle)
    # 过滤: lock_time is not null, invoice_upload_time is not null
    # 合并 2024 和 2025 数据，仅生成一个总体分布
    
    mask_period_2024 = get_period_mask(df, "invoice_upload_time", 2024)
    mask_period_2025 = get_period_mask(df, "invoice_upload_time", 2025)
    
    # Combined mask
    mask_period = mask_period_2024 | mask_period_2025
    
    if "lock_time" in df.columns:
         mask_valid_lock = df["lock_time"].notna()
         
         delivered_df = df[mask_period & mask_valid_lock].copy()
         
         # Calculate duration
         delivered_df['duration'] = (delivered_df['invoice_upload_time'] - delivered_df['lock_time']).dt.days
         
         # Filter non-negative
         delivered_df = delivered_df[delivered_df['duration'] >= 0]
         
         # Binning
         delivered_df['bin'] = pd.cut(delivered_df['duration'], bins=bins, labels=labels, right=False)
         
         counts = delivered_df['bin'].value_counts().sort_index()
         
         metrics['delivery_cycle_dist'] = {
             "total_count": delivered_df['order_number'].nunique(),
             "counts": counts
         }

    # 8. 交付分析概览 (Delivery Overview: 30-day & 98-day rates)
    # 对比 2024 和 2025
    delivery_overview = []
    
    for year in [2024, 2025]:
        # 定义基准: 该年份产生的锁单
        base_mask = get_period_mask(df, "lock_time", year)
        
        base_orders = df[base_mask].copy()
        total_locks = base_orders['order_number'].nunique()
        
        # 计算 duration (invoice_upload_time - lock_time)
        if "invoice_upload_time" in base_orders.columns:
            # 确保 invoice_upload_time 是 datetime
            # Note: We already did conversion at start of func but copies might need check
            pass
            
        # 标记是否交付 (invoice_upload_time 存在)
        has_invoice = base_orders['invoice_upload_time'].notna()
        
        # 计算交付时长 (仅对已交付的计算，未交付的设为 NaT/NaN)
        # Note: We need to handle NaT carefully in subtraction
        
        # Create a duration series, default infinite or NaN
        # Only calculate where invoice exists
        durations = (base_orders.loc[has_invoice, 'invoice_upload_time'] - base_orders.loc[has_invoice, 'lock_time']).dt.days
        
        # 30日交付数: duration <= 30
        count_30d = (durations <= 30).sum()
        
        # 98日交付数: duration <= 98
        count_98d = (durations <= 98).sum()
        
        # Calculate Rates
        rate_30d = count_30d / total_locks if total_locks > 0 else 0.0
        rate_98d = count_98d / total_locks if total_locks > 0 else 0.0
        
        delivery_overview.append({
            "year": year,
            "total_locks": total_locks,
            "count_30d": count_30d,
            "rate_30d": rate_30d,
            "count_98d": count_98d,
            "rate_98d": rate_98d
        })
        
    metrics['delivery_overview'] = delivery_overview
    
    # 9. 锁单交付率趋势 (Delivery Rate Trend)
    # 按天统计 (Daily)
    trend_data = {}
    
    for year in [2024, 2025]:
        mask = get_period_mask(df, "lock_time", year)
        df_year = df[mask].copy()
        
        if df_year.empty:
            continue
            
        # Set index to lock_time for resampling
        df_year = df_year.set_index('lock_time').sort_index()
        
        # Calculate duration for all orders in this year
        if "invoice_upload_time" in df_year.columns:
            # Duration in days
            # Fill NaT with NaNs, calculations handles NaNs automatically
            durations = (df_year['invoice_upload_time'] - df_year.index).dt.days
        else:
            durations = pd.Series([np.nan] * len(df_year), index=df_year.index)
            
        # Resample Daily
        # 1. Total Locks
        daily_total = df_year['order_number'].resample('D').nunique()
        
        # 2. 30d Deliveries
        is_30d = (durations <= 30)
        daily_30d = is_30d.resample('D').sum()
        
        # 3. 98d Deliveries
        is_98d = (durations <= 98)
        daily_98d = is_98d.resample('D').sum()
        
        # Combine
        trend_df = pd.DataFrame({
            'total': daily_total,
            'count_30d': daily_30d,
            'count_98d': daily_98d
        })
        
        # Calculate Rates
        # Avoid division by zero
        trend_df['rate_30d'] = trend_df.apply(lambda row: row['count_30d'] / row['total'] if row['total'] > 0 else 0.0, axis=1)
        trend_df['rate_98d'] = trend_df.apply(lambda row: row['count_98d'] / row['total'] if row['total'] > 0 else 0.0, axis=1)
        
        trend_data[year] = trend_df
        
    metrics['delivery_trend'] = trend_data

    # 10. 在营门店数 (Active Store Count) - Module 4
    try:
        # Prepare data
        df_store = df.copy()
        
        # Priority: order_create_date > order_create_time
        # Use order_create_date if available and valid
        if 'order_create_date' in df_store.columns:
             df_store['order_create_date'] = pd.to_datetime(df_store['order_create_date'], errors='coerce')
             df_store['date'] = df_store['order_create_date']
        
        # Fallback to order_create_time if date is missing
        if 'order_create_time' in df_store.columns:
             df_store['order_create_time'] = pd.to_datetime(df_store['order_create_time'], errors='coerce')
             if 'date' not in df_store.columns:
                 df_store['date'] = df_store['order_create_time'].dt.floor('D')
             else:
                 # Fill NaT in date with time
                 df_store['date'] = df_store['date'].fillna(df_store['order_create_time'].dt.floor('D'))

        df_store['store_create_date'] = pd.to_datetime(df_store['store_create_date'], errors='coerce')
        
        # Valid records only (must have store_name and a valid date)
        df_store = df_store.dropna(subset=['store_name', 'date'])
        
        if not df_store.empty:
            # 1. Store Opening Dates (Min per store)
            open_map = df_store.groupby('store_name')['store_create_date'].min()
            
            # 2. Daily Orders per Store
            # Date is already prepared above
            daily_counts = df_store.groupby(['date', 'store_name']).size().unstack(fill_value=0)
            
            # Full date range
            min_date = df_store['date'].min()
            max_date = df_store['date'].max()
            full_days = pd.date_range(min_date, max_date, freq='D')
            
            # Reindex daily counts
            daily_counts = daily_counts.reindex(full_days, fill_value=0)
            
            # 3. Rolling Activity (30 days)
            rolling_activity = daily_counts.rolling(window=30, min_periods=1).sum()
            
            # 4. Calculate Active Count
            active_counts = []
            for d in full_days:
                if d not in rolling_activity.index:
                    active_counts.append(0)
                    continue
                    
                # Stores with activity > 0
                activity_mask = rolling_activity.loc[d] > 0
                current_stores = activity_mask.index
                
                # Check opening date
                store_open_dates = open_map.reindex(current_stores)
                is_open = (store_open_dates <= d)
                
                # Active = Active Activity & Open
                is_active_store = activity_mask & is_open
                active_counts.append(is_active_store.sum())
                
            metrics['active_store_series'] = pd.Series(active_counts, index=full_days)
    except Exception as e:
        print(f"Error calculating active stores: {e}")

    # 11. Daily Lock Counts (for Module 4.4)
    try:
        daily_locks = df.groupby(df['lock_time'].dt.floor('D')).size()
        metrics['daily_lock_counts'] = daily_locks
    except Exception as e:
        print(f"Error calculating daily lock counts: {e}")

    # 12. Daily Lock Counts by Series (for Module 4.4 Breakdown)
    try:
        # Group by Date and Series, unstack to get columns as series names
        daily_locks_series = df.groupby([df['lock_time'].dt.floor('D'), 'series']).size().unstack(fill_value=0)
        metrics['daily_locks_series'] = daily_locks_series
    except Exception as e:
        print(f"Error calculating daily lock counts by series: {e}")

    # 13. Daily Total Invoice Amount (for Module 4.5)
    try:
        # Ensure invoice_upload_time is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['invoice_upload_time']):
             df['invoice_upload_time'] = pd.to_datetime(df['invoice_upload_time'], errors='coerce')
        
        # Ensure invoice_amount is numeric
        df['invoice_amount'] = pd.to_numeric(df['invoice_amount'], errors='coerce')
        
        # Filter for valid dates and sum amount
        daily_invoice_sum = df.groupby(df['invoice_upload_time'].dt.floor('D'))['invoice_amount'].sum()
        metrics['daily_invoice_sum'] = daily_invoice_sum
    except Exception as e:
        print(f"Error calculating daily invoice sum: {e}")

    # 14. Daily Invoice Price Trend (for Module 1.4)
    try:
        # Filter delivered orders
        df_delivered = df[df['delivery_date'].notna()].copy()
        
        # Ensure types (already done above if reused, but safe to check)
        if not pd.api.types.is_datetime64_any_dtype(df_delivered['invoice_upload_time']):
             df_delivered['invoice_upload_time'] = pd.to_datetime(df_delivered['invoice_upload_time'], errors='coerce')
        
        df_delivered['invoice_amount'] = pd.to_numeric(df_delivered['invoice_amount'], errors='coerce')
        
        # Filter valid invoice time and amount
        df_delivered = df_delivered.dropna(subset=['invoice_upload_time', 'invoice_amount'])
        
        # Group by Date: Average Invoice Price
        daily_price = df_delivered.groupby(df_delivered['invoice_upload_time'].dt.floor('D'))['invoice_amount'].mean()
        metrics['daily_invoice_price'] = daily_price
        
        # Group by Date and Series: Average Invoice Price
        daily_price_series = df_delivered.groupby([df_delivered['invoice_upload_time'].dt.floor('D'), 'series'])['invoice_amount'].mean().unstack()
        metrics['daily_invoice_price_series'] = daily_price_series
        
    except Exception as e:
        print(f"Error calculating daily invoice price: {e}")

    # 15. Module 5: Lead Conversion Rate (线索转化率)
    try:
        assign_file = Path("original/assign_data.csv")
        if assign_file.exists():
            # Try loading with utf-16 (from check script)
            try:
                df_assign = pd.read_csv(assign_file, sep='\t', encoding='utf-16')
            except:
                # Fallback to gbk or utf-8 if needed
                try:
                    df_assign = pd.read_csv(assign_file, sep='\t', encoding='gbk')
                except:
                    df_assign = pd.read_csv(assign_file, sep='\t', encoding='utf-8')

            # Ensure columns exist
            required_cols = ['Assign Time 年/月/日', '下发线索数', '下发线索当日试驾数', 
                             '下发线索 7 日试驾数', '下发线索 7 日锁单数', 
                             '下发线索 30日试驾数', '下发线索 30 日锁单数', '下发门店数']
            
            if all(col in df_assign.columns for col in required_cols):
                # Parse date
                df_assign['date'] = pd.to_datetime(df_assign['Assign Time 年/月/日'], format='%Y年%m月%d日', errors='coerce')
                df_assign = df_assign.dropna(subset=['date'])
                
                module5_stats = {}
                
                for year in [2024, 2025]:
                    df_year = df_assign[df_assign['date'].dt.year == year].copy()
                    
                    if df_year.empty:
                        # Ensure structure exists even if empty
                        module5_stats[year] = None
                        continue
                        
                    stats = {}
                    
                    # 1. Total Leads
                    total_leads = df_year['下发线索数'].sum()
                    stats['total_leads'] = total_leads
                    
                    # Helper for rates
                    def calc_rate(col_num, col_denom):
                        sum_num = df_year[col_num].sum()
                        sum_denom = df_year[col_denom].sum()
                        return sum_num / sum_denom if sum_denom > 0 else 0.0
                    
                    # 2. Rates
                    stats['rate_same_day_test_drive'] = calc_rate('下发线索当日试驾数', '下发线索数')
                    stats['rate_7d_test_drive'] = calc_rate('下发线索 7 日试驾数', '下发线索数')
                    stats['rate_7d_lock'] = calc_rate('下发线索 7 日锁单数', '下发线索数')
                    stats['rate_30d_test_drive'] = calc_rate('下发线索 30日试驾数', '下发线索数')
                    stats['rate_30d_lock'] = calc_rate('下发线索 30 日锁单数', '下发线索数')
                    
                    # 3. Store Avg Daily Leads
                    # Calculate daily leads/store then mean
                    # Handle 0 stores
                    with np.errstate(divide='ignore', invalid='ignore'):
                        df_year['daily_leads_per_store'] = df_year['下发线索数'] / df_year['下发门店数']
                        df_year['daily_leads_per_store'] = df_year['daily_leads_per_store'].replace([np.inf, -np.inf], np.nan)
                        
                    stats['avg_daily_leads_per_store'] = df_year['daily_leads_per_store'].mean()
                    
                    module5_stats[year] = stats
                    
                metrics['module5_stats'] = module5_stats
                
                # Save daily series for Module 5.1
                # Keep relevant columns
                cols_to_keep = ['date', '下发线索数', '下发线索 30日试驾数', '下发线索 30 日锁单数']
                df_daily_leads = df_assign[cols_to_keep].copy()
                df_daily_leads = df_daily_leads.rename(columns={
                    '下发线索数': 'leads_count',
                    '下发线索 30日试驾数': 'test_drive_30d',
                    '下发线索 30 日锁单数': 'lock_30d'
                })
                # Ensure date is index for easier plotting
                df_daily_leads = df_daily_leads.set_index('date').sort_index()
                metrics['module5_daily_series'] = df_daily_leads
                
    except Exception as e:
        print(f"Error calculating Module 5 stats: {e}")

    # 16. Module 6: Test Drive Analysis (试驾分析)
    try:
        td_file = Path("original/test_drive_data.csv")
        if td_file.exists():
            # Try loading with utf-16 (confirmed by check script)
            try:
                df_td = pd.read_csv(td_file, sep='\t', encoding='utf-16')
            except:
                try:
                    df_td = pd.read_csv(td_file, sep='\t', encoding='gbk')
                except:
                    df_td = pd.read_csv(td_file, sep='\t', encoding='utf-8')
                    
            # Check columns
            required_cols = ['create_date 年/月/日', '有效试驾数', 'L6有效试驾数', 'LS6有效试驾数', 'LS9有效试驾数', '试驾门店数']
            if all(col in df_td.columns for col in required_cols):
                # Parse date
                df_td['date'] = pd.to_datetime(df_td['create_date 年/月/日'], format='%Y年%m月%d日', errors='coerce')
                df_td = df_td.dropna(subset=['date'])
                
                module6_stats = {}
                
                for year in [2024, 2025]:
                    df_year = df_td[df_td['date'].dt.year == year].copy()
                    
                    if df_year.empty:
                        module6_stats[year] = None
                        continue
                        
                    stats = {}
                    
                    # 1. Total Valid Test Drives
                    stats['total_valid_td'] = df_year['有效试驾数'].sum()
                    stats['total_L6_td'] = df_year['L6有效试驾数'].sum()
                    stats['total_LS6_td'] = df_year['LS6有效试驾数'].sum()
                    stats['total_LS9_td'] = df_year['LS9有效试驾数'].sum()
                    
                    # 2. Store Daily Avg Test Drives
                    # Mean of (Daily Valid Test Drives / Daily Stores)
                    # Modified: Use 'active_store_series' (Module 4.3) as denominator if available
                    
                    if 'active_store_series' in metrics:
                         # Map dates to active store counts
                         df_year['active_store_count'] = df_year['date'].map(metrics['active_store_series'])
                         denominator = df_year['active_store_count']
                    else:
                         denominator = df_year['试驾门店数']
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        df_year['daily_avg_per_store'] = df_year['有效试驾数'] / denominator
                        df_year['daily_avg_L6'] = df_year['L6有效试驾数'] / denominator
                        df_year['daily_avg_LS6'] = df_year['LS6有效试驾数'] / denominator
                        
                        df_year['daily_avg_per_store'] = df_year['daily_avg_per_store'].replace([np.inf, -np.inf], np.nan)
                        df_year['daily_avg_L6'] = df_year['daily_avg_L6'].replace([np.inf, -np.inf], np.nan)
                        df_year['daily_avg_LS6'] = df_year['daily_avg_LS6'].replace([np.inf, -np.inf], np.nan)
                        
                    stats['store_daily_avg'] = df_year['daily_avg_per_store'].mean()
                    
                    # Store series data for Module 6.1
                    series_data = df_year[['date', 'daily_avg_per_store', 'daily_avg_L6', 'daily_avg_LS6']].copy()
                    series_data = series_data.set_index('date').sort_index()
                    stats['daily_series'] = series_data
                    
                    module6_stats[year] = stats
                    
                metrics['module6_stats'] = module6_stats
                
    except Exception as e:
        print(f"Error calculating Module 6 stats: {e}")

    # ==========================================
    # 17. 用户画像 (User Profile) - Module 7.1
    # ==========================================
    try:
        # 7.1 Age Structure
        # Filter valid age (18-100) and Lock Orders
        if 'age' in df.columns:
            valid_age_mask = (df['age'] >= 18) & (df['age'] <= 100)
            
            # Helper to calculate mean age
            def calc_mean_age(mask):
                valid_data = df[mask & valid_age_mask]
                if valid_data.empty:
                    return 0.0
                return valid_data['age'].mean()

            # 7.1.1 By Series
            age_series_stats = []
            target_series = ["L6", "LS6", "LS9"]
            
            for s in target_series:
                series_mask = df['series'] == s
                
                # 2024
                mask_2024 = series_mask & get_metric_mask("留存锁单数", 2024)
                mean_2024 = calc_mean_age(mask_2024)
                
                # 2025
                mask_2025 = series_mask & get_metric_mask("留存锁单数", 2025)
                mean_2025 = calc_mean_age(mask_2025)
                
                # Diff
                diff = mean_2025 - mean_2024
                diff_str = f"{diff:+.1f}"
                if diff < 0:
                    diff_str = f"<span style='color: red'>{diff_str}</span>"
                
                age_series_stats.append({
                    "Series": s,
                    "2024 Avg Age": mean_2024,
                    "2025 Avg Age": mean_2025,
                    "Diff": diff_str
                })
            
            # Total
            mask_total_2024 = get_metric_mask("锁单数", 2024)
            mean_total_2024 = calc_mean_age(mask_total_2024)
            
            mask_total_2025 = get_metric_mask("锁单数", 2025)
            mean_total_2025 = calc_mean_age(mask_total_2025)
            
            diff_total = mean_total_2025 - mean_total_2024
            diff_total_str = f"{diff_total:+.1f}"
            if diff_total < 0:
                diff_total_str = f"<span style='color: red'>{diff_total_str}</span>"
            
            age_series_stats.append({
                "Series": "总计",
                "2024 Avg Age": mean_total_2024,
                "2025 Avg Age": mean_total_2025,
                "Diff": diff_total_str
            })
            
            metrics['age_series_stats'] = pd.DataFrame(age_series_stats)
            
            # 7.1.2 By Parent Region
            if 'parent_region_name' in df.columns:
                age_region_stats = []
                # Get unique regions, sorted
                regions = sorted(df['parent_region_name'].dropna().unique())
                
                for region in regions:
                    region_mask = df['parent_region_name'] == region
                    
                    # 2024
                    mask_2024 = region_mask & get_metric_mask("锁单数", 2024)
                    mean_2024 = calc_mean_age(mask_2024)
                    
                    # 2025
                    mask_2025 = region_mask & get_metric_mask("锁单数", 2025)
                    mean_2025 = calc_mean_age(mask_2025)
                    
                    # Diff
                    diff = mean_2025 - mean_2024
                    diff_str = f"{diff:+.1f}"
                    if diff < 0:
                        diff_str = f"<span style='color: red'>{diff_str}</span>"
                    
                    age_region_stats.append({
                        "Region": region,
                        "2024 Avg Age": mean_2024,
                        "2025 Avg Age": mean_2025,
                        "Diff": diff_str
                    })
                    
                metrics['age_region_stats'] = pd.DataFrame(age_region_stats)

            # 7.2 Age Trends (Daily Mean Age)
            age_trends = {}
            for s in target_series:
                age_trends[s] = {}
                series_mask = df['series'] == s
                
                for year in [2024, 2025]:
                    # Strict Retained Logic
                    mask = series_mask & get_metric_mask("留存锁单数", year)
                    
                    # Also need valid age
                    valid_age_mask = (df['age'] >= 18) & (df['age'] <= 100)
                    final_mask = mask & valid_age_mask
                    
                    data = df[final_mask].copy()
                    if data.empty:
                        age_trends[s][year] = pd.Series()
                        continue
                        
                    # Group by Day of Year
                    # Use 'lock_time' as date source
                    if not pd.api.types.is_datetime64_any_dtype(data['lock_time']):
                        data['lock_time'] = pd.to_datetime(data['lock_time'], errors='coerce')
                    
                    data['doy'] = data['lock_time'].dt.dayofyear
                    
                    # Calculate mean age per day
                    daily_avg = data.groupby('doy')['age'].mean()
                    
                    age_trends[s][year] = daily_avg
            
            metrics['age_trends'] = age_trends

            # 7.3 Gender Structure (Retained Lock Orders)
            # Use 'gender' field.
            # Calculate counts for 2024 vs 2025.
            # Output:
            # 1. By Series (LS6, L6, LS9)
            # 2. By Region (parent_region_name)
            
            gender_stats = {'series': {}, 'region': {}}
            
            # Helper to standardize gender and filter Unknown
            def standardize_gender(g):
                if pd.isna(g): return None
                if g in ['男', 'Male']: return "男"
                if g in ['女', 'Female']: return "女"
                return None

            # 7.3.1 By Series
            for s in target_series:
                gender_stats['series'][s] = {}
                series_mask = df['series'] == s
                
                rows = []
                
                # Pre-calculate masks for 2024 and 2025
                mask_2024 = series_mask & get_metric_mask("留存锁单数", 2024)
                mask_2025 = series_mask & get_metric_mask("留存锁单数", 2025)
                
                data_2024 = df[mask_2024].copy()
                data_2025 = df[mask_2025].copy()
                
                data_2024['std_gender'] = data_2024['gender'].apply(standardize_gender)
                data_2025['std_gender'] = data_2025['gender'].apply(standardize_gender)
                
                # Filter out Unknown/None
                data_2024 = data_2024.dropna(subset=['std_gender'])
                data_2025 = data_2025.dropna(subset=['std_gender'])
                
                counts_2024 = data_2024['std_gender'].value_counts()
                counts_2025 = data_2025['std_gender'].value_counts()
                
                total_2024 = counts_2024.sum()
                total_2025 = counts_2025.sum()
                
                for g_label in ['男', '女']:
                    c_2024 = counts_2024.get(g_label, 0)
                    c_2025 = counts_2025.get(g_label, 0)
                    
                    share_2024 = (c_2024 / total_2024) if total_2024 > 0 else 0.0
                    share_2025 = (c_2025 / total_2025) if total_2025 > 0 else 0.0
                    
                    diff_share = share_2025 - share_2024
                    diff_share_str = f"{diff_share:.1%}"
                    if diff_share > 0:
                        diff_share_str = f"+{diff_share_str}"
                    elif diff_share < 0:
                        diff_share_str = f"<span style='color: red'>{diff_share_str}</span>"
                    
                    rows.append({
                        "Gender": g_label,
                        "2024 Count": c_2024,
                        "2024 Share": f"{share_2024:.1%}",
                        "2025 Count": c_2025,
                        "2025 Share": f"{share_2025:.1%}",
                        "Share Diff": diff_share_str
                    })
                
                gender_stats['series'][s] = pd.DataFrame(rows)
            
            # 7.3.2 By Region (parent_region_name)
            # Identify regions
            if 'parent_region_name' in df.columns:
                regions = df['parent_region_name'].dropna().unique()
                # Sort regions to ensure consistent order
                regions = sorted(regions)
                
                for region in regions:
                    # Get Region Mask
                    region_mask = df['parent_region_name'] == region
                    
                    rows = []
                    
                    mask_2024 = region_mask & get_metric_mask("留存锁单数", 2024)
                    mask_2025 = region_mask & get_metric_mask("留存锁单数", 2025)
                    
                    data_2024 = df[mask_2024].copy()
                    data_2025 = df[mask_2025].copy()
                    
                    data_2024['std_gender'] = data_2024['gender'].apply(standardize_gender)
                    data_2025['std_gender'] = data_2025['gender'].apply(standardize_gender)
                    
                    # Filter out Unknown/None
                    data_2024 = data_2024.dropna(subset=['std_gender'])
                    data_2025 = data_2025.dropna(subset=['std_gender'])
                    
                    counts_2024 = data_2024['std_gender'].value_counts()
                    counts_2025 = data_2025['std_gender'].value_counts()
                    
                    total_2024 = counts_2024.sum()
                    total_2025 = counts_2025.sum()
                    
                    # Skip empty regions if no data at all (optional, but cleaner)
                    if total_2024 == 0 and total_2025 == 0:
                        continue
                    
                    for g_label in ['男', '女']:
                        c_2024 = counts_2024.get(g_label, 0)
                        c_2025 = counts_2025.get(g_label, 0)
                        
                        share_2024 = (c_2024 / total_2024) if total_2024 > 0 else 0.0
                        share_2025 = (c_2025 / total_2025) if total_2025 > 0 else 0.0
                        
                        diff_share = share_2025 - share_2024
                        diff_share_str = f"{diff_share:.1%}"
                        if diff_share > 0:
                            diff_share_str = f"+{diff_share_str}"
                        elif diff_share < 0:
                            diff_share_str = f"<span style='color: red'>{diff_share_str}</span>"
                        
                        rows.append({
                            "Gender": g_label,
                            "2024 Count": c_2024,
                            "2024 Share": f"{share_2024:.1%}",
                            "2025 Count": c_2025,
                            "2025 Share": f"{share_2025:.1%}",
                            "Share Diff": diff_share_str
                        })
                        
                    gender_stats['region'][region] = pd.DataFrame(rows)
            
            metrics['gender_stats'] = gender_stats

    except Exception as e:
        print(f"Error calculating Module 7 stats: {e}")

    # 4.3.2 Store Tenure Analysis
    try:
        metrics['store_tenure_analysis'] = calculate_store_tenure_metrics(df)
    except Exception as e:
        print(f"Error calculating Store Tenure stats: {e}")

    return metrics

def calculate_conversion_probability(metrics):
    """
    根据历史(2024+2025)的退订和交付分布，计算各时长区间的'生存转化率'。
    
    Model:
    For an order pending at bin i:
    Prob(Deliver | Age >= i) = Sum(Deliveries[j] for j >= i) / Sum(Deliveries[j] + Refunds[j] for j >= i)
    
    Assumption:
    Pending orders follow the same conditional outcome distribution as historical closed orders.
    """
    if 'refund_duration_dist' not in metrics or 'delivery_cycle_dist' not in metrics:
        return None

    # 1. Aggregate Historical Data (2024 + 2025)
    # Note: Module 2.1 data structure is a list of dicts [{'year': 2024, 'counts': ...}, ...]
    # Module 2.3 data structure is now a dict {'total_count': ..., 'counts': ...} (merged)
    
    # Initialize total series with 0
    # Use the labels from the first available series to ensure alignment
    ref_dist = metrics['refund_duration_dist']
    del_dist = metrics['delivery_cycle_dist']
    
    # Get labels from pending_delivery_dist to ensure we match the target
    if 'pending_delivery_dist' not in metrics:
        return None
        
    pending_counts = metrics['pending_delivery_dist']['counts']
    labels = pending_counts.index
    
    # Combine Refund Counts
    total_refunds = pd.Series(0, index=labels)
    for item in ref_dist:
        # Align series to labels (fill 0 for missing)
        counts = item['counts'].reindex(labels, fill_value=0)
        total_refunds = total_refunds + counts
        
    # Combine Delivery Counts
    total_deliveries = del_dist['counts'].reindex(labels, fill_value=0)
    
    # 2. Calculate Reverse Cumulative Sums (Events occurring at >= i)
    # We iterate backwards
    
    # Convert to numeric for calculation
    R = total_refunds.values
    D = total_deliveries.values
    
    # Reverse CumSum (Total Future Events from index i onwards)
    # Using [::-1] to reverse, cumsum, then reverse back
    future_R = np.cumsum(R[::-1])[::-1]
    future_D = np.cumsum(D[::-1])[::-1]
    
    total_future_events = future_R + future_D
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        conversion_prob = future_D / total_future_events
        # If total_future_events is 0, probability is 0 (or undefined, treat as 0)
        conversion_prob = np.nan_to_num(conversion_prob, nan=0.0)
        
    return pd.Series(conversion_prob, index=labels)

def get_common_layout(title: str, xaxis_title: str = None, yaxis_title: str = None):
    """获取统一的 Plotly Layout 配置"""
    layout = dict(
        title=title,
        template="plotly_white",
        plot_bgcolor='#FFFFFF',
        hovermode="x unified",
        xaxis=dict(
            title=xaxis_title,
            gridcolor='#ebedf0',
            zerolinecolor='#ebedf0',
            tickfont=dict(color='#7B848F'),
            title_font=dict(color='#7B848F'),
            showgrid=True
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor='#ebedf0',
            zerolinecolor='#ebedf0',
            tickfont=dict(color='#7B848F'),
            title_font=dict(color='#7B848F'),
            showgrid=True
        ),
        legend=dict(
            bordercolor='#7B848F',
            font=dict(color='#7B848F')
        )
    )
    return layout

def generate_html(metrics: dict, output_file: Path):
    """生成 HTML 报告"""
    
    # CSS 样式
    css = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; color: #333; }
        h1 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; border-left: 5px solid #3498db; padding-left: 10px; }
        h3 { color: #2980b9; margin-top: 25px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        th, td { text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: 600; color: #555; }
        tr:hover { background-color: #f5f5f5; }
        .timestamp { color: #888; font-size: 0.9em; margin-bottom: 20px; }
        .summary-box { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    </style>
    """
    
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>2024 vs 2025 业务指标对比分析</title>",
        css,
        "</head>",
        "<body>",
        "<h1>2024 vs 2025 业务指标对比分析</h1>",
        f"<div class='timestamp'>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>"
    ]
    
    # 1. 指标概览
    html_content.append("<h2>1. 指标概览 (Overview)</h2>")
    html_content.append("<div class='summary-box'>")
    html_content.append("<p>统计周期说明：</p>")
    html_content.append("<ul>")
    html_content.append("<li>2024: 2024-01-01 ~ 2024-12-31</li>")
    html_content.append("<li>2025: 2025-01-01 ~ 至今 (Max Date)</li>")
    html_content.append("</ul>")
    html_content.append("</div>")
    
    df_overall = metrics['overall']
    html_content.append(df_overall.to_html(index=False, classes='table', escape=False, float_format=lambda x: '{:,.0f}'.format(x) if isinstance(x, (int, float)) else x))
    
    # 1.1 指标概览（分车型）
    html_content.append("<h2>1.1 指标概览 - 分车型 (By Series)</h2>")
    
    series_details = metrics['series_details']
    # 按特定顺序展示：锁单 -> 开票 -> 退订 -> 留存锁单
    display_order = ["锁单数", "开票数", "锁单退订数", "留存锁单数"]
    
    for metric_name in display_order:
        if metric_name in series_details:
            df_table = series_details[metric_name]
            html_content.append(f"<h3>{metric_name}</h3>")
            html_content.append(df_table.to_html(index=False, classes='table', escape=False, float_format=lambda x: '{:,.0f}'.format(x) if isinstance(x, (int, float)) else x))
            
    # 1.2 指标概览（分能源形式）
    html_content.append("<h2>1.2 指标概览 - 分能源形式 (By Energy Type)</h2>")
    
    energy_details = metrics.get('energy_details', {})
    # 按照 target_series 顺序展示
    target_series_order = ["L6", "LS6", "LS9"]
    target_energy_metrics = ["锁单数", "留存锁单数"]
    
    for s in target_series_order:
        if s in energy_details:
            html_content.append(f"<h3>{s}</h3>")
            series_data = energy_details[s]
            
            for metric_name in target_energy_metrics:
                if metric_name in series_data:
                    df_table = series_data[metric_name]
                    html_content.append(f"<h4>{metric_name}</h4>")
                    html_content.append(df_table.to_html(index=False, classes='table', escape=False, float_format=lambda x: '{:,.0f}'.format(x) if isinstance(x, (int, float)) else x))
    
    # 1.3 锁单趋势分析 (Lock Order Trends)
    html_content.append("<h2>1.3 锁单趋势分析 (Lock Order Trends)</h2>")
    html_content.append("<p>X轴: Lock Time (Day of Year), Y轴: 锁单数 (MA7 Smoothed)</p>")
    html_content.append("<p>注：数据已进行 7天移动平均 (MA7) 平滑处理。</p>")

    # 1.3.0 Summary
    if 'daily_lock_counts' in metrics:
        html_content.append("<h3>1.3.0 整体锁单趋势 (Overall Lock Trends - MA7)</h3>")
        daily_locks = metrics['daily_lock_counts']
        
        fig = go.Figure()
        for year in [2024, 2025]:
            # Filter
            data_year = daily_locks[daily_locks.index.year == year]
            if data_year.empty: continue
            
            # Apply MA7 Smoothing
            # Ensure full date range for correct rolling
            min_date = data_year.index.min()
            max_date = data_year.index.max()
            full_idx = pd.date_range(min_date, max_date, freq='D')
            data_year = data_year.reindex(full_idx, fill_value=0)
            
            # Calculate MA7
            ma7_data = data_year.rolling(window=7, min_periods=1).mean()
            
            # X = Day of Year
            x_days = ma7_data.index.dayofyear
            dates_str = ma7_data.index.strftime('%Y-%m-%d')
            color = '#3498DB' if year == 2024 else '#E67E22'
            
            fig.add_trace(go.Scatter(
                x=x_days,
                y=ma7_data.values,
                mode='lines',
                name=f'{year} (MA7)',
                line=dict(color=color, width=2),
                hovertemplate="Day %{x} (%{customdata})<br>MA7 Locks: %{y:.1f}<extra></extra>",
                customdata=dates_str
            ))
            
        layout = get_common_layout(
            title="整体锁单趋势对比 (Overall Daily Lock Counts - MA7)",
            xaxis_title="年份天数 (Day of Year)",
            yaxis_title="锁单数 (MA7)"
        )
        layout['xaxis']['range'] = [1, 366]
        fig.update_layout(layout)
        chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        html_content.append(chart_html)

    # 1.3.1 - 1.3.3 Series Breakdown
    if 'daily_locks_series' in metrics:
        df_locks_series = metrics['daily_locks_series']
        target_series = ['LS6', 'L6', 'LS9']
        
        for ser_name in target_series:
            if ser_name not in df_locks_series.columns: continue
            
            html_content.append(f"<h3>1.3.{target_series.index(ser_name)+1} {ser_name} 锁单趋势 (MA7)</h3>")
            s_locks_ser = df_locks_series[ser_name]
            
            fig = go.Figure()
            for year in [2024, 2025]:
                data_year = s_locks_ser[s_locks_ser.index.year == year]
                if data_year.empty: continue
                
                # Apply MA7 Smoothing
                min_date = data_year.index.min()
                max_date = data_year.index.max()
                full_idx = pd.date_range(min_date, max_date, freq='D')
                data_year = data_year.reindex(full_idx, fill_value=0)
                
                ma7_data = data_year.rolling(window=7, min_periods=1).mean()
                
                x_days = ma7_data.index.dayofyear
                dates_str = ma7_data.index.strftime('%Y-%m-%d')
                color = '#3498DB' if year == 2024 else '#E67E22'
                
                fig.add_trace(go.Scatter(
                    x=x_days,
                    y=ma7_data.values,
                    mode='lines',
                    name=f'{year} (MA7)',
                    line=dict(color=color, width=2),
                    hovertemplate="Day %{x} (%{customdata})<br>MA7 Locks: %{y:.1f}<extra></extra>",
                    customdata=dates_str
                ))
                
            layout = get_common_layout(
                title=f"{ser_name} 锁单趋势对比 (Daily Lock Counts - MA7)",
                xaxis_title="年份天数 (Day of Year)",
                yaxis_title="锁单数 (MA7)"
            )
            layout['xaxis']['range'] = [1, 366]
            fig.update_layout(layout)
            chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
            html_content.append(chart_html)

    # 1.4 开票价格趋势 (Invoice Price Trends)
    html_content.append("<h2>1.4 开票价格趋势 (Invoice Price Trends)</h2>")
    html_content.append("<p>X轴: Invoice Upload Time (Day of Year), Y轴: 平均开票价格 (MA7 Smoothed)</p>")
    html_content.append("<p>筛选条件: 含有 delivery_date 的已交付订单。</p>")
    html_content.append("<p>注：数据已进行 7天移动平均 (MA7) 平滑处理。</p>")

    # 1.4.0 Summary
    if 'daily_invoice_price' in metrics:
        html_content.append("<h3>1.4.0 整体开票价格趋势 (Overall Invoice Price Trends - MA7)</h3>")
        daily_price = metrics['daily_invoice_price']
        
        fig = go.Figure()
        for year in [2024, 2025]:
            # Filter
            data_year = daily_price[daily_price.index.year == year]
            if data_year.empty: continue
            
            # Apply MA7 Smoothing
            # For price, we reindex to full daily range but fill with NaN (not 0)
            # rolling().mean() will skip NaNs but provide smoothing over available data
            min_date = data_year.index.min()
            max_date = data_year.index.max()
            full_idx = pd.date_range(min_date, max_date, freq='D')
            data_year = data_year.reindex(full_idx) # Default fill_value is NaN
            
            # Calculate MA7
            ma7_data = data_year.rolling(window=7, min_periods=1).mean()
            
            # X = Day of Year
            x_days = ma7_data.index.dayofyear
            dates_str = ma7_data.index.strftime('%Y-%m-%d')
            color = '#3498DB' if year == 2024 else '#E67E22'
            
            fig.add_trace(go.Scatter(
                x=x_days,
                y=ma7_data.values,
                mode='lines',
                name=f'{year} (MA7)',
                line=dict(color=color, width=2),
                hovertemplate="Day %{x} (%{customdata})<br>MA7 Price: %{y:,.0f}<extra></extra>",
                customdata=dates_str
            ))
            
        layout = get_common_layout(
            title="整体开票价格趋势对比 (Overall Daily Average Invoice Price - MA7)",
            xaxis_title="年份天数 (Day of Year)",
            yaxis_title="平均开票价格 (CNY)"
        )
        layout['xaxis']['range'] = [1, 366]
        fig.update_layout(layout)
        chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        html_content.append(chart_html)

    # 1.4.1 - 1.4.3 Series Breakdown
    if 'daily_invoice_price_series' in metrics:
        df_price_series = metrics['daily_invoice_price_series']
        target_series = ['LS6', 'L6', 'LS9']
        
        for ser_name in target_series:
            if ser_name not in df_price_series.columns: continue
            
            html_content.append(f"<h3>1.4.{target_series.index(ser_name)+1} {ser_name} 开票价格趋势 (MA7)</h3>")
            s_price_ser = df_price_series[ser_name]
            
            fig = go.Figure()
            for year in [2024, 2025]:
                data_year = s_price_ser[s_price_ser.index.year == year]
                if data_year.empty: continue
                
                # Apply MA7 Smoothing
                min_date = data_year.index.min()
                max_date = data_year.index.max()
                full_idx = pd.date_range(min_date, max_date, freq='D')
                data_year = data_year.reindex(full_idx) # Default fill_value is NaN
                
                ma7_data = data_year.rolling(window=7, min_periods=1).mean()
                
                x_days = ma7_data.index.dayofyear
                dates_str = ma7_data.index.strftime('%Y-%m-%d')
                color = '#3498DB' if year == 2024 else '#E67E22'
                
                fig.add_trace(go.Scatter(
                    x=x_days,
                    y=ma7_data.values,
                    mode='lines',
                    name=f'{year} (MA7)',
                    line=dict(color=color, width=2),
                    hovertemplate="Day %{x} (%{customdata})<br>MA7 Price: %{y:,.0f}<extra></extra>",
                    customdata=dates_str
                ))
                
            layout = get_common_layout(
                title=f"{ser_name} 开票价格趋势对比 (Daily Average Invoice Price - MA7)",
                xaxis_title="年份天数 (Day of Year)",
                yaxis_title="平均开票价格 (CNY)"
            )
            layout['xaxis']['range'] = [1, 366]
            fig.update_layout(layout)
            chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
            html_content.append(chart_html)

    # 2. 退订分析
    html_content.append("<h2>2. 退订分析 (Refund Analysis)</h2>")
    html_content.append("<p>定义：统计周期内提交退订申请 (approve_refund_time) 且存在锁单时间 (lock_time) 的订单。</p>")
    
    if 'refund_analysis' in metrics:
        df_refund = metrics['refund_analysis']
        html_content.append(df_refund.to_html(index=False, classes='table', escape=False, float_format=lambda x: '{:,.0f}'.format(x) if isinstance(x, (int, float)) else x))
    
    # 2.1 锁单-退订周期分布
    if 'refund_duration_dist' in metrics:
        dist_data = metrics['refund_duration_dist']
        
        fig = go.Figure()
        
        # Color palette for comparison
        colors = ['#3498DB', '#E67E22']
        
        for i, item in enumerate(dist_data):
            year = item['year']
            counts = item['counts']
            total = counts.sum()
            # Calculate percentages
            percentages = counts.values / total if total > 0 else counts.values * 0
            
            # Use color based on index
            color = colors[i % len(colors)]
            
            # counts index is labels, values is count
            fig.add_trace(go.Scatter(
                x=counts.index.astype(str), 
                y=percentages, 
                customdata=counts.values,
                mode='lines+markers', 
                name=str(year),
                line=dict(color=color),
                hovertemplate="%{y:.1%}<br>(%{customdata} orders)<extra></extra>"
            ))
            
        layout = get_common_layout(
            title="2.1 锁单-退订周期分布 (Lock-to-Refund Duration)",
            xaxis_title="Duration (Days)",
            yaxis_title="Percentage"
        )
        layout['yaxis']['tickformat'] = '.0%'
        fig.update_layout(layout)
        
        # Generate div
        chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        
        html_content.append("<h3>2.1 锁单-退订周期分布</h3>")
        html_content.append(chart_html)
        
    # 2.2 锁单未交付未退订分布 (2025)
    if 'pending_delivery_dist' in metrics:
        pending_data = metrics['pending_delivery_dist']
        total_count = pending_data['total_count']
        counts = pending_data['counts']
        
        fig = go.Figure()
        
        # Bar chart for frequency (Main Metric -> Blue)
        fig.add_trace(go.Bar(
            x=counts.index.astype(str),
            y=counts.values,
            text=counts.values,
            textposition='auto',
            name="Pending Orders",
            marker_color='#3498DB'
        ))
        
        layout = get_common_layout(
            title=f"2.2 锁单未交付未退订分布 (2025) - Total: {total_count:,}",
            xaxis_title="Duration (Days since Lock)",
            yaxis_title="Count"
        )
        fig.update_layout(layout)
        
        chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        
        html_content.append(f"<h3>2.2 锁单未交付未退订分布 (2025)</h3>")
        html_content.append(f"<p>统计范围: 2025年锁单，截至当前 ({pending_data['data_timestamp'].strftime('%Y-%m-%d')}) 既未退订也未交付的订单。</p>")
        html_content.append(f"<p><strong>总积压量: {total_count:,}</strong></p>")
        html_content.append(chart_html)

    # 2.3 锁单交付周期 (Lock-to-Delivery Cycle)
    if 'delivery_cycle_dist' in metrics:
        html_content.append("<h3>2.3 锁单交付周期分布 (Lock-to-Delivery Cycle)</h3>")
        
        item = metrics['delivery_cycle_dist']
        total = item['total_count']
        counts = item['counts']
        
        if total > 0:
            # Calculate percentages
            percentages = counts.values / total
            
            # Calculate cumulative percentages
            cumsum = counts.cumsum()
            cum_percentages = cumsum / total
            
            fig = go.Figure()
            
            # Left Y-Axis: Bar Chart (Percentage) - Main Metric (#3498DB)
            fig.add_trace(go.Bar(
                x=counts.index.astype(str),
                y=percentages,
                name='占比 (Percentage)',
                marker_color='#3498DB',
                yaxis='y',
                hovertemplate="%{y:.1%}<br>(%{customdata} orders)<extra></extra>",
                customdata=counts.values
            ))
            
            # Right Y-Axis: Line Chart (Cumulative Percentage) - Comparison/Secondary (#E67E22)
            fig.add_trace(go.Scatter(
                x=counts.index.astype(str),
                y=cum_percentages,
                name='累计占比 (Cumulative)',
                mode='lines+markers',
                line=dict(color='#E67E22', width=3),
                yaxis='y2',
                hovertemplate="%{y:.1%}<extra></extra>"
            ))
            
            layout = get_common_layout(
                title=f"锁单交付周期分布 (2024 & 2025) - Total: {total:,}",
                xaxis_title="Duration (Days)",
                yaxis_title="Percentage"
            )
            
            # Customize Y-Axis 1
            layout['yaxis']['tickformat'] = '.0%'
            layout['yaxis']['side'] = 'left'
            layout['yaxis']['range'] = [0, max(percentages) * 1.2]
            
            # Add Y-Axis 2
            layout['yaxis2'] = dict(
                title="Cumulative Percentage",
                tickformat='.0%',
                overlaying='y',
                side='right',
                range=[0, 1.05],
                gridcolor='#ebedf0',
                zerolinecolor='#ebedf0',
                tickfont=dict(color='#7B848F'),
                title_font=dict(color='#7B848F'),
                showgrid=False # Don't overlap grid lines
            )
            
            layout['legend']['orientation'] = 'h'
            layout['legend']['x'] = 0.5
            layout['legend']['y'] = 1.1
            layout['legend']['xanchor'] = 'center'
            
            fig.update_layout(layout)
            
            chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
            html_content.append(chart_html)

    # 2.4 锁单未交付预估 (2025)
    if 'pending_delivery_dist' in metrics:
        pending_counts = metrics['pending_delivery_dist']['counts']
        
        # Calculate conversion probabilities
        conv_prob = calculate_conversion_probability(metrics)
        
        if conv_prob is not None:
            # Calculate Estimated Deliveries
            # Est(i) = Pending(i) * Prob(i)
            est_deliveries = pending_counts * conv_prob
            total_est_deliveries = est_deliveries.sum()
            
            # Calculate Cumulative Estimated Deliveries
            cum_est_deliveries = est_deliveries.cumsum()
            
            fig = go.Figure()
            
            # Trace 1: Conversion Probability (Left Y) - Main Metric (Line)
            # Use Blue (#3498DB) as requested for Main Metric
            fig.add_trace(go.Scatter(
                x=conv_prob.index.astype(str),
                y=conv_prob.values,
                name='转化概率估计 (Prob)',
                mode='lines+markers',
                line=dict(color='#3498DB', width=3),
                yaxis='y',
                hovertemplate="Prob: %{y:.1%}<extra></extra>"
            ))
            
            # Trace 2: Estimated Cumulative Delivery Count (Right Y) - Secondary Metric
            # Use Orange (#E67E22)
            fig.add_trace(go.Scatter(
                x=cum_est_deliveries.index.astype(str),
                y=cum_est_deliveries.values,
                name='预估累计交付数 (Cum Est)',
                mode='lines+markers', # Or 'lines' or 'bar'
                fill='tozeroy', # Optional: fill area to show accumulation
                line=dict(color='#E67E22', width=3, dash='dot'),
                yaxis='y2',
                hovertemplate="Cum Est: %{y:.0f}<extra></extra>"
            ))
            
            # Add Bar for specific bin estimate (Optional but helpful context)
            # Make it light/transparent so it doesn't distract
            fig.add_trace(go.Bar(
                x=est_deliveries.index.astype(str),
                y=est_deliveries.values,
                name='本区间预估交付 (Est)',
                marker_color='#3498DB',
                opacity=0.3,
                yaxis='y2',
                hovertemplate="Est: %{y:.1f}<extra></extra>"
            ))

            layout = get_common_layout(
                title=f"2.4 锁单未交付预估 (2025) - Total Est Conversion: {int(total_est_deliveries):,}",
                xaxis_title="Duration (Days since Lock)",
                yaxis_title="Conversion Probability"
            )
            
            # Left Y: Probability
            layout['yaxis']['tickformat'] = '.0%'
            layout['yaxis']['range'] = [0, 1.05]
            
            # Right Y: Count
            layout['yaxis2'] = dict(
                title="Estimated Delivery Count",
                overlaying='y',
                side='right',
                gridcolor='#ebedf0', # Show grid? Maybe not to avoid clutter
                zerolinecolor='#ebedf0',
                tickfont=dict(color='#7B848F'),
                title_font=dict(color='#7B848F'),
                showgrid=False
            )
            
            layout['legend']['orientation'] = 'h'
            layout['legend']['x'] = 0.5
            layout['legend']['y'] = 1.1
            layout['legend']['xanchor'] = 'center'
            
            fig.update_layout(layout)
            
            chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
            
            html_content.append("<h3>2.4 锁单未交付预估 (2025)</h3>")
            html_content.append(f"<p>基于历史交付与退订数据建模，预估当前积压订单的最终交付转化情况。</p>")
            html_content.append(f"<p><strong>当前积压总量: {pending_counts.sum():,}</strong></p>")
            html_content.append(f"<p><strong>预估最终交付: {int(total_est_deliveries):,} (转化率: {total_est_deliveries/pending_counts.sum():.1%})</strong></p>")
            html_content.append(chart_html)

    # 3. 交付分析 (Delivery Analysis)
    html_content.append("<h2>3. 交付分析</h2>")
    
    # 交付概览 (Delivery Overview)
    if 'delivery_overview' in metrics:
        html_content.append("<h3>交付效率概览 (Delivery Efficiency)</h3>")
        
        overview = metrics['delivery_overview']
        # Data structure: list of dicts [{'year': 2024, ...}, {'year': 2025, ...}]
        
        # We need to format this into a comparison table
        # Rows: 30-day rate, 98-day rate
        # Cols: 2024, 2025, Diff, Ratio
        
        # Helper to find data by year
        def get_year_data(y):
            for item in overview:
                if item['year'] == y:
                    return item
            return {}
            
        data_2024 = get_year_data(2024)
        data_2025 = get_year_data(2025)
        
        metrics_to_show = [
            ("30日锁单交付率", "rate_30d"),
            ("98日锁单交付率", "rate_98d")
        ]
        
        table_html = """
        <table>
            <tr>
                <th>指标</th>
                <th>2024 全年</th>
                <th>2025 (至今)</th>
                <th>Diff (pp)</th>
            </tr>
        """
        
        for label, key in metrics_to_show:
            val_2024 = data_2024.get(key, 0.0)
            val_2025 = data_2025.get(key, 0.0)
            
            diff = val_2025 - val_2024
            
            # Formatting
            # Rates are floats, show as percentage
            str_2024 = f"{val_2024:.1%}"
            str_2025 = f"{val_2025:.1%}"
            
            # Diff in percentage points (pp)
            str_diff = f"{diff*100:+.1f} pp"
            
            if diff < 0:
                str_diff = f"<span style='color: red'>{str_diff}</span>"
            elif diff > 0:
                str_diff = f"<span style='color: green'>{str_diff}</span>"
                
            table_html += f"""
            <tr>
                <td>{label}</td>
                <td>{str_2024}</td>
                <td>{str_2025}</td>
                <td>{str_diff}</td>
            </tr>
            """
            
        table_html += "</table>"
        html_content.append(table_html)

    # 3.1 锁单交付率趋势 (Delivery Rate Trend)
    if 'delivery_trend' in metrics:
        html_content.append("<h3>3.1 锁单交付率趋势 (Delivery Rate Trends)</h3>")
        
        trend_data = metrics['delivery_trend']
        
        for year in [2024, 2025]:
            if year not in trend_data:
                continue
                
            df_trend = trend_data[year]
            
            # Prepare data for LOWESS
            # X must be numeric (e.g., timestamps converted to float or integers)
            # We use days from start of year or simple range
            x_numeric = (df_trend.index - df_trend.index.min()).days.values
            
            # Helper to calculate LOWESS
            def calculate_lowess(y_values, frac=0.2):
                # statsmodels lowess returns (x, y) sorted by x
                # We need to map it back or just use the returned y since our x is sorted
                smoothed = sm.nonparametric.lowess(y_values, x_numeric, frac=frac)
                return smoothed[:, 1] # Return Y values
            
            # Calculate smoothed lines
            # Handle potential NaNs by filling or skipping? 
            # lowess handles NaNs poorly usually, better to interpolate or drop
            # For simplicity, let's just run on valid data points
            
            # 30d
            mask_30d = ~np.isnan(df_trend['rate_30d'])
            if mask_30d.sum() > 10: # Only smooth if enough points
                # Re-calculate x for valid points
                x_valid = x_numeric[mask_30d]
                y_valid = df_trend['rate_30d'][mask_30d].values
                y_smooth_30d = sm.nonparametric.lowess(y_valid, x_valid, frac=0.2)[:, 1]
                x_smooth_30d = df_trend.index[mask_30d]
            else:
                y_smooth_30d = []
                x_smooth_30d = []

            # 98d
            mask_98d = ~np.isnan(df_trend['rate_98d'])
            if mask_98d.sum() > 10:
                x_valid = x_numeric[mask_98d]
                y_valid = df_trend['rate_98d'][mask_98d].values
                y_smooth_98d = sm.nonparametric.lowess(y_valid, x_valid, frac=0.2)[:, 1]
                x_smooth_98d = df_trend.index[mask_98d]
            else:
                y_smooth_98d = []
                x_smooth_98d = []

            fig = go.Figure()
            
            # 1. Scatter Points (Raw Daily Data)
            # 30d Rate (Blue, transparent)
            fig.add_trace(go.Scatter(
                x=df_trend.index.astype(str),
                y=df_trend['rate_30d'],
                name='30日交付率 (Daily)',
                mode='markers',
                marker=dict(color='rgba(52, 152, 219, 0.3)', size=6), # #3498DB with opacity
                yaxis='y',
                hovertemplate="30d Rate: %{y:.1%}<br>(%{customdata} orders)<extra></extra>",
                customdata=df_trend['count_30d']
            ))
            
            # 2. LOWESS Curves (Trend)
            # 30d Trend (Blue, solid)
            if len(x_smooth_30d) > 0:
                fig.add_trace(go.Scatter(
                    x=x_smooth_30d.astype(str),
                    y=y_smooth_30d,
                    name='30日趋势 (LOWESS)',
                    mode='lines',
                    line=dict(color='#3498DB', width=3),
                    yaxis='y',
                    hovertemplate="30d Trend: %{y:.1%}<extra></extra>"
                ))
            
            # 3. Scatter Points (Raw Daily Data)
            # 98d Rate (Orange, transparent)
            fig.add_trace(go.Scatter(
                x=df_trend.index.astype(str),
                y=df_trend['rate_98d'],
                name='98日交付率 (Daily)',
                mode='markers',
                marker=dict(color='rgba(230, 126, 34, 0.3)', size=6), # #E67E22 with opacity
                yaxis='y2',
                hovertemplate="98d Rate: %{y:.1%}<br>(%{customdata} orders)<extra></extra>",
                customdata=df_trend['count_98d']
            ))
            
            # 4. LOWESS Curves (Trend)
            # 98d Trend (Orange, solid)
            if len(x_smooth_98d) > 0:
                fig.add_trace(go.Scatter(
                    x=x_smooth_98d.astype(str),
                    y=y_smooth_98d,
                    name='98日趋势 (LOWESS)',
                    mode='lines',
                    line=dict(color='#E67E22', width=3),
                    yaxis='y2',
                    hovertemplate="98d Trend: %{y:.1%}<extra></extra>"
                ))
            
            layout = get_common_layout(
                title=f"3.1 锁单交付率趋势 - {year} (Daily + LOWESS)",
                xaxis_title="Lock Time (Day)",
                yaxis_title="30-day Rate"
            )
            
            # Axis 1
            layout['yaxis']['tickformat'] = '.0%'
            layout['yaxis']['range'] = [0, 1.05]
            
            # Axis 2
            layout['yaxis2'] = dict(
                title="98-day Rate",
                tickformat='.0%',
                overlaying='y',
                side='right',
                range=[0, 1.05],
                gridcolor='#ebedf0',
                zerolinecolor='#ebedf0',
                tickfont=dict(color='#7B848F'),
                title_font=dict(color='#7B848F'),
                showgrid=False
            )
            
            layout['legend']['orientation'] = 'h'
            layout['legend']['x'] = 0.5
            layout['legend']['y'] = 1.1
            layout['legend']['xanchor'] = 'center'
            
            fig.update_layout(layout)
            
            chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
            html_content.append(chart_html)

    # 4. 渠道分析 (Channel Analysis)
    if 'active_store_series' in metrics:
        html_content.append("<h2>4. 渠道分析 (Channel Analysis)</h2>")
        html_content.append("<h3>4.1 在营门店数概览 (Active Store Overview)</h3>")
        
        s_active = metrics['active_store_series']
        
        # Calculate stats for each year
        stats = {}
        raw_stats = {} # Store raw numbers for calculation
        
        for year in [2024, 2025]:
            s_year = s_active[s_active.index.year == year]
            if s_year.empty:
                stats[year] = {'min': '-', 'max': '-', 'mean': '-'}
                raw_stats[year] = {'min': np.nan, 'max': np.nan, 'mean': np.nan}
            else:
                _min = int(s_year.min())
                _max = int(s_year.max())
                _mean = s_year.mean()
                
                stats[year] = {
                    'min': _min,
                    'max': _max,
                    'mean': f"{_mean:.1f}"
                }
                raw_stats[year] = {
                    'min': _min,
                    'max': _max,
                    'mean': _mean
                }
        
        # Helper to calculate Diff and Ratio
        def get_diff_ratio(metric_key):
            v24 = raw_stats[2024][metric_key]
            v25 = raw_stats[2025][metric_key]
            
            if pd.isna(v24) or pd.isna(v25):
                return "-", "-"
                
            diff = v25 - v24
            if v24 != 0:
                ratio = diff / v24
            else:
                ratio = np.nan
                
            # Format Diff
            diff_color = "green" if diff > 0 else "red" if diff < 0 else "black"
            diff_prefix = "+" if diff > 0 else ""
            diff_str = f"<span style='color: {diff_color}'>{diff_prefix}{diff:.1f}</span>"
            
            # Format Ratio
            if pd.isna(ratio):
                ratio_str = "-"
            else:
                ratio_color = "green" if ratio > 0 else "red" if ratio < 0 else "black"
                ratio_prefix = "+" if ratio > 0 else ""
                ratio_str = f"<span style='color: {ratio_color}'>{ratio_prefix}{ratio:.1%}</span>"
                
            return diff_str, ratio_str

        diff_min, ratio_min = get_diff_ratio('min')
        diff_max, ratio_max = get_diff_ratio('max')
        diff_mean, ratio_mean = get_diff_ratio('mean')

        # Build Transposed Table
        # Rows: Indicators
        # Columns: Years, Diff, Ratio
        table_html = f"""
        <table>
            <thead>
                <tr>
                    <th>指标 (Metric)</th>
                    <th>2024</th>
                    <th>2025</th>
                    <th>差异 (Diff)</th>
                    <th>同比 (YoY)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>最小值 (Min)</td>
                    <td>{stats[2024]['min']}</td>
                    <td>{stats[2025]['min']}</td>
                    <td>{diff_min}</td>
                    <td>{ratio_min}</td>
                </tr>
                <tr>
                    <td>最大值 (Max)</td>
                    <td>{stats[2024]['max']}</td>
                    <td>{stats[2025]['max']}</td>
                    <td>{diff_max}</td>
                    <td>{ratio_max}</td>
                </tr>
                <tr>
                    <td>平均值 (Mean)</td>
                    <td>{stats[2024]['mean']}</td>
                    <td>{stats[2025]['mean']}</td>
                    <td>{diff_mean}</td>
                    <td>{ratio_mean}</td>
                </tr>
            </tbody>
        </table>
        """
        html_content.append(table_html)

        # Prepare data for 4.2 and 4.3
        s_2024 = s_active[s_active.index.year == 2024]
        s_2025 = s_active[s_active.index.year == 2025]

        # 4.2 Total Operating Days Analysis (Moved from 4.3)
        html_content.append("<h3>4.2 营业总时长分析 (Total Operating Days Analysis)</h3>")
        html_content.append("<p>统计2024年和2025年所有在营门店的营业天数总和 (Sum of operating days for all active stores).</p>")
        
        # Calculate Total Operating Days
        total_days_2024 = int(s_2024.sum()) if not s_2024.empty else 0
        total_days_2025 = int(s_2025.sum()) if not s_2025.empty else 0
        
        # Calculate Diff and Ratio
        diff_days = total_days_2025 - total_days_2024
        
        if total_days_2024 != 0:
            ratio_days = diff_days / total_days_2024
        else:
            ratio_days = np.nan
            
        # Format Diff
        diff_color = "green" if diff_days > 0 else "red" if diff_days < 0 else "black"
        diff_prefix = "+" if diff_days > 0 else ""
        diff_str = f"<span style='color: {diff_color}'>{diff_prefix}{diff_days:,}</span>"
        
        # Format Ratio
        if pd.isna(ratio_days):
            ratio_str = "-"
        else:
            ratio_color = "green" if ratio_days > 0 else "red" if ratio_days < 0 else "black"
            ratio_prefix = "+" if ratio_days > 0 else ""
            ratio_str = f"<span style='color: {ratio_color}'>{ratio_prefix}{ratio_days:.1%}</span>"
            
        # Table for 4.2
        table_4_2 = f"""
        <table>
            <thead>
                <tr>
                    <th>指标 (Metric)</th>
                    <th>2024 总计</th>
                    <th>2025 总计</th>
                    <th>差异 (Diff)</th>
                    <th>同比 (YoY)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>营业总时长 (Total Operating Days)</td>
                    <td>{total_days_2024:,}</td>
                    <td>{total_days_2025:,}</td>
                    <td>{diff_str}</td>
                    <td>{ratio_str}</td>
                </tr>
            </tbody>
        </table>
        """
        html_content.append(table_4_2)

        # 4.3 Comparison Chart (Active Store Comparison - Day Aligned) (Moved from 4.2)
        html_content.append("<h3>4.3 在营门店数对比 (Active Store Count Comparison)</h3>")
        
        # 4.3.1 Existing Active Store Analysis
        html_content.append("<h4>4.3.1 在营门店数（现有）</h4>")
        
        fig = go.Figure()
        
        # 2024 Trace
        if not s_2024.empty:
            # X = Day of Year
            x_2024 = s_2024.index.dayofyear
            # Format dates for hover
            dates_2024 = s_2024.index.strftime('%Y-%m-%d')
            
            fig.add_trace(go.Scatter(
                x=x_2024,
                y=s_2024.values,
                mode='lines',
                name='2024',
                line=dict(color='#3498DB', width=2),
                hovertemplate="Day %{x} (%{customdata})<br>Active Stores: %{y}<extra>2024</extra>",
                customdata=dates_2024
            ))
            
        # 2025 Trace
        if not s_2025.empty:
            x_2025 = s_2025.index.dayofyear
            dates_2025 = s_2025.index.strftime('%Y-%m-%d')
            
            fig.add_trace(go.Scatter(
                x=x_2025,
                y=s_2025.values,
                mode='lines',
                name='2025',
                line=dict(color='#E67E22', width=2),
                hovertemplate="Day %{x} (%{customdata})<br>Active Stores: %{y}<extra>2025</extra>",
                customdata=dates_2025
            ))
            
        layout = get_common_layout(
            title="4.3.1 在营门店数趋势对比 (2024 vs 2025)",
            xaxis_title="年份天数 (Day of Year)",
            yaxis_title="在营门店数"
        )
        layout['xaxis']['range'] = [1, 366] # Explicitly set range to match 2024
        
        fig.update_layout(layout)
        
        chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        html_content.append(chart_html)

        # 4.3.2 Store Tenure Analysis (New vs Old)
        if 'store_tenure_analysis' in metrics:
            html_content.append("<h4>4.3.2 新老门店锁单能力对比 (New vs Old Store Lock Capability)</h4>")
            html_content.append("<p>基于各周期内门店的存续时长 (Tenure = min(周期末日期, 门店当年最后锁单) - Store Create Date) 进行分组分析。</p>")
            html_content.append("<p>统计口径：仅包含留存锁单 (Retained Lock Orders)。店日均 = 分箱内留存锁单总数 / 分箱内实际在营天数总和（与 4.2 的 active-day 口径一致）。</p>")
            
            tenure_stats = metrics['store_tenure_analysis']
            
            # Chart 1: Store Count Distribution
            fig1 = go.Figure()
            # Chart 2: Avg Locks Distribution
            fig2 = go.Figure()
            
            colors = {2024: '#3498DB', 2025: '#E67E22'}
            
            for year in [2024, 2025]:
                if year not in tenure_stats or tenure_stats[year].empty:
                    continue
                    
                stats = tenure_stats[year]
                
                # Convert index (Interval) to string for X-axis
                x_vals = stats.index.astype(str)
                y_count = stats['store_count']
                y_avg = stats['avg_daily_locks']
                
                # Trace for Count
                fig1.add_trace(go.Bar(
                    x=x_vals,
                    y=y_count,
                    name=f'{year}',
                    marker_color=colors[year]
                ))
                
                # Trace for Avg Locks
                fig2.add_trace(go.Bar(
                    x=x_vals,
                    y=y_avg,
                    name=f'{year}',
                    marker_color=colors[year]
                ))
                
            # Layout 1
            layout1 = get_common_layout(
                title="4.3.2.1 不同存续周期门店分布 (Store Count by Tenure)",
                xaxis_title="存续周期 (Tenure Days)",
                yaxis_title="门店数量 (Store Count)"
            )
            layout1['barmode'] = 'group'
            fig1.update_layout(layout1)
            html_content.append(pio.to_html(fig1, full_html=False, include_plotlyjs='cdn'))
            
            # Layout 2
            layout2 = get_common_layout(
                title="4.3.2.2 不同存续周期店日均锁单数 (Avg Daily Retained Locks per Store by Tenure)",
                xaxis_title="存续周期 (Tenure Days)",
                yaxis_title="店日均留存锁单数 (Avg Daily Retained Locks)"
            )
            layout2['barmode'] = 'group'
            fig2.update_layout(layout2)
            html_content.append(pio.to_html(fig2, full_html=False, include_plotlyjs='cdn'))

    # 4.4 Average Lock Orders per Store (Daily)
    html_content.append("<h3>4.4 店均锁单数分析 (Average Daily Lock Orders per Store)</h3>")
    html_content.append("<p>统计2024年和2025年每日的“当日锁单数”除以“当日在营门店数”。</p>")
    html_content.append("<p>定义：店均锁单数(d) = 当日锁单数(d) / 在营门店数(d)</p>")

    if 'active_store_series' in metrics and 'daily_lock_counts' in metrics:
        html_content.append("<h4>4.4.0 整体店均锁单数趋势</h4>")
        s_active = metrics['active_store_series']
        daily_locks = metrics['daily_lock_counts']
        
        fig = go.Figure()
        
        for year in [2024, 2025]:
            # Filter data for year
            s_active_year = s_active[s_active.index.year == year]
            daily_locks_year = daily_locks[daily_locks.index.year == year]
            
            if s_active_year.empty:
                continue
                
            # Align dates: Reindex daily locks to match active store dates (fill 0 for no locks)
            # Ensure index is DatetimeIndex
            daily_locks_year = daily_locks_year.reindex(s_active_year.index, fill_value=0)
            
            # Use Daily Locks instead of Cumulative
            # cum_locks = daily_locks_year.cumsum() # REMOVED
            
            # Calculate Average per Store
            # Handle division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                avg_per_store = daily_locks_year / s_active_year
                avg_per_store = avg_per_store.replace([np.inf, -np.inf], np.nan)
            
            # Add LOWESS smoothing for better readability
            # X must be numeric
            x_numeric = (s_active_year.index - s_active_year.index.min()).days.values
            
            # Filter NaNs for smoothing
            mask = ~np.isnan(avg_per_store)
            if mask.sum() > 10:
                y_smooth = sm.nonparametric.lowess(avg_per_store[mask], x_numeric[mask], frac=0.1)[:, 1]
                x_smooth = s_active_year.index[mask]
            else:
                y_smooth = []
                x_smooth = []

            # Plot
            # X = Day of Year
            x_days = s_active_year.index.dayofyear
            # Format dates for hover
            dates_str = s_active_year.index.strftime('%Y-%m-%d')
            
            color = '#3498DB' if year == 2024 else '#E67E22'
            
            # Scatter points (faint)
            fig.add_trace(go.Scatter(
                x=x_days,
                y=avg_per_store,
                mode='markers',
                name=f'{year} (Daily)',
                marker=dict(color=color, size=4, opacity=0.3),
                hovertemplate=f"Day %{{x}} (%{{customdata[2]}})<br>{year} Avg: %{{y:.2f}} orders/store<br>(Locks: %{{customdata[0]}}, Stores: %{{customdata[1]}})<extra></extra>",
                customdata=np.stack((daily_locks_year.values, s_active_year.values, dates_str.values), axis=-1),
                showlegend=False
            ))
            
            # Smooth line (Solid)
            if len(x_smooth) > 0:
                # Get dates for smooth line (subset of original dates)
                dates_smooth = x_smooth.strftime('%Y-%m-%d')
                
                fig.add_trace(go.Scatter(
                    x=x_smooth.dayofyear,
                    y=y_smooth,
                    mode='lines',
                    name=f'{year} (Trend)',
                    line=dict(color=color, width=2),
                    hovertemplate=f"Day %{{x}} (%{{customdata}})<br>{year} Trend: %{{y:.2f}} orders/store<extra></extra>",
                    customdata=dates_smooth
                ))
            
        layout = get_common_layout(
            title="4.4.0 整体店均锁单数趋势对比 (Overall Daily Locks per Store)",
            xaxis_title="年份天数 (Day of Year)",
            yaxis_title="店均锁单数 (Orders per Store)"
        )
        layout['xaxis']['range'] = [1, 366]
        layout['yaxis']['range'] = [0, 2] # Default Y-axis scale
        
        fig.update_layout(layout)
        
        chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        html_content.append(chart_html)

    # 4.4 Series Breakdown (LS6, L6, LS9)
    if 'active_store_series' in metrics and 'daily_locks_series' in metrics:
        s_active = metrics['active_store_series']
        df_locks_series = metrics['daily_locks_series']
        target_series = ['LS6', 'L6', 'LS9']
        
        for ser_name in target_series:
            if ser_name not in df_locks_series.columns:
                continue
                
            html_content.append(f"<h4>4.4.{target_series.index(ser_name)+1} {ser_name} 店均锁单数趋势</h4>")
            
            s_locks_ser = df_locks_series[ser_name]
            
            fig = go.Figure()
            
            for year in [2024, 2025]:
                # Filter data for year
                s_active_year = s_active[s_active.index.year == year]
                s_locks_year = s_locks_ser[s_locks_ser.index.year == year]
                
                if s_active_year.empty:
                    continue
                    
                # Align dates
                s_locks_year = s_locks_year.reindex(s_active_year.index, fill_value=0)
                
                # Calculate Average per Store (Series Locks / Total Active Stores)
                with np.errstate(divide='ignore', invalid='ignore'):
                    avg_per_store = s_locks_year / s_active_year
                    avg_per_store = avg_per_store.replace([np.inf, -np.inf], np.nan)
                
                # LOWESS Smoothing
                x_numeric = (s_active_year.index - s_active_year.index.min()).days.values
                mask = ~np.isnan(avg_per_store)
                
                if mask.sum() > 10:
                    y_smooth = sm.nonparametric.lowess(avg_per_store[mask], x_numeric[mask], frac=0.1)[:, 1]
                    x_smooth = s_active_year.index[mask]
                else:
                    y_smooth = []
                    x_smooth = []

                # Plot
                x_days = s_active_year.index.dayofyear
                dates_str = s_active_year.index.strftime('%Y-%m-%d')
                color = '#3498DB' if year == 2024 else '#E67E22'
                
                # Scatter points (faint)
                fig.add_trace(go.Scatter(
                    x=x_days,
                    y=avg_per_store,
                    mode='markers',
                    name=f'{year} (Daily)',
                    marker=dict(color=color, size=4, opacity=0.3),
                    hovertemplate=f"Day %{{x}} (%{{customdata[2]}})<br>{ser_name} {year} Avg: %{{y:.2f}} orders/store<br>(Locks: %{{customdata[0]}}, Total Stores: %{{customdata[1]}})<extra></extra>",
                    customdata=np.stack((s_locks_year.values, s_active_year.values, dates_str.values), axis=-1),
                    showlegend=False
                ))
                
                # Smooth line (Solid)
                if len(x_smooth) > 0:
                    dates_smooth = x_smooth.strftime('%Y-%m-%d')
                    fig.add_trace(go.Scatter(
                        x=x_smooth.dayofyear,
                        y=y_smooth,
                        mode='lines',
                        name=f'{year} (Trend)',
                        line=dict(color=color, width=2),
                        hovertemplate=f"Day %{{x}} (%{{customdata}})<br>{ser_name} {year} Trend: %{{y:.2f}} orders/store<extra></extra>",
                        customdata=dates_smooth
                    ))
            
            layout = get_common_layout(
                title=f"{ser_name} 店均锁单数趋势 (Series Locks / Total Stores)",
                xaxis_title="年份天数 (Day of Year)",
                yaxis_title="店均锁单数 (Orders per Store)"
            )
            layout['xaxis']['range'] = [1, 366]
            layout['yaxis']['range'] = [0, 2] # Default Y-axis scale
            fig.update_layout(layout)
            
            chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
            html_content.append(chart_html)

    # 4.5 Average Invoice Amount per Store
    html_content.append("<h3>4.5 店均开票价格分析 (Average Invoice Amount per Store)</h3>")
    html_content.append("<p>统计每日“总开票金额”除以“当日在营门店数”。</p>")
    html_content.append("<p>定义：店均开票金额(d) = sum(invoice_amount where upload_time=d) / 在营门店数(d)</p>")

    if 'active_store_series' in metrics and 'daily_invoice_sum' in metrics:
        s_active = metrics['active_store_series']
        daily_invoice = metrics['daily_invoice_sum']
        
        fig = go.Figure()
        
        for year in [2024, 2025]:
            # Filter data for year
            s_active_year = s_active[s_active.index.year == year]
            daily_invoice_year = daily_invoice[daily_invoice.index.year == year]
            
            if s_active_year.empty:
                continue
                
            # Align dates
            daily_invoice_year = daily_invoice_year.reindex(s_active_year.index, fill_value=0)
            
            # Calculate Average per Store
            with np.errstate(divide='ignore', invalid='ignore'):
                avg_per_store = daily_invoice_year / s_active_year
                avg_per_store = avg_per_store.replace([np.inf, -np.inf], np.nan)
            
            # LOWESS Smoothing
            x_numeric = (s_active_year.index - s_active_year.index.min()).days.values
            mask = ~np.isnan(avg_per_store)
            
            if mask.sum() > 10:
                y_smooth = sm.nonparametric.lowess(avg_per_store[mask], x_numeric[mask], frac=0.1)[:, 1]
                x_smooth = s_active_year.index[mask]
            else:
                y_smooth = []
                x_smooth = []

            # Plot
            x_days = s_active_year.index.dayofyear
            dates_str = s_active_year.index.strftime('%Y-%m-%d')
            color = '#3498DB' if year == 2024 else '#E67E22'
            
            # Scatter points (faint)
            fig.add_trace(go.Scatter(
                x=x_days,
                y=avg_per_store,
                mode='markers',
                name=f'{year} (Daily)',
                marker=dict(color=color, size=4, opacity=0.3),
                hovertemplate=f"Day %{{x}} (%{{customdata[2]}})<br>{year} Avg: ¥%{{y:,.0f}}<br>(Total: ¥%{{customdata[0]:,.0f}}, Stores: %{{customdata[1]}})<extra></extra>",
                customdata=np.stack((daily_invoice_year.values, s_active_year.values, dates_str.values), axis=-1),
                showlegend=False
            ))
            
            # Smooth line (Solid)
            if len(x_smooth) > 0:
                dates_smooth = x_smooth.strftime('%Y-%m-%d')
                fig.add_trace(go.Scatter(
                    x=x_smooth.dayofyear,
                    y=y_smooth,
                    mode='lines',
                    name=f'{year} (Trend)',
                    line=dict(color=color, width=2),
                    hovertemplate=f"Day %{{x}} (%{{customdata}})<br>{year} Trend: ¥%{{y:,.0f}}<extra></extra>",
                    customdata=dates_smooth
                ))
        
        layout = get_common_layout(
            title="4.5 店均开票金额趋势对比 (Average Invoice Amount per Store)",
            xaxis_title="年份天数 (Day of Year)",
            yaxis_title="店均开票金额 (RMB)"
        )
        layout['xaxis']['range'] = [1, 366]
        # Auto-scale Y-axis for amount
        
        fig.update_layout(layout)
        
        chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        html_content.append(chart_html)

    # 5. Module 5: Lead Conversion Rate (线索转化率)
    if 'module5_stats' in metrics:
        m5_stats = metrics['module5_stats']
        html_content.append("<h2>5. 线索转化率分析 (Lead Conversion Analysis)</h2>")
        
        # Build Table
        # (Label, Key, FormatType)
        # FormatType: 'int', 'percent', 'float'
        row_configs = [
            ("下发线索数 (Total Leads)", 'total_leads', 'int'),
            ("当日试驾率 (Same-day Test Drive Rate)", 'rate_same_day_test_drive', 'percent'),
            ("7日试驾率 (7-day Test Drive Rate)", 'rate_7d_test_drive', 'percent'),
            ("7日锁单率 (7-day Lock Rate)", 'rate_7d_lock', 'percent'),
            ("30日试驾率 (30-day Test Drive Rate)", 'rate_30d_test_drive', 'percent'),
            ("30日锁单率 (30-day Lock Rate)", 'rate_30d_lock', 'percent'),
            ("店日均下发线索数 (Avg Daily Leads per Store)", 'avg_daily_leads_per_store', 'float')
        ]
        
        tbody_html = ""
        
        for label, key, fmt in row_configs:
            # Get values safely
            stats_2024 = m5_stats.get(2024)
            stats_2025 = m5_stats.get(2025)
            
            v2024 = stats_2024.get(key, 0) if stats_2024 else 0
            v2025 = stats_2025.get(key, 0) if stats_2025 else 0
            
            # Calculate Diff and Ratio (YoY)
            diff = v2025 - v2024
            
            if v2024 != 0:
                ratio = (v2025 - v2024) / v2024
            else:
                ratio = np.nan
            
            # Formatting
            if fmt == 'int':
                v24_str = f"{v2024:,.0f}"
                v25_str = f"{v2025:,.0f}"
                diff_str = f"{diff:+,.0f}"
                ratio_str = f"{ratio:+.1%}" if not pd.isna(ratio) else "-"
            elif fmt == 'percent':
                v24_str = f"{v2024:.1%}"
                v25_str = f"{v2025:.1%}"
                diff_str = f"{diff:+.1%}" 
                ratio_str = f"{ratio:+.1%}" if not pd.isna(ratio) else "-"
            elif fmt == 'float':
                v24_str = f"{v2024:.2f}"
                v25_str = f"{v2025:.2f}"
                diff_str = f"{diff:+.2f}"
                ratio_str = f"{ratio:+.1%}" if not pd.isna(ratio) else "-"
            else:
                v24_str = str(v2024)
                v25_str = str(v2025)
                diff_str = str(diff)
                ratio_str = str(ratio)
                
            # Colors
            diff_color = "green" if diff > 0 else "red" if diff < 0 else "black"
            ratio_color = "green" if (pd.notna(ratio) and ratio > 0) else "red" if (pd.notna(ratio) and ratio < 0) else "black"
            
            diff_html = f"<span style='color: {diff_color}'>{diff_str}</span>"
            ratio_html = f"<span style='color: {ratio_color}'>{ratio_str}</span>"
            
            tbody_html += f"""
                <tr>
                    <td>{label}</td>
                    <td>{v24_str}</td>
                    <td>{v25_str}</td>
                    <td>{diff_html}</td>
                    <td>{ratio_html}</td>
                </tr>
            """
            
        table_html = f"""
        <table class="table table-striped">
            <thead>
                <tr>
                    <th>指标 (Metric)</th>
                    <th>2024</th>
                    <th>2025</th>
                    <th>差异 (Diff)</th>
                    <th>同比 (YoY)</th>
                </tr>
            </thead>
            <tbody>
                {tbody_html}
            </tbody>
        </table>
        """
        html_content.append(table_html)

    # 5.1 Module 5.1: Lead Trends (Module 5.1 模块)
    if 'module5_daily_series' in metrics:
        df_series = metrics['module5_daily_series']
        html_content.append("<h2>5.1 线索趋势对比 (Lead Trends - MA7 Smoothed)</h2>")
        
        # Helper to plot daily comparison
        def plot_daily_metric(col_name, title, y_axis_title, is_rate=False):
            fig = go.Figure()
            
            for year in [2024, 2025]:
                # Filter by year
                df_year = df_series[df_series.index.year == year].copy()
                if df_year.empty:
                    continue
                    
                # Apply MA7 Smoothing
                # 1. Reindex to full year range to handle missing days correctly
                min_date = df_year.index.min()
                max_date = df_year.index.max()
                full_idx = pd.date_range(min_date, max_date, freq='D')
                df_year = df_year.reindex(full_idx)
                
                # Calculate Y values (Daily)
                if is_rate:
                    # Avoid div by zero
                    with np.errstate(divide='ignore', invalid='ignore'):
                        y_daily = df_year[col_name] / df_year['leads_count']
                        y_daily = y_daily.replace([np.inf, -np.inf], np.nan)
                else:
                    y_daily = df_year[col_name]
                
                # 2. Calculate MA7
                # Use min_periods=1 to allow calculation even with some missing data
                y_ma7 = y_daily.rolling(window=7, min_periods=1).mean()
                
                # Prepare Date Strings for Tooltip
                dates_str = y_ma7.index.strftime('%Y-%m-%d')
                
                color = '#3498DB' if year == 2024 else '#E67E22'
                
                # Plot
                fig.add_trace(go.Scatter(
                    x=y_ma7.index.dayofyear,
                    y=y_ma7,
                    mode='lines',
                    name=f'{year} (MA7)',
                    line=dict(color=color),
                    customdata=dates_str,
                    hovertemplate=f"Day %{{x}} (%{{customdata}})<br>{year} MA7: %{{y:.1%}}<extra></extra>" if is_rate else f"Day %{{x}} (%{{customdata}})<br>{year} MA7: %{{y:,.0f}}<extra></extra>",
                ))
            
            layout = get_common_layout(
                title=f"{title} (MA7 Smoothed)",
                xaxis_title="年份天数 (Day of Year)",
                yaxis_title=y_axis_title
            )
            layout['xaxis']['range'] = [1, 366]
            if is_rate:
                 layout['yaxis']['tickformat'] = '.0%'
            
            fig.update_layout(layout)
            return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

        # 1. Daily Leads Count
        html_content.append("<h3>5.1.1 下发线索数趋势 (Daily Leads Count)</h3>")
        html_content.append(plot_daily_metric('leads_count', '每日下发线索数对比', '线索数', is_rate=False))
        
        # 2. 30d Test Drive Rate
        html_content.append("<h3>5.1.2 30日试驾率趋势 (30-day Test Drive Rate)</h3>")
        html_content.append(plot_daily_metric('test_drive_30d', '30日试驾率对比', '转化率', is_rate=True))
        
        # 3. 30d Lock Rate
        html_content.append("<h3>5.1.3 30日锁单率趋势 (30-day Lock Rate)</h3>")
        html_content.append(plot_daily_metric('lock_30d', '30日锁单率对比', '转化率', is_rate=True))

    # 6. Module 6: Test Drive Analysis (试驾分析)
    if 'module6_stats' in metrics:
        m6_stats = metrics['module6_stats']
        html_content.append("<h2>6. 试驾分析 (Test Drive Analysis)</h2>")
        
        row_configs = [
            ("有效试驾数 (Total Valid Test Drives)", 'total_valid_td', 'int'),
            ("L6有效试驾数 (L6 Valid Test Drives)", 'total_L6_td', 'int'),
            ("LS6有效试驾数 (LS6 Valid Test Drives)", 'total_LS6_td', 'int'),
            ("LS9有效试驾数 (LS9 Valid Test Drives)", 'total_LS9_td', 'int'),
            ("店日均试驾数 (Store Daily Avg Test Drives)", 'store_daily_avg', 'float')
        ]
        
        tbody_html = ""
        for label, key, fmt in row_configs:
            stats_2024 = m6_stats.get(2024)
            stats_2025 = m6_stats.get(2025)
            
            v2024 = stats_2024.get(key, 0) if stats_2024 else 0
            v2025 = stats_2025.get(key, 0) if stats_2025 else 0
            
            diff = v2025 - v2024
            if v2024 != 0:
                ratio = (v2025 - v2024) / v2024
            else:
                ratio = np.nan
            
            # Format values
            if fmt == 'int':
                v24_str = f"{int(v2024):,}"
                v25_str = f"{int(v2025):,}"
                diff_str = f"{int(diff):+,}"
            else:
                v24_str = f"{v2024:.2f}"
                v25_str = f"{v2025:.2f}"
                diff_str = f"{diff:+.2f}"
                
            if pd.isna(ratio):
                ratio_str = "N/A"
            else:
                ratio_str = f"{ratio:+.1%}"
            
            # Color coding
            diff_color = "green" if diff > 0 else "red" if diff < 0 else "black"
            ratio_color = "green" if (pd.notna(ratio) and ratio > 0) else "red" if (pd.notna(ratio) and ratio < 0) else "black"
            
            diff_html = f"<span style='color: {diff_color}'>{diff_str}</span>"
            ratio_html = f"<span style='color: {ratio_color}'>{ratio_str}</span>"
            
            tbody_html += f"""
            <tr>
                <td>{label}</td>
                <td>{v24_str}</td>
                <td>{v25_str}</td>
                <td>{diff_html}</td>
                <td>{ratio_html}</td>
            </tr>
            """
            
        table_html = f"""
        <table class="table table-striped">
            <thead>
                <tr>
                    <th>指标 (Metric)</th>
                    <th>2024</th>
                    <th>2025</th>
                    <th>差异 (Diff)</th>
                    <th>同比 (YoY)</th>
                </tr>
            </thead>
            <tbody>
                {tbody_html}
            </tbody>
        </table>
        """
        html_content.append(table_html)

        # 6.1 Module 6.1: Test Drive Trends
        html_content.append("<h2>6.1 试驾趋势分析 (Test Drive Trends)</h2>")
        
        # Define charts to generate
        charts_config = [
            ('daily_avg_per_store', '店日均试驾数 (Store Daily Avg Test Drives)', 'Avg Test Drives'),
            ('daily_avg_LS6', '店日均LS6试驾数 (Store Daily Avg LS6)', 'Avg LS6 Test Drives'),
            ('daily_avg_L6', '店日均L6试驾数 (Store Daily Avg L6)', 'Avg L6 Test Drives')
        ]
        
        for col_key, title, y_label in charts_config:
            fig = go.Figure()
            
            for year in [2024, 2025]:
                m6_stats_year = m6_stats.get(year)
                if not m6_stats_year or 'daily_series' not in m6_stats_year:
                    continue
                    
                df_series_year = m6_stats_year['daily_series']
                if df_series_year.empty:
                    continue
                
                # Reindex to full year for correct smoothing
                min_date = df_series_year.index.min()
                max_date = df_series_year.index.max()
                if pd.isna(min_date) or pd.isna(max_date):
                     continue

                full_idx = pd.date_range(min_date, max_date, freq='D')
                df_resampled = df_series_year.reindex(full_idx)
                
                # Get series
                if col_key not in df_resampled.columns:
                    continue
                    
                y_daily = df_resampled[col_key]
                
                # MA7 Smoothing
                y_ma7 = y_daily.rolling(window=7, min_periods=1).mean()
                
                # Prepare Tooltip Data
                dates_str = y_ma7.index.strftime('%Y-%m-%d')
                color = '#3498DB' if year == 2024 else '#E67E22'
                
                fig.add_trace(go.Scatter(
                    x=y_ma7.index.dayofyear,
                    y=y_ma7,
                    mode='lines',
                    name=f'{year} (MA7)',
                    line=dict(color=color),
                    customdata=dates_str,
                    hovertemplate=f"Day %{{x}} (%{{customdata}})<br>{year} MA7: %{{y:.2f}}<extra></extra>"
                ))
            
            layout = get_common_layout(
                title=f"{title} (MA7 Smoothed)",
                xaxis_title="年份天数 (Day of Year)",
                yaxis_title=y_label
            )
            layout['xaxis']['range'] = [1, 366]
            fig.update_layout(layout)
            
            chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
            html_content.append(chart_html)

    # ==========================================
    # 7. 用户画像 (User Profile)
    # ==========================================
    if 'age_series_stats' in metrics or 'age_region_stats' in metrics:
        html_content.append("<h2>7. 用户画像 (User Profile)</h2>")
        
        # 7.1 Age Structure
        html_content.append("<h3>7.1 年龄结构 (Age Structure)</h3>")
        html_content.append("<p>统计口径：基于留存锁单数据 (Retained Lock Orders，即不包含任何已退订订单)，年龄范围过滤 [18, 100]。</p>")
        
        # 7.1.1 By Series
        if 'age_series_stats' in metrics:
            html_content.append("<h4>7.1.1 分车型平均年龄 (Average Age by Series)</h4>")
            df_table = metrics['age_series_stats']
            html_content.append(df_table.to_html(index=False, classes='table', escape=False, float_format=lambda x: '{:,.1f}'.format(x) if isinstance(x, (int, float)) else x))
            
        # 7.1.2 By Parent Region
        if 'age_region_stats' in metrics:
            html_content.append("<h4>7.1.2 分大区平均年龄 (Average Age by Region)</h4>")
            df_table = metrics['age_region_stats']
            html_content.append(df_table.to_html(index=False, classes='table', escape=False, float_format=lambda x: '{:,.1f}'.format(x) if isinstance(x, (int, float)) else x))

        # 7.2 Age Trends
        if 'age_trends' in metrics:
            html_content.append("<h3>7.2 年龄趋势 (Age Trends)</h3>")
            html_content.append("<p>展示每日平均年龄走势 (Day of Year 对齐)。包含：<br>1. 每日散点 (Daily)<br>2. 7日移动平均 (MA7)<br>3. LOWESS 平滑趋势线</p>")
            
            age_trends = metrics['age_trends']
            # Series: LS6, L6, LS9
            target_series = ["L6", "LS6", "LS9"] # Align order with previous sections if possible, or just iterate dict
            
            for s in target_series:
                if s not in age_trends:
                    continue
                    
                fig = go.Figure()
                
                # Colors
                color_2024 = '#95a5a6' # Gray
                color_2025 = '#3498db' # Blue
                
                # 2024
                data_2024 = age_trends[s][2024]
                if not data_2024.empty:
                    # Sort by index (doy)
                    data_2024 = data_2024.sort_index()
                    x_2024 = data_2024.index
                    y_2024 = data_2024.values
                    
                    # 1. Scatter (Raw)
                    fig.add_trace(go.Scatter(
                        x=x_2024, y=y_2024,
                        mode='markers',
                        name='2024 Daily',
                        marker=dict(color=color_2024, size=4, opacity=0.3),
                        showlegend=False
                    ))
                    
                    # 2. MA7
                    ma7_2024 = data_2024.rolling(window=7, min_periods=1).mean()
                    fig.add_trace(go.Scatter(
                        x=ma7_2024.index, y=ma7_2024.values,
                        mode='lines',
                        name='2024 MA7',
                        line=dict(color=color_2024, width=2),
                        opacity=0.8
                    ))
                    
                    # 3. LOWESS
                    # frac=0.2 means using 20% of data for smoothing window
                    lowess_2024 = sm.nonparametric.lowess(y_2024, x_2024, frac=0.2)
                    fig.add_trace(go.Scatter(
                        x=lowess_2024[:, 0], y=lowess_2024[:, 1],
                        mode='lines',
                        name='2024 LOWESS',
                        line=dict(color=color_2024, width=3, dash='dash'),
                    ))

                # 2025
                data_2025 = age_trends[s][2025]
                if not data_2025.empty:
                    # Sort by index
                    data_2025 = data_2025.sort_index()
                    x_2025 = data_2025.index
                    y_2025 = data_2025.values
                    
                    # 1. Scatter (Raw)
                    fig.add_trace(go.Scatter(
                        x=x_2025, y=y_2025,
                        mode='markers',
                        name='2025 Daily',
                        marker=dict(color=color_2025, size=4, opacity=0.3),
                        showlegend=False
                    ))
                    
                    # 2. MA7
                    ma7_2025 = data_2025.rolling(window=7, min_periods=1).mean()
                    fig.add_trace(go.Scatter(
                        x=ma7_2025.index, y=ma7_2025.values,
                        mode='lines',
                        name='2025 MA7',
                        line=dict(color=color_2025, width=2),
                        opacity=0.8
                    ))
                    
                    # 3. LOWESS
                    lowess_2025 = sm.nonparametric.lowess(y_2025, x_2025, frac=0.2)
                    fig.add_trace(go.Scatter(
                        x=lowess_2025[:, 0], y=lowess_2025[:, 1],
                        mode='lines',
                        name='2025 LOWESS',
                        line=dict(color=color_2025, width=3, dash='dash'),
                    ))

                layout = get_common_layout(
                    title=f"7.2 {s} 平均年龄趋势 (Average Age Trend)",
                    xaxis_title="Day of Year",
                    yaxis_title="Average Age"
                )
                layout['xaxis']['range'] = [1, 366]
                # Auto Y-axis should be fine, or strictly [20, 50]? Auto is better for detail.
                fig.update_layout(layout)
                
                chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
                html_content.append(chart_html)

        # 7.3 Gender Structure
        if 'gender_stats' in metrics:
            html_content.append("<h3>7.3 性别结构 (Gender Structure)</h3>")
            html_content.append("<p>统计口径：基于留存锁单数据 (Retained Lock Orders)。</p>")
            
            gender_stats = metrics['gender_stats']
            
            # 7.3.1 By Series
            html_content.append("<h4>7.3.1 分车型性别分布 (Gender by Series)</h4>")
            if 'series' in gender_stats:
                target_series = ["L6", "LS6", "LS9"]
                for s in target_series:
                    if s in gender_stats['series']:
                        html_content.append(f"<h5>{s}</h5>")
                        df_table = gender_stats['series'][s]
                        html_content.append(df_table.to_html(index=False, classes='table', escape=False))
            
            # 7.3.2 By Region
            html_content.append("<h4>7.3.2 分区域性别分布 (Gender by Region)</h4>")
            if 'region' in gender_stats:
                # Sort regions alphabetically for display consistency
                regions = sorted(gender_stats['region'].keys())
                for region in regions:
                    df_table = gender_stats['region'][region]
                    html_content.append(f"<h5>{region}</h5>")
                    html_content.append(df_table.to_html(index=False, classes='table', escape=False))

    html_content.append("</body></html>")
    
    # 保存
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_content))
    
    print(f"✅ Report generated at: {output_file}")

def main():
    try:
        df = load_data(PARQUET_FILE)
        metrics = calculate_metrics(df)
        generate_html(metrics, DEFAULT_OUTPUT)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
