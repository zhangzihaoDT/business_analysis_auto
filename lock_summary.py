import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import json
import re

import pandas as pd


DEFAULT_INPUT = Path(
    "/Users/zihao_/Documents/coding/dataset/formatted/order_full_data.parquet"
)
OUT_DIR = Path(
    "/Users/zihao_/Documents/coding/dataset/processed/analysis_results"
)
BUSINESS_DEF_FILE = Path(
    "/Users/zihao_/Documents/github/W52_reasoning/world/business_definition.json"
)

# Will be populated from JSON
BUSINESS_GROUPS = {}


def load_business_definition(file_path: Path) -> dict:
    if not file_path.exists():
        # Fallback or raise error? User said to use it.
        raise FileNotFoundError(f"Business definition file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace(" ", " ")


def resolve_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for cand in candidates:
        if cand in df.columns:
            return cand
    cand_norm = [normalize(c) for c in candidates]
    col_norm_map = {normalize(c): c for c in df.columns}
    for cn in cand_norm:
        if cn in col_norm_map:
            return col_norm_map[cn]
    raise KeyError(
        f"Unable to resolve column from candidates: {candidates}. Available columns: {list(df.columns)}"
    )


def parse_sql_condition(df: pd.DataFrame, condition_str: str, product_col: str) -> pd.Series:
    """
    Parses simple SQL-like conditions for product filtering.
    Replaces 'Product Name' with the actual column name.
    """
    # Replace "Product Name" (case insensitive) with reference to product_col
    # We use a placeholder to avoid messing up if product_col has spaces/quotes
    
    # 1. Normalize the condition string to use a standard placeholder for the column
    # The JSON uses "Product Name". 
    # Example: "Product Name LIKE '%LS6%'"
    
    # We will simply interpret the logic by manual parsing or eval with replacement.
    # Given the complexity, eval is easiest if we sanitize/prepare the string.
    
    # Replace "Product Name" with "df[product_col]" is risky if product_col is not a valid identifier.
    # Safer: Use a variable name `series_col` in eval context.
    
    # Prepare the string for python eval:
    # LIKE '%...%' -> .str.contains('...', case=False, na=False)
    # NOT LIKE -> ~...
    # AND -> &
    # OR -> |
    
    # Note: analyze_2025.py uses regex substitution. I will use similar approach.
    
    # Step 1: Replace column name variants with a standard token
    # Regex for "Product Name" (case insensitive)
    # Be careful not to match inside quotes if possible, but here keys are usually field names.
    s = re.sub(r"(?i)product\s+name", "TARGET_COL", condition_str)
    s = re.sub(r"(?i)product_name", "TARGET_COL", s)
    
    # Step 2: NOT LIKE
    # TARGET_COL NOT LIKE '%value%'
    def not_like_replacer(match):
        val = match.group(1)
        return f"~df['{product_col}'].astype(str).str.contains('{val}', case=False, na=False)"

    s = re.sub(r"TARGET_COL\s+NOT\s+LIKE\s+'%([^%]+)%+'", not_like_replacer, s)

    # Step 3: LIKE
    # TARGET_COL LIKE '%value%'
    def like_replacer(match):
        val = match.group(1)
        return f"df['{product_col}'].astype(str).str.contains('{val}', case=False, na=False)"

    s = re.sub(r"TARGET_COL\s+LIKE\s+'%([^%]+)%+'", like_replacer, s)
    
    # Step 4: AND / OR / ELSE
    s = s.replace(" AND ", " & ").replace(" OR ", " | ")
    
    if "ELSE" in s:
        # ELSE usually means "True" (fallback) if checked last, but here we return True 
        # so it can be assigned if nothing else matched. 
        # However, usually we apply these in sequence.
        return pd.Series([True] * len(df), index=df.index)

    try:
        return eval(s)
    except Exception as e:
        print(f"Warning: Failed to parse condition: {condition_str}. Error: {e}")
        return pd.Series([False] * len(df), index=df.index)


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(空表，无数据)"
    df_fmt = df.copy()
    return df_fmt.to_markdown(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="锁单数分车型分大区汇总"
    )
    parser.add_argument(
        "--start",
        required=True,
        help="开始日期，格式 YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="结束日期，格式 YYYY-MM-DD",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="输入 Parquet 文件路径",
    )
    parser.add_argument(
        "--models",
        help="逗号分隔的车型列表，仅显示这些车型列，例如: 'CM2,CM2 增程,LS9'",
    )
    parser.add_argument(
        "--sections",
        help="逗号分隔的模块键或组：overview,deposit,region,city_level,province,city,age,gender",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="生成报告后自动发送到飞书",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    SECTION_GROUPS = {
        "overview": {"overview"},
        "deposit": {"deposit_daily"},
        "region": {"region_biz_matrix", "region_model_matrix"},
        "city_level": {"city_level"},
        "province": {"province"},
        "city": {"city_top10"},
        "age": {"age_stats", "age_distribution", "age_model_pct", "age_region_pct_by_model"},
        "gender": {"gender"},
    }
    raw_sections = set([s.strip() for s in str(args.sections).split(",") if s.strip()]) if getattr(args, "sections", None) else set()
    expanded = set()
    for t in raw_sections:
        if t in SECTION_GROUPS:
            expanded |= SECTION_GROUPS[t]
        else:
            expanded.add(t)
    include_sections = expanded
    def section_enabled(key: str) -> bool:
        return (not include_sections) or (key in include_sections)
    
    # Load Business Definition
    try:
        biz_def = load_business_definition(BUSINESS_DEF_FILE)
        series_group_logic = biz_def.get("series_group_logic", {})
        product_type_logic = biz_def.get("product_type_logic", {})
        model_series_mapping = biz_def.get("model_series_mapping", {})
    except Exception as e:
        print(f"Error loading business definition: {e}")
        # Fallback or exit?
        # Assuming required.
        raise e

    # Update global BUSINESS_GROUPS based on model_series_mapping
    # But we also need to account for "CM2 增程" if we create it.
    # We will update this after processing data or just prepopulate from mapping.
    # The script uses BUSINESS_GROUPS for filtering.
    
    # Construct initial BUSINESS_GROUPS from JSON mapping
    # Note: user wants to use "series" from new data source.
    # So keys are series names (LS6, L6).
    global BUSINESS_GROUPS
    BUSINESS_GROUPS = model_series_mapping.copy()

    df = pd.read_parquet(input_path)

    lock_col = resolve_column(
        df,
        [
            "Lock_Time",
            "Lock Time",
            "lock_time",
            "lock time",
            "LockTime",
            "锁单时间",
            "LockDate",
        ],
    )
    # Region col might be different now.
    # New data has license_province, license_city.
    # Old script grouped by "Parent Region Name". 
    # If not present, we can default to "license_province" or just try to find it.
    try:
        region_col = resolve_column(
            df,
            [
                "Parent Region Name",
                "Parent_Region_Name",
                "Parent Region",
                "Region Name",
                "区域",
                "大区",
            ],
        )
    except KeyError:
        # If absolutely no region column, create a dummy one
        print("Warning: No region column found. Using 'Unknown'.")
        df["Unknown_Region"] = "Unknown"
        region_col = "Unknown_Region"

    # Resolve approve_refund_time for Retained Lock Logic
    try:
        approve_refund_col = resolve_column(
            df,
            [
                "approve_refund_time",
                "Approve Refund Time",
                "ApproveRefundTime",
            ]
        )
        df[approve_refund_col] = pd.to_datetime(df[approve_refund_col], errors="coerce")
    except KeyError:
        approve_refund_col = None
        print("Warning: approve_refund_time not found. Retained Lock Total will equal Lock Total.")

    # We don't need model_group_col input anymore because we construct it from product_name
    product_name_col = resolve_column(
        df,
        [
            "product_name",
            "productname",
            "ProductName",
            "Product Name",
            "产品名称",
            "商品名称",
        ],
    )
    order_no_col = resolve_column(
        df,
        [
            "order_number",
            "Order Number",
            "Order_Number",
            "OrderNo",
            "Order No",
            "订单编号",
            "订单号",
            "意向单号",
            "Intention Order Number",
        ],
    )
    
    # Series col
    series_col = resolve_column(
        df,
        [
            "series",
            "Series",
            "车系",
        ]
    )

    df[lock_col] = pd.to_datetime(df[lock_col], errors="coerce")

    # --- Classification Logic ---
    # 1. Generate "车型分类" (Sub-model) using series_group_logic
    df["车型分类"] = "其他" # Default
    
    # Iterate through logic. Order matters.
    # Since JSON dicts are ordered, we follow that order.
    for code, condition in series_group_logic.items():
        mask = parse_sql_condition(df, condition, product_name_col)
        # Apply to those who are currently "其他" or overwrite?
        # Usually specific rules come first or last.
        # "series_group_logic" has "其他": "ELSE" at the end.
        # So we should probably apply if match.
        # But if multiple match? The first one usually wins in CASE WHEN logic.
        # Let's assume we fill "车型分类" if it is still "其他" or just overwrite in sequence?
        # If we overwrite in sequence, the last one wins.
        # "CM2" condition: ...
        # "CM0" condition: ...
        # They seem mutually exclusive.
        if condition == "ELSE":
             # Apply to remaining "其他"
             pass
        else:
             df.loc[mask, "车型分类"] = code

    # 2. Refine "CM2 增程" using product_type_logic
    # Check if we have "增程" logic
    range_ext_condition = product_type_logic.get("增程")
    if range_ext_condition:
        is_range_ext = parse_sql_condition(df, range_ext_condition, product_name_col)
        # If model is CM2 AND is range ext -> CM2 增程
        cm2_mask = df["车型分类"] == "CM2"
        df.loc[cm2_mask & is_range_ext, "车型分类"] = "CM2 增程"
        
        # Update BUSINESS_GROUPS to include "CM2 增程" in "LS6" (since CM2 is in LS6)
        # We find which group contains CM2
        for grp, models in BUSINESS_GROUPS.items():
            if "CM2" in models and "CM2 增程" not in models:
                # Need to append. But models might be a list from JSON which we shouldn't mutate in place if it affects others?
                # It's a new list in our dict.
                BUSINESS_GROUPS[grp] = models + ["CM2 增程"]

    # 3. Set "业务定义" (Business Group) based on series column
    # The user said: "lock_summary.py#L17-21 的部分直接使用新数据源中的 series"
    df["业务定义"] = df[series_col].fillna("其他")
    
    # Ensure consistency between BUSINESS_GROUPS and df["业务定义"]
    # If the data has a series "XYZ" not in BUSINESS_GROUPS, we might want to add it?
    # Or strict filtering?
    # The script uses BUSINESS_GROUPS keys to filter "requested_biz".
    # So if data has "L6", BUSINESS_GROUPS must have "L6".
    # JSON mapping has "L6".
    
    # Check for any Series in data that is not in BUSINESS_GROUPS
    unique_series = df["业务定义"].unique()
    for s in unique_series:
        if s not in BUSINESS_GROUPS:
            # Dynamically add it?
            # Find what sub-models belong to this series in the data
            sub_models = df.loc[df["业务定义"] == s, "车型分类"].unique().tolist()
            BUSINESS_GROUPS[s] = sub_models

    # Re-build model_to_biz for consistency (though we used series col directly)
    model_to_biz = {}
    for grp_name, models in BUSINESS_GROUPS.items():
        for m in models:
            model_to_biz[m] = grp_name
            
    # --- End Classification Logic ---

    mask = df[lock_col].notna() & (
        (df[lock_col].dt.date >= start_date) & (df[lock_col].dt.date <= end_date)
    )

    # Retained Lock Mask: Lock time exists AND approve_refund_time is NULL
    if approve_refund_col:
        retained_mask = mask & df[approve_refund_col].isna()
    else:
        retained_mask = mask

    # --- Token Logic for Filter and Grouping ---
    wanted_tokens = [m.strip() for m in str(args.models).split(",") if m.strip()] if args.models else []
    
    # Calculate expanded_models for legacy modules (that rely on 车型分类 pivot)
    expanded_models = []
    if wanted_tokens:
        for t in wanted_tokens:
            if t in BUSINESS_GROUPS:
                expanded_models.extend(BUSINESS_GROUPS[t])
            else:
                expanded_models.append(t)
        # Remove duplicates
        expanded_models = list(set(expanded_models))

    requested_biz = [t for t in wanted_tokens if t in BUSINESS_GROUPS]
    requested_models = [t for t in wanted_tokens if t not in BUSINESS_GROUPS]
    biz_mode = True if requested_biz and not requested_models else False

    # We need to construct a mask for each token to allow flexible aggregation
    # Token can be a Series (in df["业务定义"]) or a Model (in df["车型分类"])
    token_masks = {}
    if wanted_tokens:
        for t in wanted_tokens:
            # Match either Business Group (Series) or Model Classification
            # Use strict equality for now, assuming user knows exact names
            t_mask = (df["业务定义"] == t) | (df["车型分类"] == t)
            token_masks[t] = t_mask
        
        # Combine all masks for global filtering
        if token_masks:
            import operator
            from functools import reduce
            active_filter = reduce(operator.or_, token_masks.values())
        else:
            active_filter = pd.Series(True, index=df.index)
    else:
        active_filter = pd.Series(True, index=df.index)

    lock_total = int(df.loc[mask & active_filter, order_no_col].nunique())
    retained_lock_total = int(df.loc[retained_mask & active_filter, order_no_col].nunique())

    # Helper to calculate metric per token
    def calculate_token_metric(
        base_df: pd.DataFrame, 
        tokens: List[str], 
        masks: Dict[str, pd.Series], 
        groupby_col: str, 
        metric_col: str
    ) -> pd.DataFrame:
        """
        Calculates metric for each token separately and combines them.
        Returns DataFrame with index=groupby_col and columns=tokens.
        """
        results = []
        # Get all unique groups first to ensure alignment? 
        # Or just concat (outer join) which is safer.
        
        for t in tokens:
            t_mask = masks.get(t, pd.Series(False, index=base_df.index))
            # Apply token mask to base_df
            # Note: base_df usually already has some time/status filters applied (e.g. mask & active_filter)
            # But active_filter is the UNION of all tokens.
            # So we need to apply the SPECIFIC token mask AND the common filters.
            # However, base_df passed in should be the FULL df (or filtered by time only), 
            # and we apply specific mask here?
            # Actually, the caller usually passes a filtered DF. 
            # If caller passes df[mask & active_filter], then applying t_mask again is redundant but safe 
            # ONLY IF t_mask is a subset of active_filter (which it is).
            # BUT: If row A belongs to Token 1 but not Token 2, it is in active_filter.
            # When processing Token 2, we must exclude it.
            # So yes, we must apply t_mask.
            
            # Use base_df index alignment
            subset = base_df[t_mask.reindex(base_df.index, fill_value=False)]
            
            if subset.empty:
                s = pd.Series(0, index=[])
            else:
                s = subset.groupby(groupby_col, observed=False)[metric_col].nunique()
            
            s.name = t
            results.append(s)
            
        if not results:
            return pd.DataFrame()
            
        return pd.concat(results, axis=1).fillna(0).astype(int)



    # 2) 大定相关统计（周期内、含大定支付时间）
    summary_df = None
    summary_missing_msg = None
    retained_by_date_total_df = None
    retained_by_date_pivot_df = None
    per_model_df = None
    per_model_pivot_df = None
    try:
        deposit_col = resolve_column(
            df,
            [
                "deposit_payment_time", # New data
                "Deposit_Payment_Time",
                "Deposit Payment Time",
                "大定支付时间",
                "大定时间",
                "下定时间",
            ],
        )
        df[deposit_col] = pd.to_datetime(df[deposit_col], errors="coerce")
        dmask = df[deposit_col].notna() & (
            (df[deposit_col].dt.date >= start_date) & (df[deposit_col].dt.date <= end_date)
        )
        deposit_total = int(df.loc[dmask & active_filter, order_no_col].nunique())

        try:
            refund_col = resolve_column(
                df,
                [
                    "deposit_refund_time", # New data preferred
                    "Deposit_Refund_Time",
                    "Deposit Refund Time",
                    "大定退订时间",
                    "退订时间",
                    "退款时间",
                    # Removed fallback to approve_refund_time to strictly follow user request
                ],
            )
            df[refund_col] = pd.to_datetime(df[refund_col], errors="coerce")
            deposit_refund_total = int(
                df.loc[dmask & active_filter & df[refund_col].notna(), order_no_col].nunique()
            )
            deposit_retained_total = int(
                df.loc[dmask & active_filter & df[refund_col].isna() & df[lock_col].isna(), order_no_col].nunique()
            )
            _retained_mask = dmask & active_filter & df[refund_col].isna() & df[lock_col].isna()
            _retained_df = df.loc[_retained_mask, [deposit_col, order_no_col, "车型分类"]].copy()
            _retained_df["deposit_date"] = _retained_df[deposit_col].dt.date
            _retained_grp = (
                _retained_df.groupby(["deposit_date", "车型分类"]).agg(retained_orders=(order_no_col, pd.Series.nunique)).reset_index()
            )
            retained_by_date_pivot_df = (
                _retained_grp.pivot_table(index="deposit_date", columns="车型分类", values="retained_orders", fill_value=0, aggfunc="sum", observed=False).sort_index()
            )
            if expanded_models:
                cols = [m for m in expanded_models if m in retained_by_date_pivot_df.columns]
                if cols:
                    retained_by_date_pivot_df = retained_by_date_pivot_df[cols]
            retained_by_date_total_df = (
                _retained_df.groupby(["deposit_date"]).agg(retained_orders=(order_no_col, pd.Series.nunique)).reset_index().sort_values("deposit_date")
            )

            # 分车型概览统计（仅在指定多个车型时输出）
            if (not biz_mode) and (requested_models and len(requested_models) >= 2):
                rows = []
                for m in requested_models:
                    m_filter = df["车型分类"] == m
                    rows.append(
                        {
                            "车型分类": m,
                            "锁单总数": int(df.loc[mask & active_filter & m_filter, order_no_col].nunique()),
                            "留存锁单总数": int(df.loc[retained_mask & active_filter & m_filter, order_no_col].nunique()),
                            "大定总数": int(df.loc[dmask & active_filter & m_filter, order_no_col].nunique()),
                            "大定留存总数": int(df.loc[dmask & active_filter & m_filter & df[refund_col].isna() & df[lock_col].isna(), order_no_col].nunique()),
                            "大定退订数": int(df.loc[dmask & active_filter & m_filter & df[refund_col].notna(), order_no_col].nunique()),
                        }
                    )
                per_model_df = pd.DataFrame(rows)
        except KeyError as e:
            deposit_refund_total = 0
            deposit_retained_total = int(
                df.loc[dmask & active_filter & df[lock_col].isna(), order_no_col].nunique()
            )
            summary_missing_msg = f"字段缺失：{e}"
            _retained_mask = dmask & active_filter & df[lock_col].isna()
            _retained_df = df.loc[_retained_mask, [deposit_col, order_no_col, "车型分类"]].copy()
            _retained_df["deposit_date"] = _retained_df[deposit_col].dt.date
            _retained_grp = (
                _retained_df.groupby(["deposit_date", "车型分类"]).agg(retained_orders=(order_no_col, pd.Series.nunique)).reset_index()
            )
            retained_by_date_pivot_df = (
                _retained_grp.pivot_table(index="deposit_date", columns="车型分类", values="retained_orders", fill_value=0, aggfunc="sum").sort_index()
            )
            if expanded_models:
                cols = [m for m in expanded_models if m in retained_by_date_pivot_df.columns]
                if cols:
                    retained_by_date_pivot_df = retained_by_date_pivot_df[cols]
            retained_by_date_total_df = (
                _retained_df.groupby(["deposit_date"]).agg(retained_orders=(order_no_col, pd.Series.nunique)).reset_index().sort_values("deposit_date")
            )

            if (not biz_mode) and (requested_models and len(requested_models) >= 2):
                rows = []
                for m in requested_models:
                    m_filter = df["车型分类"] == m
                    rows.append(
                        {
                            "车型分类": m,
                            "锁单总数": int(df.loc[mask & active_filter & m_filter, order_no_col].nunique()),
                            "留存锁单总数": int(df.loc[retained_mask & active_filter & m_filter, order_no_col].nunique()),
                            "大定总数": int(df.loc[dmask & active_filter & m_filter, order_no_col].nunique()),
                            "大定留存总数": int(df.loc[dmask & active_filter & m_filter & df[lock_col].isna(), order_no_col].nunique()),
                            "大定退订数": 0,
                        }
                    )
                per_model_df = pd.DataFrame(rows)

        summary_df = pd.DataFrame(
            {
                "指标": ["锁单总数", "留存锁单总数", "大定总数", "大定留存总数", "大定退订数"],
                "数量": [lock_total, retained_lock_total, deposit_total, deposit_retained_total, deposit_refund_total],
            }
        )
        if per_model_df is not None and not per_model_df.empty:
            _pivot = per_model_df.set_index("车型分类").T
            _pivot.index.name = "指标"
            per_model_pivot_df = _pivot.reset_index()
    except KeyError as e:
        # 无大定支付时间字段，仅输出锁单总数并提示缺失字段
        summary_missing_msg = f"字段缺失：{e}"
        summary_df = pd.DataFrame({"指标": ["锁单总数", "留存锁单总数"], "数量": [lock_total, retained_lock_total]})
    
    # --- Pivot Logic ---
    df_period = df.loc[retained_mask & active_filter, [lock_col, region_col, "车型分类", "业务定义", order_no_col]].copy()
    
    grouped = (
        df_period.groupby([region_col, "车型分类"], observed=False).agg(订单数=(order_no_col, pd.Series.nunique)).reset_index()
    )
    grouped.rename(columns={region_col: "Parent Region Name"}, inplace=True)
    pivot_df = grouped.pivot_table(
        index="Parent Region Name",
        columns="车型分类",
        values="订单数",
        fill_value=0,
        aggfunc="sum",
        observed=False,
    )
    pivot_df = pivot_df.sort_index().sort_index(axis=1)

    if args.models:
        wanted = [m.strip() for m in str(args.models).split(",") if m.strip()]
        cols = [c for c in wanted if c in pivot_df.columns]
        if cols:
            pivot_df = pivot_df[cols]

    percent_df = pivot_df.div(pivot_df.sum(axis=0).replace(0, pd.NA), axis=1).fillna(0) * 100
    percent_df = percent_df.round(0).astype(int).astype(str) + "%"

    md_lines = []
    md_lines.append("# 锁单分车型分大区汇总")
    md_lines.append("")
    md_lines.append(f"- 源文件: `{input_path}`")
    md_lines.append(f"- 时间区间: `{args.start}` ~ `{args.end}`")
    md_lines.append("")
    if section_enabled("overview"):
        if per_model_pivot_df is not None:
            md_lines.append("## 概览统计（分车型）")
            md_lines.append(df_to_md(per_model_pivot_df))
            md_lines.append("")
        else:
            md_lines.append("## 概览统计")
            md_lines.append(df_to_md(summary_df))
            if summary_missing_msg:
                md_lines.append("")
                md_lines.append(f"缺失字段提示：{summary_missing_msg}")
            md_lines.append("")
    if section_enabled("deposit_daily"):
        if (not biz_mode) and (requested_models and len(requested_models) >= 2):
            md_lines.append("## 大定留存的 Deposit_Payment_Time 分布（按日，分车型）")
            if retained_by_date_pivot_df is not None and not retained_by_date_pivot_df.empty:
                md_lines.append(df_to_md(retained_by_date_pivot_df.reset_index()))
            else:
                md_lines.append("(空表，无数据)")
        else:
            md_lines.append("## 大定留存的 Deposit_Payment_Time 分布（按日）")
            if retained_by_date_total_df is not None and not retained_by_date_total_df.empty:
                md_lines.append(df_to_md(retained_by_date_total_df))
            else:
                md_lines.append("(空表，无数据)")
        md_lines.append("")
    preview_pivot = None
    preview_percent = None
    if biz_mode:
        if section_enabled("region_biz_matrix"):
            grouped_biz = (
                df_period.groupby([region_col, "业务定义"], observed=False).agg(订单数=(order_no_col, pd.Series.nunique)).reset_index()
            )
            grouped_biz.rename(columns={region_col: "Parent Region Name"}, inplace=True)
            pivot_biz = grouped_biz.pivot_table(
                index="Parent Region Name",
                columns="业务定义",
                values="订单数",
                fill_value=0,
                aggfunc="sum",
                observed=False,
            ).sort_index().sort_index(axis=1)
            if requested_biz:
                cols_biz = [c for c in requested_biz if c in pivot_biz.columns]
                if cols_biz:
                    pivot_biz = pivot_biz[cols_biz]
            percent_biz = pivot_biz.div(pivot_biz.sum(axis=0).replace(0, pd.NA), axis=1).fillna(0) * 100
            percent_biz = percent_biz.round(0).astype(int).astype(str) + "%"
            md_lines.append("## 区域 x 业务定义矩阵")
            md_lines.append(df_to_md(pivot_biz.reset_index()))
            md_lines.append("")
            md_lines.append("## 分 region 占比（%）（按业务定义列归一化）")
            md_lines.append(df_to_md(percent_biz.reset_index()))
            preview_pivot = pivot_biz
            preview_percent = percent_biz
    else:
        if section_enabled("region_model_matrix"):
            md_lines.append("## 区域 x 车型矩阵")
            md_lines.append(df_to_md(pivot_df.reset_index()))
            md_lines.append("")
            md_lines.append("## 分 region 占比（%）（按车型列归一化）")
            md_lines.append(df_to_md(percent_df.reset_index()))
            preview_pivot = pivot_df
            preview_percent = percent_df
    
    # Always output biz breakdown
    if section_enabled("region_biz_matrix"):
        grouped_biz = (
            df_period.groupby([region_col, "业务定义"], observed=False).agg(订单数=(order_no_col, pd.Series.nunique)).reset_index()
        )
        grouped_biz.rename(columns={region_col: "Parent Region Name"}, inplace=True)
        pivot_biz = grouped_biz.pivot_table(
            index="Parent Region Name",
            columns="业务定义",
            values="订单数",
            fill_value=0,
            aggfunc="sum",
            observed=False,
        )
        pivot_biz = pivot_biz.sort_index().sort_index(axis=1)
        percent_biz = pivot_biz.div(pivot_biz.sum(axis=0).replace(0, pd.NA), axis=1).fillna(0) * 100
        percent_biz = percent_biz.round(0).astype(int).astype(str) + "%"
        md_lines.append("")
        md_lines.append("## 区域 x 业务定义矩阵")
        md_lines.append(df_to_md(pivot_biz.reset_index()))
        md_lines.append("")
        md_lines.append("## 分 region 占比（%）（按业务定义列归一化）")
        md_lines.append(df_to_md(percent_biz.reset_index()))

    try:
        level_col = resolve_column(
            df,
            [
                "License City Level",
                "license_city_level",
                "License City Tier",
                "city_level",
                "City Level",
                "城市级别",
                "城市等级",
                "上牌城市级别",
                "上牌城市等级",
            ],
        )
        level_series = df.loc[retained_mask & active_filter, level_col]
        
        # 统计剔除前的“未知”情况（NaN 或 "nan"）
        total_count_l = len(level_series)
        def is_unknown_level(val):
            if pd.isna(val):
                return True
            s = str(val).strip().lower()
            return s == "nan" or s == "null" or s == ""

        unknown_level_mask = level_series.apply(is_unknown_level).astype(bool)
        unknown_l_count = unknown_level_mask.sum()
        unknown_l_pct = (unknown_l_count / total_count_l * 100) if total_count_l > 0 else 0.0

        # 剔除无效值
        level_series_valid = level_series[~unknown_level_mask].astype(str).str.strip()

        level_counts = level_series_valid.value_counts()
        total_orders = len(level_series_valid)
        level_share = (level_counts / max(total_orders, 1) * 100).round(0).astype(int).astype(str) + "%"
        level_df = (
            pd.DataFrame({"level": level_counts.index, "lock_orders": level_counts.values, "share_pct": level_share.values})
            .sort_values(["lock_orders", "share_pct"], ascending=[False, False])
        )
        if section_enabled("city_level"):
            md_lines.append("")
            md_lines.append("## 分 license_city_level 的锁单量与占比")
            md_lines.append(f"> 注：已剔除无效/空值样本 {unknown_l_count} 个（占比 {unknown_l_pct:.2f}%），下表基于有效样本 {total_orders} 个统计。")
            md_lines.append(df_to_md(level_df))
        
        if section_enabled("city_level"):
            level_sub_df = df.loc[retained_mask & active_filter, [level_col, "车型分类", order_no_col]].copy()
            # 过滤无效/未知值
            level_sub_df["__is_unknown__"] = level_sub_df[level_col].apply(is_unknown_level).astype(bool)
            level_sub_df = level_sub_df[~level_sub_df["__is_unknown__"]].drop(columns=["__is_unknown__"])
            if not level_sub_df.empty:
                if args.models:
                    pivot_level_model = calculate_token_metric(
                        level_sub_df, 
                        wanted_tokens, 
                        token_masks, 
                        level_col, 
                        order_no_col
                    )
                    # Align with overall level counts order
                    pivot_level_model = pivot_level_model.reindex(level_counts.index).fillna(0).astype(int)
                else:
                    level_counts_model = (
                        level_sub_df.groupby([level_col, "车型分类"], observed=False)
                        .agg(订单数=(order_no_col, pd.Series.nunique))
                        .reset_index()
                    )
                    pivot_level_model = level_counts_model.pivot_table(
                        index=level_col,
                        columns="车型分类",
                        values="订单数",
                        aggfunc="sum",
                        fill_value=0,
                        observed=False,
                    )

                pivot_level_pct = pivot_level_model.div(pivot_level_model.sum(axis=0).replace(0, pd.NA), axis=1).fillna(0) * 100
                pivot_level_pct = pivot_level_pct.round(0).astype(int).astype(str) + "%"
                
                # Only show if we have columns
                if not pivot_level_model.empty:
                    md_lines.append("")
                    md_lines.append("## 分 license_city_level 的锁单量（分车型占比%）")
                    md_lines.append(df_to_md(pivot_level_pct.reset_index()).replace("nan%", "0%"))
    except KeyError as e:
        md_lines.append("")
        md_lines.append("## 分 license_city_level 的锁单量与占比")
        md_lines.append(f"字段缺失：{e}")

    # 追加：分 License Province 的锁单量与占比
    try:
        prov_col = resolve_column(
            df,
            [
                "License Province",
                "license_province",
                "License Province Name",
                "Province",
                "province",
                "上牌省份",
                "上牌省",
                "省份",
            ],
        )
        prov_series = df.loc[retained_mask & active_filter, prov_col].astype(str).fillna("未知")
        prov_counts = prov_series.value_counts()
        total_orders_p = int(prov_series.size)
        prov_share = (prov_counts / max(total_orders_p, 1) * 100).round(0).astype(int).astype(str) + "%"
        prov_df = (
            pd.DataFrame({"province": prov_counts.index, "lock_orders": prov_counts.values, "share_pct": prov_share.values})
            .sort_values(["lock_orders", "share_pct"], ascending=[False, False])
        )
        if section_enabled("province"):
            md_lines.append("")
            md_lines.append("## 分 License Province 的锁单量与占比")
            md_lines.append(df_to_md(prov_df))
        
        if section_enabled("province"):
            prov_sub_df = df.loc[retained_mask & active_filter, [prov_col, "车型分类", order_no_col, "业务定义"]].copy()
            if not prov_sub_df.empty:
                if args.models:
                    pivot_prov_model = calculate_token_metric(
                        prov_sub_df, 
                        wanted_tokens, 
                        token_masks, 
                        prov_col, 
                        order_no_col
                    )
                    pivot_prov_model = pivot_prov_model.reindex(prov_counts.index).fillna(0).astype(int)
                else:
                    prov_counts_model = (
                        prov_sub_df.groupby([prov_col, "车型分类"], observed=False)
                        .agg(订单数=(order_no_col, pd.Series.nunique))
                        .reset_index()
                    )
                    pivot_prov_model = prov_counts_model.pivot_table(
                        index=prov_col,
                        columns="车型分类",
                        values="订单数",
                        aggfunc="sum",
                        fill_value=0,
                        observed=False,
                    )
                
                pivot_prov_pct = pivot_prov_model.div(pivot_prov_model.sum(axis=0).replace(0, pd.NA), axis=1).fillna(0) * 100
                pivot_prov_pct = pivot_prov_pct.round(0).astype(int).astype(str) + "%"
                
                if not pivot_prov_model.empty:
                    md_lines.append("")
                    md_lines.append("## 分 License Province 的锁单量（分车型占比%）")
                    md_lines.append(df_to_md(pivot_prov_pct.reset_index()).replace("nan%", "0%"))
    except KeyError as e:
        md_lines.append("")
        md_lines.append("## 分 License Province 的锁单量与占比")
        md_lines.append(f"字段缺失：{e}")

    # 追加：分 License City 的锁单量与占比（Top 10）
    try:
        city_col = resolve_column(
            df,
            [
                "License City",
                "license_city",
                "License City Name",
                "Store City",
                "store_city",
                "城市",
                "上牌城市",
                "上牌市",
            ],
        )
        city_series = df.loc[mask & active_filter, city_col].astype(str).fillna("未知")
        city_counts = city_series.value_counts()
        total_orders_c = int(city_series.size)
        city_share = (city_counts / max(total_orders_c, 1) * 100).round(0).astype(int).astype(str) + "%"
        city_df = (
            pd.DataFrame({"city": city_counts.index, "lock_orders": city_counts.values, "share_pct": city_share.values})
            .sort_values(["lock_orders", "share_pct"], ascending=[False, False])
            .head(10)
        )
        if section_enabled("city_top10"):
            md_lines.append("")
            md_lines.append("## 分 License City 的锁单量与占比（Top 10）")
            md_lines.append(df_to_md(city_df))
        
        if section_enabled("city_top10"):
            if args.models:
                # Combined table for requested tokens
                city_sub_df = df.loc[mask & active_filter, [city_col, "车型分类", order_no_col, "业务定义"]].copy()
                pivot_city_model = calculate_token_metric(
                    city_sub_df, 
                    wanted_tokens, 
                    token_masks, 
                    city_col, 
                    order_no_col
                )
                # Reindex to Top 10 cities (by total)
                top10_cities = city_df["city"].tolist()
                pivot_city_model = pivot_city_model.reindex(top10_cities).fillna(0).astype(int)
                
                # Add percentages? Or just counts? User usually wants counts for Top 10.
                # Let's show counts.
                if not pivot_city_model.empty:
                    md_lines.append("")
                    md_lines.append("## 分 License City 的锁单量（Top 10 Cities Breakdown）")
                    md_lines.append(df_to_md(pivot_city_model.reset_index()))
            
            else:
                # Original logic: Loop over models found in data
                models_for_cross = list(df.loc[mask & active_filter, "车型分类"].unique())
                if models_for_cross:
                    for m in models_for_cross:
                        sub_series = df.loc[mask & (df["车型分类"] == m), city_col].astype(str).fillna("未知")
                        if sub_series.empty:
                            continue
                        sub_counts = sub_series.value_counts()
                        sub_total = int(sub_series.size)
                        sub_share = (sub_counts / max(sub_total, 1) * 100).round(0).astype(int).astype(str) + "%"
                        sub_df = (
                            pd.DataFrame({"city": sub_counts.index, "lock_orders": sub_counts.values, "share_pct": sub_share.values})
                            .sort_values(["lock_orders", "share_pct"], ascending=[False, False])
                            .head(10)
                        )
                        md_lines.append("")
                        md_lines.append(f"## 分 License City 的锁单量与占比（Top 10，{m}）")
                        md_lines.append(df_to_md(sub_df))
    except KeyError as e:
        md_lines.append("")
        md_lines.append("## 分 License City 的锁单量与占比（Top 10）")
        md_lines.append(f"字段缺失：{e}")

    # 追加：分年龄段的锁单量与占比
    try:
        age_col = resolve_column(
            df,
            [
                "owner_age",
                "Owner Age",
                "Owner_Age",
                "age",
                "Age",
                "车主年龄",
            ],
        )
        
        # 提取包含年龄、车型、订单号的子集
        age_sub_df = df.loc[mask & active_filter, [age_col, "车型分类", order_no_col, region_col]].copy()
        age_sub_df["age_raw"] = pd.to_numeric(age_sub_df[age_col], errors="coerce")
        
        # 1. 全局统计（均值/中位数等）
        age_series_raw = age_sub_df["age_raw"]
        total_count_age_raw = len(age_series_raw)
        
        out_of_range_mask = age_series_raw.notna() & ~((age_series_raw >= 16) & (age_series_raw <= 85))
        out_of_range_count = int(out_of_range_mask.sum())
        out_of_range_pct = (out_of_range_count / total_count_age_raw * 100) if total_count_age_raw > 0 else 0.0
        
        # 仅保留有效区间内的用于统计数值特征
        valid_ages_stats = age_series_raw.where((age_series_raw >= 16) & (age_series_raw <= 85)).dropna()
        
        if not valid_ages_stats.empty:
            if section_enabled("age_stats"):
                age_mean = valid_ages_stats.mean()
                age_median = valid_ages_stats.median()
                age_std = valid_ages_stats.std()
                md_lines.append("")
                md_lines.append("## 车主年龄统计")
                md_lines.append(f"- 平均值: {age_mean:.2f}")
                md_lines.append(f"- 中位数: {age_median:.2f}")
                md_lines.append(f"- 标准差: {age_std:.2f}")
                md_lines.append(f"> 注：已根据区间过滤剔除不在 [16,85] 的样本 {out_of_range_count} 个（占比 {out_of_range_pct:.2f}%）。")
                if (not biz_mode) and (requested_models and len(requested_models) >= 2):
                    age_valid_df = age_sub_df[
                        age_sub_df["age_raw"].notna()
                        & (age_sub_df["age_raw"] >= 16)
                        & (age_sub_df["age_raw"] <= 85)
                    ].copy()
                    if not age_valid_df.empty:
                        per_model_age = (
                            age_valid_df.groupby("车型分类")["age_raw"]
                            .agg(["mean", "median", "std"])
                            .round(2)
                        )
                        if args.models:
                            wanted = [m.strip() for m in str(args.models).split(",") if m.strip()]
                            cols = [m for m in wanted if m in per_model_age.index]
                            if cols:
                                per_model_age = per_model_age.loc[cols]
                        per_model_age = per_model_age.rename(
                            columns={"mean": "平均值", "median": "中位数", "std": "标准差"}
                        ).T
                        per_model_age.index.name = "指标"
                        md_lines.append("")
                        md_lines.append("## 车主年龄统计（分车型）")
                        md_lines.append(df_to_md(per_model_age.reset_index()))
        else:
            if section_enabled("age_stats"):
                md_lines.append("")
                md_lines.append("## 车主年龄统计")
                md_lines.append("(无有效年龄数据)")
                md_lines.append(f"> 注：已根据区间过滤剔除不在 [16,85] 的样本 {out_of_range_count} 个（占比 {out_of_range_pct:.2f}%）。")

        # 修正：使用 start_date 的年份作为基准
        current_year = start_date.year

        def map_age_group(age):
            if pd.isna(age):
                return "未知"
            birth_year = current_year - age
            if birth_year >= 2000:
                return "00后"
            elif birth_year >= 1995:
                return "95后"
            elif birth_year >= 1990:
                return "90后"
            elif birth_year >= 1985:
                return "85后"
            elif birth_year >= 1980:
                return "80后"
            elif birth_year >= 1975:
                return "75后"
            elif birth_year >= 1970:
                return "70后"
            else:
                return "70前"

        # 2. 分组分布（分车型）
        # 先处理区间外的值：设为 NaN，以便 map_age_group 映射为 "未知"
        # 注意：之前的 valid_ages_stats 已经 dropna 了，这里我们要对 DataFrame 操作
        age_sub_df["age_clean"] = age_sub_df["age_raw"].where(
            (age_sub_df["age_raw"] >= 16) & (age_sub_df["age_raw"] <= 85), pd.NA
        )
        age_sub_df["age_group"] = age_sub_df["age_clean"].apply(map_age_group)
        
        # 统计剔除前的“未知”情况（包含原始NaN和区间外的）
        raw_na_count = age_sub_df["age_raw"].isna().sum()
        total_excluded = raw_na_count + out_of_range_count
        total_excluded_pct = (total_excluded / total_count_age_raw * 100) if total_count_age_raw > 0 else 0.0
        
        # 过滤掉 "未知"
        valid_group_df = age_sub_df[age_sub_df["age_group"] != "未知"].copy()
        valid_sample_count = len(valid_group_df)
        
        # --- A. 总体分布 (Backward Compatibility) ---
        group_counts = valid_group_df["age_group"].value_counts()
        total_orders_age = len(valid_group_df)
        age_share = (group_counts / max(total_orders_age, 1) * 100).round(0).astype(int).astype(str) + "%"

        age_df = pd.DataFrame({
            "age_group": group_counts.index,
            "lock_orders": group_counts.values,
            "share_pct": age_share.values
        })
        sort_order = ["00后", "95后", "90后", "85后", "80后", "75后", "70后", "70前"]
        age_df["age_group"] = pd.Categorical(age_df["age_group"], categories=sort_order, ordered=True)
        age_df = age_df.sort_values("age_group")

        if section_enabled("age_distribution"):
            md_lines.append("")
            md_lines.append("## 分年龄段的锁单量与占比")
            md_lines.append(f"> 注：已剔除年龄未知或不在[16,85]区间的样本共 {total_excluded} 个（占比 {total_excluded_pct:.2f}%），下表基于有效样本 {valid_sample_count} 个统计。")
            md_lines.append(df_to_md(age_df))

        # --- B. 分车型分布 (New Feature) ---
        if not valid_group_df.empty and section_enabled("age_model_pct"):
            age_pivot = valid_group_df.pivot_table(
                index="age_group",
                columns="车型分类",
                values=order_no_col,
                aggfunc="nunique",
                fill_value=0,
                observed=False,
            )
            age_pivot = age_pivot.reindex(sort_order)
            if args.models:
                wanted = [m.strip() for m in str(args.models).split(",") if m.strip()]
                cols = [c for c in wanted if c in age_pivot.columns]
                if cols:
                    age_pivot = age_pivot[cols]
            age_pct_df = age_pivot.div(age_pivot.sum(axis=0).replace(0, pd.NA), axis=1).fillna(0) * 100
            age_pct_df = age_pct_df.round(0).astype(int).astype(str) + "%"
            md_lines.append("")
            md_lines.append("## 分年龄段的锁单量与占比（分车型占比%）")
            md_lines.append(df_to_md(age_pct_df.reset_index()))
            
            # Regional breakdown by age
            age_region_df = valid_group_df[[ "age_group", "车型分类", region_col, order_no_col ]].copy()
            age_region_df = age_region_df[age_region_df[region_col].notna()]
            if not age_region_df.empty:
                age_region_counts = (
                    age_region_df.groupby(["age_group", region_col, "车型分类"], observed=False)
                    .agg(订单数=(order_no_col, pd.Series.nunique))
                    .reset_index()
                )
                models_for_cross = []
                if requested_models:
                    models_for_cross = [m for m in requested_models if m in age_region_counts["车型分类"].unique()]
                else:
                    models_for_cross = list(age_region_counts["车型分类"].unique())
                for m in models_for_cross:
                    sub = age_region_counts[age_region_counts["车型分类"] == m]
                    if sub.empty:
                        continue
                    pivot_counts = sub.pivot_table(
                        index="age_group",
                        columns=region_col,
                        values="订单数",
                        aggfunc="sum",
                        fill_value=0,
                        observed=False,
                    )
                    pivot_counts = pivot_counts.reindex(sort_order)
                    row_sum = pivot_counts.sum(axis=1).replace(0, pd.NA)
                    pivot_pct = pivot_counts.div(row_sum, axis=0).fillna(0) * 100
                    pivot_pct = pivot_pct.round(0).astype(int).astype(str) + "%"
                    if section_enabled("age_region_pct_by_model"):
                        md_lines.append("")
                        md_lines.append(f"## 分年龄段的锁单在不同区域的占比（{m}，按年龄段行归一化）")
                        md_lines.append(df_to_md(pivot_pct.reset_index()))

    except KeyError as e:
        md_lines.append("")
        md_lines.append("## 分年龄段的锁单量与占比")
        md_lines.append(f"字段缺失：{e}")

    # 追加：分性别的锁单量与占比
    try:
        gender_col = resolve_column(
            df,
            [
                "owner_gender",
                "Owner Gender",
                "Owner_Gender",
                "gender",
                "Gender",
                "车主性别",
                "性别",
            ],
        )
        raw_gender_series = df.loc[retained_mask & active_filter, gender_col]

        def is_unknown_gender(val):
            if pd.isna(val):
                return True
            s = str(val).strip()
            if not s:  # empty string
                return True
            s_lower = s.lower()
            if s_lower in ["默认未知", "none", "nan", "null", "未知"]:
                return True
            return False

        # 识别未知/无效值
        unknown_mask = raw_gender_series.apply(is_unknown_gender)
        
        # 统计
        total_count_g = len(raw_gender_series)
        unknown_g_count = unknown_mask.sum()
        unknown_g_pct = (unknown_g_count / total_count_g * 100) if total_count_g > 0 else 0.0

        # 提取有效值
        gender_series_valid = raw_gender_series[~unknown_mask].astype(str).str.strip()
        
        gender_counts = gender_series_valid.value_counts()
        total_orders_gender = len(gender_series_valid)
        gender_share = (gender_counts / max(total_orders_gender, 1) * 100).round(0).astype(int).astype(str) + "%"

        gender_df = (
            pd.DataFrame({"gender": gender_counts.index, "lock_orders": gender_counts.values, "share_pct": gender_share.values})
            .sort_values(["lock_orders", "share_pct"], ascending=[False, False])
        )

        if section_enabled("gender"):
            md_lines.append("")
            md_lines.append("## 分性别的锁单量与占比")
            md_lines.append(f"> 注：已剔除性别为“默认未知/None/空”的样本 {unknown_g_count} 个（占比 {unknown_g_pct:.2f}%），下表基于有效样本 {total_orders_gender} 个统计。")
            md_lines.append(df_to_md(gender_df))

    except KeyError as e:
        md_lines.append("")
        md_lines.append("## 分性别的锁单量与占比")
        md_lines.append(f"字段缺失：{e}")

    report_name = f"lock_summary_{args.start}_to_{args.end}.md"
    report_path = OUT_DIR / report_name
    report_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Report saved: {report_path}")
    print("\n预览：")
    if preview_pivot is not None:
        print(df_to_md(preview_pivot.reset_index()))
    print("\n占比预览：")
    if preview_percent is not None:
        print(df_to_md(preview_percent.reset_index()))

    if args.send:
        print("\nSending report to Feishu...")
        import subprocess
        import sys
        
        script_dir = Path(__file__).parent
        send_script = script_dir / "send_lock_summary_to_feishu.py"
        
        if send_script.exists():
            try:
                subprocess.run(
                    [sys.executable, str(send_script), "--file", str(report_path)],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error sending report: {e}")
        else:
            print(f"Warning: Send script not found at {send_script}")


if __name__ == "__main__":
    main()
