#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import numpy as np
    import pandas as pd
    from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
except Exception:
    raise RuntimeError("请先安装依赖：pip install pandas pyarrow numpy")


TARGET_PARQUET = Path(__file__).parent.parent / "formatted" / "order_data.parquet"
OUTPUT_MD = Path(__file__).parent / "order_data_schema_with_examples.md"

BUSINESS_MEANING = {
    "apply_refund_time": "退订提交时间",
    "deposit_refund_time": "大订退订时间",
    "finance_product": "金融产品",
    "final_payment_time": "最终支付时间",
    "intention_refund_time": "小订退订时间",
    "first_test_drive_time": "首次试驾时间",
    "actual_refund_time": "实际退订时间",
    "invoice_upload_time": "开票时间",
    "delivery_date": "交付时间",
    "vin": "车架号",
    "final_payment_way": "付款方式",
    "intention_payment_time": "小订支付时间",
    "lock_time": "锁单时间",
    "deposit_payment_time": "大订支付时间",
    "owner_age": "车主年龄",
    "owner_identity_no": "车主身份证号",
    "buyer_identity_no": "买家身份证号",
    "buyer_age": "买家年龄",
    "owner_gender": "车主性别",
    "order_type": "订单类型",
    "first_touch_time": "首触时间",
    "first_assign_time": "线索首次下发时间",
    "order_gender": "订单性别",
    "license_city": "上牌城市",
    "store_city": "门店城市",
    "parent_region_name": "大区名称",
    "store_name": "门店名称",
    "store_create_date": "门店创建时间",
}


def _file_size_mb(p: Path) -> float:
    try:
        return round(p.stat().st_size / (1024 * 1024), 2)
    except Exception:
        return float("nan")


def _identify_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric_cols: List[str] = []
    date_cols: List[str] = []
    categorical_cols: List[str] = []

    for c in df.columns:
        s = df[c]
        if is_datetime64_any_dtype(s):
            date_cols.append(c)
        elif is_numeric_dtype(s):
            numeric_cols.append(c)
        else:
            lc = str(c).lower()
            if any(k in lc for k in ["time", "date", "时间", "日期"]):
                date_cols.append(c)
            else:
                categorical_cols.append(c)

    return {"numeric": numeric_cols, "categorical": categorical_cols, "date": date_cols}


def _completeness(df: pd.DataFrame) -> float:
    total = df.shape[0] * df.shape[1]
    non_null = int(df.notna().sum().sum())
    return round(100.0 * non_null / total, 2) if total > 0 else float("nan")


def _read_parquet(file_path: Path) -> pd.DataFrame:
    return pd.read_parquet(file_path)


def _stringify_value(v: object) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "null"
    if isinstance(v, (np.datetime64, pd.Timestamp)):
        ts = pd.to_datetime(v)
        if pd.isna(ts):
            return "null"
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts.isoformat(sep=" ", timespec="seconds")
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (np.floating, float)):
        vv = float(v)
        if math.isfinite(vv) and abs(vv - round(vv)) < 1e-9:
            return str(int(round(vv)))
        return str(vv)
    s = str(v)
    s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    if len(s) > 80:
        s = s[:77] + "..."
    return s


def _sample_examples(series: pd.Series, max_examples: int = 5) -> List[str]:
    try:
        s = series.dropna()
        if s.empty:
            return []
        if is_datetime64_any_dtype(s):
            s = pd.to_datetime(s, errors="coerce").dropna()
            if s.empty:
                return []
            uniq = s.drop_duplicates().sort_values()
            picks = uniq.iloc[:max_examples].tolist()
            return [_stringify_value(v) for v in picks]
        if is_numeric_dtype(s):
            s_num = pd.to_numeric(s, errors="coerce").dropna()
            if s_num.empty:
                return []
            uniq = pd.Series(s_num.unique())
            uniq = uniq.dropna().sort_values()
            picks = uniq.iloc[:max_examples].tolist()
            return [_stringify_value(v) for v in picks]
        uniq = s.astype("string").drop_duplicates()
        picks = uniq.iloc[:max_examples].tolist()
        return [_stringify_value(v) for v in picks]
    except Exception:
        return []


def analyze_dataset(file_path: Path) -> Dict:
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    print(f"正在读取文件: {file_path} ...")
    df = _read_parquet(file_path)
    print("读取成功，开始分析...")

    types = _identify_types(df)
    comp = _completeness(df)
    dup_count = int(df.duplicated().sum())

    missing_counts = df.isna().sum()
    missing = missing_counts[missing_counts > 0].sort_values(ascending=False).to_dict()
    missing_percent = {
        k: round(100.0 * v / df.shape[0], 2) if df.shape[0] > 0 else float("nan")
        for k, v in missing.items()
    }

    target_time_cols = ["order_create_date", "lock_time", "invoice_upload_time"]
    time_max_values = {}
    for col in target_time_cols:
        if col in df.columns:
            try:
                max_val = df[col].max()
                time_max_values[col] = str(max_val) if pd.notna(max_val) else "None"
            except Exception:
                time_max_values[col] = "Error"
        else:
            time_max_values[col] = "Column Not Found"

    col_details: List[Dict[str, object]] = []
    for c in df.columns:
        s = df[c]
        dtype = str(s.dtype)
        nulls = int(s.isna().sum())
        non_null = int(s.notna().sum())
        null_pct = round(100.0 * nulls / df.shape[0], 2) if df.shape[0] > 0 else float("nan")
        examples = _sample_examples(s, max_examples=5)
        meaning = BUSINESS_MEANING.get(str(c), "")
        col_details.append(
            {
                "name": str(c),
                "dtype": dtype,
                "nulls": nulls,
                "non_null": non_null,
                "null_pct": null_pct,
                "examples": examples,
                "meaning": meaning,
            }
        )

    return {
        "path": str(file_path),
        "size_mb": _file_size_mb(file_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "shape": (df.shape[0], df.shape[1]),
        "columns": list(df.columns),
        "types": types,
        "completeness": comp,
        "duplicates": dup_count,
        "missing": missing,
        "missing_percent": missing_percent,
        "time_max_values": time_max_values,
        "col_details": col_details,
    }


def _format_missing_table(stats: Dict) -> List[str]:
    lines: List[str] = []
    if not stats["missing"]:
        lines.append("✅ 未发现缺失值")
        return lines
    lines.append("")
    lines.append("### 缺失值异常")
    lines.append("")
    lines.append("| 列名 | 缺失数量 | 缺失比例 | 业务释义 |")
    lines.append("|------|----------|----------|----------|")
    for col, cnt in stats["missing"].items():
        pct = stats["missing_percent"].get(col, float("nan"))
        pct_str = f"{pct:.2f}%" if not math.isnan(pct) else "NA"
        meaning = BUSINESS_MEANING.get(str(col), "")
        lines.append(f"| {col} | {cnt} | {pct_str} | {meaning} |")
    return lines


def format_md_for_dataset(title: str, stats: Dict) -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## 数据概览")
    lines.append(f"- **数据文件**: {stats['path']}")
    lines.append(f"- **生成时间**: {stats['generated_at']}")
    if not math.isnan(stats["size_mb"]):
        lines.append(f"- **文件大小**: {stats['size_mb']:.2f} MB")
    lines.append("")

    lines.append("## 数据基本信息")
    lines.append(f"- **数据形状**: {stats['shape'][0]} 行 × {stats['shape'][1]} 列")
    lines.append(f"- **数据完整性**: {stats['completeness']:.2f}%")
    lines.append(f"- **重复行数**: {stats['duplicates']}")
    lines.append("")

    if "time_max_values" in stats:
        lines.append("## 关键时间字段最大值")
        for col, val in stats["time_max_values"].items():
            lines.append(f"- **{col}**: {val}")
        lines.append("")

    lines.append("## 数据类型分布")
    lines.append(f"- **数值列**: {len(stats['types']['numeric'])} 个")
    lines.append(f"- **分类列**: {len(stats['types']['categorical'])} 个  ")
    lines.append(f"- **日期列**: {len(stats['types']['date'])} 个")
    lines.append("")

    lines.append("## 字段 Schema + 示例值")
    lines.append("")
    lines.append("| # | 字段 | dtype | 缺失数 | 缺失率 | 示例值(最多5个) | 业务释义 |")
    lines.append("|---:|---|---|---:|---:|---|---|")
    for i, d in enumerate(stats["col_details"]):
        name = d["name"]
        dtype = d["dtype"]
        nulls = d["nulls"]
        null_pct = d["null_pct"]
        examples = d["examples"]
        meaning = d.get("meaning", "")
        examples_s = ", ".join(examples) if examples else ""
        null_pct_s = f"{null_pct:.2f}%" if not math.isnan(float(null_pct)) else "NA"
        lines.append(
            f"| {i+1} | `{name}` | `{dtype}` | {nulls} | {null_pct_s} | {examples_s} | {meaning} |"
        )
    lines.append("")

    lines.extend(_format_missing_table(stats))
    lines.append("")

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        stats = analyze_dataset(TARGET_PARQUET)
        md_content = format_md_for_dataset("Order 数据 Schema（含示例值）", stats)
        OUTPUT_MD.write_text(md_content, encoding="utf-8")
        print(f"已生成字段说明: {OUTPUT_MD}")
        return 0
    except Exception as e:
        print(f"处理出错: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
