#!/usr/bin/env python3
"""
车型预售/上市分析报告生成工具

功能：
1. 分析各车型预售期小订数据（每小时小订数、预售周期日小订数）
2. 分析各车型上市期锁单数据（每小时锁单数）
3. 生成HTML可视化报告

使用规则：
1. 基本用法（分析所有车型）：
   python3 scripts/analyze_order.py

2. 指定分析特定车型：
   python3 scripts/analyze_order.py --models LS8 LS9
   python3 scripts/analyze_order.py --models LS8

3. 指定输出文件路径：
   python3 scripts/analyze_order.py --output scripts/reports/custom_report.html
   python3 scripts/analyze_order.py --models LS8 --output scripts/reports/ls8_report.html

4. 查看帮助：
   python3 scripts/analyze_order.py --help

支持的车型：
- CM0, DM0, CM1, DM1, CM2, LS9, LS8

数据源：
- 订单数据：/Users/zihao_/Documents/coding/dataset/formatted/order_data.parquet
- 业务定义：/Users/zihao_/Documents/github/26W06_Tool_calls/schema/business_definition.json

报告内容：
1. 预售期分析
   1.1 汇总表（预售期开始日小订数统计）
   1.2 可视化（预售期开始日每小时小订数）
   1.3 预售周期日小订数（start～end范围每日小订数）
   1.4 历史预售至N日累计留存分 product_name（并统计上市后30日锁单）
   1.5 明细表（series_group_logic × order_gender）
   1.6 明细表（series_group_logic × parent_region_name：含在营门店数）
   1.7 可视化（series_group_logic × buyer_age：年龄占比折线图）
   1.8 IMADS 配置分布（Attribute Name × Value Dispaly Name × parent_region_name）

2. 上市期分析
   2.1 汇总表（上市日期锁单数统计）
   2.2 可视化（上市日期每小时锁单数）

输出：
- 默认：scripts/reports/analyze_order.html
- 可通过 --output 参数自定义路径
"""

import json
import re
import argparse
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

_missing = [m for m in ["numpy", "pandas", "plotly"] if importlib.util.find_spec(m) is None]
if _missing:
    print("💥 缺少依赖模块，无法运行 analyze_order.py")
    print(f"缺少模块: {', '.join(_missing)}")
    print("请先安装依赖后重试，例如：")
    print(f"{sys.executable} -m pip install numpy pandas plotly")
    raise SystemExit(1)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


PARQUET_FILE = Path("/Users/zihao_/Documents/coding/dataset/formatted/order_data.parquet")
BUSINESS_DEF_FILE = Path(
    "/Users/zihao_/Documents/github/26W06_Tool_calls/schema/business_definition.json"
)
ACTIVE_STORE_OPERATOR_FILE = Path(
    "/Users/zihao_/Documents/github/26W06_Tool_calls/operators/active_store.py"
)
OP_STEER_TRANSPOSED_FILE = Path("/Users/zihao_/Documents/coding/dataset/processed/LS8_Configuration_Details_transposed.csv")
SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT = SCRIPT_DIR / "reports" / "analyze_order.html"
ALL_TARGET_GROUPS = ["CM0","DM0","CM1","DM1","CM2","LS9","LS8"]


def load_business_definition(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_imads_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame()
    bom = b""
    try:
        with open(file_path, "rb") as f:
            bom = f.read(4) or b""
    except Exception:
        bom = b""

    encodings = []
    if bom.startswith(b"\xff\xfe") or bom.startswith(b"\xfe\xff"):
        encodings.extend(["utf-16"])
    if bom.startswith(b"\xef\xbb\xbf"):
        encodings.extend(["utf-8-sig"])
    encodings.extend(["utf-8", "gb18030"])

    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(file_path, sep="\t", dtype="string", encoding=enc)
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    return pd.DataFrame()

def load_op_steer_from_transposed(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, dtype="string", encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(file_path, dtype="string")
    if "order_number" not in df.columns:
        return pd.DataFrame()
    if "OP-Steer" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        "Order Number": df["order_number"],
        "Attribute Code": "OP-Steer",
        "Value Dispaly Name": df["OP-Steer"]
    })
    out = out.dropna(subset=["Order Number"])
    return out


def load_transposed_configuration_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, dtype="string", encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(file_path, dtype="string")
    if "order_number" not in df.columns:
        return pd.DataFrame()
    df["order_number"] = df["order_number"].astype("string")
    return df

_ACTIVE_STORE_OPERATOR = None


def _load_active_store_operator():
    global _ACTIVE_STORE_OPERATOR
    if _ACTIVE_STORE_OPERATOR is not None:
        return _ACTIVE_STORE_OPERATOR
    if not ACTIVE_STORE_OPERATOR_FILE.exists():
        _ACTIVE_STORE_OPERATOR = None
        return None
    spec = importlib.util.spec_from_file_location("active_store_operator", str(ACTIVE_STORE_OPERATOR_FILE))
    if spec is None or spec.loader is None:
        _ACTIVE_STORE_OPERATOR = None
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _ACTIVE_STORE_OPERATOR = mod
    return mod


def parse_sql_condition(df: pd.DataFrame, condition_str: str) -> pd.Series:
    def not_like_replacer(match):
        val = match.group(1)
        return f"~df['product_name'].str.contains('{val}', na=False, regex=False)"

    condition_str = re.sub(
        r"product_name\s+NOT\s+LIKE\s+'%([^%]+)%+'",
        not_like_replacer,
        condition_str,
    )

    def like_replacer(match):
        val = match.group(1)
        return f"df['product_name'].str.contains('{val}', na=False, regex=False)"

    condition_str = re.sub(
        r"product_name\s+LIKE\s+'%([^%]+)%+'",
        like_replacer,
        condition_str,
    )

    condition_str = condition_str.replace(" AND ", " & ").replace(" OR ", " | ")

    try:
        return eval(condition_str)
    except Exception as e:
        print(f"⚠️ 解析条件失败: {condition_str}, Error: {e}")
        return pd.Series([False] * len(df), index=df.index)


def load_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    print(f"📖 Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"✅ Loaded {len(df)} rows.")
    return df


def apply_series_group_logic(df: pd.DataFrame, business_def: dict) -> pd.DataFrame:
    logic: Dict[str, str] = business_def.get("series_group_logic", {})
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


def build_hourly_intention_counts(
    df: pd.DataFrame, business_def: dict, target_groups: List[str]
) -> pd.DataFrame:
    if "intention_payment_time" not in df.columns:
        raise KeyError("数据缺少列: intention_payment_time")
    if "order_number" not in df.columns:
        raise KeyError("数据缺少列: order_number")
    if not pd.api.types.is_datetime64_any_dtype(df["intention_payment_time"]):
        df["intention_payment_time"] = pd.to_datetime(df["intention_payment_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    rows: List[Dict[str, object]] = []

    for g in target_groups:
        start_str = (time_periods.get(g, {}) or {}).get("start")
        if not start_str:
            continue
        start_day = pd.Timestamp(start_str)
        end_day = start_day + pd.Timedelta(days=1)

        m_group = df["series_group_logic"].eq(g)
        m_time = df["intention_payment_time"].notna()
        m_day = (df["intention_payment_time"] >= start_day) & (df["intention_payment_time"] < end_day)
        df_day = df.loc[m_group & m_time & m_day, ["order_number", "intention_payment_time"]].copy()
        if df_day.empty:
            for hour in range(24):
                rows.append(
                    {
                        "series_group_logic": g,
                        "start_date": start_day.date().isoformat(),
                        "hour": hour,
                        "intention_orders": 0,
                    }
                )
            continue

        df_day["hour"] = df_day["intention_payment_time"].dt.hour.astype("int64")
        hourly = df_day.groupby("hour")["order_number"].nunique()
        for hour in range(24):
            rows.append(
                {
                    "series_group_logic": g,
                    "start_date": start_day.date().isoformat(),
                    "hour": hour,
                    "intention_orders": int(hourly.get(hour, 0)),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(["__group_order", "hour"]).drop(columns="__group_order").reset_index(drop=True)
    return out


def build_presale_summary(
    df: pd.DataFrame, business_def: dict, target_groups: List[str]
) -> pd.DataFrame:
    if "intention_payment_time" not in df.columns:
        raise KeyError("数据缺少列: intention_payment_time")
    if "order_number" not in df.columns:
        raise KeyError("数据缺少列: order_number")
    if "series_group_logic" not in df.columns:
        raise KeyError("数据缺少列: series_group_logic")
    if not pd.api.types.is_datetime64_any_dtype(df["intention_payment_time"]):
        df["intention_payment_time"] = pd.to_datetime(df["intention_payment_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    rows: List[Dict[str, object]] = []

    base = df.loc[
        df["intention_payment_time"].notna(), ["order_number", "intention_payment_time", "series_group_logic"]
    ].copy()

    for g in target_groups:
        tp = time_periods.get(g, {}) or {}
        start_str = tp.get("start")
        end_str = tp.get("end")
        if not start_str:
            continue

        start_day = pd.Timestamp(start_str)
        end_day = pd.Timestamp(end_str) if end_str else start_day

        base_g = base.loc[base["series_group_logic"].eq(g)].copy()

        start_end_excl = start_day + pd.Timedelta(days=1)
        start_day_slice = base_g.loc[
            (base_g["intention_payment_time"] >= start_day)
            & (base_g["intention_payment_time"] < start_end_excl),
            ["order_number", "intention_payment_time"],
        ].copy()
        if start_day_slice.empty:
            hourly_full = pd.Series([0] * 24, index=pd.RangeIndex(0, 24), dtype="int64")
        else:
            start_day_slice["hour"] = start_day_slice["intention_payment_time"].dt.hour.astype("int64")
            hourly = start_day_slice.groupby("hour")["order_number"].nunique()
            hourly_full = hourly.reindex(range(24), fill_value=0).astype("int64")

        peak_hour = int(hourly_full.idxmax())
        peak_count = int(hourly_full.iloc[peak_hour])
        next_hour_count = int(hourly_full.iloc[peak_hour + 1]) if peak_hour < 23 else 0
        start_day_total = int(hourly_full.sum())

        end_limit_excl = end_day + pd.Timedelta(days=1)
        first_week_end_excl = min(start_day + pd.Timedelta(days=7), end_limit_excl)
        week_slice = base_g.loc[
            (base_g["intention_payment_time"] >= start_day)
            & (base_g["intention_payment_time"] < first_week_end_excl),
            ["order_number", "intention_payment_time"],
        ].copy()
        if week_slice.empty:
            first_week_total = 0
        else:
            week_slice["date"] = week_slice["intention_payment_time"].dt.floor("D")
            daily = week_slice.groupby("date")["order_number"].nunique()
            first_week_total = int(daily.sum())

        delta_to_sat = (5 - start_day.weekday()) % 7
        if delta_to_sat == 0:
            delta_to_sat = 7
        weekend_start = start_day + pd.Timedelta(days=int(delta_to_sat))
        weekend_days = [weekend_start, weekend_start + pd.Timedelta(days=1)]
        weekend_total = 0
        for d in weekend_days:
            if d > end_day:
                continue
            day_end_excl = d + pd.Timedelta(days=1)
            day_slice = base_g.loc[
                (base_g["intention_payment_time"] >= d) & (base_g["intention_payment_time"] < day_end_excl),
                "order_number",
            ]
            weekend_total += int(day_slice.nunique())

        rows.append(
            {
                "series_group_logic": g,
                "预售日期": start_day.date().isoformat(),
                "峰值小时小订数": peak_count,
                "峰值后第二小时小订数": next_hour_count,
                "预售当日累计小订数": start_day_total,
                "第一个周末小订数": weekend_total,
                "第一周累计小订数": first_week_total,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(["__group_order"]).drop(columns="__group_order").reset_index(drop=True)
    return out


def build_presale_daily_counts(
    df: pd.DataFrame, business_def: dict, target_groups: List[str]
) -> pd.DataFrame:
    if "intention_payment_time" not in df.columns:
        raise KeyError("数据缺少列: intention_payment_time")
    if "order_number" not in df.columns:
        raise KeyError("数据缺少列: order_number")
    if "series_group_logic" not in df.columns:
        raise KeyError("数据缺少列: series_group_logic")
    if not pd.api.types.is_datetime64_any_dtype(df["intention_payment_time"]):
        df["intention_payment_time"] = pd.to_datetime(df["intention_payment_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    rows: List[Dict[str, object]] = []

    for g in target_groups:
        period = time_periods.get(g, {}) or {}
        start_str = period.get("start")
        end_str = period.get("end")
        if not start_str or not end_str:
            continue
        start_day = pd.Timestamp(start_str)
        end_day = pd.Timestamp(end_str)

        m_group = df["series_group_logic"].eq(g)
        m_time = df["intention_payment_time"].notna()
        m_range = (df["intention_payment_time"] >= start_day) & (df["intention_payment_time"] < end_day + pd.Timedelta(days=1))
        df_range = df.loc[m_group & m_time & m_range, ["order_number", "intention_payment_time"]].copy()

        date_range = pd.date_range(start=start_day, end=end_day, freq="D")
        if df_range.empty:
            for d in date_range:
                rows.append(
                    {
                        "series_group_logic": g,
                        "date": d.date().isoformat(),
                        "days_from_start": (d - start_day).days,
                        "intention_orders": 0,
                    }
                )
            continue

        df_range["date"] = df_range["intention_payment_time"].dt.normalize()
        daily = df_range.groupby("date")["order_number"].nunique()
        for d in date_range:
            d_ts = pd.Timestamp(d)
            rows.append(
                {
                    "series_group_logic": g,
                    "date": d.date().isoformat(),
                    "days_from_start": (d_ts - start_day).days,
                    "intention_orders": int(daily.get(d_ts, 0)),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(["__group_order", "days_from_start"]).drop(columns="__group_order").reset_index(drop=True)
    return out


def build_presale_retention_product_lock_30d(
    df: pd.DataFrame, business_def: dict, target_groups: List[str]
) -> pd.DataFrame:
    required_cols = [
        "series_group_logic",
        "order_number",
        "product_name",
        "intention_payment_time",
        "intention_refund_time",
        "lock_time",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"数据缺少列: {', '.join(missing)}")

    if not pd.api.types.is_datetime64_any_dtype(df["intention_payment_time"]):
        df["intention_payment_time"] = pd.to_datetime(df["intention_payment_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["intention_refund_time"]):
        df["intention_refund_time"] = pd.to_datetime(df["intention_refund_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["lock_time"]):
        df["lock_time"] = pd.to_datetime(df["lock_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    rows: List[Dict[str, object]] = []

    base_intention = df.loc[
        df["intention_payment_time"].notna(),
        [
            "series_group_logic",
            "order_number",
            "product_name",
            "intention_payment_time",
            "intention_refund_time",
        ],
    ].copy()

    base_lock = df.loc[
        df["lock_time"].notna(),
        ["series_group_logic", "order_number", "lock_time"],
    ].copy()

    for g in target_groups:
        tp = time_periods.get(g, {}) or {}
        start_str = tp.get("start")
        end_str = tp.get("end")
        finish_str = tp.get("finish")
        if not start_str or not end_str:
            continue

        presale_start = pd.Timestamp(start_str)
        presale_end = pd.Timestamp(end_str)
        listing_day = presale_end + pd.Timedelta(days=1) if g == "CM0" else presale_end
        finish_day = pd.Timestamp(finish_str) if finish_str else listing_day

        m_group = base_intention["series_group_logic"].eq(g)
        m_range = (base_intention["intention_payment_time"] >= presale_start) & (
            base_intention["intention_payment_time"] < presale_end + pd.Timedelta(days=1)
        )
        df_presale = base_intention.loc[m_group & m_range].copy()
        if df_presale.empty:
            continue

        retention_cutoff = listing_day + pd.Timedelta(days=1)
        m_retained = df_presale["intention_refund_time"].isna() | (
            df_presale["intention_refund_time"] >= retention_cutoff
        )
        retained = df_presale.loc[m_retained, ["order_number", "product_name"]].copy()
        if retained.empty:
            continue

        retained["product_name"] = retained["product_name"].fillna("(空)")
        retained = retained.drop_duplicates(subset=["order_number"])

        total_retained = int(retained["order_number"].nunique())
        if total_retained <= 0:
            continue

        after_30d_end_excl = min(listing_day + pd.Timedelta(days=31), finish_day + pd.Timedelta(days=1))

        lock_g = base_lock.loc[base_lock["series_group_logic"].eq(g)].copy()
        lock_window = lock_g.loc[
            (lock_g["lock_time"] >= listing_day) & (lock_g["lock_time"] < after_30d_end_excl),
            ["order_number", "lock_time"],
        ].copy()
        if lock_window.empty:
            locked_order_set = set()
            lock_by_product = pd.Series(dtype="int64")
        else:
            lock_window = lock_window[lock_window["order_number"].isin(retained["order_number"])]
            if lock_window.empty:
                locked_order_set = set()
                lock_by_product = pd.Series(dtype="int64")
            else:
                lock_window = lock_window.merge(retained, on="order_number", how="left")
                lock_window["date"] = lock_window["lock_time"].dt.floor("D")
                lock_by_product = (
                    lock_window.groupby(["product_name", "date"])["order_number"]
                    .nunique()
                    .groupby("product_name")
                    .sum()
                )
                locked_order_set = set(lock_window["order_number"].unique())

        retention_by_product = retained.groupby("product_name")["order_number"].nunique().sort_values(ascending=False)

        total_locked_30d = int(len(locked_order_set))
        total_rate = round(total_locked_30d / total_retained * 100, 1) if total_retained > 0 else 0.0
        rows.append(
            {
                "series_group_logic": g,
                "product_name": "合计",
                "留存小订数": total_retained,
                "留存占比": "100.0%",
                "上市后30日锁单数": total_locked_30d,
                "锁单转化率": f"{total_rate:.1f}%",
            }
        )

        for product_name, cnt in retention_by_product.items():
            share = round(cnt / total_retained * 100, 1) if total_retained > 0 else 0.0
            lock_cnt = int(lock_by_product.get(product_name, 0)) if lock_by_product is not None else 0
            rate = round(lock_cnt / cnt * 100, 1) if cnt > 0 else 0.0
            rows.append(
                {
                    "series_group_logic": g,
                    "product_name": product_name,
                    "留存小订数": int(cnt),
                    "留存占比": f"{share:.1f}%",
                    "上市后30日锁单数": lock_cnt,
                    "锁单转化率": f"{rate:.1f}%",
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out["__is_total"] = out["product_name"].eq("合计").astype("int64") * -1
        out = (
            out.sort_values(["__group_order", "__is_total", "留存小订数"], ascending=[True, True, False])
            .drop(columns=["__group_order", "__is_total"])
            .reset_index(drop=True)
        )
    return out


def build_presale_retention_by_product(
    df: pd.DataFrame, business_def: dict, target_groups: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = [
        "intention_payment_time",
        "intention_refund_time",
        "lock_time",
        "order_number",
        "series_group_logic",
        "product_name",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"数据缺少列: {', '.join(missing)}")

    if not pd.api.types.is_datetime64_any_dtype(df["intention_payment_time"]):
        df["intention_payment_time"] = pd.to_datetime(df["intention_payment_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["intention_refund_time"]):
        df["intention_refund_time"] = pd.to_datetime(df["intention_refund_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["lock_time"]):
        df["lock_time"] = pd.to_datetime(df["lock_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    detail_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for g in target_groups:
        tp = time_periods.get(g, {}) or {}
        start_str = tp.get("start")
        end_str = tp.get("end")
        finish_str = tp.get("finish")
        if not start_str or not end_str:
            continue

        start_day = pd.Timestamp(start_str)
        presale_end_day = pd.Timestamp(end_str)
        finish_day = pd.Timestamp(finish_str) if finish_str else presale_end_day
        n_days = int((presale_end_day.normalize() - start_day.normalize()).days + 1)
        n_days = max(1, n_days)

        listing_day = presale_end_day
        if g == "CM0":
            listing_day = listing_day + pd.Timedelta(days=1)

        presale_end_excl = presale_end_day + pd.Timedelta(days=1)
        window_end_excl = start_day + pd.Timedelta(days=int(n_days))
        window_end_excl = min(window_end_excl, presale_end_excl)
        m_retention = (
            df["series_group_logic"].eq(g)
            & df["intention_payment_time"].notna()
            & (df["intention_payment_time"] >= start_day)
            & (df["intention_payment_time"] < window_end_excl)
            & ((df["intention_refund_time"] > window_end_excl) | df["intention_refund_time"].isna())
        )
        retention_orders = df.loc[m_retention, ["order_number", "product_name"]].dropna(subset=["order_number"])
        retention_orders = retention_orders.drop_duplicates(subset=["order_number", "product_name"]).copy()
        retention_total = int(retention_orders["order_number"].nunique())

        finish_excl = finish_day + pd.Timedelta(days=1)
        lock_30d_end_excl = min(listing_day + pd.Timedelta(days=31), finish_excl)
        m_lock_30d = (
            df["series_group_logic"].eq(g)
            & df["lock_time"].notna()
            & (df["lock_time"] >= listing_day)
            & (df["lock_time"] < lock_30d_end_excl)
        )
        lock_orders = df.loc[m_lock_30d, ["order_number"]].dropna(subset=["order_number"]).drop_duplicates()
        lock_orders = lock_orders.assign(_locked_30d=1)

        m_lock_to_date = (
            df["series_group_logic"].eq(g)
            & df["lock_time"].notna()
            & (df["lock_time"] >= listing_day)
            & (df["lock_time"] < finish_excl)
        )
        lock_to_date_orders = df.loc[m_lock_to_date, ["order_number"]].dropna(subset=["order_number"]).drop_duplicates()

        if retention_orders.empty:
            summary_rows.append(
                {
                    "series_group_logic": g,
                    "预售期": f"{start_day.date().isoformat()} ~ {presale_end_day.date().isoformat()}",
                    "预售周期（日）": int(n_days),
                    "预售至N日累计留存小订数": 0,
                    "上市后30日锁单数": 0,
                    "上市后30日转化率": "0.0%",
                    "上市至今留存小订锁单数": 0,
                    "留存小订转化率": "0.0%",
                }
            )
            continue

        merged = retention_orders.merge(lock_orders, on="order_number", how="left")
        merged["_locked_30d"] = merged["_locked_30d"].fillna(0).astype("int64")

        agg = (
            merged.groupby("product_name", as_index=False)
            .agg(
                retention_intention_orders=("order_number", "nunique"),
                lock_30d_orders=("_locked_30d", "sum"),
            )
            .sort_values(["retention_intention_orders", "lock_30d_orders"], ascending=[False, False])
            .reset_index(drop=True)
        )
        agg["retention_share"] = agg["retention_intention_orders"].apply(
            lambda x: round(x / retention_total * 100, 1) if retention_total > 0 else 0.0
        )
        agg["lock_30d_rate"] = agg.apply(
            lambda row: round(row["lock_30d_orders"] / row["retention_intention_orders"] * 100, 1)
            if row["retention_intention_orders"] > 0
            else 0.0,
            axis=1,
        )

        locked_total = int(merged.loc[merged["_locked_30d"].eq(1), "order_number"].nunique())
        retention_order_numbers = retention_orders["order_number"].dropna().drop_duplicates()
        locked_to_date_total = int(retention_order_numbers.isin(lock_to_date_orders["order_number"]).sum())
        summary_rows.append(
            {
                "series_group_logic": g,
                "预售期": f"{start_day.date().isoformat()} ~ {presale_end_day.date().isoformat()}",
                "预售周期（日）": int(n_days),
                "预售至N日累计留存小订数": retention_total,
                "上市后30日锁单数": locked_total,
                "上市后30日转化率": f"{(locked_total / retention_total * 100) if retention_total > 0 else 0.0:.1f}%",
                "上市至今留存小订锁单数": locked_to_date_total,
                "留存小订转化率": f"{(locked_to_date_total / retention_total * 100) if retention_total > 0 else 0.0:.1f}%",
            }
        )

        for _, r in agg.iterrows():
            detail_rows.append(
                {
                    "series_group_logic": g,
                    "product_name": str(r["product_name"]),
                    "预售周期（日）": int(n_days),
                    "预售至N日累计留存小订数": int(r["retention_intention_orders"]),
                    "预售至N日累计留存占比": f"{float(r['retention_share']):.1f}%",
                    "上市后30日锁单数": int(r["lock_30d_orders"]),
                    "上市后30日转化率": f"{float(r['lock_30d_rate']):.1f}%",
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        summary_df["__group_order"] = summary_df["series_group_logic"].map(group_order).fillna(len(group_order))
        summary_df = summary_df.sort_values(["__group_order"]).drop(columns="__group_order").reset_index(drop=True)
    if not detail_df.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        detail_df["__group_order"] = detail_df["series_group_logic"].map(group_order).fillna(len(group_order))
        detail_df = detail_df.sort_values(["__group_order", "预售至N日累计留存小订数"], ascending=[True, False]).drop(
            columns="__group_order"
        ).reset_index(drop=True)
    return detail_df, summary_df


def build_presale_retention_region_detail(
    df: pd.DataFrame, business_def: dict, target_groups: List[str]
) -> pd.DataFrame:
    required_cols = [
        "intention_payment_time",
        "intention_refund_time",
        "order_number",
        "series_group_logic",
        "product_name",
        "parent_region_name",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"数据缺少列: {', '.join(missing)}")

    if not pd.api.types.is_datetime64_any_dtype(df["intention_payment_time"]):
        df["intention_payment_time"] = pd.to_datetime(df["intention_payment_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["intention_refund_time"]):
        df["intention_refund_time"] = pd.to_datetime(df["intention_refund_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    rows: List[Dict[str, object]] = []

    for g in target_groups:
        tp = time_periods.get(g, {}) or {}
        start_str = tp.get("start")
        end_str = tp.get("end")
        if not start_str or not end_str:
            continue

        start_day = pd.Timestamp(start_str)
        presale_end_day = pd.Timestamp(end_str)
        n_days = int((presale_end_day.normalize() - start_day.normalize()).days + 1)
        n_days = max(1, n_days)

        presale_end_excl = presale_end_day + pd.Timedelta(days=1)
        window_end_excl = start_day + pd.Timedelta(days=int(n_days))
        window_end_excl = min(window_end_excl, presale_end_excl)

        m_retention = (
            df["series_group_logic"].eq(g)
            & df["intention_payment_time"].notna()
            & (df["intention_payment_time"] >= start_day)
            & (df["intention_payment_time"] < window_end_excl)
            & ((df["intention_refund_time"] > window_end_excl) | df["intention_refund_time"].isna())
        )

        retained = df.loc[m_retention, ["order_number", "product_name", "parent_region_name"]].dropna(
            subset=["order_number"]
        )
        if retained.empty:
            continue

        retained["parent_region_name"] = retained["parent_region_name"].fillna("NA").astype("string")
        retained["product_name"] = retained["product_name"].fillna("NA").astype("string")
        retained = retained.drop_duplicates(subset=["order_number", "product_name", "parent_region_name"])

        counts = (
            retained.groupby(["product_name", "parent_region_name"])["order_number"]
            .nunique()
            .rename("cnt")
            .reset_index()
        )
        region_totals = counts.groupby("parent_region_name")["cnt"].sum().rename("region_total")
        counts = counts.merge(region_totals, on="parent_region_name", how="left")
        counts["share_in_region"] = counts.apply(
            lambda r: round(r["cnt"] / r["region_total"] * 100, 1) if r["region_total"] > 0 else 0.0, axis=1
        )

        for _, r in counts.iterrows():
            rows.append(
                {
                    "series_group_logic": g,
                    "product_name": str(r["product_name"]),
                    "parent_region_name": str(r["parent_region_name"]),
                    "预售周期（日）": int(n_days),
                    "预售至N日累计留存小订数": int(r["cnt"]),
                    "产品占比（该region内）": float(r["share_in_region"]),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(
            ["__group_order", "series_group_logic", "product_name", "预售至N日累计留存小订数"],
            ascending=[True, True, True, False],
        ).drop(columns="__group_order").reset_index(drop=True)
    return out


def build_presale_retention_region_summary_with_active_stores(
    df: pd.DataFrame, business_def: dict, target_groups: List[str]
) -> pd.DataFrame:
    required_cols = [
        "intention_payment_time",
        "intention_refund_time",
        "order_number",
        "series_group_logic",
        "parent_region_name",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"数据缺少列: {', '.join(missing)}")

    if not pd.api.types.is_datetime64_any_dtype(df["intention_payment_time"]):
        df["intention_payment_time"] = pd.to_datetime(df["intention_payment_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["intention_refund_time"]):
        df["intention_refund_time"] = pd.to_datetime(df["intention_refund_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    op = _load_active_store_operator()

    rows: List[Dict[str, object]] = []

    for g in target_groups:
        tp = time_periods.get(g, {}) or {}
        start_str = tp.get("start")
        end_str = tp.get("end")
        if not start_str or not end_str:
            continue

        start_day = pd.Timestamp(start_str)
        presale_end_day = pd.Timestamp(end_str)
        n_days = int((presale_end_day.normalize() - start_day.normalize()).days + 1)
        n_days = max(1, n_days)

        presale_end_excl = presale_end_day + pd.Timedelta(days=1)
        window_end_excl = start_day + pd.Timedelta(days=int(n_days))
        window_end_excl = min(window_end_excl, presale_end_excl)

        m_retention = (
            df["series_group_logic"].eq(g)
            & df["intention_payment_time"].notna()
            & (df["intention_payment_time"] >= start_day)
            & (df["intention_payment_time"] < window_end_excl)
            & ((df["intention_refund_time"] > window_end_excl) | df["intention_refund_time"].isna())
        )
        retained = df.loc[m_retention, ["order_number", "parent_region_name"]].dropna(subset=["order_number"]).copy()
        if retained.empty:
            continue

        retained["parent_region_name"] = retained["parent_region_name"].fillna("NA").astype("string")
        retained = retained.drop_duplicates(subset=["order_number", "parent_region_name"])

        total_retained = int(retained["order_number"].nunique())
        by_region = (
            retained.groupby("parent_region_name")["order_number"].nunique().sort_values(ascending=False)
        )

        listing_day = presale_end_day
        if g == "CM0":
            listing_day = listing_day + pd.Timedelta(days=1)
        listing_day = listing_day.normalize()

        active_by_region: Dict[str, int] = {}
        if op is not None and hasattr(op, "run_active_store_operator"):
            for region_name in by_region.index.astype(str).tolist():
                df_region = df.loc[df["parent_region_name"].astype("string").fillna("NA").eq(region_name)].copy()
                if df_region.empty:
                    active_by_region[region_name] = 0
                    continue
                start = listing_day.strftime("%Y-%m-%d")
                end = (listing_day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                result = op.run_active_store_operator(df_region, start=start, end=end)
                if isinstance(result, dict) and result.get("daily_rows"):
                    active_by_region[region_name] = int(result["daily_rows"][0].get("active_store_count", 0))
                else:
                    active_by_region[region_name] = 0

        for region_name, cnt in by_region.items():
            share = round(cnt / total_retained * 100, 1) if total_retained > 0 else 0.0
            rows.append(
                {
                    "series_group_logic": g,
                    "预售周期（日）": int(n_days),
                    "上市日(d)": listing_day.date().isoformat(),
                    "parent_region_name": str(region_name),
                    "预售至N日累计留存小订数": int(cnt),
                    "占比": share,
                    "在营门店数": int(active_by_region.get(str(region_name), 0)) if active_by_region else None,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(
            ["__group_order", "series_group_logic", "预售至N日累计留存小订数"],
            ascending=[True, True, False],
        ).drop(columns="__group_order").reset_index(drop=True)
    return out


def build_presale_retention_age_detail(
    df: pd.DataFrame,
    business_def: dict,
    target_groups: List[str],
    age_col: str,
    age_label: str,
) -> pd.DataFrame:
    required_cols = [
        "intention_payment_time",
        "intention_refund_time",
        "order_number",
        "series_group_logic",
        age_col,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"数据缺少列: {', '.join(missing)}")

    if not pd.api.types.is_datetime64_any_dtype(df["intention_payment_time"]):
        df["intention_payment_time"] = pd.to_datetime(df["intention_payment_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["intention_refund_time"]):
        df["intention_refund_time"] = pd.to_datetime(df["intention_refund_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    rows: List[Dict[str, object]] = []

    age_num = pd.to_numeric(df[age_col], errors="coerce")
    age_int = age_num.round().astype("Int64")

    for g in target_groups:
        tp = time_periods.get(g, {}) or {}
        start_str = tp.get("start")
        end_str = tp.get("end")
        if not start_str or not end_str:
            continue

        start_day = pd.Timestamp(start_str)
        presale_end_day = pd.Timestamp(end_str)
        n_days = int((presale_end_day.normalize() - start_day.normalize()).days + 1)
        n_days = max(1, n_days)

        presale_end_excl = presale_end_day + pd.Timedelta(days=1)
        window_end_excl = start_day + pd.Timedelta(days=int(n_days))
        window_end_excl = min(window_end_excl, presale_end_excl)

        m_retention = (
            df["series_group_logic"].eq(g)
            & df["intention_payment_time"].notna()
            & (df["intention_payment_time"] >= start_day)
            & (df["intention_payment_time"] < window_end_excl)
            & ((df["intention_refund_time"] > window_end_excl) | df["intention_refund_time"].isna())
        )
        retained = df.loc[m_retention, ["order_number"]].dropna(subset=["order_number"]).copy()
        if retained.empty:
            continue

        retained = retained.drop_duplicates(subset=["order_number"])
        retained[age_label] = age_int.loc[retained.index]

        total = int(retained["order_number"].nunique())
        by_age = retained.dropna(subset=[age_label]).groupby(age_label)["order_number"].nunique().sort_index()

        for age_val, cnt in by_age.items():
            share = round(cnt / total * 100, 1) if total > 0 else 0.0
            rows.append(
                {
                    "series_group_logic": g,
                    "预售周期（日）": int(n_days),
                    age_label: int(age_val) if pd.notna(age_val) else None,
                    "预售至N日累计留存小订数": int(cnt),
                    "占比": share,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(["__group_order", age_label]).drop(columns="__group_order").reset_index(drop=True)
    return out


def build_presale_retention_category_detail(
    df: pd.DataFrame,
    business_def: dict,
    target_groups: List[str],
    category_col: str,
    category_label: str,
) -> pd.DataFrame:
    required_cols = [
        "intention_payment_time",
        "intention_refund_time",
        "order_number",
        "series_group_logic",
        category_col,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"数据缺少列: {', '.join(missing)}")

    if not pd.api.types.is_datetime64_any_dtype(df["intention_payment_time"]):
        df["intention_payment_time"] = pd.to_datetime(df["intention_payment_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["intention_refund_time"]):
        df["intention_refund_time"] = pd.to_datetime(df["intention_refund_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    rows: List[Dict[str, object]] = []

    cat_series = df[category_col].astype("string").fillna("NA").str.strip()

    for g in target_groups:
        tp = time_periods.get(g, {}) or {}
        start_str = tp.get("start")
        end_str = tp.get("end")
        if not start_str or not end_str:
            continue

        start_day = pd.Timestamp(start_str)
        presale_end_day = pd.Timestamp(end_str)
        n_days = int((presale_end_day.normalize() - start_day.normalize()).days + 1)
        n_days = max(1, n_days)

        presale_end_excl = presale_end_day + pd.Timedelta(days=1)
        window_end_excl = start_day + pd.Timedelta(days=int(n_days))
        window_end_excl = min(window_end_excl, presale_end_excl)

        m_retention = (
            df["series_group_logic"].eq(g)
            & df["intention_payment_time"].notna()
            & (df["intention_payment_time"] >= start_day)
            & (df["intention_payment_time"] < window_end_excl)
            & ((df["intention_refund_time"] > window_end_excl) | df["intention_refund_time"].isna())
        )
        retained = df.loc[m_retention, ["order_number"]].dropna(subset=["order_number"]).copy()
        if retained.empty:
            continue

        retained = retained.drop_duplicates(subset=["order_number"])
        retained[category_label] = cat_series.loc[retained.index]

        total = int(retained["order_number"].nunique())
        by_cat = (
            retained.groupby(category_label)["order_number"].nunique().sort_values(ascending=False)
        )
        for cat_val, cnt in by_cat.items():
            share = round(cnt / total * 100, 1) if total > 0 else 0.0
            rows.append(
                {
                    "series_group_logic": g,
                    "预售周期（日）": int(n_days),
                    category_label: str(cat_val),
                    "预售至N日累计留存小订数": int(cnt),
                    "占比": share,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(["__group_order", "预售至N日累计留存小订数"], ascending=[True, False]).drop(
            columns="__group_order"
        ).reset_index(drop=True)
    return out


def build_imads_attribute_region_distribution(
    df_orders: pd.DataFrame, business_def: dict, target_groups: List[str], df_imads: pd.DataFrame
) -> pd.DataFrame:
    if df_imads is None or df_imads.empty:
        return pd.DataFrame()
    if "Order Number" not in df_imads.columns:
        raise KeyError("IMADS数据缺少列: Order Number")
    if "Attribute Name" not in df_imads.columns:
        raise KeyError("IMADS数据缺少列: Attribute Name")
    if "Value Dispaly Name" not in df_imads.columns:
        raise KeyError("IMADS数据缺少列: Value Dispaly Name")

    required_cols = ["order_number", "parent_region_name", "series_group_logic", "intention_payment_time", "intention_refund_time"]
    missing = [c for c in required_cols if c not in df_orders.columns]
    if missing:
        raise KeyError(f"订单数据缺少列: {', '.join(missing)}")

    if not pd.api.types.is_datetime64_any_dtype(df_orders["intention_payment_time"]):
        df_orders["intention_payment_time"] = pd.to_datetime(df_orders["intention_payment_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df_orders["intention_refund_time"]):
        df_orders["intention_refund_time"] = pd.to_datetime(df_orders["intention_refund_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    retained_rows: List[pd.DataFrame] = []
    for g in target_groups:
        tp = time_periods.get(g, {}) or {}
        start_str = tp.get("start")
        end_str = tp.get("end")
        if not start_str or not end_str:
            continue
        start_day = pd.Timestamp(start_str)
        end_day = pd.Timestamp(end_str)
        n_days = int((end_day.normalize() - start_day.normalize()).days + 1)
        n_days = max(1, n_days)

        end_excl = end_day + pd.Timedelta(days=1)
        window_end_excl = min(start_day + pd.Timedelta(days=int(n_days)), end_excl)
        m_retention = (
            df_orders["series_group_logic"].eq(g)
            & df_orders["intention_payment_time"].notna()
            & (df_orders["intention_payment_time"] >= start_day)
            & (df_orders["intention_payment_time"] < window_end_excl)
            & ((df_orders["intention_refund_time"] > window_end_excl) | df_orders["intention_refund_time"].isna())
        )
        retained = df_orders.loc[m_retention, ["order_number", "series_group_logic", "parent_region_name"]].dropna(
            subset=["order_number"]
        )
        if retained.empty:
            continue
        retained["parent_region_name"] = retained["parent_region_name"].fillna("NA").astype("string")
        retained = retained.drop_duplicates(subset=["order_number"])
        retained_rows.append(retained)

    if not retained_rows:
        return pd.DataFrame()
    retained_all = pd.concat(retained_rows, ignore_index=True)

    imads = df_imads.loc[:, ["Order Number", "Attribute Name", "Value Dispaly Name"]].copy()
    imads = imads.dropna(subset=["Order Number", "Attribute Name", "Value Dispaly Name"])
    imads = imads.rename(
        columns={
            "Order Number": "order_number",
            "Attribute Name": "attribute_name",
            "Value Dispaly Name": "value_display_name",
        }
    )
    imads["order_number"] = imads["order_number"].astype("string")

    matched = retained_all.merge(imads, on="order_number", how="inner")
    if matched.empty:
        return pd.DataFrame()

    grp = (
        matched.groupby(
            ["series_group_logic", "attribute_name", "value_display_name", "parent_region_name"], as_index=False
        )["order_number"]
        .nunique()
        .rename(columns={"order_number": "order_cnt"})
    )
    value_totals = grp.groupby(["series_group_logic", "attribute_name", "value_display_name"])["order_cnt"].sum().rename(
        "value_total"
    )
    grp = grp.merge(value_totals, on=["series_group_logic", "attribute_name", "value_display_name"], how="left")
    grp["share_in_value"] = grp.apply(
        lambda r: round(r["order_cnt"] / r["value_total"] * 100, 1) if r["value_total"] > 0 else 0.0, axis=1
    )
    grp = grp.drop(columns=["value_total"])
    return grp


def build_imads_op_steer_region_summary(
    df_orders: pd.DataFrame, business_def: dict, target_groups: List[str], df_imads: pd.DataFrame
) -> pd.DataFrame:
    if df_imads is None or df_imads.empty:
        return pd.DataFrame()
    required_imads_cols = ["Order Number", "Attribute Code", "Value Dispaly Name"]
    missing_imads = [c for c in required_imads_cols if c not in df_imads.columns]
    if missing_imads:
        raise KeyError(f"IMADS数据缺少列: {', '.join(missing_imads)}")

    required_order_cols = [
        "order_number",
        "parent_region_name",
        "product_name",
        "series_group_logic",
        "intention_payment_time",
        "intention_refund_time",
    ]
    missing_orders = [c for c in required_order_cols if c not in df_orders.columns]
    if missing_orders:
        raise KeyError(f"订单数据缺少列: {', '.join(missing_orders)}")

    if not pd.api.types.is_datetime64_any_dtype(df_orders["intention_payment_time"]):
        df_orders["intention_payment_time"] = pd.to_datetime(df_orders["intention_payment_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df_orders["intention_refund_time"]):
        df_orders["intention_refund_time"] = pd.to_datetime(df_orders["intention_refund_time"], errors="coerce")

    imads = df_imads.loc[:, ["Order Number", "Attribute Code", "Value Dispaly Name"]].copy()
    imads = imads.dropna(subset=["Order Number", "Attribute Code", "Value Dispaly Name"])
    imads = imads.rename(
        columns={
            "Order Number": "order_number",
            "Attribute Code": "attribute_code",
            "Value Dispaly Name": "value_display_name",
        }
    )
    imads["order_number"] = imads["order_number"].astype("string")
    imads["attribute_code"] = imads["attribute_code"].astype("string")
    imads["value_display_name"] = imads["value_display_name"].astype("string")

    imads = imads[imads["attribute_code"].eq("OP-Steer")].copy()
    if imads.empty:
        return pd.DataFrame()

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    rows: List[Dict[str, object]] = []

    for g in target_groups:
        tp = time_periods.get(g, {}) or {}
        start_str = tp.get("start")
        end_str = tp.get("end")
        if not start_str or not end_str:
            continue

        start_day = pd.Timestamp(start_str)
        end_day = pd.Timestamp(end_str)
        n_days = int((end_day.normalize() - start_day.normalize()).days + 1)
        n_days = max(1, n_days)

        end_excl = end_day + pd.Timedelta(days=1)
        window_end_excl = min(start_day + pd.Timedelta(days=int(n_days)), end_excl)
        m_retention = (
            df_orders["series_group_logic"].eq(g)
            & df_orders["intention_payment_time"].notna()
            & (df_orders["intention_payment_time"] >= start_day)
            & (df_orders["intention_payment_time"] < window_end_excl)
            & ((df_orders["intention_refund_time"] > window_end_excl) | df_orders["intention_refund_time"].isna())
        )

        retained = df_orders.loc[m_retention, ["order_number", "parent_region_name", "product_name"]].dropna(
            subset=["order_number"]
        ).copy()
        if retained.empty:
            continue

        retained["order_number"] = retained["order_number"].astype("string")
        retained["parent_region_name"] = retained["parent_region_name"].fillna("NA").astype("string")
        retained["product_name"] = retained["product_name"].fillna("").astype("string")
        retained = retained.drop_duplicates(subset=["order_number"])

        pre_total = retained[retained["product_name"].ne("")].groupby("parent_region_name")["order_number"].nunique()
        high66 = retained[retained["product_name"].str.contains("66 Ultra", na=False, regex=False)]
        high66_total = high66.groupby("parent_region_name")["order_number"].nunique()

        matched = retained[["order_number", "parent_region_name", "product_name"]].merge(
            imads[["order_number", "value_display_name"]],
            on="order_number",
            how="left",
        )

        v = matched["value_display_name"].fillna("").astype("string")
        yes_mask = v.eq("是").fillna(False)
        no_mask = v.eq("否").fillna(False)

        yes_cnt = matched[yes_mask].groupby("parent_region_name")["order_number"].nunique()
        no_cnt = matched[no_mask].groupby("parent_region_name")["order_number"].nunique()

        yes66_mask = matched["product_name"].str.contains("66 Ultra", na=False, regex=False) & yes_mask
        yes66_cnt = matched[yes66_mask].groupby("parent_region_name")["order_number"].nunique()

        regions = pd.Index(pre_total.index.union(yes_cnt.index).union(no_cnt.index).union(high66_total.index)).astype("string")
        for region in regions.tolist():
            base_total = int(pre_total.get(region, 0))
            base66_total = int(high66_total.get(region, 0))
            y_yes = int(yes_cnt.get(region, 0))
            y_no = int(no_cnt.get(region, 0))
            y_yes66 = int(yes66_cnt.get(region, 0))
            rate = (y_yes / base_total * 100) if base_total > 0 else 0.0
            rate66 = (y_yes66 / base66_total * 100) if base66_total > 0 else 0.0
            rows.append(
                {
                    "series_group_logic": g,
                    "parent_region_name": str(region),
                    "Value Dispaly Name=是": y_yes,
                    "Value Dispaly Name=否": y_no,
                    "小订总数": base_total,
                    "高配（66ultra）总数": base66_total,
                    "线控选装率": f"{rate:.1f}%",
                    "线控在（66ultra）选装率": f"{rate66:.1f}%",
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(
            ["__group_order", "小订总数"],
            ascending=[True, False],
        ).drop(columns="__group_order").reset_index(drop=True)
    return out


def build_transposed_configuration_option_summary(
    df_orders: pd.DataFrame,
    business_def: dict,
    target_groups: List[str],
    df_transposed: pd.DataFrame,
) -> pd.DataFrame:
    if df_transposed is None or df_transposed.empty:
        return pd.DataFrame()

    if "order_number" not in df_transposed.columns:
        raise KeyError("转置配置数据缺少列: order_number")

    required_order_cols = [
        "order_number",
        "series_group_logic",
        "intention_payment_time",
        "intention_refund_time",
    ]
    missing_orders = [c for c in required_order_cols if c not in df_orders.columns]
    if missing_orders:
        raise KeyError(f"订单数据缺少列: {', '.join(missing_orders)}")

    if not pd.api.types.is_datetime64_any_dtype(df_orders["intention_payment_time"]):
        df_orders["intention_payment_time"] = pd.to_datetime(df_orders["intention_payment_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df_orders["intention_refund_time"]):
        df_orders["intention_refund_time"] = pd.to_datetime(df_orders["intention_refund_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    configuration_name_map = business_def.get("configuration", {}) or {}
    if not isinstance(configuration_name_map, dict):
        configuration_name_map = {}
    configuration_name_map = {str(k): str(v) for k, v in configuration_name_map.items()}
    if "DATE([Order Intent Pay Time])" in configuration_name_map and "intention_pay_time" not in configuration_name_map:
        configuration_name_map["intention_pay_time"] = configuration_name_map["DATE([Order Intent Pay Time])"]

    exclude_cols = {
        "order_number",
        "DATE([Order Intent Pay Time])",
        "intention_pay_time",
        "lock_time",
        "Is Staff",
        "Product Name",
        "Product_Types",
    }

    rows: List[Dict[str, object]] = []
    df_transposed = df_transposed.copy()
    df_transposed["order_number"] = df_transposed["order_number"].astype("string")

    for g in target_groups:
        tp = time_periods.get(g, {}) or {}
        start_str = tp.get("start")
        end_str = tp.get("end")
        if not start_str or not end_str:
            continue

        start_day = pd.Timestamp(start_str)
        presale_end_day = pd.Timestamp(end_str)
        n_days = int((presale_end_day.normalize() - start_day.normalize()).days + 1)
        n_days = max(1, n_days)

        presale_end_excl = presale_end_day + pd.Timedelta(days=1)
        window_end_excl = start_day + pd.Timedelta(days=int(n_days))
        window_end_excl = min(window_end_excl, presale_end_excl)

        m_retention = (
            df_orders["series_group_logic"].eq(g)
            & df_orders["intention_payment_time"].notna()
            & (df_orders["intention_payment_time"] >= start_day)
            & (df_orders["intention_payment_time"] < window_end_excl)
            & ((df_orders["intention_refund_time"] > window_end_excl) | df_orders["intention_refund_time"].isna())
        )
        retained_orders = (
            df_orders.loc[m_retention, ["order_number"]]
            .dropna(subset=["order_number"])
            .drop_duplicates(subset=["order_number"])
            .copy()
        )
        retained_orders["order_number"] = retained_orders["order_number"].astype("string")
        retention_total = int(retained_orders["order_number"].nunique())
        if retention_total <= 0:
            continue

        matched = df_transposed.merge(retained_orders, on="order_number", how="inner")
        if matched.empty:
            continue

        attr_cols = [c for c in matched.columns if c not in exclude_cols]
        if not attr_cols:
            continue

        long_df = matched.melt(
            id_vars=["order_number"],
            value_vars=attr_cols,
            var_name="Attribute",
            value_name="Value Dispaly Name",
        )
        v = long_df["Value Dispaly Name"].astype("string").fillna("").str.strip()
        long_df["Value Dispaly Name"] = v
        long_df = long_df[long_df["Value Dispaly Name"].ne("")]
        if long_df.empty:
            continue

        agg = (
            long_df.groupby(["Attribute", "Value Dispaly Name"], as_index=False)
            .agg(订单数=("order_number", "nunique"))
            .sort_values(["Attribute", "订单数"], ascending=[True, False])
            .reset_index(drop=True)
        )
        agg["选配比例"] = agg["订单数"].apply(
            lambda x: f"{(float(x) / retention_total * 100) if retention_total > 0 else 0.0:.1f}%"
        )
        agg.insert(0, "series_group_logic", g)
        agg.insert(2, "中文名称", agg["Attribute"].map(lambda x: configuration_name_map.get(str(x), "")).astype("string"))

        rows.append(agg)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    group_order = {g: i for i, g in enumerate(target_groups)}
    out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
    out = out.sort_values(["__group_order", "Attribute", "订单数"], ascending=[True, True, False]).drop(
        columns="__group_order"
    )
    return out.reset_index(drop=True)


def _render_age_share_line_figure(
    age_df: pd.DataFrame,
    target_groups: List[str],
    age_label: str,
    fig_title: str,
) -> go.Figure:
    n_groups = len(target_groups)
    n_cols = 3
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=target_groups,
    )

    positions = [(r + 1, c + 1) for r in range(n_rows) for c in range(n_cols)]
    for g, (r, c) in zip(target_groups, positions):
        dfg = age_df[age_df["series_group_logic"].eq(g)].copy()
        if dfg.empty:
            continue
        dfg = dfg.dropna(subset=[age_label, "占比", "预售至N日累计留存小订数"]).sort_values(age_label)
        if dfg.empty:
            continue

        x = dfg[age_label].astype(int).tolist()
        y = dfg["占比"].astype(float).tolist()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", showlegend=False), row=r, col=c)

        weights = dfg["预售至N日累计留存小订数"].astype(float).tolist()
        if weights and sum(weights) > 0:
            avg_age = float(np.average(np.array(x, dtype="float64"), weights=np.array(weights, dtype="float64")))
            fig.add_vline(
                x=avg_age,
                line_width=1,
                line_dash="dot",
                line_color="rgba(50,50,50,0.6)",
                row=r,
                col=c,
            )
            fig.add_annotation(
                x=avg_age,
                y=9.5,
                text=f"Avg {avg_age:.1f}",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.7)",
                row=r,
                col=c,
            )
        fig.update_xaxes(title_text=age_label, tickmode="linear", dtick=5, row=r, col=c)
        fig.update_yaxes(title_text="Share (%)", range=[0, 10], row=r, col=c)

    fig.update_layout(height=360 * n_rows, title=fig_title, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def build_hourly_lock_counts(df: pd.DataFrame, business_def: dict, target_groups: List[str]) -> pd.DataFrame:
    if "lock_time" not in df.columns:
        raise KeyError("数据缺少列: lock_time")
    if "order_number" not in df.columns:
        raise KeyError("数据缺少列: order_number")
    if not pd.api.types.is_datetime64_any_dtype(df["lock_time"]):
        df["lock_time"] = pd.to_datetime(df["lock_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    rows: List[Dict[str, object]] = []

    for g in target_groups:
        end_str = (time_periods.get(g, {}) or {}).get("end")
        if not end_str:
            continue
        end_day = pd.Timestamp(end_str)
        if g == "CM0":
            end_day = end_day + pd.Timedelta(days=1)
        end_day_excl = end_day + pd.Timedelta(days=1)

        m_group = df["series_group_logic"].eq(g)
        m_time = df["lock_time"].notna()
        m_day = (df["lock_time"] >= end_day) & (df["lock_time"] < end_day_excl)
        df_day = df.loc[m_group & m_time & m_day, ["order_number", "lock_time"]].copy()
        if df_day.empty:
            for hour in range(24):
                rows.append(
                    {
                        "series_group_logic": g,
                        "end_date": end_day.date().isoformat(),
                        "hour": hour,
                        "lock_orders": 0,
                    }
                )
            continue

        df_day["hour"] = df_day["lock_time"].dt.hour.astype("int64")
        hourly = df_day.groupby("hour")["order_number"].nunique()
        for hour in range(24):
            rows.append(
                {
                    "series_group_logic": g,
                    "end_date": end_day.date().isoformat(),
                    "hour": hour,
                    "lock_orders": int(hourly.get(hour, 0)),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(["__group_order", "hour"]).drop(columns="__group_order").reset_index(drop=True)
    return out


def build_listing_summary(df: pd.DataFrame, business_def: dict, target_groups: List[str]) -> pd.DataFrame:
    if "lock_time" not in df.columns:
        raise KeyError("数据缺少列: lock_time")
    if "order_number" not in df.columns:
        raise KeyError("数据缺少列: order_number")
    if "series_group_logic" not in df.columns:
        raise KeyError("数据缺少列: series_group_logic")
    if not pd.api.types.is_datetime64_any_dtype(df["lock_time"]):
        df["lock_time"] = pd.to_datetime(df["lock_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    rows: List[Dict[str, object]] = []

    base = df.loc[df["lock_time"].notna(), ["order_number", "lock_time", "series_group_logic"]].copy()

    for g in target_groups:
        tp = time_periods.get(g, {}) or {}
        end_str = tp.get("end")
        finish_str = tp.get("finish")
        if not end_str:
            continue

        end_day = pd.Timestamp(end_str)
        if g == "CM0":
            end_day = end_day + pd.Timedelta(days=1)
        finish_day = pd.Timestamp(finish_str) if finish_str else end_day

        base_g = base.loc[base["series_group_logic"].eq(g)].copy()

        end_excl = end_day + pd.Timedelta(days=1)
        end_day_slice = base_g.loc[
            (base_g["lock_time"] >= end_day) & (base_g["lock_time"] < end_excl),
            ["order_number", "lock_time"],
        ].copy()
        if end_day_slice.empty:
            hourly_full = pd.Series([0] * 24, index=pd.RangeIndex(0, 24), dtype="int64")
        else:
            end_day_slice["hour"] = end_day_slice["lock_time"].dt.hour.astype("int64")
            hourly = end_day_slice.groupby("hour")["order_number"].nunique()
            hourly_full = hourly.reindex(range(24), fill_value=0).astype("int64")

        peak_hour = int(hourly_full.idxmax())
        peak_count = int(hourly_full.iloc[peak_hour])
        next_hour_count = int(hourly_full.iloc[peak_hour + 1]) if peak_hour < 23 else 0
        end_day_total = int(hourly_full.sum())

        finish_limit_excl = finish_day + pd.Timedelta(days=1)
        after_30d_end_excl = min(end_day + pd.Timedelta(days=31), finish_limit_excl)
        window_slice = base_g.loc[
            (base_g["lock_time"] >= end_day) & (base_g["lock_time"] < after_30d_end_excl),
            ["order_number", "lock_time"],
        ].copy()
        if window_slice.empty:
            after_30d_total = 0
        else:
            window_slice["date"] = window_slice["lock_time"].dt.floor("D")
            daily = window_slice.groupby("date")["order_number"].nunique()
            after_30d_total = int(daily.sum())

        delta_to_sat = (5 - end_day.weekday()) % 7
        if delta_to_sat == 0:
            delta_to_sat = 7
        weekend_start = end_day + pd.Timedelta(days=int(delta_to_sat))
        weekend_days = [weekend_start, weekend_start + pd.Timedelta(days=1)]
        weekend_total = 0
        for d in weekend_days:
            if d > finish_day:
                continue
            day_excl = d + pd.Timedelta(days=1)
            day_slice = base_g.loc[(base_g["lock_time"] >= d) & (base_g["lock_time"] < day_excl), "order_number"]
            weekend_total += int(day_slice.nunique())

        rows.append(
            {
                "series_group_logic": g,
                "上市日期": end_day.date().isoformat(),
                "峰值小时锁单数": peak_count,
                "峰值后第二小时锁单数": next_hour_count,
                "上市当日累计锁单数": end_day_total,
                "第一个周末锁单数": weekend_total,
                "上市后30日累计锁单数": after_30d_total,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(["__group_order"]).drop(columns="__group_order").reset_index(drop=True)
    return out


def _render_hourly_bar_figure(
    hourly_df: pd.DataFrame,
    target_groups: List[str],
    date_col: str,
    value_col: str,
    fig_title: str,
    y_title: str,
    subplot_date_label: str,
) -> go.Figure:
    summary = (
        hourly_df.groupby(["series_group_logic", date_col], as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "day_total"})
    )

    n_groups = len(target_groups)
    n_cols = 3
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[
            f"{g} ({subplot_date_label}: {summary.loc[summary['series_group_logic'].eq(g), date_col].iloc[0] if (summary['series_group_logic'].eq(g).any()) else ''})"
            for g in target_groups
        ],
    )

    positions = [(r + 1, c + 1) for r in range(n_rows) for c in range(n_cols)]
    for g, (r, c) in zip(target_groups, positions):
        dfg = hourly_df[hourly_df["series_group_logic"].eq(g)]
        x = dfg["hour"].tolist() if not dfg.empty else list(range(24))
        y = dfg[value_col].tolist() if not dfg.empty else [0] * 24
        fig.add_trace(go.Bar(x=x, y=y, name=g, showlegend=False), row=r, col=c)
        if x and y:
            peak_i = int(np.argmax(y))
            peak_x = x[peak_i]
            peak_y = int(y[peak_i])
            fig.add_annotation(
                x=peak_x,
                y=peak_y,
                text=str(peak_y),
                showarrow=False,
                yshift=10,
                bgcolor="rgba(255,255,255,0.7)",
                row=r,
                col=c,
            )
        fig.update_xaxes(title_text="Hour", tickmode="linear", dtick=1, row=r, col=c)
        fig.update_yaxes(title_text=y_title, row=r, col=c)

    fig.update_layout(height=360 * n_rows, title=fig_title, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def _render_daily_line_figure(
    daily_df: pd.DataFrame,
    target_groups: List[str],
    value_col: str,
    fig_title: str,
    y_title: str,
) -> go.Figure:
    n_groups = len(target_groups)
    n_cols = 3
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=target_groups,
    )

    positions = [(r + 1, c + 1) for r in range(n_rows) for c in range(n_cols)]
    for g, (r, c) in zip(target_groups, positions):
        dfg = daily_df[daily_df["series_group_logic"].eq(g)]
        if dfg.empty:
            continue
        x = dfg["days_from_start"].tolist()
        y = dfg[value_col].tolist()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=g, showlegend=False), row=r, col=c)
        fig.update_xaxes(title_text="Days from Start", row=r, col=c)
        fig.update_yaxes(title_text=y_title, row=r, col=c)

    fig.update_layout(height=360 * n_rows, title=fig_title, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def render_report(
    presale_hourly_df: pd.DataFrame,
    presale_summary_df: pd.DataFrame,
    presale_daily_df: pd.DataFrame,
    presale_retention_product_df: pd.DataFrame,
    presale_retention_summary_df: pd.DataFrame,
    presale_retention_region_df: pd.DataFrame,
    presale_retention_order_gender_df: pd.DataFrame,
    presale_retention_parent_region_df: pd.DataFrame,
    presale_retention_buyer_age_df: pd.DataFrame,
    imads_dist_df: pd.DataFrame,
    config_option_df: pd.DataFrame,
    listing_hourly_df: pd.DataFrame,
    listing_summary_df: pd.DataFrame,
    target_groups: List[str],
) -> str:
    css = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; color: #333; }
        h1 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; border-left: 5px solid #3498db; padding-left: 10px; }
        h3 { color: #2980b9; margin-top: 25px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        th, td { text-align: left; padding: 10px; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: 600; color: #555; }
        tr:hover { background-color: #f5f5f5; }
        .timestamp { color: #888; font-size: 0.9em; margin-bottom: 20px; }
        .summary-box { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    </style>
    """

    html_content: List[str] = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>订单分析报告</title>",
        css,
        "</head>",
        "<body>",
        "<h1>订单分析报告 (Order Data)</h1>",
        f"<div class='timestamp'>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>",
        "<h2>1. 预售期每小时小订数（按 series_group_logic）</h2>",
        "<div class='summary-box'>",
        "<p>口径：每个 series_group_logic 使用业务定义中的预售期 start 日期；统计该日期内每小时小订数（intention_payment_time 非空的 order_number 去重计数）。</p>",
        "</div>",
    ]

    if presale_hourly_df.empty:
        html_content.append("<p>⚠️ 未生成任何统计结果（可能缺少 time_periods 或数据列）。</p>")
        html_content.append("</body></html>")
        return "\n".join(html_content)

    html_content.append("<h3>1.1 汇总</h3>")
    if presale_summary_df is None or presale_summary_df.empty:
        html_content.append("<p>⚠️ 汇总表为空（可能缺少 time_periods 的 start/end 或分组无数据）。</p>")
    else:
        html_content.append(
            presale_summary_df.to_html(
                index=False,
                classes="table",
                escape=False,
                float_format=lambda x: "{:,.0f}".format(x) if isinstance(x, (int, float)) else x,
            )
        )
        html_content.append(
            "<div class='summary-box'>"
            "<p><b>备注</b></p>"
            "<ul>"
            "<li>预售日期：业务定义 time_periods 中的 start_date</li>"
            "<li>峰值小时小订数：取预售当日每小时小订数的峰值小时</li>"
            "<li>峰值后第二小时小订数：取峰值小时后的第二个小时（peak_hour + 1）</li>"
            "<li>第一个周末小订数：取 startday 后第一个双休日（周六+周日）两日求和</li>"
            "<li>第一周累计小订数：startday ~ startday+6 的累计求和，且不超过 end day</li>"
            "</ul>"
            "</div>"
        )

    html_content.append("<h3>1.2 可视化</h3>")
    fig1 = _render_hourly_bar_figure(
        presale_hourly_df,
        target_groups,
        date_col="start_date",
        value_col="intention_orders",
        fig_title="预售期开始日每小时小订数（series_group_logic）",
        y_title="Intention Orders",
        subplot_date_label="预售期开始日",
    )
    html_content.append(pio.to_html(fig1, full_html=False, include_plotlyjs="cdn"))

    html_content.append("<h3>1.3 预售周期日小订数</h3>")
    html_content.append("<div class='summary-box'>")
    html_content.append(
        "<p>口径：统计预售期（start ~ end）内每日小订数（intention_payment_time 非空的 order_number 去重计数）。X轴为距离预售开始日的天数。</p>"
    )
    html_content.append("</div>")
    if presale_daily_df.empty:
        html_content.append("<p>⚠️ 无数据。</p>")
    else:
        fig_daily = _render_daily_line_figure(
            presale_daily_df,
            target_groups,
            value_col="intention_orders",
            fig_title="预售周期日小订数（series_group_logic）",
            y_title="Intention Orders",
        )
        html_content.append(pio.to_html(fig_daily, full_html=False, include_plotlyjs="cdn"))

    html_content.append("<h3>1.4 历史预售至N日累计留存（分 product_name）与上市后30日锁单</h3>")
    html_content.append("<div class='summary-box'>")
    html_content.append(
        "<p>口径：N = end day - start day + 1。对每个 series_group_logic，统计预售开始日起至第N天（含）的小订订单，并以“截至N日未退订”作为留存：intention_refund_time 为空，或 intention_refund_time > 窗口结束时间。然后在这些订单中统计上市后30日（上市日 ~ 上市日+30）发生锁单（lock_time落在窗口内）的订单数。</p>"
    )
    html_content.append("</div>")
    if presale_retention_summary_df is None or presale_retention_summary_df.empty:
        html_content.append("<p>⚠️ 无数据（可能缺少 intention_refund_time / lock_time / product_name 或分组无数据）。</p>")
    else:
        html_content.append("<h4>1.4.1 汇总（按 series_group_logic）</h4>")
        html_content.append(
            presale_retention_summary_df.to_html(
                index=False,
                classes="table",
                escape=False,
                float_format=lambda x: "{:,.0f}".format(x) if isinstance(x, (int, float)) else x,
            )
        )
        html_content.append("<h4>1.4.2 明细（按 series_group_logic × product_name）</h4>")
        if presale_retention_product_df is None or presale_retention_product_df.empty:
            html_content.append("<p>⚠️ 明细表为空。</p>")
        else:
            html_content.append(
                presale_retention_product_df.to_html(
                    index=False,
                    classes="table",
                    escape=False,
                    float_format=lambda x: "{:,.0f}".format(x) if isinstance(x, (int, float)) else x,
                )
            )

        html_content.append("<h4>1.4.3 明细（按 series_group_logic × product_name × parent_region_name）</h4>")
        if presale_retention_region_df is None or presale_retention_region_df.empty:
            html_content.append("<p>⚠️ 明细表为空。</p>")
        else:
            for g in target_groups:
                dfg = presale_retention_region_df[presale_retention_region_df["series_group_logic"].eq(g)].copy()
                if dfg.empty:
                    continue
                n_val = int(dfg["预售周期（日）"].iloc[0]) if "预售周期（日）" in dfg.columns else None
                title = f"{g}（预售周期={n_val}）" if n_val is not None else g
                html_content.append(f"<h5>{title}</h5>")

                product_totals = (
                    dfg.groupby("product_name")["预售至N日累计留存小订数"].sum().sort_values(ascending=False)
                )
                region_totals = (
                    dfg.groupby("parent_region_name")["预售至N日累计留存小订数"].sum().sort_values(ascending=False)
                )
                pivot_cnt = dfg.pivot_table(
                    index="product_name",
                    columns="parent_region_name",
                    values="预售至N日累计留存小订数",
                    aggfunc="sum",
                    fill_value=0,
                )
                pivot_cnt = pivot_cnt.reindex(index=product_totals.index, columns=region_totals.index)
                html_content.append("<h6>预售至N日累计留存小订数</h6>")
                html_content.append(pivot_cnt.to_html(classes="table", escape=False))

                dfg["share_str"] = dfg["产品占比（该region内）"].apply(lambda x: f"{float(x):.1f}%")
                pivot_pct = dfg.pivot_table(
                    index="product_name",
                    columns="parent_region_name",
                    values="share_str",
                    aggfunc="first",
                    fill_value="",
                )
                pivot_pct = pivot_pct.reindex(index=product_totals.index, columns=region_totals.index)
                html_content.append("<h6>占比（该region内）</h6>")
                html_content.append(pivot_pct.to_html(classes="table", escape=False))

        html_content.append("<h4>1.5 明细（series_group_logic × order_gender）</h4>")
        html_content.append("<div class='summary-box'>")
        html_content.append(
            "<p>口径：基于“预售至N日累计留存小订数”的订单集合，按 order_gender 分组统计小订订单数与占比。</p>"
        )
        html_content.append("</div>")
        if presale_retention_order_gender_df is None or presale_retention_order_gender_df.empty:
            html_content.append("<p>⚠️ 明细表为空（可能缺少 order_gender 或无数据）。</p>")
        else:
            gender_show = presale_retention_order_gender_df.copy()
            gender_show["占比"] = gender_show["占比"].apply(lambda x: f"{float(x):.1f}%")
            html_content.append(
                gender_show.to_html(
                    index=False,
                    classes="table",
                    escape=False,
                    float_format=lambda x: "{:,.0f}".format(x) if isinstance(x, (int, float)) else x,
                )
            )

        html_content.append("<h4>1.6 明细（series_group_logic × parent_region_name）</h4>")
        html_content.append("<div class='summary-box'>")
        html_content.append(
            "<p>口径：基于“预售至N日累计留存小订数”的订单集合，按 parent_region_name 分组统计小订订单数与占比；并在上市日(d)补充“在营门店数”（近30天有活动且已开店的门店数），在营门店数由固定算子 active_store 计算。</p>"
        )
        html_content.append("</div>")
        if presale_retention_parent_region_df is None or presale_retention_parent_region_df.empty:
            html_content.append("<p>⚠️ 明细表为空（可能缺少 parent_region_name 或无数据）。</p>")
        else:
            region_show = presale_retention_parent_region_df.copy()
            region_show["占比"] = region_show["占比"].apply(lambda x: f"{float(x):.1f}%")
            html_content.append(
                region_show.to_html(
                    index=False,
                    classes="table",
                    escape=False,
                    float_format=lambda x: "{:,.0f}".format(x) if isinstance(x, (int, float)) else x,
                )
            )

        html_content.append("<h4>1.7 可视化（series_group_logic × buyer_age：年龄占比）</h4>")
        html_content.append("<div class='summary-box'>")
        html_content.append(
            "<p>口径：基于“预售至N日累计留存小订数”的订单集合，按 buyer_age 计算占比；横轴为年龄，纵轴为占比（%）。</p>"
        )
        html_content.append("</div>")
        if presale_retention_buyer_age_df is None or presale_retention_buyer_age_df.empty:
            html_content.append("<p>⚠️ 无数据（可能缺少 buyer_age 或无数据）。</p>")
        else:
            fig_buyer_age = _render_age_share_line_figure(
                presale_retention_buyer_age_df,
                target_groups,
                age_label="buyer_age",
                fig_title="预售至N日累计留存：Buyer Age 占比（series_group_logic）",
            )
            html_content.append(pio.to_html(fig_buyer_age, full_html=False, include_plotlyjs="cdn"))

        html_content.append("<h4>1.8 IMADS（线控转向 OP-Steer）分区域选装率</h4>")
        html_content.append("<div class='summary-box'>")
        html_content.append("<p>口径：加载 processed/LS8_Configuration_Details_transposed.csv，以 order_number 与目标订单数据集的 order_number 匹配；读取列 OP-Steer 的“是/否”，按 parent_region_name 汇总并计算选装率（是/预选配总数；是/高配66总数）。</p>")
        html_content.append("</div>")
        if imads_dist_df is None or imads_dist_df.empty:
            html_content.append("<p>⚠️ 无数据（可能缺少 OP-Steer 列或匹配不到订单）。</p>")
        else:
            cols = [
                "parent_region_name",
                "Value Dispaly Name=是",
                "Value Dispaly Name=否",
                "小订总数",
                "高配（66ultra）总数",
                "线控选装率",
                "线控在（66ultra）选装率",
            ]
            for g in target_groups:
                df_g = imads_dist_df[imads_dist_df["series_group_logic"].eq(g)].copy()
                if df_g.empty:
                    continue
                html_content.append(f"<h5>{g}</h5>")
                show = df_g.loc[:, [c for c in cols if c in df_g.columns]].copy()
                html_content.append(
                    show.to_html(
                        index=False,
                        classes="table",
                        escape=False,
                        float_format=lambda x: "{:,.0f}".format(x) if isinstance(x, (int, float)) else x,
                    )
                )

        html_content.append("<h4>1.9 配置选配情况（转置配置表）</h4>")
        html_content.append("<div class='summary-box'>")
        html_content.append(
            "<p>口径：加载 processed/LS8_Configuration_Details_transposed.csv；以 order_number 与“预售至N日累计留存小订数”的订单集合匹配；对各 Attribute（列名）下不同 Value Dispaly Name（字段值）统计订单数与选配比例（订单数/预售至N日累计留存小订数）；中文名称来自 business_definition.json 的 configuration 映射。</p>"
        )
        html_content.append("</div>")
        if config_option_df is None or config_option_df.empty:
            html_content.append("<p>⚠️ 无数据（可能缺少转置配置文件、列名不一致或匹配不到留存订单）。</p>")
        else:
            cols = ["Attribute", "中文名称", "Value Dispaly Name", "订单数", "选配比例"]
            for g in target_groups:
                df_g = config_option_df[config_option_df["series_group_logic"].eq(g)].copy()
                if df_g.empty:
                    continue
                html_content.append(f"<h5>{g}</h5>")
                show = df_g.loc[:, [c for c in cols if c in df_g.columns]].copy()
                html_content.append(
                    show.to_html(
                        index=False,
                        classes="table",
                        escape=False,
                        float_format=lambda x: "{:,.0f}".format(x) if isinstance(x, (int, float)) else x,
                    )
                )

    html_content.append("<h2>2. 上市期每小时锁单数（按 series_group_logic）</h2>")
    html_content.append("<div class='summary-box'>")
    html_content.append(
        "<p>口径：先用业务定义 series_group_logic 根据 product_name 对订单归类；然后对每个 series_group_logic 使用业务定义 time_periods 中的 end 日期（上市日期），统计该日期内每小时锁单数（lock_time 非空的 order_number 去重计数）。</p>"
    )
    html_content.append("</div>")

    html_content.append("<h3>2.1 汇总</h3>")
    if listing_summary_df is None or listing_summary_df.empty:
        html_content.append("<p>⚠️ 汇总表为空（可能缺少 time_periods 的 end/finish 或分组无数据）。</p>")
    else:
        html_content.append(
            listing_summary_df.to_html(
                index=False,
                classes="table",
                escape=False,
                float_format=lambda x: "{:,.0f}".format(x) if isinstance(x, (int, float)) else x,
            )
        )
        html_content.append(
            "<div class='summary-box'>"
            "<p><b>备注</b></p>"
            "<ul>"
            "<li>上市日期：业务定义 time_periods 中的 end_date；其中 CM0 特殊处理：上市日期取 end_date + 1（事故）</li>"
            "<li>峰值小时锁单数：取上市当日每小时锁单数的峰值小时</li>"
            "<li>峰值后第二小时锁单数：取峰值小时后的第二个小时（peak_hour + 1）</li>"
            "<li>第一个周末锁单数：取 endday 后第一个双休日（周六+周日）两日求和（不超过 finish day）</li>"
            "<li>上市后30日累计锁单数：endday ~ endday+30 的累计求和，且不超过 finish day</li>"
            "</ul>"
            "</div>"
        )

    html_content.append("<h3>2.2 可视化</h3>")
    fig2 = _render_hourly_bar_figure(
        listing_hourly_df,
        target_groups,
        date_col="end_date",
        value_col="lock_orders",
        fig_title="上市日期每小时锁单数（series_group_logic）",
        y_title="Lock Orders",
        subplot_date_label="上市日期",
    )
    html_content.append(pio.to_html(fig2, full_html=False, include_plotlyjs="cdn"))
    html_content.append("</body></html>")
    return "\n".join(html_content)



def main() -> int:
    parser = argparse.ArgumentParser(description="生成车型预售/上市分析报告")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=ALL_TARGET_GROUPS,
        help="指定要分析的车型（可多选），例如：--models LS8 LS9"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"输出HTML文件路径（默认：{DEFAULT_OUTPUT}）"
    )
    args = parser.parse_args()
    
    target_groups = args.models if args.models else ALL_TARGET_GROUPS
    output_path = Path(args.output)
    
    print(f"📊 分析车型: {', '.join(target_groups)}")
    print(f"📁 输出文件: {output_path}")
    
    business_def = load_business_definition(BUSINESS_DEF_FILE)
    df = load_data(PARQUET_FILE)
    df = apply_series_group_logic(df, business_def)

    presale_hourly_df = build_hourly_intention_counts(df, business_def, target_groups)
    presale_summary_df = build_presale_summary(df, business_def, target_groups)
    presale_daily_df = build_presale_daily_counts(df, business_def, target_groups)
    presale_retention_product_df, presale_retention_summary_df = build_presale_retention_by_product(
        df, business_def, target_groups
    )
    presale_retention_region_df = build_presale_retention_region_detail(df, business_def, target_groups)
    presale_retention_order_gender_df = build_presale_retention_category_detail(
        df, business_def, target_groups, category_col="order_gender", category_label="order_gender"
    )
    presale_retention_parent_region_df = build_presale_retention_region_summary_with_active_stores(
        df, business_def, target_groups
    )
    presale_retention_buyer_age_df = build_presale_retention_age_detail(
        df, business_def, target_groups, age_col="buyer_age", age_label="buyer_age"
    )
    df_imads = load_op_steer_from_transposed(OP_STEER_TRANSPOSED_FILE)
    imads_dist_df = build_imads_op_steer_region_summary(df, business_def, target_groups, df_imads)
    df_transposed = load_transposed_configuration_data(OP_STEER_TRANSPOSED_FILE)
    config_option_df = build_transposed_configuration_option_summary(df, business_def, target_groups, df_transposed)
    listing_hourly_df = build_hourly_lock_counts(df, business_def, target_groups)
    listing_summary_df = build_listing_summary(df, business_def, target_groups)

    html = render_report(
        presale_hourly_df,
        presale_summary_df,
        presale_daily_df,
        presale_retention_product_df,
        presale_retention_summary_df,
        presale_retention_region_df,
        presale_retention_order_gender_df,
        presale_retention_parent_region_df,
        presale_retention_buyer_age_df,
        imads_dist_df,
        config_option_df,
        listing_hourly_df,
        listing_summary_df,
        target_groups,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"✅ 已生成报告: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
