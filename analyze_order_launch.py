#!/usr/bin/env python3
"""
车型上市期锁单分析报告生成工具

功能：
1. 统计各 series_group_logic 上市日（time_periods.end）每小时锁单数
2. 统计上市期汇总指标（上市当日累计、峰值小时、周末、上市后30日累计等）
3. 统计上市后每日锁单数（默认上市日起至上市后30日或 finish）
4. 生成 HTML 可视化报告
"""

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

_missing = [m for m in ["numpy", "pandas", "plotly"] if importlib.util.find_spec(m) is None]
_has_parquet_engine = (
    importlib.util.find_spec("pyarrow") is not None
    or importlib.util.find_spec("fastparquet") is not None
)
if _missing:
    print("💥 缺少依赖模块，无法运行 analyze_order_launch.py")
    print(f"缺少模块: {', '.join(_missing)}")
    print("请先安装依赖后重试，例如：")
    print(f"{sys.executable} -m pip install numpy pandas plotly")
    raise SystemExit(1)
if not _has_parquet_engine:
    print("💥 缺少 Parquet 读取引擎，无法运行 analyze_order_launch.py")
    print("缺少模块: pyarrow 或 fastparquet")
    print("请先安装依赖后重试，例如：")
    print(f"{sys.executable} -m pip install pyarrow")
    raise SystemExit(1)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

PARQUET_FILE = Path("/Users/zihao_/Documents/coding/dataset/formatted/order_data.parquet")
CONFIG_ATTRIBUTE_FILE = Path("/Users/zihao_/Documents/coding/dataset/formatted/config_attribute.parquet")
BUSINESS_DEF_FILE = Path("/Users/zihao_/Documents/github/26W06_Tool_calls/schema/business_definition.json")
SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT = SCRIPT_DIR / "reports" / "analyze_order_launch.html"
ALL_TARGET_GROUPS = ["CM0", "DM0", "CM1", "DM1", "CM2", "LS9", "LS8"]


def load_business_definition(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    print(f"📖 Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"✅ Loaded {len(df)} rows.")
    return df


def load_configuration_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    last_modified = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    df = pd.read_parquet(file_path)
    print(f"📖 Loaded configuration data from {file_path} (Last modified: {last_modified})")
    return df


def _standardize_config_attribute_long(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    rename_map = {}
    if "Order Number" in df.columns:
        rename_map["Order Number"] = "order_number"
    elif "order_number" in df.columns:
        rename_map["order_number"] = "order_number"
    if "Attribute" in df.columns:
        rename_map["Attribute"] = "Attribute"
    elif "Attribute Code" in df.columns:
        rename_map["Attribute Code"] = "Attribute"
    if "Value Dispaly Name" in df.columns:
        rename_map["Value Dispaly Name"] = "Value Dispaly Name"
    elif "value" in df.columns:
        rename_map["value"] = "Value Dispaly Name"

    df = df.rename(columns=rename_map)
    required = {"order_number", "Attribute", "Value Dispaly Name"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    out = df.loc[:, ["order_number", "Attribute", "Value Dispaly Name"]].copy()
    out["order_number"] = out["order_number"].astype("string")
    out["Attribute"] = out["Attribute"].astype("string").str.strip()
    out["Value Dispaly Name"] = out["Value Dispaly Name"].astype("string").str.strip()
    out = out.dropna(subset=["order_number"])
    return out


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


def build_daily_lock_counts(df: pd.DataFrame, business_def: dict, target_groups: List[str]) -> pd.DataFrame:
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

        finish_limit_excl = finish_day + pd.Timedelta(days=1)
        after_30d_end_excl = min(end_day + pd.Timedelta(days=31), finish_limit_excl)

        base_g = base.loc[base["series_group_logic"].eq(g), ["order_number", "lock_time"]].copy()
        window_slice = base_g.loc[
            (base_g["lock_time"] >= end_day) & (base_g["lock_time"] < after_30d_end_excl),
            ["order_number", "lock_time"],
        ].copy()

        date_index = pd.date_range(start=end_day.floor("D"), end=(after_30d_end_excl - pd.Timedelta(days=1)).floor("D"), freq="D")

        if window_slice.empty:
            daily_full = pd.Series([0] * len(date_index), index=date_index, dtype="int64")
        else:
            window_slice["date"] = window_slice["lock_time"].dt.floor("D")
            daily = window_slice.groupby("date")["order_number"].nunique()
            daily_full = daily.reindex(date_index, fill_value=0).astype("int64")

        for d, cnt in daily_full.items():
            rows.append(
                {
                    "series_group_logic": g,
                    "end_date": end_day.date().isoformat(),
                    "date": d.date().isoformat(),
                    "lock_orders": int(cnt),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(["__group_order", "date"]).drop(columns="__group_order").reset_index(drop=True)
    return out


def build_model_mix(
    df: pd.DataFrame, business_def: dict, target_groups: List[str], staff_orders: set
) -> pd.DataFrame:
    if "order_number" not in df.columns:
        raise KeyError("数据缺少列: order_number")
    if "series_group_logic" not in df.columns:
        raise KeyError("数据缺少列: series_group_logic")
    if "product_name" not in df.columns:
        raise KeyError("数据缺少列: product_name")
    if "lock_time" not in df.columns:
        raise KeyError("数据缺少列: lock_time")

    if not pd.api.types.is_datetime64_any_dtype(df["lock_time"]):
        df["lock_time"] = pd.to_datetime(df["lock_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    rows: List[Dict[str, object]] = []

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

        finish_limit_excl = finish_day + pd.Timedelta(days=1)
        after_30d_end_excl = min(end_day + pd.Timedelta(days=31), finish_limit_excl)

        window = df.loc[
            df["series_group_logic"].eq(g)
            & df["lock_time"].notna()
            & (df["lock_time"] >= end_day)
            & (df["lock_time"] < after_30d_end_excl),
            ["order_number", "product_name"],
        ].copy()
        if staff_orders:
            window = window.loc[~window["order_number"].astype("string").isin(staff_orders)].copy()
        if window.empty:
            continue

        window["product_name"] = window["product_name"].fillna("").astype("string")
        agg_g = (
            window.groupby(["product_name"], as_index=False)["order_number"]
            .nunique()
            .rename(columns={"order_number": "锁单数"})
        )
        total = int(agg_g["锁单数"].sum())
        if total <= 0:
            continue
        agg_g["series_group_logic"] = g
        agg_g["占比"] = (agg_g["锁单数"] / total).fillna(0.0).map(lambda x: f"{x:.1%}")
        rows.extend(agg_g[["series_group_logic", "product_name", "锁单数", "占比"]].to_dict("records"))

    if not rows:
        return pd.DataFrame(columns=["series_group_logic", "product_name", "锁单数", "占比"])

    agg = pd.DataFrame(rows)

    group_order = {g: i for i, g in enumerate(target_groups)}
    agg["__group_order"] = agg["series_group_logic"].map(group_order).fillna(len(group_order))
    agg = (
        agg.sort_values(["__group_order", "锁单数", "product_name"], ascending=[True, False, True])
        .drop(columns="__group_order")
        .reset_index(drop=True)
    )
    return agg


def build_configuration_version_summary(
    df_orders: pd.DataFrame,
    business_def: dict,
    target_groups: List[str],
    config_long_df: pd.DataFrame,
    staff_orders: set,
) -> pd.DataFrame:
    if config_long_df is None or config_long_df.empty:
        return pd.DataFrame()
    if "order_number" not in df_orders.columns:
        raise KeyError("数据缺少列: order_number")
    if "series_group_logic" not in df_orders.columns:
        raise KeyError("数据缺少列: series_group_logic")
    if "lock_time" not in df_orders.columns:
        raise KeyError("数据缺少列: lock_time")
    if not pd.api.types.is_datetime64_any_dtype(df_orders["lock_time"]):
        df_orders["lock_time"] = pd.to_datetime(df_orders["lock_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})

    rows: List[pd.DataFrame] = []
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

        finish_limit_excl = finish_day + pd.Timedelta(days=1)
        after_30d_end_excl = min(end_day + pd.Timedelta(days=31), finish_limit_excl)

        m_lock_window = (
            df_orders["series_group_logic"].eq(g)
            & df_orders["lock_time"].notna()
            & (df_orders["lock_time"] >= end_day)
            & (df_orders["lock_time"] < after_30d_end_excl)
        )
        if staff_orders:
            m_lock_window = m_lock_window & ~df_orders["order_number"].astype("string").isin(staff_orders)
        lock_orders = (
            df_orders.loc[m_lock_window, ["order_number"]]
            .dropna(subset=["order_number"])
            .drop_duplicates(subset=["order_number"])
            .copy()
        )
        lock_orders["order_number"] = lock_orders["order_number"].astype("string")
        if lock_orders.empty:
            continue
        series_total = int(lock_orders["order_number"].nunique())

        matched_lock = config_long_df.merge(lock_orders, left_on="order_number", right_on="order_number", how="inner")
        if matched_lock.empty:
            continue
        matched_lock = matched_lock.loc[:, ["order_number", "Attribute", "Value Dispaly Name"]].copy()
        matched_lock["Attribute"] = matched_lock["Attribute"].astype("string").str.strip()
        matched_lock["Value Dispaly Name"] = matched_lock["Value Dispaly Name"].astype("string").fillna("").str.strip()
        matched_lock = matched_lock[matched_lock["Value Dispaly Name"].ne("") & matched_lock["Attribute"].ne("Is Staff")]

        lock_agg = (
            matched_lock.groupby(["Attribute", "Value Dispaly Name"], as_index=False)
            .agg(锁单数=("order_number", "nunique"))
            .reset_index(drop=True)
        )

        attr_totals = lock_agg.groupby("Attribute")["锁单数"].transform("sum")
        ratio = (lock_agg["锁单数"] / attr_totals).fillna(0.0)
        lock_agg["比例"] = ratio.map(lambda x: f"{x:.1%}")
        lock_agg["选装率"] = (lock_agg["锁单数"] / float(series_total) if series_total > 0 else 0.0).fillna(0.0).map(
            lambda x: f"{x:.1%}"
        )

        out_g = lock_agg.copy()
        out_g.insert(0, "series_group_logic", g)
        out_g = out_g.sort_values(["Attribute", "锁单数"], ascending=[True, False]).reset_index(drop=True)
        rows.append(out_g)

    if not rows:
        return pd.DataFrame()
    group_order = {g: i for i, g in enumerate(target_groups)}
    out = pd.concat(rows, ignore_index=True)
    out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
    out = out.sort_values(["__group_order", "Attribute", "锁单数"], ascending=[True, True, False]).drop(columns="__group_order")
    return out


def build_after30d_lock_totals(df: pd.DataFrame, business_def: dict, target_groups: List[str], staff_orders: set) -> pd.DataFrame:
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

    for g in target_groups:
        tp = time_periods.get(g, {}) or {}
        end_str = tp.get("end")
        finish_str = tp.get("finish")
        if not end_str:
            rows.append({"series_group_logic": g, "锁单数": 0})
            continue

        end_day = pd.Timestamp(end_str)
        if g == "CM0":
            end_day = end_day + pd.Timedelta(days=1)
        finish_day = pd.Timestamp(finish_str) if finish_str else end_day

        finish_limit_excl = finish_day + pd.Timedelta(days=1)
        after_30d_end_excl = min(end_day + pd.Timedelta(days=31), finish_limit_excl)

        m = (
            df["series_group_logic"].eq(g)
            & df["lock_time"].notna()
            & (df["lock_time"] >= end_day)
            & (df["lock_time"] < after_30d_end_excl)
        )
        if staff_orders:
            m = m & ~df["order_number"].astype("string").isin(staff_orders)
        cnt = int(df.loc[m, "order_number"].nunique())
        rows.append({"series_group_logic": g, "锁单数": cnt})

    out = pd.DataFrame(rows)
    group_order = {g: i for i, g in enumerate(target_groups)}
    out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
    out = out.sort_values(["__group_order"]).drop(columns="__group_order").reset_index(drop=True)
    return out


def build_after30d_gender_detail(
    df: pd.DataFrame, business_def: dict, target_groups: List[str], staff_orders: set
) -> pd.DataFrame:
    required_cols = ["lock_time", "order_number", "series_group_logic", "order_gender"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"数据缺少列: {', '.join(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(df["lock_time"]):
        df["lock_time"] = pd.to_datetime(df["lock_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    rows: List[Dict[str, object]] = []
    raw_gender = df["order_gender"].astype("string").fillna("").str.strip()
    raw_gender_u = raw_gender.str.upper()
    gender_series = pd.Series("未知", index=df.index, dtype="string")
    gender_series = gender_series.where(~raw_gender.isin(["男", "男性"]), "男")
    gender_series = gender_series.where(~raw_gender.isin(["女", "女性"]), "女")
    gender_series = gender_series.where(~raw_gender_u.isin(["M", "MALE", "1"]), "男")
    gender_series = gender_series.where(~raw_gender_u.isin(["F", "FEMALE", "2"]), "女")

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

        finish_limit_excl = finish_day + pd.Timedelta(days=1)
        after_30d_end_excl = min(end_day + pd.Timedelta(days=31), finish_limit_excl)

        m = (
            df["series_group_logic"].eq(g)
            & df["lock_time"].notna()
            & (df["lock_time"] >= end_day)
            & (df["lock_time"] < after_30d_end_excl)
        )
        if staff_orders:
            m = m & ~df["order_number"].astype("string").isin(staff_orders)
        locked = df.loc[m, ["order_number"]].dropna(subset=["order_number"]).copy()
        if locked.empty:
            continue
        locked = locked.drop_duplicates(subset=["order_number"])
        locked["order_gender"] = gender_series.loc[locked.index]

        total_all = int(locked["order_number"].nunique())

        locked_known = locked.loc[locked["order_gender"].isin(["男", "女"])].copy()
        total_known = int(locked_known["order_number"].nunique())
        by_gender = (
            locked_known.groupby("order_gender")["order_number"]
            .nunique()
            .reindex(["男", "女"])
            .dropna()
            .astype("int64")
        )

        for gender_val, cnt in by_gender.items():
            share_known = round(cnt / total_known * 100, 1) if total_known > 0 else 0.0
            share_total = round(cnt / total_all * 100, 1) if total_all > 0 else 0.0
            rows.append(
                {
                    "series_group_logic": g,
                    "order_gender": str(gender_val),
                    "锁单数": int(cnt),
                    "占比": share_known,
                    "合计百分比": share_total,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(["__group_order", "锁单数"], ascending=[True, False]).drop(columns="__group_order")
    return out.reset_index(drop=True)


def build_after30d_age_detail(
    df: pd.DataFrame, business_def: dict, target_groups: List[str], staff_orders: set, age_col: str, age_label: str
) -> pd.DataFrame:
    required_cols = ["lock_time", "order_number", "series_group_logic", age_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"数据缺少列: {', '.join(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(df["lock_time"]):
        df["lock_time"] = pd.to_datetime(df["lock_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    rows: List[Dict[str, object]] = []

    age_num = pd.to_numeric(df[age_col], errors="coerce")
    age_int = age_num.round().astype("Int64")

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

        finish_limit_excl = finish_day + pd.Timedelta(days=1)
        after_30d_end_excl = min(end_day + pd.Timedelta(days=31), finish_limit_excl)

        m = (
            df["series_group_logic"].eq(g)
            & df["lock_time"].notna()
            & (df["lock_time"] >= end_day)
            & (df["lock_time"] < after_30d_end_excl)
        )
        if staff_orders:
            m = m & ~df["order_number"].astype("string").isin(staff_orders)

        locked = df.loc[m, ["order_number"]].dropna(subset=["order_number"]).copy()
        if locked.empty:
            continue
        locked = locked.drop_duplicates(subset=["order_number"])
        locked[age_label] = age_int.loc[locked.index]

        total = int(locked["order_number"].nunique())
        by_age = locked.dropna(subset=[age_label]).groupby(age_label)["order_number"].nunique().sort_index()
        for age_val, cnt in by_age.items():
            share = round(cnt / total * 100, 1) if total > 0 else 0.0
            rows.append(
                {
                    "series_group_logic": g,
                    age_label: int(age_val) if pd.notna(age_val) else None,
                    "锁单数": int(cnt),
                    "占比": share,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(["__group_order", age_label]).drop(columns="__group_order").reset_index(drop=True)
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
    fig_title: str,
    y_title: str,
) -> go.Figure:
    summary = (
        daily_df.groupby(["series_group_logic", "end_date"], as_index=False)["lock_orders"]
        .sum()
        .rename(columns={"lock_orders": "window_total"})
    )

    n_groups = len(target_groups)
    n_cols = 3
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[
            f"{g} (上市日期: {summary.loc[summary['series_group_logic'].eq(g), 'end_date'].iloc[0] if (summary['series_group_logic'].eq(g).any()) else ''})"
            for g in target_groups
        ],
    )

    positions = [(r + 1, c + 1) for r in range(n_rows) for c in range(n_cols)]
    for g, (r, c) in zip(target_groups, positions):
        dfg = daily_df[daily_df["series_group_logic"].eq(g)]
        x = dfg["date"].tolist()
        y = dfg["lock_orders"].tolist()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=g, showlegend=False), row=r, col=c)
        if y:
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
        fig.update_xaxes(title_text="Date", tickangle=-45, row=r, col=c)
        fig.update_yaxes(title_text=y_title, row=r, col=c)

    fig.update_layout(height=400 * n_rows, title=fig_title, margin=dict(l=40, r=20, t=60, b=80))
    return fig


def _render_age_share_line_figure(
    age_df: pd.DataFrame,
    target_groups: List[str],
    age_label: str,
    fig_title: str,
) -> go.Figure:
    n_groups = len(target_groups)
    n_cols = 3
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=target_groups)

    positions = [(r + 1, c + 1) for r in range(n_rows) for c in range(n_cols)]
    for g, (r, c) in zip(target_groups, positions):
        dfg = age_df[age_df["series_group_logic"].eq(g)].copy()
        if dfg.empty:
            continue
        dfg = dfg.dropna(subset=[age_label, "占比", "锁单数"]).sort_values(age_label)
        if dfg.empty:
            continue

        x = dfg[age_label].astype(int).tolist()
        y = dfg["占比"].astype(float).tolist()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", showlegend=False), row=r, col=c)

        weights = dfg["锁单数"].astype(float).tolist()
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
                y=max(y) if y else 0,
                text=f"Avg {avg_age:.1f}",
                showarrow=False,
                yshift=10,
                bgcolor="rgba(255,255,255,0.7)",
                row=r,
                col=c,
            )

        ymax = max([float(max(y)) if y else 0.0, 10.0])
        ymax = float(np.ceil(ymax / 5.0) * 5.0) if ymax > 0 else 10.0
        fig.update_xaxes(title_text=age_label, tickmode="linear", dtick=5, row=r, col=c)
        fig.update_yaxes(title_text="Share (%)", range=[0, ymax], row=r, col=c)

    fig.update_layout(height=360 * n_rows, title=fig_title, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def render_launch_report(
    listing_hourly_df: pd.DataFrame,
    listing_daily_df: pd.DataFrame,
    listing_summary_df: pd.DataFrame,
    model_mix_df: pd.DataFrame,
    config_version_df: pd.DataFrame,
    gender_df: pd.DataFrame,
    age_df: pd.DataFrame,
    lock_totals_df: pd.DataFrame,
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
        "<title>上市期锁单分析</title>",
        css,
        "</head>",
        "<body>",
        "<h1>上市期锁单分析报告</h1>",
        f"<div class='timestamp'>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>",
        "<h2>2. 上市期锁单数</h2>",
        "<div class='summary-box'>",
        "<p>口径：先用业务定义 series_group_logic 根据 product_name 对订单归类；然后对每个 series_group_logic 使用业务定义 time_periods 中的 end 日期（上市日期），统计上市日期每小时锁单数，并展示上市日起（至上市后30日或 finish）每天锁单数（lock_time 非空的 order_number 去重计数）。</p>",
        "</div>",
    ]

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

    html_content.append("<h3>2.2 上市日期每小时锁单数（series_group_logic）</h3>")
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

    html_content.append("<h3>2.3 上市日期每天锁单数（series_group_logic）</h3>")
    fig3 = _render_daily_line_figure(
        listing_daily_df,
        target_groups,
        fig_title="上市日期每天锁单数（series_group_logic）",
        y_title="Lock Orders",
    )
    html_content.append(pio.to_html(fig3, full_html=False, include_plotlyjs=False))

    html_content.append("<h2>5. 订单结构</h2>")
    lock_lines: List[str] = []
    if lock_totals_df is not None and not lock_totals_df.empty and {"series_group_logic", "锁单数"}.issubset(set(lock_totals_df.columns)):
        m = lock_totals_df.set_index("series_group_logic")["锁单数"].to_dict()
        lock_lines = [f"{g}：{int(m.get(g, 0))}" for g in target_groups]
    html_content.append("<div class='summary-box'>")
    html_content.append(
        "<p>口径：基于各 series_group_logic 的上市日期 end_day（见业务定义 time_periods），统计上市后30日累计窗口 "
        "end_day ~ min(end_day+30, finish) 的锁单订单（lock_time 非空的 order_number 去重计数）；"
        "并在此基础上剔除 is_staff=True 的订单（即仅保留 is_staff=False）。</p>"
    )
    if lock_lines:
        html_content.append("<p><b>本口径锁单数</b></p>")
        html_content.append("<pre>" + "\n".join(lock_lines) + "</pre>")
    html_content.append("</div>")
    html_content.append("<h3>5.1 车型Mix（按 series_group_logic × product_name）</h3>")
    if model_mix_df is None or model_mix_df.empty:
        html_content.append("<p>⚠️ 车型Mix表为空（可能无锁单数据或缺少必要字段）。</p>")
    else:
        for g in target_groups:
            html_content.append(f"<h4>{g}</h4>")
            if "series_group_logic" not in model_mix_df.columns:
                html_content.append("<p>⚠️ 车型Mix表缺少 series_group_logic 列。</p>")
                break
            dfg = model_mix_df.loc[model_mix_df["series_group_logic"].eq(g)].copy()
            if dfg.empty:
                html_content.append("<p>⚠️ 无数据</p>")
                continue
            dfg = dfg.drop(columns=["series_group_logic"])
            html_content.append(dfg.to_html(index=False, classes="table", escape=False))

    html_content.append("<h3>5.2 配置版本（按 series_group_logic × Attribute）</h3>")
    if config_version_df is None or config_version_df.empty:
        html_content.append("<p>⚠️ 配置版本表为空（可能无配置数据或缺少必要字段）。</p>")
    else:
        cols = ["Attribute", "Value Dispaly Name", "锁单数", "比例", "选装率"]
        for g in target_groups:
            html_content.append(f"<h4>{g}</h4>")
            if "series_group_logic" not in config_version_df.columns:
                html_content.append("<p>⚠️ 配置版本表缺少 series_group_logic 列。</p>")
                break
            dfg = config_version_df.loc[config_version_df["series_group_logic"].eq(g)].copy()
            if dfg.empty:
                html_content.append("<p>⚠️ 无数据</p>")
                continue
            for c in cols:
                if c not in dfg.columns:
                    html_content.append(f"<p>⚠️ 配置版本表缺少列: {c}</p>")
                    dfg = None
                    break
            if dfg is None:
                break
            dfg = dfg.loc[:, cols].copy()
            html_content.append(dfg.to_html(index=False, classes="table", escape=False))

    html_content.append("<h3>5.3 性别（series_group_logic × order_gender）</h3>")
    html_content.append("<div class='summary-box'>")
    html_content.append(
        "<p>口径：使用“本口径锁单数”的订单集合（after_30d_total 窗口 + is_staff=False）。性别仅保留 男/女；占比按（男+女）计算，剔除未知。合计百分比按（男/女）各自锁单数 / 该车系整体锁单数计算。</p>"
    )
    html_content.append("</div>")
    if gender_df is None or gender_df.empty:
        html_content.append("<p>⚠️ 无数据（可能缺少 order_gender 或无数据）。</p>")
    else:
        cols = ["order_gender", "锁单数", "占比", "合计百分比"]
        for g in target_groups:
            html_content.append(f"<h4>{g}</h4>")
            if "series_group_logic" not in gender_df.columns:
                html_content.append("<p>⚠️ 性别明细表缺少 series_group_logic 列。</p>")
                break
            dfg = gender_df.loc[gender_df["series_group_logic"].eq(g)].copy()
            if dfg.empty:
                html_content.append("<p>⚠️ 无数据</p>")
                continue
            dfg["占比"] = dfg["占比"].apply(lambda x: f"{float(x):.1f}%")
            if "合计百分比" in dfg.columns:
                dfg["合计百分比"] = dfg["合计百分比"].apply(lambda x: f"{float(x):.1f}%")
            dfg = dfg.loc[:, [c for c in cols if c in dfg.columns]].copy()
            html_content.append(dfg.to_html(index=False, classes="table", escape=False))

    html_content.append("<h3>5.4 年龄（series_group_logic × buyer_age）</h3>")
    html_content.append("<div class='summary-box'>")
    html_content.append(
        "<p>口径：使用“本口径锁单数”的订单集合（after_30d_total 窗口 + is_staff=False），按 buyer_age 计算占比；横轴为年龄，纵轴为占比（%）。</p>"
    )
    html_content.append("</div>")
    if age_df is None or age_df.empty:
        html_content.append("<p>⚠️ 无数据（可能缺少 buyer_age 或无数据）。</p>")
    else:
        fig_age = _render_age_share_line_figure(
            age_df,
            target_groups,
            age_label="buyer_age",
            fig_title="上市后30日锁单：Buyer Age 占比（series_group_logic）",
        )
        html_content.append(pio.to_html(fig_age, full_html=False, include_plotlyjs=False))
    html_content.append("</body></html>")
    return "\n".join(html_content)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成车型上市期锁单分析报告")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=ALL_TARGET_GROUPS,
        help="指定要分析的车型（可多选），例如：--models LS8 LS9",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"输出HTML文件路径（默认：{DEFAULT_OUTPUT}）",
    )
    args = parser.parse_args()

    target_groups = args.models if args.models else ALL_TARGET_GROUPS
    output_path = Path(args.output)

    print(f"📊 分析车型: {', '.join(target_groups)}")
    print(f"📁 输出文件: {output_path}")

    business_def = load_business_definition(BUSINESS_DEF_FILE)
    df = load_data(PARQUET_FILE)
    df = apply_series_group_logic(df, business_def)

    config_raw_df = load_configuration_data(CONFIG_ATTRIBUTE_FILE)
    config_long_df = _standardize_config_attribute_long(config_raw_df)
    staff_orders = set()
    if "is_staff" in config_raw_df.columns:
        order_col = "Order Number" if "Order Number" in config_raw_df.columns else "order_number"
        s = config_raw_df["is_staff"]
        if pd.api.types.is_bool_dtype(s):
            m_staff = s.fillna(False)
        else:
            s2 = s.astype("string").fillna("").str.strip().str.upper()
            m_staff = s2.isin(["Y", "YES", "TRUE", "T", "1"])
        if order_col in config_raw_df.columns and m_staff.any():
            staff_orders = set(config_raw_df.loc[m_staff, order_col].dropna().astype("string"))
    else:
        print("⚠️ config_attribute.parquet 缺少 is_staff 列，将不做 isstaff=N 过滤。")
    listing_hourly_df = build_hourly_lock_counts(df, business_def, target_groups)
    listing_daily_df = build_daily_lock_counts(df, business_def, target_groups)
    listing_summary_df = build_listing_summary(df, business_def, target_groups)
    model_mix_df = build_model_mix(df, business_def, target_groups, staff_orders)
    config_version_df = build_configuration_version_summary(df, business_def, target_groups, config_long_df, staff_orders)
    gender_df = build_after30d_gender_detail(df, business_def, target_groups, staff_orders)
    age_df = build_after30d_age_detail(df, business_def, target_groups, staff_orders, age_col="buyer_age", age_label="buyer_age")
    lock_totals_df = build_after30d_lock_totals(df, business_def, target_groups, staff_orders)

    html = render_launch_report(
        listing_hourly_df,
        listing_daily_df,
        listing_summary_df,
        model_mix_df,
        config_version_df,
        gender_df,
        age_df,
        lock_totals_df,
        target_groups,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"✅ 已生成报告: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
