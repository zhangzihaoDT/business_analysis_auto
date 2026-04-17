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
from typing import Dict, List, Tuple

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


def build_presale_retention_summary(df: pd.DataFrame, business_def: dict, target_groups: List[str]) -> pd.DataFrame:
    required_cols = ["series_group_logic", "order_number", "intention_payment_time", "intention_refund_time", "lock_time"]
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

    for g in target_groups:
        tp = time_periods.get(g, {}) or {}
        start_str = tp.get("start")
        end_str = tp.get("end")
        finish_str = tp.get("finish")
        if not start_str or not end_str:
            continue

        start_day = pd.Timestamp(start_str)
        presale_end_day = pd.Timestamp(end_str)
        listing_day = presale_end_day + pd.Timedelta(days=1) if g == "CM0" else presale_end_day
        finish_day = pd.Timestamp(finish_str) if finish_str else listing_day

        presale_days = int((presale_end_day - start_day).days + 1)
        window_end_excl = presale_end_day + pd.Timedelta(days=1)

        finish_limit_excl = finish_day + pd.Timedelta(days=1)
        lock_30d_end_excl = min(listing_day + pd.Timedelta(days=31), finish_limit_excl)
        m_lock_total = (
            df["series_group_logic"].eq(g)
            & df["lock_time"].notna()
            & (df["lock_time"] >= listing_day)
            & (df["lock_time"] < lock_30d_end_excl)
        )
        total_lock_30d_cnt = int(df.loc[m_lock_total, "order_number"].dropna().nunique())

        m_presale = (
            df["series_group_logic"].eq(g)
            & df["intention_payment_time"].notna()
            & (df["intention_payment_time"] >= start_day)
            & (df["intention_payment_time"] < window_end_excl)
        )
        df_presale = df.loc[m_presale, ["order_number", "intention_refund_time", "lock_time"]].copy()
        if df_presale.empty:
            rows.append(
                {
                    "series_group_logic": g,
                    "预售期": f"{start_day.date().isoformat()} ~ {presale_end_day.date().isoformat()}",
                    "预售周期（日）": presale_days,
                    "留存小订数": 0,
                    "上市后30日锁单数": total_lock_30d_cnt,
                    "上市后30日留存小订转化数": 0,
                    "留存小订转化占比": "0.0%",
                    "转化率": "0.0%",
                }
            )
            continue

        df_presale["order_number"] = df_presale["order_number"].astype("string")
        df_presale = df_presale.dropna(subset=["order_number"]).drop_duplicates(subset=["order_number"])

        m_retained = df_presale["intention_refund_time"].isna() | (df_presale["intention_refund_time"] >= window_end_excl)
        retained = df_presale.loc[m_retained, ["order_number", "lock_time"]].copy()
        retained_cnt = int(retained["order_number"].nunique())

        m_lock_30d = (
            retained["lock_time"].notna()
            & (retained["lock_time"] >= listing_day)
            & (retained["lock_time"] < lock_30d_end_excl)
        )
        retained_lock_30d_cnt = int(retained.loc[m_lock_30d, "order_number"].nunique()) if retained_cnt > 0 else 0

        rate = retained_lock_30d_cnt / float(retained_cnt) if retained_cnt > 0 else 0.0
        retained_lock_share = (
            retained_lock_30d_cnt / float(total_lock_30d_cnt) if total_lock_30d_cnt > 0 else 0.0
        )
        rows.append(
            {
                "series_group_logic": g,
                "预售期": f"{start_day.date().isoformat()} ~ {presale_end_day.date().isoformat()}",
                "预售周期（日）": presale_days,
                "留存小订数": retained_cnt,
                "上市后30日锁单数": total_lock_30d_cnt,
                "上市后30日留存小订转化数": retained_lock_30d_cnt,
                "留存小订转化占比": f"{retained_lock_share:.1%}",
                "转化率": f"{rate:.1%}",
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        ordered_cols = [
            "series_group_logic",
            "预售期",
            "预售周期（日）",
            "留存小订数",
            "上市后30日锁单数",
            "上市后30日留存小订转化数",
            "留存小订转化占比",
            "转化率",
        ]
        out = out.loc[:, [c for c in ordered_cols if c in out.columns]].copy()
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(["__group_order"]).drop(columns="__group_order").reset_index(drop=True)
    return out


def build_presale_phase_conversion_and_ls8_projection(
    df: pd.DataFrame,
    business_def: dict,
    history_groups: List[str],
    target_group: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    required_cols = ["series_group_logic", "order_number", "intention_payment_time", "intention_refund_time", "lock_time"]
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

    conv_rows: List[Dict[str, object]] = []
    retained_dist: Dict[str, Dict[str, int]] = {}

    for g in history_groups:
        tp = time_periods.get(g, {}) or {}
        start_str = tp.get("start")
        end_str = tp.get("end")
        finish_str = tp.get("finish")
        if not start_str or not end_str:
            continue

        start_day = pd.Timestamp(start_str)
        presale_end_day = pd.Timestamp(end_str)
        listing_day = presale_end_day + pd.Timedelta(days=1) if g == "CM0" else presale_end_day
        finish_day = pd.Timestamp(finish_str) if finish_str else listing_day

        window_end_excl = presale_end_day + pd.Timedelta(days=1)

        df_g = df.loc[df["series_group_logic"].eq(g)].copy()
        m_presale = (
            df_g["intention_payment_time"].notna()
            & (df_g["intention_payment_time"] >= start_day)
            & (df_g["intention_payment_time"] < window_end_excl)
        )
        m_retained = df_g["intention_refund_time"].isna() | (df_g["intention_refund_time"] >= window_end_excl)
        retained_df = (
            df_g.loc[m_presale & m_retained, ["order_number", "intention_payment_time"]]
            .dropna(subset=["order_number"])
            .drop_duplicates(subset=["order_number"])
            .copy()
        )
        if retained_df.empty:
            continue

        retained_df["intention_days_from_start"] = (
            retained_df["intention_payment_time"].dt.normalize() - start_day.normalize()
        ).dt.days
        retained_df["intention_days_to_end"] = (
            presale_end_day.normalize() - retained_df["intention_payment_time"].dt.normalize()
        ).dt.days

        retained_count = int(retained_df["order_number"].nunique())

        base_day1 = int((retained_df["intention_days_from_start"] == 0).sum())
        base_top3 = int((retained_df["intention_days_from_start"] < 3).sum())
        base_last_day3 = int((retained_df["intention_days_to_end"] == 2).sum())
        base_last_day2 = int((retained_df["intention_days_to_end"] == 1).sum())
        base_last_day1 = int((retained_df["intention_days_to_end"] == 0).sum())
        base_middle = int(
            retained_count - base_top3 - base_last_day3 - base_last_day2 - base_last_day1
        )

        finish_limit_excl = finish_day + pd.Timedelta(days=1)
        lock_30d_end_excl = min(listing_day + pd.Timedelta(days=31), finish_limit_excl)
        m_lock = (
            df_g["lock_time"].notna()
            & (df_g["lock_time"] >= listing_day)
            & (df_g["lock_time"] < lock_30d_end_excl)
        )
        locked_orders = (
            df_g.loc[m_lock, ["order_number", "lock_time"]]
            .dropna(subset=["order_number"])
            .drop_duplicates(subset=["order_number"])
            .copy()
        )
        if locked_orders.empty:
            conv_rows.append(
                {
                    "车型": g,
                    "Day1留存小订": base_day1,
                    "Day1转化率": "0.0%",
                    "前3日留存小订": base_top3,
                    "前3日转化率": "0.0%",
                    "中间期留存小订": base_middle,
                    "中间期转化率": "0.0%",
                    "倒数Day2留存小订": base_last_day3,
                    "倒数Day2转化率": "0.0%",
                    "倒数Day1留存小订": base_last_day2,
                    "倒数Day1转化率": "0.0%",
                    "倒数Day0(上市当天)留存小订": base_last_day1,
                    "倒数Day0转化率": "0.0%",
                }
            )
            retained_dist[g] = {
                "total": retained_count,
                "top3": base_top3,
                "middle": base_middle,
                "last_day3": base_last_day3,
                "last_day2": base_last_day2,
                "last_day1": base_last_day1,
            }
            continue

        locked_retained_df = locked_orders.loc[
            locked_orders["order_number"].isin(retained_df["order_number"])
        ].copy()
        if locked_retained_df.empty:
            conv_rows.append(
                {
                    "车型": g,
                    "Day1留存小订": base_day1,
                    "Day1转化率": "0.0%",
                    "前3日留存小订": base_top3,
                    "前3日转化率": "0.0%",
                    "中间期留存小订": base_middle,
                    "中间期转化率": "0.0%",
                    "倒数Day2留存小订": base_last_day3,
                    "倒数Day2转化率": "0.0%",
                    "倒数Day1留存小订": base_last_day2,
                    "倒数Day1转化率": "0.0%",
                    "倒数Day0(上市当天)留存小订": base_last_day1,
                    "倒数Day0转化率": "0.0%",
                }
            )
            retained_dist[g] = {
                "total": retained_count,
                "top3": base_top3,
                "middle": base_middle,
                "last_day3": base_last_day3,
                "last_day2": base_last_day2,
                "last_day1": base_last_day1,
            }
            continue

        locked_retained_df = locked_retained_df.merge(
            retained_df[
                ["order_number", "intention_days_from_start", "intention_days_to_end"]
            ],
            on="order_number",
            how="left",
        )

        locked_total = int(locked_retained_df["order_number"].nunique())
        lock_day1 = int((locked_retained_df["intention_days_from_start"] == 0).sum())
        lock_top3 = int((locked_retained_df["intention_days_from_start"] < 3).sum())
        lock_last_day3 = int((locked_retained_df["intention_days_to_end"] == 2).sum())
        lock_last_day2 = int((locked_retained_df["intention_days_to_end"] == 1).sum())
        lock_last_day1 = int((locked_retained_df["intention_days_to_end"] == 0).sum())
        lock_middle = int(
            locked_total - lock_top3 - lock_last_day3 - lock_last_day2 - lock_last_day1
        )

        def conv_ratio(lock_c: int, base_c: int) -> str:
            return f"{(lock_c / float(base_c)):.1%}" if base_c > 0 else "0.0%"

        conv_rows.append(
            {
                "车型": g,
                "Day1留存小订": base_day1,
                "Day1转化率": conv_ratio(lock_day1, base_day1),
                "前3日留存小订": base_top3,
                "前3日转化率": conv_ratio(lock_top3, base_top3),
                "中间期留存小订": base_middle,
                "中间期转化率": conv_ratio(lock_middle, base_middle),
                "倒数Day2留存小订": base_last_day3,
                "倒数Day2转化率": conv_ratio(lock_last_day3, base_last_day3),
                "倒数Day1留存小订": base_last_day2,
                "倒数Day1转化率": conv_ratio(lock_last_day2, base_last_day2),
                "倒数Day0(上市当天)留存小订": base_last_day1,
                "倒数Day0转化率": conv_ratio(lock_last_day1, base_last_day1),
            }
        )
        retained_dist[g] = {
            "total": retained_count,
            "top3": base_top3,
            "middle": base_middle,
            "last_day3": base_last_day3,
            "last_day2": base_last_day2,
            "last_day1": base_last_day1,
        }

    conv_df = pd.DataFrame(conv_rows)
    if not conv_df.empty:
        group_order = {g: i for i, g in enumerate(history_groups)}
        conv_df["__group_order"] = conv_df["车型"].map(group_order).fillna(len(group_order))
        conv_df = conv_df.sort_values(["__group_order"]).drop(columns="__group_order").reset_index(drop=True)

    if target_group not in time_periods:
        return conv_df, pd.DataFrame(), ""

    tp_t = time_periods.get(target_group, {}) or {}
    start_str_t = tp_t.get("start")
    end_str_t = tp_t.get("end")
    finish_str_t = tp_t.get("finish")
    if not start_str_t or not end_str_t:
        return conv_df, pd.DataFrame(), ""

    start_day_t = pd.Timestamp(start_str_t)
    presale_end_day_t = pd.Timestamp(end_str_t)
    listing_day_t = (
        presale_end_day_t + pd.Timedelta(days=1) if target_group == "CM0" else presale_end_day_t
    )
    finish_day_t = pd.Timestamp(finish_str_t) if finish_str_t else listing_day_t
    window_end_excl_t = presale_end_day_t + pd.Timedelta(days=1)

    df_t = df.loc[df["series_group_logic"].eq(target_group)].copy()
    m_presale_t = (
        df_t["intention_payment_time"].notna()
        & (df_t["intention_payment_time"] >= start_day_t)
        & (df_t["intention_payment_time"] < window_end_excl_t)
    )
    m_retained_t = df_t["intention_refund_time"].isna() | (
        df_t["intention_refund_time"] >= window_end_excl_t
    )
    retained_t = (
        df_t.loc[m_presale_t & m_retained_t, ["order_number", "intention_payment_time"]]
        .dropna(subset=["order_number"])
        .drop_duplicates(subset=["order_number"])
        .copy()
    )
    if retained_t.empty:
        return conv_df, pd.DataFrame(), ""

    retained_t["intention_days_from_start"] = (
        retained_t["intention_payment_time"].dt.normalize() - start_day_t.normalize()
    ).dt.days
    retained_t["intention_days_to_end"] = (
        presale_end_day_t.normalize() - retained_t["intention_payment_time"].dt.normalize()
    ).dt.days

    retained_total_t = int(retained_t["order_number"].nunique())
    ls8_top3_cnt = int((retained_t["intention_days_from_start"] < 3).sum())
    actual_last_day3 = int((retained_t["intention_days_to_end"] == 2).sum())
    actual_last_day2 = int((retained_t["intention_days_to_end"] == 1).sum())
    actual_last_day1 = int((retained_t["intention_days_to_end"] == 0).sum())

    ls8_max_day = retained_t["intention_days_from_start"].max()
    total_presale_days = int((presale_end_day_t - start_day_t).days + 1)
    is_presale_complete = (
        pd.notna(ls8_max_day) and int(ls8_max_day) >= int(total_presale_days - 1)
    )
    ls8_effective_max_day = (
        int(ls8_max_day) if is_presale_complete else int(max(0, int(ls8_max_day) - 1))
    ) if pd.notna(ls8_max_day) else 0
    middle_period_days = int(max(0, total_presale_days - 3 - 3))
    max_middle_day_idx = int(total_presale_days - 1 - 3)
    effective_middle_max = int(min(ls8_effective_max_day, max_middle_day_idx))
    passed_middle_days = int(effective_middle_max - 3 + 1) if effective_middle_max >= 3 else 0

    middle_actual_df = retained_t.loc[
        (retained_t["intention_days_from_start"] >= 3)
        & (retained_t["intention_days_from_start"] <= ls8_effective_max_day)
        & (retained_t["intention_days_to_end"] >= 3)
    ].copy()
    ls8_middle_actual_cnt = int(len(middle_actual_df))
    if is_presale_complete:
        remaining_middle_days = 0
        ls8_projected_middle = float(ls8_middle_actual_cnt)
    else:
        ls8_middle_avg = ls8_middle_actual_cnt / float(passed_middle_days) if passed_middle_days > 0 else 0.0
        remaining_middle_days = int(max(0, middle_period_days - passed_middle_days))
        ls8_projected_middle = float(ls8_middle_actual_cnt + ls8_middle_avg * remaining_middle_days)

    last_day3_idx = int(total_presale_days - 3)
    last_day2_idx = int(total_presale_days - 2)
    last_day1_idx = int(total_presale_days - 1)
    is_last_day3_complete = ls8_effective_max_day >= last_day3_idx
    is_last_day2_complete = ls8_effective_max_day >= last_day2_idx
    is_last_day1_complete = ls8_effective_max_day >= last_day1_idx

    ls8_rows: List[Dict[str, object]] = []

    def parse_pct(s: object) -> float:
        if s is None:
            return 0.0
        if isinstance(s, (int, float)):
            return float(s)
        text = str(s).strip()
        if not text:
            return 0.0
        if text.endswith("%"):
            text = text[:-1]
        try:
            return float(text) / 100.0
        except Exception:
            return 0.0

    conv_map = {str(r["车型"]): r for r in conv_rows if "车型" in r}
    for hist, dist in retained_dist.items():
        total = int(dist.get("total", 0))
        if total <= 0:
            continue
        hist_top3_ratio = float(dist.get("top3", 0)) / float(total)
        hist_middle_ratio = float(dist.get("middle", 0)) / float(total)
        hist_last_day3_ratio = float(dist.get("last_day3", 0)) / float(total)
        hist_last_day2_ratio = float(dist.get("last_day2", 0)) / float(total)
        hist_last_day1_ratio = float(dist.get("last_day1", 0)) / float(total)

        base_count = float(ls8_top3_cnt) + float(ls8_projected_middle)
        base_ratio = float(hist_top3_ratio + hist_middle_ratio)
        if is_last_day3_complete:
            base_count += float(actual_last_day3)
            base_ratio += float(hist_last_day3_ratio)
        if is_last_day2_complete:
            base_count += float(actual_last_day2)
            base_ratio += float(hist_last_day2_ratio)
        if is_last_day1_complete:
            base_count += float(actual_last_day1)
            base_ratio += float(hist_last_day1_ratio)

        proj_total_base = base_count / base_ratio if base_ratio > 0 else 0.0

        proj_last_day3 = float(actual_last_day3) if is_last_day3_complete else max(float(actual_last_day3), proj_total_base * hist_last_day3_ratio)
        proj_last_day2 = float(actual_last_day2) if is_last_day2_complete else max(float(actual_last_day2), proj_total_base * hist_last_day2_ratio)
        proj_last_day1 = float(actual_last_day1) if is_last_day1_complete else max(float(actual_last_day1), proj_total_base * hist_last_day1_ratio)
        proj_last3_total = float(proj_last_day3 + proj_last_day2 + proj_last_day1)
        proj_retained_total = float(ls8_top3_cnt) + float(ls8_projected_middle) + float(proj_last3_total)

        conv = conv_map.get(hist, {})
        top3_rate = parse_pct(conv.get("前3日转化率"))
        middle_rate = parse_pct(conv.get("中间期转化率"))
        last_day3_rate = parse_pct(conv.get("倒数Day2转化率"))
        last_day2_rate = parse_pct(conv.get("倒数Day1转化率"))
        last_day1_rate = parse_pct(conv.get("倒数Day0转化率"))

        lock_top3 = float(ls8_top3_cnt) * top3_rate
        lock_middle = float(ls8_projected_middle) * middle_rate
        lock_last_day3 = float(proj_last_day3) * last_day3_rate
        lock_last_day2 = float(proj_last_day2) * last_day2_rate
        lock_last_day1 = float(proj_last_day1) * last_day1_rate
        lock_last3 = float(lock_last_day3 + lock_last_day2 + lock_last_day1)
        lock_total = float(lock_top3 + lock_middle + lock_last3)

        rate_last3_overall = lock_last3 / float(proj_last3_total) if proj_last3_total > 0 else 0.0
        rate_total = lock_total / float(proj_retained_total) if proj_retained_total > 0 else 0.0

        ls8_rows.append(
            {
                "参考历史车型": hist,
                "推演留存小订": int(round(proj_retained_total)),
                "推演30日锁单": int(round(lock_total)),
                "推演转化率": f"{rate_total:.1%}",
                "历史前3日转化率": f"{top3_rate:.1%}",
                "前3日推演锁单": int(round(lock_top3)),
                "历史中间期转化率": f"{middle_rate:.1%}",
                "中间期推演锁单": int(round(lock_middle)),
                "综合末尾3日转化率": f"{rate_last3_overall:.1%}",
                "末尾3日推演锁单": int(round(lock_last3)),
            }
        )

    ls8_df = pd.DataFrame(ls8_rows)
    if not ls8_df.empty:
        group_order = {g: i for i, g in enumerate(history_groups)}
        ls8_df["__group_order"] = ls8_df["参考历史车型"].map(group_order).fillna(len(group_order))
        ls8_df = ls8_df.sort_values(["__group_order"]).drop(columns="__group_order").reset_index(drop=True)
    last3_note = ""
    if is_last_day3_complete and is_last_day2_complete and is_last_day1_complete:
        last3_note = f", 末尾3日基数: {actual_last_day3 + actual_last_day2 + actual_last_day1}"
    note = f"前3日已知基数: {ls8_top3_cnt}, 中间期推演基数: {int(round(ls8_projected_middle))}{last3_note}"
    return conv_df, ls8_df, note


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


def build_retained_intention_conversion_curve(
    df: pd.DataFrame, business_def: dict, target_groups: List[str]
) -> pd.DataFrame:
    required_cols = ["series_group_logic", "order_number", "intention_payment_time", "intention_refund_time", "lock_time"]
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

    for g in target_groups:
        tp = time_periods.get(g, {}) or {}
        start_str = tp.get("start")
        end_str = tp.get("end")
        finish_str = tp.get("finish")
        if not start_str or not end_str:
            continue

        start_day = pd.Timestamp(start_str)
        presale_end_day = pd.Timestamp(end_str)
        listing_day = presale_end_day + pd.Timedelta(days=1) if g == "CM0" else presale_end_day
        finish_day = pd.Timestamp(finish_str) if finish_str else listing_day

        presale_end_excl = presale_end_day + pd.Timedelta(days=1)
        finish_limit_excl = finish_day + pd.Timedelta(days=1)
        after_30d_end_excl = min(listing_day + pd.Timedelta(days=31), finish_limit_excl)

        m_presale = (
            df["series_group_logic"].eq(g)
            & df["intention_payment_time"].notna()
            & (df["intention_payment_time"] >= start_day)
            & (df["intention_payment_time"] < presale_end_excl)
        )
        df_presale = df.loc[m_presale, ["order_number", "intention_refund_time", "lock_time"]].copy()
        df_presale["order_number"] = df_presale["order_number"].astype("string")
        df_presale = df_presale.dropna(subset=["order_number"]).drop_duplicates(subset=["order_number"])

        m_retained = df_presale["intention_refund_time"].isna() | (df_presale["intention_refund_time"] >= presale_end_excl)
        retained = df_presale.loc[m_retained, ["order_number", "lock_time"]].copy()
        retained_cnt = int(retained["order_number"].nunique())

        date_index = pd.date_range(
            start=listing_day.floor("D"),
            end=(after_30d_end_excl - pd.Timedelta(days=1)).floor("D"),
            freq="D",
        )

        if retained_cnt <= 0 or retained.empty:
            daily_full = pd.Series([0] * len(date_index), index=date_index, dtype="int64")
        else:
            m_lock_window = (
                retained["lock_time"].notna()
                & (retained["lock_time"] >= listing_day)
                & (retained["lock_time"] < after_30d_end_excl)
            )
            lock_slice = retained.loc[m_lock_window, ["order_number", "lock_time"]].copy()
            if lock_slice.empty:
                daily_full = pd.Series([0] * len(date_index), index=date_index, dtype="int64")
            else:
                lock_slice["date"] = lock_slice["lock_time"].dt.floor("D")
                daily = lock_slice.groupby("date")["order_number"].nunique()
                daily_full = daily.reindex(date_index, fill_value=0).astype("int64")

        cum = daily_full.cumsum()
        listing_day_floor = listing_day.floor("D")
        for d, cum_cnt in cum.items():
            day_n = int((d - listing_day_floor).days)
            rate = float(cum_cnt) / float(retained_cnt) if retained_cnt > 0 else 0.0
            rows.append(
                {
                    "series_group_logic": g,
                    "end_date": listing_day.date().isoformat(),
                    "day": day_n,
                    "date": d.date().isoformat(),
                    "retained_orders": retained_cnt,
                    "converted_orders_cum": int(cum_cnt),
                    "conversion_rate": rate,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(["__group_order", "day"]).drop(columns="__group_order").reset_index(drop=True)
    return out


def build_retained_intention_refund_curve(df: pd.DataFrame, business_def: dict, target_groups: List[str]) -> pd.DataFrame:
    required_cols = ["series_group_logic", "order_number", "intention_payment_time", "intention_refund_time"]
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
        finish_str = tp.get("finish")
        if not start_str or not end_str:
            continue

        start_day = pd.Timestamp(start_str)
        presale_end_day = pd.Timestamp(end_str)
        listing_day = presale_end_day + pd.Timedelta(days=1) if g == "CM0" else presale_end_day
        finish_day = pd.Timestamp(finish_str) if finish_str else listing_day

        presale_end_excl = presale_end_day + pd.Timedelta(days=1)
        finish_limit_excl = finish_day + pd.Timedelta(days=1)
        after_30d_end_excl = min(listing_day + pd.Timedelta(days=31), finish_limit_excl)

        m_presale = (
            df["series_group_logic"].eq(g)
            & df["intention_payment_time"].notna()
            & (df["intention_payment_time"] >= start_day)
            & (df["intention_payment_time"] < presale_end_excl)
        )
        df_presale = df.loc[m_presale, ["order_number", "intention_refund_time"]].copy()
        df_presale["order_number"] = df_presale["order_number"].astype("string")
        df_presale = df_presale.dropna(subset=["order_number"]).drop_duplicates(subset=["order_number"])

        m_retained = df_presale["intention_refund_time"].isna() | (df_presale["intention_refund_time"] >= presale_end_excl)
        retained = df_presale.loc[m_retained, ["order_number", "intention_refund_time"]].copy()
        retained_cnt = int(retained["order_number"].nunique())

        date_index = pd.date_range(
            start=listing_day.floor("D"),
            end=(after_30d_end_excl - pd.Timedelta(days=1)).floor("D"),
            freq="D",
        )

        if retained_cnt <= 0 or retained.empty:
            daily_full = pd.Series([0] * len(date_index), index=date_index, dtype="int64")
        else:
            m_refund_window = (
                retained["intention_refund_time"].notna()
                & (retained["intention_refund_time"] >= listing_day)
                & (retained["intention_refund_time"] < after_30d_end_excl)
            )
            refund_slice = retained.loc[m_refund_window, ["order_number", "intention_refund_time"]].copy()
            if refund_slice.empty:
                daily_full = pd.Series([0] * len(date_index), index=date_index, dtype="int64")
            else:
                refund_slice["date"] = refund_slice["intention_refund_time"].dt.floor("D")
                daily = refund_slice.groupby("date")["order_number"].nunique()
                daily_full = daily.reindex(date_index, fill_value=0).astype("int64")

        cum = daily_full.cumsum()
        listing_day_floor = listing_day.floor("D")
        for d, cum_cnt in cum.items():
            day_n = int((d - listing_day_floor).days)
            rate = float(cum_cnt) / float(retained_cnt) if retained_cnt > 0 else 0.0
            remaining = int(retained_cnt - int(cum_cnt))
            rows.append(
                {
                    "series_group_logic": g,
                    "end_date": listing_day.date().isoformat(),
                    "day": day_n,
                    "date": d.date().isoformat(),
                    "retained_orders": retained_cnt,
                    "refunded_orders_cum": int(cum_cnt),
                    "remaining_orders": remaining,
                    "refund_rate": rate,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_order = {g: i for i, g in enumerate(target_groups)}
        out["__group_order"] = out["series_group_logic"].map(group_order).fillna(len(group_order))
        out = out.sort_values(["__group_order", "day"]).drop(columns="__group_order").reset_index(drop=True)
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


def _render_retained_conversion_rate_line_figure(
    curve_df: pd.DataFrame,
    target_groups: List[str],
    fig_title: str,
) -> go.Figure:
    fig = go.Figure()
    for g in target_groups:
        dfg = curve_df.loc[curve_df["series_group_logic"].eq(g)].copy()
        if dfg.empty:
            continue
        dfg = dfg.sort_values("day")
        customdata = np.stack(
            [
                dfg["date"].astype("string").fillna("").to_numpy(),
                dfg["retained_orders"].astype("int64").to_numpy(),
                dfg["converted_orders_cum"].astype("int64").to_numpy(),
            ],
            axis=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dfg["day"].astype("int64"),
                y=dfg["conversion_rate"].astype(float),
                mode="lines+markers",
                name=g,
                customdata=customdata,
                hovertemplate=(
                    "series=%{fullData.name}<br>"
                    "上市后Day=%{x}<br>"
                    "date=%{customdata[0]}<br>"
                    "累计转化=%{customdata[2]} / %{customdata[1]}<br>"
                    "转化率=%{y:.1%}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=fig_title,
        xaxis=dict(title="上市后天数", tickmode="linear", dtick=2),
        yaxis=dict(title="留存小订转化率", tickformat=".1%"),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _render_retained_refund_rate_line_figure(
    curve_df: pd.DataFrame,
    target_groups: List[str],
    fig_title: str,
) -> go.Figure:
    fig = go.Figure()
    for g in target_groups:
        dfg = curve_df.loc[curve_df["series_group_logic"].eq(g)].copy()
        if dfg.empty:
            continue
        dfg = dfg.sort_values("day")
        customdata = np.stack(
            [
                dfg["date"].astype("string").fillna("").to_numpy(),
                dfg["retained_orders"].astype("int64").to_numpy(),
                dfg["refunded_orders_cum"].astype("int64").to_numpy(),
                dfg["remaining_orders"].astype("int64").to_numpy(),
            ],
            axis=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dfg["day"].astype("int64"),
                y=dfg["refund_rate"].astype(float),
                mode="lines+markers",
                name=g,
                customdata=customdata,
                hovertemplate=(
                    "series=%{fullData.name}<br>"
                    "上市后Day=%{x}<br>"
                    "date=%{customdata[0]}<br>"
                    "累计退订=%{customdata[2]} / %{customdata[1]}<br>"
                    "未退订=%{customdata[3]}<br>"
                    "退订率=%{y:.1%}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=fig_title,
        xaxis=dict(title="上市后天数", tickmode="linear", dtick=2),
        yaxis=dict(title="留存小订退订率", tickformat=".1%"),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
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
    presale_summary_df: pd.DataFrame,
    phase_conv_df: pd.DataFrame,
    ls8_projection_df: pd.DataFrame,
    ls8_projection_note: str,
    listing_hourly_df: pd.DataFrame,
    listing_daily_df: pd.DataFrame,
    retained_conv_curve_df: pd.DataFrame,
    retained_refund_curve_df: pd.DataFrame,
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
    ]

    html_content.append("<h2>1. 预售期小订回顾</h2>")
    html_content.append("<div class='summary-box'>")
    html_content.append(
        "<p>口径：预售周期（日）= end_day - start_day + 1。对每个 series_group_logic，统计预售开始日起至预售结束日 endday 的小订订单（intention_payment_time 落在 start~endday 窗口内，order_number 去重）；"
        "以“截至 endday 未退订”作为留存小订：intention_refund_time 为空，或 intention_refund_time &gt; endday（等价于 intention_refund_time &gt;= endday+1 的 00:00）。"
        "上市后30日锁单窗口为 listing_day ~ min(listing_day+30, finish)（CM0 特殊：listing_day=end_day+1）。转化率 = 上市后30日留存小订转化数 / 留存小订数；留存小订转化占比 = 上市后30日留存小订转化数 / 上市后30日锁单数。</p>"
    )
    html_content.append("</div>")
    html_content.append("<h3>1.1 汇总表</h3>")
    if presale_summary_df is None or presale_summary_df.empty:
        html_content.append("<p>⚠️ 汇总表为空（可能缺少 time_periods 的 start/end 或分组无数据）。</p>")
    else:
        html_content.append(presale_summary_df.to_html(index=False, classes="table", escape=False))

    html_content.append("<h3>1.2 LS8 上市后30日锁单数推演</h3>")
    html_content.append("<div class='summary-box'>")
    html_content.append(
        "<p>步骤1：基于历史车型 CM0/DM0/CM1/DM1/CM2/LS9 的预售留存小订，按 Day1、前3日、中间期、倒数Day2/Day1/Day0 拆分各阶段留存小订基数，并计算对应的上市后30日锁单转化率。</p>"
    )
    html_content.append(
        "<p>步骤2：对 LS8 的预售留存小订做相同阶段拆分，分别代入历史车型的阶段转化率，计算“前3日 + 中间期 + 末尾3日”的推演锁单数，得到 LS8 上市后30日推演锁单量及推演转化率。</p>"
    )
    html_content.append("</div>")
    if phase_conv_df is None or phase_conv_df.empty:
        html_content.append("<p>⚠️ 历史车型预售各阶段留存小订及转化率表为空（可能缺少预售相关字段或历史车型无数据）。</p>")
    else:
        html_content.append("<h4>1.2.1 历史车型预售各阶段留存小订及转化率</h4>")
        html_content.append(phase_conv_df.to_html(index=False, classes="table", escape=False))
    if ls8_projection_df is None or ls8_projection_df.empty:
        html_content.append("<p>⚠️ LS8 推演结果为空（可能缺少 LS8 预售数据或预售留存小订）。</p>")
    else:
        html_content.append("<h4>1.2.2 LS8 上市后30日锁单数推演</h4>")
        html_content.append("<pre>--- LS8 上市后30日锁单数推演 (前3日 + 中间期 + 末尾3日分别代入转化率) ---</pre>")
        if ls8_projection_note:
            html_content.append(f"<pre>{ls8_projection_note}</pre>")
        html_content.append(ls8_projection_df.to_html(index=False, classes="table", escape=False))

    html_content.append("<h2>2. 上市期锁单数</h2>")
    html_content.append("<div class='summary-box'>")
    html_content.append(
        "<p>口径：先用业务定义 series_group_logic 根据 product_name 对订单归类；然后对每个 series_group_logic 使用业务定义 time_periods 中的 end 日期（上市日期），统计上市日期每小时锁单数，并展示上市日起（至上市后30日或 finish）每天锁单数（lock_time 非空的 order_number 去重计数）。</p>"
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

    html_content.append("<h3>2.4 留存小订转化率（series_group_logic）</h3>")
    html_content.append("<div class='summary-box'>")
    html_content.append(
        "<p>口径：对每个 series_group_logic，以预售期 start~end 内的小订订单（intention_payment_time 落在窗口内，order_number 去重）为基数；"
        "以截至 endday 未退订的小订作为“留存小订”（intention_refund_time 为空，或 intention_refund_time &gt; endday）。"
        "这里 endday 指预售结束日当天结束时刻（等价于 intention_refund_time &gt;= end+1 的 00:00）。"
        "统计这些留存小订在上市后窗口 endday~min(endday+30, finish) 的锁单（lock_time 非空，order_number 去重），并按日期做累计，得到累计转化率曲线。"
        "其中 CM0 上市日特殊处理：endday=end+1。</p>"
    )
    html_content.append("</div>")
    if retained_conv_curve_df is None or retained_conv_curve_df.empty:
        html_content.append("<p>⚠️ 留存小订转化率明细为空（可能缺少 time_periods.start/end 或无留存小订）。</p>")
    else:
        fig4 = _render_retained_conversion_rate_line_figure(
            retained_conv_curve_df,
            target_groups,
            fig_title="留存小订转化率：上市后30日累计（series_group_logic）",
        )
        html_content.append(pio.to_html(fig4, full_html=False, include_plotlyjs=False))

    html_content.append("<h3>2.5 留存小订退订率（series_group_logic）</h3>")
    html_content.append("<div class='summary-box'>")
    html_content.append(
        "<p>口径：沿用 2.4 的“留存小订”定义作为基数（截至 endday 未退订的小订，"
        "即 intention_refund_time 为空，或 intention_refund_time &gt; endday，"
        "其中 endday 指预售结束日当天结束时刻）；"
        "统计这些留存小订在上市后窗口 endday~min(endday+30, finish) 内发生退订（intention_refund_time 落在窗口内，order_number 去重）的累计数量，"
        "累计退订率 = 累计退订数 / 留存小订数；未退订数 = 留存小订数 - 累计退订数。"
        "其中 CM0 上市日特殊处理：endday=end+1。</p>"
    )
    html_content.append("</div>")
    if retained_refund_curve_df is None or retained_refund_curve_df.empty:
        html_content.append("<p>⚠️ 留存小订退订率明细为空（可能缺少 time_periods.start/end 或无留存小订）。</p>")
    else:
        fig5 = _render_retained_refund_rate_line_figure(
            retained_refund_curve_df,
            target_groups,
            fig_title="留存小订退订率：上市后30日累计（series_group_logic）",
        )
        html_content.append(pio.to_html(fig5, full_html=False, include_plotlyjs=False))

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
    presale_groups = ["CM0", "DM0", "CM1", "DM1", "CM2", "LS9", "LS8"]
    presale_summary_df = build_presale_retention_summary(df, business_def, presale_groups)
    phase_conv_df, ls8_projection_df, ls8_projection_note = build_presale_phase_conversion_and_ls8_projection(
        df,
        business_def,
        history_groups=["CM0", "DM0", "CM1", "DM1", "CM2", "LS9"],
        target_group="LS8",
    )
    listing_hourly_df = build_hourly_lock_counts(df, business_def, target_groups)
    listing_daily_df = build_daily_lock_counts(df, business_def, target_groups)
    retained_conv_curve_df = build_retained_intention_conversion_curve(df, business_def, target_groups)
    retained_refund_curve_df = build_retained_intention_refund_curve(df, business_def, target_groups)
    listing_summary_df = build_listing_summary(df, business_def, target_groups)
    model_mix_df = build_model_mix(df, business_def, target_groups, staff_orders)
    config_version_df = build_configuration_version_summary(df, business_def, target_groups, config_long_df, staff_orders)
    gender_df = build_after30d_gender_detail(df, business_def, target_groups, staff_orders)
    age_df = build_after30d_age_detail(df, business_def, target_groups, staff_orders, age_col="buyer_age", age_label="buyer_age")
    lock_totals_df = build_after30d_lock_totals(df, business_def, target_groups, staff_orders)

    html = render_launch_report(
        presale_summary_df,
        phase_conv_df,
        ls8_projection_df,
        ls8_projection_note,
        listing_hourly_df,
        listing_daily_df,
        retained_conv_curve_df,
        retained_refund_curve_df,
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
