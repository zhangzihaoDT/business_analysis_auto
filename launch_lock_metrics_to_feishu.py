#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reads order_data.parquet, computes launch lock-order monitoring metrics for a specific model, and sends an interactive card via Feishu webhook.
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()


PARQUET_FILE = Path("/Users/zihao_/Documents/coding/dataset/formatted/order_data.parquet")
BUSINESS_DEF_FILE = Path("/Users/zihao_/Documents/github/26W06_Tool_calls/schema/business_definition.json")

WEBHOOK_URL = os.getenv("FS_WEBHOOK_URL")


def load_business_definition(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    return json.loads(file_path.read_text(encoding="utf-8"))


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
    except Exception:
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


def resolve_launch_date(time_periods: dict, key: str) -> pd.Timestamp | None:
    tp = (time_periods or {}).get(key, {}) or {}
    date_str = tp.get("end") or tp.get("finish") or tp.get("start")
    if not date_str:
        return None
    return pd.Timestamp(date_str)


def compute_launch_lock_metrics(df: pd.DataFrame, business_def: dict, today: pd.Timestamp, target_model: str) -> dict:
    required_cols = [
        "lock_time",
        "order_number",
        "approve_refund_time",
        "owner_identity_no",
        "series_group_logic",
        "order_type",
        "store_city",
        "store_name",
        "intention_payment_time",
        "intention_refund_time",
        "deposit_payment_time",
        "deposit_refund_time",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"数据缺少列: {', '.join(missing)}")

    if not pd.api.types.is_datetime64_any_dtype(df["lock_time"]):
        df["lock_time"] = pd.to_datetime(df["lock_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["approve_refund_time"]):
        df["approve_refund_time"] = pd.to_datetime(df["approve_refund_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["intention_payment_time"]):
        df["intention_payment_time"] = pd.to_datetime(df["intention_payment_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["intention_refund_time"]):
        df["intention_refund_time"] = pd.to_datetime(df["intention_refund_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["deposit_payment_time"]):
        df["deposit_payment_time"] = pd.to_datetime(df["deposit_payment_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["deposit_refund_time"]):
        df["deposit_refund_time"] = pd.to_datetime(df["deposit_refund_time"], errors="coerce")

    run_date = today.normalize()
    max_lock_time = df["lock_time"].max()
    max_deposit_payment_time = df["deposit_payment_time"].max()
    max_intention_payment_time = df["intention_payment_time"].max()
    as_of_date = run_date
    for v in [max_lock_time, max_deposit_payment_time, max_intention_payment_time]:
        if pd.notna(v):
            as_of_date = min(as_of_date, pd.Timestamp(v).normalize())

    time_periods = business_def.get("time_periods", {}) or {}
    target_launch = resolve_launch_date(time_periods, target_model)
    cm2_launch = resolve_launch_date(time_periods, "CM2")
    ls9_launch = resolve_launch_date(time_periods, "LS9")

    n = 1
    n_raw = 1
    if target_launch is not None:
        n_raw = int((today.normalize() - target_launch.normalize()).days + 1)
        n = max(1, n_raw)

    base = df.loc[
        df["lock_time"].notna(),
        ["order_number", "lock_time", "approve_refund_time", "owner_identity_no", "series_group_logic"],
    ].copy()

    target_cum = 0
    target_retention = 0
    target_retention_users = 0
    target_peak_hour = None
    target_peak_count = 0
    target_next_hour_count = 0
    target_launch_day_total = 0
    cm2_n_day_retention = 0
    ls9_n_day_retention = 0
    path_intention = 0
    path_retained_deposit_conv = 0
    path_today_deposit_conv = 0
    path_today_deposit_unlocked = 0
    today_lock_count = 0
    today_user_car_lock_count = 0

    if target_launch is not None:
        today_excl = today + pd.Timedelta(days=1)

        target_slice = base.loc[
            base["series_group_logic"].eq(target_model)
            & (base["lock_time"] >= target_launch)
            & (base["lock_time"] < today_excl),
            "order_number",
        ]
        target_cum = int(target_slice.nunique())

        target_retention_slice = base.loc[
            base["series_group_logic"].eq(target_model)
            & (base["lock_time"] >= target_launch)
            & (base["lock_time"] < today_excl)
            & (base["approve_refund_time"].isna()),
            ["order_number", "owner_identity_no"],
        ]
        target_retention = int(target_retention_slice["order_number"].nunique())

        if not target_retention_slice.empty:
            order_counts_per_user = target_retention_slice.groupby("owner_identity_no")["order_number"].nunique()
            target_retention_users = int((order_counts_per_user == 1).sum())

        launch_excl = target_launch + pd.Timedelta(days=1)
        day_slice = base.loc[
            base["series_group_logic"].eq(target_model) & (base["lock_time"] >= target_launch) & (base["lock_time"] < launch_excl),
            ["order_number", "lock_time"],
        ].copy()

        if not day_slice.empty:
            day_slice["hour"] = day_slice["lock_time"].dt.hour.astype("int64")
            hourly = day_slice.groupby("hour")["order_number"].nunique().reindex(range(24), fill_value=0)
            target_peak_hour = int(hourly.idxmax())
            target_peak_count = int(hourly.iloc[target_peak_hour])
            target_next_hour_count = int(hourly.iloc[target_peak_hour + 1]) if target_peak_hour < 23 else 0
            target_launch_day_total = int(hourly.sum())

        n_days = (today - target_launch).days + 1

        if cm2_launch is not None:
            cm2_window_end = cm2_launch + pd.Timedelta(days=n_days)
            cm2_slice = base.loc[
                base["series_group_logic"].eq("CM2")
                & (base["lock_time"] >= cm2_launch)
                & (base["lock_time"] < cm2_window_end)
                & ((base["approve_refund_time"] > cm2_window_end) | base["approve_refund_time"].isna()),
                "order_number",
            ]
            cm2_n_day_retention = int(cm2_slice.nunique())

        if ls9_launch is not None:
            ls9_window_end = ls9_launch + pd.Timedelta(days=n_days)
            ls9_slice = base.loc[
                base["series_group_logic"].eq("LS9")
                & (base["lock_time"] >= ls9_launch)
                & (base["lock_time"] < ls9_window_end)
                & ((base["approve_refund_time"] > ls9_window_end) | base["approve_refund_time"].isna()),
                "order_number",
            ]
            ls9_n_day_retention = int(ls9_slice.nunique())

    start = as_of_date
    end = start + pd.Timedelta(days=1)
    df_model = df[df["series_group_logic"].eq(target_model)].copy()
    if not df_model.empty:
        lock_mask = df_model["lock_time"].notna() & (df_model["lock_time"] >= start) & (df_model["lock_time"] < end)
        user_car_mask = df_model["order_type"] == "用户车"
        hq_mask = (df_model["store_city"].astype("string") == "拉萨市") & (
            df_model["store_name"].astype("string") == "总部主理店"
        )
        hq_mask = hq_mask.fillna(False)
        user_lock_mask = lock_mask & user_car_mask & (~hq_mask)
        today_lock_count = int(df_model.loc[lock_mask, "order_number"].nunique())
        today_user_car_lock_count = int(df_model.loc[user_lock_mask, "order_number"].nunique())
        locked_user_car = df_model.loc[
            user_lock_mask,
            [
                "order_number",
                "intention_payment_time",
                "intention_refund_time",
                "deposit_payment_time",
                "lock_time",
            ],
        ]
        if not locked_user_car.empty:
            cond_intention = locked_user_car["intention_payment_time"].notna() & (
                locked_user_car["intention_refund_time"].isna()
                | (locked_user_car["intention_refund_time"] > locked_user_car["lock_time"])
            )
            cond_deposit = (~cond_intention) & locked_user_car["deposit_payment_time"].notna()
            locked_deposit = locked_user_car[cond_deposit]

            path_intention = int(locked_user_car.loc[cond_intention, "order_number"].nunique())
            path_retained_deposit_conv = int(
                locked_deposit.loc[locked_deposit["deposit_payment_time"] < start, "order_number"].nunique()
            )
            path_today_deposit_conv = int(
                locked_deposit.loc[
                    (locked_deposit["deposit_payment_time"] >= start) & (locked_deposit["deposit_payment_time"] < end),
                    "order_number",
                ].nunique()
            )

        historic_deposit = (
            user_car_mask
            & (~hq_mask)
            & df_model["deposit_payment_time"].notna()
            & (df_model["deposit_payment_time"] >= start)
            & (df_model["deposit_payment_time"] < end)
        )
        not_locked_yet = df_model["lock_time"].isna() | (df_model["lock_time"] >= end)
        not_refunded_yet = df_model["deposit_refund_time"].isna() | (df_model["deposit_refund_time"] >= end)
        path_today_deposit_unlocked = int(
            df_model.loc[historic_deposit & not_locked_yet & not_refunded_yet, "order_number"].nunique()
        )

    return {
        "target_model": target_model,
        "today": today.date().isoformat(),
        "target_launch": target_launch.date().isoformat() if target_launch is not None else None,
        "n": n,
        "n_raw": n_raw,
        "target_cum": target_cum,
        "target_retention": target_retention,
        "target_retention_users": target_retention_users,
        "target_peak_hour": target_peak_hour,
        "target_peak_count": target_peak_count,
        "target_next_hour_count": target_next_hour_count,
        "target_launch_day_total": target_launch_day_total,
        "cm2_n_day_retention": cm2_n_day_retention,
        "ls9_n_day_retention": ls9_n_day_retention,
        "as_of_date": as_of_date.date().isoformat(),
        "run_date": run_date.date().isoformat(),
        "max_lock_time": (pd.Timestamp(max_lock_time).isoformat(sep=" ", timespec="seconds") if pd.notna(max_lock_time) else None),
        "max_deposit_payment_time": (
            pd.Timestamp(max_deposit_payment_time).isoformat(sep=" ", timespec="seconds") if pd.notna(max_deposit_payment_time) else None
        ),
        "max_intention_payment_time": (
            pd.Timestamp(max_intention_payment_time).isoformat(sep=" ", timespec="seconds") if pd.notna(max_intention_payment_time) else None
        ),
        "today_lock_count": today_lock_count,
        "today_user_car_lock_count": today_user_car_lock_count,
        "path_intention": path_intention,
        "path_retained_deposit_conv": path_retained_deposit_conv,
        "path_today_deposit_conv": path_today_deposit_conv,
        "path_today_deposit_unlocked": path_today_deposit_unlocked,
    }


def build_feishu_card(metrics: dict) -> dict:
    lines: List[str] = []

    target_model = metrics["target_model"]
    peak_hour_str = f"{metrics['target_peak_hour']:02d}:00" if metrics["target_peak_hour"] is not None else "NA"
    display_date = metrics.get("as_of_date") or metrics["today"]

    lines.append(f"{target_model} 上市锁单指标（{display_date}）")

    lines.append("")
    lines.append(f"**锁单数：** {metrics['today_lock_count']}")
    lines.append(f"**锁单数（order_type =用户车&非总部）：** {metrics['today_user_car_lock_count']}")

    lines.append("")
    lines.append("**锁单路径盘点：**")
    lines.append(f"- 留存小订转化：{metrics['path_intention']}")
    lines.append(f"- 留存大定转化：{metrics['path_retained_deposit_conv']}")
    lines.append(f"- 当日大定转化：{metrics['path_today_deposit_conv']}")
    lines.append(f"- 当日大定留存：{metrics['path_today_deposit_unlocked']}")

    lines.append("")
    lines.append(f"**{target_model} 当前累计锁单数：** {metrics['target_cum']}")
    lines.append(f"- 峰值小时锁单数：{metrics['target_peak_count']}（{peak_hour_str}）")
    lines.append(f"- 峰值后1h：{metrics['target_next_hour_count']}")
    lines.append(f"- 上市当日累计：{metrics['target_launch_day_total']}")
    lines.append(f"- 上市至今累计留存：{metrics['target_retention']}")
    lines.append(f"- 累计留存唯一订单用户数：{metrics['target_retention_users']}")
    lines.append(
        f"- 历史上市至N日累计：CM2（{metrics['cm2_n_day_retention']}）｜LS9（{metrics['ls9_n_day_retention']}）"
    )

    if metrics.get("target_launch"):
        lines.append("")
        lines.append(f"上市日： {metrics['target_launch']}")
        lines.append(f"N（日）： {metrics['n']}（定义：当前日期 - {target_model} 上市日 + 1）")

    if metrics.get("as_of_date") and metrics.get("run_date") and metrics["as_of_date"] != metrics["run_date"]:
        lines.append("")
        lines.append(f"当日指标日期：{metrics['as_of_date']}（运行日 {metrics['run_date']}，数据未覆盖到运行日）")

    body_md = "\n".join(lines)
    return {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": f"📈 {target_model} 上市锁单监控（{display_date}）"},
                "template": "blue",
            },
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": body_md}},
                {
                    "tag": "note",
                    "elements": [{"tag": "plain_text", "content": f"统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}],
                },
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="读取 order_data.parquet 并发送指定车型上市锁单监控指标到飞书")
    parser.add_argument("--models", type=str, required=True, help="要监控的车型（对应 business_definition.json 中的 series_group_logic 键，例如 LS8 或 LS9）")
    parser.add_argument("--dry-run", action="store_true", help="只打印不发送飞书")
    args = parser.parse_args()

    if not PARQUET_FILE.exists():
        print(f"❌ 文件不存在: {PARQUET_FILE}")
        return 1

    business_def = load_business_definition(BUSINESS_DEF_FILE)
    
    target_model = args.models.strip()
    if target_model not in business_def.get("series_group_logic", {}):
        print(f"⚠️ 警告: 车型 '{target_model}' 不在 business_definition.json 的 series_group_logic 中，但仍会继续执行。")

    print(f"📖 Loading: {PARQUET_FILE}")
    df = pd.read_parquet(PARQUET_FILE)
    df = apply_series_group_logic(df, business_def)

    today = pd.Timestamp(datetime.now().date())
    metrics = compute_launch_lock_metrics(df, business_def, today, target_model)
    card = build_feishu_card(metrics)

    if args.dry_run or not WEBHOOK_URL:
        if not WEBHOOK_URL:
            print("⚠️ 未设置 FS_WEBHOOK_URL，跳过发送")
        print(json.dumps(card, ensure_ascii=False, indent=2))
        return 0

    try:
        resp = requests.post(WEBHOOK_URL, json=card, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        code = result.get("StatusCode", result.get("code"))
        if code == 0:
            print("✅ 飞书消息发送成功")
            return 0
        print(f"❌ 飞书返回异常: {result}")
        return 1
    except Exception as e:
        print(f"❌ 发送飞书消息失败: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
