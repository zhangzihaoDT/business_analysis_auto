#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
读取 order_data.parquet，按 business_definition.json 的口径计算 LS8 预售小订监控指标，并通过飞书自定义机器人 Webhook 发送交互式卡片。
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


def compute_presale_metrics(df: pd.DataFrame, business_def: dict, today: pd.Timestamp) -> dict:
    if "intention_payment_time" not in df.columns:
        raise KeyError("数据缺少列: intention_payment_time")
    if "order_number" not in df.columns:
        raise KeyError("数据缺少列: order_number")
    if "series_group_logic" not in df.columns:
        raise KeyError("数据缺少列: series_group_logic")
    if "intention_refund_time" not in df.columns:
        raise KeyError("数据缺少列: intention_refund_time")
    if "product_name" not in df.columns:
        raise KeyError("数据缺少列: product_name")
    if "parent_region_name" not in df.columns:
        raise KeyError("数据缺少列: parent_region_name")
    if "buyer_identity_no" not in df.columns:
        raise KeyError("数据缺少列: buyer_identity_no")

    if not pd.api.types.is_datetime64_any_dtype(df["intention_payment_time"]):
        df["intention_payment_time"] = pd.to_datetime(df["intention_payment_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["intention_refund_time"]):
        df["intention_refund_time"] = pd.to_datetime(df["intention_refund_time"], errors="coerce")

    time_periods: Dict[str, Dict[str, str]] = business_def.get("time_periods", {})
    ls8_tp = time_periods.get("LS8", {}) or {}
    ls8_start = pd.Timestamp(ls8_tp.get("start")) if ls8_tp.get("start") else None
    ls8_end = pd.Timestamp(ls8_tp.get("end")) if ls8_tp.get("end") else None

    n = 1
    if ls8_start is not None:
        n_raw = int((today.normalize() - ls8_start.normalize()).days + 1)
        n = max(1, n_raw)

    base = df.loc[
        df["intention_payment_time"].notna(),
        [
            "order_number",
            "intention_payment_time",
            "intention_refund_time",
            "series_group_logic",
            "product_name",
            "parent_region_name",
            "buyer_identity_no",
            "store_name",
        ],
    ].copy()

    ls8_cum = 0
    ls8_retention = 0
    ls8_retention_users = 0
    ls8_retention_by_product: List[dict] = []
    ls8_retention_by_region: List[dict] = []
    ls8_peak_hour = None
    ls8_peak_count = 0
    ls8_next_hour_count = 0
    ls8_start_day_total = 0
    ls8_n_day_cum = 0
    cm2_n_day_retention = 0
    ls9_n_day_retention = 0

    if ls8_start is not None:
        ls8_slice = base.loc[
            base["series_group_logic"].eq("LS8")
            & (base["intention_payment_time"] >= ls8_start)
            & (base["intention_payment_time"] < (today + pd.Timedelta(days=1))),
            "order_number",
        ]
        ls8_cum = int(ls8_slice.nunique())

        ls8_retention_slice = base.loc[
            base["series_group_logic"].eq("LS8")
            & (base["intention_payment_time"] >= ls8_start)
            & (base["intention_payment_time"] < (today + pd.Timedelta(days=1)))
            & (base["intention_refund_time"].isna()),
            ["order_number", "product_name", "parent_region_name", "buyer_identity_no", "store_name"],
        ]
        ls8_retention = int(ls8_retention_slice["order_number"].nunique())

        if not ls8_retention_slice.empty:
            order_counts_per_user = ls8_retention_slice.groupby("buyer_identity_no")["order_number"].nunique()
            ls8_retention_users = int((order_counts_per_user == 1).sum())
        else:
            ls8_retention_users = 0

        if ls8_retention > 0 and not ls8_retention_slice.empty:
            product_counts = ls8_retention_slice.groupby("product_name")["order_number"].nunique()
            for product_name, count in product_counts.items():
                share = count / ls8_retention * 100 if ls8_retention > 0 else 0
                ls8_retention_by_product.append(
                    {
                        "product_name": product_name,
                        "count": int(count),
                        "share": round(share, 1),
                    }
                )
            ls8_retention_by_product = sorted(ls8_retention_by_product, key=lambda x: x["count"], reverse=True)

            region_counts = ls8_retention_slice.groupby("parent_region_name")["order_number"].nunique()
            for region_name, count in region_counts.items():
                share = count / ls8_retention * 100 if ls8_retention > 0 else 0

                cr5 = None
                if "store_name" in ls8_retention_slice.columns:
                    region_slice = ls8_retention_slice[ls8_retention_slice["parent_region_name"] == region_name]
                    if not region_slice.empty:
                        store_counts = (
                            region_slice.dropna(subset=["store_name"]).groupby("store_name")["order_number"].nunique()
                        )
                        total = float(store_counts.sum())
                        if total > 0:
                            top5 = float(store_counts.nlargest(5).sum())
                            cr5 = round(top5 / total * 100, 1)

                ls8_retention_by_region.append(
                    {
                        "region_name": region_name,
                        "count": int(count),
                        "share": round(share, 1),
                        "cr5": cr5,
                    }
                )
            ls8_retention_by_region = sorted(ls8_retention_by_region, key=lambda x: x["count"], reverse=True)

        start_excl = ls8_start + pd.Timedelta(days=1)
        day_slice = base.loc[
            base["series_group_logic"].eq("LS8")
            & (base["intention_payment_time"] >= ls8_start)
            & (base["intention_payment_time"] < start_excl),
            ["order_number", "intention_payment_time"],
        ].copy()

        if not day_slice.empty:
            day_slice["hour"] = day_slice["intention_payment_time"].dt.hour.astype("int64")
            hourly = day_slice.groupby("hour")["order_number"].nunique().reindex(range(24), fill_value=0)
            ls8_peak_hour = int(hourly.idxmax())
            ls8_peak_count = int(hourly.iloc[ls8_peak_hour])
            ls8_next_hour_count = int(hourly.iloc[ls8_peak_hour + 1]) if ls8_peak_hour < 23 else 0
            ls8_start_day_total = int(hourly.sum())

        if ls8_end is not None:
            end_limit_excl = ls8_end + pd.Timedelta(days=1)
            window_end_excl = min(ls8_start + pd.Timedelta(days=n), end_limit_excl)
        else:
            window_end_excl = ls8_start + pd.Timedelta(days=n)

        window_slice = base.loc[
            base["series_group_logic"].eq("LS8")
            & (base["intention_payment_time"] >= ls8_start)
            & (base["intention_payment_time"] < window_end_excl),
            "order_number",
        ]
        ls8_n_day_cum = int(window_slice.nunique())

        if ls8_start is not None:
            cm2_start = pd.to_datetime(business_def["time_periods"]["CM2"]["start"])
            ls9_start = pd.to_datetime(business_def["time_periods"]["LS9"]["start"])

            n_days = (today - ls8_start).days + 1

            cm2_window_end = cm2_start + pd.Timedelta(days=n_days)
            cm2_slice = base.loc[
                base["series_group_logic"].eq("CM2")
                & (base["intention_payment_time"] >= cm2_start)
                & (base["intention_payment_time"] < cm2_window_end)
                & ((base["intention_refund_time"] > cm2_window_end) | base["intention_refund_time"].isna()),
                "order_number",
            ]
            cm2_n_day_retention = int(cm2_slice.nunique())

            ls9_window_end = ls9_start + pd.Timedelta(days=n_days)
            ls9_slice = base.loc[
                base["series_group_logic"].eq("LS9")
                & (base["intention_payment_time"] >= ls9_start)
                & (base["intention_payment_time"] < ls9_window_end)
                & ((base["intention_refund_time"] > ls9_window_end) | base["intention_refund_time"].isna()),
                "order_number",
            ]
            ls9_n_day_retention = int(ls9_slice.nunique())

    return {
        "today": today.date().isoformat(),
        "ls8_start": ls8_start.date().isoformat() if ls8_start is not None else None,
        "ls8_end": ls8_end.date().isoformat() if ls8_end is not None else None,
        "n": n,
        "n_raw": n_raw,
        "ls8_cum": ls8_cum,
        "ls8_retention": ls8_retention,
        "ls8_retention_users": ls8_retention_users,
        "ls8_retention_by_product": ls8_retention_by_product,
        "ls8_retention_by_region": ls8_retention_by_region,
        "ls8_peak_hour": ls8_peak_hour,
        "ls8_peak_count": ls8_peak_count,
        "ls8_next_hour_count": ls8_next_hour_count,
        "ls8_start_day_total": ls8_start_day_total,
        "ls8_n_day_cum": ls8_n_day_cum,
        "cm2_n_day_retention": cm2_n_day_retention,
        "ls9_n_day_retention": ls9_n_day_retention,
    }


def build_feishu_card(metrics: dict) -> dict:
    lines: List[str] = []

    peak_hour_str = f"{metrics['ls8_peak_hour']:02d}:00" if metrics["ls8_peak_hour"] is not None else "NA"

    lines.append(f"**LS8 预售指标（{metrics['today']}）**")
    lines.append("")
    lines.append(f"LS8 当前累计小订数： {metrics['ls8_cum']}")
    lines.append(f"- 峰值小时小订数：{metrics['ls8_peak_count']}（{peak_hour_str}）")
    lines.append(f"- 峰值后1h：{metrics['ls8_next_hour_count']}")
    lines.append(f"- 预售当日累计：{metrics['ls8_start_day_total']}")
    lines.append(f"- 预售至今累计留存：{metrics['ls8_retention']}")
    lines.append(f"- 累计留存唯一订单用户数：{metrics['ls8_retention_users']}")
    lines.append(
        f"- 历史预售至N日累计：CM2（{metrics['cm2_n_day_retention']}）｜LS9（{metrics['ls9_n_day_retention']}）"
    )

    if metrics.get("ls8_retention_by_product"):
        lines.append("")
        lines.append("**累计留存分 product_name：**")
        for item in metrics["ls8_retention_by_product"]:
            lines.append(f"  - {item['product_name']}：{item['count']}（{item['share']}%）")

    if metrics.get("ls8_retention_by_region"):
        lines.append("")
        lines.append("**累计留存分 parent_region_name：**")
        for item in metrics["ls8_retention_by_region"]:
            cr5_str = f"｜CR5（{item['cr5']}%）" if item.get("cr5") is not None else ""
            lines.append(f"  - {item['region_name']}：{item['count']}（{item['share']}%）{cr5_str}")

    if metrics.get("ls8_start") and metrics.get("ls8_end"):
        lines.append("")
        lines.append(f"**预售期：** {metrics['ls8_start']} ~ {metrics['ls8_end']}")
        lines.append(f"**N（日）：** {metrics['n']}（定义：当前日期 - LS8 startday + 1）")

    body_md = "\n".join(lines)
    return {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": f"📊 LS8 预售小订监控（{metrics['today']}）"},
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
    parser = argparse.ArgumentParser(description="读取 order_data.parquet 并发送 LS8 预售小订监控指标到飞书")
    parser.add_argument("--dry-run", action="store_true", help="只打印不发送飞书")
    args = parser.parse_args()

    if not PARQUET_FILE.exists():
        print(f"❌ 文件不存在: {PARQUET_FILE}")
        return 1

    business_def = load_business_definition(BUSINESS_DEF_FILE)

    print(f"📖 Loading: {PARQUET_FILE}")
    df = pd.read_parquet(PARQUET_FILE)
    df = apply_series_group_logic(df, business_def)

    today = pd.Timestamp(datetime.now().date())
    metrics = compute_presale_metrics(df, business_def, today)
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
