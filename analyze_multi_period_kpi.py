#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BASE_DIR = Path("/Users/zihao_/Documents/coding/dataset")
ASSIGN_CSV = BASE_DIR / "original" / "assign_data.csv"
ORDER_PARQUET = BASE_DIR / "formatted" / "order_full_data.parquet"
BUSINESS_DEF_FILE = Path("/Users/zihao_/Documents/github/W52_reasoning/world/business_definition.json")

COLOR_MAIN = "#3498DB"
COLOR_CONTRAST = "#E67E22"
COLOR_DARK = "#373f4a"
COLOR_GRID = "#ebedf0"
COLOR_TEXT = "#7B848F"
COLOR_BG = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="多指标双时间窗对比分析（assign_data.csv + order_full_data.parquet）"
    )
    parser.add_argument("--start1", required=True, help="时间窗1起始日期 YYYY-MM-DD")
    parser.add_argument("--end1", required=True, help="时间窗1结束日期 YYYY-MM-DD")
    parser.add_argument("--start2", required=True, help="时间窗2起始日期 YYYY-MM-DD")
    parser.add_argument("--end2", required=True, help="时间窗2结束日期 YYYY-MM-DD")
    parser.add_argument(
        "--out",
        default=str(BASE_DIR / "reports" / "multi_period_kpi.html"),
        help="输出 HTML 报告路径",
    )
    return parser.parse_args()


def load_assign_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-16", sep="\t")
    df = df.rename(
        columns={
            "Assign Time 年/月/日": "date",
            "下发线索当日锁单数 (门店)": "same_day_lock_store",
            "下发线索数 (门店)": "leads_store",
            "下发线索数": "leads_total",
        }
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y年%m月%d日", errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    df["distribution_ratio"] = df["leads_store"] / df["leads_total"]
    df = df[df["leads_total"] > 0].copy()
    df["conversion_efficiency"] = df["same_day_lock_store"] / df["leads_store"]
    return df


def load_business_definition(json_path: Path) -> dict | None:
    try:
        import json
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _norm_text(s: str) -> str:
    return s.strip().lower()


def _match_condition(text: str, product_name: str) -> bool:
    t = _norm_text(text)
    name = _norm_text(product_name or "")
    if "like" in t and "%" in t:
        # simple LIKE '%keyword%'
        parts = [p.strip() for p in t.replace("like", "").replace("'", "").split("%") if p.strip()]
        return all(p in name for p in parts)
    # fallback contains
    return t in name if t else False


def apply_business_logic(df: pd.DataFrame, business_def: dict | None) -> pd.DataFrame:
    if business_def is None or "product_rules" not in business_def:
        return df
    rules = business_def.get("product_rules", [])
    df = df.copy()
    if "series_derived" not in df.columns:
        df["series_derived"] = df.get("series", pd.Series([None] * len(df)))
    if "product_type" not in df.columns:
        df["product_type"] = None
    for r in rules:
        cond = r.get("condition", "")
        set_fields = r.get("set", {})
        if not cond or not set_fields:
            continue
        mask = df["product_name"].astype(str).apply(lambda x: _match_condition(cond, x))
        for k, v in set_fields.items():
            df.loc[mask, k] = v
    return df


def load_order_data(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if "lock_time" in df.columns:
        df["lock_time"] = pd.to_datetime(df["lock_time"], errors="coerce")
    if "invoice_upload_time" in df.columns:
        df["invoice_upload_time"] = pd.to_datetime(
            df["invoice_upload_time"], errors="coerce"
        )
    if "delivery_date" in df.columns:
        df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    return df


def filter_assign_window(df_assign: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    mask = (df_assign["date"] >= start_dt) & (df_assign["date"] <= end_dt)
    return df_assign.loc[mask].copy()


def filter_order_window(df_order: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    if "lock_time" not in df_order.columns:
        return df_order.iloc[0:0].copy()
    mask = (df_order["lock_time"] >= start_dt) & (df_order["lock_time"] <= end_dt)
    return df_order.loc[mask].copy()


def _resolve_assign_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in cols:
            return cols[key]
    return None


def calc_assign_metrics(df_window: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    if df_window.empty:
        return pd.DataFrame(
            [
                {
                    "group": "traffic",
                    "metric": "assign_leads_total",
                    "label": "下发线索数",
                    "window": label,
                    "value": 0.0,
                    "sigma": np.nan,
                },
                {
                    "group": "traffic",
                    "metric": "assign_leads_store_total",
                    "label": "下发线索数(门店)",
                    "window": label,
                    "value": 0.0,
                    "sigma": np.nan,
                },
                {
                    "group": "traffic",
                    "metric": "assign_store_leads_share",
                    "label": "门店线索占比",
                    "window": label,
                    "value": np.nan,
                    "sigma": np.nan,
                },
                {
                    "group": "traffic",
                    "metric": "assign_same_day_lock_store_total",
                    "label": "门店线索当日锁单数",
                    "window": label,
                    "value": 0.0,
                    "sigma": np.nan,
                },
                {
                    "group": "conversion",
                    "metric": "assign_same_day_lock_rate",
                    "label": "门店线索当日锁单率",
                    "window": label,
                    "value": np.nan,
                    "sigma": np.nan,
                },
                {
                    "group": "conversion",
                    "metric": "rate_7d_lock",
                    "label": "下发线索7日锁单率",
                    "window": label,
                    "value": np.nan,
                    "sigma": np.nan,
                },
                {
                    "group": "conversion",
                    "metric": "rate_30d_lock",
                    "label": "下发线索30日锁单率",
                    "window": label,
                    "value": np.nan,
                    "sigma": np.nan,
                },
            ]
        )
    leads_total_series = df_window["leads_total"].astype(float)
    leads_store_series = df_window["leads_store"].astype(float)
    same_day_lock_series = df_window["same_day_lock_store"].astype(float)
    leads_total = float(leads_total_series.sum())
    leads_store = float(leads_store_series.sum())
    same_day_lock_store = float(same_day_lock_series.sum())
    store_leads_share = leads_store / leads_total if leads_total > 0 else np.nan

    # optional 7d/30d counts
    col_7d = _resolve_assign_col(
        df_window,
        [
            "下发线索7日锁单数",
            "下发线索 7 日锁单数",
            "7日线索锁单数",
            "7日内锁单线索数",
            "7日锁单数",
        ],
    )
    col_30d = _resolve_assign_col(
        df_window,
        [
            "下发线索30日锁单数",
            "下发线索 30 日锁单数",
            "30日线索锁单数",
            "30日内锁单线索数",
            "30日锁单数",
        ],
    )
    lock_7d_series = df_window[col_7d].astype(float) if col_7d else pd.Series(dtype=float)
    lock_30d_series = df_window[col_30d].astype(float) if col_30d else pd.Series(dtype=float)
    lock_7d = float(lock_7d_series.sum()) if col_7d else np.nan
    lock_30d = float(lock_30d_series.sum()) if col_30d else np.nan
    rate_7d = lock_7d / leads_total if (col_7d and leads_total > 0) else np.nan
    rate_30d = lock_30d / leads_total if (col_30d and leads_total > 0) else np.nan

    sigma_assign_leads_total = leads_total_series.std(ddof=0) if not leads_total_series.empty else np.nan
    sigma_assign_leads_store = leads_store_series.std(ddof=0) if not leads_store_series.empty else np.nan
    daily_share_series = np.where(
        leads_total_series > 0, leads_store_series / leads_total_series, np.nan
    )
    sigma_store_leads_share = (
        pd.Series(daily_share_series).dropna().std(ddof=0) if leads_total_series.size > 0 else np.nan
    )
    sigma_same_day_lock_total = same_day_lock_series.std(ddof=0) if not same_day_lock_series.empty else np.nan
    daily_same_day_rate = np.where(
        leads_store_series > 0, same_day_lock_series / leads_store_series, np.nan
    )
    if leads_store_series.size > 0:
        same_day_rate_series = pd.Series(daily_same_day_rate).replace([np.inf, -np.inf], np.nan).dropna()
        if not same_day_rate_series.empty:
            same_day_rate = same_day_rate_series.mean()
            sigma_same_day_rate = same_day_rate_series.std(ddof=0)
        else:
            same_day_rate = np.nan
            sigma_same_day_rate = np.nan
    else:
        same_day_rate = np.nan
        sigma_same_day_rate = np.nan
    if col_7d:
        daily_rate_7d = np.where(
            leads_total_series > 0, lock_7d_series / leads_total_series, np.nan
        )
        rate_7d_series = pd.Series(daily_rate_7d).replace([np.inf, -np.inf], np.nan).dropna()
        if not rate_7d_series.empty:
            rate_7d = rate_7d_series.mean()
            sigma_rate_7d = rate_7d_series.std(ddof=0)
        else:
            rate_7d = np.nan
            sigma_rate_7d = np.nan
    else:
        rate_7d = np.nan
        sigma_rate_7d = np.nan
    if col_30d:
        daily_rate_30d = np.where(
            leads_total_series > 0, lock_30d_series / leads_total_series, np.nan
        )
        rate_30d_series = pd.Series(daily_rate_30d).replace([np.inf, -np.inf], np.nan).dropna()
        if not rate_30d_series.empty:
            rate_30d = rate_30d_series.mean()
            sigma_rate_30d = rate_30d_series.std(ddof=0)
        else:
            rate_30d = np.nan
            sigma_rate_30d = np.nan
    else:
        rate_30d = np.nan
        sigma_rate_30d = np.nan

    rows.append(
        {
            "group": "traffic",
            "metric": "assign_leads_total",
            "label": "下发线索数",
            "window": label,
            "value": leads_total,
            "sigma": float(sigma_assign_leads_total) if pd.notna(sigma_assign_leads_total) else np.nan,
        }
    )
    rows.append(
        {
            "group": "traffic",
            "metric": "assign_leads_store_total",
            "label": "下发线索数(门店)",
            "window": label,
            "value": leads_store,
            "sigma": float(sigma_assign_leads_store) if pd.notna(sigma_assign_leads_store) else np.nan,
        }
    )
    rows.append(
        {
            "group": "traffic",
            "metric": "assign_store_leads_share",
            "label": "门店线索占比",
            "window": label,
            "value": store_leads_share,
            "sigma": float(sigma_store_leads_share) if pd.notna(sigma_store_leads_share) else np.nan,
        }
    )
    rows.append(
        {
            "group": "conversion",
            "metric": "assign_same_day_lock_store_total",
            "label": "门店线索当日锁单数",
            "window": label,
            "value": same_day_lock_store / len(df_window) if len(df_window) > 0 else np.nan,
            "sigma": float(sigma_same_day_lock_total) if pd.notna(sigma_same_day_lock_total) else np.nan,
        }
    )
    rows.append(
        {
            "group": "conversion",
            "metric": "assign_same_day_lock_rate",
            "label": "门店线索当日锁单率",
            "window": label,
            "value": same_day_rate,
            "sigma": float(sigma_same_day_rate) if pd.notna(sigma_same_day_rate) else np.nan,
        }
    )
    rows.append(
        {
            "group": "conversion",
            "metric": "rate_7d_lock",
            "label": "下发线索7日锁单率",
            "window": label,
            "value": rate_7d,
            "sigma": float(sigma_rate_7d) if pd.notna(sigma_rate_7d) else np.nan,
        }
    )
    rows.append(
        {
            "group": "conversion",
            "metric": "rate_30d_lock",
            "label": "下发线索30日锁单率",
            "window": label,
            "value": rate_30d,
            "sigma": float(sigma_rate_30d) if pd.notna(sigma_rate_30d) else np.nan,
        }
    )
    return pd.DataFrame(rows)


def calc_order_metrics_full(df_order: pd.DataFrame, start: str, end: str, label: str) -> pd.DataFrame:
    rows = []
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    # lock-window
    if "lock_time" in df_order.columns:
        lock_win = df_order[(df_order["lock_time"] >= start_dt) & (df_order["lock_time"] <= end_dt)].copy()
    else:
        lock_win = df_order.iloc[0:0].copy()
    # invoice-window
    if "invoice_upload_time" in df_order.columns:
        inv_win = df_order[(df_order["invoice_upload_time"] >= start_dt) & (df_order["invoice_upload_time"] <= end_dt)].copy()
        # ensure lock presence for invoice stats, aligned with skills script
        if "lock_time" in inv_win.columns:
            inv_win = inv_win[inv_win["lock_time"].notna()]
    else:
        inv_win = df_order.iloc[0:0].copy()
    # delivery-window
    if "delivery_date" in df_order.columns:
        deliv_win = df_order[(df_order["delivery_date"] >= start_dt) & (df_order["delivery_date"] <= end_dt)].copy()
    else:
        deliv_win = df_order.iloc[0:0].copy()

    if lock_win.empty and inv_win.empty and deliv_win.empty:
        base = [
            ("order_lock_total", "订单锁单数", "traffic"),
            ("order_invoice_total", "订单开票数", "traffic"),
            ("order_delivery_total", "订单交付数", "traffic"),
            ("order_user_invoice_total", "用户车开票数", "structure"),
            ("order_user_invoice_share", "用户车开票占比", "structure"),
            ("order_atp_overall", "整体ATP(用户车,万元)", "structure"),
            ("share_l6", "L6 销量占比", "structure"),
            ("share_ls6", "LS6 销量占比", "structure"),
            ("share_ls9", "LS9 销量占比", "structure"),
            ("share_reev", "增程车型销量占比", "structure"),
            ("share_ls6_reev", "LS6增程销量占比", "structure"),
            ("cr5_store_conc", "CR5门店销量集中度", "structure"),
            ("cr5_city_conc", "CR5门店城市销量集中度", "structure"),
        ]
        return pd.DataFrame(
            [
                {
                    "group": g,
                    "metric": m,
                    "label": l,
                    "window": label,
                    "value": np.nan if ("share" in m or "atp" in m) else 0.0,
                    "sigma": np.nan,
                }
                for m, l, g in base
            ]
        )
    # daily series for volatility
    if not lock_win.empty:
        lock_win = lock_win.copy()
        lock_win["_date"] = lock_win["lock_time"].dt.date
        daily_lock_total = lock_win.groupby("_date")["order_number"].nunique()
    else:
        daily_lock_total = pd.Series(dtype=float)
    if not inv_win.empty:
        inv_win = inv_win.copy()
        inv_win["_date"] = inv_win["invoice_upload_time"].dt.date
        daily_invoice_total = inv_win.groupby("_date")["order_number"].nunique()
    else:
        daily_invoice_total = pd.Series(dtype=float)
    if not deliv_win.empty:
        deliv_win = deliv_win.copy()
        deliv_win["_date"] = deliv_win["delivery_date"].dt.date
        daily_delivery_total = deliv_win.groupby("_date")["order_number"].nunique()
    else:
        daily_delivery_total = pd.Series(dtype=float)

    # conversion: core totals
    lock_total = lock_win["order_number"].astype(str).nunique()
    invoice_total = inv_win["order_number"].astype(str).nunique()
    user_orders = inv_win[inv_win["order_type"] == "用户车"] if "order_type" in inv_win.columns else inv_win.iloc[0:0]
    user_invoice_total = user_orders["order_number"].astype(str).nunique()
    user_share = user_invoice_total / invoice_total if invoice_total > 0 else np.nan
    if not user_orders.empty:
        user_orders = user_orders.copy()
        user_orders["_date"] = user_orders["invoice_upload_time"].dt.date
        daily_user_invoice_total = user_orders.groupby("_date")["order_number"].nunique()
    else:
        daily_user_invoice_total = pd.Series(dtype=float)
    if "invoice_amount" in user_orders.columns:
        atp = user_orders["invoice_amount"]
        atp = atp[atp > 0].mean()
        atp_wan = atp / 10000 if pd.notna(atp) else np.nan
        daily_atp_wan = (
            user_orders.groupby("_date")["invoice_amount"].mean() / 10000
        )
        sigma_atp_wan = daily_atp_wan.dropna().std(ddof=0)
    else:
        atp_wan = np.nan
        sigma_atp_wan = np.nan
    delivery_total = deliv_win["order_number"].astype(str).nunique()

    sigma_lock_total = daily_lock_total.std(ddof=0) if not daily_lock_total.empty else np.nan
    sigma_invoice_total = daily_invoice_total.std(ddof=0) if not daily_invoice_total.empty else np.nan
    sigma_delivery_total = daily_delivery_total.std(ddof=0) if not daily_delivery_total.empty else np.nan
    if not daily_invoice_total.empty:
        aligned_user = daily_user_invoice_total.reindex(daily_invoice_total.index, fill_value=0)
        daily_share_user = aligned_user / daily_invoice_total.replace(0, np.nan)
        sigma_user_share = daily_share_user.dropna().std(ddof=0)
    else:
        sigma_user_share = np.nan
    sigma_user_invoice_total = (
        daily_user_invoice_total.std(ddof=0) if not daily_user_invoice_total.empty else np.nan
    )
    rows.extend(
        [
            {
                "group": "traffic",
                "metric": "order_lock_total",
                "label": "订单锁单数",
                "window": label,
                "value": float(lock_total),
                "sigma": float(sigma_lock_total) if pd.notna(sigma_lock_total) else np.nan,
            },
            {
                "group": "traffic",
                "metric": "order_invoice_total",
                "label": "订单开票数",
                "window": label,
                "value": float(invoice_total),
                "sigma": float(sigma_invoice_total) if pd.notna(sigma_invoice_total) else np.nan,
            },
            {
                "group": "traffic",
                "metric": "order_delivery_total",
                "label": "订单交付数",
                "window": label,
                "value": float(delivery_total),
                "sigma": float(sigma_delivery_total) if pd.notna(sigma_delivery_total) else np.nan,
            },
            {
                "group": "structure",
                "metric": "order_user_invoice_total",
                "label": "用户车开票数",
                "window": label,
                "value": float(user_invoice_total),
                "sigma": float(sigma_user_invoice_total) if pd.notna(sigma_user_invoice_total) else np.nan,
            },
            {
                "group": "structure",
                "metric": "order_user_invoice_share",
                "label": "用户车开票占比",
                "window": label,
                "value": float(user_share) if pd.notna(user_share) else np.nan,
                "sigma": float(sigma_user_share) if pd.notna(sigma_user_share) else np.nan,
            },
            {
                "group": "structure",
                "metric": "order_atp_overall",
                "label": "整体ATP(用户车,万元)",
                "window": label,
                "value": float(atp_wan) if pd.notna(atp_wan) else np.nan,
                "sigma": float(sigma_atp_wan) if pd.notna(sigma_atp_wan) else np.nan,
            },
        ]
    )
    # structure: LS6/LS9 shares, REEV share, CR5
    total_locks = float(lock_total)
    if total_locks > 0:
        if "series" in lock_win.columns:
            l6_mask = lock_win["series"] == "L6"
            ls6_mask = lock_win["series"] == "LS6"
            ls9_mask = lock_win["series"] == "LS9"
        else:
            l6_mask = ls6_mask = ls9_mask = pd.Series([False] * len(lock_win), index=lock_win.index)
        l6_count = lock_win[l6_mask]["order_number"].astype(str).nunique()
        ls6_count = lock_win[ls6_mask]["order_number"].astype(str).nunique()
        ls9_count = lock_win[ls9_mask]["order_number"].astype(str).nunique()
        share_l6 = l6_count / total_locks
        share_ls6 = ls6_count / total_locks
        share_ls9 = ls9_count / total_locks
        if not lock_win.empty:
            daily_total = daily_lock_total.replace(0, np.nan)
            daily_l6 = (
                lock_win[l6_mask].groupby("_date")["order_number"].nunique()
                if l6_mask.any()
                else pd.Series(dtype=float)
            )
            daily_ls6 = (
                lock_win[ls6_mask].groupby("_date")["order_number"].nunique()
                if ls6_mask.any()
                else pd.Series(dtype=float)
            )
            daily_ls9 = (
                lock_win[ls9_mask].groupby("_date")["order_number"].nunique()
                if ls9_mask.any()
                else pd.Series(dtype=float)
            )
            daily_share_l6 = daily_l6.reindex(daily_total.index, fill_value=0) / daily_total
            daily_share_ls6 = daily_ls6.reindex(daily_total.index, fill_value=0) / daily_total
            daily_share_ls9 = daily_ls9.reindex(daily_total.index, fill_value=0) / daily_total
            sigma_share_l6 = daily_share_l6.dropna().std(ddof=0)
            sigma_share_ls6 = daily_share_ls6.dropna().std(ddof=0)
            sigma_share_ls9 = daily_share_ls9.dropna().std(ddof=0)
        else:
            sigma_share_l6 = sigma_share_ls6 = sigma_share_ls9 = np.nan
    else:
        share_l6 = share_ls6 = share_ls9 = np.nan
        sigma_share_l6 = sigma_share_ls6 = sigma_share_ls9 = np.nan
    # REEV share via product_type == '增程'
    if "product_type" in lock_win.columns:
        reev_mask = lock_win["product_type"] == "增程"
    else:
        reev_mask = (
            lock_win["product_name"].astype(str).str.contains("52|66", regex=True, na=False)
            if "product_name" in lock_win.columns
            else pd.Series([False] * len(lock_win), index=lock_win.index)
        )
    if total_locks > 0:
        reev_count = lock_win.loc[reev_mask, "order_number"].astype(str).nunique()
        share_reev = reev_count / total_locks
        ls6_reev_mask = reev_mask & ls6_mask
        ls6_reev_count = lock_win.loc[ls6_reev_mask, "order_number"].astype(str).nunique()
        share_ls6_reev = ls6_reev_count / ls6_count if ls6_count > 0 else np.nan
        if not lock_win.empty:
            daily_total = daily_lock_total.replace(0, np.nan)
            daily_reev = (
                lock_win[reev_mask].groupby("_date")["order_number"].nunique()
                if reev_mask.any()
                else pd.Series(dtype=float)
            )
            daily_share_reev = daily_reev.reindex(daily_total.index, fill_value=0) / daily_total
            sigma_share_reev = daily_share_reev.dropna().std(ddof=0)
            daily_ls6_total = (
                lock_win[ls6_mask].groupby("_date")["order_number"].nunique()
                if ls6_mask.any()
                else pd.Series(dtype=float)
            )
            daily_ls6_total = daily_ls6_total.replace(0, np.nan)
            daily_ls6_reev = (
                lock_win[ls6_reev_mask].groupby("_date")["order_number"].nunique()
                if ls6_reev_mask.any()
                else pd.Series(dtype=float)
            )
            daily_share_ls6_reev = daily_ls6_reev.reindex(daily_ls6_total.index, fill_value=0) / daily_ls6_total
            sigma_share_ls6_reev = daily_share_ls6_reev.dropna().std(ddof=0)
        else:
            sigma_share_reev = np.nan
            sigma_share_ls6_reev = np.nan
    else:
        share_reev = np.nan
        share_ls6_reev = np.nan
        sigma_share_reev = np.nan
        sigma_share_ls6_reev = np.nan
    if "store_name" in lock_win.columns and total_locks > 0:
        store_daily = lock_win.groupby(["_date", "store_name"])["order_number"].nunique()
        daily_total_store = store_daily.groupby(level=0).sum()
        top5_daily = (
            store_daily.sort_values(ascending=False)
            .groupby(level=0)
            .head(5)
            .groupby(level=0)
            .sum()
        )
        daily_cr5 = top5_daily / daily_total_store.replace(0, np.nan)
        cr5 = float(daily_cr5.mean()) if not daily_cr5.empty else np.nan
        sigma_cr5 = daily_cr5.dropna().std(ddof=0) if not daily_cr5.empty else np.nan
    else:
        cr5 = np.nan
        sigma_cr5 = np.nan
    if "store_city" in lock_win.columns and total_locks > 0:
        city_daily = lock_win.groupby(["_date", "store_city"])["order_number"].nunique()
        daily_total_city = city_daily.groupby(level=0).sum()
        top5_city_daily = (
            city_daily.sort_values(ascending=False)
            .groupby(level=0)
            .head(5)
            .groupby(level=0)
            .sum()
        )
        daily_cr5_city = top5_city_daily / daily_total_city.replace(0, np.nan)
        cr5_city = float(daily_cr5_city.mean()) if not daily_cr5_city.empty else np.nan
        sigma_cr5_city = daily_cr5_city.dropna().std(ddof=0) if not daily_cr5_city.empty else np.nan
    else:
        cr5_city = np.nan
        sigma_cr5_city = np.nan
    rows.extend(
        [
            {
                "group": "structure",
                "metric": "share_l6",
                "label": "L6 销量占比",
                "window": label,
                "value": float(share_l6) if pd.notna(share_l6) else np.nan,
                "sigma": float(sigma_share_l6) if pd.notna(sigma_share_l6) else np.nan,
            },
            {
                "group": "structure",
                "metric": "share_ls6",
                "label": "LS6 销量占比",
                "window": label,
                "value": float(share_ls6) if pd.notna(share_ls6) else np.nan,
                "sigma": float(sigma_share_ls6) if pd.notna(sigma_share_ls6) else np.nan,
            },
            {
                "group": "structure",
                "metric": "share_ls9",
                "label": "LS9 销量占比",
                "window": label,
                "value": float(share_ls9) if pd.notna(share_ls9) else np.nan,
                "sigma": float(sigma_share_ls9) if pd.notna(sigma_share_ls9) else np.nan,
            },
            {
                "group": "structure",
                "metric": "share_reev",
                "label": "增程车型销量占比",
                "window": label,
                "value": float(share_reev) if pd.notna(share_reev) else np.nan,
                "sigma": float(sigma_share_reev) if pd.notna(sigma_share_reev) else np.nan,
            },
            {
                "group": "structure",
                "metric": "share_ls6_reev",
                "label": "LS6增程销量占比",
                "window": label,
                "value": float(share_ls6_reev) if pd.notna(share_ls6_reev) else np.nan,
                "sigma": float(sigma_share_ls6_reev) if pd.notna(sigma_share_ls6_reev) else np.nan,
            },
            {
                "group": "structure",
                "metric": "cr5_store_conc",
                "label": "CR5门店销量集中度",
                "window": label,
                "value": float(cr5) if pd.notna(cr5) else np.nan,
                "sigma": float(sigma_cr5) if pd.notna(sigma_cr5) else np.nan,
            },
            {
                "group": "structure",
                "metric": "cr5_city_conc",
                "label": "CR5门店城市销量集中度",
                "window": label,
                "value": float(cr5_city) if pd.notna(cr5_city) else np.nan,
                "sigma": float(sigma_cr5_city) if pd.notna(sigma_cr5_city) else np.nan,
            },
        ]
    )
    return pd.DataFrame(rows)


def build_comparison_table(df_metrics: pd.DataFrame) -> pd.DataFrame:
    pivot_val = df_metrics.pivot_table(
        index=["group", "metric", "label"],
        columns="window",
        values="value",
        aggfunc="first",
    )
    pivot = pivot_val.reset_index()
    if {"W1", "W2"}.issubset(pivot.columns):
        w1 = pivot["W1"].astype(float)
        w2 = pivot["W2"].astype(float)
        diff = w2 - w1
        with np.errstate(divide="ignore", invalid="ignore"):
            diff_pct = diff / w1.replace(0, np.nan)
        pivot["Diff"] = diff
        pivot["Diff_Pct"] = diff_pct
        sigma_pivot = df_metrics.pivot_table(
            index=["group", "metric", "label"],
            columns="window",
            values="sigma",
            aggfunc="first",
        )
        sigma_pivot = sigma_pivot.rename(columns={"W1": "Sigma_W1", "W2": "Sigma_W2"})
        sigma_pivot["Sigma_Diff"] = sigma_pivot["Sigma_W2"] - sigma_pivot["Sigma_W1"]
        sigma_pivot = sigma_pivot.reset_index()
        pivot = pivot.merge(sigma_pivot, on=["group", "metric", "label"], how="left")
    else:
        pivot["Diff"] = np.nan
        pivot["Diff_Pct"] = np.nan
        pivot["Sigma_W1"] = np.nan
        pivot["Sigma_W2"] = np.nan
        pivot["Sigma_Diff"] = np.nan
    return pivot


def make_figure(df_comparison: pd.DataFrame, start1: str, end1: str, start2: str, end2: str) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.33, 0.33, 0.34],
        vertical_spacing=0.08,
        specs=[[{"type": "table"}], [{"type": "table"}], [{"type": "table"}]],
        subplot_titles=(
            f"流量侧",
            f"转化侧",
            f"结构侧",
        ),
    )
    metric_order_map = {
        "traffic": [
            "order_lock_total",
            "order_invoice_total",
            "order_delivery_total",
            "assign_leads_total",
            "assign_leads_store_total",
            "assign_store_leads_share",
        ],
        "conversion": [
            "assign_same_day_lock_store_total",
            "assign_same_day_lock_rate",
            "rate_7d_lock",
            "rate_30d_lock",
        ],
        "structure": [
            "order_user_invoice_share",
            "order_atp_overall",
            "order_user_invoice_total",
            "share_l6",
            "share_ls6",
            "share_ls9",
            "share_reev",
            "share_ls6_reev",
            "cr5_store_conc",
            "cr5_city_conc",
        ],
    }
    percent_metrics = {
        "assign_store_leads_share",
        "assign_same_day_lock_rate",
        "rate_7d_lock",
        "rate_30d_lock",
        "order_user_invoice_share",
        "share_l6",
        "share_ls6",
        "share_ls9",
        "share_reev",
        "share_ls6_reev",
        "cr5_store_conc",
        "cr5_city_conc",
    }
    for idx, group_name in enumerate(["traffic", "conversion", "structure"], start=1):
        sub = df_comparison[df_comparison["group"] == group_name].copy()
        if sub.empty:
            continue
        desired_order = metric_order_map.get(group_name)
        if desired_order:
            sub["metric"] = pd.Categorical(sub["metric"], categories=desired_order, ordered=True)
            sub = sub.sort_values("metric")
        labels = sub["label"].tolist()
        metrics = sub["metric"].tolist()
        raw_w1 = sub.get("W1", pd.Series([np.nan] * len(labels)))
        raw_w2 = sub.get("W2", pd.Series([np.nan] * len(labels)))
        raw_diff = sub["Diff"]
        raw_sigma_w1 = sub.get("Sigma_W1", pd.Series([np.nan] * len(labels)))
        raw_sigma_w2 = sub.get("Sigma_W2", pd.Series([np.nan] * len(labels)))
        raw_sigma_diff = sub.get("Sigma_Diff", pd.Series([np.nan] * len(labels)))
        w1_vals = []
        w2_vals = []
        diff_vals = []
        sigma_w1_vals = []
        sigma_w2_vals = []
        sigma_diff_vals = []
        for m, v1, v2, dv, s1, s2, sd in zip(
            metrics, raw_w1, raw_w2, raw_diff, raw_sigma_w1, raw_sigma_w2, raw_sigma_diff
        ):
            if m in percent_metrics:
                w1_vals.append(f"{v1*100:.1f}%" if pd.notna(v1) else "")
                w2_vals.append(f"{v2*100:.1f}%" if pd.notna(v2) else "")
                diff_vals.append(f"{dv*100:.1f}%" if pd.notna(dv) else "")
                sigma_w1_vals.append(f"{s1*100:.1f}%" if pd.notna(s1) else "")
                sigma_w2_vals.append(f"{s2*100:.1f}%" if pd.notna(s2) else "")
                sigma_diff_vals.append(f"{sd*100:.1f}%" if pd.notna(sd) else "")
            elif m == "order_atp_overall":
                w1_vals.append(f"{v1:.1f}" if pd.notna(v1) else "")
                w2_vals.append(f"{v2:.1f}" if pd.notna(v2) else "")
                diff_vals.append(f"{dv:.1f}" if pd.notna(dv) else "")
                sigma_w1_vals.append(f"{s1:.1f}" if pd.notna(s1) else "")
                sigma_w2_vals.append(f"{s2:.1f}" if pd.notna(s2) else "")
                sigma_diff_vals.append(f"{sd:.1f}" if pd.notna(sd) else "")
            else:
                w1_vals.append(f"{v1:.1f}" if pd.notna(v1) else "")
                w2_vals.append(f"{v2:.1f}" if pd.notna(v2) else "")
                diff_vals.append(f"{dv:.1f}" if pd.notna(dv) else "")
                sigma_w1_vals.append(f"{s1:.1f}" if pd.notna(s1) else "")
                sigma_w2_vals.append(f"{s2:.1f}" if pd.notna(s2) else "")
                sigma_diff_vals.append(f"{sd:.1f}" if pd.notna(sd) else "")
        diff_pct_vals = []
        diff_pct_colors = []
        for v in sub["Diff_Pct"].tolist():
            if pd.isna(v):
                diff_pct_vals.append("")
                diff_pct_colors.append(COLOR_DARK)
            else:
                sign = "+" if v > 0 else "-" if v < 0 else ""
                abs_pct = f"{abs(v)*100:.1f}%"
                text = f"{sign}{abs_pct}" if sign else abs_pct
                diff_pct_vals.append(text)
                diff_pct_colors.append("#2ECC71" if v > 0 else "#E74C3C" if v < 0 else "#7B848F")
        sigma_w1_colors = []
        sigma_w2_colors = []
        sigma_diff_colors = []
        for s1, s2, sd in zip(raw_sigma_w1, raw_sigma_w2, raw_sigma_diff):
            sigma_w1_colors.append("#E74C3C" if pd.notna(s1) and s1 < 0 else COLOR_DARK)
            sigma_w2_colors.append("#E74C3C" if pd.notna(s2) and s2 < 0 else COLOR_DARK)
            sigma_diff_colors.append("#E74C3C" if pd.notna(sd) and sd < 0 else COLOR_DARK)
        n_rows = len(labels)
        dark_column = [COLOR_DARK] * n_rows
        font_colors = [
            dark_column,
            dark_column,
            dark_column,
            dark_column,
            dark_column,
            diff_pct_colors,
            sigma_w1_colors,
            sigma_w2_colors,
            sigma_diff_colors,
        ]
        fig.add_table(
            header=dict(
                values=[
                    "指标",
                    "说明",
                    "W1 值",
                    "W2 值",
                    "差值(W2-W1)",
                    "份额变化",
                    "W1 σ",
                    "W2 σ",
                    "σ差(W2-W1)",
                ],
                fill_color="#f8f9fa",
                font=dict(color="#555555", size=12),
                align="center",
                height=30,
            ),
            cells=dict(
                values=[
                    metrics,
                    labels,
                    w1_vals,
                    w2_vals,
                    diff_vals,
                    diff_pct_vals,
                    sigma_w1_vals,
                    sigma_w2_vals,
                    sigma_diff_vals,
                ],
                align=[
                    "left",
                    "left",
                    "right",
                    "right",
                    "right",
                    "right",
                    "right",
                    "right",
                    "right",
                ],
                fill_color=COLOR_BG,
                font=dict(color=font_colors, size=11),
            ),
            row=idx,
            col=1,
        )
    rows_traffic = len(df_comparison[df_comparison["group"] == "traffic"])
    rows_conversion = len(df_comparison[df_comparison["group"] == "conversion"])
    rows_structure = len(df_comparison[df_comparison["group"] == "structure"])
    row_height = 42
    estimated_height = (
        max(rows_traffic, 1) * row_height
        + max(rows_conversion, 1) * row_height
        + max(rows_structure, 1) * row_height
        + 220
    )
    fig.update_layout(
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(bordercolor=COLOR_TEXT, borderwidth=1, font=dict(color=COLOR_TEXT)),
        margin=dict(l=40, r=40, t=80, b=40),
        height=estimated_height,
        title=f"多指标双时间窗对比（{start1}~{end1} vs {start2}~{end2}）",
    )
    return fig


def main():
    args = parse_args()
    assign_df = load_assign_data(ASSIGN_CSV)
    order_df = load_order_data(ORDER_PARQUET)
    bizdef = load_business_definition(BUSINESS_DEF_FILE)
    order_df = apply_business_logic(order_df, bizdef)
    w1_assign = filter_assign_window(assign_df, args.start1, args.end1)
    w2_assign = filter_assign_window(assign_df, args.start2, args.end2)
    m_assign_w1 = calc_assign_metrics(w1_assign, "W1")
    m_assign_w2 = calc_assign_metrics(w2_assign, "W2")
    m_order_w1 = calc_order_metrics_full(order_df, args.start1, args.end1, "W1")
    m_order_w2 = calc_order_metrics_full(order_df, args.start2, args.end2, "W2")
    metrics_all = pd.concat(
        [m_assign_w1, m_assign_w2, m_order_w1, m_order_w2], ignore_index=True
    )
    comp = build_comparison_table(metrics_all)
    fig = make_figure(comp, args.start1, args.end1, args.start2, args.end2)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


if __name__ == "__main__":
    main()
