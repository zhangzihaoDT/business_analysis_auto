import argparse
from datetime import datetime
import html
import re
from pathlib import Path

import pandas as pd


def read_mix_csv(file_path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "utf-16", "utf-16le", "utf-16be", "gbk"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, sep="\t", encoding=encoding)
            if len(df.columns) == 1:
                df = pd.read_csv(file_path, encoding=encoding)
            return df
        except Exception as e:
            last_error = e

    raise RuntimeError(f"读取失败: {file_path} (已尝试编码: {encodings})\n原始错误: {last_error}")


def parse_month(value: str) -> pd.Timestamp:
    s = str(value).strip()
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", s)
    if not m:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            raise ValueError(f"无法解析月份: {value!r}")
        return dt.to_period("M").to_timestamp()

    year = int(m.group(1))
    month = int(m.group(2))
    return pd.Timestamp(year=year, month=month, day=1)


def month_label(month_ts: pd.Timestamp) -> str:
    return f"{month_ts.year} 年 {month_ts.month} 月"


def build_monthly_sales_wide_table(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    brand_col = "品牌"
    submodel_col = "子车型"
    date_col = "日期 年/月"
    sales_col = "销量"

    missing = [c for c in [brand_col, submodel_col, date_col, sales_col] if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必要列: {missing}，当前列: {list(df.columns)}")

    df[date_col] = df[date_col].astype(str)
    df["month"] = df[date_col].map(parse_month)
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce").fillna(0).astype(int)

    if start:
        start_ts = pd.Period(start, freq="M").to_timestamp()
        df = df[df["month"] >= start_ts]
    if end:
        end_ts = pd.Period(end, freq="M").to_timestamp()
        df = df[df["month"] <= end_ts]

    if df.empty:
        return pd.DataFrame(columns=["品牌", "子车型", "月均销量", "近 3 月CV（std/mean）"])

    agg = df.groupby([brand_col, submodel_col, "month"], as_index=False)[sales_col].sum()
    wide = agg.pivot(index=[brand_col, submodel_col], columns="month", values=sales_col).reset_index()
    wide.columns.name = None

    month_ts_cols = [c for c in wide.columns if isinstance(c, pd.Timestamp)]
    month_ts_cols = sorted(month_ts_cols)
    month_name_map = {c: f"{month_label(c)}销量" for c in month_ts_cols}
    wide = wide.rename(columns=month_name_map)

    month_sales_cols = [month_name_map[c] for c in month_ts_cols]
    wide["月均销量"] = wide[month_sales_cols].mean(axis=1, skipna=True)

    target_month_ts = month_ts_cols[-1]
    if end:
        target_month_ts = pd.Period(end, freq="M").to_timestamp()
    target_month_ts = pd.Timestamp(year=target_month_ts.year, month=target_month_ts.month, day=1)
    prev_month_ts = (target_month_ts.to_period("M") - 1).to_timestamp()
    prev2_month_ts = (target_month_ts.to_period("M") - 2).to_timestamp()

    target_col = f"{month_label(target_month_ts)}销量"
    prev_col = f"{month_label(prev_month_ts)}销量"
    prev2_col = f"{month_label(prev2_month_ts)}销量"

    if target_col not in wide.columns:
        wide[target_col] = pd.NA
    if prev_col not in wide.columns:
        wide[prev_col] = pd.NA
    if prev2_col not in wide.columns:
        wide[prev2_col] = pd.NA

    growth_col = f"{month_label(target_month_ts)}销量环比（环比前一个月，即 {month_label(prev_month_ts)}）"
    target_vals = pd.to_numeric(wide[target_col], errors="coerce")
    prev_vals = pd.to_numeric(wide[prev_col], errors="coerce")
    growth_pct = (target_vals - prev_vals) / prev_vals * 100
    growth_pct = growth_pct.mask(prev_vals.fillna(0) <= 0)
    wide[growth_col] = growth_pct

    last3_cols = [prev2_col, prev_col, target_col]
    last3_std = wide[last3_cols].std(axis=1, skipna=True)
    last3_mean = wide[last3_cols].mean(axis=1, skipna=True)
    last3_cv = last3_std / last3_mean
    last3_cv = last3_cv.mask(last3_mean.fillna(0) <= 0)
    wide["近 3 月CV（std/mean）"] = last3_cv

    month_sales_cols_ordered = [month_name_map[c] for c in month_ts_cols if month_name_map[c] in wide.columns]
    if target_col in wide.columns and target_col not in month_sales_cols_ordered:
        month_sales_cols_ordered.append(target_col)

    wide = wide.rename(columns={brand_col: "品牌", submodel_col: "子车型"})
    base_cols = ["品牌", "子车型"]
    tail_cols = ["月均销量", growth_col, "近 3 月CV（std/mean）"]
    wide = wide[base_cols + month_sales_cols_ordered + tail_cols]
    wide = wide.sort_values(["品牌", "子车型"], ascending=[True, True])
    return wide


def format_wide_for_html(wide: pd.DataFrame) -> pd.DataFrame:
    w = wide.copy()
    for c in w.columns:
        if c.endswith("销量") and c not in {"月均销量"} and "环比" not in c:
            w[c] = pd.to_numeric(w[c], errors="coerce").round(0).astype("Int64")

    if "月均销量" in w.columns:
        w["月均销量"] = pd.to_numeric(w["月均销量"], errors="coerce").map(
            lambda x: "" if pd.isna(x) else f"{x:.0f}"
        )

    for c in w.columns:
        if "CV" in str(c):
            w[c] = pd.to_numeric(w[c], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x * 100:.0f}%")

    for c in w.columns:
        if "销量环比" in c:
            w[c] = pd.to_numeric(w[c], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.0f}%")

    return w


def build_city_tier_share_change_table(
    df: pd.DataFrame,
    submodels: list[str],
    month_a: str,
    month_b: str,
) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    brand_col = "品牌"
    submodel_col = "子车型"
    city_tier_col = "25年城市级别"
    date_col = "日期 年/月"
    sales_col = "销量"

    missing = [c for c in [brand_col, submodel_col, city_tier_col, date_col, sales_col] if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必要列: {missing}，当前列: {list(df.columns)}")

    df[date_col] = df[date_col].astype(str)
    df["month"] = df[date_col].map(parse_month)
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce").fillna(0).astype(int)

    month_a_ts = pd.Period(month_a, freq="M").to_timestamp()
    month_b_ts = pd.Period(month_b, freq="M").to_timestamp()

    sdf = df[df[submodel_col].isin(submodels)].copy()
    sdf = sdf[sdf["month"].isin([month_a_ts, month_b_ts])]

    if sdf.empty:
        month_a_label = month_label(month_a_ts)
        month_b_label = month_label(month_b_ts)
        return pd.DataFrame(
            columns=[
                "品牌",
                "子车型",
                "25年城市级别",
                f"{month_a_label}销量",
                f"{month_a_label}占比",
                f"{month_b_label}销量",
                f"{month_b_label}占比",
                "占比变化(pp)",
            ]
        )

    agg = (
        sdf.groupby([brand_col, submodel_col, city_tier_col, "month"], as_index=False)[sales_col]
        .sum()
        .rename(columns={sales_col: "销量"})
    )
    totals = agg.groupby([brand_col, submodel_col, "month"], as_index=False)["销量"].sum().rename(
        columns={"销量": "总销量"}
    )
    agg = agg.merge(totals, on=[brand_col, submodel_col, "month"], how="left")
    agg["占比"] = agg["销量"] / agg["总销量"]
    agg.loc[agg["总销量"].fillna(0) <= 0, "占比"] = pd.NA

    a = agg[agg["month"] == month_a_ts].copy()
    b = agg[agg["month"] == month_b_ts].copy()

    month_a_label = month_label(month_a_ts)
    month_b_label = month_label(month_b_ts)

    a = a.rename(
        columns={
            "销量": f"{month_a_label}销量",
            "占比": f"{month_a_label}占比",
        }
    ).drop(columns=["month", "总销量"])
    b = b.rename(
        columns={
            "销量": f"{month_b_label}销量",
            "占比": f"{month_b_label}占比",
        }
    ).drop(columns=["month", "总销量"])

    out = a.merge(b, on=[brand_col, submodel_col, city_tier_col], how="outer")
    out["占比变化(pp)"] = out[f"{month_b_label}占比"] - out[f"{month_a_label}占比"]

    tier_order = ["一线", "新一线", "二线", "三线", "四五线"]
    out[city_tier_col] = pd.Categorical(out[city_tier_col], categories=tier_order, ordered=True)
    out = out.sort_values([brand_col, submodel_col, city_tier_col], ascending=[True, True, True])

    out = out.rename(columns={brand_col: "品牌", submodel_col: "子车型", city_tier_col: "25年城市级别"})
    out = out[
        [
            "品牌",
            "子车型",
            "25年城市级别",
            f"{month_a_label}销量",
            f"{month_a_label}占比",
            f"{month_b_label}销量",
            f"{month_b_label}占比",
            "占比变化(pp)",
        ]
    ]
    return out


def format_city_tier_table_for_html(city_tier: pd.DataFrame) -> pd.DataFrame:
    df = city_tier.copy()

    for c in df.columns:
        if str(c).endswith("销量"):
            df[c] = pd.to_numeric(df[c], errors="coerce").round(0).astype("Int64")

    for c in df.columns:
        if str(c).endswith("占比"):
            df[c] = pd.to_numeric(df[c], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x * 100:.0f}%")

    if "占比变化(pp)" in df.columns:
        df["占比变化(pp)"] = pd.to_numeric(df["占比变化(pp)"], errors="coerce").map(
            lambda x: "" if pd.isna(x) else f"{x * 100:+.0f}pp"
        )

    return df


def build_region_conversion_drop_table(df: pd.DataFrame, target_month: str = "2026-03") -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    region_col = "lc_region_name"
    date_col = "lc_assign_time_min 年/月"
    conv_col = "[30 日内锁单线索数]/[下发线索数]"

    missing = [c for c in [region_col, date_col, conv_col] if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必要列: {missing}，当前列: {list(df.columns)}")

    df[date_col] = df[date_col].astype(str)
    df["month"] = df[date_col].map(parse_month)
    df["conv30"] = pd.to_numeric(df[conv_col], errors="coerce")

    target_ts = pd.Period(target_month, freq="M").to_timestamp()
    target_label = month_label(target_ts)

    g = df.groupby([region_col, "month"], as_index=False)["conv30"].mean()
    max_conv = g.groupby(region_col, as_index=False)["conv30"].max().rename(columns={"conv30": "历史最高月转化率"})
    target_conv = g[g["month"] == target_ts][[region_col, "conv30"]].rename(columns={"conv30": f"{target_label}转化率"})

    out = max_conv.merge(target_conv, on=region_col, how="left")
    max_vals = pd.to_numeric(out["历史最高月转化率"], errors="coerce")
    tgt_vals = pd.to_numeric(out.get(f"{target_label}转化率"), errors="coerce")
    rel_drop = (max_vals - tgt_vals) / max_vals
    rel_drop = rel_drop.mask(max_vals.fillna(0) <= 0)
    out["下降幅度(%)"] = rel_drop

    out = out.rename(columns={region_col: "Ic_region_name"})
    out = out[["Ic_region_name", f"{target_label}转化率", "历史最高月转化率", "下降幅度(%)"]]
    out = out.sort_values("下降幅度(%)", ascending=True)
    return out


def format_region_drop_for_html(reg: pd.DataFrame) -> pd.DataFrame:
    df = reg.copy()
    for c in df.columns:
        if "转化率" in str(c):
            df[c] = pd.to_numeric(df[c], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x * 100:.1f}%")
    if "下降幅度(%)" in df.columns:
        df["下降幅度(%)"] = pd.to_numeric(df["下降幅度(%)"], errors="coerce").map(
            lambda x: "" if pd.isna(x) else f"{x * 100:.1f}%"
        )
    return df


def build_price_segment_top10_tables(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    bucket_col = "2026 年 2 月分车型 TP重心 (数据桶)"
    fuel_group_col = "燃料类型 (组) 1"
    body_col = "车身形式"
    brand_col = "品牌"
    model_col = "车型"
    submodel_col = "子车型"
    month_col = "日期 年/月"
    metric_col = "度量名称"
    value_col = "度量值"

    missing = [
        c
        for c in [
            bucket_col,
            fuel_group_col,
            body_col,
            brand_col,
            model_col,
            submodel_col,
            month_col,
            metric_col,
            value_col,
        ]
        if c not in df.columns
    ]
    if missing:
        raise KeyError(f"缺少必要列: {missing}，当前列: {list(df.columns)}")

    def normalize_group_value(value: object) -> str:
        s = "" if value is None else str(value)
        s = s.replace("\ufeff", "").replace("\u200b", "").replace("\u00a0", " ").replace("\u3000", " ")
        for dash in ["–", "—", "−", "－"]:
            s = s.replace(dash, "-")
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"\s*-\s*", "-", s)
        s = s.replace(" 万", "万")
        return s

    def map_bucket_to_price_band(value: object) -> str:
        s = normalize_group_value(value)
        m = re.match(r"^(\d+(?:\.\d+)?)\s*[Kk]\s*$", s)
        if not m:
            s2 = s.replace("~", "～").replace("-", "～")
            if s2.endswith("万以上"):
                return "50万以及以上" if s2.startswith("50") else s2
            if s2 == "10万以下":
                return s2
            if re.match(r"^\d+～\d+\s*万$", s2):
                return s2.replace(" ", "")
            return s

        k = float(m.group(1))
        if k < 100:
            return "10万以下"
        if k < 150:
            return "10～15万"
        if k < 200:
            return "15～20万"
        if k < 250:
            return "20～25万"
        if k < 300:
            return "25～30万"
        if k < 350:
            return "30～35万"
        if k < 400:
            return "35～40万"
        if k < 450:
            return "40～45万"
        if k < 500:
            return "45～50万"
        return "50万以及以上"

    for c in [
        bucket_col,
        fuel_group_col,
        body_col,
        brand_col,
        model_col,
        submodel_col,
        month_col,
    ]:
        df[c] = df[c].map(normalize_group_value)

    price_group_col = "TP价格段"
    df[price_group_col] = df[bucket_col].map(map_bucket_to_price_band)

    df[metric_col] = df[metric_col].astype(str).str.strip()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # base_keys 不再直接使用，保留注释以便理解原始维度

    tp_metric_candidates = {
        "2026年2月分车型TP重心",
        "2026年2月分车型 TP重心",
        "2026 年 2 月分车型TP重心",
        "2026 年 2 月分车型 TP重心",
        "TP重心",
    }

    # 销量按分组求和
    sales = (
        df[df[metric_col] == "销量"]
        .groupby([price_group_col, fuel_group_col, body_col, brand_col, model_col, submodel_col, month_col], as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "销量"})
    )
    # TP重心按车型/子车型取唯一值（优先选“2026年2月分车型TP重心”命名），避免重复均值导致偏差
    tp_raw = df[df[metric_col].isin(tp_metric_candidates)].copy()
    tp_raw["metric_priority"] = tp_raw[metric_col].map(
        lambda m: 0 if "2026" in m and "TP重心" in m else 1
    )
    tp_sorted = tp_raw.sort_values(["metric_priority"])
    tp_unique = (
        tp_sorted.groupby([brand_col, submodel_col], as_index=False)[value_col]
        .first()
        .rename(columns={value_col: "成交价格"})
    )

    merged = sales.merge(tp_unique, on=[brand_col, submodel_col], how="left")
    merged["销量"] = pd.to_numeric(merged["销量"], errors="coerce").fillna(0)
    merged = merged[merged[body_col].isin(["轿车", "SUV", "MPV"])].copy()

    merged = (
        merged.groupby([price_group_col, fuel_group_col, body_col, brand_col, submodel_col], as_index=False)
        .agg(销量=("销量", "sum"), 成交价格=("成交价格", "first"))
        .rename(columns={submodel_col: "子车型", brand_col: "品牌"})
    )
    merged = merged[merged["销量"] > 0].copy()

    body_order = {"轿车": 0, "SUV": 1, "MPV": 2}
    merged["_body_order"] = merged[body_col].map(body_order)

    price_order = {
        "10万以下": 0,
        "10～15万": 1,
        "15～20万": 2,
        "20～25万": 3,
        "25～30万": 4,
        "30～35万": 5,
        "35～40万": 6,
        "40～45万": 7,
        "45～50万": 8,
        "50万以及以上": 9,
    }

    groups = (
        merged[[price_group_col, fuel_group_col, body_col]]
        .drop_duplicates()
        .assign(_price_order=lambda x: x[price_group_col].map(price_order).fillna(9999))
        .assign(_body_order=lambda x: x[body_col].map(body_order))
        .sort_values(["_price_order", price_group_col, fuel_group_col, "_body_order"])
        .drop(columns=["_price_order", "_body_order"])
    )

    tables: list[tuple[str, pd.DataFrame]] = []
    seen_titles: set[str] = set()
    for _, g in groups.iterrows():
        a = g[price_group_col]
        b = g[fuel_group_col]
        c = g[body_col]

        sdf = merged[
            (merged[price_group_col] == a) & (merged[fuel_group_col] == b) & (merged[body_col] == c)
        ].copy()
        sdf = sdf.sort_values("销量", ascending=False).head(10)

        out = sdf.rename(
            columns={
                price_group_col: "TP 5万1档（组）",
                fuel_group_col: "燃料类型（组）1",
                body_col: "车身形式",
            }
        )[
            [
                "子车型",
                "销量",
                "成交价格",
                "品牌",
            ]
        ]
        out = out[["子车型", "品牌", "销量", "成交价格"]]

        title = f"{a} | {b} | {c}"
        if title in seen_titles:
            continue
        seen_titles.add(title)
        tables.append((title, out))

    return tables


def format_price_segment_table_for_html(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "销量" in out.columns:
        out["销量"] = pd.to_numeric(out["销量"], errors="coerce").round(0).astype("Int64")
    if "成交价格" in out.columns:
        out["成交价格"] = pd.to_numeric(out["成交价格"], errors="coerce").round(0).astype("Int64")
    return out


def build_html_report(
    ls6_wide: pd.DataFrame,
    ls9_wide: pd.DataFrame,
    city_tier: pd.DataFrame,
    city_tier_ls9: pd.DataFrame,
    region_drop: pd.DataFrame,
    ls6_input_path: Path,
    ls9_input_path: Path,
    city_tier_input_path: Path,
    region_input_path: Path,
    region_history_range: str,
    price_segment_tables: list[tuple[str, pd.DataFrame]],
    price_input_path: Path,
    start: str | None,
    end: str | None,
) -> str:
    ls6_wide_f = format_wide_for_html(ls6_wide)
    ls9_wide_f = format_wide_for_html(ls9_wide)
    city_tier_f = format_city_tier_table_for_html(city_tier)
    city_tier_ls9_f = format_city_tier_table_for_html(city_tier_ls9)
    region_drop_f = format_region_drop_for_html(region_drop)
    range_label = f"{start or ''} ~ {end or ''}".strip() if (start or end) else ""

    def render_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        col_count = max(len(cols), 1)
        col_width_pct = 100.0 / col_count
        colgroup = "<colgroup>" + "".join(f'<col style="width:{col_width_pct:.6f}%">' for _ in cols) + "</colgroup>"

        thead = "<thead><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in cols) + "</tr></thead>"
        rows_html = []

        for _, row in df.iterrows():
            brand = str(row.get("品牌", "")).strip()
            tr_class = ' class="highlight-im"' if brand == "智己" else ""
            tds = "".join(f"<td>{html.escape('' if pd.isna(v) else str(v))}</td>" for v in row.tolist())
            rows_html.append(f"<tr{tr_class}>{tds}</tr>")

        tbody = "<tbody>" + "".join(rows_html) + "</tbody>"
        return f'<table class="dataframe">{colgroup}{thead}{tbody}</table>'

    body_order = ["轿车", "SUV", "MPV"]
    grouped_price: dict[tuple[str, str], dict[str, tuple[str, pd.DataFrame]]] = {}
    group_order: list[tuple[str, str]] = []

    for title, tdf in price_segment_tables:
        parts = [p.strip() for p in title.split("|")]
        if len(parts) != 3:
            continue
        price, fuel, body = parts
        key = (price, fuel)
        if key not in grouped_price:
            grouped_price[key] = {}
            group_order.append(key)
        grouped_price[key][body] = (title, tdf)

    price_rows_html = []
    for key in group_order:
        items = grouped_price.get(key, {})
        cells = []
        for body in body_order:
            if body in items:
                title, tdf = items[body]
                table_html = render_table(format_price_segment_table_for_html(tdf))
                cells.append(f'<div class="price-item"><h3>{html.escape(title)}</h3>{table_html}</div>')
            else:
                cells.append('<div class="price-item empty"></div>')
        price_rows_html.append(f"<div class=\"price-row\">{''.join(cells)}</div>")

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; margin: 24px; color: #111; }
    h1 { font-size: 20px; margin: 0 0 8px 0; }
    .meta { color: #555; font-size: 12px; margin-bottom: 16px; }
    .section { margin-top: 18px; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; table-layout: fixed; }
    th, td { border: 1px solid #e5e7eb; padding: 6px 8px; vertical-align: top; }
    th { background: #f8fafc; text-align: left; position: sticky; top: 0; }
    tr:nth-child(even) td { background: #fcfcfd; }
    tr.highlight-im td { background: #D67D58 !important; font-weight: 700; color: #ffffff; }
    td, th { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .price-rows { display: flex; flex-direction: column; gap: 16px; }
    .price-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
    .price-item h3 { font-size: 14px; margin: 0 0 8px 0; }
    .price-item.empty { visibility: hidden; }
    .container { max-width: 1600px; }
    """

    ls6_html = render_table(ls6_wide_f)
    ls9_html = render_table(ls9_wide_f)
    city_tier_html = render_table(city_tier_f)
    city_tier_ls9_html = render_table(city_tier_ls9_f)
    region_drop_html = render_table(region_drop_f)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""
    <html>
    <head>
      <meta charset="utf-8">
      <title>MIX 销量分析</title>
      <style>{css}</style>
    </head>
    <body>
      <div class="container">
        <h1>MIX 销量分析：月度销量 & 波动率</h1>
        <div class="meta">
          <div>时间范围：{range_label}</div>
          <div>生成时间：{now}</div>
        </div>

        <div class="section">
          <h2>LS6 主要竞对车型销量观察</h2>
          <div class="meta">数据源：{ls6_input_path}</div>
          {ls6_html}
        </div>

        <div class="section">
          <h2>LS9 主要竞对车型销量观察</h2>
          <div class="meta">数据源：{ls9_input_path}</div>
          {ls9_html}
        </div>

        <div class="section">
          <h2>LS6 主要竞对车型分城市线级销量变化</h2>
          <div class="meta">数据源：{city_tier_input_path}</div>
          {city_tier_html}
        </div>

        <div class="section">
          <h2>LS9 主要竞对车型分城市线级销量变化</h2>
          <div class="meta">数据源：{city_tier_input_path}</div>
          {city_tier_ls9_html}
        </div>

        <div class="section">
          <h2>分区域下发线索30日转化率变化幅度（2026-03 vs 历史最高）</h2>
          <div class="meta">数据源：{region_input_path}</div>
          <div class="meta">历史时间范围：{region_history_range}</div>
          {region_drop_html}
        </div>

        <div class="section">
          <h2>分价格段量价（各分组TOP10）</h2>
          <div class="meta">数据源：{price_input_path}</div>
          <div class="price-rows">
            {''.join(price_rows_html)}
          </div>
        </div>
      </div>
    </body>
    </html>
    """


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_ls6_input = project_root / "original" / "MIX销量_data_LS6.csv"
    default_ls9_input = project_root / "original" / "MIX销量_LS9_data.csv"
    default_city_tier_input = project_root / "original" / "区域销量_data.csv"
    default_region_input = project_root / "original" / "leads_assign_region_data.csv"
    default_price_input = project_root / "original" / "分价格段量价_data_26 年2月.csv"
    default_output_dir = Path(__file__).resolve().parent / "reports"
    default_output = default_output_dir / "analyze_mix.html"

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(default_ls6_input))
    ap.add_argument("--ls6-input", type=str, default=None)
    ap.add_argument("--ls9-input", type=str, default=str(default_ls9_input))
    ap.add_argument("--city-tier-input", type=str, default=str(default_city_tier_input))
    ap.add_argument("--region-input", type=str, default=str(default_region_input))
    ap.add_argument("--price-input", type=str, default=str(default_price_input))
    ap.add_argument("--output", type=str, default=str(default_output))
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    args = ap.parse_args()

    ls6_input_path = Path(args.ls6_input) if args.ls6_input else Path(args.input)
    ls9_input_path = Path(args.ls9_input)
    city_tier_input_path = Path(args.city_tier_input)
    output_path = Path(args.output)

    ls6_df = read_mix_csv(ls6_input_path)
    ls6_wide = build_monthly_sales_wide_table(ls6_df, start=args.start, end=args.end)

    ls9_df = read_mix_csv(ls9_input_path)
    ls9_wide = build_monthly_sales_wide_table(ls9_df, start=args.start, end=args.end)

    city_df = read_mix_csv(city_tier_input_path)
    city_tier_submodels = [
        "智己LS6 EV",
        "智己LS6 REEV",
        "极氪7X",
        "理想i6",
        "理想L6",
        "阿维塔07 EV",
        "阿维塔07 REEV",
    ]
    city_tier = build_city_tier_share_change_table(
        city_df,
        submodels=city_tier_submodels,
        month_a="2026-01",
        month_b="2026-02",
    )

    city_tier_ls9_submodels = [
        "问界M9 EV",
        "问界M9 REEV",
        "智己LS9",
        "极氪9X",
        "腾势N8L",
        "领克900",
    ]
    city_tier_ls9 = build_city_tier_share_change_table(
        city_df,
        submodels=city_tier_ls9_submodels,
        month_a="2026-01",
        month_b="2026-02",
    )

    region_input_path = Path(args.region_input)
    region_df = read_mix_csv(region_input_path)
    region_drop = build_region_conversion_drop_table(region_df, target_month="2026-03")
    # 历史时间范围说明
    region_df["month"] = region_df["lc_assign_time_min 年/月"].astype(str).map(parse_month)
    if not region_df["month"].isna().all():
        min_m = pd.to_datetime(region_df["month"]).min()
        max_m = pd.to_datetime(region_df["month"]).max()
        region_history_range = f"{month_label(pd.Timestamp(min_m))} ~ {month_label(pd.Timestamp(max_m))}"
    else:
        region_history_range = ""

    price_input_path = Path(args.price_input)
    price_df = read_mix_csv(price_input_path)
    price_segment_tables = build_price_segment_top10_tables(price_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = build_html_report(
        ls6_wide=ls6_wide,
        ls9_wide=ls9_wide,
        city_tier=city_tier,
        city_tier_ls9=city_tier_ls9,
        region_drop=region_drop,
        ls6_input_path=ls6_input_path,
        ls9_input_path=ls9_input_path,
        city_tier_input_path=city_tier_input_path,
        region_input_path=region_input_path,
        region_history_range=region_history_range,
        price_segment_tables=price_segment_tables,
        price_input_path=price_input_path,
        start=args.start,
        end=args.end,
    )
    output_path.write_text(html, encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
