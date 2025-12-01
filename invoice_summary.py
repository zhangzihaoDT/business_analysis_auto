import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_INPUT = Path(
    "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"
)
OUT_DIR = Path(
    "/Users/zihao_/Documents/coding/dataset/processed/analysis_results"
)


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


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(空表，无数据)"
    df_fmt = df.copy()
    return df_fmt.to_markdown(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="发票上传订单数（分车型）汇总"
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
        help="输入 Parquet 文件路径（默认为 intention_order_analysis.parquet）",
    )
    parser.add_argument(
        "--models",
        help="逗号分隔的车型列表，仅显示这些车型列，例如: 'CM2,CM2 增程,LS9'",
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

    df = pd.read_parquet(input_path)

    invoice_col = resolve_column(
        df,
        [
            "Invoice_Upload_Time",
            "Invoice Upload Time",
            "invoice_upload_time",
            "InvoiceUploadTime",
            "Invoice Time",
            "开票上传时间",
            "发票上传时间",
            "开票时间",
        ],
    )
    model_group_col = resolve_column(
        df,
        [
            "车型分组",
            "Model Group",
            "Vehicle Group",
            "Car Group",
            "车型",
        ],
    )
    product_name_col = resolve_column(
        df,
        [
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

    df[invoice_col] = pd.to_datetime(df[invoice_col], errors="coerce")

    df["车型分类"] = df[model_group_col].astype(str)
    _cm2_all = df[model_group_col].astype(str).str.upper() == "CM2"
    _is_range_ext_all = df[product_name_col].astype(str).str.contains(r"52|66", case=False, na=False)
    df.loc[_cm2_all & _is_range_ext_all, "车型分类"] = "CM2 增程"
    df.loc[_cm2_all & ~_is_range_ext_all, "车型分类"] = "CM2"

    mask = df[invoice_col].notna() & (
        (df[invoice_col].dt.date >= start_date) & (df[invoice_col].dt.date <= end_date)
    )
    wanted_models = [m.strip() for m in str(args.models).split(",") if m.strip()] if args.models else []
    model_filter = df["车型分类"].isin(wanted_models) if wanted_models else pd.Series(True, index=df.index)

    invoice_total = int(df.loc[mask & model_filter, order_no_col].nunique())

    grouped_df = (
        df.loc[mask & model_filter, ["车型分类", order_no_col]]
        .groupby(["车型分类"])
        .agg(订单数=(order_no_col, pd.Series.nunique))
        .reset_index()
        .sort_values("车型分类")
    )

    if wanted_models:
        cols = [m for m in wanted_models if m in grouped_df["车型分类"].tolist()]
        if cols:
            grouped_df = grouped_df[grouped_df["车型分类"].isin(cols)]

    total_by_models = grouped_df["订单数"].sum()
    percent_df = grouped_df.copy()
    percent_df["占比%"] = (percent_df["订单数"] / max(total_by_models, 1) * 100).round(2)

    summary_df = pd.DataFrame({"指标": ["发票上传订单数"], "数量": [invoice_total]})

    md_lines = []
    md_lines.append("# 发票上传订单数汇总")
    md_lines.append("")
    md_lines.append(f"- 源文件: `{input_path}`")
    md_lines.append(f"- 时间区间: `{args.start}` ~ `{args.end}`")
    md_lines.append("")
    md_lines.append("## 概览统计")
    md_lines.append(df_to_md(summary_df))
    md_lines.append("")
    md_lines.append("## 分车型订单数")
    md_lines.append(df_to_md(grouped_df))
    md_lines.append("")
    md_lines.append("## 分车型占比（%）")
    md_lines.append(df_to_md(percent_df))

    report_name = f"invoice_summary_{args.start}_to_{args.end}.md"
    report_path = OUT_DIR / report_name
    report_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Report saved: {report_path}")
    print("\n预览：")
    print(df_to_md(grouped_df))
    print("\n占比预览：")
    print(df_to_md(percent_df))


if __name__ == "__main__":
    main()

