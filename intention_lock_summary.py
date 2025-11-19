import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd


# Default input and output paths based on your dataset structure
DEFAULT_INPUT = Path(
    "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"
)
OUT_DIR = Path(
    "/Users/zihao_/Documents/coding/dataset/processed/analysis_results"
)


def normalize(name: str) -> str:
    """Normalize column names for case-insensitive, space/underscore-insensitive matching."""
    return name.strip().lower().replace("_", " ").replace("  ", " ")


def resolve_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Resolve a column name from a list of candidates with robust matching.

    Tries exact candidate first; falls back to normalized, case-insensitive comparison.
    """
    # Prefer exact matches first
    for cand in candidates:
        if cand in df.columns:
            return cand

    # Fallback: normalized comparison
    cand_norm = [normalize(c) for c in candidates]
    col_norm_map = {normalize(c): c for c in df.columns}
    for cn in cand_norm:
        if cn in col_norm_map:
            return col_norm_map[cn]

    raise KeyError(
        f"Unable to resolve column from candidates: {candidates}. Available columns: {list(df.columns)}"
    )


def df_to_md(df: pd.DataFrame) -> str:
    """Convert DataFrame to a GitHub-flavored Markdown table."""
    if df.empty:
        return "(空表，无数据)"
    # Ensure string columns for safety
    df_fmt = df.copy()
    return df_fmt.to_markdown(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="锁单数与分城市门店数/锁单数汇总"
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_parquet(input_path)

    # Resolve required columns
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
    city_col = resolve_column(
        df,
        [
            "Store City",
            "Store_City",
            "城市",
            "门店城市",
            "City",
        ],
    )
    store_col = resolve_column(
        df,
        [
            "Store Name",
            "Store_Name",
            "门店名称",
            "网点名称",
            "店名",
        ],
    )

    # Ensure datetime type for lock time
    df[lock_col] = pd.to_datetime(df[lock_col], errors="coerce")

    # Filter: lock time present and within date range (inclusive)
    mask = df[lock_col].notna() & (
        (df[lock_col].dt.date >= start_date) & (df[lock_col].dt.date <= end_date)
    )
    df_period = df.loc[mask, [lock_col, city_col, store_col]].copy()

    # Totals
    total_locks = int(len(df_period))

    # Per-city aggregation
    city_summary = (
        df_period.groupby(city_col)
        .agg(
            锁单数=(lock_col, "size"),
            门店数=(store_col, pd.Series.nunique),
        )
        .reset_index()
    )
    city_summary.rename(columns={city_col: "城市"}, inplace=True)
    city_summary.sort_values(by="锁单数", ascending=False, inplace=True)

    # Build markdown report
    md_lines = []
    md_lines.append("# 锁单汇总报告")
    md_lines.append("")
    md_lines.append(f"- 源文件: `{input_path}`")
    md_lines.append(f"- 时间区间: `{args.start}` ~ `{args.end}`")
    md_lines.append("")
    md_lines.append("## 区间总计")
    md_lines.append(f"- 锁单数（Lock_Time 在区间且非空）: `{total_locks}`")
    md_lines.append("")
    md_lines.append("## 分城市汇总")
    md_lines.append("(门店数为区间内 countd(Store Name))")
    md_lines.append(df_to_md(city_summary))

    report_name = f"intention_lock_summary_{args.start}_to_{args.end}.md"
    report_path = OUT_DIR / report_name
    report_path.write_text("\n".join(md_lines), encoding="utf-8")

    # Console preview
    print(f"Report saved: {report_path}")
    print("\nTop 10 cities by 锁单数:")
    print(df_to_md(city_summary.head(10)))


if __name__ == "__main__":
    main()