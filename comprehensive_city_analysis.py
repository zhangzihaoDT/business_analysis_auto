#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
from pathlib import Path
from typing import List, Tuple

import pandas as pd


BASE = Path("/Users/zihao_/Documents/coding/dataset")
SCRIPTS_DIR = BASE / "scripts"
OUT_DIR = BASE / "processed" / "analysis_results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="综合分析：线索识别数 vs 锁单数 vs 门店数（分城市）")
    p.add_argument("--start", required=True, help="开始日期 YYYY-MM-DD")
    p.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD")
    # 可选：第二时间段，用于对比
    p.add_argument("--start2", required=False, help="第二时间段开始 YYYY-MM-DD")
    p.add_argument("--end2", required=False, help="第二时间段结束 YYYY-MM-DD")
    p.add_argument(
        "--out-csv",
        required=False,
        help="输出城市比值差异 CSV 路径（默认写入 processed/analysis_results/）",
    )
    return p.parse_args()


def run_script(path: Path, start: str, end: str) -> None:
    cmd = ["python3", str(path), "--start", start, "--end", end]
    subprocess.run(cmd, check=True)


def read_file(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def extract_table_lines(lines: List[str], section_prefix: str) -> List[str]:
    # Find the section header
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith(section_prefix):
            start_idx = i
            break
    if start_idx is None:
        return []

    # From next lines, find the first table start (line starting with '|')
    table_start = None
    for j in range(start_idx + 1, len(lines)):
        if lines[j].strip().startswith("|"):
            table_start = j
            break
    if table_start is None:
        return []

    # Collect table lines until a non-table line appears
    table_lines = []
    for k in range(table_start, len(lines)):
        s = lines[k].strip()
        if s.startswith("|"):
            table_lines.append(s)
        else:
            break
    return table_lines


def parse_md_table(table_lines: List[str]) -> pd.DataFrame:
    if not table_lines:
        return pd.DataFrame()
    # First line: headers, second line: separator, rest: rows
    header_line = table_lines[0]
    sep_line = table_lines[1] if len(table_lines) > 1 else ""
    data_lines = table_lines[2:]

    def split_row(s: str) -> List[str]:
        parts = [p.strip() for p in s.strip().strip("|").split("|")]
        return parts

    headers = split_row(header_line)
    # Normalize headers mapping common variants
    hdr_map = {}
    for h in headers:
        key = h.strip()
        if key in ("线索识别数(sum)", "线索识别数"):
            hdr_map[key] = "线索识别数"
        elif key in ("下发门店数(max)", "下发门店数"):
            hdr_map[key] = "下发门店数"
        else:
            hdr_map[key] = key

    rows = [split_row(dl) for dl in data_lines]
    df = pd.DataFrame(rows, columns=headers)
    df = df.rename(columns=hdr_map)

    # Coerce numeric columns if present
    for col in ["锁单数", "门店数", "线索识别数", "下发门店数"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_report(start: str, end: str) -> Path:
    # Ensure output dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run dependent scripts so their reports exist
    leads_script = SCRIPTS_DIR / "leads_assign_summary.py"
    lock_script = SCRIPTS_DIR / "intention_lock_summary.py"
    run_script(leads_script, start, end)
    run_script(lock_script, start, end)

    # Read their markdown reports
    leads_md = OUT_DIR / f"leads_assign_summary_{start}_to_{end}.md"
    lock_md = OUT_DIR / f"intention_lock_summary_{start}_to_{end}.md"
    leads_lines = read_file(leads_md)
    lock_lines = read_file(lock_md)

    # Extract and parse per-city tables
    leads_table_lines = extract_table_lines(leads_lines, "## 分城市汇总")
    lock_table_lines = extract_table_lines(lock_lines, "## 分城市汇总")
    leads_df = parse_md_table(leads_table_lines)
    lock_df = parse_md_table(lock_table_lines)

    # Select relevant columns
    leads_sel = leads_df[["城市", "区域", "线索识别数"]].copy()
    lock_sel = lock_df[["城市", "锁单数", "门店数"]].copy()

    # Merge
    merged = pd.merge(lock_sel, leads_sel, on="城市", how="outer")
    # Compute ratio (handle zero/NaN)
    merged["线索识别数"].fillna(0, inplace=True)
    merged["锁单数"].fillna(0, inplace=True)
    def ratio(row):
        lead = row["线索识别数"]
        lock = row["锁单数"]
        return (lock / lead) if (pd.notna(lead) and lead > 0) else 0.0
    merged["锁单数/线索识别数比值"] = merged.apply(ratio, axis=1)

    # Sort by 锁单数 desc (secondary by 线索识别数 desc)
    merged.sort_values(by=["锁单数", "线索识别数"], ascending=[False, False], inplace=True)

    # Format ratio
    merged["锁单数/线索识别数比值"] = merged["锁单数/线索识别数比值"].round(4)

    # Build markdown
    md_lines = []
    md_lines.append("# 分城市综合分析报告")
    md_lines.append("")
    md_lines.append(f"- 时间区间: `{start}` ~ `{end}`")
    md_lines.append("- 来源报告: leads_assign_summary 与 intention_lock_summary")
    md_lines.append("")
    md_lines.append("## 分城市综合对比（按锁单数降序）")
    md_lines.append(merged.to_markdown(index=False))

    out_path = OUT_DIR / f"comprehensive_city_analysis_{start}_to_{end}.md"
    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    return out_path


def build_comparison_csv(start1: str, end1: str, start2: str, end2: str, out_csv: Path | None = None) -> Path:
    """生成双时间段的城市比值差异（后期-前期）CSV，包括锁单数与线索识别数，便于校验。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 确保依赖报告存在：分别跑两个时间段
    leads_script = SCRIPTS_DIR / "leads_assign_summary.py"
    lock_script = SCRIPTS_DIR / "intention_lock_summary.py"
    run_script(leads_script, start1, end1)
    run_script(lock_script, start1, end1)
    run_script(leads_script, start2, end2)
    run_script(lock_script, start2, end2)

    # 读取两段 Markdown 报告
    leads_md_1 = OUT_DIR / f"leads_assign_summary_{start1}_to_{end1}.md"
    lock_md_1 = OUT_DIR / f"intention_lock_summary_{start1}_to_{end1}.md"
    leads_md_2 = OUT_DIR / f"leads_assign_summary_{start2}_to_{end2}.md"
    lock_md_2 = OUT_DIR / f"intention_lock_summary_{start2}_to_{end2}.md"

    leads_lines_1 = read_file(leads_md_1)
    lock_lines_1 = read_file(lock_md_1)
    leads_lines_2 = read_file(leads_md_2)
    lock_lines_2 = read_file(lock_md_2)

    # 提取与解析“分城市汇总”表
    leads_df_1 = parse_md_table(extract_table_lines(leads_lines_1, "## 分城市汇总"))
    lock_df_1 = parse_md_table(extract_table_lines(lock_lines_1, "## 分城市汇总"))
    leads_df_2 = parse_md_table(extract_table_lines(leads_lines_2, "## 分城市汇总"))
    lock_df_2 = parse_md_table(extract_table_lines(lock_lines_2, "## 分城市汇总"))

    # 选择列并合并（每时间段）
    leads_sel_1 = leads_df_1[["城市", "区域", "线索识别数"]].copy()
    lock_sel_1 = lock_df_1[["城市", "锁单数", "门店数"]].copy()
    m1 = pd.merge(lock_sel_1, leads_sel_1, on="城市", how="outer")
    m1["线索识别数"].fillna(0, inplace=True)
    m1["锁单数"].fillna(0, inplace=True)
    m1["ratio"] = m1.apply(lambda r: (r["锁单数"] / r["线索识别数"]) if (pd.notna(r["线索识别数"]) and r["线索识别数"] > 0) else 0.0, axis=1)

    leads_sel_2 = leads_df_2[["城市", "区域", "线索识别数"]].copy()
    lock_sel_2 = lock_df_2[["城市", "锁单数", "门店数"]].copy()
    m2 = pd.merge(lock_sel_2, leads_sel_2, on="城市", how="outer")
    m2["线索识别数"].fillna(0, inplace=True)
    m2["锁单数"].fillna(0, inplace=True)
    m2["ratio"] = m2.apply(lambda r: (r["锁单数"] / r["线索识别数"]) if (pd.notna(r["线索识别数"]) and r["线索识别数"] > 0) else 0.0, axis=1)

    # 重命名列以区分时间段
    m1 = m1.rename(columns={
        "区域": "区域_前期",
        "锁单数": "锁单数_前期",
        "线索识别数": "线索识别数_前期",
        "门店数": "门店数_前期",
        "ratio": "锁单/线索比值_前期",
    })
    m2 = m2.rename(columns={
        "区域": "区域_后期",
        "锁单数": "锁单数_后期",
        "线索识别数": "线索识别数_后期",
        "门店数": "门店数_后期",
        "ratio": "锁单/线索比值_后期",
    })

    # 合并两期并计算差异
    merged = pd.merge(m1, m2, on="城市", how="outer")
    merged["比值差异(后期-前期)"] = merged["锁单/线索比值_后期"].fillna(0.0) - merged["锁单/线索比值_前期"].fillna(0.0)

    # 排序（按差异降序，其次后期锁单数降序）
    if "锁单数_后期" in merged.columns:
        merged = merged.sort_values(by=["比值差异(后期-前期)", "锁单数_后期"], ascending=[False, False])
    else:
        merged = merged.sort_values(by=["比值差异(后期-前期)"] , ascending=[False])

    # 比值四舍五入
    for col in ["锁单/线索比值_前期", "锁单/线索比值_后期", "比值差异(后期-前期)"]:
        if col in merged.columns:
            merged[col] = merged[col].round(4)

    # 输出 CSV
    if out_csv is None:
        out_csv = OUT_DIR / (
            f"comprehensive_city_analysis_{start1}_to_{end1}_vs_{start2}_to_{end2}.csv"
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    args = parse_args()
    # 若提供了两段时间窗，则生成 CSV 对比；否则生成单段 Markdown 报告
    if args.start2 and args.end2:
        out_csv = Path(args.out_csv) if args.out_csv else None
        csv_path = build_comparison_csv(args.start, args.end, args.start2, args.end2, out_csv)
        print(f"Comparison CSV saved: {csv_path}")
    else:
        report_path = build_report(args.start, args.end)
        print(f"Report saved: {report_path}")
        # Preview top 15
        lines = report_path.read_text(encoding="utf-8").splitlines()
        # Find table and print a short preview
        tbl = extract_table_lines(lines, "## 分城市综合对比")
        preview = "\n".join(tbl[:20]) if tbl else "(no table)"
        print("\nPreview (top rows):\n" + preview)


if __name__ == "__main__":
    main()