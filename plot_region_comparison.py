#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read_lines(file_path: str, start_line: int, end_line: int) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines[start_line - 1 : end_line]


def get_section_lines(all_lines: List[str], header_title: str) -> List[str]:
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(all_lines):
        if header_title in line:
            start_idx = i
        if start_idx != -1 and i > start_idx and line.startswith("## "):
            end_idx = i
            break
    
    if start_idx != -1:
        if end_idx == -1: end_idx = len(all_lines)
        return all_lines[start_idx:end_idx]
    return []


def infer_group_label(file_path: str) -> str:
    m = re.search(r"lock_summary_(.+)\.md", Path(file_path).name)
    if m:
        return m.group(1)
    m = re.search(r"意向订单简报_([^_]+)_", Path(file_path).name)
    return m.group(1) if m else Path(file_path).stem


def parse_region_matrix(lines: List[str]) -> Tuple[List[str], List[float], List[float]]:
    """
    Parses the '区域 x 车型矩阵' or '分 region 占比' table.
    Supports both integer counts and float percentages.
    """
    bins: List[str] = []
    counts: List[float] = []
    
    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("|"):
            continue
            
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 3:
            continue
            
        col0 = parts[1]
        
        # Skip header and separator
        if "Region" in col0 or "---" in col0 or "大区" in col0 and "FAC" not in col0 and "虚拟" not in col0: 
            if "Parent Region Name" in col0:
                continue
            if set(col0) <= {':', '-', ' '}:
                continue
        
        region_name = col0
        
        # Sum numeric columns
        row_sum = 0.0
        valid_row = False
        for val_str in parts[2:-1]: # Columns after region name
            val_str = val_str.replace(",", "")
            if not val_str: continue
            try:
                val = float(val_str)
                row_sum += val
                valid_row = True
            except ValueError:
                pass
        
        if valid_row or region_name:
             bins.append(region_name)
             counts.append(row_sum)

    # Calculate percentages
    total = sum(counts)
    # If the input was already percentages (sum ~ 100), this re-normalization 
    # keeps them roughly the same (maybe slight rounding diffs).
    # If input was counts, this calculates percentages.
    pcts = [(c / total * 100.0) if total > 0 else 0.0 for c in counts]
    
    return bins, counts, pcts


def parse_region_distribution_bullets(lines: List[str]) -> Tuple[List[str], List[float], List[float]]:
    # Fallback for bullet points
    bins: List[str] = []
    counts: List[float] = []
    pcts: List[float] = []
    bullet_re = re.compile(
        r"^\s*-\s*(?P<name>[^:]+):\s*(?P<count>[\d,]+)\s*[（(]\s*(?P<pct>[\d.]+)%\s*[)）]\s*$",
        re.UNICODE,
    )
    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("-"):
            continue
        m = bullet_re.match(ln)
        if not m:
            continue
        name = m.group("name").strip()
        count = float(m.group("count").replace(",", ""))
        pct = float(m.group("pct"))
        bins.append(name)
        counts.append(count)
        pcts.append(pct)
    return bins, counts, pcts


def align_bins_multi(series_list: List[Dict]) -> Tuple[List[str], List[List[float]], List[List[float]]]:
    """Align bins across multiple series."""
    all_bins = []
    for s in series_list:
        all_bins.extend(s['bins'])
    
    # Unique bins, preserve order roughly?
    # Usually regions have a specific order (e.g. by volume or geographic).
    # We can sort by total volume across all series, or just alphabetic, or keep first appearance.
    # Let's keep first appearance to respect the source file order (usually sorted by volume).
    order = list(dict.fromkeys(all_bins))
    
    aligned_counts_list = []
    aligned_pcts_list = []
    
    for s in series_list:
        map_c = {b: c for b, c in zip(s['bins'], s['counts'])}
        map_p = {b: p for b, p in zip(s['bins'], s['pcts'])}
        
        aligned_counts = [map_c.get(b, 0) for b in order]
        aligned_pcts = [map_p.get(b, 0.0) for b in order]
        
        aligned_counts_list.append(aligned_counts)
        aligned_pcts_list.append(aligned_pcts)

    return order, aligned_counts_list, aligned_pcts_list


def make_dashboard_multi(
    bins: List[str],
    series_data: List[Dict],
    title: str = "父区域分布对比",
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.12,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=("父区域分布（占比对比）", "父区域分布明细（数量与占比）"),
    )

    colors = ["#27AD00", "#005783", "#E63F00", "#6A00A8", "#CC004C"]
    
    # 1. Bar Chart
    for i, s in enumerate(series_data):
        color = colors[i % len(colors)]
        label = s['label']
        
        fig.add_bar(
            x=bins,
            y=s['aligned_pcts'],
            name=label,
            marker_color=color,
            hovertemplate="区域=%{x}<br>占比=%{y:.2f}%<br>数量=%{customdata}<extra></extra>",
            customdata=s['aligned_counts'],
            row=1,
            col=1,
        )

    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="占比%", tickformat=".0f", row=1, col=1)

    # 2. Table
    header_vals = ["父区域"]
    fill_colors = ["#EDEDED"]
    text_colors = ["#333333"]
    
    for i, s in enumerate(series_data):
        label = s['label']
        header_vals.append(f"数量<br>({label})")
        header_vals.append(f"占比%<br>({label})")
        
        c = colors[i % len(colors)]
        fill_colors.extend([c, c])
        text_colors.extend(["white", "white"])

    cells_vals = [bins + ["总计"]]
    
    for s in series_data:
        counts = s['aligned_counts']
        pcts = s['aligned_pcts']
        total = sum(counts)
        
        # Format counts: if all are integers, show as int; otherwise .2f
        is_all_int = all(isinstance(x, (int, float)) and float(x).is_integer() for x in counts)
        if is_all_int:
             formatted_counts = [f"{int(x)}" for x in counts]
             formatted_total = f"{int(total)}"
        else:
             formatted_counts = [f"{x:.2f}" for x in counts]
             formatted_total = f"{total:.2f}"

        cells_vals.append(formatted_counts + [formatted_total])
        cells_vals.append([f"{p:.2f}" for p in pcts] + ["100.00"])

    fig.add_table(
        header=dict(
            values=header_vals,
            fill_color=fill_colors,
            font=dict(color=text_colors, size=11),
            align="center",
            height=30,
        ),
        cells=dict(
            values=cells_vals,
            align=["left"] + ["right"] * (len(series_data) * 2),
            height=26,
            fill=dict(color=["#F9F9F9", "white"]),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=dict(text=title),
        margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="父区域分布对比（分组柱状图+表格，支持多文件）")
    parser.add_argument("files", nargs="+", help="简报文件路径列表")
    parser.add_argument("--output", type=str, default="reports/region_comparison_multi.html", help="输出 HTML 路径")

    args = parser.parse_args()

    series_data = []
    
    for fpath in args.files:
        with open(fpath, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        
        # Priority 1: User requested section "分 region 占比（%）（按车型列归一化）"
        target_lines = get_section_lines(all_lines, "## 分 region 占比")
        if target_lines:
             bins, counts, pcts = parse_region_matrix(target_lines)
             print(f"[{Path(fpath).name}] Parsed matrix from '分 region 占比'.")
        else:
            # Priority 2: "区域 x 车型矩阵" (Counts)
            target_lines = get_section_lines(all_lines, "## 区域 x 车型矩阵")
            if target_lines:
                 bins, counts, pcts = parse_region_matrix(target_lines)
                 print(f"[{Path(fpath).name}] Parsed matrix from '区域 x 车型矩阵'.")
            else:
                 # Priority 3: Bullet points fallback
                 print(f"[{Path(fpath).name}] '区域 x 车型矩阵' not found, trying bullets...")
                 bins, counts, pcts = parse_region_distribution_bullets(all_lines)

        series_data.append({
            "label": infer_group_label(fpath),
            "bins": bins,
            "counts": counts,
            "pcts": pcts
        })

    # Align
    aligned_bins, aligned_counts_list, aligned_pcts_list = align_bins_multi(series_data)
    
    for i, s in enumerate(series_data):
        s['aligned_counts'] = aligned_counts_list[i]
        s['aligned_pcts'] = aligned_pcts_list[i]

    fig = make_dashboard_multi(aligned_bins, series_data)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved dashboard to {out_path}")


if __name__ == "__main__":
    main()