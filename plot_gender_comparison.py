#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read_lines(path: Path, start: int, end: int) -> List[str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    return lines[start - 1 : end]


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


def infer_group_from_filename(path: Path) -> str:
    m = re.search(r"lock_summary_(.+)\.md", path.name)
    if m:
        return m.group(1)
    m = re.search(r"意向订单简报_([^_]+)_", path.name)
    return m.group(1) if m else path.stem


def parse_gender(lines: List[str]) -> Dict[str, Dict[str, float]]:
    # Expect lines like: "- 男: 9932（73.64%）"
    # Or table rows: "| 男 | 1297 | 58.06 |"
    
    pattern_full = re.compile(r"^-\s*(.+?):\s*([0-9,]+)（([0-9.]+)%）\s*$")
    pattern_half = re.compile(r"^-\s*(.+?):\s*([0-9,]+)\(([^%]+)%\)\s*$")
    
    table_row_re = re.compile(
        r"^\s*\|\s*(?P<gender>[^|]+?)\s*\|\s*(?P<count>[\d,]+)\s*\|\s*(?P<pct>[\d.]+)\s*\|"
    )

    out: Dict[str, Dict[str, float]] = {}
    
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
            
        # Try table row
        m_table = table_row_re.match(ln)
        if m_table:
            gender = m_table.group("gender").strip()
            if gender in ["gender", ":---------"]: # Skip header/separator
                continue
                
            count = int(m_table.group("count").replace(",", ""))
            pct = float(m_table.group("pct"))
            out[gender] = {"count": count, "percent": pct}
            continue

        if not ln.startswith("-"):
            continue
            
        m = pattern_full.match(ln) or pattern_half.match(ln)
        if not m:
            continue
        name, count, pct = m.group(1), m.group(2), m.group(3)
        count = int(count.replace(",", ""))
        pct = float(pct)
        out[name] = {"count": count, "percent": pct}
    return out


def make_dashboard_multi(series_list: List[Dict[str, Any]], title: str = "性别分布对比") -> go.Figure:
    # 仅保留“男、女”
    include_order = ["男", "女"]
    
    # Colors for table columns
    colors = ["#27AD00", "#005783", "#E63F00", "#6A00A8", "#CC004C"]
    
    # 1. Prepare data for Bar Chart and Table
    categories = include_order
    
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.15,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=("性别分布（占比柱状图对比）", "性别分布明细（数量与占比）"),
    )

    # Bar Chart
    for i, s in enumerate(series_list):
        data = s['data']
        label = s['label']
        color = colors[i % len(colors)]
        
        # Re-calculate percentages based on just Male/Female total?
        # Or use the provided percentages?
        # Usually user wants the percentage relative to the total valid samples (Male+Female).
        # The input file says "下表基于有效样本 2234 个统计" (which excludes unknowns).
        # So the percentages in the file should sum to 100% for Male+Female.
        # Let's verify or re-calc.
        
        total_count = sum(data.get(cat, {}).get("count", 0) for cat in include_order)
        
        y_vals = []
        custom_data = []
        for cat in include_order:
            cnt = data.get(cat, {}).get("count", 0)
            if total_count > 0:
                pct = (cnt / total_count) * 100.0
            else:
                pct = 0.0
            y_vals.append(pct)
            custom_data.append(cnt)
            
        fig.add_trace(
            go.Bar(
                x=categories,
                y=y_vals,
                name=label,
                marker_color=color,
                text=[f"{y:.1f}%" for y in y_vals],
                textposition='auto',
                hovertemplate="性别=%{x}<br>占比=%{y:.2f}%<br>数量=%{customdata}<extra></extra>",
                customdata=custom_data
            ),
            row=1, col=1
        )

    fig.update_yaxes(title_text="占比%", row=1, col=1)

    # Table
    header_vals = ["类别"]
    fill_colors = ["#EDEDED"]
    text_colors = ["#333333"]
    
    for i, s in enumerate(series_list):
        label = s['label']
        header_vals.append(f"数量<br>({label})")
        header_vals.append(f"占比%<br>({label})")
        
        c = colors[i % len(colors)]
        fill_colors.extend([c, c])
        text_colors.extend(["white", "white"])

    cells_vals = [categories + ["总计"]]
    
    for s in series_list:
        data = s['data']
        total = sum(data.get(cat, {}).get("count", 0) for cat in include_order)
        
        col_counts = []
        col_pcts = []
        
        for cat in include_order:
            cnt = data.get(cat, {}).get("count", 0)
            if total > 0:
                pct = (cnt / total) * 100.0
            else:
                pct = 0.0
            col_counts.append(cnt)
            col_pcts.append(f"{pct:.2f}")
            
        cells_vals.append(col_counts + [total])
        cells_vals.append(col_pcts + ["100.00"])

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
            align=["left"] + ["right"] * (len(series_list) * 2),
            height=26,
            fill=dict(color=["#F9F9F9", "white"]),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=80, b=20),
        height=600,
        barmode='group'
    )
    return fig


def main():
    ap = argparse.ArgumentParser(description="解析多份简报性别分布，生成对比看板")
    ap.add_argument("files", nargs="+", help="简报文件路径列表")
    ap.add_argument("--out", default="reports/gender_comparison_multi.html", help="输出HTML路径")
    args = ap.parse_args()

    series_data = []
    
    for fpath in args.files:
        path = Path(fpath)
        text = path.read_text(encoding="utf-8")
        all_lines = text.splitlines()
        
        # Try to find section
        target_lines = get_section_lines(all_lines, "## 分性别的锁单量与占比")
        if not target_lines:
             # Fallback: try to find lines that look like gender stats if section not found
             # Or just use the whole file? No, risky.
             print(f"[{path.name}] Section '## 分性别的锁单量与占比' not found, trying whole file...")
             target_lines = all_lines
        else:
             print(f"[{path.name}] Found section lines.")

        data = parse_gender(target_lines)
        series_data.append({
            "label": infer_group_from_filename(path),
            "data": data
        })

    fig = make_dashboard_multi(series_data)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved gender comparison dashboard to: {out_path}")


if __name__ == "__main__":
    main()