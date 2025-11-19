import argparse
import re
from pathlib import Path
from typing import List, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read_lines(file_path: str, start_line: int, end_line: int) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines[start_line - 1 : end_line]


def infer_group_label(file_path: str) -> str:
    m = re.search(r"意向订单简报_([^_]+)_", Path(file_path).name)
    return m.group(1) if m else Path(file_path).stem


def parse_region_distribution(lines: List[str]) -> Tuple[List[str], List[int], List[float]]:
    bins: List[str] = []
    counts: List[int] = []
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
        count = int(m.group("count").replace(",", ""))
        pct = float(m.group("pct"))
        bins.append(name)
        counts.append(count)
        pcts.append(pct)
    return bins, counts, pcts


def align_bins(
    bins1: List[str], counts1: List[int], pcts1: List[float],
    bins2: List[str], counts2: List[int], pcts2: List[float],
) -> Tuple[List[str], List[int], List[int], List[float], List[float]]:
    order = list(dict.fromkeys(bins1 + bins2))
    map1c = {b: c for b, c in zip(bins1, counts1)}
    map1p = {b: p for b, p in zip(bins1, pcts1)}
    map2c = {b: c for b, c in zip(bins2, counts2)}
    map2p = {b: p for b, p in zip(bins2, pcts2)}
    ac1 = [map1c.get(b, 0) for b in order]
    ac2 = [map2c.get(b, 0) for b in order]
    ap1 = [map1p.get(b, 0.0) for b in order]
    ap2 = [map2p.get(b, 0.0) for b in order]
    return order, ac1, ac2, ap1, ap2


def make_dashboard(
    bins: List[str], counts1: List[int], counts2: List[int], pcts1: List[float], pcts2: List[float],
    group1: str, group2: str, color1: str, color2: str,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.12,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=("父区域分布（占比对比）", "父区域分布明细（数量与占比）"),
    )

    # Bar chart of counts
    fig.add_bar(
        x=bins,
        y=pcts1,
        name=group1,
        marker_color=color1,
        hovertemplate="区域=%{x}<br>占比=%{y:.2f}%<br>数量=%{customdata}<extra></extra>",
        customdata=counts1,
        row=1,
        col=1,
    )
    fig.add_bar(
        x=bins,
        y=pcts2,
        name=group2,
        marker_color=color2,
        hovertemplate="区域=%{x}<br>占比=%{y:.2f}%<br>数量=%{customdata}<extra></extra>",
        customdata=counts2,
        row=1,
        col=1,
    )
    fig.update_layout(barmode="group")

    # Y轴改为百分比（0-100）
    fig.update_yaxes(title_text="占比%", range=[0, 20], tickformat=".0f", row=1, col=1)

    # Table
    total1 = sum(counts1)
    total2 = sum(counts2)
    header_vals = [
        ["父区域"],
        [f"数量（{group1}）"],
        [f"占比%（{group1}）"],
        [f"数量（{group2}）"],
        [f"占比%（{group2}）"],
        [f"占比差%（{group2}−{group1}）"],
    ]
    cells_bins = bins + ["总计"]
    cells_counts1 = counts1 + [total1]
    cells_pcts1 = [f"{p:.2f}" for p in pcts1] + ["100.00"]
    cells_counts2 = counts2 + [total2]
    cells_pcts2 = [f"{p:.2f}" for p in pcts2] + ["100.00"]
    # 差异列：占比差（group2 - group1），保留两位小数并带符号，合计行为 +0.00
    diff_pcts = [f"{(p2 - p1):+.2f}" for p1, p2 in zip(pcts1, pcts2)] + ["+0.00"]

    fig.add_table(
        header=dict(
            values=[header_vals[0], header_vals[1], header_vals[2], header_vals[3], header_vals[4], header_vals[5]],
            fill_color=["#EDEDED", color1, color1, color2, color2, "#EDEDED"],
            font=dict(color=["#333333", "white", "white", "white", "white", "#333333"], size=12),
            align=["left", "center", "center", "center", "center", "center"],
            height=30,
        ),
        cells=dict(
            values=[cells_bins, cells_counts1, cells_pcts1, cells_counts2, cells_pcts2, diff_pcts],
            align=["left", "right", "right", "right", "right", "right"],
            height=26,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=dict(text=f"父区域分布对比 | {group1} vs {group2}"),
        margin=dict(l=40, r=20, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="父区域分布对比（分组柱状图+表格）")
    parser.add_argument("--file1", required=True, help="第一份简报文件路径")
    parser.add_argument("--file2", required=True, help="第二份简报文件路径")
    parser.add_argument("--start", type=int, default=58, help="起始行（一基）")
    parser.add_argument("--end", type=int, default=70, help="结束行（一基，含本行)")
    parser.add_argument("--group1", type=str, default=None, help="第一组标签（默认从文件名推断）")
    parser.add_argument("--group2", type=str, default=None, help="第二组标签（默认从文件名推断）")
    parser.add_argument("--color1", type=str, default="#27AD00", help="第一组颜色")
    parser.add_argument("--color2", type=str, default="#005783", help="第二组颜色")
    parser.add_argument("--output", type=str, default="reports/父区域分布对比.html", help="输出 HTML 路径")

    args = parser.parse_args()

    group1 = args.group1 or infer_group_label(args.file1)
    group2 = args.group2 or infer_group_label(args.file2)

    lines1 = read_lines(args.file1, args.start, args.end)
    lines2 = read_lines(args.file2, args.start, args.end)

    bins1, counts1, pcts1 = parse_region_distribution(lines1)
    bins2, counts2, pcts2 = parse_region_distribution(lines2)

    bins, ac1, ac2, ap1, ap2 = align_bins(bins1, counts1, pcts1, bins2, counts2, pcts2)

    fig = make_dashboard(bins, ac1, ac2, ap1, ap2, group1, group2, args.color1, args.color2)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved dashboard to {out_path}")


if __name__ == "__main__":
    main()