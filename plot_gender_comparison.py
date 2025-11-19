#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List

import plotly.graph_objects as go


def read_lines(path: Path, start: int, end: int) -> List[str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    return lines[start - 1 : end]


def infer_group_from_filename(path: Path) -> str:
    m = re.search(r"意向订单简报_([^_]+)_", path.name)
    return m.group(1) if m else path.stem


def parse_gender(lines: List[str]) -> Dict[str, Dict[str, float]]:
    # Expect lines like: "- 男: 9932（73.64%）"
    pattern_full = re.compile(r"^-\s*(.+?):\s*([0-9,]+)（([0-9.]+)%）\s*$")
    pattern_half = re.compile(r"^-\s*(.+?):\s*([0-9,]+)\(([^%]+)%\)\s*$")
    out: Dict[str, Dict[str, float]] = {}
    for ln in lines:
        ln = ln.strip()
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


def make_table(data1: Dict[str, Dict[str, float]], label1: str, data2: Dict[str, Dict[str, float]], label2: str, color1: str, color2: str) -> go.Figure:
    # 仅保留“男、女”，同时剔除“默认未知/未知”后重新计算占比
    include_order = ["男", "女"]

    # 以展示类别的总数为分母，确保占比相加为 100%
    total1 = sum(int(data1.get(cat, {}).get("count", 0)) for cat in include_order)
    total2 = sum(int(data2.get(cat, {}).get("count", 0)) for cat in include_order)

    rows = []
    for cat in include_order:
        c1 = int(data1.get(cat, {}).get("count", 0))
        c2 = int(data2.get(cat, {}).get("count", 0))
        p1 = (c1 / total1 * 100.0) if total1 > 0 else 0.0
        p2 = (c2 / total2 * 100.0) if total2 > 0 else 0.0
        rows.append((cat, c1, p1, c2, p2))

    header_vals = [
        "类别",
        f"数量（{label1}）",
        f"占比%（{label1}）",
        f"数量（{label2}）",
        f"占比%（{label2}）",
        f"占比差%（{label2}−{label1}）",
    ]
    header_colors = ["#f0f0f0", f"#{color1}", f"#{color1}", f"#{color2}", f"#{color2}", "#f0f0f0"]

    diffs = [f"{(r[4] - r[2]):+.2f}%" for r in rows]
    cells_vals = [
        [r[0] for r in rows],
        [int(r[1]) for r in rows],
        [f"{r[2]:.2f}%" for r in rows],
        [int(r[3]) for r in rows],
        [f"{r[4]:.2f}%" for r in rows],
        diffs,
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=header_vals, fill_color=header_colors, align="center", font=dict(color="white", size=12)),
                cells=dict(values=cells_vals, align="center")
            )
        ]
    )
    fig.update_layout(title="性别分布对比表", margin=dict(l=20, r=20, t=60, b=20), width=900, height=400)
    return fig


def main():
    ap = argparse.ArgumentParser(description="解析两份简报性别分布，剔除默认未知与未知，生成对比表格")
    ap.add_argument("--file1", required=True, help="第一份简报路径")
    ap.add_argument("--file2", required=True, help="第二份简报路径")
    ap.add_argument("--line-start", type=int, default=34, help="起始行（1-index）")
    ap.add_argument("--line-end", type=int, default=38, help="结束行（1-index，包含）")
    ap.add_argument("--color1", default="27AD00", help="第一组颜色（不含#）")
    ap.add_argument("--color2", default="005783", help="第二组颜色（不含#）")
    ap.add_argument("--out", default=None, help="输出HTML路径，默认保存在reports目录")
    args = ap.parse_args()

    p1 = Path(args.file1)
    p2 = Path(args.file2)
    lines1 = read_lines(p1, args.line_start, args.line_end)
    lines2 = read_lines(p2, args.line_start, args.line_end)

    g1 = infer_group_from_filename(p1)
    g2 = infer_group_from_filename(p2)
    data1 = parse_gender(lines1)
    data2 = parse_gender(lines2)

    fig = make_table(data1, g1, data2, g2, args.color1, args.color2)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("reports") / f"性别分布对比_{g1}_vs_{g2}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved gender comparison table to: {out_path}")


if __name__ == "__main__":
    main()