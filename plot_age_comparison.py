import argparse
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read_lines(file_path: str, start_line: int, end_line: int) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # one-indexed inclusive range
    return lines[start_line - 1 : end_line]


def infer_group_label(file_path: str) -> str:
    # Try to extract token between '意向订单简报_' and next '_'
    m = re.search(r"意向订单简报_([A-Za-z0-9]+)_", Path(file_path).name)
    if m:
        return m.group(1)
    # Fallback to stem
    return Path(file_path).stem


def parse_age_distribution(lines: List[str]) -> Tuple[List[str], List[int], List[float], Dict[str, float]]:
    bins: List[str] = []
    counts: List[int] = []
    pcts: List[float] = []
    stats: Dict[str, float] = {}

    # Regex for bullet lines like: "- 25-34: 1608（23.84%）"
    bullet_re = re.compile(
        r"^\s*-\s*(?P<bin>[^:]+):\s*(?P<count>[\d,]+)\s*[（(]\s*(?P<pct>[\d.]+)%\s*[)）]",
        re.UNICODE,
    )
    # Regex for summary stats: "年龄均值：42.4；年龄中位数：38.0"
    stats_re = re.compile(r"年龄均值：\s*([\d.]+)；年龄中位数：\s*([\d.]+)")

    for ln in lines:
        ln_clean = ln.strip()
        if not ln_clean:
            continue
        m = bullet_re.match(ln_clean)
        if m:
            bin_label = m.group("bin").strip()
            count_str = m.group("count").replace(",", "")
            pct_str = m.group("pct")
            try:
                count_val = int(count_str)
            except ValueError:
                # Fallback: try float then int
                count_val = int(float(count_str))
            pct_val = float(pct_str)
            bins.append(bin_label)
            counts.append(count_val)
            pcts.append(pct_val)
            continue
        sm = stats_re.search(ln_clean)
        if sm:
            stats["mean"] = float(sm.group(1))
            stats["median"] = float(sm.group(2))

    return bins, counts, pcts, stats


def align_bins(
    bins1: List[str], counts1: List[int], pcts1: List[float],
    bins2: List[str], counts2: List[int], pcts2: List[float],
) -> Tuple[List[str], List[int], List[int], List[float], List[float]]:
    """Align bins across two series.

    If bins are integer-like (e.g., per-year buckets: "25", "26"), sort ascending
    so we can render a continuous bell curve. Otherwise, preserve union order.
    """
    set_union = list(dict.fromkeys(bins1 + bins2))

    def to_int_safe(s: str) -> Optional[int]:
        try:
            return int(str(s).strip())
        except Exception:
            return None

    ints = [to_int_safe(b) for b in set_union]
    if all(v is not None for v in ints):
        # sort unique ints ascending, then convert back to strings for table use
        unique_sorted_ints = sorted(set(ints))
        order = [str(v) for v in unique_sorted_ints]
    else:
        order = set_union

    map1c = {b: c for b, c in zip(bins1, counts1)}
    map1p = {b: p for b, p in zip(bins1, pcts1)}
    map2c = {b: c for b, c in zip(bins2, counts2)}
    map2p = {b: p for b, p in zip(bins2, pcts2)}

    aligned_counts1 = [map1c.get(b, 0) for b in order]
    aligned_counts2 = [map2c.get(b, 0) for b in order]
    aligned_pcts1 = [map1p.get(b, 0.0) for b in order]
    aligned_pcts2 = [map2p.get(b, 0.0) for b in order]

    return order, aligned_counts1, aligned_counts2, aligned_pcts1, aligned_pcts2


def make_dashboard(
    bins: List[str],
    counts1: List[int], counts2: List[int], pcts1: List[float], pcts2: List[float],
    group1: str, group2: str, color1: str, color2: str,
    stats1: Dict[str, float], stats2: Dict[str, float],
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.12,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=("年龄分布（占比面积图对比）", "年龄分布明细（数量与占比）"),
    )

    # Line chart (percentages) for bell-curve visualization
    # Use numeric x when possible for continuous axis
    def bins_to_int(bins: List[str]) -> Optional[List[int]]:
        try:
            xs = [int(str(b).strip()) for b in bins]
            return xs
        except Exception:
            return None

    x_numeric = bins_to_int(bins)
    x_for_plot = x_numeric if x_numeric is not None else bins

    fig.add_scatter(
        x=x_for_plot,
        y=pcts1,
        name=group1,
        mode="lines+markers",
        line=dict(color=color1, shape="spline"),
        marker=dict(color=color1, size=5),
        fill="tozeroy",
        opacity=0.10,
        hovertemplate="年龄=%{x}<br>占比=%{y:.2f}%<br>数量=%{customdata}<extra></extra>",
        customdata=counts1,
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=x_for_plot,
        y=pcts2,
        name=group2,
        mode="lines+markers",
        line=dict(color=color2, shape="spline"),
        marker=dict(color=color2, size=5),
        fill="tozeroy",
        opacity=0.10,
        hovertemplate="年龄=%{x}<br>占比=%{y:.2f}%<br>数量=%{customdata}<extra></extra>",
        customdata=counts2,
        row=1,
        col=1,
    )
    # Y-axis: percentage with suffix
    y_max = max([0.0] + pcts1 + pcts2) if (pcts1 or pcts2) else 1.0
    fig.update_yaxes(title_text="占比%", tickformat=".2f", ticksuffix="%", row=1, col=1)
    fig.update_xaxes(title_text="年龄（岁）", row=1, col=1)

    # Add dashed vertical lines at means (computed from distribution if not provided)
    def weighted_mean(x_vals: Optional[List[int]], counts: List[int], stats: Dict[str, float]) -> Optional[float]:
        if stats and "mean" in stats:
            try:
                return float(stats.get("mean"))
            except Exception:
                pass
        if x_vals is None:
            return None
        total = sum(counts)
        if total <= 0:
            return None
        return sum(x * c for x, c in zip(x_vals, counts)) / total

    mean1 = weighted_mean(x_numeric, counts1, stats1)
    mean2 = weighted_mean(x_numeric, counts2, stats2)

    if mean1 is not None:
        fig.add_vline(x=mean1, line_dash="dash", line_color=color1, row=1, col=1)
        fig.add_annotation(x=mean1, y=y_max, xref="x1", yref="y1", text=f"{group1} 均值 {mean1:.1f}",
                           showarrow=False, font=dict(color=color1, size=11))
    if mean2 is not None:
        fig.add_vline(x=mean2, line_dash="dash", line_color=color2, row=1, col=1)
        fig.add_annotation(x=mean2, y=y_max, xref="x1", yref="y1", text=f"{group2} 均值 {mean2:.1f}",
                           showarrow=False, font=dict(color=color2, size=11))

    # Table data
    total1 = sum(counts1)
    total2 = sum(counts2)

    # Aggregate table bins into 5-year buckets
    def to_int_safe(s: str) -> Optional[int]:
        try:
            return int(str(s).strip())
        except Exception:
            return None

    ages = [to_int_safe(b) for b in bins]
    use_5yr = all(a is not None for a in ages)

    if use_5yr:
        # build 5-year buckets like 15-19, 20-24, ...
        starts = [ (a // 5) * 5 for a in ages ]
        # ensure deterministic order
        unique_starts = sorted(set(starts))
        labels_5yr = [f"{s}-{s+4}" for s in unique_starts]

        # map age->index for counts lookup
        idx_by_age: Dict[int, int] = {a: i for i, a in enumerate(ages)}

        def sum_counts_for_start(start: int, counts: List[int]) -> int:
            s = 0
            for a in range(start, start + 5):
                i = idx_by_age.get(a)
                if i is not None:
                    s += counts[i]
            return s

        agg_counts1 = [sum_counts_for_start(s, counts1) for s in unique_starts]
        agg_counts2 = [sum_counts_for_start(s, counts2) for s in unique_starts]
        agg_pcts1 = [ (c / total1 * 100.0) if total1 > 0 else 0.0 for c in agg_counts1 ]
        agg_pcts2 = [ (c / total2 * 100.0) if total2 > 0 else 0.0 for c in agg_counts2 ]

        table_bins = labels_5yr
        table_counts1 = agg_counts1
        table_counts2 = agg_counts2
        table_pcts1 = agg_pcts1
        table_pcts2 = agg_pcts2
    else:
        # fallback to original bins
        table_bins = bins
        table_counts1 = counts1
        table_counts2 = counts2
        table_pcts1 = pcts1
        table_pcts2 = pcts2
    header_values = [
        ["类别"],
        [f"数量（{group1}）"],
        [f"占比%（{group1}）"],
        [f"数量（{group2}）"],
        [f"占比%（{group2}）"],
        [f"占比差%（{group2}−{group1}）"],
    ]
    cells_bins = table_bins + ["总计"]
    cells_counts1 = table_counts1 + [total1]
    cells_pcts1 = [f"{p:.2f}" for p in table_pcts1] + ["100.00"]
    cells_counts2 = table_counts2 + [total2]
    cells_pcts2 = [f"{p:.2f}" for p in table_pcts2] + ["100.00"]
    # Difference column: LS9 - CM2 with sign, total row is 0.00
    diff_pcts = [f"{(p2 - p1):+.2f}" for p1, p2 in zip(table_pcts1, table_pcts2)] + ["+0.00"]

    fig.add_table(
        header=dict(
            values=[
                header_values[0], header_values[1], header_values[2], header_values[3], header_values[4], header_values[5]
            ],
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

    # Title and layout
    title_text = f"年龄分布对比 | {group1} vs {group2}"
    subtitle_text = ""
    if stats1:
        subtitle_text += f"{group1} 均值 {stats1.get('mean', '-')}, 中位数 {stats1.get('median', '-')}"
    if stats2:
        if subtitle_text:
            subtitle_text += " | "
        subtitle_text += f"{group2} 均值 {stats2.get('mean', '-')}, 中位数 {stats2.get('median', '-')}"

    fig.update_layout(
        title=dict(text=title_text + (f"<br><span style='font-size:12px;color:#666'>{subtitle_text}</span>" if subtitle_text else "")),
        margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="生成年龄分布对比看板（分组柱状图+表格）")
    parser.add_argument("--file1", required=True, help="第一份简报文件路径")
    parser.add_argument("--file2", required=True, help="第二份简报文件路径")
    # Support different ranges per file for fine-grained 1-year buckets
    parser.add_argument("--start", type=int, default=44, help="起始行（一基），两文件共用时使用")
    parser.add_argument("--end", type=int, default=50, help="结束行（一基，含本行），两文件共用时使用")
    parser.add_argument("--start1", type=int, default=None, help="文件1起始行（一基）")
    parser.add_argument("--end1", type=int, default=None, help="文件1结束行（一基，含本行)")
    parser.add_argument("--start2", type=int, default=None, help="文件2起始行（一基）")
    parser.add_argument("--end2", type=int, default=None, help="文件2结束行（一基，含本行)")
    parser.add_argument("--group1", type=str, default=None, help="第一组标签（默认从文件名推断）")
    parser.add_argument("--group2", type=str, default=None, help="第二组标签（默认从文件名推断）")
    parser.add_argument("--color1", type=str, default="#27AD00", help="第一组颜色")
    parser.add_argument("--color2", type=str, default="#005783", help="第二组颜色")
    parser.add_argument("--output", type=str, default="reports/年龄分布对比.html", help="输出 HTML 路径")

    args = parser.parse_args()

    group1 = args.group1 or infer_group_label(args.file1)
    group2 = args.group2 or infer_group_label(args.file2)

    s1 = args.start1 if args.start1 is not None else args.start
    e1 = args.end1 if args.end1 is not None else args.end
    s2 = args.start2 if args.start2 is not None else args.start
    e2 = args.end2 if args.end2 is not None else args.end

    lines1 = read_lines(args.file1, s1, e1)
    lines2 = read_lines(args.file2, s2, e2)

    bins1, counts1, pcts1, stats1 = parse_age_distribution(lines1)
    bins2, counts2, pcts2, stats2 = parse_age_distribution(lines2)

    bins, ac1, ac2, ap1, ap2 = align_bins(bins1, counts1, pcts1, bins2, counts2, pcts2)

    fig = make_dashboard(
        bins, ac1, ac2, ap1, ap2, group1, group2, args.color1, args.color2, stats1, stats2
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved dashboard to {out_path}")


if __name__ == "__main__":
    main()