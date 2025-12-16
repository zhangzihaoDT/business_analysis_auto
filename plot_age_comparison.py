import argparse
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read_lines(file_path: str, start_line: int, end_line: int) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # one-indexed inclusive range
    return lines[start_line - 1 : end_line]


def infer_group_label(file_path: str) -> str:
    # Try to extract date range or other identifier
    # lock_summary_2024-12-01_to_2024-12-14.md -> 2024-12-01_to_2024-12-14
    m = re.search(r"lock_summary_(.+)\.md", Path(file_path).name)
    if m:
        return m.group(1)
    # Fallback to stem
    return Path(file_path).stem


def parse_age_distribution(lines: List[str]) -> Tuple[List[str], List[int], List[float], Dict[str, float]]:
    bins: List[str] = []
    counts: List[int] = []
    pcts: List[float] = []
    stats: Dict[str, float] = {}

    # Regex for table rows like: "| 00后        |            55 |        2.52 |"
    table_row_re = re.compile(
        r"^\s*\|\s*(?P<bin>[^|]+?)\s*\|\s*(?P<count>[\d,]+)\s*\|\s*(?P<pct>[\d.]+)\s*\|"
    )
    
    # Also support bullet lines for backward compatibility (though user asked for table parsing)
    # "- 25-34: 1608（23.84%）"
    bullet_re = re.compile(
        r"^\s*-\s*(?P<bin>[^:]+):\s*(?P<count>[\d,]+)\s*[（(]\s*(?P<pct>[\d.]+)%\s*[)）]",
        re.UNICODE,
    )
    
    # Regex for summary stats: "- 平均值: 40.00"
    stats_mean_re = re.compile(r"平均值[:：]\s*([\d.]+)")
    stats_median_re = re.compile(r"中位数[:：]\s*([\d.]+)")

    for ln in lines:
        ln_clean = ln.strip()
        if not ln_clean:
            continue
        
        # Check for stats first (lines usually before table)
        m_mean = stats_mean_re.search(ln_clean)
        if m_mean:
            stats["mean"] = float(m_mean.group(1))
            continue
            
        m_median = stats_median_re.search(ln_clean)
        if m_median:
            stats["median"] = float(m_median.group(1))
            continue

        # Check for table row
        m_table = table_row_re.match(ln_clean)
        if m_table:
            bin_label = m_table.group("bin").strip()
            # Skip header separator line like "|:---|---:|---:|"
            if set(bin_label) <= {':', '-'}: 
                continue
            # Skip header line like "| age_group | lock_orders | share_pct |"
            if bin_label == "age_group":
                continue
                
            count_str = m_table.group("count").replace(",", "")
            pct_str = m_table.group("pct")
            try:
                count_val = int(count_str)
            except ValueError:
                count_val = int(float(count_str))
            pct_val = float(pct_str)
            bins.append(bin_label)
            counts.append(count_val)
            pcts.append(pct_val)
            continue

        # Check for bullet format
        m_bullet = bullet_re.match(ln_clean)
        if m_bullet:
            bin_label = m_bullet.group("bin").strip()
            count_str = m_bullet.group("count").replace(",", "")
            pct_str = m_bullet.group("pct")
            try:
                count_val = int(count_str)
            except ValueError:
                count_val = int(float(count_str))
            pct_val = float(pct_str)
            bins.append(bin_label)
            counts.append(count_val)
            pcts.append(pct_val)
            continue

    return bins, counts, pcts, stats


def align_bins_multi(series_list: List[Dict]) -> Tuple[List[str], List[List[int]], List[List[float]]]:
    """Align bins across multiple series."""
    all_bins = []
    for s in series_list:
        all_bins.extend(s['bins'])
    
    set_union = list(dict.fromkeys(all_bins))
    
    # Custom sort order for age groups if present
    age_order_map = {
        "00后": 0, "95后": 1, "90后": 2, "85后": 3, 
        "80后": 4, "75后": 5, "70后": 6, "70前": 7
    }
    
    # Check if all bins are in our known age groups
    if all(b in age_order_map for b in set_union):
        order = sorted(set_union, key=lambda x: age_order_map.get(x, 99))
    else:
        # Try numeric sort
        def to_int_safe(s: str) -> Optional[int]:
            try:
                return int(str(s).strip())
            except Exception:
                return None
        ints = [to_int_safe(b) for b in set_union]
        if all(v is not None for v in ints):
            unique_sorted_ints = sorted(set(ints))
            order = [str(v) for v in unique_sorted_ints]
        else:
            order = set_union

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


def make_dashboard(
    bins: List[str],
    series_data: List[Dict],
    title: str = "年龄分布对比",
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.12,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=("年龄分布（占比折线图对比）", "年龄分布明细（数量与占比）"),
    )

    colors = ["#27AD00", "#005783", "#E63F00", "#6A00A8", "#CC004C"]
    
    # 1. Line Chart
    for i, s in enumerate(series_data):
        color = colors[i % len(colors)]
        group_label = s['label']
        
        fig.add_scatter(
            x=bins,
            y=s['aligned_pcts'],
            name=group_label,
            mode="lines+markers",
            line=dict(color=color, shape="linear"), # Use linear for categorical
            marker=dict(color=color, size=6),
            hovertemplate="年龄组=%{x}<br>占比=%{y:.2f}%<br>数量=%{customdata}<extra></extra>",
            customdata=s['aligned_counts'],
            row=1,
            col=1,
        )
        
        # Add mean line if available (not applicable for categorical "00后" etc easily, 
        # but if we had numeric stats we could plot them. For categorical bins, vertical line is tricky.
        # We'll just skip vertical mean lines for categorical groups unless we map them to x-axis indices)

    fig.update_yaxes(title_text="占比%", tickformat=".2f", ticksuffix="%", row=1, col=1)
    fig.update_xaxes(title_text="年龄段", row=1, col=1)

    # 2. Table
    header_values = ["类别"]
    for s in series_data:
        header_values.append(f"数量<br>({s['label']})")
        header_values.append(f"占比%<br>({s['label']})")
    
    cells_values = [bins + ["总计"]]
    
    for s in series_data:
        total = sum(s['aligned_counts'])
        cells_values.append(s['aligned_counts'] + [total])
        cells_values.append([f"{p:.2f}" for p in s['aligned_pcts']] + ["100.00"])

    # Table formatting
    header_fill = ["#EDEDED"] + [colors[i % len(colors)] for i in range(len(series_data)) for _ in range(2)]
    # Adjust opacity for header colors or use lighter versions? 
    # Let's keep it simple: Gray for Category, specific color for each group's columns
    
    # We need to interleave colors: Col 1 (Gray), Col 2 (Group1), Col 3 (Group1), Col 4 (Group2), Col 5 (Group2)...
    fill_colors = ["#EDEDED"]
    text_colors = ["#333333"]
    for i in range(len(series_data)):
        c = colors[i % len(colors)]
        fill_colors.extend([c, c])
        text_colors.extend(["white", "white"])

    fig.add_table(
        header=dict(
            values=header_values,
            fill_color=fill_colors,
            font=dict(color=text_colors, size=11),
            align="center",
            height=30,
        ),
        cells=dict(
            values=cells_values,
            align=["left"] + ["right"] * (len(series_data) * 2),
            height=26,
            fill=dict(color=["#F9F9F9", "white"]),
        ),
        row=2,
        col=1,
    )

    # Title and stats subtitle
    subtitle_parts = []
    for s in series_data:
        stats = s['stats']
        part = f"<b>{s['label']}</b>"
        if 'mean' in stats:
            part += f": 均值 {stats['mean']:.2f}"
        if 'median' in stats:
            part += f", 中位数 {stats['median']:.2f}"
        subtitle_parts.append(part)
    
    subtitle_text = " | ".join(subtitle_parts)

    fig.update_layout(
        title=dict(text=title + (f"<br><span style='font-size:12px;color:#666'>{subtitle_text}</span>" if subtitle_text else "")),
        margin=dict(l=40, r=20, t=100, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


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


def main():
    parser = argparse.ArgumentParser(description="生成年龄分布对比看板（支持多文件）")
    parser.add_argument("files", nargs="+", help="简报文件路径列表")
    parser.add_argument("--start", type=int, default=90, help="默认起始行（一基）")
    parser.add_argument("--end", type=int, default=102, help="默认结束行（一基，含本行）")
    parser.add_argument("--output", type=str, default="reports/age_comparison_multi.html", help="输出 HTML 路径")

    args = parser.parse_args()

    series_data = []
    
    for fpath in args.files:
        with open(fpath, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        
        # 1. Try to find age section
        age_lines = get_section_lines(all_lines, "## 分年龄段的锁单量与占比")
        
        # 2. Try to find stats section (optional, usually precedes age section)
        stats_lines = get_section_lines(all_lines, "## 车主年龄统计")
        
        if age_lines:
            target_lines = stats_lines + age_lines
        else:
            # Fallback to hardcoded lines if section not found
            target_lines = read_lines(fpath, args.start, args.end)

        bins, counts, pcts, stats = parse_age_distribution(target_lines)
        
        series_data.append({
            "label": infer_group_label(fpath),
            "bins": bins,
            "counts": counts,
            "pcts": pcts,
            "stats": stats
        })

    # Align
    aligned_bins, aligned_counts_list, aligned_pcts_list = align_bins_multi(series_data)
    
    for i, s in enumerate(series_data):
        s['aligned_counts'] = aligned_counts_list[i]
        s['aligned_pcts'] = aligned_pcts_list[i]

    fig = make_dashboard(aligned_bins, series_data)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved dashboard to {out_path}")


if __name__ == "__main__":
    main()
