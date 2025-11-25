#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
from typing import Dict, Tuple, List, Optional
import os

import numpy as np
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

def safe_float(x: str) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

def safe_int(x: str) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None

def parse_md_ratios(md_path: str) -> Dict[str, Tuple[Optional[float], Optional[int]]]:
    """
    解析综合城市Markdown表，返回 {城市: (锁单数/线索识别数比值, 线索识别数)}
    """
    res: Dict[str, Tuple[Optional[float], Optional[int]]] = {}
    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            # 过滤表头和对齐分隔行
            if not line.startswith("| "):
                continue
            if "| 城市" in line or "|:---" in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            # parts[0]为空字符串（因为以|开头），实际列位于 1..-2
            cols = parts[1:-1]
            if len(cols) < 6:
                # 保护：不符合预期表结构的行跳过
                continue
            city = cols[0]
            leads = safe_int(cols[4])
            ratio = safe_float(cols[5])
            res[city] = (ratio, leads)
    return res

def parse_md_orders(md_path: str) -> Dict[str, Optional[int]]:
    """
    解析综合城市Markdown表，返回 {城市: 锁单数}
    预期列：城市 | 锁单数 | 门店数 | 区域 | 线索识别数 | 锁单数/线索识别数比值
    """
    res: Dict[str, Optional[int]] = {}
    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("| "):
                continue
            if "| 城市" in line or "|:---" in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            cols = parts[1:-1]
            if len(cols) < 6:
                continue
            city = cols[0]
            orders = safe_int(cols[1])
            res[city] = orders
    return res

def parse_ev_csv(ev_csv_path: str) -> Dict[str, Optional[float]]:
    """
    解析 cm2_city_range_ev_counts.csv，返回 {城市: 增程/纯电比值}
    兼容城市列为 "License City" 或 "Store City"。
    """
    res: Dict[str, Optional[float]] = {}
    with open(ev_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        # 动态识别城市列
        city_col = None
        if "Store City" in fieldnames:
            city_col = "Store City"
        elif "License City" in fieldnames:
            city_col = "License City"
        else:
            # 尝试其他可能的变体
            for cand in ["store_city", "license_city", "城市", "City"]:
                if cand in fieldnames:
                    city_col = cand
                    break
        # 若无法识别，返回空结果
        if not city_col:
            return res

        for row in reader:
            city = (row.get(city_col) or "").strip()
            ev_ratio = safe_float(row.get("增程/纯电比值") or "")
            if city:
                res[city] = ev_ratio
    return res

def pearsonr(x: List[float], y: List[float]) -> float:
    """
    纯Python实现Pearson相关系数（不依赖SciPy）
    """
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    vx = sum((xi - mx) ** 2 for xi in x)
    vy = sum((yi - my) ** 2 for yi in y)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)

def rankdata(vals: List[float]) -> List[float]:
    """
    计算平均秩（处理并列），返回每个元素的秩；值为NaN的会被剔除前置处理
    """
    # 记录值与原索引
    indexed = list(enumerate(vals))
    # 排序（按值）
    indexed.sort(key=lambda kv: kv[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(indexed):
        j = i
        # 查找并列范围
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        # 平均秩：位置从 i 到 j，对应秩从 i+1 到 j+1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks

def spearmanr(x: List[float], y: List[float]) -> float:
    """
    纯Python实现Spearman相关系数（用平均秩后做Pearson）
    """
    rx = rankdata(x)
    ry = rankdata(y)
    return pearsonr(rx, ry)


def lowess(x: np.ndarray, y: np.ndarray, frac: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    简易LOWESS实现（局部加权线性回归，tricube核）。
    - x, y: 一维数组
    - frac: 平滑带宽比例（相对x范围）。
    返回：按x排序后的 (x_sorted, y_smooth)
    """
    if len(x) == 0:
        return np.array([]), np.array([])
    # 排序并去除NaN
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_range = x.max() - x.min()
    if x_range == 0:
        return x, np.repeat(np.nanmean(y), len(y))
    bandwidth = max(1e-8, frac * x_range)

    yhat = np.empty_like(y)
    for i, xi in enumerate(x):
        # 权重：tricube((|x - xi|) / bandwidth)
        dist = np.abs(x - xi) / bandwidth
        w = (1 - dist ** 3) ** 3
        w[dist >= 1] = 0.0
        # 加权线性回归 y = a + b*x
        W = np.diag(w)
        X = np.column_stack([np.ones_like(x), x])
        try:
            beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
            yhat[i] = beta[0] + beta[1] * xi
        except Exception:
            yhat[i] = np.nan
    return x, yhat

def main():
    ap = argparse.ArgumentParser(description="城市效率比值差异与EV比值相关性分析")
    ap.add_argument("--nov_md", required=True, help="后期综合报告Markdown路径（如 2025-11-12_to_2025-11-16.md）")
    ap.add_argument("--sep_md", required=True, help="前期综合报告Markdown路径（如 2025-09-10_to_2025-09-14.md）")
    ap.add_argument("--ev_csv", required=True, help="cm2_city_range_ev_counts.csv 路径")
    ap.add_argument("--min_leads", type=int, default=0, help="按后期线索识别数过滤的阈值（默认不过滤）")
    ap.add_argument("--out_md", default="processed/analysis_results/city_efficiency_ev_correlation_对比.md",
                    help="输出Markdown路径")
    ap.add_argument("--pairs_output", default=None, help="导出配对数据CSV（城市, 后期比值, EV比值, 差异）")
    ap.add_argument("--plot_output", default=None, help="Plotly 散点+LOWESS输出HTML路径")
    ap.add_argument("--label_top", type=int, default=10, help="标注TOP城市数量（默认10）")
    ap.add_argument("--label_small_top", type=int, default=0, help="标注差异最小TOP数量（默认0不标注）")
    ap.add_argument(
        "--label_mode",
        default="diff_abs",
        choices=["diff_abs", "efficiency", "ev"],
        help="TOP选择依据：差异绝对值/后期效率比值/EV比值"
    )
    args = ap.parse_args()

    nov = parse_md_ratios(args.nov_md)
    sep = parse_md_ratios(args.sep_md)
    ev = parse_ev_csv(args.ev_csv)
    nov_orders = parse_md_orders(args.nov_md)

    exclude_cities = {"拉萨市"}
    for c in list(nov.keys()):
        if c in exclude_cities:
            nov.pop(c, None)
    for c in list(sep.keys()):
        if c in exclude_cities:
            sep.pop(c, None)
    for c in list(ev.keys()):
        if c in exclude_cities:
            ev.pop(c, None)
    for c in list(nov_orders.keys()):
        if c in exclude_cities:
            nov_orders.pop(c, None)

    # 计算比值差异（nov - sep）
    diff_rows: List[Tuple[str, Optional[float], Optional[float], Optional[float], Optional[int]]] = []
    for city, (nov_ratio, nov_leads) in nov.items():
        sep_ratio, _ = sep.get(city, (None, None))
        diff = None
        if nov_ratio is not None and sep_ratio is not None:
            diff = nov_ratio - sep_ratio
        diff_rows.append((city, sep_ratio, nov_ratio, diff, nov_leads))

    # 过滤：后期线索识别数 >= min_leads（如果有值）
    if args.min_leads > 0:
        diff_rows = [r for r in diff_rows if (r[4] is not None and r[4] >= args.min_leads)]

    # 排序输出：按差异升序（跌幅最大在前）
    diff_sorted = sorted(diff_rows, key=lambda r: (r[3] if r[3] is not None else float("inf")))

    # 差异映射（用于相关性与标注）
    diff_map = {c: d for c, _, _, d, _ in diff_rows}

    # 相关性：取交集城市 + 有效数值（比值差异 与 EV比值）
    x_vals: List[float] = []
    y_vals: List[float] = []
    common_cities: List[str] = []
    order_vals: List[Optional[int]] = []
    for city, (nov_ratio, nov_leads) in nov.items():
        if args.min_leads and (nov_leads is None or nov_leads < args.min_leads):
            continue
        ev_ratio = ev.get(city)
        diff_val = diff_map.get(city)
        if diff_val is not None and ev_ratio is not None:
            x_vals.append(diff_val)
            y_vals.append(ev_ratio)
            common_cities.append(city)
            order_vals.append(nov_orders.get(city))

    pear = pearsonr(x_vals, y_vals) if x_vals else float("nan")
    spear = spearmanr(x_vals, y_vals) if x_vals else float("nan")

    # 导出配对CSV（如有要求）
    if args.pairs_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.pairs_output)), exist_ok=True)
        with open(args.pairs_output, "w", encoding="utf-8", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["城市", "后期比值", "EV比值", "比值差异(后-前)"])
            for c in common_cities:
                nov_ratio = nov.get(c, (None, None))[0]
                ev_ratio = ev.get(c)
                diff_val = diff_map.get(c)
                writer.writerow([
                    c,
                    "" if nov_ratio is None else f"{nov_ratio:.6f}",
                    "" if ev_ratio is None else f"{ev_ratio:.6f}",
                    "" if diff_val is None else f"{diff_val:.6f}"
                ])

    # 写Markdown
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# 城市效率比值差异与EV比值相关性分析\n\n")
        f.write(f"- 前期报告: `{args.sep_md}`\n")
        f.write(f"- 后期报告: `{args.nov_md}`\n")
        f.write(f"- EV比值: `{args.ev_csv}`\n")
        f.write(f"- 过滤条件: 后期线索识别数 >= `{args.min_leads}`\n\n")
        f.write("## 城市比值差异（后期 - 前期）\n")
        f.write("| 城市 | 前期比值 | 后期比值 | 差异(后-前) | 后期线索识别数 |\n")
        f.write("|:-----|---------:|---------:|------------:|---------------:|\n")
        for city, sep_ratio, nov_ratio, diff, nov_leads in diff_sorted:
            f.write(f"| {city} | "
                    f"{'' if sep_ratio is None else f'{sep_ratio:.4f}'} | "
                    f"{'' if nov_ratio is None else f'{nov_ratio:.4f}'} | "
                    f"{'' if diff is None else f'{diff:.4f}'} | "
                    f"{'' if nov_leads is None else nov_leads} |\n")

        f.write("\n## 相关性（比值差异 vs EV比值）\n")
        f.write(f"- Pearson r: `{pear:.4f}`\n")
        f.write(f"- Spearman ρ: `{spear:.4f}`\n")
        f.write(f"- 样本城市数: `{len(common_cities)}`\n")

        # 可选：打印前若干城市的匹配示意
        f.write("\n### 样本城市示意（前10）\n")
        f.write("| 城市 | 比值差异(后-前) | EV比值 |\n")
        f.write("|:-----|----------------:|-------:|\n")
        for i in range(min(10, len(common_cities))):
            c = common_cities[i]
            dv = diff_map.get(c)
            evv = ev.get(c)
            f.write(f"| {c} | {'' if dv is None else f'{dv:.4f}'} | {'' if evv is None else f'{evv:.4f}'} |\n")

    # 生成Plotly散点+LOWESS（如有要求）
    if args.plot_output:
        try:
            if PLOTLY_AVAILABLE and x_vals:
                x_arr = np.array(x_vals, dtype=float)
                y_arr = np.array(y_vals, dtype=float)
                xs, ys = lowess(x_arr, y_arr, frac=0.35)
                # 线性拟合
                try:
                    b, a = np.polyfit(x_arr, y_arr, 1)
                except Exception:
                    b, a = 0.0, float(np.nanmean(y_arr))

                fig = go.Figure()
                # 计算散点大小（按锁单数线性缩放）
                min_size, max_size = 6, 18
                # 提取有效orders用于缩放
                ords = [o for o in order_vals if (o is not None and np.isfinite(o))]
                if len(ords) >= 1:
                    o_min = min(ords)
                    o_max = max(ords)
                else:
                    o_min = o_max = None
                marker_sizes: List[float] = []
                for o in order_vals:
                    if o is None or o_min is None or o_max is None or o_max == o_min:
                        marker_sizes.append(float((min_size + max_size) / 2))
                    else:
                        s = min_size + (max_size - min_size) * (float(o - o_min) / float(o_max - o_min))
                        marker_sizes.append(float(s))
                # 散点（主层）
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, mode="markers",
                    marker=dict(color="#005783", size=marker_sizes, opacity=0.85),
                    name="城市样本",
                    hovertext=common_cities,
                    customdata=order_vals,
                    hovertemplate="城市:%{hovertext}<br>比值差异(后-前):%{x:.3f}<br>EV比值:%{y:.3f}<br>锁单数:%{customdata}<extra></extra>"
                ))
                # LOWESS
                fig.add_trace(go.Scatter(
                    x=xs.tolist(), y=ys.tolist(), mode="lines",
                    line=dict(color="#27AD00", width=3),
                    name="LOWESS"
                ))
                # 线性回归
                x_line = np.linspace(float(min(x_vals)), float(max(x_vals)), 100)
                y_line = a + b * x_line
                fig.add_trace(go.Scatter(
                    x=x_line.tolist(), y=y_line.tolist(), mode="lines",
                    line=dict(color="#A3ACB9", width=2, dash="dash"),
                    name="线性拟合"
                ))

                # 计算TOP标签索引
                metrics = []
                for i, c in enumerate(common_cities):
                    if args.label_mode == "efficiency":
                        m = x_arr[i]
                    elif args.label_mode == "ev":
                        m = y_arr[i]
                    else:  # diff_abs
                        # 使用差异绝对值；若无差异则赋-1保证不入选
                        m = abs(diff_map.get(c)) if diff_map.get(c) is not None else -1.0
                    metrics.append((m, i))
                # 按指标降序选择TOP N
                metrics_sorted = sorted(metrics, key=lambda t: t[0], reverse=True)
                top_n = max(0, min(args.label_top, len(metrics_sorted)))
                idx_top = [i for _, i in metrics_sorted[:top_n]]
                # 标签颜色按差异符号区分（正: 绿色，负: 蓝色）
                label_colors = []
                for i in idx_top:
                    c = common_cities[i]
                    d = diff_map.get(c)
                    if d is None:
                        label_colors.append("#7B848F")
                    elif d >= 0:
                        label_colors.append("#27AD00")
                    else:
                        label_colors.append("#005783")
                fig.add_trace(go.Scatter(
                    x=[x_vals[i] for i in idx_top],
                    y=[y_vals[i] for i in idx_top],
                    mode="markers+text",
                    marker=dict(color=label_colors, size=10, line=dict(color="#C8D0D9", width=1)),
                    text=[common_cities[i] for i in idx_top],
                    textposition="top center",
                    name=f"TOP城市（{args.label_mode}）"
                ))

                # 差异最小 TOP（按差异绝对值升序），避免与上面的重复
                if args.label_small_top and args.label_small_top > 0:
                    metrics_small: List[Tuple[float, int]] = []
                    for i, c in enumerate(common_cities):
                        d = diff_map.get(c)
                        m = abs(d) if d is not None else float("inf")
                        metrics_small.append((m, i))
                    metrics_small.sort(key=lambda t: t[0])
                    existing = set(idx_top)
                    idx_small: List[int] = []
                    for _, i in metrics_small:
                        if i not in existing:
                            idx_small.append(i)
                            if len(idx_small) >= args.label_small_top:
                                break
                    if idx_small:
                        fig.add_trace(go.Scatter(
                            x=[x_vals[i] for i in idx_small],
                            y=[y_vals[i] for i in idx_small],
                            mode="markers+text",
                            marker=dict(color="#7B848F", size=10, line=dict(color="#C8D0D9", width=1)),
                            text=[common_cities[i] for i in idx_small],
                            textposition="bottom center",
                            name="差异最小TOP"
                        ))

                title = f"比值差异 vs EV比值（N={len(x_vals)}，Pearson={pear:.3f}，Spearman={spear:.3f})"
                fig.update_layout(
                    title=title,
                    xaxis_title="比值差异(后-前)",
                    yaxis_title="增程/纯电比值",
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    xaxis=dict(gridcolor="#C8D0D9", zerolinecolor="#7B848F", linecolor="#7B848F"),
                    yaxis=dict(gridcolor="#C8D0D9", zerolinecolor="#7B848F", linecolor="#7B848F"),
                    legend=dict(bgcolor="#FFFFFF")
                )
                os.makedirs(os.path.dirname(os.path.abspath(args.plot_output)), exist_ok=True)
                fig.write_html(args.plot_output, include_plotlyjs="cdn")
                print(f"Plotly 图已生成：{args.plot_output}")
            else:
                print("未安装Plotly或无数据，跳过图形生成。")
        except Exception as e:
            print(f"生成Plotly图时出错：{e}")

    # 控制台提示
    print(f"差异与相关性分析已输出到: {args.out_md}")
    print(f"Pearson r = {pear:.4f}, Spearman rho = {spear:.4f}, N = {len(common_cities)}")

if __name__ == "__main__":
    main()
