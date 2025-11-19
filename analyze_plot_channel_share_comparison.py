#!/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 尝试从 intention_lock_summary 复用默认输入路径
DEFAULT_INPUT = None
try:
    from scripts.intention_lock_summary import DEFAULT_INPUT as _DEFAULT_INPUT

    DEFAULT_INPUT = _DEFAULT_INPUT
except Exception:
    DEFAULT_INPUT = Path(
        "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"
    )


def normalize(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("  ", " ")


def resolve_column(df: pd.DataFrame, logical_name: str, candidates: Dict[str, List[str]]) -> str:
    cand_list = candidates.get(logical_name, [])
    # 优先精确匹配
    for cand in cand_list:
        if cand in df.columns:
            return cand
    # 退化到归一化匹配
    cand_norm = [normalize(c) for c in cand_list]
    col_norm_map = {normalize(c): c for c in df.columns}
    for cn in cand_norm:
        if cn in col_norm_map:
            return col_norm_map[cn]
    raise KeyError(f"无法解析列: {logical_name}; 备选: {cand_list}; 实际: {list(df.columns)}")


COL_CANDIDATES = {
    "lock_time": [
        "Lock Time",
        "Lock_Time",
        "日(Lock Time)",
        "锁单时间",
        "lock_time",
    ],
    "channel": [
        "first_main_channel_group",
    ],
    "model_group": [
        "车型分组",
        "model_group",
        "car_model_group",
        "车型",
    ],
}


def load_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    return df


def filter_locked_in_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    col_lock = resolve_column(df, "lock_time", COL_CANDIDATES)
    # 解析日期
    df = df.copy()
    df[col_lock] = pd.to_datetime(df[col_lock], errors="coerce")
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    # 锁单非空 且 在区间内（含边界）
    df = df[df[col_lock].notna()]
    df = df[(df[col_lock] >= start) & (df[col_lock] <= end)]
    return df


def aggregate_by_channel_model(df: pd.DataFrame) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    col_ch = resolve_column(df, "channel", COL_CANDIDATES)
    col_md = resolve_column(df, "model_group", COL_CANDIDATES)
    df_use = df.copy()

    def safe_fillna(series: pd.Series, fill_value: str) -> pd.Series:
        try:
            if pd.api.types.is_categorical_dtype(series):
                series = series.cat.add_categories([fill_value]).fillna(fill_value)
            else:
                series = series.astype(object).fillna(fill_value)
        except Exception:
            series = series.astype(object).fillna(fill_value)
        return series

    df_use[col_ch] = safe_fillna(df_use[col_ch], "未知")
    df_use[col_md] = safe_fillna(df_use[col_md], "其他")
    grouped = df_use.groupby([col_ch, col_md]).size().reset_index(name="count")
    channels = grouped[col_ch].drop_duplicates().tolist()
    # 构造: {model_group: {channel: count}}
    model_groups = grouped[col_md].drop_duplicates().tolist()
    matrix: Dict[str, Dict[str, int]] = {mg: {c: 0 for c in channels} for mg in model_groups}
    for _, row in grouped.iterrows():
        matrix[row[col_md]][row[col_ch]] = int(row["count"])
    return channels, matrix


def compute_channel_totals(matrix: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for mg, ch_map in matrix.items():
        for ch, cnt in ch_map.items():
            totals[ch] = totals.get(ch, 0) + int(cnt)
    return totals


def build_color_map(model_groups: List[str]) -> Dict[str, str]:
    other_palette = ["#A3ACB9", "#C8D0D9", "#7B848F"]
    color_map: Dict[str, str] = {}
    for mg in model_groups:
        key = str(mg).strip().upper()
        if key == "LS9":
            color_map[mg] = "#005783"
        elif key == "CM2":
            color_map[mg] = "#27AD00"
        else:
            # 其他车型分配随机色（循环三色）
            idx = len(color_map) % len(other_palette)
            color_map[mg] = other_palette[idx]
    return color_map


def make_period_comparison_chart(
    channels: List[str],
    matrix1: Dict[str, Dict[str, int]],
    matrix2: Dict[str, Dict[str, int]],
    start1: str,
    end1: str,
    start2: str,
    end2: str,
    model1: str,
    model2: str,
) -> go.Figure:
    # 排序：按第二周期总数降序
    totals2 = compute_channel_totals(matrix2)
    order = sorted(channels, key=lambda c: totals2.get(c, 0), reverse=True)

    # 模型组顺序：LS9，CM2，其余按字典序
    all_groups = set(list(matrix1.keys()) + list(matrix2.keys()))
    prefer = ["LS9", "CM2"]
    def key_norm(x):
        return str(x).strip().upper()
    others = sorted([g for g in all_groups if key_norm(g) not in {"LS9", "CM2"}])
    model_groups = prefer + others
    colors = build_color_map(model_groups)

    # 计算总数与占比（用于悬停）
    totals1 = compute_channel_totals(matrix1)
    total_sum1 = max(sum(totals1.values()), 1)
    share1 = {c: totals1.get(c, 0) / total_sum1 * 100.0 for c in order}

    total_sum2 = max(sum(totals2.values()), 1)
    share2 = {c: totals2.get(c, 0) / total_sum2 * 100.0 for c in order}

    # 差异（周期2 - 周期1）
    diff_share = {c: share2.get(c, 0.0) - share1.get(c, 0.0) for c in order}

    # 构建上下两行：上方为堆叠柱图，下方为对比表格
    fig = make_subplots(
        rows=3,
        cols=1,
        # 保证每行高度不低于 ~400px：结合整体高度配置
        row_heights=[0.40, 0.40, 0.40],
        vertical_spacing=0.08,
        specs=[[{"type": "xy"}], [{"type": "table"}], [{"type": "xy"}]],
        subplot_titles=(
            f"分渠道锁单占比（{start1} 至 {end1} vs {start2} 至 {end2}）",
            "分渠道锁单数/占比/占比差异（周期2-周期1）",
            f"分渠道订单占比：{model1}（{start1} 至 {end1}） vs {model2}（{start2} 至 {end2}）",
        ),
    )

    # 周期1堆叠（Y 轴为占比：各车型对全周期总数的百分比，按渠道堆叠相加为渠道占比）
    for mg in model_groups:
        y1 = [matrix1.get(mg, {}).get(c, 0) / total_sum1 * 100.0 for c in order]
        fig.add_bar(
            name=f"{mg}（{start1} 至 {end1}）",
            x=order,
            y=y1,
            marker_color=colors.get(mg, "#7B848F"),
            offsetgroup="P1",
            legendgroup=f"P1-{mg}",
            meta=mg,
            hovertemplate=(
                "渠道=%{x}<br>车型=%{meta}<br>占比(车型对周期总)=%{y:.2f}%<br>渠道占比(周期总)=%{customdata[0]:.2f}%<br>占比差异(周期2-1)=%{customdata[1]:+.2f}%<extra></extra>"
            ),
            customdata=[[share1[c], diff_share[c]] for c in order],
            row=1,
            col=1,
        )

    # 周期2堆叠（同理为占比）
    for mg in model_groups:
        y2 = [matrix2.get(mg, {}).get(c, 0) / total_sum2 * 100.0 for c in order]
        fig.add_bar(
            name=f"{mg}（{start2} 至 {end2}）",
            x=order,
            y=y2,
            marker_color=colors.get(mg, "#7B848F"),
            offsetgroup="P2",
            legendgroup=f"P2-{mg}",
            meta=mg,
            hovertemplate=(
                "渠道=%{x}<br>车型=%{meta}<br>占比(车型对周期总)=%{y:.2f}%<br>渠道占比(周期总)=%{customdata[0]:.2f}%<br>占比差异(周期2-1)=%{customdata[1]:+.2f}%<extra></extra>"
            ),
            customdata=[[share2[c], diff_share[c]] for c in order],
            row=1,
            col=1,
        )

    # 下方表格：对比两个周期的分渠道锁单数、分渠道比例和比例差异
    counts1 = [int(totals1.get(c, 0)) for c in order]
    counts2 = [int(totals2.get(c, 0)) for c in order]
    shares1 = [share1.get(c, 0.0) for c in order]
    shares2 = [share2.get(c, 0.0) for c in order]
    diffs = [shares2[i] - shares1[i] for i in range(len(order))]

    # 合计行
    total_count1 = int(sum(counts1))
    total_count2 = int(sum(counts2))

    header_vals = [
        "渠道",
        f"数量（{start1} 至 {end1}）",
        f"占比%（{start1} 至 {end1}）",
        f"数量（{start2} 至 {end2}）",
        f"占比%（{start2} 至 {end2}）",
        "占比差异%（周期2-周期1）",
    ]
    cells_channels = order + ["总计"]
    cells_counts1 = counts1 + [total_count1]
    cells_shares1 = [f"{p:.2f}%" for p in shares1] + ["100.00%"]
    cells_counts2 = counts2 + [total_count2]
    cells_shares2 = [f"{p:.2f}%" for p in shares2] + ["100.00%"]
    cells_diffs = [f"{d:+.2f}%" for d in diffs] + ["+0.00%"]

    fig.add_table(
        header=dict(values=header_vals, fill_color="#f0f0f0", align="center", font=dict(color="#333", size=12), height=30),
        cells=dict(
            values=[
                cells_channels,
                cells_counts1,
                cells_shares1,
                cells_counts2,
                cells_shares2,
                cells_diffs,
            ],
            align="center",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        barmode="relative",
        title=f"分渠道锁单占比对比：{start1} 至 {end1} vs {start2} 至 {end2}",
        xaxis_title="渠道",
        yaxis_title="锁单占比（%）",
        legend_title="车型分组 × 周期",
        margin=dict(l=40, r=40, t=60, b=80),
        # 设置整体高度，结合 row_heights，确保每行不低于 ~400px
        height=1200,
    )
    
    # 追加对比柱状图（第三行）：周期1指定车型 vs 周期2指定车型
    # 单车型归一化：各渠道该车型占比 = 该车型在该渠道的锁单数 / 该车型在该周期的总锁单数 * 100
    model_total1 = max(sum(matrix1.get(model1, {}).values()), 1)
    model_total2 = max(sum(matrix2.get(model2, {}).values()), 1)
    y1_sel = [matrix1.get(model1, {}).get(c, 0) / model_total1 * 100.0 for c in order]
    y2_sel = [matrix2.get(model2, {}).get(c, 0) / model_total2 * 100.0 for c in order]

    fig.add_bar(
        name=f"{model1}（{start1} 至 {end1}）",
        x=order,
        y=y1_sel,
        marker_color=colors.get(model1, "#7B848F"),
        offsetgroup="M1",
        legendgroup=f"M1-{model1}",
        meta=model1,
        hovertemplate=(
            "渠道=%{x}<br>车型=%{meta}<br>占比(该车型在该周期内)=%{y:.2f}%<extra></extra>"
        ),
        row=3,
        col=1,
    )

    fig.add_bar(
        name=f"{model2}（{start2} 至 {end2}）",
        x=order,
        y=y2_sel,
        marker_color=colors.get(model2, "#7B848F"),
        offsetgroup="M2",
        legendgroup=f"M2-{model2}",
        meta=model2,
        hovertemplate=(
            "渠道=%{x}<br>车型=%{meta}<br>占比(该车型在该周期内)=%{y:.2f}%<extra></extra>"
        ),
        row=3,
        col=1,
    )

    # 第三行设置分组显示
    fig.update_layout(
        barmode="relative",
    )
    return fig


def main():
    ap = argparse.ArgumentParser(
        description="读取锁单数据（Parquet），按两个周期输出分渠道的锁单数/占比对比柱状图（柱内区分车型分组）"
    )
    ap.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="输入 Parquet 路径（默认读取 intention_lock_summary.DEFAULT_INPUT）",
    )
    ap.add_argument("--start1", required=True, help="周期1开始日期，格式YYYY-MM-DD")
    ap.add_argument("--end1", required=True, help="周期1结束日期，格式YYYY-MM-DD")
    ap.add_argument("--start2", required=True, help="周期2开始日期，格式YYYY-MM-DD")
    ap.add_argument("--end2", required=True, help="周期2结束日期，格式YYYY-MM-DD")
    # 车型参数：支持 --model1/--model2；并提供便捷开关 --CM2、--LS9（与示例保持一致）
    ap.add_argument("--model1", default=None, help="周期1车型分组（如 LS9/CM2）")
    ap.add_argument("--model2", default=None, help="周期2车型分组（如 LS9/CM2）")
    ap.add_argument("--CM2", action="store_true", help="快捷指定周期1车型为 CM2（若未显式指定 --model1）")
    ap.add_argument("--LS9", action="store_true", help="快捷指定周期2车型为 LS9（若未显式指定 --model2）")
    ap.add_argument(
        "--out",
        default=None,
        help="输出HTML路径，默认写入 reports/分渠道锁单数_含车型分组_对比_<period1>_vs_<period2>.html",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    df = load_data(input_path)

    # 构造两个周期的锁单样本
    df1 = filter_locked_in_range(df, args.start1, args.end1)
    df2 = filter_locked_in_range(df, args.start2, args.end2)

    # 聚合：分渠道 × 车型分组 计数
    ch1, mat1 = aggregate_by_channel_model(df1)
    ch2, mat2 = aggregate_by_channel_model(df2)
    # 渠道全集（并集）
    channels = sorted(set(ch1 + ch2))

    # 解析车型参数
    model1 = args.model1
    model2 = args.model2
    if args.CM2 and not model1:
        model1 = "CM2"
    if args.LS9 and not model2:
        model2 = "LS9"
    if not model1 or not model2:
        raise SystemExit("请通过 --model1 与 --model2（或 --CM2 / --LS9）指定周期对应的车型分组")

    fig = make_period_comparison_chart(
        channels=channels,
        matrix1=mat1,
        matrix2=mat2,
        start1=args.start1,
        end1=args.end1,
        start2=args.start2,
        end2=args.end2,
        model1=model1,
        model2=model2,
    )

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("reports") / (
            f"分渠道锁单数_含车型分组_对比_{args.start1}_to_{args.end1}_vs_{args.start2}_to_{args.end2}.html"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()