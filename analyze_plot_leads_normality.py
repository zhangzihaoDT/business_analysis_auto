import argparse
import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def find_latest_leads_csv(base_dir: Path) -> Path:
    pattern = str(base_dir / "processed" / "leads_daily_*.csv")
    files = [Path(p) for p in glob.glob(pattern)]
    if not files:
        raise FileNotFoundError(f"未找到匹配文件: {pattern}")
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def load_daily_leads(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError("CSV 缺少 'date' 列")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        df = df.dropna(subset=["date"])  # 清理不可解析日期
    # 主要使用 '线索识别数' 作为每日线索量
    leads_col = "线索识别数"
    if leads_col not in df.columns:
        # 兜底：尝试可能的英文列名
        for alt in ["leads", "total_leads", "lead_count"]:
            if alt in df.columns:
                leads_col = alt
                break
        else:
            raise ValueError("CSV 缺少每日线索量列 '线索识别数' 或常见替代列")
    return df[["date", leads_col]].rename(columns={leads_col: "daily_leads"})


def compute_normality_metrics(x: np.ndarray) -> dict:
    metrics = {}
    x = np.asarray(x, dtype=float)
    metrics["n"] = int(x.size)
    metrics["mean"] = float(np.mean(x))
    metrics["std"] = float(np.std(x, ddof=1))
    # 使用 pandas 计算偏度与峰度，避免额外依赖
    s = pd.Series(x)
    metrics["skew"] = float(s.skew())
    metrics["kurtosis"] = float(s.kurt())  # Fisher 定义（正态为 0）
    # 如可用，加入 SciPy 的检验
    try:
        from scipy import stats  # type: ignore
        shapiro_stat, shapiro_p = stats.shapiro(x)
        metrics["shapiro_stat"] = float(shapiro_stat)
        metrics["shapiro_p"] = float(shapiro_p)
        k2_stat, k2_p = stats.normaltest(x)
        metrics["dagostino_k2"] = float(k2_stat)
        metrics["dagostino_p"] = float(k2_p)
        z = (x - metrics["mean"]) / (metrics["std"] if metrics["std"] else 1.0)
        ks_stat, ks_p = stats.kstest(z, "norm")
        metrics["ks_stat"] = float(ks_stat)
        metrics["ks_p"] = float(ks_p)
    except Exception:
        # 无 SciPy 时跳过 p 值检验
        pass
    return metrics


def make_figure(dates: pd.Series, x: np.ndarray, metrics: dict, start: str, end: str) -> go.Figure:
    title = f"{start}～{end} 日线索量正态性检验"
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12, subplot_titles=("直方图与正态曲线", "Q-Q 对比图（与标准正态）"))

    # 直方图
    bin_count = min(30, max(10, int(len(x) ** 0.5)))
    hist = go.Histogram(x=x, nbinsx=bin_count, name="日线索量", marker_color="#3b82f6", opacity=0.85)
    fig.add_trace(hist, row=1, col=1)

    # 正态曲线（按直方图面积缩放）
    bins = np.histogram_bin_edges(x, bins=bin_count)
    xs = np.linspace(np.min(x), np.max(x), 200)
    bin_width = (np.max(x) - np.min(x)) / (len(bins) - 1) if len(bins) > 1 else 1.0
    mu, sigma = metrics["mean"], metrics["std"] if metrics["std"] else 1.0
    pdf = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
    scaled_pdf = pdf * bin_width * len(x)
    fig.add_trace(go.Scatter(x=xs, y=scaled_pdf, name="正态拟合曲线", mode="lines", line=dict(color="#ef4444", width=2.5)), row=1, col=1)

    # Q-Q 图：与同规模的标准正态样本比较
    z = (x - mu) / (sigma if sigma else 1.0)
    z_sorted = np.sort(z)
    rng = np.random.default_rng(42)
    norm_sorted = np.sort(rng.normal(loc=0.0, scale=1.0, size=len(z_sorted)))
    fig.add_trace(go.Scatter(x=norm_sorted, y=z_sorted, mode="markers", name="样本 vs 正态", marker=dict(size=6, color="#10b981")), row=2, col=1)
    # 参考线（y=x）
    mn = float(min(z_sorted.min(), norm_sorted.min()))
    mx = float(max(z_sorted.max(), norm_sorted.max()))
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="参考线 y=x", line=dict(color="#6b7280", dash="dash")), row=2, col=1)

    # 注释文本
    lines = [
        f"样本量 N={metrics['n']}",
        f"均值={metrics['mean']:.1f}",
        f"标准差={metrics['std']:.1f}",
        f"偏度={metrics['skew']:.2f}",
        f"峰度={metrics['kurtosis']:.2f}",
    ]
    if "shapiro_p" in metrics:
        lines.append(f"Shapiro p={metrics['shapiro_p']:.4f}")
    if "dagostino_p" in metrics:
        lines.append(f"D'Agostino p={metrics['dagostino_p']:.4f}")
    if "ks_p" in metrics:
        lines.append(f"K-S p={metrics['ks_p']:.4f}")
    fig.add_annotation(
        text="<br>".join(lines),
        showarrow=False,
        xref="paper",
        yref="paper",
        x=1.0,
        y=1.0,
        xanchor="right",
        yanchor="top",
        bordercolor="#e5e7eb",
        borderwidth=1,
        bgcolor="#ffffff",
        font=dict(size=12),
    )

    fig.update_layout(
        title=title,
        bargap=0.05,
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="right", x=1.0),
        margin=dict(l=60, r=60, t=70, b=60),
    )
    fig.update_xaxes(title_text="日线索量", row=1, col=1)
    fig.update_yaxes(title_text="频数", row=1, col=1)
    fig.update_xaxes(title_text="标准正态分位", row=2, col=1)
    fig.update_yaxes(title_text="样本分位（标准化）", row=2, col=1)
    return fig


def main():
    parser = argparse.ArgumentParser(description="探索日线索量是否符合正态分布并可视化（Plotly）")
    parser.add_argument("--start", default="2025-09-10", help="开始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end", default="2025-11-18", help="结束日期，格式 YYYY-MM-DD")
    parser.add_argument("--out_html", default=None, help="输出 HTML 文件路径（默认写入 reports/ 下）")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    latest_csv = find_latest_leads_csv(root)
    df = load_daily_leads(latest_csv)

    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)
    mask = (df["date"] >= start_dt) & (df["date"] <= end_dt)
    df_range = df.loc[mask].sort_values("date")
    if df_range.empty:
        raise ValueError(f"在区间 {args.start}～{args.end} 内无数据")

    x = df_range["daily_leads"].astype(float).values
    metrics = compute_normality_metrics(x)
    fig = make_figure(df_range["date"], x, metrics, args.start, args.end)

    out_dir = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = args.out_html or out_dir / f"analyze_leads_normality_{args.start}_to_{args.end}.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"已生成可视化: {out_html}")
    print(f"使用数据: {latest_csv}")
    print(f"样本量 N={metrics['n']}, 均值={metrics['mean']:.1f}, 标准差={metrics['std']:.1f}, 偏度={metrics['skew']:.2f}, 峰度={metrics['kurtosis']:.2f}")
    if "shapiro_p" in metrics:
        print(f"Shapiro p={metrics['shapiro_p']:.4f}")
    if "dagostino_p" in metrics:
        print(f"D'Agostino p={metrics['dagostino_p']:.4f}")
    if "ks_p" in metrics:
        print(f"K-S p={metrics['ks_p']:.4f}")


if __name__ == "__main__":
    main()