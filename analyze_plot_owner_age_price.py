#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
import numpy as np

DEFAULT_INPUT = Path(
    "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"
)
OUT_DIR = Path(
    "/Users/zihao_/Documents/coding/dataset/processed/analysis_results"
)


def normalize(name: str) -> str:
    return str(name).strip().lower().replace("_", " ")


def resolve_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    cand_norm = [normalize(c) for c in candidates]
    col_norm_map = {normalize(c): c for c in df.columns}
    for cn in cand_norm:
        if cn in col_norm_map:
            return col_norm_map[cn]
    raise KeyError(f"Unable to resolve column: {candidates}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分车型的车主年龄 vs 开票价格散点图（Plotly）")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="输入 Parquet 文件路径",
    )
    parser.add_argument(
        "--out",
        default=str(OUT_DIR / "analyze_owner_age_price_scatter.html"),
        help="输出 HTML 文件路径",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="逗号分隔的车型列表，仅显示这些车型分类",
    )
    parser.add_argument(
        "--exclude-regions",
        default="FAC大区,虚拟大区",
        help="逗号分隔的需排除大区列表，默认剔除 FAC大区 和 虚拟大区",
    )
    return parser.parse_args()


def build_color_map(categories: List[str]) -> Dict[str, str]:
    base_map = {
        "LS9": "#27AD00",
        "CM2": "#005783",
        "CM2 增程": "#A3ACB9",
    }
    extra_palette = ["#C8D0D9", "#7B848F", "#27AD00", "#005783", "#A3ACB9"]
    used_colors = set()
    cmap: Dict[str, str] = {}
    for cat in categories:
        if cat in base_map:
            cmap[cat] = base_map[cat]
            used_colors.add(base_map[cat])
    for cat in categories:
        if cat not in cmap:
            for color in extra_palette:
                if color not in used_colors:
                    cmap[cat] = color
                    used_colors.add(color)
                    break
            if cat not in cmap:
                cmap[cat] = extra_palette[0]
    return cmap


def lowess(x: np.ndarray, y: np.ndarray, frac: float = 0.3, grid_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    优化的 LOWESS 实现：在网格点上计算平滑值，而非全量数据点，大幅提升大数据量下的性能。
    """
    if len(x) == 0:
        return np.array([]), np.array([])
    
    # 排序并去除无效值
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        return np.array([]), np.array([])
    
    # 大数据量降采样优化：如果数据点超过 2000，随机采样 2000 个点用于计算权重
    # 这能显著降低计算复杂度，同时对整体趋势影响较小
    if len(x) > 2000:
        indices = np.random.choice(len(x), 2000, replace=False)
        x = x[indices]
        y = y[indices]
        
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    
    x_range = x.max() - x.min()
    if x_range == 0:
        return x, np.repeat(np.nanmean(y), len(y))
        
    bandwidth = max(1e-8, frac * x_range)
    
    # 使用网格点进行计算，而非全量点
    if len(x) > grid_points:
        x_eval = np.linspace(x.min(), x.max(), grid_points)
    else:
        x_eval = x
        
    y_eval = np.empty_like(x_eval)
    
    # 预计算
    # 为加速计算，当数据量过大时（>5000），可以对原始数据也进行降采样用于权重计算（可选，此处暂保留全量权重计算以保证精度）
    # 如果依然慢，可以进一步对 x, y 进行降采样
    
    X = np.column_stack([np.ones_like(x), x])
    
    for i, xi in enumerate(x_eval):
        # 权重计算
        dist = np.abs(x - xi)
        # 仅考虑带宽内的数据点，加速计算
        window_mask = dist < bandwidth
        
        if not np.any(window_mask):
            y_eval[i] = np.nan
            continue
            
        dist_window = dist[window_mask] / bandwidth
        w_window = (1 - dist_window ** 3) ** 3
        
        # 局部回归
        X_window = X[window_mask]
        y_window = y[window_mask]
        W_window = np.diag(w_window)
        
        try:
            # 求解 (X^T W X) beta = X^T W y
            # 使用 pinv 或 solve，solve 更快但 pinv 更稳定
            XtW = X_window.T @ W_window
            beta = np.linalg.pinv(XtW @ X_window) @ (XtW @ y_window)
            y_eval[i] = beta[0] + beta[1] * xi
        except Exception:
            y_eval[i] = np.nan
            
    # 移除计算失败的点
    valid_eval = np.isfinite(y_eval)
    return x_eval[valid_eval], y_eval[valid_eval]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)

    model_group_col = resolve_column(
        df,
        ["车型分组", "Model Group", "Vehicle Group", "Car Group", "车型", "model_group"],
    )
    product_name_col = None
    try:
        product_name_col = resolve_column(
            df,
            ["productname", "ProductName", "Product Name", "产品名称", "商品名称"],
        )
    except Exception:
        product_name_col = None

    owner_age_col = resolve_column(
        df,
        ["owner_age", "Owner Age", "车主年龄", "年龄"],
    )
    price_col = resolve_column(
        df,
        ["开票价格", "Invoice Price", "invoice_price", "price", "开票 价格"],
    )

    # 解析 Parent Region Name 列
    try:
        region_col = resolve_column(
            df,
            ["Parent Region Name", "Parent_Region_Name", "Parent Region", "大区", "区域"]
        )
    except KeyError:
        region_col = None
        print("未找到区域列，将不进行分面展示")

    df["车型分类"] = df[model_group_col].astype(str)
    if product_name_col is not None:
        cm2_mask = df[model_group_col].astype(str).str.upper() == "CM2"
        is_range_ext = df[product_name_col].astype(str).str.contains(r"52|66", case=False, na=False)
        df.loc[cm2_mask & is_range_ext, "车型分类"] = "CM2 增程"
        df.loc[cm2_mask & ~is_range_ext, "车型分类"] = "CM2"

    wanted_models = [m.strip() for m in str(args.models).split(",") if m.strip()] if args.models else []
    model_filter = df["车型分类"].isin(wanted_models) if wanted_models else pd.Series(True, index=df.index)

    df_plot = df.loc[model_filter].copy()
    df_plot["x"] = pd.to_numeric(df_plot[owner_age_col], errors="coerce")
    df_plot["y"] = pd.to_numeric(df_plot[price_col], errors="coerce")
    
    # 价格单位转换
    valid_y = df_plot["y"].dropna()
    if not valid_y.empty and valid_y.median() > 1000:
        df_plot["y"] = df_plot["y"] / 10000.0
        
    # 筛选范围
    rng_mask = (df_plot["x"] >= 16) & (df_plot["x"] <= 85) & (df_plot["y"] >= 10) & (df_plot["y"] <= 50)
    df_plot = df_plot.loc[rng_mask]
    
    if df_plot.empty:
        raise ValueError("无有效数据用于绘图")

    cats = sorted(df_plot["车型分类"].unique().tolist())
    cmap = build_color_map(cats)
    
    # 获取区域列表
    if region_col:
        # 大区筛选
        exclude_list = [r.strip() for r in str(args.exclude_regions).split(",") if r.strip()]
        if exclude_list:
            exclude_mask = ~df_plot[region_col].astype(str).isin(exclude_list)
            df_plot = df_plot.loc[exclude_mask]
        
        if df_plot.empty:
            raise ValueError("排除指定大区后无有效数据")
            
        regions = sorted(df_plot[region_col].astype(str).unique().tolist())
    else:
        regions = ["All"]
        df_plot["All"] = "All"
        region_col = "All"

    # 计算子图布局行列
    n_regions = len(regions)
    n_cols = 3  # 每行3列
    n_rows = (n_regions + n_cols - 1) // n_cols
    
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols, 
        subplot_titles=regions,
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
        shared_yaxes=True  # 共享Y轴，确保所有子图Y轴范围一致
    )

    # 默认展示的车型列表
    default_visible_models = {"LS9", "CM2", "CM2 增程"}
    
    # 记录已显示图例的车型，避免重复或遗漏
    legend_shown = set()

    # 遍历每个区域绘制子图
    for i, region in enumerate(regions):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        df_reg = df_plot[df_plot[region_col].astype(str) == region]
        
        for cat in cats:
            mask = df_reg["车型分类"] == cat
            if not mask.any():
                continue
                
            x_data = df_reg.loc[mask, "x"]
            y_data = df_reg.loc[mask, "y"]
            
            # 判断该车型是否默认可见
            visible_status = True
            if default_visible_models and (cat not in default_visible_models):
                visible_status = "legendonly"
            
            # 仅在该车型首次出现时显示图例
            if cat not in legend_shown:
                show_legend = True
                legend_shown.add(cat)
            else:
                show_legend = False
            
            # 散点
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode="markers",
                    name=cat,
                    marker=dict(color=cmap[cat], size=5, opacity=0.3), # 缩小点大小适应多图
                    hovertemplate="区域=%{customdata}<br>车型=%{text}<br>年龄=%{x}<br>价格=%{y:.2f}<extra></extra>",
                    text=[cat] * len(x_data),
                    customdata=[region] * len(x_data),
                    visible=visible_status,
                    legendgroup=cat, # 同一车型归为一组
                    showlegend=show_legend
                ),
                row=row,
                col=col
            )
            
            # LOWESS
            xi = x_data.values
            yi = y_data.values
            xs, ys = lowess(xi, yi, frac=0.6) # 增加平滑度适应数据稀疏
            if len(xs) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        name=f"{cat} LOWESS",
                        line=dict(color=cmap[cat], width=2),
                        hovertemplate="区域=%{customdata}<br>车型=%{text}<br>年龄=%{x}<br>LOWESS=%{y:.2f}<extra></extra>",
                        text=[cat] * len(xs),
                        customdata=[region] * len(xs),
                        visible=visible_status,
                        legendgroup=cat, # 与散点同组，实现点击联动
                        showlegend=False # 不显示独立图例
                    ),
                    row=row,
                    col=col
                )

    fig.update_layout(
        title="分区域 x 分车型的车主年龄 vs 开票价格",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(bordercolor="#7B848F", borderwidth=1, font=dict(color="#7B848F")),
        margin=dict(l=40, r=40, t=80, b=40),
        height=300 * n_rows, # 动态调整高度
    )
    
    # 统一设置轴样式
    fig.update_xaxes(
        showline=True,
        linecolor="#7B848F",
        tickfont=dict(color="#7B848F", size=10),
        gridcolor="#ebedf0",
        zerolinecolor="#ebedf0",
        title_text="车主年龄",
        title_font=dict(size=10)
    )
    fig.update_yaxes(
        showline=True,
        linecolor="#7B848F",
        tickfont=dict(color="#7B848F", size=10),
        gridcolor="#ebedf0",
        zerolinecolor="#ebedf0",
        title_text="开票价格(万)",
        title_font=dict(size=10)
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Plot saved: {out_path}")
    print(f"Input: {input_path}")


if __name__ == "__main__":
    main()
