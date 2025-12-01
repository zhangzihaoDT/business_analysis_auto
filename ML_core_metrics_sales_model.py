#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心指标销量贝叶斯建模脚本（PyMC）

用途：
- 使用 `processed/Core_Metrics_transposed.csv` 的日级核心指标数据，构建销量（锁单数）与多因素的贝叶斯线性模型
- 输出各因素影响的 β 后验分布、N 日销量预测（含 95% 可信区间）、Posterior Predictive Decomposition 因素贡献图（Plotly）

数据要求：
- 必须包含日期列（优先匹配 `日(日期)`），目标列（默认 `锁单数`），以及若干特征列（如 `小订数`、`有效线索数`、`下发线索数`、`有效试驾数`、`7 日内锁单线索数`、`试驾锁单数`、`抖音战队线索数`，若存在还将使用 `价格`、`开票价格`、`折扣`）
- 数值列可包含千分位逗号，脚本会自动清洗为数值类型

默认时间窗：
- 历史训练：2025-09-10 ～ 2025-10-15
- 新车型评估：2025-11-12 ～ 2025-11-27

模型说明：
- 形式：y ~ Normal(alpha + X·beta, sigma)
- 特征标准化：对特征进行 z-score（按训练段均值与标准差）
- 采样：NUTS，tune=1000，draws=2000，chains=2，target_accept=0.9
- 新车型段的后验预测：从后验样本抽取 (alpha, beta, sigma)，在新段特征上生成 y 的后验样本

输出文件：
- β 后验分布：`processed/core_metrics_sales_model_beta_posterior.csv`
- N 日销量预测：`processed/core_metrics_sales_model_forecast_<N>d.csv`
- 因素贡献图：`processed/core_metrics_sales_model_decomposition.html`

依赖：
- 需要 `pymc`、`arviz`、`plotly`（已在虚拟环境内安装可直接运行）

用法示例：
    python scripts/ML_core_metrics_sales_model.py \
        --hist-start 2025-09-10 --hist-end 2025-10-15 \
        --new-start 2025-11-12 --new-end 2025-11-27 \
        --forecast-days 14 --out-prefix core_metrics_sales_model

参数：
- `--input`      输入CSV路径，默认 `processed/Core_Metrics_transposed.csv`
- `--hist-start` 历史训练开始日期（YYYY-MM-DD）
- `--hist-end`   历史训练结束日期（YYYY-MM-DD）
- `--new-start`  新车型评估开始日期（YYYY-MM-DD）
- `--new-end`    新车型评估结束日期（YYYY-MM-DD）
- `--forecast-days` 预测天数 N（默认 14）
- `--out-prefix` 输出文件前缀（默认 `core_metrics_sales_model`）
"""
import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np

# 依赖库（pymc 与 arviz、plotly）
PYMC_AVAILABLE = True
ARVIZ_AVAILABLE = True
PLOTLY_AVAILABLE = True
try:
    import pymc as pm
except Exception:
    PYMC_AVAILABLE = False
try:
    import arviz as az
except Exception:
    ARVIZ_AVAILABLE = False
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    pio.templates.default = "plotly_white"
except Exception:
    PLOTLY_AVAILABLE = False

BASE_DIR = Path("/Users/zihao_/Documents/coding/dataset")
DEFAULT_INPUT = BASE_DIR / "processed" / "Core_Metrics_transposed.csv"
OUTPUT_DIR = BASE_DIR / "processed"


def parse_date_cn(s: str) -> datetime:
    s = str(s).strip()
    s = s.replace("年", "-").replace("月", "-").replace("日", "")
    return datetime.strptime(s, "%Y-%m-%d")


def select_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    date_col_candidates = ["日(日期)", "日期", "Date"]
    target_candidates = ["锁单数"]
    date_col = next((c for c in date_col_candidates if c in df.columns), None)
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if date_col is None:
        raise ValueError("未找到日期列（如 '日(日期)'）")
    if target_col is None:
        raise ValueError("未找到销量目标列（如 '锁单数'）")
    return {"date": date_col, "target": target_col}


def filter_by_range(df: pd.DataFrame, date_col: str, start: str, end: str) -> pd.DataFrame:
    dt = df[date_col].apply(parse_date_cn)
    mask = (dt >= datetime.strptime(start, "%Y-%m-%d")) & (dt <= datetime.strptime(end, "%Y-%m-%d"))
    out = df.loc[mask].copy()
    out["__date_dt"] = dt.loc[mask]
    return out


def to_numeric_clean(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(',', ''), errors='coerce')


def clean_numeric_block(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = to_numeric_clean(out[c])
    out = out.dropna(subset=cols)
    return out


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in [
        "7 日内锁单线索数",
        "有效线索数",
        "小订数",
        "小订退订数",
        "小订留存锁单数",
    ]:
        if c in out.columns:
            out[c] = to_numeric_clean(out[c])
    if "7 日内锁单线索数" in out.columns and "有效线索数" in out.columns:
        denom = out["有效线索数"].replace(0, np.nan)
        out["7 日线索转化率"] = (out["7 日内锁单线索数"] / denom).fillna(0.0)
    out = out.sort_values("__date_dt")
    if "小订数" in out.columns:
        out["累计小订数"] = out["小订数"].cumsum()
    if "小订退订数" in out.columns:
        out["累计小订退订数"] = out["小订退订数"].cumsum()
    if "累计小订数" in out.columns and "累计小订退订数" in out.columns:
        denom2 = out["累计小订数"].replace(0, np.nan)
        out["小订退订率"] = (out["累计小订退订数"] / denom2).fillna(0.0)
    if "累计小订数" in out.columns and "累计小订退订数" in out.columns and "小订留存锁单数" in out.columns:
        out["留存小订数"] = out["累计小订数"] - out["累计小订退订数"] - out["小订留存锁单数"].fillna(0.0)
    return out


def standardize_features(X: pd.DataFrame) -> (pd.DataFrame, pd.Series, pd.Series):
    mean = X.mean()
    std = X.std().replace(0, 1)
    Z = (X - mean) / std
    return Z, mean, std


def build_and_sample_model(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 2025):
    if not PYMC_AVAILABLE or not ARVIZ_AVAILABLE:
        raise RuntimeError("缺少pymc或arviz依赖，请安装: pip install pymc arviz")

    X_val = X_train.values
    y_val = y_train.values
    n_feat = X_val.shape[1]

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=n_feat)
        sigma = pm.HalfNormal("sigma", sigma=5.0)

        mu = alpha + pm.math.dot(X_val, beta)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_val)

        idata = pm.sample(2000, tune=1000, target_accept=0.9, random_seed=seed, chains=2)

    return model, idata


def draw_posterior_predictions(idata, X_new: pd.DataFrame) -> np.ndarray:
    Xn = X_new.values  # shape (N, F)
    post = idata.posterior
    alpha = post["alpha"].values  # (chains, draws)
    beta = post["beta"].values    # (chains, draws, F)
    sigma = post["sigma"].values  # (chains, draws)
    chains, draws = alpha.shape
    S = chains * draws
    F = beta.shape[-1]
    alpha = alpha.reshape(S)
    sigma = sigma.reshape(S)
    beta = beta.reshape(S, F)
    mu = alpha[:, None] + np.dot(beta, Xn.T)  # (S, N)
    eps = np.random.normal(loc=0.0, scale=sigma[:, None], size=mu.shape)
    y_samples = mu + eps
    return y_samples


def beta_summary(idata, feature_names: List[str]) -> pd.DataFrame:
    sel = az.summary(idata, var_names=["beta"], kind="stats", hdi_prob=0.95)
    df_sum = sel.copy()
    df_sum.index = feature_names
    df_sum.reset_index(inplace=True)
    df_sum.rename(columns={"index": "feature"}, inplace=True)
    return df_sum


def compute_contributions(idata, Z: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    posterior = idata.posterior
    beta_arr = posterior["beta"].values  # (chains, draws, F)
    beta_mean = beta_arr.mean(axis=(0, 1))  # (F,)
    contrib = Z.values * beta_mean
    df_contrib = pd.DataFrame(contrib, columns=feature_names, index=None)
    return df_contrib


def make_decomposition_plot(date_series: pd.Series, df_contrib: pd.DataFrame, y_pred_mean: np.ndarray, y_pred_hdi: np.ndarray, html_out: Path):
    if not PLOTLY_AVAILABLE:
        return None
    x = list(date_series.dt.strftime("%Y-%m-%d"))
    fig = go.Figure()
    for col in df_contrib.columns:
        fig.add_trace(go.Bar(x=x, y=df_contrib[col], name=col, opacity=0.85))
    fig.add_trace(go.Scatter(x=x, y=y_pred_mean, name="预测销量均值", mode="lines", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=x, y=y_pred_hdi[:, 0], name="下限95%CI", mode="lines", line=dict(color="gray", dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=y_pred_hdi[:, 1], name="上限95%CI", mode="lines", line=dict(color="gray", dash="dot")))
    fig.update_layout(barmode="stack", title="后验预测分解 (Posterior Predictive Decomposition)", xaxis_title="日期", yaxis_title="销量(锁单数标准化单位)")
    html_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_html(str(html_out), include_plotlyjs="cdn")
    except Exception:
        pass
    return fig


def generate_beta_report(beta_df: pd.DataFrame, html_out: Path, hist_range: str, new_range: str):
    if not PLOTLY_AVAILABLE:
        return None
    df = beta_df.copy()
    upper = df["hdi_97.5%"].values - df["mean"].values
    lower = df["mean"].values - df["hdi_2.5%"].values
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15, specs=[[{"type": "xy"}], [{"type": "table"}]], subplot_titles=("β 后验均值与95%区间", "参数表"))
    fig.add_trace(go.Bar(x=df["feature"], y=df["mean"], error_y=dict(type="data", array=upper, arrayminus=lower), name="β均值"), row=1, col=1)
    fig.add_trace(go.Table(header=dict(values=list(df.columns)), cells=dict(values=[df[c] for c in df.columns])), row=2, col=1)
    fig.update_layout(title=f"核心指标销量建模报告 | 历史:{hist_range} 新车型:{new_range}")
    html_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_html(str(html_out), include_plotlyjs="cdn")
    except Exception:
        pass
    return fig


def forecast_N_days(idata, feat_mean: pd.Series, feat_std: pd.Series, feature_names: List[str], last_date: datetime, N: int) -> pd.DataFrame:
    mu_alpha = float(idata.posterior["alpha"].values.mean())
    mu_beta = np.array(idata.posterior["beta"].values.mean(axis=(0, 1)))
    dates = [last_date + timedelta(days=i) for i in range(1, N + 1)]

    base_vals = feat_mean.copy()
    Z = (base_vals - feat_mean) / feat_std
    mu = float(mu_alpha + np.dot(Z.values, mu_beta))
    sigma = float(idata.posterior["sigma"].values.mean())

    y_mean = np.full(N, mu)
    y_lower = y_mean - 1.96 * sigma
    y_upper = y_mean + 1.96 * sigma

    out = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "pred_mean": y_mean,
        "pred_ci_lower": y_lower,
        "pred_ci_upper": y_upper,
    })
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="核心指标销量Bayesian建模 (PyMC)")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="输入数据集CSV路径")
    parser.add_argument("--hist-start", default="2025-08-15", help="历史数据开始日期 (YYYY-MM-DD)")
    parser.add_argument("--hist-end", default="2025-10-15", help="历史数据结束日期 (YYYY-MM-DD)")
    parser.add_argument("--new-start", default="2025-11-04", help="新车型开始日期 (YYYY-MM-DD)")
    parser.add_argument("--new-end", default="2025-11-27", help="新车型结束日期 (YYYY-MM-DD)")
    parser.add_argument("--forecast-days", type=int, default=14, help="预测天数 N")
    parser.add_argument("--out-prefix", default="core_metrics_sales_model", help="输出前缀名")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input)
    cols = select_columns(df)

    df_hist = filter_by_range(df, cols["date"], args.hist_start, args.hist_end)
    df_new = filter_by_range(df, cols["date"], args.new_start, args.new_end)
    df_hist = compute_derived_features(df_hist)
    df_new = compute_derived_features(df_new)

    feature_candidates = [
        "有效线索数",
        "7 日线索转化率",
        "小订留存锁单数",
        "大定数",
        "留存小订数",
        "小订退订率",
    ]
    X_cols = [c for c in feature_candidates if c in df_hist.columns]
    y_col = cols["target"]

    df_hist = clean_numeric_block(df_hist, X_cols + [y_col])
    X_train = df_hist[X_cols]
    y_train = df_hist[y_col]
    Z_train, feat_mean, feat_std = standardize_features(X_train)

    model, idata = build_and_sample_model(Z_train, y_train)

    df_new = clean_numeric_block(df_new, X_cols)
    Z_new = (df_new[X_cols] - feat_mean) / feat_std
    y_new_samples = draw_posterior_predictions(idata, Z_new)
    y_new_mean = y_new_samples.mean(axis=0)
    hdi = az.hdi(y_new_samples, hdi_prob=0.95)
    y_new_hdi = np.column_stack([hdi[:, 0], hdi[:, 1]])

    beta_df = beta_summary(idata, X_cols)
    beta_out = OUTPUT_DIR / "core_metrics_sales_model_beta_posterior.csv"
    beta_df.to_csv(beta_out, index=False)
    report_html = OUTPUT_DIR / "core_metrics_sales_model_report.html"
    hist_range = f"{args.hist_start}~{args.hist_end}"
    new_range = f"{args.new_start}~{args.new_end}"
    generate_beta_report(beta_df, report_html, hist_range, new_range)

    contrib_df = compute_contributions(idata, Z_new, X_cols)
    decomp_out = OUTPUT_DIR / "core_metrics_sales_model_newrange_decomposition.csv"
    decomp = pd.DataFrame({
        "date": df_new["__date_dt"].dt.strftime("%Y-%m-%d"),
        "pred_mean": y_new_mean,
        "pred_ci_lower": y_new_hdi[:, 0],
        "pred_ci_upper": y_new_hdi[:, 1],
    })
    for c in X_cols:
        decomp[c] = contrib_df[c].values
    decomp.to_csv(decomp_out, index=False)

    actual_new = to_numeric_clean(df_new.get("锁单数", pd.Series(index=df_new.index))).fillna(0).sum()
    cumulative_pred = float(np.sum(y_new_mean))

    print(f"β 后验分布已保存: {beta_out}")
    print(f"HTML 报告已生成: {report_html}")
    print(f"新车型段分解CSV: {decomp_out}")
    print(f"新车型段累计锁单数（实际）: {actual_new:.2f}")
    print(f"新车型段累计锁单数（预测均值）: {cumulative_pred:.2f}")
    print("锁单数与因素构成关系解释：系数为标准化特征对锁单数的边际影响，均值为方向与强度，区间反映不确定性。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
