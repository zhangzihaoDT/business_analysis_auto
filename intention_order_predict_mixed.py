#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意向订单每日计数与双峰混合预测（基于 intention_order_analysis.parquet）

功能：
- 读取数据（默认路径：../formatted/intention_order_analysis.parquet）
- 部分1：按车型分组与意向支付时间范围筛选，输出每天（Intention Payment Time）的订单数
- 部分2：使用 CM1 与 CM2 两段样本做训练，构建「BSTS + Boosting」混合模型，预测新车型在累计 X 天周期内第4～X日每日订单数
- 部分3：使用 Plotly 可视化历史曲线与预测曲线，输出 HTML 文件
- 增强：自动读取 processed/ 下最新的 leads_daily_*.csv，提取“线索识别数”并按 date 合并到每日表，生成线索滞后与滚动特征（leads_lag_1/2/3，leads_cum_7），用于提升预测能力。

示例：
python intention_order_daily_counts.py -m CM2 --start-date 2025-08-15 --end-date 2025-09-10 \
  --train-cm1-start 2024-08-30 --train-cm1-end 2024-09-26 \
  --train-cm2-start 2025-08-15 --train-cm2-end 2025-09-10 \
  --new-model LS9 --new-start-date 2025-11-04 --new-end-date 2025-11-06 --cycle-days 9
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

# 尝试导入XGBoost，失败则回退到sklearn的梯度提升
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor
    XGB_AVAILABLE = False

# Plotly 可视化
try:
    import plotly.graph_objects as go
    from plotly.offline import plot as plotly_save
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


# 默认数据路径（相对于 scripts 目录）
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "formatted", "intention_order_analysis.parquet")


def normalize_col(name: str) -> str:
    """标准化列名用于匹配（去空格、下划线统一小写）。"""
    return name.strip().lower().replace(" ", "_")


def resolve_column(df: pd.DataFrame, key: str, candidates_map: Dict[str, List[str]]) -> str:
    """根据候选集合在 DataFrame 中解析出真实列名。"""
    norm_to_actual = {normalize_col(c): c for c in df.columns}
    for cand in candidates_map.get(key, []):
        norm_cand = normalize_col(cand)
        if norm_cand in norm_to_actual:
            return norm_to_actual[norm_cand]
    raise KeyError(f"未在数据集中找到需要的字段: {key}. 可用字段: {list(df.columns)}")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    return pd.read_parquet(path)


# === Leads 日级数据加载与合并 ===

def find_latest_leads_daily(processed_dir: Optional[str] = None, pattern: str = 'leads_daily_*.csv') -> Optional[str]:
    """查找 processed 目录下最新的 leads_daily_*.csv 文件。"""
    base = Path(processed_dir) if processed_dir is not None else (Path(__file__).resolve().parents[1] / 'processed')
    candidates = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    return str(candidates[0])


def load_leads_daily(path: str) -> pd.DataFrame:
    """读取 leads_daily CSV，并解析 date 与线索识别数列，输出两列：date, leads_recognition_count。"""
    df = pd.read_csv(path)
    # 解析日期列
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        # 兜底：尝试从可能的日期列名解析
        for cand in ['日期', 'day', 'dt']:
            if cand in df.columns:
                df['date'] = pd.to_datetime(df[cand], errors='coerce')
                break
        if 'date' not in df.columns:
            raise KeyError('leads_daily 文件缺少日期列')

    # 识别“线索识别数”列名候选
    lead_candidates = [
        '线索识别数', '识别数', 'leads_count', '线索数', 'lead_count', 'leads', 'count', '日识别数'
    ]
    leads_col = None
    for c in lead_candidates:
        if c in df.columns:
            leads_col = c
            break
    if leads_col is None:
        # 若没有明确列名，选择第一个数值列作为近似（除 date）
        num_cols = [c for c in df.columns if c != 'date']
        # 尝试转数值
        num_cols_numeric = []
        for c in num_cols:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                num_cols_numeric.append(c)
            except Exception:
                pass
        leads_col = num_cols_numeric[0] if num_cols_numeric else None

    if leads_col is None:
        df['leads_recognition_count'] = 0.0
    else:
        df['leads_recognition_count'] = pd.to_numeric(df[leads_col], errors='coerce').fillna(0.0)

    return df[['date', 'leads_recognition_count']].dropna(subset=['date'])


def merge_leads_features(daily_df: pd.DataFrame, leads_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """按日期合并线索识别数到每日数据，缺失填 0。"""
    df = daily_df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if leads_df is not None and not leads_df.empty:
        ldf = leads_df.copy()
        ldf['date'] = pd.to_datetime(ldf['date'], errors='coerce')
        df = df.merge(ldf, on='date', how='left')
    if 'leads_recognition_count' not in df.columns:
        df['leads_recognition_count'] = 0.0
    df['leads_recognition_count'] = df['leads_recognition_count'].fillna(0.0)
    return df


def filter_data(
    df: pd.DataFrame,
    model_group_value: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    candidates = {
        "model_group": [
            "车型分组",
            "model_group",
            "car_model_group",
            "车型",
            "modelgroup",
        ],
        "intention_payment_time": [
            "Intention Payment Time",
            "Intention_Payment_Time",
            "意向支付时间",
            "intention_payment_time",
            "intent_payment_time",
            "intentpaytime",
        ],
    }

    col_model = resolve_column(df, "model_group", candidates)
    col_time = resolve_column(df, "intention_payment_time", candidates)

    # 车型分组筛选
    df = df[df[col_model].astype(str) == str(model_group_value)].copy()

    # 时间筛选（包含边界）
    df[col_time] = pd.to_datetime(df[col_time], errors="coerce")
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(df[col_time] >= start) & (df[col_time] <= end)].copy()

    return df, col_time


def compute_daily_counts(df: pd.DataFrame, col_time: str) -> pd.DataFrame:
    """按天统计订单数量（基于意向支付时间）。"""
    s_time = pd.to_datetime(df[col_time], errors="coerce")
    day = s_time.dt.floor("D")
    res = pd.DataFrame({"day": day}).dropna().groupby("day", dropna=False).size().reset_index(name="orders")
    res = res.sort_values("day")
    return res


def compute_daily_counts_for_group(
    df: pd.DataFrame, group: str, start_date: str, end_date: str,
    leads_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """给定车型分组与日期范围，返回每日订单数，并按日期合并线索识别数，生成 cycle_day 与 model_group 标识。"""
    df_filt, col_time = filter_data(df, group, start_date, end_date)
    daily = compute_daily_counts(df_filt, col_time)
    daily = daily.rename(columns={"day": "date", "orders": "count"})
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["cycle_day"] = np.arange(1, len(daily) + 1)
    daily["model_group"] = group
    # 合并线索识别数
    daily = merge_leads_features(daily, leads_df)
    # 数值编码用于混合（CM1->0, CM2->1）
    code = 1 if group.strip().upper() == "CM2" else 0
    daily["model_group_code"] = float(code)
    return daily


def add_time_features(daily: pd.DataFrame) -> pd.DataFrame:
    """为每日数据添加时间相关特征与滞后/累计特征。"""
    df = daily.copy()
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    # 滞后特征
    df["lag_1"] = df["count"].shift(1)
    df["lag_2"] = df["count"].shift(2)
    df["lag_3"] = df["count"].shift(3)
    # 7日累计（含当日）
    df["cum_7"] = df["count"].rolling(window=7, min_periods=1).sum()
    # 线索识别数特征（滞后与滚动）
    if "leads_recognition_count" not in df.columns:
        df["leads_recognition_count"] = 0.0
    df["leads_lag_1"] = df["leads_recognition_count"].shift(1)
    df["leads_lag_2"] = df["leads_recognition_count"].shift(2)
    df["leads_lag_3"] = df["leads_recognition_count"].shift(3)
    df["leads_cum_7"] = df["leads_recognition_count"].rolling(window=7, min_periods=1).sum()
    # 归一化相位与截止天特征（用于刻画双峰尾部上扬）
    seg_len = int(df["cycle_day"].max()) if len(df) else 1
    if seg_len <= 1:
        df["phase"] = 0.0
        df["days_to_deadline_norm"] = 1.0
    else:
        df["phase"] = (df["cycle_day"] - 1) / float(seg_len - 1)
        df["days_to_deadline_norm"] = 1.0 - df["phase"]
    return df


def build_training_features(segments: List[pd.DataFrame]) -> pd.DataFrame:
    """将多个训练段（CM1/CM2）拼接为特征数据集，过滤掉前3日缺滞后的样本。"""
    feats = []
    for seg in segments:
        df = add_time_features(seg)
        # 去除缺少 lag_1~lag_3 的样本（>=第4日）
        df = df.loc[df["cycle_day"] >= 4].copy()
        feats.append(df)
    Xy = pd.concat(feats, ignore_index=True)
    # 目标
    Xy["y"] = Xy["count"].astype(float)
    return Xy


def build_normalized_baseline(segments: List[pd.DataFrame], new_length: int) -> List[float]:
    """在归一化相位空间构造双峰混合基线，并重采样到 new_length。

    步骤：
    - 对每个训练段，按其长度将 (cycle_day, count) 映射为相位 curve（0～1）。
    - 在统一网格上线性插值，得到每段的曲线。
    - 对所有段取均值，得到总体双峰形状。
    - 重采样到目标长度 new_length，使尾部峰值在任意周期长度下都靠近周期末尾。
    """
    if new_length <= 0:
        return []
    target_phase = np.linspace(0.0, 1.0, new_length)
    curves = []
    for seg in segments:
        seg = seg.sort_values("cycle_day").copy()
        L = int(seg["cycle_day"].max()) if len(seg) else 0
        if L <= 1:
            # 退化：平坦曲线
            curves.append(np.zeros(new_length))
            continue
        src_phase = np.linspace(0.0, 1.0, L)
        src_vals = seg["count"].astype(float).values
        # 线性插值到目标相位
        curve = np.interp(target_phase, src_phase, src_vals)
        curves.append(curve)
    if not curves:
        return [0.0] * new_length
    baseline = np.mean(np.vstack(curves), axis=0)
    return baseline.tolist()


def make_baseline_for_cycle_days(norm_curve: List[float], cycle_days: int) -> List[float]:
    """将归一化基线曲线重采样到指定周期天数。"""
    if cycle_days <= 0:
        return []
    if not norm_curve:
        return [0.0] * cycle_days
    src_phase = np.linspace(0.0, 1.0, len(norm_curve))
    tgt_phase = np.linspace(0.0, 1.0, cycle_days)
    curve = np.interp(tgt_phase, src_phase, np.array(norm_curve))
    return curve.tolist()


def orchestrate_training(segments: List[pd.DataFrame]):
    """组织训练：构造归一化双峰基线 + 拟合Boosting（残差）。"""
    # 归一化双峰基线（用较细网格拟合残差）
    norm_curve = build_normalized_baseline(segments, new_length=100)
    grid_phase = np.linspace(0.0, 1.0, 100)

    # 构造训练特征
    feats = []
    for seg in segments:
        df = add_time_features(seg)
        # 仅使用第4日起的样本（确保lag特征齐备）
        df = df.loc[df["cycle_day"] >= 4].copy()
        feats.append(df)
    Xy = pd.concat(feats, ignore_index=True)

    # 将每行的基线值按相位插值映射
    Xy["baseline"] = np.interp(Xy["phase"].astype(float), grid_phase, np.array(norm_curve))
    Xy["y"] = Xy["count"].astype(float)
    Xy["residual"] = Xy["y"] - Xy["baseline"]

    # 选择特征列（增加相位与截止天归一化，帮助尾部上扬）
    feat_cols = [
        "lag_1", "lag_2", "lag_3", "cum_7", "day_of_week", "is_weekend",
        "model_group_code", "phase", "days_to_deadline_norm",
        # 线索识别数的滞后与滚动特征
        "leads_lag_1", "leads_lag_2", "leads_lag_3", "leads_cum_7",
    ]
    X = Xy[feat_cols].astype(float).fillna(0.0)
    y = Xy["residual"].astype(float)

    # 训练 Boosting
    if XGB_AVAILABLE:
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
        )
        model.fit(X, y)
    else:
        model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.08,
            random_state=42,
        )
        model.fit(X, y)

    return norm_curve, model, feat_cols


def predict_new_series(
    model, feat_cols: List[str], baseline: List[float],
    new_start_date: str, new_end_date: str, cycle_days: int,
    initial_daily: pd.DataFrame,
) -> pd.DataFrame:
    """根据前1~3日的真实订单数，递归预测第4～X日每日订单。"""
    # 准备时间线
    start_dt = pd.to_datetime(new_start_date)
    dates = [start_dt + pd.Timedelta(days=i) for i in range(cycle_days)]
    # 初始已知：根据输入数据获取第1～3日的订单数
    init_map = {pd.to_datetime(r["date"]).date(): float(r["count"]) for _, r in initial_daily.iterrows()}
    init_leads_map = {pd.to_datetime(r["date"]).date(): float(r.get("leads_recognition_count", 0.0)) for _, r in initial_daily.iterrows()}
    counts = []
    leads_counts = []
    for i, d in enumerate(dates):
        if i < len(initial_daily):
            counts.append(init_map.get(d.date(), 0.0))
            leads_counts.append(init_leads_map.get(d.date(), 0.0))
        else:
            counts.append(np.nan)  # 待预测
            leads_counts.append(np.nan)

    # 递归预测从第4日开始
    for i in range(cycle_days):
        cycle_day = i + 1
        if not np.isnan(counts[i]):
            continue  # 已知天

        # 构造特征
        lag_1 = counts[i - 1] if i - 1 >= 0 else 0.0
        lag_2 = counts[i - 2] if i - 2 >= 0 else 0.0
        lag_3 = counts[i - 3] if i - 3 >= 0 else 0.0
        # cum_7：最近7天累计（含当日，但当日未知，用基线填充0再由残差修正）
        window_start = max(0, i - 6)
        past_vals = [c for c in counts[window_start:i] if not np.isnan(c)]
        cum_7 = float(np.sum(past_vals))
        dow = int(dates[i].weekday())
        is_weekend = 1 if dow in (5, 6) else 0
        # 新车型的编码置为 0.5，用以表达CM1/CM2的混合经验
        model_group_code = 0.5
        # 相位与截止天（归一化）
        if cycle_days <= 1:
            phase = 0.0
            dtd_norm = 1.0
        else:
            phase = i / float(cycle_days - 1)
            dtd_norm = 1.0 - phase

        # 线索识别数特征（使用已知的前几日并前向填充缺失）
        def lead_val(idx: int) -> float:
            if idx < 0:
                return 0.0
            v = leads_counts[idx]
            return 0.0 if np.isnan(v) else float(v)

        leads_lag_1 = lead_val(i - 1)
        leads_lag_2 = lead_val(i - 2)
        leads_lag_3 = lead_val(i - 3)
        l_window_start = max(0, i - 6)
        l_past_vals = [lead_val(k) for k in range(l_window_start, i)]
        leads_cum_7 = float(np.sum(l_past_vals))

        X_row = pd.DataFrame({
            "lag_1": [lag_1],
            "lag_2": [lag_2],
            "lag_3": [lag_3],
            "cum_7": [cum_7],
            "day_of_week": [dow],
            "is_weekend": [is_weekend],
            "model_group_code": [model_group_code],
            "phase": [phase],
            "days_to_deadline_norm": [dtd_norm],
            "leads_lag_1": [leads_lag_1],
            "leads_lag_2": [leads_lag_2],
            "leads_lag_3": [leads_lag_3],
            "leads_cum_7": [leads_cum_7],
        })
        X_row = X_row[feat_cols].astype(float)
        residual_pred = float(model.predict(X_row)[0])

        base = baseline[i] if i < len(baseline) else 0.0
        pred = max(0.0, base + residual_pred)
        counts[i] = pred

    pred_df = pd.DataFrame({
        "date": dates,
        "cycle_day": np.arange(1, cycle_days + 1),
        "pred_count": counts,
        "baseline": [baseline[i] if i < len(baseline) else 0.0 for i in range(cycle_days)],
    })
    return pred_df


def generate_plot(
    segments: List[pd.DataFrame], pred_df: pd.DataFrame, new_model: str,
    html_path: str
):
    """生成历史+预测曲线的Plotly图并保存为HTML。"""
    if not PLOTLY_AVAILABLE:
        return None
    fig = go.Figure()
    # 历史段
    for seg in segments:
        fig.add_trace(go.Scatter(
            x=seg["date"], y=seg["count"],
            mode="lines+markers",
            name=f"历史-{seg['model_group'].iloc[0]}",
        ))
    # 预测段
    fig.add_trace(go.Scatter(
        x=pred_df["date"], y=pred_df["pred_count"],
        mode="lines+markers",
        name=f"预测-{new_model}",
        line=dict(color="firebrick")
    ))
    fig.add_trace(go.Scatter(
        x=pred_df["date"], y=pred_df["baseline"],
        mode="lines",
        name="基线",
        line=dict(dash="dash", color="gray")
    ))
    fig.update_layout(
        title=f"意向订单双峰混合预测（{new_model}）",
        xaxis_title="日期",
        yaxis_title="每日订单数",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2)
    )
    plotly_save(fig, filename=html_path, auto_open=False)
    return html_path


def ensure_reports_dir(root_dir: str) -> str:
    reports_dir = os.path.join(root_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir


def generate_report(
    df_daily: pd.DataFrame, model_group: str, start_date: str, end_date: str,
    leads_df: Optional[pd.DataFrame] = None,
) -> str:
    lines: List[str] = []
    lines.append(f"意向订单每日计数（车型分组={model_group}，时间范围={start_date} 至 {end_date}）")
    total = int(df_daily["orders"].sum()) if len(df_daily) else 0
    lines.append(f"总订单数：{total}")
    lines.append("")
    lines.append("【每日订单数（按意向支付时间）】")

    # 合并线索识别数，并在每日列表中追加显示
    daily_for_merge = df_daily.rename(columns={"day": "date", "orders": "count"}).copy()
    daily_for_merge = merge_leads_features(daily_for_merge, leads_df)

    for _, row in daily_for_merge.iterrows():
        day_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        orders_val = int(row.get("count", row.get("orders", 0)))
        leads_val = int(row.get("leads_recognition_count", 0))
        lines.append(f"- {day_str}: {orders_val}（线索识别数={leads_val}）")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="意向订单每日计数与双峰混合预测")
    # 部分1：每日计数
    parser.add_argument("-m", "--model-group", default="CM2", help="每日计数用的车型分组（默认 CM2）")
    parser.add_argument("--start-date", default="2025-08-15", help="每日计数开始日期（默认 2025-08-15）")
    parser.add_argument("--end-date", default="2025-09-10", help="每日计数结束日期（默认 2025-09-10）")
    # 通用数据路径
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help=f"数据文件路径（默认：{DEFAULT_DATA_PATH}）")
    parser.add_argument("-o", "--output", default=None, help="输出简报文件路径（默认写入到项目 reports 目录）")
    # 部分2：训练段（CM1/CM2）
    parser.add_argument("--train-cm1-start", default="2024-08-30")
    parser.add_argument("--train-cm1-end", default="2024-09-26")
    parser.add_argument("--train-cm2-start", default="2025-08-15")
    parser.add_argument("--train-cm2-end", default="2025-09-10")
    # 新车型预测参数
    parser.add_argument("--new-model", default="LS9")
    parser.add_argument("--new-start-date", default="2025-11-04")
    parser.add_argument("--new-end-date", default="2025-11-06")
    parser.add_argument("--cycle-days", type=int, default=9, help="累计订单周期（X 天），默认 9")
    parser.add_argument("--plot-output", default=None, help="Plotly 图输出路径（默认写入 reports/）")

    args = parser.parse_args(argv)

    # 读取数据
    df = load_data(args.data_path)

    # 自动读取最新 leads_daily 并构造线索识别数表（先于报告生成）
    leads_daily_path = find_latest_leads_daily()
    leads_df = None
    if leads_daily_path and os.path.exists(leads_daily_path):
        try:
            leads_df = load_leads_daily(leads_daily_path)
            print(f"已加载最新线索日级文件：{leads_daily_path}")
        except Exception as e:
            print(f"读取线索日级文件失败，继续不使用线索特征：{e}")

    # 部分1：每日计数（用于快速查看）
    df_filtered, col_time = filter_data(df, args.model_group, args.start_date, args.end_date)
    daily_df = compute_daily_counts(df_filtered, col_time)
    report_text = generate_report(daily_df, args.model_group, args.start_date, args.end_date, leads_df=leads_df)
    # 部分2：训练段准备（CM1/CM2）
    seg_cm1 = compute_daily_counts_for_group(df, "CM1", args.train_cm1_start, args.train_cm1_end, leads_df=leads_df)
    seg_cm2 = compute_daily_counts_for_group(df, "CM2", args.train_cm2_start, args.train_cm2_end, leads_df=leads_df)
    segments = [seg_cm1, seg_cm2]
    norm_curve, model, feat_cols = orchestrate_training(segments)

    # 部分3：新车型（LS9）输入与预测
    seg_new_init = compute_daily_counts_for_group(df, args.new_model, args.new_start_date, args.new_end_date, leads_df=leads_df)
    # 只保留前3日作为已知（若不到3日，则按实际保留）
    seg_new_init = seg_new_init.iloc[:3].copy()
    # 基线按目标周期重采样（确保尾部峰值贴近周期末端）
    baseline = make_baseline_for_cycle_days(norm_curve, args.cycle_days)
    pred_df = predict_new_series(
        model, feat_cols, baseline,
        args.new_start_date, args.new_end_date, args.cycle_days,
        seg_new_init[["date", "count", "leads_recognition_count"]]
    )

    # 扩展文本：先写入前1～3日“已知值”，再附上预测第4～X日
    report_text += "\n\n【新车型预测】"\
        + f"\n车型={args.new_model}，起始={args.new_start_date}，累计周期={args.cycle_days}天"

    # 已知值（前1～3日）
    report_text += "\n【已知值（前1～3日）】"
    for i, r in seg_new_init.iterrows():
        report_text += f"\n- 第{i+1}日({pd.to_datetime(r['date']).strftime('%Y-%m-%d')}): {int(r['count'])}（线索识别数={int(r.get('leads_recognition_count', 0))}）"

    # 预测值（第4～X日）
    report_text += "\n【预测（第4～X日）】"
    for _, r in pred_df.iterrows():
        if int(r["cycle_day"]) >= 4:
            report_text += f"\n- 第{int(r['cycle_day'])}日({pd.to_datetime(r['date']).strftime('%Y-%m-%d')}): {float(r['pred_count']):.0f}"

    # 输出路径
    script_dir = os.path.dirname(__file__)
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))
    reports_dir = ensure_reports_dir(root_dir)

    if args.output:
        out_path = args.output
    else:
        today_str = datetime.now().strftime("%Y-%m-%d")
        out_filename = f"意向订单每日计数与混合预测_{args.model_group}_{args.start_date}_至_{args.end_date}_{today_str}.txt"
        out_path = os.path.join(reports_dir, out_filename)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print(f"简报已生成：{out_path}")

    # 生成Plotly图
    if args.plot_output:
        html_path = args.plot_output
    else:
        today_str = datetime.now().strftime("%Y-%m-%d")
        html_name = f"意向订单双峰混合预测_{args.new_model}_{today_str}.html"
        html_path = os.path.join(reports_dir, html_name)

    try:
        if PLOTLY_AVAILABLE:
            generate_plot(segments, pred_df, args.new_model, html_path)
            print(f"预测图已生成：{html_path}")
        else:
            print("未安装Plotly，跳过图形生成。")
    except Exception as e:
        print(f"生成图形时出现警告：{e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())