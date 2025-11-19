import argparse
import itertools
from pathlib import Path
from typing import List, Tuple, Set

import numpy as np
import pandas as pd


def _to_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _age_to_bin(age: float) -> str:
    if pd.isna(age):
        return "age=未知"
    try:
        age = float(age)
    except Exception:
        return "age=未知"
    if age < 25:
        return "age=<25"
    elif age < 35:
        return "age=25-34"
    elif age < 45:
        return "age=35-44"
    elif age < 55:
        return "age=45-54"
    else:
        return "age=55+"


def _datediff_to_bin(lock_time: pd.Timestamp, first_touch_time: pd.Timestamp) -> str:
    if pd.isna(lock_time) or pd.isna(first_touch_time):
        return "touch_to_lock=未知"
    try:
        diff_days = (pd.Timestamp(lock_time) - pd.Timestamp(first_touch_time)).days
    except Exception:
        return "touch_to_lock=未知"
    # 将时间差分箱，便于 Apriori 处理
    if diff_days <= 0:
        return "touch_to_lock=0d"
    elif diff_days <= 7:
        return "touch_to_lock=1-7d"
    elif diff_days <= 14:
        return "touch_to_lock=8-14d"
    elif diff_days <= 30:
        return "touch_to_lock=15-30d"
    elif diff_days <= 60:
        return "touch_to_lock=31-60d"
    else:
        return "touch_to_lock=60d+"


def _is_age_known(val) -> bool:
    if pd.isna(val):
        return False
    s = str(val).strip()
    if s == "" or s in {"未知", "None", "none", "nan", "NaN", "NAN"}:
        return False
    try:
        f = float(s)
        return not np.isnan(f)
    except Exception:
        return False


def load_orders(orders_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(orders_path)
    return df


def filter_and_select_ls9(df: pd.DataFrame) -> pd.DataFrame:
    # 仅保留车型分组=LS9
    if "车型分组" not in df.columns:
        raise KeyError("源数据缺少列: 车型分组")
    df = df[df["车型分组"] == "LS9"].copy()

    # 只保留指定列
    required_cols = [
        "Order Number",
        "Lock_Time",
        "first_touch_time",
        "Parent Region Name",
        "buyer_age",
        "order_gender",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"源数据缺少列: {missing}")

    df = df[required_cols].copy()

    # 规范字段名以便后续合并
    df = df.rename(columns={"Order Number": "order_number"})

    # 时间列转为日期
    df["Lock_Time"] = _to_datetime_safe(df["Lock_Time"])  # 锁单时间
    df["first_touch_time"] = _to_datetime_safe(df["first_touch_time"])  # 首次触达时间

    # 剔除 gender=默认未知 与 age=未知 的记录（避免在分类类型列上直接 fillna）
    gender_str = df["order_gender"].astype(str).str.strip()
    df = df[(gender_str != "默认未知") & (gender_str != "")].copy()
    df = df[df["buyer_age"].apply(_is_age_known)].copy()

    return df


def load_ls9_exclude_orders(exclude_path: Path) -> Set[str]:
    """从重复购买订单列表中读取需要剔除的 LS9 订单号集合"""
    df_ex = pd.read_csv(exclude_path, dtype=str)
    required_cols = ["Order Number", "车型分组"]
    missing = [c for c in required_cols if c not in df_ex.columns]
    if missing:
        raise KeyError(f"排除文件缺少列: {missing}")
    df_ex = df_ex[df_ex["车型分组"] == "LS9"].copy()
    return set(df_ex["Order Number"].dropna().astype(str).str.strip().tolist())


def load_config(config_path: Path) -> pd.DataFrame:
    # 读取配置 CSV
    dfc = pd.read_csv(config_path, dtype=str)
    # 标准化主键字段名
    if "order_number" not in dfc.columns:
        # 尝试可能的变体
        if "Order Number" in dfc.columns:
            dfc = dfc.rename(columns={"Order Number": "order_number"})
        else:
            raise KeyError("配置数据缺少列: order_number")
    return dfc


def join_orders_config(df_orders: pd.DataFrame, df_cfg: pd.DataFrame) -> pd.DataFrame:
    merged = df_orders.merge(df_cfg, on="order_number", how="inner")
    if merged.empty:
        raise ValueError("合并结果为空：请检查订单号是否匹配")
    return merged


def build_transactions(df: pd.DataFrame) -> List[List[str]]:
    transactions: List[List[str]] = []

    # 用户画像字段（前缀 U:）
    user_cols = {
        "Parent Region Name": "region",
        "order_gender": "gender",
    }

    # 配置字段（前缀 C:）
    config_cols = [
        "Product Name",
        "EXCOLOR",
        "INCOLOR",
        "OP-Hitch",
        "OP-SW",
        "WHEEL",
    ]

    # 构造交易项
    for _, row in df.iterrows():
        items: List[str] = []

        # 用户画像：地区、性别、年龄分箱、时间分箱
        for col, tag in user_cols.items():
            val = row.get(col, None)
            if pd.notna(val) and str(val).strip() != "":
                items.append(f"U:{tag}={str(val).strip()}")

        # 年龄分箱（变量：buyer_age）
        items.append(f"U:{_age_to_bin(row.get('buyer_age', np.nan))}")

        # datediff分箱（变量：Lock_Time 与 first_touch_time 的差值）
        items.append(f"U:{_datediff_to_bin(row.get('Lock_Time', pd.NaT), row.get('first_touch_time', pd.NaT))}")

        # 配置项
        for col in config_cols:
            # 检查列是否存在于行数据中，因为merge可能不完整
            if col in row and pd.notna(row[col]):
                val = str(row[col]).strip()
                if val != "":
                    items.append(f"C:{col}={val}")

        transactions.append(items)

    return transactions


def transactions_to_onehot(transactions: List[List[str]]) -> pd.DataFrame:
    # 手工 One-Hot，避免额外依赖
    all_items = sorted(set(itertools.chain.from_iterable(transactions)))
    data = []
    for tx in transactions:
        s = set(tx)
        row = {item: (item in s) for item in all_items}
        data.append(row)
    onehot = pd.DataFrame(data, columns=all_items)
    return onehot


def run_apriori(onehot: pd.DataFrame, min_support: float, min_confidence: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except Exception as e:
        raise ImportError(
            "未找到 mlxtend，请先安装: pip install mlxtend\n"
            f"原始错误: {e}"
        )

    frequent_itemsets = apriori(onehot, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return frequent_itemsets, rules


def filter_user_to_config_rules(rules: pd.DataFrame) -> pd.DataFrame:
    # 筛选规则：前件至少包含一个U:项，后件至少包含一个C:项
    def check_antecedents(fs: frozenset) -> bool:
        return any(str(x).startswith("U:") for x in fs)

    def check_consequents(fs: frozenset) -> bool:
        return any(str(x).startswith("C:") for x in fs)

    mask = rules["antecedents"].apply(check_antecedents) & rules["consequents"].apply(check_consequents)
    filtered = rules[mask].copy()
    filtered["antecedent_len"] = filtered["antecedents"].apply(lambda x: len(x))
    filtered["consequent_len"] = filtered["consequents"].apply(lambda x: len(x))
    filtered = filtered.sort_values(["confidence", "lift"], ascending=[False, False])
    return filtered


def main():
    parser = argparse.ArgumentParser(description="生成 LS9 用户画像→配置 的关联规则（Apriori）")
    parser.add_argument(
        "--orders-path",
        type=Path,
        default=Path("/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"),
        help="订单意向分析 parquet 文件路径",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("/Users/zihao_/Documents/coding/dataset/processed/LS9_Configuration_Details_transposed_20251119_092014.csv"),
        help="LS9 配置详情 CSV 文件路径",
    )
    parser.add_argument(
        "--exclude-orders-path",
        type=Path,
        default=Path("/Users/zihao_/Documents/coding/dataset/original/repeat_buyer_orders_list_LS9_2025-11-12_to_2025-11-18.csv"),
        help="需剔除的重复购买 LS9 订单列表 CSV 文件路径",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("/Users/zihao_/Documents/coding/dataset/processed/analysis_results/LS9_user_to_config_association_rules.csv"),
        help="输出规则 CSV 路径",
    )
    parser.add_argument("--min-support", type=float, default=0.05, help="最小支持度")
    parser.add_argument("--min-confidence", type=float, default=0.4, help="最小置信度")

    args = parser.parse_args()

    print("[1/6] 读取订单意向数据…", args.orders_path)
    df_orders = load_orders(args.orders_path)

    print("[2/6] 筛选 LS9 并选取所需列…")
    df_orders = filter_and_select_ls9(df_orders)

    # 额外过滤：剔除 original 文件中列示的 LS9 重复购买订单
    if args.exclude_orders_path.exists():
        try:
            exclude_set = load_ls9_exclude_orders(args.exclude_orders_path)
            before = len(df_orders)
            df_orders = df_orders[~df_orders["order_number"].astype(str).str.strip().isin(exclude_set)].copy()
            removed = before - len(df_orders)
            print(f"已剔除重复购买 LS9 订单 {removed} 条（来源: {args.exclude_orders_path.name}）")
        except Exception as e:
            print(f"警告：读取排除订单失败，跳过该过滤。原因: {e}")
    else:
        print("提示：未找到排除订单文件，跳过该过滤。")

    print("[3/6] 读取 LS9 配置数据…", args.config_path)
    df_cfg = load_config(args.config_path)

    print("[4/6] 按订单号合并…")
    df_merged = join_orders_config(df_orders, df_cfg)
    print(f"合并后行数: {len(df_merged)}")

    print("[5/6] 构造交易并进行 One-Hot 编码…")
    transactions = build_transactions(df_merged)
    onehot = transactions_to_onehot(transactions)
    print(f"交易数: {len(transactions)}, 项目维度: {onehot.shape[1]}")

    print("[6/6] 运行 Apriori 并生成关联规则…")
    frequent_itemsets, rules = run_apriori(onehot, args.min_support, args.min_confidence)
    user_to_config_rules = filter_user_to_config_rules(rules)

    # 输出
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    # 将集合列转换为字符串，便于查看
    to_save = user_to_config_rules.copy()
    to_save["antecedents"] = to_save["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    to_save["consequents"] = to_save["consequents"].apply(lambda s: ", ".join(sorted(list(s))))

    to_save.to_csv(args.out_path, index=False)

    # 打印 支持数>30 且按 lift 降序 的 Top10 概览
    print("完成！规则已保存:", args.out_path)
    n_tx = len(transactions)
    support_threshold = 30
    preview_df = to_save.copy()
    preview_df["support_count"] = (preview_df["support"] * n_tx).round().astype(int)
    preview_df = preview_df[preview_df["support_count"] > support_threshold]
    preview_df = preview_df.sort_values(["lift", "confidence"], ascending=[False, False])
    preview_cols = ["antecedents", "consequents", "support_count", "support", "confidence", "lift"]
    print(preview_df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()