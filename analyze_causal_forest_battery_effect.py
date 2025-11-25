#!/usr/bin/env python3
import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

DEFAULT_DATA_PATH = "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"

PRICE_MAP = {
    "智己LS9 52 Ultra": {"price": 322800,"acc_0_100_s": 4.9, "range_est_km": 1454,"battery_types":"A_LFP","battery_supplier":"CATL"},
    "智己LS9 66 Ultra": {"price": 352800,"acc_0_100_s": 4.9, "range_est_km": 1508,"battery_types":"B_NCM","battery_supplier":"CATL"},
    "新一代智己LS6 66 Max+": {"price": 249900,"acc_0_100_s": 6.4, "range_est_km": 1502,"battery_types":"B_NCM","battery_supplier":"CATL"},
    "新一代智己LS6 66 Max": {"price": 234900,"acc_0_100_s": 6.4, "range_est_km": 1502,"battery_types":"B_NCM","battery_supplier":"CATL"},
    "新一代智己LS6 52 Max+": {"price": 229900,"acc_0_100_s": 6.4, "range_est_km": 1400,"battery_types":"A_LFP","battery_supplier":"JSL"},
    "新一代智己LS6 52 Max": {"price": 214900,"acc_0_100_s": 6.4, "range_est_km": 1400,"battery_types":"A_LFP","battery_supplier":"JSL"},
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_DATA_PATH)
    p.add_argument("--cm2_start", default="2025-09-10")
    p.add_argument("--cm2_end", default="2025-10-15")
    p.add_argument("--ls9_start", default="2025-11-12")
    p.add_argument("--ls9_end", default="2025-11-20")
    p.add_argument("--outcome", choices=["sales", "share"], default="sales")
    p.add_argument("--method", choices=["econml", "xlearner"], default="econml")
    return p.parse_args()

def resolve_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_and_filter(df):
    col_lock = resolve_col(df, ["Lock_Time", "lock_time", "Lock Time"])
    col_group = resolve_col(df, ["车型分组", "model_group", "车型", "Model Group"])
    col_product = resolve_col(df, ["Product Name", "产品名称", "product_name"])
    col_order = resolve_col(df, ["Order Number", "order_number", "订单编号"])
    col_age = resolve_col(df, ["buyer_age", "年龄", "age"])
    col_gender = resolve_col(df, ["order_gender", "性别", "gender"])
    col_region = resolve_col(df, ["Parent Region Name", "License Province", "省份", "province"])
    for c in [col_lock, col_group, col_product]:
        if c is None:
            raise ValueError("缺少必要列")
    df[col_lock] = pd.to_datetime(df[col_lock], errors="coerce")
    df_cm2_time = df[(df[col_group].astype(str) == "CM2") & (df[col_lock] >= pd.to_datetime(args.cm2_start)) & (df[col_lock] <= pd.to_datetime(args.cm2_end))]
    df_ls9_time = df[(df[col_group].astype(str) == "LS9") & (df[col_lock] >= pd.to_datetime(args.ls9_start)) & (df[col_lock] <= pd.to_datetime(args.ls9_end))]
    df_cm2 = df_cm2_time[df_cm2_time[col_product].astype(str).str.contains("52|66", regex=True, na=False)]
    df_ls9 = df_ls9_time[df_ls9_time[col_product].astype(str).str.contains("52|66", regex=True, na=False)]
    df_u = pd.concat([df_cm2, df_ls9], ignore_index=True)
    print(f"CM2 订单数: {len(df_cm2)}")
    print(f"LS9 订单数: {len(df_ls9)}")
    battery = np.where(df_u[col_product].astype(str).str.contains("66"), 66, np.where(df_u[col_product].astype(str).str.contains("52"), 52, np.nan))
    df_u = df_u.assign(battery_kwh=battery)
    df_u = df_u.dropna(subset=["battery_kwh"])
    prod_info = df_u[col_product].map(PRICE_MAP)
    listed_price = prod_info.apply(lambda x: x["price"] if isinstance(x, dict) else np.nan)
    acc_0_100_s = prod_info.apply(lambda x: x.get("acc_0_100_s") if isinstance(x, dict) else np.nan)
    range_est_km = prod_info.apply(lambda x: x.get("range_est_km") if isinstance(x, dict) else np.nan)
    battery_types = prod_info.apply(lambda x: x.get("battery_types") if isinstance(x, dict) else np.nan)
    battery_supplier = prod_info.apply(lambda x: x.get("battery_supplier") if isinstance(x, dict) else np.nan)
    ls9_base = PRICE_MAP.get("智己LS9 52 Ultra", {}).get("price", np.nan)
    ls6_base_plus = PRICE_MAP.get("新一代智己LS6 52 Max+", {}).get("price", np.nan)
    ls6_base = PRICE_MAP.get("新一代智己LS6 52 Max", {}).get("price", np.nan)
    def _price_diff(name, price):
        name = str(name)
        if pd.isna(price):
            return np.nan
        if "智己LS9" in name:
            if "66 Ultra" in name:
                return float(price) - float(ls9_base) if not pd.isna(ls9_base) else np.nan
            if "52 Ultra" in name:
                return 0.0
        if "新一代智己LS6" in name:
            if "66 Max+" in name:
                return float(price) - float(ls6_base_plus) if not pd.isna(ls6_base_plus) else np.nan
            if "66 Max" in name:
                return float(price) - float(ls6_base) if not pd.isna(ls6_base) else np.nan
            if "52 Max+" in name or "52 Max" in name:
                return 0.0
        return np.nan
    price_diff = df_u[col_product].astype(str).combine(listed_price, _price_diff)
    df_u = df_u.assign(listed_price=listed_price, acc_0_100_s=acc_0_100_s, range_est_km=range_est_km, battery_types=battery_types, battery_supplier=battery_supplier, price_diff_to_low=price_diff)

    op_laser = None
    try:
        base = "/Users/zihao_/Documents/coding/dataset/processed"
        files = [f for f in os.listdir(base) if f.endswith('.csv')]
        cm2_files = [f for f in files if f.startswith('CM2_Configuration_Details_transposed_')]
        ls9_files = [f for f in files if f.startswith('LS9_Configuration_Details_transposed_')]
        pick = []
        if cm2_files:
            cm2_latest = max(cm2_files, key=lambda f: os.path.getmtime(os.path.join(base, f)))
            pick.append(os.path.join(base, cm2_latest))
        if ls9_files:
            ls9_latest = max(ls9_files, key=lambda f: os.path.getmtime(os.path.join(base, f)))
            pick.append(os.path.join(base, ls9_latest))
        dfs = []
        for pth in pick:
            dtmp = pd.read_csv(pth)
            cols = [c for c in dtmp.columns]
            c_order = "order_number" if "order_number" in cols else resolve_col(dtmp, ["Order Number", "order_number"]) or "order_number"
            c_op = "OP-LASER" if "OP-LASER" in cols else resolve_col(dtmp, ["OP-LASER", "op_laser"]) or "OP-LASER"
            dtmp = dtmp[[c_order, c_op]].rename(columns={c_order: "order_number", c_op: "op_laser"})
            dfs.append(dtmp)
        if dfs:
            cfg = pd.concat(dfs, ignore_index=True)
            cfg = cfg.dropna(subset=["order_number"]).drop_duplicates("order_number", keep="last")
            if col_order is not None and col_order in df_u.columns:
                df_u = df_u.merge(cfg, left_on=col_order, right_on="order_number", how="left")
                df_u = df_u.drop(columns=["order_number"]) if "order_number" in df_u.columns else df_u
            else:
                df_u = df_u.assign(op_laser=np.nan)
        else:
            df_u = df_u.assign(op_laser=np.nan)
    except Exception:
        df_u = df_u.assign(op_laser=np.nan)
    is_ls9 = (df_u[col_group].astype(str) == "LS9").astype(int)
    age = pd.to_numeric(df_u[col_age], errors="coerce") if col_age is not None else pd.Series(np.nan, index=df_u.index)
    gender_raw = df_u[col_gender].astype(str) if col_gender is not None else pd.Series("", index=df_u.index)
    gender = gender_raw.map({"男": 1, "male": 1, "M": 1, "女": 0, "female": 0, "F": 0}).fillna(0).astype(int)
    region = df_u[col_region].astype(str) if col_region is not None else pd.Series("", index=df_u.index)
    op_laser_val = df_u.get("op_laser", pd.Series(np.nan, index=df_u.index)).astype(str).fillna("")

    age_filled = age.fillna(age.dropna().median() if age.dropna().size > 0 else 0)
    bins = [0, 30, 40, 50, 100]
    age_bin = pd.cut(age_filled, bins=bins, labels=False, include_lowest=True).fillna(0).astype(int)

    df_feat = pd.DataFrame({
        "is_ls9": is_ls9.astype(int),
        "gender": gender.astype(int),
        "age_bin": age_bin.astype(int),
        "region_name": region.fillna("").astype(str),
        "battery_kwh": df_u["battery_kwh"].astype(int),
        "listed_price": pd.to_numeric(df_u["listed_price"], errors="coerce"),
        "acc_0_100_s": pd.to_numeric(df_u["acc_0_100_s"], errors="coerce"),
        "range_est_km": pd.to_numeric(df_u["range_est_km"], errors="coerce"),
        "op_laser": op_laser_val,
        "battery_types": df_u["battery_types"].astype(str),
        "battery_supplier": df_u["battery_supplier"].astype(str),
        "price_diff_to_low": pd.to_numeric(df_u["price_diff_to_low"], errors="coerce"),
    })

    keys = ["is_ls9", "gender", "age_bin", "region_name", "op_laser", "battery_kwh"]
    grp = df_feat.groupby(keys, dropna=False).agg(
        count=("battery_kwh", "size"),
        listed_price=("listed_price", "mean"),
        acc_0_100_s=("acc_0_100_s", "mean"),
        range_est_km=("range_est_km", "mean"),
        price_diff_to_low=("price_diff_to_low", "mean")
    ).reset_index()

    total_grp = df_feat.groupby(["is_ls9", "gender", "age_bin", "region_name", "op_laser"], dropna=False).agg(
        total=("battery_kwh", "size")
    ).reset_index()
    grp = grp.merge(total_grp, on=["is_ls9", "gender", "age_bin", "region_name", "op_laser"], how="left")
    grp["share"] = grp["count"] / grp["total"].replace(0, np.nan)
    grp["share"] = grp["share"].fillna(0.0)

    bt_dum = pd.get_dummies(df_feat["battery_types"].fillna("").astype(str), prefix="battery_types")
    bs_dum = pd.get_dummies(df_feat["battery_supplier"].fillna("").astype(str), prefix="battery_supplier")
    feat_dum = pd.concat([df_feat[keys], bt_dum, bs_dum], axis=1)
    dum_mean = feat_dum.groupby(keys, dropna=False).mean().reset_index()
    grp = grp.merge(dum_mean, on=keys, how="left")

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    region_oh = enc.fit_transform(grp[["region_name"]])
    op_dummies = pd.get_dummies(grp["op_laser"].fillna("").astype(str), prefix="op_laser")

    X_num = grp[["gender", "age_bin", "listed_price", "price_diff_to_low", "acc_0_100_s", "range_est_km"]].fillna(0.0).to_numpy()
    bt_cols = [c for c in grp.columns if c.startswith("battery_types_")]
    bs_cols = [c for c in grp.columns if c.startswith("battery_supplier_")]
    bt_mat = grp[bt_cols].to_numpy() if bt_cols else np.zeros((len(grp), 0))
    bs_mat = grp[bs_cols].to_numpy() if bs_cols else np.zeros((len(grp), 0))
    X = np.column_stack([X_num, region_oh, op_dummies.to_numpy(), bt_mat, bs_mat])
    T = (grp["battery_kwh"] == 66).astype(int).to_numpy()
    if args.outcome == "sales":
        y = np.log1p(grp["count"].to_numpy())
    else:
        y = grp["share"].to_numpy()
    meta = {
        "is_ls9": grp["is_ls9"].to_numpy(),
        "gender": grp["gender"].to_numpy(),
        "age_bin": grp["age_bin"].to_numpy(),
        "region_labels": enc.get_feature_names_out(["region_name"]),
        "region_values": grp["region_name"].to_numpy(),
        "group_col": col_group,
        "groups": grp
    }
    return X, T, y, meta, df_u

def run_econml(X, T, y):
    from econml.dml import CausalForestDML
    est = CausalForestDML(model_t=RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42), model_y=RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42), discrete_treatment=True, n_estimators=500, max_depth=6, random_state=42)
    est.fit(y, T, X=X)
    te = est.effect(X)
    return te
def run_x_learner(X, T, y):
    mu1 = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    mu0 = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    mu1.fit(X[T == 1], y[T == 1])   
    mu0.fit(X[T == 0], y[T == 0])
    p = LogisticRegression(max_iter=1000)
    p.fit(X, T)
    prop = p.predict_proba(X)[:, 1]
    d1 = y[T == 1] - mu0.predict(X[T == 1])
    d0 = mu1.predict(X[T == 0]) - y[T == 0]
    tau1 = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    tau0 = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    tau1.fit(X[T == 1], d1)
    tau0.fit(X[T == 0], d0)
    cate = prop * tau0.predict(X) + (1 - prop) * tau1.predict(X)
    return cate

def summarize(te, meta, df_u):
    print(f"ATE(66 vs 52) 对结果: {float(np.mean(te)):.4f}")
    g = meta["is_ls9"]
    print(f"LS9 ATE: {float(np.mean(te[g == 1])):.4f}")
    print(f"CM2 ATE: {float(np.mean(te[g == 0])):.4f}")
    ge = meta["gender"]
    print(f"男性 ATE: {float(np.mean(te[ge == 1])):.4f}")
    print(f"女性 ATE: {float(np.mean(te[ge == 0])):.4f}")
    ageb = meta["age_bin"]
    for b in sorted(np.unique(ageb)):
        idx = ageb == b
        print(f"年龄Bin {int(b)} ATE: {float(np.mean(te[idx])):.4f}")
    regions = pd.Series(meta["region_values"]).fillna("")
    top_regions = regions.value_counts().head(5).index.tolist()
    for r in top_regions:
        idx = (regions == r).to_numpy()
        if np.any(idx):
            print(f"{r} ATE: {float(np.mean(te[idx])):.4f}")

    try:
        grp = meta["groups"]
        X_cols = pd.DataFrame({
            "Gender": grp["gender"].astype(int),
            "Age_Bin": grp["age_bin"].astype(int),
            "Price": pd.to_numeric(grp["listed_price"], errors="coerce").fillna(0.0),
            "Price_Diff": pd.to_numeric(grp["price_diff_to_low"], errors="coerce").fillna(0.0),
            "Acc_0_100_s": pd.to_numeric(grp["acc_0_100_s"], errors="coerce").fillna(0.0),
            "Range_Est_km": pd.to_numeric(grp["range_est_km"], errors="coerce").fillna(0.0),
        })
        region_dummies = pd.get_dummies(grp["region_name"].fillna("").astype(str), prefix="region")
        op_dummies2 = pd.get_dummies(grp["op_laser"].fillna("").astype(str), prefix="op_laser")
        bt_cols = [c for c in grp.columns if c.startswith("battery_types_")]
        bs_cols = [c for c in grp.columns if c.startswith("battery_supplier_")]
        bt_df = grp[bt_cols] if bt_cols else pd.DataFrame(index=grp.index)
        bs_df = grp[bs_cols] if bs_cols else pd.DataFrame(index=grp.index)
        X_cols = pd.concat([X_cols, region_dummies, op_dummies2, bt_df, bs_df], axis=1)

        rf = RandomForestRegressor(n_estimators=300, random_state=42)
        rf.fit(X_cols.to_numpy(), te)
        importances = rf.feature_importances_
        names = X_cols.columns.tolist()

        imp_df = pd.DataFrame({"feature": names, "importance": importances})
        imp_df["category"] = imp_df["feature"].apply(lambda s: "Region" if s.startswith("region_") else ("OP-LASER" if s.startswith("op_laser_") else ("Battery_Type" if s.startswith("battery_types_") else ("Battery_Supplier" if s.startswith("battery_supplier_") else ("Price_Diff" if s == "Price_Diff" else s)))))
        cat_imp = imp_df.groupby("category")["importance"].sum().reset_index().sort_values("importance", ascending=False)
        print("因素驱动差异（汇总类别重要性，Top 5）：")
        for i, (_, row) in enumerate(cat_imp.head(5).iterrows(), 1):
            print(f"{i}\t{row['category']}\t{row['importance']:.4f}")
        return imp_df, cat_imp
    except Exception:
        return None, None

def main(args):
    df = pd.read_parquet(args.data)
    col_lock = resolve_col(df, ["Lock_Time", "lock_time", "Lock Time"])
    col_group = resolve_col(df, ["车型分组", "model_group", "车型", "Model Group"])
    col_product = resolve_col(df, ["Product Name", "产品名称", "product_name"])
    df[col_lock] = pd.to_datetime(df[col_lock], errors="coerce")
    cm2 = df[(df[col_group].astype(str) == "CM2") & (df[col_lock] >= pd.to_datetime(args.cm2_start)) & (df[col_lock] <= pd.to_datetime(args.cm2_end))]
    cm2 = cm2[cm2[col_product].astype(str).str.contains("52|66", regex=True, na=False)]
    ls9 = df[(df[col_group].astype(str) == "LS9") & (df[col_lock] >= pd.to_datetime(args.ls9_start)) & (df[col_lock] <= pd.to_datetime(args.ls9_end))]
    ls9 = ls9[ls9[col_product].astype(str).str.contains("52|66", regex=True, na=False)]
    cm2_total = len(cm2)
    ls9_total = len(ls9)
    cm2_52 = cm2[cm2[col_product].astype(str).str.contains("52", regex=False, na=False)]
    cm2_66 = cm2[cm2[col_product].astype(str).str.contains("66", regex=False, na=False)]
    ls9_52 = ls9[ls9[col_product].astype(str).str.contains("52", regex=False, na=False)]
    ls9_66 = ls9[ls9[col_product].astype(str).str.contains("66", regex=False, na=False)]
    print(f"CM2 订单数: {cm2_total}")
    print(f"CM2 52 数量: {len(cm2_52)} ({(len(cm2_52)/cm2_total if cm2_total>0 else 0):.2%})")
    print(f"CM2 66 数量: {len(cm2_66)} ({(len(cm2_66)/cm2_total if cm2_total>0 else 0):.2%})")
    print(f"LS9 订单数: {ls9_total}")
    print(f"LS9 52 数量: {len(ls9_52)} ({(len(ls9_52)/ls9_total if ls9_total>0 else 0):.2%})")
    print(f"LS9 66 数量: {len(ls9_66)} ({(len(ls9_66)/ls9_total if ls9_total>0 else 0):.2%})")
    X, T, y, meta, df_u = load_and_filter(df)
    if args.method == "econml":
        te = run_econml(X, T, y)
    else:
        te = run_x_learner(X, T, y)
    imp_df, cat_imp = summarize(te, meta, df_u)
    out_dir = "/Users/zihao_/Documents/coding/dataset/processed/analysis_results"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cat_imp is not None:
        if args.outcome == "sales":
            rank_path = os.path.join(out_dir, f"factor_ranking_sales_{ts}.csv")
            cat_imp.sort_values("importance", ascending=False)[["category", "importance"]].to_csv(rank_path, index=False)
            print(f"保存销量因素排名: {rank_path}")
        else:
            rank_path = os.path.join(out_dir, f"factor_ranking_share_{ts}.csv")
            cat_imp.sort_values("importance", ascending=False)[["category", "importance"]].to_csv(rank_path, index=False)
            print(f"保存占比因素排名: {rank_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
