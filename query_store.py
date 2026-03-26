from datetime import datetime, timedelta

from pathlib import Path

import pandas as pd

PARQUET_FILE = Path("/Users/zihao_/Documents/coding/dataset/formatted/order_data.parquet")


def main():
    df = pd.read_parquet(PARQUET_FILE)
    
    if "store_name" not in df.columns:
        print("数据中没有 store_name 列")
        return
    
    baoding_stores = df[df["store_name"].str.contains("保定", na=False)]["store_name"].unique()
    
    print(f"含有'保定'的门店名称 (共 {len(baoding_stores)} 个):")
    for store in sorted(baoding_stores):
        print(f"  - {store}")
    
    print("\n" + "=" * 60)
    
    target_store = "保定莲池"
    store_df = df[df["store_name"] == target_store]
    
    if len(store_df) == 0:
        print(f"\n未找到门店: {target_store}")
        return
    
    store_create_date = store_df["store_create_date"].dropna().iloc[0] if "store_create_date" in store_df.columns else None
    print(f"\n【{target_store}】门店详情:")
    print(f"  门店创建时间: {store_create_date}")
    
    locked_orders = store_df[store_df["lock_time"].notna()]
    total_locked = locked_orders["order_number"].nunique()
    print(f"\n  自开业以来锁单数: {total_locked}")
    
    latest_lock_time = df["lock_time"].max()
    if pd.notna(latest_lock_time):
        cutoff_date = latest_lock_time - timedelta(days=30)
        recent_locked = locked_orders[locked_orders["lock_time"] >= cutoff_date]
        recent_locked_count = recent_locked["order_number"].nunique()
        print(f"\n  近30日锁单数: {recent_locked_count}")
        print(f"    (统计截止时间: {latest_lock_time}, 近30日起始: {cutoff_date})")
        
        print(f"\n  近30日分 product_name 锁单数:")
        recent_by_product = recent_locked.groupby("product_name")["order_number"].nunique().sort_values(ascending=False)
        for product, count in recent_by_product.items():
            print(f"    - {product}: {count}")


if __name__ == "__main__":
    main()
