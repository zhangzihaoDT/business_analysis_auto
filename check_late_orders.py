import pandas as pd
from pathlib import Path

PARQUET_FILE = Path("/Users/zihao_/Documents/coding/dataset/formatted/order_full_data.parquet")

df = pd.read_parquet(PARQUET_FILE)
df['order_create_date'] = pd.to_datetime(df['order_create_date'], errors='coerce')
df['order_create_time'] = pd.to_datetime(df['order_create_time'], errors='coerce')

# Check records after Dec 20
late_orders = df[df['order_create_date'] > '2025-12-20']
print(f"Orders after Dec 20: {len(late_orders)}")
print(late_orders[['order_create_date', 'order_create_time']].head())
print(f"Null order_create_time in late orders: {late_orders['order_create_time'].isna().sum()}")
