import pandas as pd
from pathlib import Path

PARQUET_FILE = Path("/Users/zihao_/Documents/coding/dataset/formatted/order_full_data.parquet")

try:
    df = pd.read_parquet(PARQUET_FILE)
    if 'order_create_time' in df.columns:
        df['order_create_time'] = pd.to_datetime(df['order_create_time'], errors='coerce')
        max_date = df['order_create_time'].max()
        print(f"Max order_create_time: {max_date}")
        
        # Check if there are other date columns that might be later
        date_cols = [c for c in df.columns if 'time' in c or 'date' in c]
        for c in date_cols:
            if df[c].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(df[c]):
                try:
                    d = pd.to_datetime(df[c], errors='coerce')
                    print(f"Max {c}: {d.max()}")
                except:
                    pass
    else:
        print("order_create_time column not found")
        
except Exception as e:
    print(f"Error: {e}")
