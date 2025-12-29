import pandas as pd
from pathlib import Path

PARQUET_FILE = Path("/Users/zihao_/Documents/coding/dataset/formatted/order_full_data.parquet")

def inspect_data():
    df = pd.read_parquet(PARQUET_FILE)
    print("Columns:", df.columns.tolist())
    
    if 'age' in df.columns:
        print("\nAge stats:")
        print(df['age'].describe())
        print("Age sample:", df['age'].head().tolist())
        print("Missing age count:", df['age'].isna().sum())
    else:
        print("\n'age' column not found!")

    if 'parent_region_name' in df.columns:
        print("\nParent Region stats:")
        print(df['parent_region_name'].value_counts())
        print("Missing region count:", df['parent_region_name'].isna().sum())
    else:
        print("\n'parent_region_name' column not found!")

if __name__ == "__main__":
    inspect_data()
