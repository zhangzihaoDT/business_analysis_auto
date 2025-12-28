import pandas as pd
from pathlib import Path

FILE_PATH = Path("/Users/zihao_/Documents/coding/dataset/original/test_drive_data.csv")

try:
    encodings = ['utf-8', 'gbk', 'utf-16', 'utf-8-sig']
    df = None
    
    for enc in encodings:
        try:
            print(f"Trying encoding: {enc}")
            # Try tab separator
            df = pd.read_csv(FILE_PATH, sep='\t', encoding=enc)
            if len(df.columns) <= 1:
                print("Tab separator failed/one column, trying comma...")
                df = pd.read_csv(FILE_PATH, sep=',', encoding=enc)
            
            print("Success!")
            break
        except Exception as e:
            print(f"Failed with {enc}: {e}")
            
    if df is not None:
        print("Columns:", df.columns.tolist())
        print("First row:", df.iloc[0].to_dict())
        
        # Check date range
        date_col = 'create_date 年/月/日'
        df['date'] = pd.to_datetime(df[date_col], format='%Y年%m月%d日')
        print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
        print(f"Years: {df['date'].dt.year.unique()}")
    else:
        print("All encodings failed")
        
except Exception as e:
    print(f"Error: {e}")
