import pandas as pd
import re

file_path = '/Users/zihao_/Documents/coding/dataset/original/业务数据记录_竞争PK（LS6）_表格.csv'

def parse_pk_count(count_str):
    if pd.isna(count_str):
        return 0
    if isinstance(count_str, (int, float)):
        return count_str
    return int(str(count_str).replace(',', ''))

def normalize_date_range(date_range_str):
    # Replace Chinese tilde with standard tilde
    return date_range_str.replace('～', '~')

def analyze_periods():
    df = pd.read_csv(file_path)
    
    # Normalize Week column
    df['Week_Norm'] = df['Week'].apply(normalize_date_range)
    
    # Define periods by matching substrings in Week column
    # We look for weeks that fall into the requested ranges.
    
    # Period 1: 2024-12-01 to 2024-12-14
    # Weeks: 2024-12-02~2024-12-08, 2024-12-09~2024-12-15
    p1_keywords = ['2024-12-02', '2024-12-09']
    
    # Period 2: 2025-11-01 to 2025-11-14
    # Weeks: 2025-11-03~11-09, 2025-11-10~11-16
    # Note: The CSV might use "2025-11-03~11-09" or "2025-11-03~2025-11-09"
    p2_keywords = ['2025-11-03', '2025-11-10']
    
    # Period 3: 2025-12-01 to 2025-12-14
    # Weeks: 2025-12-01~12-07, 2025-12-08~12-14
    p3_keywords = ['2025-12-01', '2025-12-08']
    
    periods = {
        '2024-12 (Period 1)': p1_keywords,
        '2025-11 (Period 2)': p2_keywords,
        '2025-12 (Period 3)': p3_keywords
    }
    
    results = {}
    
    for period_name, keywords in periods.items():
        # Filter rows where Week_Norm contains any of the keywords
        # We assume the start date is unique enough
        mask = df['Week_Norm'].apply(lambda x: any(k in x for k in keywords))
        period_df = df[mask].copy()
        
        # Clean PK count
        period_df['PK_Count'] = period_df['PK次数'].apply(parse_pk_count)
        
        # Aggregate by Car Series
        agg = period_df.groupby(['车系', '品牌'])['PK_Count'].sum().reset_index()
        agg = agg.sort_values('PK_Count', ascending=False)
        
        results[period_name] = agg
        
        print(f"--- {period_name} ---")
        print(f"Weeks found: {period_df['Week_Norm'].unique()}")
        print(agg.head(5).to_string(index=False))
        print("\n")

if __name__ == "__main__":
    analyze_periods()
