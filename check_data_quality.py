
import pandas as pd

file_path = '/Users/zihao_/Documents/coding/dataset/original/业务数据记录_竞争PK（LS6）.csv'
df = pd.read_csv(file_path)

print("Unique Car Series:")
print(df['车系'].unique())

print("\nRows per week (Head 10):")
print(df['Week'].value_counts().head(10))

# Check for Xiaomi
print("\nXiaomi entries:")
print(df[df['车系'].str.contains('Xiaomi|小米', case=False)]['车系'].unique())

# Check for Link
print("\nLink/Lingke entries:")
print(df[df['车系'].str.contains('Link|领克', case=False)]['车系'].unique())
