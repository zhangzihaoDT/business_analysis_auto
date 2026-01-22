import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load data
file_path = "/Users/zihao_/Documents/coding/dataset/original/assign_data.csv"
# The file is utf-16le encoded based on 'file -I' output
df = pd.read_csv(file_path, encoding='utf-16', sep='\t')

# Rename columns for easier access
# 'Assign Time 年/月/日' -> 'date'
# '下发线索当日锁单数 (门店)' -> 'same_day_lock_store'
# '下发线索数 (门店)' -> 'leads_store'
# '下发线索数' -> 'leads_total'

df.rename(columns={
    'Assign Time 年/月/日': 'date',
    '下发线索当日锁单数 (门店)': 'same_day_lock_store',
    '下发线索数 (门店)': 'leads_store',
    '下发线索数': 'leads_total'
}, inplace=True)

# Parse date
df['date'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce')
df = df.sort_values('date')

# Calculate the ratio: Store Leads / Total Leads
# This represents the average number of stores a lead is distributed to (or similar interpretation)
# Wait, '下发线索数 (门店)' is usually sum of leads received by all stores.
# '下发线索数' is unique leads count.
# So ratio = leads_store / leads_total represents "Average Distribution Count per Lead" (Avg Fan-out).
# Or if it's "Leads per Store", it would be leads_store / store_count.
# The user asked for "下发线索数 (门店)/下发线索数", which is effectively "Avg Distribution Count".

df['distribution_ratio'] = df['leads_store'] / df['leads_total']

# Filter out potential anomalies (e.g., ratio < 1 is impossible if logic holds, or division by zero)
df = df[df['leads_total'] > 0].copy()

# Metric of interest: 'same_day_lock_store'
# We want to see how 'distribution_ratio' affects 'same_day_lock_store'.
# Note: 'same_day_lock_store' is absolute count. Maybe we should normalize it?
# The user asked: "下发线索数 (门店)/下发线索数在多少时，对下发线索当日锁单数 (门店)影响最大"
# This implies looking for a peak or correlation in the absolute number, or maybe efficiency?
# Let's look at absolute number first as requested.

# Create a scatter plot
fig = px.scatter(
    df, 
    x='distribution_ratio', 
    y='same_day_lock_store',
    hover_data=['date', 'leads_total', 'leads_store'],
    title='Impact of Lead Distribution Ratio on Same Day Store Lock Orders',
    labels={
        'distribution_ratio': 'Distribution Ratio (Store Leads / Total Leads)',
        'same_day_lock_store': 'Same Day Store Lock Orders'
    }
)

# Add a trend line (LOWESS)
fig.add_traces(
    px.scatter(df, x='distribution_ratio', y='same_day_lock_store', trendline="lowess").data[1]
)

# Calculate correlation
corr = df['distribution_ratio'].corr(df['same_day_lock_store'])
print(f"Correlation: {corr:.4f}")

# Binning analysis to find the "optimal" range
# We can bin the ratio and calculate mean/median lock orders
df['ratio_bin'] = pd.cut(df['distribution_ratio'], bins=10)
bin_stats = df.groupby('ratio_bin')['same_day_lock_store'].agg(['mean', 'count', 'std']).reset_index()
print("\nBinned Statistics:")
print(bin_stats)

# Save plot
fig.write_html("reports/eda_assign_ratio_impact.html")
print("Report saved to reports/eda_assign_ratio_impact.html")

# Additional Analysis: Efficiency (Conversion Rate)
# same_day_lock_store / leads_store
df['conversion_efficiency'] = df['same_day_lock_store'] / df['leads_store']

fig2 = px.scatter(
    df, 
    x='distribution_ratio', 
    y='conversion_efficiency',
    hover_data=['date'],
    title='Impact of Lead Distribution Ratio on Conversion Efficiency',
    labels={
        'distribution_ratio': 'Distribution Ratio (Store Leads / Total Leads)',
        'conversion_efficiency': 'Conversion Efficiency (Locks / Store Leads)'
    }
)
fig2.add_traces(
    px.scatter(df, x='distribution_ratio', y='conversion_efficiency', trendline="lowess").data[1]
)
fig2.write_html("reports/eda_assign_ratio_efficiency.html")
print("Efficiency report saved to reports/eda_assign_ratio_efficiency.html")

