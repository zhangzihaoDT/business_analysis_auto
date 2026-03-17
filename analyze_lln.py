import csv
from datetime import datetime
import os
import sys
import random
from collections import Counter

file_path = '/Users/zihao_/Documents/github/W52_reasoning/锁单归因_2025.csv'

def parse_date(date_str):
    if not date_str or not date_str.strip():
        return None
    try:
        # Expected format: YYYY年MM月DD日
        return datetime.strptime(date_str.strip(), '%Y年%m月%d日')
    except ValueError:
        return None

data = []
# It seems to be UTF-16 with tab delimiter
encodings_to_try = ['utf-16', 'utf-16-le', 'utf-16-be']

for encoding in encodings_to_try:
    try:
        print(f"Trying encoding: {encoding} with tab delimiter")
        with open(file_path, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter='\t')
            temp_data = []
            for row in reader:
                temp_data.append(row)
            
            if temp_data:
                # Check if keys look reasonable (more than 1 key)
                keys = list(temp_data[0].keys())
                if len(keys) > 1:
                    data = temp_data
                    print(f"Successfully read with {encoding}")
                    break
                else:
                    print(f"Read with {encoding} but only found 1 column: {keys}")
    except (UnicodeDecodeError, csv.Error) as e:
        print(f"Failed with {encoding}: {e}")
        continue

if not data:
    print("Could not read file properly.")
    sys.exit(1)

print(f"Total rows read: {len(data)}")

# Identify columns
headers = list(data[0].keys())
print("Headers:", headers)

# Find relevant column names
user_col = next((col for col in headers if 'lc_user_phone_md5' in col), None)
channel_col = next((col for col in headers if 'lc_small_channel_name' in col), None)
lock_time_col = next((col for col in headers if 'lc_order_lock_time_min' in col), None)
create_time_col = next((col for col in headers if 'lc_create_time' in col), None)

print(f"Using User Column: {user_col}")
print(f"Using Channel Column: {channel_col}")
print(f"Using Lock Time Column: {lock_time_col}")
print(f"Using Create Time Column: {create_time_col}")

if not channel_col or not lock_time_col or not create_time_col or not user_col:
    print("Could not find required columns.")
    sys.exit(1)

# Group data by User
user_journeys = {}
for row in data:
    user_id = row.get(user_col)
    if not user_id: continue
    
    if user_id not in user_journeys:
        user_journeys[user_id] = []
    
    # Parse dates
    row['_parsed_create_date'] = parse_date(row.get(create_time_col))
    row['_parsed_lock_date'] = parse_date(row.get(lock_time_col))
    
    user_journeys[user_id].append(row)

# Filter for users who have at least one locked order
locked_users = []
for user_id, rows in user_journeys.items():
    # Check if any row has lock time
    has_lock = any(r.get(lock_time_col) and r.get(lock_time_col).strip() for r in rows)
    if has_lock:
        locked_users.append((user_id, rows))

print(f"\nTotal Users with Locked Orders: {len(locked_users)}")

# --- Analysis: Detailed User Journey Profiling ---
print("\n--- Typical User Journey Profiles ---")

journey_details = []

for user_id, rows in locked_users:
    # Sort rows by create time
    rows.sort(key=lambda x: x['_parsed_create_date'] if x['_parsed_create_date'] else datetime.min)
    
    # Find the earliest lock date for this user
    lock_dates = [r['_parsed_lock_date'] for r in rows if r['_parsed_lock_date']]
    if not lock_dates: continue
    first_lock_date = min(lock_dates)
    
    # Filter rows that happened BEFORE or ON the lock date
    path_rows = [r for r in rows if r['_parsed_create_date'] and r['_parsed_create_date'] <= first_lock_date]
    if not path_rows: path_rows = rows
    
    # Calculate duration
    first_touch = path_rows[0]['_parsed_create_date']
    if first_touch and first_lock_date:
        duration_days = (first_lock_date - first_touch).days
    else:
        duration_days = 0
        
    path = [r.get(channel_col) for r in path_rows]
    unique_channels = len(set(path))
    touch_count = len(path)
    
    journey_details.append({
        'user_id': user_id,
        'path': path,
        'duration_days': duration_days,
        'touch_count': touch_count,
        'unique_channels': unique_channels,
        'path_rows': path_rows # Keep full data for printing details
    })

# Define categories
one_touch = [j for j in journey_details if j['touch_count'] == 1]
hesitant = [j for j in journey_details if j['touch_count'] > 1 and j['unique_channels'] == 1]
cross_channel = [j for j in journey_details if j['unique_channels'] > 1]
long_duration = [j for j in journey_details if j['duration_days'] > 30]

print(f"Profile 1: One-Touch (Decisive) - {len(one_touch)} users ({len(one_touch)/len(locked_users):.1%})")
print(f"Profile 2: Hesitant (Same Channel, Multiple Touches) - {len(hesitant)} users ({len(hesitant)/len(locked_users):.1%})")
print(f"Profile 3: Cross-Channel (Comparison Shopper) - {len(cross_channel)} users ({len(cross_channel)/len(locked_users):.1%})")
print(f"Profile 4: Long Consideration (>30 Days) - {len(long_duration)} users ({len(long_duration)/len(locked_users):.1%})")

# Helper to print journey
def print_journey_example(title, journey_list, count=3):
    print(f"\n--- Examples: {title} ---")
    if not journey_list:
        print("No examples found.")
        return

    # Randomly sample to avoid bias, but seed for reproducibility
    random.seed(42)
    examples = random.sample(journey_list, min(count, len(journey_list)))
    
    for j in examples:
        print(f"User: {j['user_id'][:8]}... | Duration: {j['duration_days']} days | Touches: {j['touch_count']}")
        for r in j['path_rows']:
            date_str = r.get(create_time_col, 'Unknown Date')
            ch = r.get(channel_col, 'Unknown Channel')
            is_lock = "LOCK" if r.get(lock_time_col) else ""
            print(f"  - {date_str}: {ch} {is_lock}")
        print("-" * 30)

print_journey_example("Hesitant Users (Same Channel)", hesitant)
print_journey_example("Cross-Channel Users", cross_channel)
print_journey_example("Long Consideration Users", long_duration)

# Analyze Duration Distribution
durations = [j['duration_days'] for j in journey_details]
avg_duration = sum(durations) / len(durations)
median_duration = sorted(durations)[len(durations)//2]
print(f"\nAverage Duration to Lock: {avg_duration:.1f} days")
print(f"Median Duration to Lock: {median_duration} days")

# Duration buckets
buckets = Counter()
for d in durations:
    if d == 0: buckets['Same Day'] += 1
    elif d <= 7: buckets['1 Week'] += 1
    elif d <= 30: buckets['1 Month'] += 1
    elif d <= 90: buckets['3 Months'] += 1
    else: buckets['>3 Months'] += 1

print("\nDuration Distribution:")
for k, v in buckets.most_common():
    print(f"{k}: {v} ({v/len(journey_details):.1%})")
