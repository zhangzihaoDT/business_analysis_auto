import csv
import argparse
from datetime import date, datetime, timedelta

def parse_date(s: str) -> date:
    return datetime.strptime(s.strip(), '%Y-%m-%d').date()

def normalize_week(week_str: str):
    s = week_str.replace('～', '~').strip()
    parts = s.split('~')
    if len(parts) != 2:
        return None, None, None
    start_s, end_s = parts[0].strip(), parts[1].strip()
    if len(end_s) == 5 and end_s[2] == '-':
        end_s = f"{start_s[:4]}-{end_s}"
    try:
        start = parse_date(start_s)
        end = parse_date(end_s)
    except Exception:
        return None, None, None
    return f"{start.strftime('%Y-%m-%d')}~{end.strftime('%Y-%m-%d')}", start, end

def week_value(start: date) -> str:
    y = start.year
    d = date(y, 1, 1)
    offset = (0 - d.weekday()) % 7
    first_monday = d + timedelta(days=offset)
    num = ((start - first_monday).days // 7) + 1
    return f"{str(y % 100).zfill(2)} 年 {str(num).zfill(2)} 周"

def process(path: str):
    rows = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    normalized = []
    for r in rows:
        wk = r.get('Week', '').strip()
        wk_norm, start, end = normalize_week(wk)
        if not wk_norm:
            continue
        val = week_value(start)
        normalized.append({
            'Week_original': wk,
            'Week_normalized': wk_norm,
            'Week_value': val,
            'Week_start': start,
            '车系': r.get('车系', ''),
            '品牌': r.get('品牌', ''),
            'PK正向排名': r.get('PK正向排名', ''),
        })
    print('总记录数:', len(rows))
    print('成功规范 Week 的记录数:', len(normalized))
    example = '2024-12-30~2025-01-05'
    ex_norm, ex_start, ex_end = normalize_week(example)
    print('示例区间 -> 值表达:', example, '=>', week_value(ex_start))
    print('\n示例前10条规范结果:')
    for r in normalized[:10]:
        print(r['Week_original'], '=>', r['Week_normalized'], '=>', r['Week_value'])
    from collections import Counter
    def summarize_window(title: str, start_bound: date, end_bound: date):
        cnt = Counter()
        for r in normalized:
            try:
                rank = int(str(r['PK正向排名']).replace(',', '').strip())
            except Exception:
                continue
            d = r.get('Week_start')
            if not isinstance(d, date):
                continue
            if start_bound <= d <= end_bound and rank == 1:
                key = (r['品牌'], r['车系'])
                cnt[key] += 1
        print(f"\nPK正向排名为1的品牌-车系上榜次数（{title}）:")
        for (brand, series), n in sorted(cnt.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
            print(f"{brand}\t{series}\t{n}")

    summarize_window('2024-12-01 至 2025-11-30', date(2024,12,1), date(2025,11,30))
    summarize_window('2023-12-01 至 2024-11-30', date(2023,12,1), date(2024,11,30))
    return normalized

def write_output(output_path: str, source_path: str, normalized_rows):
    if not output_path:
        return
    with open(source_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
    extra = ['Week_normalized', 'Week_value']
    out_fields = list(fieldnames) + [c for c in extra if c not in fieldnames]
    index = {}
    for r in normalized_rows:
        index[r['Week_original'], r['品牌'], r['车系']] = r
    with open(source_path, 'r', encoding='utf-8-sig') as f_in, open(output_path, 'w', encoding='utf-8', newline='') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()
        for r in reader:
            key = (r.get('Week',''), r.get('品牌',''), r.get('车系',''))
            n = index.get(key)
            if n:
                r['Week_normalized'] = n['Week_normalized']
                r['Week_value'] = n['Week_value']
            writer.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='/Users/zihao_/Documents/coding/dataset/original/业务数据记录_竞争PK（LS6）_表格.csv')
    ap.add_argument('--output', default='')
    args = ap.parse_args()
    normalized_rows = process(args.input)
    write_output(args.output, args.input, normalized_rows)

if __name__ == '__main__':
    main()
