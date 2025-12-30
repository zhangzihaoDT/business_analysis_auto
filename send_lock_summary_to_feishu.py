import os
import argparse
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import time

# 加载 .env 文件中的环境变量
load_dotenv()

def parse_markdown_table(table_text):
    """
    解析 Markdown 表格为 DataFrame
    """
    try:
        lines = table_text.strip().split('\n')
        valid_lines = []
        for l in lines:
            l = l.strip()
            if not l: continue
            # 忽略引用块和无管道符的行
            if l.startswith('>') or '|' not in l: continue
            
            # 过滤掉分隔行 (e.g. |---|---|)
            # 只有 | - : 空格 的行被视为分隔行
            # 注意：数据行可能包含这些字符，但通常还有其他字符
            if not set(l) <= {'|', '-', ':', ' '}:
                valid_lines.append(l)
        
        if not valid_lines:
            return pd.DataFrame()
        
        # 手动解析，比 read_csv 更稳健
        rows = []
        for l in valid_lines:
            # 按 | 分割
            parts = l.split('|')
            # 移除首尾可能的空字符串 (Markdown 表格通常以 | 开始和结束)
            if len(parts) > 0 and parts[0].strip() == '':
                parts.pop(0)
            if len(parts) > 0 and parts[-1].strip() == '':
                parts.pop(-1)
            # 去除单元格空格
            rows.append([p.strip() for p in parts])
            
        if not rows:
            return pd.DataFrame()
            
        header = rows[0]
        data = rows[1:]
        
        # 处理列数不一致的情况 (以表头为准)
        expected_cols = len(header)
        cleaned_data = []
        for r in data:
            if len(r) == expected_cols:
                cleaned_data.append(r)
            elif len(r) < expected_cols:
                # 补全
                cleaned_data.append(r + [''] * (expected_cols - len(r)))
            else:
                # 截断
                cleaned_data.append(r[:expected_cols])
                
        df = pd.DataFrame(cleaned_data, columns=header)
        return df
    except Exception as e:
        print(f"表格解析失败: {e}")
        return pd.DataFrame()

def render_df_as_columns(df):
    """
    将 DataFrame 渲染为飞书 ColumnSet 结构
    """
    if df.empty:
        return []
        
    elements = []
    
    # 1. 表头
    header_cols = []
    for col in df.columns:
        header_cols.append({
            "tag": "column",
            "width": "weighted",
            "weight": 1,
            "elements": [{
                "tag": "markdown",
                "content": f"**{col}**"
            }]
        })
    
    elements.append({
        "tag": "column_set",
        "flex_mode": "none",
        "background_style": "grey",
        "columns": header_cols
    })
    
    # 2. 数据行
    # 限制行数以防消息过大 (e.g. max 20 rows)
    MAX_ROWS = 20
    for idx, row in df.head(MAX_ROWS).iterrows():
        row_cols = []
        for val in row:
            row_cols.append({
                "tag": "column",
                "width": "weighted",
                "weight": 1,
                "elements": [{
                    "tag": "markdown",
                    "content": str(val)
                }]
            })
        elements.append({
            "tag": "column_set",
            "flex_mode": "none",
            "columns": row_cols
        })
        
    if len(df) > MAX_ROWS:
        elements.append({
             "tag": "div",
             "text": {
                 "tag": "plain_text",
                 "content": f"... (剩余 {len(df) - MAX_ROWS} 行已省略)"
             }
        })
        
    return elements

def extract_section(content, header):
    """
    提取指定标题下的内容
    """
    try:
        parts = content.split(f"## {header}")
        if len(parts) < 2:
            return ""
        # 查找下一个标题（以 ## 开头）或文件结束
        section = parts[1].split("\n## ")[0].strip()
        return section
    except Exception:
        return ""

def format_overview(content):
    """
    格式化概览统计
    """
    section = extract_section(content, "概览统计")
    if not section:
        return []
    
    df = parse_markdown_table(section)
    if df.empty:
        return []
    
    fields = []
    for _, row in df.iterrows():
        key = str(row.iloc[0])
        val = str(row.iloc[1])
        fields.append({
            "is_short": True,
            "text": {
                "tag": "lark_md",
                "content": f"**{key}**\n{val}"
            }
        })
    return fields

def format_table_section(content, header, title, emoji="📊"):
    """
    将表格部分格式化为飞书 ColumnSet，保持对齐
    """
    section = extract_section(content, header)
    if not section:
        return None
    
    # 尝试提取注释（引用块）
    lines = section.split('\n')
    notes = [l.strip('> ').strip() for l in lines if l.strip().startswith('>')]
    note_text = "\n".join(notes)
    
    # 解析表格
    df = parse_markdown_table(section)
    
    if df.empty:
        return None
        
    elements = []
    # 标题
    elements.append({
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": f"**{emoji} {title}**"
        }
    })
    
    # 表格内容 (ColumnSet)
    elements.extend(render_df_as_columns(df))
    
    # 注释（如果有）
    if note_text:
        elements.append({
            "tag": "note",
            "elements": [{
                "tag": "plain_text",
                "content": f"注: {note_text}"
            }]
        })
        
    return elements

def format_age_stats(content):
    """
    格式化车主年龄统计（列表转字段）
    """
    section = extract_section(content, "车主年龄统计")
    if not section:
        return None
        
    lines = [l.strip('- ').strip() for l in section.split('\n') if l.strip().startswith('-')]
    if not lines:
        return None
        
    fields = []
    for line in lines:
        if ':' in line:
            k, v = line.split(':', 1)
            fields.append({
                "is_short": True,
                "text": {
                    "tag": "lark_md",
                    "content": f"**{k.strip()}**\n{v.strip()}"
                }
            })
            
    if not fields:
        return None

    return {
        "tag": "div",
        "fields": fields
    }

def send_report(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"错误: 文件不存在 {file_path}")
        return

    content = file_path.read_text(encoding='utf-8')
    
    # 提取基本信息
    time_range = "未知"
    for line in content.split('\n'):
        if line.strip().startswith("- 时间区间:"):
            time_range = line.split(":", 1)[1].strip().replace('`', '')
            break
    
    def detect_models(txt):
        candidates = [
            "分年龄段的锁单量与占比（分车型占比%）",
            "分 license_city_level 的锁单量（分车型占比%）",
            "分 License Province 的锁单量（分车型占比%）",
            "分 License City 的锁单量（Top 10 Cities Breakdown）",
        ]
        for h in candidates:
            s = extract_section(txt, h)
            if not s:
                continue
            df = parse_markdown_table(s)
            if df.empty:
                continue
            cols = list(df.columns)
            if len(cols) >= 2:
                models = cols[1:]
            else:
                models = []
            models = [m.strip() for m in models if m.strip()]
            if models:
                return models
        return []
    
    def detect_sections(txt):
        headers_map = {
            "overview": ["概览统计（分车型）", "概览统计"],
            "deposit": ["大定留存的 Deposit_Payment_Time 分布（按日，分车型）", "大定留存的 Deposit_Payment_Time 分布（按日）"],
            "region": ["区域 x 业务定义矩阵", "区域 x 车型矩阵"],
            "city_level": ["分 license_city_level 的锁单量与占比", "分 license_city_level 的锁单量（分车型占比%）"],
            "province": ["分 License Province 的锁单量与占比", "分 License Province 的锁单量（分车型占比%）"],
            "city": ["分 License City 的锁单量与占比（Top 10）", "分 License City 的锁单量（Top 10 Cities Breakdown）"],
            "age": ["车主年龄统计", "车主年龄统计（分车型）", "分年龄段的锁单量与占比", "分年龄段的锁单量与占比（分车型占比%）"],
            "gender": ["分性别的锁单量与占比"],
        }
        included = []
        for key, headers in headers_map.items():
            for h in headers:
                if f"## {h}" in txt:
                    included.append(key)
                    break
        order = ["overview", "deposit", "region", "city_level", "province", "city", "age", "gender"]
        return [k for k in order if k in included]
    
    def build_title(models_list, sections_list):
        cn_map = {
            "overview": "概览",
            "deposit": "大定",
            "region": "地域",
            "city_level": "城市级别",
            "province": "省份",
            "city": "城市",
            "age": "年龄",
            "gender": "性别",
        }
        sec_cn = [cn_map.get(s, s) for s in sections_list]
        sec_text = "、".join(sec_cn) if sec_cn else "全部模块"
        if models_list:
            mod_text = "、".join(models_list[:4]) + (" 等" if len(models_list) > 4 else "")
        else:
            mod_text = "全部车型"
        return f"📊 锁单汇总｜模块：{sec_text}｜车型：{mod_text}"
    
    models = detect_models(content)
    sections = detect_sections(content)
    dynamic_title = build_title(models, sections)
    
    # 构建飞书卡片
    webhook = os.getenv("FS_WEBHOOK_URL")
    if not webhook:
        print("错误: 未找到 FS_WEBHOOK_URL 环境变量。")
        return

    card_elements = []

    # --- 1. 概览统计 (L4-13) ---
    card_elements.append({
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": f"🕒 **统计周期**\n{time_range}"
        }
    })
    
    overview_fields = format_overview(content)
    if overview_fields:
        card_elements.append({"tag": "hr"})
        card_elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": "**📈 核心指标**"
            }
        })
        card_elements.append({
            "tag": "div",
            "fields": overview_fields
        })

    # --- 2. 地域分布 (L147-183) ---
    card_elements.append({"tag": "hr"})
    card_elements.append({
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": "**🗺️ 地域分布**"
        }
    })

    # 城市级别
    level_elems = format_table_section(content, "分 license_city_level 的锁单量与占比", "城市级别", "🏙️")
    if level_elems:
        card_elements.extend(level_elems)

    level_model_elems = format_table_section(content, "分 license_city_level 的锁单量（分车型占比%）", "城市级别（分车型%）", "🚙")
    if level_model_elems:
        card_elements.extend(level_model_elems)

    # Top 省份
    # 注意：lock_summary.py 中标题为 "分 License Province 的锁单量与占比"
    prov_elems = format_table_section(content, "分 License Province 的锁单量与占比", "Top 省份", "🏛️")
    if prov_elems:
        card_elements.extend(prov_elems)
    
    prov_model_elems = format_table_section(content, "分 License Province 的锁单量（分车型占比%）", "Top 省份（分车型%）", "🚙")
    if prov_model_elems:
        card_elements.extend(prov_model_elems)

    # Top 城市
    city_elems = format_table_section(content, "分 License City 的锁单量与占比（Top 10）", "Top 10 城市", "🌆")
    if city_elems:
        card_elements.extend(city_elems)
        
    city_model_elems = format_table_section(content, "分 License City 的锁单量（Top 10 Cities Breakdown）", "Top 10 城市（分车型）", "🚙")
    if city_model_elems:
        card_elements.extend(city_model_elems)

    # --- 3. 用户画像 (L184-208) ---
    card_elements.append({"tag": "hr"})
    card_elements.append({
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": "**👥 用户画像**"
        }
    })

    # 年龄统计 (均值/中位数)
    age_stats_elem = format_age_stats(content)
    if age_stats_elem:
        card_elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": "**🎂 年龄概览**"
            }
        })
        card_elements.append(age_stats_elem)

    # 年龄段分布
    age_dist_elems = format_table_section(content, "分年龄段的锁单量与占比", "年龄段分布", "📊")
    if age_dist_elems:
        card_elements.extend(age_dist_elems)

    age_model_elems = format_table_section(content, "分年龄段的锁单量与占比（分车型占比%）", "年龄段分布（分车型%）", "🚙")
    if age_model_elems:
        card_elements.extend(age_model_elems)

    # 性别分布
    gender_elems = format_table_section(content, "分性别的锁单量与占比", "性别分布", "👫")
    if gender_elems:
        card_elements.extend(gender_elems)

    # 底部
    card_elements.append({"tag": "hr"})
    card_elements.append({
        "tag": "note",
        "elements": [
            {
                "tag": "plain_text",
                "content": f"数据来源: {file_path.name}"
            }
        ]
    })

    card_data = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": dynamic_title
                },
                "template": "blue"
            },
            "elements": card_elements
        }
    }

    def try_send_with_retry(webhook_url, payload, retries=(2, 5, 10)):
        print("正在发送飞书消息...")
        for i, wait_s in enumerate([0] + list(retries)):
            if wait_s > 0:
                print(f"频率限制或异常，{wait_s}s 后重试（第 {i} 次）...")
                time.sleep(wait_s)
            try:
                resp = requests.post(webhook_url, json=payload)
                resp.raise_for_status()
                result = {}
                try:
                    result = resp.json()
                except Exception:
                    pass
                status_ok = (result.get("StatusCode") == 0) or (result.get("code") == 0)
                if status_ok:
                    print(f"✅ 消息推送成功")
                    return True
                msg = result.get("msg", "")
                code = result.get("code")
                print(f"❌ 消息推送异常: {result}")
                # 11232: frequency limited
                if code == 11232 or ("frequency limited" in msg.lower()):
                    continue
                # 其他错误不重试
                break
            except requests.exceptions.RequestException as e:
                print(f"❌ 网络异常: {e}")
                continue
        return False
    
    ok = try_send_with_retry(webhook, card_data)
    if not ok:
        # 兜底文本消息，包含标题信息
        fallback_text = dynamic_title
        try:
            requests.post(webhook, json={
                "msg_type": "text",
                "content": {"text": fallback_text}
            })
        except requests.exceptions.RequestException as e:
            print(f"❌ 文本消息推送失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="发送锁单汇总报告到飞书")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--file", 
        help="指定报告文件名 (需完整文件名，例如: lock_summary_2024-01-01_to_2025-12-21.md)。脚本将优先在 processed/analysis_results/ 目录下查找，也可以提供文件路径。"
    )
    group.add_argument(
        "--latest",
        action="store_true",
        help="自动选择 processed/analysis_results/ 目录下最新的 lock_summary_*.md 报告"
    )

    args = parser.parse_args()
    
    base_dir = Path("processed/analysis_results")
    default_file = base_dir / "lock_summary_2024-01-01_to_2025-12-21.md"

    if args.latest:
        files = list(base_dir.glob("lock_summary_*.md"))
        if not files:
            print(f"错误: 在 {base_dir} 下未找到任何 lock_summary 报告")
            return
        file_path = max(files, key=os.path.getmtime)
        print(f"已选择最新报告: {file_path}")
    elif args.file:
        # 1. 优先在默认目录下查找文件名
        candidate = base_dir / args.file
        if candidate.exists():
            file_path = candidate
        else:
            # 2. 检查是否为提供的直接路径
            candidate = Path(args.file)
            if candidate.exists():
                file_path = candidate
            else:
                print(f"错误: 未找到文件 '{args.file}'")
                print(f"  - 已尝试目录: {base_dir}")
                print(f"  - 已尝试路径: {Path(args.file).absolute()}")
                return
    else:
        file_path = default_file
        
    send_report(file_path)

if __name__ == "__main__":
    main()
