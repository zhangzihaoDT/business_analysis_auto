import os
import argparse
import requests
import pandas as pd
import io
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

def parse_markdown_table(table_text):
    """
    简单的 Markdown 表格解析器，返回 DataFrame
    """
    try:
        # 移除 markdown 表格的分隔行 (e.g. |---|---|)
        lines = table_text.strip().split('\n')
        lines = [l for l in lines if not set(l.strip()) <= {'|', '-', ':', ' '}]
        if not lines:
            return pd.DataFrame()
        
        # 使用 pandas 读取
        df = pd.read_csv(io.StringIO('\n'.join(lines)), sep='|', skipinitialspace=True)
        # 清理列名和数据（移除首尾空格和空列）
        df = df.dropna(axis=1, how='all')
        df.columns = [c.strip() for c in df.columns]
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        return df
    except Exception as e:
        print(f"表格解析失败: {e}")
        return pd.DataFrame()

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
    将表格部分格式化为代码块，保持对齐
    """
    section = extract_section(content, header)
    if not section:
        return None
    
    # 保留表格行
    lines = section.split('\n')
    table_lines = [l for l in lines if '|' in l]
    
    # 尝试提取注释（引用块）
    notes = [l.strip('> ').strip() for l in lines if l.strip().startswith('>')]
    note_text = "\n".join(notes)
    
    if not table_lines:
        return None
        
    text_content = "\n".join(table_lines)
    
    elements = []
    # 标题
    elements.append({
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": f"**{emoji} {title}**"
        }
    })
    
    # 表格内容
    elements.append({
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": f"```text\n{text_content}\n```"
        }
    })
    
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

    # Top 省份
    prov_elems = format_table_section(content, "分 License Province 的锁单量与占比（Top 10）", "Top 10 省份", "🏛️")
    if prov_elems:
        card_elements.extend(prov_elems)

    # Top 城市
    city_elems = format_table_section(content, "分 License City 的锁单量与占比（Top 10）", "Top 10 城市", "🌆")
    if city_elems:
        card_elements.extend(city_elems)

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
                    "content": "📊 锁单汇总报告"
                },
                "template": "blue"
            },
            "elements": card_elements
        }
    }

    try:
        print("正在发送飞书消息...")
        response = requests.post(webhook, json=card_data)
        response.raise_for_status()
        result = response.json()
        if result.get("StatusCode") == 0:
            print(f"✅ 消息推送成功: {result.get('StatusMessage')}")
        else:
            print(f"❌ 消息推送异常: {result}")
            # 简单兜底
            requests.post(webhook, json={
                "msg_type": "text",
                "content": {"text": f"锁单汇总报告\n时间: {time_range}\n(卡片渲染失败)"}
            })
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 消息推送失败: {e}")

if __name__ == "__main__":
    main()
