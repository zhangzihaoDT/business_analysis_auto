import os
import argparse
import time
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
    格式化数据概览
    """
    section = extract_section(content, "数据概览")
    if not section:
        return None
    
    df = parse_markdown_table(section)
    if df.empty:
        return None
    
    # 转换为 Feishu 字段格式
    fields = []
    
    # 如果是包含 "用户类型", "锁单数", "交付数" 的表
    if "用户类型" in df.columns:
        for _, row in df.iterrows():
            user_type = str(row["用户类型"])
            locked = str(row["锁单数"])
            delivered = str(row["交付数"])
            
            fields.append({
                "is_short": True,
                "text": {
                    "tag": "lark_md",
                    "content": f"**{user_type}**\n🔒 {locked} | 🚚 {delivered}"
                }
            })
    # 兼容旧格式或无 Staff info 的格式
    elif "指标" in df.columns:
         for _, row in df.iterrows():
            key = str(row["指标"])
            val = str(row["数量"])
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

def main():
    parser = argparse.ArgumentParser(description="发送配置汇总报告到飞书")
    # 变更为必须的参数，接受文件名或路径
    parser.add_argument("file_path", help="指定报告文件名 (processed/analysis_results/ 下的文件名或完整路径)")
    
    args = parser.parse_args()
    
    base_dir = Path("processed/analysis_results")
    
    # 处理文件路径
    candidate = Path(args.file_path)
    
    # 1. 如果是直接存在的路径（绝对或相对）
    if candidate.exists():
        file_path = candidate
    # 2. 如果只是文件名，尝试在默认目录下查找
    elif (base_dir / args.file_path).exists():
        file_path = base_dir / args.file_path
    else:
        print(f"❌ 错误: 未找到文件 '{args.file_path}'")
        print(f"  - 请确认文件路径是否正确，或文件是否在 {base_dir} 目录下")
        return

    print(f"✅ 准备推送报告: {file_path}")

    content = file_path.read_text(encoding='utf-8')
    
    # 提取基本信息
    # 标题通常是第一行 "# CM2 配置情况分析报告"
    lines = content.split('\n')
    title_line = [l for l in lines if l.startswith('# ')][0]
    report_title = title_line.replace('# ', '').strip()
    
    time_range = "未知"
    for line in lines:
        if "数据时间范围" in line:
            time_range = line.split(":", 1)[1].strip().replace('`', '')
            break
            
    # 构建飞书卡片
    webhook = os.getenv("FS_WEBHOOK_URL")
    if not webhook:
        print("错误: 未找到 FS_WEBHOOK_URL 环境变量。")
        return

    card_elements = []

    # --- 1. 基本信息 ---
    card_elements.append({
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": f"🕒 **统计周期**\n{time_range}"
        }
    })
    
    # --- 2. 数据概览 ---
    overview_fields = format_overview(content)
    if overview_fields:
        card_elements.append({"tag": "hr"})
        card_elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": "**📈 数据概览**"
            }
        })
        card_elements.append({
            "tag": "div",
            "fields": overview_fields
        })

    # --- 2.1 配置数据完整度 ---
    completeness_elems = format_table_section(content, "配置数据完整度", "配置数据完整度", "✅")
    if completeness_elems:
        card_elements.append({"tag": "hr"})
        card_elements.extend(completeness_elems)

    # --- 3. 激光雷达整体分布 ---
    laser_elems = format_table_section(content, "激光雷达 (OP-LASER) 整体分布", "激光雷达整体分布", "🎯")
    if laser_elems:
        card_elements.append({"tag": "hr"})
        card_elements.extend(laser_elems)

    # --- 4. 分员工单分布 (激光雷达) ---
    staff_elems = format_table_section(content, "分员工单 (Is Staff) 激光雷达分布", "员工单激光雷达分布", "👥")
    if staff_elems:
        card_elements.append({"tag": "hr"})
        card_elements.extend(staff_elems)
        
    # --- 5. 分车型分布 (激光雷达) ---
    # 标题可能是 "分车型 (Product Name) 高阶+Thor 分布"
    # 我们需要找到包含 "分车型" 且包含 "Thor" 的标题 (为了区分轮毂)
    laser_model_header = None
    for line in lines:
        if line.startswith("## 分车型") and "Thor" in line:
            laser_model_header = line.replace("## ", "").strip()
            break
            
    if laser_model_header:
        model_elems = format_table_section(content, laser_model_header, laser_model_header, "🚗")
        if model_elems:
            card_elements.append({"tag": "hr"})
            card_elements.extend(model_elems)

    # --- 6. 轮毂 (WHEEL) 整体分布 ---
    wheel_elems = format_table_section(content, "轮毂 (WHEEL) 整体分布", "轮毂整体分布", "🛞")
    if wheel_elems:
        card_elements.append({"tag": "hr"})
        card_elements.extend(wheel_elems)

    # --- 7. 分员工单分布 (轮毂) ---
    wheel_staff_elems = format_table_section(content, "分员工单 (Is Staff) 轮毂分布", "员工单轮毂分布", "👥")
    if wheel_staff_elems:
        card_elements.append({"tag": "hr"})
        card_elements.extend(wheel_staff_elems)

    # --- 8. 分车型分布 (轮毂) ---
    wheel_model_header = None
    for line in lines:
        if line.startswith("## 分车型") and "轮毂" in line:
            wheel_model_header = line.replace("## ", "").strip()
            break
    
    if wheel_model_header:
        wheel_model_elems = format_table_section(content, wheel_model_header, wheel_model_header, "🚗")
        if wheel_model_elems:
            card_elements.append({"tag": "hr"})
            card_elements.extend(wheel_model_elems)

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
                    "content": f"📊 {report_title}"
                },
                "template": "blue"
            },
            "elements": card_elements
        }
    }

    max_retries = 3
    print("正在发送飞书消息...")
    for attempt in range(max_retries):
        try:
            response = requests.post(webhook, json=card_data)
            response.raise_for_status()
            result = response.json()
            
            # 兼容 StatusCode 和 code
            code = result.get("StatusCode")
            if code is None:
                code = result.get("code")
                
            if code == 0:
                print(f"✅ 消息推送成功: {result.get('StatusMessage', '')}")
                return
            elif code == 11232: # Frequency limited
                wait_time = 2 * (attempt + 1)
                print(f"⚠️ 飞书消息发送频率限制 (11232)，等待 {wait_time} 秒后重试 ({attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            else:
                print(f"❌ 消息推送异常: {result}")
                return
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 消息推送失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                 print("❌ 重试次数耗尽，发送失败")

if __name__ == "__main__":
    main()
