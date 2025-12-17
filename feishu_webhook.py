import os
import requests
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

webhook = os.getenv("FS_WEBHOOK_URL")

if not webhook:
    print("错误: 未找到 FS_WEBHOOK_URL 环境变量。请检查 .env 文件。")
    exit(1)

data = {
    "msg_type": "interactive",
    "card": {
        "header": {
            "title": {
                "tag": "plain_text",
                "content": "🚀 部署通知: W50 Push"
            },
            "template": "blue"
        },
        "elements": [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": "**当前状态：** ✅ 测试通过\n**执行环境：** macOS / Python 3.13\n**更新内容：**\n- 升级为消息卡片格式\n- 增加环境变量支持"
                }
            },
            {
                "tag": "hr"
            },
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": "来自自动测试脚本"
                    }
                ]
            },
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {
                            "tag": "plain_text",
                            "content": "查看代码仓库"
                        },
                        "url": "https://github.com/zihao/W50_push",
                        "type": "primary"
                    }
                ]
            }
        ]
    }
}

try:
    response = requests.post(webhook, json=data)
    response.raise_for_status() # 检查 HTTP 错误
    result = response.json()
    if result.get("StatusCode") == 0:
         print(f"消息推送状态: {result.get('StatusMessage')}")
    else:
         print(f"消息推送异常: {result}")
except requests.exceptions.RequestException as e:
    print(f"消息推送失败: {e}")
