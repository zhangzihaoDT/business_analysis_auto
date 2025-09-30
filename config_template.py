#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件模板

请复制此文件为 config.py 并填入您的实际配置信息
注意：config.py 已在 .gitignore 中，不会被提交到 Git 仓库
"""

# Tableau 配置
TABLEAU_CONFIG = {
    "token_name": "YOUR_TOKEN_NAME",  # 替换为您的 Tableau 令牌名称
    "token_value": "YOUR_TABLEAU_PAT_TOKEN",  # 替换为您的 Tableau 个人访问令牌
    "server_url": "YOUR_TABLEAU_SERVER_URL",  # 替换为您的 Tableau 服务器地址
    "site_id": "YOUR_SITE_ID",  # 替换为您的站点 ID（可选）
    "timeout": 300  # 超时时间（秒）
}

# 数据文件路径配置
DATA_PATHS = {
    "original_dir": "../original/",
    "formatted_dir": "../formatted/",
    "reports_dir": "../reports/"
}

# Flomo 配置
FLOMO_CONFIG = {
    "api_url": "YOUR_FLOMO_API_URL"  # 替换为您的 Flomo API URL
}

# 使用示例：
# from config import TABLEAU_CONFIG
# token_name = TABLEAU_CONFIG["token_name"]
# token_value = TABLEAU_CONFIG["token_value"]