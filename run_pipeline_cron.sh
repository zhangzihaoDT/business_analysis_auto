#!/bin/bash

# --- 单例模式保护 ---
# 防止脚本重复运行 (例如 cron 配置错误导致同时跑了两个)
LOCKFILE="/tmp/data_pipeline.lock"
if [ -e ${LOCKFILE} ] && kill -0 `cat ${LOCKFILE}` 2>/dev/null; then
    echo "⚠️  脚本已在运行中 (PID: $(cat ${LOCKFILE}))"
    echo "❌ 本次启动已取消，避免冲突。"
    if [ -t 1 ]; then
        sleep 5
    fi
    exit 0
fi

# 注册清理函数：脚本退出时自动删除锁文件
trap "rm -f ${LOCKFILE}" INT TERM EXIT
echo $$ > ${LOCKFILE}
# ------------------

source /Users/zihao_/Documents/coding/dataset/venv/bin/activate
cd /Users/zihao_/Documents/coding/dataset

if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -dimsu -w $$ &
fi

echo "🚀 开始运行自动化数据处理流水线..."
echo "⏰ 开始时间: $(date)"
echo "----------------------------------------"

# 强制开启 Python 无缓冲模式，确保实时输出
export PYTHONUNBUFFERED=1

# 运行主脚本
# -u 参数进一步确保 stdout/stderr 不被缓冲
# tee -a 同时显示在屏幕并追加到日志
python -u scripts/automated_data_pipeline.py 2>&1 | sed -E -e 's/(--token-value[= ]+)[^ ]+/\1***REDACTED***/g' -e 's#(https://flomoapp\\.com/iwh/)[^ ]+#\\1***REDACTED***#g' | tee -a /Users/zihao_/Documents/coding/dataset/logs/cron_pipeline.log

echo "----------------------------------------"
echo "✅ 任务完成!"
echo "⏰ 结束时间: $(date)"
echo "日志已保存至: /Users/zihao_/Documents/coding/dataset/logs/cron_pipeline.log"
echo "您可以手动关闭此窗口。"
