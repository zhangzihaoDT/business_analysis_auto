#!/bin/bash

set -o pipefail

SCHEDULE_HOUR=8
SCHEDULE_MINUTE=45
LOG_FILE="/Users/zihao_/Documents/coding/dataset/logs/cron_pipeline.log"
LAST_RUN_FILE="/tmp/data_pipeline.last_run_date"
FORCE_RUN="${FORCE_RUN:-0}"

echo "🚀 pipeline (wrapper) start: $(date)" | tee -a "${LOG_FILE}"

if [ "${FORCE_RUN}" != "1" ]; then
    today="$(date +%Y-%m-%d)"
    now_hm="$(date +%H%M)"
    target_hm="$(printf "%02d%02d" "${SCHEDULE_HOUR}" "${SCHEDULE_MINUTE}")"

    if [ -f "${LAST_RUN_FILE}" ] && [ "$(cat "${LAST_RUN_FILE}" 2>/dev/null)" = "${today}" ]; then
        echo "⏭️  skip: already ran today (${today})" | tee -a "${LOG_FILE}"
        exit 0
    fi

    if [ "${now_hm}" -lt "${target_hm}" ]; then
        echo "⏭️  skip: before schedule (now ${now_hm} < target ${target_hm})" | tee -a "${LOG_FILE}"
        exit 0
    fi
fi

LOCKFILE="/tmp/data_pipeline.lock"
if [ -e ${LOCKFILE} ] && kill -0 `cat ${LOCKFILE}` 2>/dev/null; then
    echo "⚠️  脚本已在运行中 (PID: $(cat ${LOCKFILE}))" | tee -a "${LOG_FILE}"
    echo "❌ 本次启动已取消，避免冲突。" | tee -a "${LOG_FILE}"
    if [ -t 1 ]; then
        sleep 5
    fi
    exit 0
fi

trap "rm -f ${LOCKFILE}" INT TERM EXIT
echo $$ > ${LOCKFILE}

source /Users/zihao_/Documents/coding/dataset/venv/bin/activate
cd /Users/zihao_/Documents/coding/dataset

if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -dimsu -w $$ &
fi

echo "🚀 开始运行自动化数据处理流水线..." | tee -a "${LOG_FILE}"
echo "⏰ 开始时间: $(date)" | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"

export PYTHONUNBUFFERED=1

python -u scripts/automated_data_pipeline.py 2>&1 | sed -E -e 's/(--token-value[= ]+)[^ ]+/\1***REDACTED***/g' -e 's#(https://flomoapp\\.com/iwh/)[^ ]+#\\1***REDACTED***#g' | tee -a /Users/zihao_/Documents/coding/dataset/logs/cron_pipeline.log
pipeline_status="${PIPESTATUS[0]}"
if [ "${pipeline_status}" -eq 0 ] && [ "${FORCE_RUN}" != "1" ]; then
    echo "$(date +%Y-%m-%d)" > "${LAST_RUN_FILE}"
fi

echo "----------------------------------------" | tee -a "${LOG_FILE}"
echo "✅ 任务完成!" | tee -a "${LOG_FILE}"
echo "⏰ 结束时间: $(date)" | tee -a "${LOG_FILE}"
echo "日志已保存至: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "🏁 pipeline (wrapper) end: $(date) (exit=${pipeline_status})" | tee -a "${LOG_FILE}"
exit "${pipeline_status}"
