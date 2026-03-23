#!/bin/bash

set -o pipefail

SCHEDULE_HOUR=8
SCHEDULE_MINUTE=15
LAST_RUN_FILE="/tmp/lock_attribution_to_parquet.last_run_date"
FORCE_RUN="${FORCE_RUN:-0}"

echo "🚀 lock-attribution-to-parquet (wrapper) start: $(date)"

if [ "${FORCE_RUN}" != "1" ]; then
    today="$(date +%Y-%m-%d)"
    now_hm="$(date +%H%M)"
    target_hm="$(printf "%02d%02d" "${SCHEDULE_HOUR}" "${SCHEDULE_MINUTE}")"

    if [ -f "${LAST_RUN_FILE}" ] && [ "$(cat "${LAST_RUN_FILE}" 2>/dev/null)" = "${today}" ]; then
        echo "⏭️  skip: already ran today (${today})"
        exit 0
    fi

    if [ "${now_hm}" -lt "${target_hm}" ]; then
        echo "⏭️  skip: before schedule (now ${now_hm} < target ${target_hm})"
        exit 0
    fi
fi

LOCKFILE="/tmp/lock_attribution_to_parquet.lock"
if [ -e "${LOCKFILE}" ] && kill -0 "$(cat "${LOCKFILE}")" 2>/dev/null; then
    pid="$(cat "${LOCKFILE}")"
    echo "already running (PID: ${pid})"
    exit 0
fi

trap "rm -f ${LOCKFILE}" INT TERM EXIT
echo $$ > "${LOCKFILE}"

source /Users/zihao_/Documents/coding/dataset/venv/bin/activate
cd /Users/zihao_/Documents/coding/dataset

if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -dimsu -w $$ &
fi

export PYTHONUNBUFFERED=1
python -u scripts/lock_attribution_data_to_parquet.py
status=$?

if [ "${status}" -eq 0 ] && [ "${FORCE_RUN}" != "1" ]; then
    echo "$(date +%Y-%m-%d)" > "${LAST_RUN_FILE}"
fi

echo "🏁 lock-attribution-to-parquet (wrapper) end: $(date) (exit=${status})"
exit "${status}"
