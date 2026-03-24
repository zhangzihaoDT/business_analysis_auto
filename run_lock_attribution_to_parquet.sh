#!/bin/bash

TARGET_PRIMARY="/Users/zihao_/Documents/coding/dataset/launchd_daemon/run_lock_attribution_to_parquet.sh"
TARGET_FALLBACK="/Users/zihao_/Documents/coding/dataset/scripts/launchd_daemon/run_lock_attribution_to_parquet.sh"

if [ -f "${TARGET_PRIMARY}" ]; then
    exec /bin/bash "${TARGET_PRIMARY}" "$@"
fi

if [ -f "${TARGET_FALLBACK}" ]; then
    exec /bin/bash "${TARGET_FALLBACK}" "$@"
fi

echo "wrapper target not found: ${TARGET_PRIMARY} / ${TARGET_FALLBACK}" >&2
exit 127
