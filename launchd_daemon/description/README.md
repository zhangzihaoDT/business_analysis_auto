# 说明文档

本目录用于存放 `scripts/launchd_daemon/` 相关的说明文档（description），方便统一维护。

## 日志目录说明

实际日志目录固定为：

- `/Users/zihao_/Documents/coding/dataset/logs/`

该目录会被 LaunchDaemon 直接写入（`StandardOutPath`/`StandardErrorPath` 使用绝对路径），不建议移动。

### 主要日志

- **pipeline 业务主日志（推荐优先看）**
  - `cron_pipeline.log`

- **pipeline daemon 日志**
  - `launchd_pipeline.daemon.stdout.log`
  - `launchd_pipeline.daemon.stderr.log`

- **lock-attribution daemon 日志**
  - `launchd_lock_attribution.daemon.stdout.log`
  - `launchd_lock_attribution.daemon.stderr.log`

