# LaunchDaemon 自动化运行说明

本目录用于集中管理与 macOS `launchd` 自动化相关的配置与脚本，当前采用 **LaunchDaemon（system 域）** 方案，满足“合盖也稳定执行”的需求。

## 运行目标与时间

- **锁单归因 → Parquet**
  - 任务：运行 `scripts/lock_attribution_data_to_parquet.py`
  - 计划：每天 **08:15** 触发
  - 补跑：每 **300 秒** 检查一次（错过时间点后可补跑）
  - 防重复：同一天成功后写入 `/tmp/lock_attribution_to_parquet.last_run_date`
  - 防并发：`/tmp/lock_attribution_to_parquet.lock`

- **自动化数据流水线（pipeline）**
  - 任务：运行 `scripts/automated_data_pipeline.py`
  - 计划：每天 **08:45** 触发
  - 补跑：每 **300 秒** 检查一次（错过时间点后可补跑）
  - 防重复：同一天成功后写入 `/tmp/data_pipeline.last_run_date`
  - 防并发：`/tmp/data_pipeline.lock`

## 文件结构

- **Daemon 配置（模板）**
  - `com.zihao.dataset.lock-attribution-to-parquet.daemon.plist`
  - `com.zihao.dataset.pipeline.daemon.plist`

- **Agent 配置（模板，保留备用）**
  - `com.zihao.dataset.lock-attribution-to-parquet.plist`
  - `com.zihao.dataset.pipeline.plist`

- **包装脚本（统一入口）**
  - `run_lock_attribution_to_parquet.sh`
  - `run_pipeline_cron.sh`

- **日志说明**
  - `description/README.md`

## 安装与生效（system 域）

以下命令会把模板安装到 `/Library/LaunchDaemons/` 并加载：

```bash
sudo cp /Users/zihao_/Documents/coding/dataset/scripts/launchd_daemon/com.zihao.dataset.lock-attribution-to-parquet.daemon.plist \
  /Library/LaunchDaemons/com.zihao.dataset.lock-attribution-to-parquet.daemon.plist
sudo chown root:wheel /Library/LaunchDaemons/com.zihao.dataset.lock-attribution-to-parquet.daemon.plist
sudo chmod 644 /Library/LaunchDaemons/com.zihao.dataset.lock-attribution-to-parquet.daemon.plist
sudo launchctl bootout system /Library/LaunchDaemons/com.zihao.dataset.lock-attribution-to-parquet.daemon.plist 2>/dev/null || true
sudo launchctl bootstrap system /Library/LaunchDaemons/com.zihao.dataset.lock-attribution-to-parquet.daemon.plist

sudo cp /Users/zihao_/Documents/coding/dataset/scripts/launchd_daemon/com.zihao.dataset.pipeline.daemon.plist \
  /Library/LaunchDaemons/com.zihao.dataset.pipeline.daemon.plist
sudo chown root:wheel /Library/LaunchDaemons/com.zihao.dataset.pipeline.daemon.plist
sudo chmod 644 /Library/LaunchDaemons/com.zihao.dataset.pipeline.daemon.plist
sudo launchctl bootout system /Library/LaunchDaemons/com.zihao.dataset.pipeline.daemon.plist 2>/dev/null || true
sudo launchctl bootstrap system /Library/LaunchDaemons/com.zihao.dataset.pipeline.daemon.plist
```

## 日常运维

- **查看状态**
  - `sudo launchctl print system/com.zihao.dataset.lock-attribution-to-parquet.daemon`
  - `sudo launchctl print system/com.zihao.dataset.pipeline.daemon`

- **手动触发（立即跑一轮）**
  - `sudo launchctl kickstart -k system/com.zihao.dataset.lock-attribution-to-parquet.daemon`
  - `sudo launchctl kickstart -k system/com.zihao.dataset.pipeline.daemon`

- **强制重跑（绕过每日去重）**
  - 删除当日标记：`sudo rm -f /tmp/lock_attribution_to_parquet.last_run_date /tmp/data_pipeline.last_run_date`
