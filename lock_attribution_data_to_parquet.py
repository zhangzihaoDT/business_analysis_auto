#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
锁单归因数据自动更新流水线

1. 从 Tableau 导出最新视图数据到 original/锁单归因_data_<year>.csv
2. 合并 original/锁单归因_data_20*.csv（每年取最新的一个）并更新 Parquet

默认运行: python scripts/lock_attribution_data_to_parquet.py
仅做 step2: python scripts/lock_attribution_data_to_parquet.py --no-export
仅更新 assign/test_drive: python scripts/lock_attribution_data_to_parquet.py --only-assign-test-drive

Tableau PAT 从 scripts/.env 读取:
- TABLEAU_TOKEN_NAME="..."
- TABLEAU_TOKEN_VALUE="..."

输入文件:
- original/锁单归因_data_20*.csv（按年份分组，每年取最新的一个）
输出文件:
- formatted/lock_attribution_data.parquet
"""

from __future__ import annotations

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


BASE_DIR = Path("/Users/zihao_/Documents/coding/dataset")
SCRIPTS_DIR = BASE_DIR / "scripts"
ORIGINAL_DIR = BASE_DIR / "original"
FORMATTED_DIR = BASE_DIR / "formatted"
OUTPUT_FILE = FORMATTED_DIR / "lock_attribution_data.parquet"


def read_env_file(file_path: Path) -> dict[str, str]:
    if not file_path.exists():
        return {}

    env: dict[str, str] = {}
    for raw_line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
            value = value[1:-1]
        if key:
            env[key] = value
    return env


def normalize_phone_md5(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    s = s.replace(
        {
            "nan": pd.NA,
            "none": pd.NA,
            "null": pd.NA,
            "": pd.NA,
            "-": pd.NA,
        }
    )
    valid = s.str.match(r"^[0-9a-f]{32}$", na=False)
    return s.where(valid, pd.NA).astype("string")


def read_csv_smart(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        print(f"⚠️ 文件不存在: {file_path}")
        return pd.DataFrame()

    try:
        if file_path.stat().st_size == 0:
            print(f"❌ 文件为空(0字节): {file_path}")
            return pd.DataFrame()
    except Exception:
        pass

    print(f"📖 正在读取: {file_path.name} ...")

    encodings = ["utf-16", "utf-8", "utf-8-sig", "gb18030", "gbk"]
    separators = ["\t", ","]

    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(file_path, encoding=enc, sep=sep)
                if df.shape[1] == 1 and sep in str(df.columns[0]):
                    continue
                if df.shape[1] > 1:
                    print(
                        f"✅ 读取成功 (编码: {enc}, 分隔符: '{sep if sep != '\t' else '\\t'}') - 形状: {df.shape}"
                    )
                    return df
            except Exception:
                continue

    try:
        print("⚠️ 尝试默认参数读取...")
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return pd.DataFrame()


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.astype(str)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("_年/月/日", " 年/月/日", regex=False)

    rename_map = {
        "lc_create_time 年/月/日": "lc_create_time",
        "lc_order_lock_time_min 年/月/日": "lc_order_lock_time_min",
        "度量名称": "metric_name",
        "度量值": "metric_value",
    }
    df = df.rename(columns=rename_map)
    df.columns = df.columns.str.replace(" ", "_")
    return df


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    print("🔄 开始类型转换...")

    date_cols = ["lc_create_time", "lc_order_lock_time_min"]
    for col in date_cols:
        if col in df.columns:
            s = df[col].astype("string")
            s = s.str.strip()
            s = s.str.replace("年", "-", regex=False)
            s = s.str.replace("月", "-", regex=False)
            s = s.str.replace("日", "", regex=False)
            s = s.replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
            df[col] = pd.to_datetime(s, errors="coerce")

    if "metric_value" in df.columns:
        v = pd.to_numeric(df["metric_value"], errors="coerce")
        if (v.dropna() % 1 == 0).all():
            df["metric_value"] = v.round(0).astype("Int64")
        else:
            df["metric_value"] = v

    if "lc_main_code" in df.columns:
        df["lc_main_code"] = df["lc_main_code"].astype("string").str.strip()

    if "lc_small_channel_name" in df.columns:
        if df["lc_small_channel_name"].nunique(dropna=True) < df.shape[0] * 0.5:
            df["lc_small_channel_name"] = df["lc_small_channel_name"].astype("category")
        else:
            df["lc_small_channel_name"] = df["lc_small_channel_name"].astype("string")

    if "metric_name" in df.columns:
        if df["metric_name"].nunique(dropna=True) < df.shape[0] * 0.5:
            df["metric_name"] = df["metric_name"].astype("category")
        else:
            df["metric_name"] = df["metric_name"].astype("string")

    if "lc_user_phone_md5" in df.columns:
        df["lc_user_phone_md5"] = normalize_phone_md5(df["lc_user_phone_md5"])

    return df


def transpose_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if "metric_name" not in df.columns or "metric_value" not in df.columns:
        return df

    base_cols = [
        c
        for c in [
            "lc_user_phone_md5",
            "lc_main_code",
            "lc_small_channel_name",
            "lc_create_time",
            "lc_order_lock_time_min",
        ]
        if c in df.columns
    ]
    if not base_cols:
        return df

    df2 = df.copy()
    df2["metric_name"] = df2["metric_name"].astype("string").str.strip()

    sort_cols = [
        c
        for c in ["__file_mtime", "lc_order_lock_time_min", "lc_create_time"]
        if c in df2.columns
    ]
    if sort_cols:
        df2 = df2.sort_values(by=sort_cols, na_position="first")

    df2 = df2.drop_duplicates(subset=base_cols + ["metric_name"], keep="last")

    wide = df2.pivot(index=base_cols, columns="metric_name", values="metric_value").reset_index()
    wide.columns = [
        (c if isinstance(c, str) else str(c)).replace(" ", "_") for c in wide.columns
    ]

    metric_cols = [c for c in wide.columns if c not in base_cols]
    for col in metric_cols:
        v = pd.to_numeric(wide[col], errors="coerce")
        if (v.dropna() % 1 == 0).all():
            wide[col] = v.round(0).astype("Int64")
        else:
            wide[col] = v

    if "lc_small_channel_name" in wide.columns:
        if wide["lc_small_channel_name"].nunique(dropna=True) < wide.shape[0] * 0.5:
            wide["lc_small_channel_name"] = wide["lc_small_channel_name"].astype("category")
        else:
            wide["lc_small_channel_name"] = wide["lc_small_channel_name"].astype("string")

    return wide


def process_lock_attribution_parquet() -> int:
    global pd
    import pandas as pd

    globals()["pd"] = pd

    csv_files = list(ORIGINAL_DIR.glob("锁单归因_data_20*.csv"))
    if not csv_files:
        print(f"❌ 未在 {ORIGINAL_DIR} 下找到任何 锁单归因_data_20*.csv")
        return 1

    latest_by_year: dict[int, Path] = {}
    for p in csv_files:
        name = p.name
        year_part = name.replace("锁单归因_data_", "").split(".csv")[0]
        year_str = year_part.split("_")[0]
        if not year_str.isdigit():
            continue
        y = int(year_str)
        prev = latest_by_year.get(y)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            latest_by_year[y] = p

    input_files = [latest_by_year[y] for y in sorted(latest_by_year.keys())]
    if not input_files:
        print(f"❌ 未能从 {ORIGINAL_DIR} 解析到有效年份文件: 锁单归因_data_20XX.csv")
        return 1

    print(f"🔍 找到 {len(input_files)} 个年度数据文件，将按以下顺序处理:")
    for f in input_files:
        print(f"   - {f.name}")

    dfs: list[pd.DataFrame] = []
    failed_files: list[str] = []
    for file_path in input_files:
        df = read_csv_smart(file_path)
        if df.empty:
            failed_files.append(file_path.name)
            continue
        df = clean_column_names(df)
        df = convert_types(df)
        df["__file_mtime"] = float(file_path.stat().st_mtime)
        df = transpose_metrics(df)
        dfs.append(df)

    if failed_files:
        print("❌ 以下文件读取失败或为空，已终止更新以避免静默缺数:")
        for n in failed_files:
            print(f"   - {n}")
        return 1

    df_new = pd.concat(dfs, ignore_index=True)
    print(f"✅ 合并完成: {df_new.shape[0]} 行, {df_new.shape[1]} 列")

    if OUTPUT_FILE.exists():
        print(f"📚 发现现有 Parquet 文件: {OUTPUT_FILE}")
        try:
            df_existing = pd.read_parquet(OUTPUT_FILE)
            print(f"   现有数据: {df_existing.shape[0]} 行")

            legacy_map = {
                "lc_create_time_年/月/日": "lc_create_time",
                "lc_order_lock_time_min_年/月/日": "lc_order_lock_time_min",
                "lc_create_time 年/月/日": "lc_create_time",
                "lc_order_lock_time_min 年/月/日": "lc_order_lock_time_min",
                "度量名称": "metric_name",
                "度量值": "metric_value",
            }
            for old_col, new_col in legacy_map.items():
                if old_col in df_existing.columns:
                    if new_col in df_existing.columns:
                        df_existing[new_col] = df_existing[new_col].combine_first(
                            df_existing[old_col]
                        )
                        df_existing = df_existing.drop(columns=[old_col])
                    else:
                        df_existing = df_existing.rename(columns={old_col: new_col})

            df_existing["__file_mtime"] = float("nan")
            df_existing = transpose_metrics(df_existing)

            all_cols = list(set(df_existing.columns) | set(df_new.columns))
            df_existing = df_existing.reindex(columns=all_cols)
            df_new = df_new.reindex(columns=all_cols)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception as e:
            print(f"❌ 读取现有 Parquet 文件失败: {e}")
            print("   将仅使用新数据。")
            df_final = df_new
    else:
        print("📝 未发现现有 Parquet 文件，创建新文件...")
        df_final = df_new

    if "lc_user_phone_md5" in df_final.columns:
        df_final["lc_user_phone_md5"] = normalize_phone_md5(df_final["lc_user_phone_md5"])

    subset_keys = [c for c in ["lc_user_phone_md5", "lc_main_code"] if c in df_final.columns]
    if subset_keys:
        print(f"✂️  执行最终去重 (键: {subset_keys}) ...")
        before_count = len(df_final)
        sort_cols = [
            c
            for c in ["__file_mtime", "lc_order_lock_time_min", "lc_create_time"]
            if c in df_final.columns
        ]
        if sort_cols:
            df_final = df_final.sort_values(by=sort_cols, na_position="first")
        df_final = df_final.drop_duplicates(subset=subset_keys, keep="last")
        after_count = len(df_final)
        print(f"   去重前: {before_count}, 去重后: {after_count}, 移除: {before_count - after_count}")

    if "__file_mtime" in df_final.columns:
        df_final = df_final.drop(columns=["__file_mtime"])

    if not FORMATTED_DIR.exists():
        FORMATTED_DIR.mkdir(parents=True)

    print(f"💾 保存到: {OUTPUT_FILE} ...")
    df_final.to_parquet(OUTPUT_FILE, index=False)

    if OUTPUT_FILE.exists():
        size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        print(f"✅ 保存成功! 文件大小: {size_mb:.2f} MB")
        print(f"   最终行数: {df_final.shape[0]}")
        return 0

    print("❌ 保存失败")
    return 1


def _redact_command(cmd: list[str]) -> str:
    redacted: list[str] = []
    skip_next = False
    for part in cmd:
        if skip_next:
            redacted.append("***")
            skip_next = False
            continue
        if part in {"--token-value", "--password"}:
            redacted.append(part)
            skip_next = True
            continue
        redacted.append(part)
    return " ".join(redacted)


def run_command_with_output(command: list[str], cwd: Path | None = None) -> int:
    print(f"🚀 执行命令: {_redact_command(command)}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(cwd) if cwd else None,
            bufsize=1,
            universal_newlines=True,
        )

        assert process.stdout is not None
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())

        return int(process.poll() or 0)
    except Exception as e:
        print(f"❌ 执行命令时发生错误: {str(e)}")
        return -1


def export_tableau_csv_to_original(
    *,
    view: str,
    output_path: Path,
    token_name: str,
    token_value: str,
    timeout: int,
    mobile: bool,
) -> bool:
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass

    print(f"视图: {view}")
    print(f"目标文件: {output_path}")

    command = [
        "python3",
        "-u",
        "tableau_export.py",
        "--token-name",
        token_name,
        "--token-value",
        token_value,
        "--view",
        view,
        "--output",
        str(tmp_path),
        "--timeout",
        str(timeout),
    ]
    if mobile:
        command.append("--mobile")

    returncode = run_command_with_output(command, cwd=SCRIPTS_DIR)
    if returncode != 0 or not tmp_path.exists():
        print("❌ Tableau 数据导出失败")
        return False

    try:
        if tmp_path.stat().st_size == 0:
            print("❌ Tableau 导出结果为空文件")
            return False
    except Exception:
        pass

    tmp_path.replace(output_path)
    print(f"✅ Tableau 数据导出成功: {output_path}")
    return True


def step1_export_tableau(
    *,
    view: str,
    year: int | None = None,
    mobile: bool = False,
    timeout: int = 600,
) -> bool:
    print("\n" + "=" * 60)
    print("步骤 1: 从 Tableau 导出最新数据")
    print("=" * 60)

    if year is None:
        year = int(datetime.now().strftime("%Y"))

    output_filename = f"锁单归因_data_{year}.csv"
    output_path = ORIGINAL_DIR / output_filename

    env_file = read_env_file(SCRIPTS_DIR / ".env")
    token_name = env_file.get("TABLEAU_TOKEN_NAME")
    token_value = env_file.get("TABLEAU_TOKEN_VALUE")
    if not token_name or not token_value:
        print("❌ 缺少 Tableau PAT。请在 scripts/.env#L3-4 配置 TABLEAU_TOKEN_NAME / TABLEAU_TOKEN_VALUE")
        return False

    ok_test_drive = export_tableau_csv_to_original(
        view="https://tableau-hs.immotors.com/#/views/core_metric_observation/7",
        output_path=ORIGINAL_DIR / "test_drive_data.csv",
        token_name=token_name,
        token_value=token_value,
        timeout=timeout,
        mobile=mobile,
    )
    if not ok_test_drive:
        return False

    ok_assign = export_tableau_csv_to_original(
        view="https://tableau-hs.immotors.com/#/views/core_metric_observation/assign",
        output_path=ORIGINAL_DIR / "assign_data.csv",
        token_name=token_name,
        token_value=token_value,
        timeout=timeout,
        mobile=mobile,
    )
    if not ok_assign:
        return False

    ok_lock = export_tableau_csv_to_original(
        view=view,
        output_path=output_path,
        token_name=token_name,
        token_value=token_value,
        timeout=timeout,
        mobile=mobile,
    )
    return ok_lock


def step_export_assign_and_test_drive(*, mobile: bool = False, timeout: int = 600) -> bool:
    print("\n" + "=" * 60)
    print("步骤 1: 从 Tableau 导出 assign/test_drive 数据")
    print("=" * 60)

    env_file = read_env_file(SCRIPTS_DIR / ".env")
    token_name = env_file.get("TABLEAU_TOKEN_NAME")
    token_value = env_file.get("TABLEAU_TOKEN_VALUE")
    if not token_name or not token_value:
        print("❌ 缺少 Tableau PAT。请在 scripts/.env#L3-4 配置 TABLEAU_TOKEN_NAME / TABLEAU_TOKEN_VALUE")
        return False

    ok_test_drive = export_tableau_csv_to_original(
        view="https://tableau-hs.immotors.com/#/views/core_metric_observation/7",
        output_path=ORIGINAL_DIR / "test_drive_data.csv",
        token_name=token_name,
        token_value=token_value,
        timeout=timeout,
        mobile=mobile,
    )
    if not ok_test_drive:
        return False

    ok_assign = export_tableau_csv_to_original(
        view="https://tableau-hs.immotors.com/#/views/core_metric_observation/assign",
        output_path=ORIGINAL_DIR / "assign_data.csv",
        token_name=token_name,
        token_value=token_value,
        timeout=timeout,
        mobile=mobile,
    )
    return ok_assign


def main() -> int:
    parser = argparse.ArgumentParser(description="锁单归因数据：从 Tableau 导出并更新 Parquet")
    parser.add_argument(
        "--view",
        default="https://tableau-hs.immotors.com/#/views/1h/sheet10",
        help="Tableau 视图路径 (Workbook/Sheet) 或完整URL",
    )
    parser.add_argument("--year", type=int, default=None, help="导出年份 (默认当前年份)")
    parser.add_argument("--timeout", type=int, default=600, help="导出超时（秒）")
    parser.add_argument("--mobile", action="store_true", help="使用移动端/非办公网络服务器地址")
    parser.add_argument("--no-export", action="store_true", help="跳过导出，仅处理本地 original 目录")
    parser.add_argument(
        "--only-assign-test-drive",
        action="store_true",
        help="仅更新 original/assign_data.csv 与 original/test_drive_data.csv",
    )
    args = parser.parse_args()

    print("🚀 开始锁单归因数据更新流水线")
    start_time = datetime.now()

    if args.only_assign_test_drive:
        ok = step_export_assign_and_test_drive(mobile=args.mobile, timeout=args.timeout)
        end_time = datetime.now()
        print(f"⏰ 总耗时: {(end_time - start_time).total_seconds():.1f} 秒")
        return 0 if ok else 1

    ok = True
    if not args.no_export:
        ok = step1_export_tableau(
            view=args.view,
            year=args.year,
            mobile=args.mobile,
            timeout=args.timeout,
        )

    if ok:
        print("\n" + "=" * 60)
        print("步骤 2: 合并并更新 Parquet 文件")
        print("=" * 60)
        code = process_lock_attribution_parquet()
    else:
        code = 1

    end_time = datetime.now()
    print(f"⏰ 总耗时: {(end_time - start_time).total_seconds():.1f} 秒")
    return code


if __name__ == "__main__":
    sys.exit(main())
