#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 original/order_data_2023.csv ~ order_data_2026.csv 进行清洗与类型转换，合并为一个 Parquet 文件。
输出：formatted/order_data.parquet
"""

import sys
import os
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path("/Users/zihao_/Documents/coding/dataset")
SCRIPTS_DIR = BASE_DIR / "scripts"
ORIGINAL_DIR = BASE_DIR / "original"
FORMATTED_DIR = BASE_DIR / "formatted"
OUTPUT_FILE = FORMATTED_DIR / "order_data.parquet"

TABLEAU_VIEW = "core_metric_observation/11"
TABLEAU_OUTPUT_2026 = ORIGINAL_DIR / "order_data_2026.csv"

INPUT_FILES = [
    Path("/Users/zihao_/Documents/coding/dataset/original/order_data_2023.csv"),
    Path("/Users/zihao_/Documents/coding/dataset/original/order_data_2024.csv"),
    Path("/Users/zihao_/Documents/coding/dataset/original/order_data_2025.csv"),
    Path("/Users/zihao_/Documents/coding/dataset/original/order_data_2026.csv"),
]


def normalize_owner_cell_phone(series: pd.Series) -> pd.Series:
    s_raw = series.astype("string").str.strip()
    num = pd.to_numeric(s_raw, errors="coerce").round(0)
    num_int = num.astype("Int64")
    s_num = num_int.astype("string")
    s = s_raw.where(num.isna(), s_num)
    s = s.str.strip().str.lower()
    s = s.replace(
        {
            "nan": pd.NA,
            "none": pd.NA,
            "null": pd.NA,
            "": pd.NA,
            "-": pd.NA,
            "无": pd.NA,
        }
    )
    digits = s.str.replace(r"\D", "", regex=True)
    digits = digits.str.replace(r"^0086", "", regex=True)
    digits = digits.str.replace(r"^86", "", regex=True)
    valid = digits.str.match(r"^1\d{10}$", na=False)
    return digits.where(valid, pd.NA).astype("string")


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


def _redact_command(cmd: list[str]) -> str:
    redacted: list[str] = []
    skip_next = False
    for i, part in enumerate(cmd):
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
            cwd=str(cwd) if cwd is not None else None,
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


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        return


def step_export_tableau_order_data_2026(mobile: bool = False, timeout: int = 600) -> bool:
    print("\n" + "=" * 60)
    print("步骤 0: 从 Tableau 导出最新 order_data_2026.csv")
    print("=" * 60)

    output_path = TABLEAU_OUTPUT_2026
    print(f"目标文件: {output_path}")

    token_name = os.getenv("TABLEAU_TOKEN_NAME")
    token_value = os.getenv("TABLEAU_TOKEN_VALUE")

    command: list[str] = [
        sys.executable,
        "-u",
        str(SCRIPTS_DIR / "tableau_export.py"),
        "--view",
        TABLEAU_VIEW,
        "--output",
        str(output_path),
        "--timeout",
        str(int(timeout)),
    ]
    if mobile:
        command.append("--mobile")
    if token_name and token_value:
        command.extend(["--token-name", token_name, "--token-value", token_value])

    returncode = run_command_with_output(command, cwd=SCRIPTS_DIR)
    if returncode == 0 and output_path.exists():
        print(f"✅ Tableau 数据导出成功: {output_path}")
        return True

    print("❌ Tableau 数据导出失败")
    return False


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.astype(str).str.strip()
    df.columns = df.columns.str.replace("_年/月/日", " 年/月/日", regex=False)

    rename_map = {
        "first_touch_time 年/月/日": "first_touch_time",
        "delivery_date 年/月/日": "delivery_date",
        "deposit_payment_time 年/月/日": "deposit_payment_time",
        "deposit_refund_time 年/月/日": "deposit_refund_time",
        "first_test_drive_time 年/月/日": "first_test_drive_time",
        "intention_payment_time 年/月/日": "intention_payment_time",
        "intention_refund_time 年/月/日": "intention_refund_time",
        "invoice_upload_time 年/月/日": "invoice_upload_time",
        "lock_time 年/月/日": "lock_time",
        "order_create_time 年/月/日": "order_create_date",
        "store_create_date 年/月/日": "store_create_date",
        "approve_refund_time 年/月/日": "approve_refund_time",
        "apply_refund_time 年/月/日": "apply_refund_time",
        "first_assign_time 年/月/日": "first_assign_time",
        "lead_assign_time_max 年/月/日": "lead_assign_time_max",
        "final_payment_time 年/月/日": "final_payment_time",
        "actual_refund_time 年/月/日": "actual_refund_time",
        "Td CountD": "td_countd",
        "Drive Series Cn": "drive_series_cn",
        "Main Lead Id": "main_lead_id",
        "Parent Region Name": "parent_region_name",
        "Parent_Region_Name": "parent_region_name",
        "DATE([invoice_upload_time])": "invoice_upload_time",
        "DATE([first_assign_time])": "first_assign_time",
        "DATE([store_create_date])": "store_create_date",
        "Buyer Identity No": "buyer_identity_no",
        "Owner Identity No": "owner_identity_no",
        "Product Name": "product_name",
        "Series": "series",
        "Store Name": "store_name",
        "DATE([Invoice Upload Time])": "invoice_upload_time",
        "Deposit Payment Time": "deposit_payment_time",
        "Final Payment Way": "final_payment_way",
        "Finance Product": "finance_product",
        "Intention Payment Time": "intention_payment_time",
        "License City": "license_city",
        "Lock Time": "lock_time",
        "Order Number": "order_number",
        "Store City": "store_city",
        "Vin": "vin",
        "Invoice Amount": "invoice_amount",
        "Actual Refund Time 年/月/日": "actual_refund_time",
        "Apply Refund Time 年/月/日": "apply_refund_time",
        "Approve Refund Time 年/月/日": "approve_refund_time",
        "Delivery Date 年/月/日": "delivery_date",
        "Final Payment Time 年/月/日": "final_payment_time",
        "First Test Drive Time 年/月/日": "first_test_drive_time",
        "Order Create Time 年/月/日": "order_create_date",
    }

    df = df.rename(columns=rename_map)
    df.columns = df.columns.str.replace(" ", "_")
    return df


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = [
        "first_touch_time",
        "delivery_date",
        "deposit_payment_time",
        "deposit_refund_time",
        "first_test_drive_time",
        "intention_payment_time",
        "intention_refund_time",
        "invoice_upload_time",
        "lock_time",
        "order_create_date",
        "store_create_date",
        "order_create_time",
        "approve_refund_time",
        "apply_refund_time",
        "first_assign_time",
        "lead_assign_time_max",
        "final_payment_time",
        "actual_refund_time",
    ]
    for col in date_cols:
        if col in df.columns:
            s = df[col].astype(str)
            s = (
                s.str.replace("年", "-", regex=False)
                .str.replace("月", "-", regex=False)
                .str.replace("日", "", regex=False)
            )
            s = s.replace({"nan": None, "None": None, "": None})
            df[col] = pd.to_datetime(s, errors="coerce")

    numeric_cols = ["age", "invoice_amount", "Invoice_Amount", "td_countd", "buyer_age", "owner_age"]
    for col in numeric_cols:
        if col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                s = df[col].astype(str).str.replace(",", "", regex=False).str.replace("￥", "", regex=False).str.replace("¥", "", regex=False)
                df[col] = pd.to_numeric(s, errors="coerce")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["buyer_age", "owner_age"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            valid = s.between(18, 100, inclusive="both")
            df[col] = s.where(valid)

    cat_cols = [
        "product_name",
        "final_payment_way",
        "finance_product",
        "first_middle_channel_name",
        "gender",
        "is_hold",
        "is_staff",
        "license_city",
        "license_city_level",
        "license_province",
        "order_type",
        "series",
        "store_city",
        "belong_intent_series",
        "drive_series_cn",
        "parent_region_name",
    ]
    for col in cat_cols:
        if col in df.columns:
            if df[col].nunique(dropna=True) < df.shape[0] * 0.5:
                df[col] = df[col].astype("category")
            else:
                df[col] = df[col].astype("string")

    if "order_number" in df.columns:
        df["order_number"] = df["order_number"].astype("string")

    if "owner_cell_phone" in df.columns:
        df["owner_cell_phone"] = normalize_owner_cell_phone(df["owner_cell_phone"])

    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="order_data 合并清洗并写入 Parquet（可选自动从 Tableau 更新 2026 数据）")
    parser.add_argument("--skip-export", action="store_true", help="跳过从 Tableau 导出 2026 数据步骤")
    parser.add_argument("--mobile", action="store_true", help="使用移动端/非办公网络服务器地址导出")
    parser.add_argument("--timeout", type=int, default=600, help="Tableau 导出超时（秒）")
    args = parser.parse_args(argv)

    if not args.skip_export:
        load_env_file(SCRIPTS_DIR / ".env")
        ok = step_export_tableau_order_data_2026(mobile=args.mobile, timeout=args.timeout)
        if not ok:
            return 1

    missing = [p for p in INPUT_FILES if not p.exists()]
    if missing:
        print("❌ 输入文件不存在:")
        for p in missing:
            print(f"   - {p}")
        return 1

    dfs: list[pd.DataFrame] = []
    failed: list[str] = []
    for p in INPUT_FILES:
        df = read_csv_smart(p)
        if df.empty:
            failed.append(p.name)
            continue
        df = clean_column_names(df)
        df = convert_types(df)
        dfs.append(df)

    if failed:
        print("❌ 以下文件读取失败或为空，已终止以避免静默缺数:")
        for n in failed:
            print(f"   - {n}")
        return 1

    df_all = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"✅ 合并完成: {df_all.shape[0]} 行, {df_all.shape[1]} 列")

    if "order_number" in df_all.columns:
        before = int(df_all.shape[0])
        df_all = df_all.drop_duplicates(subset=["order_number"], keep="last")
        after = int(df_all.shape[0])
        print(f"✂️ 去重(order_number): {before} -> {after} (移除 {before - after})")

    if not FORMATTED_DIR.exists():
        FORMATTED_DIR.mkdir(parents=True)

    print(f"💾 保存到: {OUTPUT_FILE}")
    df_all.to_parquet(OUTPUT_FILE, index=False)

    if not OUTPUT_FILE.exists():
        print("❌ 保存失败")
        return 1

    try:
        df_check = pd.read_parquet(OUTPUT_FILE, columns=None)
        print("\n🔎 数据检查（Max 时间）")
        for col in ["order_create_date", "intention_payment_time", "lock_time"]:
            if col not in df_check.columns:
                print(f" - max {col}: Column Not Found")
                continue
            if not pd.api.types.is_datetime64_any_dtype(df_check[col]):
                df_check[col] = pd.to_datetime(df_check[col], errors="coerce")
            max_val = df_check[col].max()
            print(f" - max {col}: {str(max_val) if pd.notna(max_val) else 'None'}")
    except Exception as e:
        print(f"\n⚠️ 数据检查失败: {e}")

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"✅ 保存成功! 文件大小: {size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
