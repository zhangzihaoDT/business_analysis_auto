#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


MONTHS_12 = [
    "1999-11",
    "1999-12",
    "2000-01",
    "2000-02",
    "2000-03",
    "2000-04",
    "2000-05",
    "2000-06",
    "2000-07",
    "2000-08",
    "2000-09",
    "2000-10",
]

TARGET_MONTHS = ["2000-01", "2000-03", "2000-08", "2000-10", "1999-12"]


@dataclass(frozen=True)
class TestResult:
    month: str
    regions: int
    mean_month: float
    mean_baseline: float
    mean_diff: float
    share_regions_above: float
    p_value_one_sided: float
    p_value_bonferroni: float
    significant_0_05: bool
    significant_0_05_bonf: bool


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    default_in = project_dir / "original" / "statsgov_rkpc5rp_t0112_births_1999-11_2000-10_tidy.csv"
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(default_in))
    p.add_argument("--sex", default="total", choices=["total", "male", "female"])
    p.add_argument("--include-total-row", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=20260319)
    p.add_argument("--permutations", type=int, default=20000)
    return p.parse_args(argv)


def read_births(path: Path, sex: str) -> Dict[Tuple[str, str], int]:
    data: Dict[Tuple[str, str], int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("sex") != sex:
                continue
            region = (row.get("region") or "").strip()
            month = (row.get("month") or "").strip()
            if month == "1999-11_to_2000-10":
                continue
            births_raw = (row.get("births") or "").strip()
            if not births_raw.isdigit():
                continue
            births = int(births_raw)
            if not region or not month:
                continue
            data[(region, month)] = births
    return data


def region_list(data: Dict[Tuple[str, str], int]) -> List[str]:
    regions = sorted({k[0] for k in data.keys()})
    return regions


def build_matrix(
    data: Dict[Tuple[str, str], int],
    regions: Sequence[str],
    months: Sequence[str],
) -> List[List[Optional[int]]]:
    matrix: List[List[Optional[int]]] = []
    for region in regions:
        row: List[Optional[int]] = []
        for month in months:
            row.append(data.get((region, month)))
        matrix.append(row)
    return matrix


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def permutation_p_value_one_sided_mean_gt_zero(diffs: Sequence[float], rng: random.Random, permutations: int) -> float:
    diffs = [d for d in diffs if not math.isnan(d)]
    n = len(diffs)
    if n == 0:
        return float("nan")
    obs = sum(diffs) / n
    count = 1
    total = permutations + 1
    for _ in range(permutations):
        s = 0.0
        for d in diffs:
            s += d if rng.getrandbits(1) else -d
        m = s / n
        if m >= obs:
            count += 1
    return count / total


def compute_results(
    data: Dict[Tuple[str, str], int],
    months: Sequence[str],
    target_months: Sequence[str],
    include_total_row: bool,
    seed: int,
    permutations: int,
) -> Tuple[List[TestResult], Optional[Dict[str, float]]]:
    regions = region_list(data)
    if not include_total_row:
        regions = [r for r in regions if r != "合计"]

    matrix = build_matrix(data, regions, months)
    baselines: List[float] = []
    month_means: Dict[str, float] = {}
    for i, region in enumerate(regions):
        vals = [v for v in matrix[i] if v is not None]
        if len(vals) != len(months):
            continue
        baselines.append(sum(vals) / len(months))
    baseline_mean = mean(baselines)

    for j, month in enumerate(months):
        vals = []
        for i in range(len(regions)):
            v = matrix[i][j]
            if v is None:
                vals = []
                break
            vals.append(v)
        if vals:
            month_means[month] = sum(vals) / len(vals)

    rng = random.Random(seed)
    results: List[TestResult] = []
    k = len(target_months)
    for month in target_months:
        if month not in month_means:
            continue
        diffs: List[float] = []
        above = 0
        kept = 0
        month_idx = months.index(month)
        for i in range(len(regions)):
            row = matrix[i]
            if any(v is None for v in row):
                continue
            baseline_i = sum(row) / len(months)  # type: ignore[arg-type]
            diff = float(row[month_idx]) - baseline_i  # type: ignore[operator]
            diffs.append(diff)
            kept += 1
            if diff > 0:
                above += 1

        p = permutation_p_value_one_sided_mean_gt_zero(diffs, rng, permutations)
        p_b = min(1.0, p * k)
        results.append(
            TestResult(
                month=month,
                regions=kept,
                mean_month=month_means[month],
                mean_baseline=baseline_mean,
                mean_diff=mean(diffs),
                share_regions_above=(above / kept) if kept else float("nan"),
                p_value_one_sided=p,
                p_value_bonferroni=p_b,
                significant_0_05=(p < 0.05),
                significant_0_05_bonf=(p_b < 0.05),
            )
        )
    overall_series = None
    if ("合计", months[0]) in data:
        diffs_overall: Dict[str, float] = {}
        overall_vals = [data.get(("合计", m)) for m in months]
        if all(v is not None for v in overall_vals):
            overall_mean = sum(overall_vals) / len(months)  # type: ignore[arg-type]
            diffs_overall = {m: float(data[("合计", m)]) - overall_mean for m in months}  # type: ignore[index]
            overall_series = diffs_overall
    return results, overall_series


def fmt_pct(x: float) -> str:
    if math.isnan(x):
        return "NA"
    return f"{x*100:.1f}%"


def fmt_num(x: float) -> str:
    if math.isnan(x):
        return "NA"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.2f}"


def print_table(results: Sequence[TestResult]) -> None:
    headers = [
        "month",
        "regions",
        "mean_month",
        "mean_baseline",
        "mean_diff",
        "share_regions_above",
        "p_one_sided",
        "p_bonf",
        "sig_0.05",
        "sig_0.05_bonf",
    ]
    rows = []
    for r in results:
        rows.append(
            [
                r.month,
                str(r.regions),
                fmt_num(r.mean_month),
                fmt_num(r.mean_baseline),
                fmt_num(r.mean_diff),
                fmt_pct(r.share_regions_above),
                f"{r.p_value_one_sided:.4f}" if not math.isnan(r.p_value_one_sided) else "NA",
                f"{r.p_value_bonferroni:.4f}" if not math.isnan(r.p_value_bonferroni) else "NA",
                "Y" if r.significant_0_05 else "N",
                "Y" if r.significant_0_05_bonf else "N",
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def write_row(row: Sequence[str]) -> None:
        sys.stdout.write("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + "\n")

    write_row(headers)
    write_row(["-" * w for w in widths])
    for row in rows:
        write_row(row)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    include_total = bool(args.include_total_row)
    data = read_births(Path(args.input).expanduser().resolve(), args.sex)
    results, overall_series = compute_results(
        data=data,
        months=MONTHS_12,
        target_months=TARGET_MONTHS,
        include_total_row=include_total,
        seed=args.seed,
        permutations=args.permutations,
    )
    print_table(results)
    if overall_series is not None:
        overall_mean = sum(data[("合计", m)] for m in MONTHS_12) / len(MONTHS_12)
        sys.stdout.write("\n合计行（仅描述性对比，不做显著性检验）\n")
        sys.stdout.write(f"overall_mean_monthly={overall_mean:,.2f}\n")
        for m in TARGET_MONTHS:
            v = data.get(("合计", m))
            if v is None:
                continue
            diff = v - overall_mean
            sys.stdout.write(f"{m}: births={v:,} diff_vs_mean={diff:,.2f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
