#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import logging
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Tuple

import requests


URL = "https://www.stats.gov.cn/sj/pcsj/rkpc/5rp/html/t0112.htm"


def _normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ").replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_int_str(s: str) -> bool:
    return bool(re.fullmatch(r"\d+", s))


@dataclass
class ParsedTable:
    rows: List[List[str]]


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._tables: List[ParsedTable] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._cell_parts: List[str] = []
        self._current_row: List[str] = []
        self._current_rows: List[List[str]] = []

    @property
    def tables(self) -> List[ParsedTable]:
        return self._tables

    def handle_starttag(self, tag: str, attrs) -> None:
        t = tag.lower()
        if t == "table":
            self._in_table = True
            self._current_rows = []
        elif self._in_table and t == "tr":
            self._in_row = True
            self._current_row = []
        elif self._in_table and self._in_row and t in {"td", "th"}:
            self._in_cell = True
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if not self._in_cell:
            return
        if data:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in {"td", "th"} and self._in_cell:
            self._in_cell = False
            cell = _normalize_text("".join(self._cell_parts))
            if cell != "":
                self._current_row.append(cell)
            self._cell_parts = []
        elif t == "tr" and self._in_row:
            self._in_row = False
            if self._current_row:
                self._current_rows.append(self._current_row)
            self._current_row = []
        elif t == "table" and self._in_table:
            self._in_table = False
            if self._current_rows:
                self._tables.append(ParsedTable(rows=self._current_rows))
            self._current_rows = []


def fetch_html(url: str, timeout: int = 30) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def parse_best_table(html: str) -> List[List[str]]:
    parser = _HTMLTableParser()
    parser.feed(html)
    tables = parser.tables
    if not tables:
        raise RuntimeError("未解析到任何 HTML 表格")

    def score_table(t: ParsedTable) -> Tuple[int, int, int]:
        row_count = len(t.rows)
        max_len = max((len(r) for r in t.rows), default=0)
        likely_data_rows = 0
        for r in t.rows:
            if len(r) < 10:
                continue
            numeric = sum(_is_int_str(x) for x in r[1:])
            if numeric >= 30:
                likely_data_rows += 1
        return (likely_data_rows, max_len, row_count)

    tables_sorted = sorted(tables, key=score_table, reverse=True)
    return tables_sorted[0].rows


def extract_region_series(rows: List[List[str]]) -> List[Tuple[str, List[int]]]:
    candidates: List[Tuple[str, List[int]]] = []
    for r in rows:
        rr = [_normalize_text(x) for x in r if _normalize_text(x) != ""]
        if len(rr) < 10:
            continue
        region = rr[0]
        if _is_int_str(region):
            continue
        nums = [x for x in rr[1:] if _is_int_str(x)]
        if len(nums) >= 35:
            values = [int(x) for x in nums]
            candidates.append((region, values))

    if not candidates:
        raise RuntimeError("未从表格中识别到数据行（地区 + 数值列）")

    preferred = [(reg, vals) for reg, vals in candidates if len(vals) >= 39]
    if preferred:
        return [(reg, vals[:39]) for reg, vals in preferred]

    max_len = max(len(vals) for _, vals in candidates)
    return [(reg, vals) for reg, vals in candidates if len(vals) == max_len]


def build_tidy_rows(region_series: List[Tuple[str, List[int]]]) -> List[dict]:
    months = ["1999-11_to_2000-10"] + [f"1999-{m:02d}" for m in (11, 12)] + [f"2000-{m:02d}" for m in range(1, 11)]
    sexes = ["total", "male", "female"]

    records: List[dict] = []
    for region, values in region_series:
        if len(values) < 39:
            continue
        overall = values[:3]
        monthly = values[3:39]

        for sex, v in zip(sexes, overall):
            records.append({"region": region, "month": months[0], "sex": sex, "births": int(v)})

        for i, month in enumerate(months[1:]):
            chunk = monthly[i * 3 : (i + 1) * 3]
            if len(chunk) != 3:
                continue
            for sex, v in zip(sexes, chunk):
                records.append({"region": region, "month": month, "sex": sex, "births": int(v)})

    if not records:
        raise RuntimeError("生成的结果为空，请检查解析逻辑或网页结构是否变化")

    month_order = {m: i for i, m in enumerate(months)}
    sex_order = {"total": 0, "male": 1, "female": 2}
    records_sorted = sorted(
        records,
        key=lambda r: (
            r["region"],
            month_order.get(r["month"], 10**9),
            sex_order.get(r["sex"], 10**9),
        ),
    )
    return records_sorted


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        raise RuntimeError("写入 CSV 失败：rows 为空")
    fieldnames = ["region", "month", "sex", "births"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    default_out = project_dir / "original" / "statsgov_rkpc5rp_t0112_births_1999-11_2000-10_tidy.csv"

    p = argparse.ArgumentParser()
    p.add_argument("--url", default=URL)
    p.add_argument("--output", default=str(default_out))
    p.add_argument("--timeout", type=int, default=30)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    args = parse_args(argv)

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("开始抓取: %s", args.url)
    html = fetch_html(args.url, timeout=args.timeout)

    rows = parse_best_table(html)
    region_series = extract_region_series(rows)
    tidy_rows = build_tidy_rows(region_series)
    write_csv(out_path, tidy_rows)
    logging.info("已写入: %s (rows=%d)", out_path, len(tidy_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
