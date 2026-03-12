#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

"""
Converts _Events.txt file into tsv

LLM Code
"""

HEADER_RE = re.compile(r"^Table Header for (.+):\s*$", re.IGNORECASE)
EVENT_RE = re.compile(r"^(Fixation|Saccade|Blink)(?:\s+([LR]))?\s*$", re.IGNORECASE)


def _normalize_header_name(raw_name: str) -> str:
    name = raw_name.strip().lower()
    if name.endswith(" events"):
        name = name[: -len(" events")]
    if name.endswith("s"):
        name = name[:-1]
    return name


def _parse_table_headers(lines: list[str]) -> dict[str, list[str]]:
    headers: dict[str, list[str]] = {}
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        match = HEADER_RE.match(line)
        if not match:
            continue

        header_name = _normalize_header_name(match.group(1))
        header_idx = idx + 1
        while header_idx < len(lines) and not lines[header_idx].strip():
            header_idx += 1
        if header_idx >= len(lines):
            continue

        columns = [col.strip().lower() for col in lines[header_idx].rstrip("\r\n").split("\t")]
        headers[header_name] = columns
    return headers


def _resolve_indices(columns: list[str] | None) -> tuple[int, int]:
    if columns is None:
        return 3, 5

    try:
        start_idx = columns.index("start")
        duration_idx = columns.index("duration")
    except ValueError:
        return 3, 5
    return start_idx, duration_idx


def parse_smi_events(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    headers = _parse_table_headers(lines)

    records: list[dict[str, str]] = []
    for line_no, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue

        fields = raw_line.rstrip("\r\n").split("\t")
        first_field = fields[0].strip()
        event_match = EVENT_RE.match(first_field)
        if not event_match:
            continue

        event_type = event_match.group(1).lower()
        eye = (event_match.group(2) or "").upper()
        if eye not in {"L", "R"}:
            print(f"[warn] Missing/invalid eye at line {line_no}: {first_field!r}", file=sys.stderr)
            continue

        start_idx, duration_idx = _resolve_indices(headers.get(event_type))
        if len(fields) <= max(start_idx, duration_idx):
            print(f"[warn] Skipping short row at line {line_no}", file=sys.stderr)
            continue

        onset = fields[start_idx].strip()
        duration = fields[duration_idx].strip()
        if not onset or not duration:
            print(f"[warn] Empty onset/duration at line {line_no}", file=sys.stderr)
            continue

        records.append(
            {
                "type": event_type,
                "eye": eye,
                "onset": onset,
                "duration": duration,
            }
        )

    return records


def write_tsv(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "eye", "onset", "duration"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _default_output_path(in_path: Path) -> Path:
    if in_path.suffix:
        return in_path.with_suffix(".tsv")
    return in_path.with_name(f"{in_path.name}.tsv")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SMI IDF *_Events exports to a simplified TSV with columns: "
            "type, eye, onset, duration."
        )
    )
    parser.add_argument("input", type=Path, help="Path to SMI *_Events.txt file")
    parser.add_argument("-o", "--output", type=Path, help="Output TSV path (default: input with .tsv)")
    args = parser.parse_args()

    in_path: Path = args.input
    out_path: Path = args.output if args.output else _default_output_path(in_path)

    rows = parse_smi_events(in_path)
    write_tsv(rows, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
