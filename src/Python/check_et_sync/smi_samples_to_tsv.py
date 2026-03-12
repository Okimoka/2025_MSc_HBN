#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

"""
Converts _Samples.txt file into tsv

LLM Code
"""

OUTPUT_COLUMNS = [
    "time",
    "L Raw X [px]",
    "L Raw Y [px]",
    "R Raw X [px]",
    "R Raw Y [px]",
    "L Mapped Diameter [mm]",
    "R Mapped Diameter [mm]",
    "L Validity",
    "R Validity",
    "Pupil Confidence",
]

INPUT_COLUMN_BY_OUTPUT = {
    "time": "Time",
    "L Raw X [px]": "L Raw X [px]",
    "L Raw Y [px]": "L Raw Y [px]",
    "R Raw X [px]": "R Raw X [px]",
    "R Raw Y [px]": "R Raw Y [px]",
    "L Mapped Diameter [mm]": "L Mapped Diameter [mm]",
    "R Mapped Diameter [mm]": "R Mapped Diameter [mm]",
    "L Validity": "L Validity",
    "R Validity": "R Validity",
    "Pupil Confidence": "Pupil Confidence",
}


def _default_output_path(in_path: Path) -> Path:
    if in_path.suffix:
        return in_path.with_suffix(".tsv")
    return in_path.with_name(f"{in_path.name}.tsv")


def _find_header(lines: list[str]) -> tuple[int, list[str]]:
    for i, raw_line in enumerate(lines):
        if raw_line.startswith("##"):
            continue
        fields = [field.strip() for field in raw_line.rstrip("\r\n").split("\t")]
        if fields and fields[0] == "Time":
            return i, fields
    raise ValueError("Could not find tabular header row starting with 'Time'.")


def _resolve_indices(header: list[str]) -> dict[str, int]:
    missing = [in_col for in_col in INPUT_COLUMN_BY_OUTPUT.values() if in_col not in header]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns in header: {missing_str}")
    return {out_col: header.index(in_col) for out_col, in_col in INPUT_COLUMN_BY_OUTPUT.items()}


def parse_smi_samples(path: Path) -> tuple[list[dict[str, str]], int, int]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header_line_idx, header = _find_header(lines)
    indices = _resolve_indices(header)
    type_idx = header.index("Type") if "Type" in header else None

    short_row_count = 0
    non_sample_count = 0
    rows: list[dict[str, str]] = []

    for raw_line in lines[header_line_idx + 1 :]:
        if not raw_line or raw_line.startswith("##"):
            continue

        fields = raw_line.rstrip("\r\n").split("\t")
        row: dict[str, str] = {out_col: "" for out_col in OUTPUT_COLUMNS}

        time_idx = indices["time"]
        if time_idx < len(fields):
            row["time"] = fields[time_idx].strip()
        else:
            short_row_count += 1
            rows.append(row)
            continue

        row_type = ""
        if type_idx is not None and type_idx < len(fields):
            row_type = fields[type_idx].strip().upper()

        if row_type and row_type != "SMP":
            non_sample_count += 1
            rows.append(row)
            continue

        row_was_short = False
        for out_col in OUTPUT_COLUMNS:
            if out_col == "time":
                continue
            idx = indices[out_col]
            if idx < len(fields):
                row[out_col] = fields[idx].strip()
            else:
                row_was_short = True

        if row_was_short:
            short_row_count += 1
        rows.append(row)

    return rows, short_row_count, non_sample_count


def write_tsv(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SMI IDF *_Samples exports to TSV with columns: "
            "time, L/R raw X/Y, mapped diameters, validity, and pupil confidence."
        )
    )
    parser.add_argument("input", type=Path, help="Path to SMI *_Samples.txt file")
    parser.add_argument("-o", "--output", type=Path, help="Output TSV path (default: input with .tsv)")
    args = parser.parse_args()

    in_path: Path = args.input
    out_path: Path = args.output if args.output else _default_output_path(in_path)

    rows, short_row_count, non_sample_count = parse_smi_samples(in_path)
    write_tsv(rows, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")
    if non_sample_count:
        print(
            f"Note: {non_sample_count} non-SMP row(s) were retained with only 'time' populated."
        )
    if short_row_count:
        print(
            f"Note: {short_row_count} row(s) were shorter than the header and were padded with empty fields."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
