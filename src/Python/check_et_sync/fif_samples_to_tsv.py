#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mne
import numpy as np

"""
Extract Samples.tsv and Events.tsv from a .fif file

LLM Code
"""

OUTPUT_COLUMNS = [
    "time",
    "L Raw X [px]",
    "L Raw Y [px]",
    "R Raw X [px]",
    "R Raw Y [px]",
    "L POR X [px]",
    "L POR Y [px]",
    "R POR X [px]",
    "R POR Y [px]",
    "L Mapped Diameter [mm]",
    "R Mapped Diameter [mm]",
    "L Validity",
    "R Validity",
    "Pupil Confidence",
    "E8",
    "E32",
    "E82",
]

# Prefer exact SMI names first; include common aliases seen in ET FIF exports.
SOURCE_CANDIDATES = {
    "L Raw X [px]": ["L Raw X [px]", "xpos_left"],
    "L Raw Y [px]": ["L Raw Y [px]", "ypos_left"],
    "R Raw X [px]": ["R Raw X [px]", "xpos_right"],
    "R Raw Y [px]": ["R Raw Y [px]", "ypos_right"],
    "L POR X [px]": ["L POR X [px]"],
    "L POR Y [px]": ["L POR Y [px]"],
    "R POR X [px]": ["R POR X [px]"],
    "R POR Y [px]": ["R POR Y [px]"],
    "L Mapped Diameter [mm]": ["L Mapped Diameter [mm]", "pupil_left"],
    "R Mapped Diameter [mm]": ["R Mapped Diameter [mm]", "pupil_right"],
    "L Validity": ["L Validity", "validity_left"],
    "R Validity": ["R Validity", "validity_right"],
    "Pupil Confidence": ["Pupil Confidence", "pupil_confidence"],
    "E8": ["E8", "EEG E8"],
    "E32": ["E32", "EEG E32"],
    "E82": ["E82", "EEG E82"],
}


def _default_output_path(in_path: Path) -> Path:
    if in_path.suffix:
        return in_path.with_suffix(".tsv")
    return in_path.with_name(f"{in_path.name}.tsv")


def _resolve_source_channels(ch_names: list[str]) -> tuple[dict[str, str], list[str]]:
    found: dict[str, str] = {}
    missing: list[str] = []
    present = set(ch_names)

    for out_col, candidates in SOURCE_CANDIDATES.items():
        src = next((cand for cand in candidates if cand in present), None)
        if src is None:
            missing.append(out_col)
        else:
            found[out_col] = src
    return found, missing


def _fmt(value: float, digits: int = 6) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def convert_fif_to_rows(path: Path) -> tuple[list[dict[str, str]], dict[str, str], list[str]]:
    raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    n_times = int(raw.n_times)
    time_s = raw.times.astype(np.float64)

    found, missing = _resolve_source_channels(raw.ch_names)

    data_by_output = {
        col: np.full(n_times, np.nan, dtype=np.float64) for col in OUTPUT_COLUMNS if col != "time"
    }

    if found:
        src_names = sorted(set(found.values()), key=raw.ch_names.index)
        picks = [raw.ch_names.index(ch) for ch in src_names]
        src_data = raw.get_data(picks=picks).astype(np.float64)
        src_to_data = {src: src_data[idx] for idx, src in enumerate(src_names)}
        for out_col, src_col in found.items():
            data_by_output[out_col] = src_to_data[src_col]

    rows: list[dict[str, str]] = []
    for i in range(n_times):
        row = {"time": _fmt(float(time_s[i]))}
        for out_col in OUTPUT_COLUMNS[1:]:
            row[out_col] = _fmt(float(data_by_output[out_col][i]))
        rows.append(row)

    return rows, found, missing


def write_tsv(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert eye-tracking data from a FIF file into TSV columns: "
            "time plus selected SMI gaze/pupil/validity channels."
        )
    )
    parser.add_argument("input", type=Path, help="Path to input FIF file")
    parser.add_argument("-o", "--output", type=Path, help="Output TSV path (default: input with .tsv)")
    args = parser.parse_args()

    in_path: Path = args.input
    out_path: Path = args.output if args.output else _default_output_path(in_path)

    rows, found, missing = convert_fif_to_rows(in_path)
    write_tsv(rows, out_path)

    print(f"Wrote {len(rows)} rows to {out_path}")
    if found:
        print("Resolved channels:")
        for out_col in OUTPUT_COLUMNS[1:]:
            if out_col in found:
                print(f"  - {out_col} <- {found[out_col]}")
    if missing:
        print("Missing channels (written as empty values):")
        for out_col in missing:
            print(f"  - {out_col}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
