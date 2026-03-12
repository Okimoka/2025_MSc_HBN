#!/usr/bin/env python3
"""
Evaluates suitable subjects from the metrics overview csv

Filter subjects in a CSV using sequential QC criteria and log:
- how many subjects were dropped at each step (among those still in consideration)
- what percentage of ALL original subjects fulfill each criterion

Usage:
  python filter_subjects.py input.csv -o filtered.csv

Notes:
- Any NaN / non-numeric value in a criterion column is treated as FAIL for that criterion.
- The output is sorted descending by xcorr_peak.

LLM Code
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Tuple

import pandas as pd


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def make_criteria() -> list[Tuple[str, Callable[[pd.DataFrame], pd.Series]]]:
    return [
        ("et_sampling_rate_hz >= 58",
         lambda d: _to_num(d["et_sampling_rate_hz"]).ge(58)),
        ("|gauss_mu| <= 200",
         lambda d: _to_num(d["gauss_mu"]).abs().le(200)),
        ("gauss_xcorr_cosine_similarity_95 >= 0.98",
         lambda d: _to_num(d["gauss_xcorr_cosine_similarity_95"]).ge(0.98)),
        ("gauss_xcorr_cosine_similarity_full >= 0.90",
         lambda d: _to_num(d["gauss_xcorr_cosine_similarity_full"]).ge(0.90)),
        ("|xcorr_peak_idx| <= 200",
         lambda d: _to_num(d["xcorr_peak_idx"]).abs().le(200)),
        ("shared_events >= 10",
         lambda d: _to_num(d["shared_events"]).ge(10)),
    ]


def apply_filters(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if "subject" not in df.columns:
        raise ValueError("Expected a 'subject' column.")

    # Keep a pristine copy for "% of all subjects fulfill this criterion"
    df_all = df.copy()
    n_all = len(df_all)

    # Optional: drop exact duplicate subjects (keep first)
    if df["subject"].duplicated().any():
        n_dup = int(df["subject"].duplicated().sum())
        logger.warning("Found %d duplicated subject ids; keeping first occurrence.", n_dup)
        df = df.drop_duplicates(subset=["subject"], keep="first").copy()
        df_all = df_all.drop_duplicates(subset=["subject"], keep="first").copy()
        n_all = len(df_all)

    logger.info("Total subjects (after de-dup, if any): %d", n_all)

    criteria = make_criteria()

    remaining = df.copy()
    for step_idx, (name, fn) in enumerate(criteria, start=1):
        n_before = len(remaining)

        # Criterion for remaining set (sequential filtering)
        mask_remaining = fn(remaining).fillna(False)
        remaining = remaining.loc[mask_remaining].copy()
        n_after = len(remaining)
        dropped = n_before - n_after

        # Criterion fulfillment among ALL subjects
        mask_all = fn(df_all).fillna(False)
        n_fulfill_all = int(mask_all.sum())
        pct_all = (n_fulfill_all / n_all * 100.0) if n_all else 0.0

        logger.info(
            "Step %d: %s | dropped this step: %d (from %d -> %d) | "
            "fulfill among ALL: %d/%d (%.2f%%)",
            step_idx, name, dropped, n_before, n_after, n_fulfill_all, n_all, pct_all
        )

    # Sort descending by xcorr_peak (numeric; NaNs go last)
    remaining["xcorr_peak"] = _to_num(remaining["xcorr_peak"])
    remaining = remaining.sort_values(by="xcorr_peak", ascending=False, na_position="last")

    logger.info("Final selected subjects: %d", len(remaining))
    return remaining


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter suitable subjects from a CSV.")
    ap.add_argument("input_csv", type=Path, help="Path to input CSV/TSV file")
    ap.add_argument("-o", "--output_csv", type=Path, default=None, help="Path to write filtered CSV")
    ap.add_argument("--log", type=Path, default=None, help="Optional path to write a log file")
    ap.add_argument("--sep", default=None,
                    help="Optional delimiter (e.g. ',' or '\\t'). If omitted, pandas will try to infer.")
    args = ap.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log is not None:
        handlers.append(logging.FileHandler(args.log, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )
    logger = logging.getLogger("subject_filter")

    if not args.input_csv.exists():
        raise FileNotFoundError(args.input_csv)

    # Try to infer delimiter if not provided
    read_kwargs = {}
    if args.sep is None:
        read_kwargs.update(dict(sep=None, engine="python"))
    else:
        read_kwargs.update(dict(sep=args.sep))

    df = pd.read_csv(args.input_csv, **read_kwargs)

    filtered = apply_filters(df, logger)

    if args.output_csv is not None:
        filtered.to_csv(args.output_csv, index=False)
        logger.info("Wrote filtered CSV: %s", args.output_csv)


if __name__ == "__main__":
    main()