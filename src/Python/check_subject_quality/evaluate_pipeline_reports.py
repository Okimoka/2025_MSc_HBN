#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


"""
Given a folder of mne bids pipeline report.html files, it filters subjects
based on the criteria given in accept_table and then prints them  

For quicker reexecution, it stores a cache of the ICA Component table that is scraped out of the html

LLM Code
"""

# If needed:
#   pip install beautifulsoup4 lxml
from bs4 import BeautifulSoup


def accept_table(df: pd.DataFrame) -> bool:
    """
    Accept iff within the first 5 rows there exists a row where:
      - Predicted Label == "brain"
      - Max Prob > 0.9
    """
    first5 = df.head(5)
    max_prob = pd.to_numeric(first5["Max Prob"], errors="raise")
    return ((first5["Predicted Label"] == "brain") & (max_prob > 0.9)).any()
    
    """
    Accept iff within the first 30 rows, the count of rows where
    Predicted Label == "brain" is > 20.
    """
    #first30 = df.head(30)
    #brain_count = (first30["Predicted Label"] == "brain").sum()
    #return brain_count > 10

    """
    Accept iff within the first 20 rows:
      - count(Predicted Label == "brain") > 10
      - count(Predicted Label == "other") < 5
    """
    #first20 = df.head(30)
    #brain_count = (first20["Predicted Label"] == "brain").sum()
    #other_count = (first20["Predicted Label"] == "other").sum()
    #return (brain_count > 5) and (other_count < 10)


def report_id_for_path(report_path: Path, base_dir: Path) -> str:
    """Stable ID derived from the report path to name cache files safely."""
    try:
        rel = report_path.relative_to(base_dir)
        s = str(rel)
    except ValueError:
        s = str(report_path)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def is_component_table(table_tag) -> bool:
    """True if this <table> has a <th> whose text is exactly 'Component'."""
    for th in table_tag.find_all("th"):
        if th.get_text(strip=True) == "Component":
            return True
    return False


def load_cached_tables(cache_dir: Path, rep_id: str, mtime_ns: int) -> list[pd.DataFrame] | None:
    """
    Returns:
      - list of DataFrames if cache hit (possibly empty if 'no tables' marker exists)
      - None if no cache info exists for this report file version
    """
    no_tables_marker = cache_dir / f"{rep_id}__{mtime_ns}__NO_COMPONENT_TABLES"
    if no_tables_marker.exists():
        return []

    table_files = sorted(cache_dir.glob(f"{rep_id}__{mtime_ns}__component_tbl*.pkl"))
    if table_files:
        dfs: list[pd.DataFrame] = []
        for pkl in table_files:
            try:
                dfs.append(pd.read_pickle(pkl))
            except Exception:
                return None  # treat as cache miss if corrupted
        return dfs

    return None


def write_cache(cache_dir: Path, rep_id: str, mtime_ns: int, dfs: list[pd.DataFrame]) -> None:
    """Write one pickle per table; or a 'no tables' marker if none."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not dfs:
        marker = cache_dir / f"{rep_id}__{mtime_ns}__NO_COMPONENT_TABLES"
        try:
            marker.write_text("no component tables\n", encoding="utf-8")
        except Exception:
            pass
        return

    for i, df in enumerate(dfs):
        out = cache_dir / f"{rep_id}__{mtime_ns}__component_tbl{i:03d}.pkl"
        try:
            df.to_pickle(out)
        except Exception:
            pass


def parse_component_tables_from_html(html: str) -> list[pd.DataFrame]:
    """Parse all <table> elements that contain <th>Component</th> into DataFrames."""
    soup = BeautifulSoup(html, "html.parser")
    dfs: list[pd.DataFrame] = []

    for table in soup.find_all("table"):
        if not is_component_table(table):
            continue
        df = pd.read_html(str(table))[0]
        dfs.append(df)

    return dfs


def decide_report(dfs: list[pd.DataFrame]) -> bool:
    """
    Decide accept/reject at the report level.

    Current policy:
      - If there are no matching tables, reject the report.
      - Otherwise accept if ANY table passes accept_table(df).
        (Easy to change to ALL, majority vote, max rows, etc.)
    """
    if not dfs:
        return False
    return any(accept_table(df) for df in dfs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize *_report.html component tables with caching + accept/reject.")
    ap.add_argument(
        "report_dir",
        type=Path,
        help="Directory containing *_report.html files to evaluate.",
    )
    ap.add_argument(
        "--max-tables",
        type=int,
        default=None,
        help="Stop after finding this many <th>Component</th> tables (across all reports).",
    )
    ap.add_argument(
        "--cache-dir",
        type=str,
        default="component_table_cache",
        help="Cache directory (created in the current working directory).",
    )
    args = ap.parse_args()

    report_dir = args.report_dir.expanduser().resolve()
    if not report_dir.is_dir():
        raise SystemExit(f"Report directory not found: {report_dir}")

    cache_dir = Path(args.cache_dir).resolve()

    report_paths = sorted(report_dir.glob("*_report.html"))

    report_files_discovered = len(report_paths)
    report_files_processed = 0

    component_tables_found = 0
    total_rows = 0

    reports_with_component_tables = 0
    accepted_reports = 0
    rejected_reports = 0

    accepted_report_files: list[str] = []
    rejected_report_files: list[str] = []


    for report_path in report_paths:
        if args.max_tables is not None and component_tables_found >= args.max_tables:
            break

        report_files_processed += 1

        try:
            mtime_ns = report_path.stat().st_mtime_ns
        except OSError:
            continue

        rep_id = report_id_for_path(report_path, report_dir)

        # Try cache first
        dfs = load_cached_tables(cache_dir, rep_id, mtime_ns)
        if dfs is None:
            # Cache miss: parse HTML
            try:
                html = report_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            dfs = parse_component_tables_from_html(html)
            write_cache(cache_dir, rep_id, mtime_ns, dfs)

        if dfs:
            reports_with_component_tables += 1

        # Update table/row aggregates (respect --max-tables)
        for df in dfs:
            if args.max_tables is not None and component_tables_found >= args.max_tables:
                break
            component_tables_found += 1
            total_rows += len(df)

        # Report-level decision (based on all tables in this report)
        #if decide_report(dfs):
        #    accepted_reports += 1
        #else:
        #    rejected_reports += 1

        accepted = decide_report(dfs)

        if accepted:
            accepted_reports += 1
            accepted_report_files.append(str(report_path))
        else:
            rejected_reports += 1
            rejected_report_files.append(str(report_path))

    avg_rows = (total_rows / component_tables_found) if component_tables_found else 0.0
    total_decided = accepted_reports + rejected_reports
    acc_pct = (accepted_reports / total_decided * 100.0) if total_decided else 0.0
    rej_pct = (rejected_reports / total_decided * 100.0) if total_decided else 0.0

    print(f"Report files found: {report_files_discovered}")
    print(f"ICA component tables: {reports_with_component_tables}")
    print(f"Accepted reports: {accepted_reports} ({acc_pct:.2f}%)")
    print(f"Rejected reports: {rejected_reports} ({rej_pct:.2f}%)")
    #print(f"Cache directory: {cache_dir}")


    print("\n=== Accepted report files ===")
    for p in accepted_report_files:
        print(p)

    #print("\n=== Rejected report files ===")
    #for p in rejected_report_files:
    #    print(p)


if __name__ == "__main__":
    main()
