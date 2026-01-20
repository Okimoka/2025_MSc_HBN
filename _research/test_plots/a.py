#!/usr/bin/env python3
"""
Compute correlations between every numeric column and 'mean_abs_sync_error_ms',
rank them by absolute correlation, and plot:
  1) the strongest-correlated metric, and
  2) ALWAYS also 'avg_fixation_duration_ms' (if present and with ≥2 valid pairs).

Filters:
  - Ignore rows where mean_abs_sync_error_ms > 2.5 ms.
  - Only include metrics in the ranking that have at least 10 distinct numeric values
    across ALL rows (the forced plot ignores this uniqueness requirement).

Usage:
  python correlate_and_plot_min_unique_plus_fixation.py overview.xlsx
"""

import sys
import pathlib
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: if SciPy is installed, we'll compute p-values via pearsonr
try:
    from scipy.stats import pearsonr
except Exception:
    pearsonr = None  # p-values will be omitted

TARGET_COL = "mean_abs_sync_error_ms"
FORCED_METRIC = "et_samples"
MAX_TARGET_MS = 200000.5
MIN_UNIQUE_VALUES = 10

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "plot"

def scatter_with_fit(x, y, x_label, out_path, title_prefix, r=None, p=None):
    n = len(x)
    plt.figure()
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel(x_label)
    plt.ylabel("Mean abs sync error (ms)")
    title = f"{title_prefix} (≤ {MAX_TARGET_MS} ms filter, n={n}"
    if r is not None and np.isfinite(r):
        title += f", r={r:.2f}"
    if p is not None and np.isfinite(p):
        title += f", p={p:.3g}"
    title += ")"
    plt.title(title)
    plt.grid(True)

    if n >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(np.min(x), np.max(x), 200)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, linewidth=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.show()

def main(path: str = "overview.xlsx") -> None:
    xlsx_path = pathlib.Path(path)
    if not xlsx_path.exists():
        print(f"Error: file not found: {xlsx_path}")
        sys.exit(1)

    df = pd.read_excel(xlsx_path)

    if TARGET_COL not in df.columns:
        print(f"Error: missing required column '{TARGET_COL}'.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Target + filter
    target = pd.to_numeric(df[TARGET_COL], errors="coerce")
    keep_mask = target.notna() & (target <= MAX_TARGET_MS)
    if not keep_mask.any():
        print(f"Error: no rows remain after filtering {TARGET_COL} <= {MAX_TARGET_MS}.")
        sys.exit(1)

    results = []
    for col in df.columns:
        if col == TARGET_COL:
            continue

        metric_all = pd.to_numeric(df[col], errors="coerce")
        unique_values = metric_all.dropna().nunique()
        if unique_values < MIN_UNIQUE_VALUES:
            continue

        pair_mask = keep_mask & metric_all.notna()
        x = metric_all[pair_mask]
        y = target[pair_mask]
        n = len(x)
        if n < 2:
            continue
        if np.isclose(np.nanstd(x), 0.0) or np.isclose(np.nanstd(y), 0.0):
            continue

        r = np.corrcoef(x.values, y.values)[0, 1]
        if not np.isfinite(r):
            continue

        pval = None
        if pearsonr is not None:
            try:
                r_sc, p_sc = pearsonr(x.values, y.values)
                # pearsonr may return slightly different r due to numeric details; keep r from np for sorting; store p
                pval = float(p_sc)
            except Exception:
                pval = None

        results.append({
            "metric": col,
            "pearson_r": float(r),
            "abs_r": float(abs(r)),
            "n": int(n),
            "unique_values": int(unique_values),
            **({"p_value": pval} if pval is not None else {}),
        })

    if not results:
        print(f"Error: no metrics passed the checks (>= {MIN_UNIQUE_VALUES} unique values, >= 2 paired samples, non-zero variance).")
        sys.exit(1)

    res_df = pd.DataFrame(results).sort_values("abs_r", ascending=False).reset_index(drop=True)

    # Print ranking
    print("\nCorrelation with mean_abs_sync_error_ms (filtered to <= %.3f ms):" % MAX_TARGET_MS)
    display_cols = ["metric", "pearson_r", "abs_r", "n", "unique_values"]
    if "p_value" in res_df.columns:
        display_cols.append("p_value")
    print(res_df[display_cols].to_string(
        index=True,
        justify="left",
        header=True,
        float_format=lambda v: f"{v:0.4f}" if isinstance(v, float) else str(v))
    )

    # Save CSV
    out_csv = xlsx_path.with_name("metric_correlations_vs_sync_error_min_unique.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved full ranking to {out_csv.resolve()}")

    # Plot strongest metric
    top = res_df.iloc[0]
    best_metric = str(top["metric"])
    best_series = pd.to_numeric(df[best_metric], errors="coerce")
    pair_mask = keep_mask & best_series.notna()
    x_best = best_series[pair_mask].values
    y_best = target[pair_mask].values
    n_best = len(x_best)
    r_best = np.corrcoef(x_best, y_best)[0, 1] if n_best >= 2 else float("nan")
    p_best = None
    if pearsonr is not None and n_best >= 2:
        try:
            _, p_best = pearsonr(x_best, y_best)
        except Exception:
            p_best = None
    out_png_best = xlsx_path.with_name(f"{sanitize_filename(best_metric)}_vs_sync_error_filtered.png")
    scatter_with_fit(x_best, y_best, best_metric, str(out_png_best),
                     title_prefix=f"{best_metric} vs sync error", r=r_best, p=p_best)

    # Always also plot the forced metric (if present and with ≥2 valid pairs)
    if FORCED_METRIC in df.columns:
        forced_series = pd.to_numeric(df[FORCED_METRIC], errors="coerce")
        pair_mask_forced = keep_mask & forced_series.notna()
        x_forced = forced_series[pair_mask_forced].values
        y_forced = target[pair_mask_forced].values
        n_forced = len(x_forced)
        if n_forced >= 2:
            r_forced = np.corrcoef(x_forced, y_forced)[0, 1]
            p_forced = None
            if pearsonr is not None:
                try:
                    _, p_forced = pearsonr(x_forced, y_forced)
                except Exception:
                    p_forced = None
            out_png_forced = xlsx_path.with_name(f"{sanitize_filename(FORCED_METRIC)}_vs_sync_error_filtered.png")
            scatter_with_fit(x_forced, y_forced, FORCED_METRIC, str(out_png_forced),
                             title_prefix=f"{FORCED_METRIC} vs sync error", r=r_forced, p=p_forced)
        else:
            print(f"Warning: '{FORCED_METRIC}' cannot be plotted (need ≥2 valid paired samples after filtering).")
    else:
        print(f"Warning: column '{FORCED_METRIC}' not found; forced plot skipped.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "overview.xlsx")
