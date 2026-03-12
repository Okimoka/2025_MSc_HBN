from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # for headless servers
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

"""
Plots the specified ET channels over the time of the entire recording
Plot background shows whether the Events file currently considers the timespan a saccade/fixation/blink
Just for sanity checking synchronity
"""


DEFAULT_ET_SANITY_CHANNELS = [
    #"E8",
    #"E32",
    "L POR X [px]",
    #"R POR Y [px]",
    "L Raw X [px]",
    "L Raw Y [px]",
    #"R Raw X [px]",
    #"R Raw Y [px]",
    "L Mapped Diameter [mm]",
    #"R Mapped Diameter [mm]",
    "L Validity",
    #"R Validity",
    "Pupil Confidence",
]

DERIVED_CHANNELS = {
    "HEOG": ("L Raw X [px]", "R Raw X [px]"),
    "VEOG": ("L Raw Y [px]", "R Raw Y [px]"),
}

EVENT_CODE_TO_NAME = {1: "Fixation", 2: "Saccade", 3: "Blink"}
EVENT_CODE_TO_COLOR = {1: "#6dbf4b", 2: "#f39c12", 3: "#d62728"}
EOG_CHANNELS = {"HEOG", "VEOG"}
CONFIDENCE_CHANNEL = "Pupil Confidence"


def _parse_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    value = value.strip()
    if not value:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _infer_timestamp_scale(timestamps: np.ndarray) -> tuple[float, str]:
    diffs = np.diff(timestamps)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 1.0, "s"

    median_diff = float(np.median(diffs))
    candidates = [
        (1.0, "s"),
        (1e-3, "ms"),
        (1e-6, "us"),
        (1e-9, "ns"),
    ]

    plausible = []
    for scale, unit in candidates:
        step_s = median_diff * scale
        if step_s <= 0:
            continue
        sfreq = 1.0 / step_s
        if 10.0 <= sfreq <= 5000.0:
            plausible.append((abs(sfreq - 120.0), scale, unit))

    if plausible:
        _, scale, unit = min(plausible, key=lambda x: x[0])
        return scale, unit

    if median_diff >= 1000:
        return 1e-6, "us"
    if median_diff >= 1:
        return 1e-3, "ms"
    return 1.0, "s"


def _estimate_sample_rate_hz(time_s: np.ndarray) -> float:
    diffs = np.diff(time_s)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 1.0
    return float(1.0 / np.median(diffs))


def _event_type_to_code(event_type: str | None) -> int:
    if event_type is None:
        return 0
    t = event_type.strip().lower()
    if t == "blink":
        return 3
    if t == "saccade":
        return 2
    if t == "fixation":
        return 1
    return 0


def _channel_eye_and_kind(ch_name: str) -> tuple[str, bool, bool]:
    is_eog = ch_name in EOG_CHANNELS
    is_et = not is_eog
    if ch_name.startswith("L "):
        return "L", is_et, is_eog
    if ch_name.startswith("R "):
        return "R", is_et, is_eog
    return "B", is_et, is_eog


def _channel_relevant_for_event(
    *,
    ch_eye: str,
    is_et: bool,
    is_eog: bool,
    event_code: int,
    event_eye: str | None,
) -> bool:
    if event_code == 3:
        return is_et or is_eog

    if is_eog:
        return True
    if event_eye is None:
        return True
    if ch_eye == "B":
        return True
    return ch_eye == event_eye


def _mask_broken_et_samples_for_plot(
    *,
    data: np.ndarray,
    channel_names: list[str],
    mask_mode: str = "zero_or_validity",
    validity_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
    zero_eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Mask broken ET samples so they are not drawn as valid zeros.

    `mask_mode` options:
    - "off": no masking
    - "zero": only classic all-zero/monocular-zero masking
    - "validity": only validity/confidence-threshold masking
    - "zero_or_validity": combine both approaches (recommended for mixed sources)
    """
    cleaned = data.copy()
    n_times = cleaned.shape[1] if cleaned.ndim == 2 else 0
    if cleaned.size == 0 or n_times == 0:
        return cleaned, {
            "all_zero": 0,
            "left_eye_zero": 0,
            "right_eye_zero": 0,
            "left_validity_low": 0,
            "right_validity_low": 0,
            "confidence_low": 0,
            "all_invalid": 0,
        }
    if mask_mode == "off":
        return cleaned, {
            "all_zero": 0,
            "left_eye_zero": 0,
            "right_eye_zero": 0,
            "left_validity_low": 0,
            "right_validity_low": 0,
            "confidence_low": 0,
            "all_invalid": 0,
        }

    left_idx = [i for i, ch in enumerate(channel_names) if ch.startswith("L ")]
    right_idx = [i for i, ch in enumerate(channel_names) if ch.startswith("R ")]
    eog_idx = [i for i, ch in enumerate(channel_names) if ch in EOG_CHANNELS]
    idx_by_name = {ch: i for i, ch in enumerate(channel_names)}
    all_idx = list(range(cleaned.shape[0]))

    def _all_zero(indices: list[int]) -> np.ndarray:
        if not indices:
            return np.zeros(n_times, dtype=bool)
        vals = cleaned[indices]
        finite = np.isfinite(vals)
        any_finite = np.any(finite, axis=0)
        all_zero = np.all(np.where(finite, np.abs(vals) <= zero_eps, True), axis=0)
        return any_finite & all_zero

    def _any_nonzero(indices: list[int]) -> np.ndarray:
        if not indices:
            return np.zeros(n_times, dtype=bool)
        vals = cleaned[indices]
        finite = np.isfinite(vals)
        return np.any(finite & (np.abs(vals) > zero_eps), axis=0)

    all_zero = np.zeros(n_times, dtype=bool)
    left_eye_zero = np.zeros(n_times, dtype=bool)
    right_eye_zero = np.zeros(n_times, dtype=bool)
    if mask_mode in {"zero", "zero_or_validity"}:
        all_zero = _all_zero(all_idx)
        left_all_zero = _all_zero(left_idx)
        right_all_zero = _all_zero(right_idx)
        left_any_nonzero = _any_nonzero(left_idx)
        right_any_nonzero = _any_nonzero(right_idx)
        left_eye_zero = left_all_zero & right_any_nonzero & ~all_zero
        right_eye_zero = right_all_zero & left_any_nonzero & ~all_zero

    left_validity_low = np.zeros(n_times, dtype=bool)
    right_validity_low = np.zeros(n_times, dtype=bool)
    confidence_low = np.zeros(n_times, dtype=bool)
    if mask_mode in {"validity", "zero_or_validity"}:
        if "L Validity" in idx_by_name:
            lv = cleaned[idx_by_name["L Validity"]]
            left_validity_low = np.isfinite(lv) & (lv < validity_threshold)
        if "R Validity" in idx_by_name:
            rv = cleaned[idx_by_name["R Validity"]]
            right_validity_low = np.isfinite(rv) & (rv < validity_threshold)
        if CONFIDENCE_CHANNEL in idx_by_name:
            conf = cleaned[idx_by_name[CONFIDENCE_CHANNEL]]
            confidence_low = np.isfinite(conf) & (conf < confidence_threshold)

    all_invalid = all_zero | confidence_low | (left_validity_low & right_validity_low)
    left_invalid = (left_eye_zero | left_validity_low) & ~all_invalid
    right_invalid = (right_eye_zero | right_validity_low) & ~all_invalid

    if np.any(all_invalid):
        cleaned[:, all_invalid] = np.nan
    if np.any(left_invalid) and left_idx:
        left_cols = np.flatnonzero(left_invalid)
        cleaned[np.ix_(left_idx, left_cols)] = np.nan
    if np.any(right_invalid) and right_idx:
        right_cols = np.flatnonzero(right_invalid)
        cleaned[np.ix_(right_idx, right_cols)] = np.nan
    monocular_invalid = left_invalid | right_invalid
    if np.any(monocular_invalid) and eog_idx:
        mono_cols = np.flatnonzero(monocular_invalid)
        cleaned[np.ix_(eog_idx, mono_cols)] = np.nan

    stats = {
        "all_zero": int(np.sum(all_zero)),
        "left_eye_zero": int(np.sum(left_eye_zero)),
        "right_eye_zero": int(np.sum(right_eye_zero)),
        "left_validity_low": int(np.sum(left_validity_low)),
        "right_validity_low": int(np.sum(right_validity_low)),
        "confidence_low": int(np.sum(confidence_low)),
        "all_invalid": int(np.sum(all_invalid)),
    }
    return cleaned, stats


def _apply_robust_ylim(ax: plt.Axes, values: np.ndarray, central_pct: float) -> None:
    if not (0.0 < central_pct <= 100.0):
        return
    finite = values[np.isfinite(values)]
    if finite.size < 5:
        return

    tail = (100.0 - central_pct) / 2.0
    low = float(np.percentile(finite, tail))
    high = float(np.percentile(finite, 100.0 - tail))

    if not np.isfinite(low) or not np.isfinite(high):
        return
    if high <= low:
        center = float(np.median(finite))
        span = float(np.std(finite))
        if span <= 0:
            span = max(abs(center), 1.0) * 0.05
        low = center - span
        high = center + span

    pad = 0.05 * (high - low)
    ax.set_ylim(low - pad, high + pad)


def _load_samples_tsv(
    samples_path: Path,
    requested_channels: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str], float, str]:
    required_source_columns: set[str] = set()
    for ch in requested_channels:
        if ch in DERIVED_CHANNELS:
            required_source_columns.update(DERIVED_CHANNELS[ch])
        else:
            required_source_columns.add(ch)

    with samples_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or "time" not in reader.fieldnames:
            raise RuntimeError(
                f"{samples_path} does not have the required tabular header with 'time' column."
            )

        present_source = [c for c in required_source_columns if c in set(reader.fieldnames)]
        source_lists = {c: [] for c in present_source}
        sample_times_raw: list[float] = []

        for row in reader:
            ts_raw = _parse_float(row.get("time"))
            if not np.isfinite(ts_raw):
                continue
            sample_times_raw.append(ts_raw)
            for col in present_source:
                source_lists[col].append(_parse_float(row.get(col)))

    if not sample_times_raw:
        raise RuntimeError(f"No valid sample rows found in {samples_path}.")

    raw_timestamps = np.asarray(sample_times_raw, dtype=np.float64)
    scale_to_seconds, timestamp_unit = _infer_timestamp_scale(raw_timestamps)
    t0_raw = float(raw_timestamps[0])
    time_s = (raw_timestamps - t0_raw) * scale_to_seconds

    channel_data: dict[str, np.ndarray] = {
        col: np.asarray(values, dtype=np.float32) for col, values in source_lists.items()
    }

    for out_ch, (left_ch, right_ch) in DERIVED_CHANNELS.items():
        if out_ch not in requested_channels:
            continue
        if left_ch in channel_data and right_ch in channel_data:
            channel_data[out_ch] = channel_data[left_ch] - channel_data[right_ch]

    selected_channels = [ch for ch in requested_channels if ch in channel_data]
    if not selected_channels:
        raise RuntimeError(
            "None of the requested ET sanity channels were found in the samples TSV. "
            f"Requested: {requested_channels}"
        )

    return raw_timestamps, time_s, channel_data, selected_channels, scale_to_seconds, timestamp_unit


def _load_events_tsv(
    events_path: Path,
    *,
    sample_t0_raw: float,
    sample_scale_to_seconds: float,
) -> list[dict[str, str | float | int]]:
    events: list[dict[str, str | float | int]] = []
    with events_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"type", "eye", "onset", "duration"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise RuntimeError(
                f"{events_path} must contain columns: type, eye, onset, duration."
            )

        for row in reader:
            event_code = _event_type_to_code(row.get("type"))
            if event_code == 0:
                continue

            eye = (row.get("eye") or "").strip().upper()
            if eye not in {"L", "R"}:
                continue

            onset_raw = _parse_float(row.get("onset"))
            duration_raw = _parse_float(row.get("duration"))
            if not np.isfinite(onset_raw) or not np.isfinite(duration_raw):
                continue

            onset_s = (float(onset_raw) - sample_t0_raw) * sample_scale_to_seconds
            duration_s = max(0.0, float(duration_raw) * sample_scale_to_seconds)
            events.append(
                {
                    "type": EVENT_CODE_TO_NAME[event_code].lower(),
                    "eye": eye,
                    "event_code": event_code,
                    "onset_s": onset_s,
                    "duration_s": duration_s,
                }
            )

    events.sort(key=lambda r: (float(r["onset_s"]), int(r["event_code"]), str(r["eye"])))
    return events


def _build_channel_event_code_timelines(
    *,
    time: np.ndarray,
    event_rows: list[dict[str, str | float | int]],
    sfreq: float,
    channel_names: list[str],
) -> np.ndarray:
    n_ch = len(channel_names)
    code = np.zeros((n_ch, len(time)), dtype=np.uint8)
    if len(time) == 0 or n_ch == 0:
        return code

    meta = [_channel_eye_and_kind(ch) for ch in channel_names]
    min_dur = 1.0 / max(float(sfreq), 1e-9)
    t_start = float(time[0])
    t_stop = float(time[-1] + min_dur)

    for row in event_rows:
        event_code = int(row["event_code"])
        event_eye = str(row["eye"])
        t0 = float(row["onset_s"])
        t1 = float(row["onset_s"]) + max(float(row["duration_s"]), min_dur)
        if t1 <= t_start or t0 >= t_stop:
            continue

        i0 = int(np.searchsorted(time, t0, side="left"))
        i1 = int(np.searchsorted(time, t1, side="right"))
        if i0 >= code.shape[1]:
            continue
        i0 = max(i0, 0)
        i1 = min(max(i1, i0 + 1), code.shape[1])

        for ch_idx, (ch_eye, is_et, is_eog) in enumerate(meta):
            if not _channel_relevant_for_event(
                ch_eye=ch_eye,
                is_et=is_et,
                is_eog=is_eog,
                event_code=event_code,
                event_eye=event_eye,
            ):
                continue
            window = code[ch_idx, i0:i1]
            np.maximum(window, event_code, out=window)

    return code


def _event_segments_from_code(time: np.ndarray, event_code: np.ndarray) -> list[tuple[float, float, int]]:
    if len(time) == 0 or len(event_code) == 0:
        return []

    edges = np.flatnonzero(np.diff(event_code) != 0) + 1
    edges = np.concatenate(([0], edges, [len(event_code)]))
    dt = float(np.median(np.diff(time))) if len(time) > 1 else 0.0

    segments: list[tuple[float, float, int]] = []
    for start_idx, stop_idx in zip(edges[:-1], edges[1:]):
        c = int(event_code[start_idx])
        if c == 0:
            continue
        x0 = float(time[start_idx])
        if stop_idx < len(time):
            x1 = float(time[stop_idx])
        else:
            x1 = float(time[-1] + dt)
        if x1 > x0:
            segments.append((x0, x1, c))
    return segments


def plot_eyetracking_event_sanity_from_tsv(
    *,
    samples_path: str | Path = "Samples.tsv",
    events_path: str | Path = "Events.tsv",
    channels: list[str] | tuple[str, ...] | None = None,
    dpi: int = 1000,
    width_in: float = 100,
    height_per_channel_in: float = 1.35,
    line_width: float = 0.45,
    max_points: int = 300_000,
    shade_alpha: float = 0.18,
    mask_mode: str = "zero_or_validity",
    validity_threshold: float = 0.9,
    confidence_threshold: float = 1.0,
    robust_ylim_percentile: float = 95.0,
    out_path: str | Path | None = None,
) -> Path:
    samples_path = Path(samples_path)
    events_path = Path(events_path)

    requested = list(DEFAULT_ET_SANITY_CHANNELS if channels is None else channels)
    raw_timestamps, time_s, channel_data, selected, scale_to_seconds, timestamp_unit = _load_samples_tsv(
        samples_path=samples_path,
        requested_channels=requested,
    )
    event_rows = _load_events_tsv(
        events_path=events_path,
        sample_t0_raw=float(raw_timestamps[0]),
        sample_scale_to_seconds=scale_to_seconds,
    )

    sfreq = _estimate_sample_rate_hz(time_s)
    decim = max(1, int(np.ceil(len(time_s) / max_points)))
    time_plot = time_s[::decim].astype(np.float64)
    data = np.vstack([channel_data[ch] for ch in selected]).astype(np.float32)[:, ::decim]
    data, broken_stats = _mask_broken_et_samples_for_plot(
        data=data,
        channel_names=selected,
        mask_mode=mask_mode,
        validity_threshold=validity_threshold,
        confidence_threshold=confidence_threshold,
    )

    channel_event_code = _build_channel_event_code_timelines(
        time=time_plot,
        event_rows=event_rows,
        sfreq=sfreq,
        channel_names=selected,
    )
    segments_per_channel = [
        _event_segments_from_code(time_plot, channel_event_code[i]) for i in range(len(selected))
    ]

    n_ch = len(selected)
    height_in = max(9.0, n_ch * height_per_channel_in)
    fig, axes = plt.subplots(n_ch, 1, sharex=True, figsize=(width_in, height_in), dpi=dpi)
    if n_ch == 1:
        axes = [axes]

    for ch_idx, (ax, ch_name, ch_data) in enumerate(zip(axes, selected, data)):
        for x0, x1, c in segments_per_channel[ch_idx]:
            ax.axvspan(x0, x1, color=EVENT_CODE_TO_COLOR[c], alpha=shade_alpha, linewidth=0, zorder=0)
        ax.plot(time_plot, ch_data, color="black", lw=line_width, zorder=1)
        _apply_robust_ylim(ax, ch_data, robust_ylim_percentile)
        ax.set_ylabel(ch_name, fontsize=8)
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.3)

    legend_handles = [
        Patch(facecolor=EVENT_CODE_TO_COLOR[c], edgecolor="none", alpha=shade_alpha, label=EVENT_CODE_TO_NAME[c])
        for c in (1, 2, 3)
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", framealpha=0.85)
    axes[-1].set_xlabel("Time (s)")

    fig.suptitle(
        f"ET sanity plot | {samples_path.name} + {events_path.name} | timestamp unit: {timestamp_unit}",
        fontsize=12,
        y=0.995,
    )
    fig.subplots_adjust(left=0.13, right=0.995, top=0.97, bottom=0.05, hspace=0.08)

    if out_path is None:
        out_path = samples_path.with_name(f"{samples_path.stem}_et_event_sanity.png")
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    combined_event_code = (
        np.max(channel_event_code, axis=0) if channel_event_code.size else np.array([], dtype=np.uint8)
    )
    kept_counts = {name: int(np.sum(combined_event_code == c)) for c, name in EVENT_CODE_TO_NAME.items()}
    print(f"[et_sanity_tsv] Used channels ({len(selected)}): {selected}")
    print(f"[et_sanity_tsv] Decimation factor: {decim}")
    print(
        "[et_sanity_tsv] Mask mode / thresholds: "
        f"mode={mask_mode}, validity<{validity_threshold}, confidence<{confidence_threshold}"
    )
    print(f"[et_sanity_tsv] Robust y-limit percentile: {robust_ylim_percentile}")
    print(
        "[et_sanity_tsv] Masked broken samples: "
        f"all-zero={broken_stats['all_zero']}, "
        f"left-eye-zero={broken_stats['left_eye_zero']}, "
        f"right-eye-zero={broken_stats['right_eye_zero']}, "
        f"left-validity-low={broken_stats['left_validity_low']}, "
        f"right-validity-low={broken_stats['right_validity_low']}, "
        f"confidence-low={broken_stats['confidence_low']}, "
        f"all-invalid={broken_stats['all_invalid']}"
    )
    print(f"[et_sanity_tsv] Sample rows: {len(time_s)} | Event rows: {len(event_rows)}")
    print(f"[et_sanity_tsv] Event-coded samples: {kept_counts}")
    print(f"[et_sanity_tsv] Wrote: {out_path}")
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot ET sanity channels from Samples.tsv and shade Fixation/Saccade/Blink "
            "regions from Events.tsv."
        )
    )
    parser.add_argument(
        "--samples-path",
        default="Samples.tsv",
        help="Path to samples TSV (default: Samples.tsv)",
    )
    parser.add_argument(
        "--events-path",
        default="Events.tsv",
        help="Path to events TSV (default: Events.tsv)",
    )
    parser.add_argument(
        "--out-path",
        default=None,
        help="Output PNG path (default: <samples_stem>_et_event_sanity.png)",
    )
    parser.add_argument(
        "--channels",
        default=None,
        help="Comma-separated channel names to plot (default matches plot_data.py).",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--width-in", type=float, default=30)
    parser.add_argument("--height-per-channel-in", type=float, default=1.35)
    parser.add_argument("--line-width", type=float, default=0.45)
    parser.add_argument("--max-points", type=int, default=300_000)
    parser.add_argument("--shade-alpha", type=float, default=0.18)
    parser.add_argument(
        "--mask-mode",
        choices=("off", "zero", "validity", "zero_or_validity"),
        default="zero_or_validity",
        help=(
            "How to detect invalid ET samples: off|zero|validity|zero_or_validity "
            "(default: zero_or_validity)."
        ),
    )
    parser.add_argument(
        "--validity-threshold",
        type=float,
        default=0.9,
        help="Mask samples with L/R Validity below this threshold (default: 0.9).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=1.0,
        help="Mask samples with Pupil Confidence below this threshold (default: 1.0).",
    )
    parser.add_argument(
        "--robust-ylim-percentile",
        type=float,
        default=95.0,
        help=(
            "Set per-channel y-limits to central percentile range (0-100]. "
            "Use 100 to disable outlier clipping (default: 95.0)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    channels = None
    if args.channels:
        channels = [c.strip() for c in args.channels.split(",") if c.strip()]

    plot_eyetracking_event_sanity_from_tsv(
        samples_path=args.samples_path,
        events_path=args.events_path,
        channels=channels,
        dpi=args.dpi,
        width_in=args.width_in,
        height_per_channel_in=args.height_per_channel_in,
        line_width=args.line_width,
        max_points=args.max_points,
        shade_alpha=args.shade_alpha,
        mask_mode=args.mask_mode,
        validity_threshold=args.validity_threshold,
        confidence_threshold=args.confidence_threshold,
        robust_ylim_percentile=args.robust_ylim_percentile,
        out_path=args.out_path,
    )


if __name__ == "__main__":
    main()
