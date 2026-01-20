from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")  # for headless servers
import matplotlib.pyplot as plt
import mne
import matplotlib.ticker as mticker

def plot_processed_eeg_overview(
    *,
    subject: str,
    task: str,
    processed_dir: str | Path = "processed",
    epo: bool = False,
    run: str | int | None = None,  # optional disambiguation; not required
    exclude_bads: bool = True,     # toggles inclusion/exclusion of bad channels (info['bads'] + *_bads.tsv)
    dpi: int = 200,
    width_in: float = 40.0,
    height_per_channel_in: float = 0.22,
    line_width: float = 0.35,
    color_cycle: str = "tab10",
    max_points: int = 300_000,     # automatic decimation target to keep PNG size/memory reasonable
    offset_multiplier: float = 1.2,
    label_fontsize: float = 7.0,
    out_path: str | Path | None = None,
) -> Path:
    """
    Create a large PNG showing all EEG channels at once (no normalization; traces may overlap).

    - Removes annotations immediately after reading (Raw or Epochs' underlying raw if present).
    - Robust per-channel vertical offset is chosen automatically from data percentiles.
    - For epochs: reconstructs an accurate timeline with gaps for dropped epochs using epochs.selection/drop_log.
      Also draws vertical dotted lines at epoch boundaries (including dropped ones).
    - Optionally excludes channels listed in processed/*_bads.tsv (column 'name') and info['bads'].

    Returns
    -------
    Path to the written PNG.
    """
    processed_dir = Path(processed_dir)

    fif_suffix = "_proc-icafit_epo.fif" if epo else "_proc-filt_raw.fif"
    run_pat = f"_run-{run}_" if run is not None else "_run-*_"

    # Build a glob pattern that works with or without run in the filename
    base_pat = f"sub-{subject}_task-{task}_"
    if run is None:
        fif_glob = f"{base_pat}*{fif_suffix}"
    else:
        fif_glob = f"{base_pat}{run_pat}*{fif_suffix}"

    candidates = sorted(processed_dir.glob(fif_glob))
    if not candidates:
        # fallback: sometimes files may omit run or have extra entities; broaden slightly
        candidates = sorted(processed_dir.glob(f"sub-{subject}_task-{task}*{fif_suffix}"))
    if not candidates:
        raise FileNotFoundError(f"No matching FIF found in {processed_dir} for: {fif_glob}")

    fif_path = candidates[0]
    if len(candidates) > 1:
        # deterministic choice without interactive prompt
        # (use run=... to disambiguate if needed)
        print(f"[plot_processed_eeg_overview] Multiple matches; using: {fif_path.name}")
        print("  Other matches:")
        for p in candidates[1:]:
            print(f"   - {p.name}")

    # Read bad channels from TSV (preferred: same task; fallback: any task for subject)
    bads_from_tsv: set[str] = set()
    tsv_candidates = sorted(processed_dir.glob(f"sub-{subject}_task-{task}*_bads.tsv"))
    if not tsv_candidates:
        tsv_candidates = sorted(processed_dir.glob(f"sub-{subject}_task-*_bads.tsv"))
    if tsv_candidates:
        tsv_path = tsv_candidates[0]
        try:
            with tsv_path.open("r", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                if "name" in (reader.fieldnames or []):
                    for row in reader:
                        name = (row.get("name") or "").strip()
                        if name:
                            bads_from_tsv.add(name)
        except Exception as e:
            print(f"[plot_processed_eeg_overview] Could not read TSV {tsv_path.name}: {e}")

    if epo:
        epochs = mne.read_epochs(fif_path, preload=True, verbose="ERROR")

        # Remove annotations immediately after reading, if an underlying raw exists
        # Remove annotations immediately after reading, if an underlying raw exists
        raw_like = getattr(epochs, "_raw", None)
        empty_annot = mne.Annotations(onset=[], duration=[], description=[])

        if raw_like is None:
            pass
        elif isinstance(raw_like, list):
            for r in raw_like:
                if hasattr(r, "set_annotations"):
                    r.set_annotations(empty_annot)
        elif hasattr(raw_like, "set_annotations"):
            raw_like.set_annotations(empty_annot)


        info = epochs.info
        all_bads = set(info.get("bads", [])) | bads_from_tsv
        exclude_list = list(all_bads) if exclude_bads else []
        picks = mne.pick_types(info, eeg=True, meg=False, eog=False, ecg=False, emg=False, stim=False,
                              misc=False, exclude=exclude_list)
        ch_names = [info["ch_names"][i] for i in picks]
        n_ch = len(ch_names)
        if n_ch == 0:
            raise RuntimeError("No EEG channels selected (check exclude_bads / channel types).")

        # Reconstruct fixed-length schedule (with gaps) from selection/events/drop_log
        sfreq = float(info["sfreq"])
        n_original = len(epochs.drop_log)
        selection = np.array(epochs.selection, dtype=int)  # indices into original epoch list
        kept_samples = np.array(epochs.events[:, 0], dtype=float)

        if len(selection) >= 2:
            d_sel = np.diff(selection)
            d_samp = np.diff(kept_samples)
            mask = d_sel > 0
            if np.any(mask):
                step_samples = np.median(d_samp[mask] / d_sel[mask])
            else:
                step_samples = np.median(np.diff(kept_samples))
        else:
            # degenerate case: only one epoch kept
            step_samples = int(round((epochs.tmax - epochs.tmin) * sfreq))

        step_samples = int(round(step_samples))
        if step_samples <= 0:
            step_samples = int(round((epochs.tmax - epochs.tmin) * sfreq))

        # Estimate total number of samples to size/decimate sensibly
        epoch_len_sec = float(epochs.times[-1] - epochs.times[0] + np.median(np.diff(epochs.times)))
        total_dur_sec = (n_original - 1) * (step_samples / sfreq) + epoch_len_sec
        approx_total_points = total_dur_sec * sfreq

        decim = max(1, int(np.ceil(approx_total_points / max_points)))
        if decim > 1:
            epochs.decimate(decim, verbose="ERROR")

        # After decimation, update timing quantities
        times_ep = epochs.times
        dt = float(np.median(np.diff(times_ep))) if len(times_ep) > 1 else 1.0 / sfreq
        epoch_len = len(times_ep)
        epoch_t0 = float(times_ep[0])
        step_sec = step_samples / sfreq

        total_dur_sec = (n_original - 1) * step_sec + float(times_ep[-1] - times_ep[0] + dt)
        time = (np.arange(0, total_dur_sec / dt, 1.0, dtype=np.float64) * dt) + epoch_t0
        n_points = len(time)

        data_ep = epochs.get_data(picks=picks).astype(np.float32)  # (n_kept, n_ch, n_t)

        # Accumulate into a continuous timeline with NaN gaps (and averaging if overlap exists)
        sums = np.zeros((n_ch, n_points), dtype=np.float32)
        counts = np.zeros((n_ch, n_points), dtype=np.uint16)

        for k, orig_idx in enumerate(selection):
            start_sec = orig_idx * step_sec + epoch_t0
            i0 = int(round((start_sec - time[0]) / dt))
            i1 = i0 + epoch_len
            if i1 <= 0 or i0 >= n_points:
                continue
            sl0 = max(i0, 0)
            sl1 = min(i1, n_points)
            ep0 = sl0 - i0
            ep1 = ep0 + (sl1 - sl0)
            sums[:, sl0:sl1] += data_ep[k, :, ep0:ep1]
            counts[:, sl0:sl1] += 1

        data = np.full((n_ch, n_points), np.nan, dtype=np.float32)
        mask = counts > 0
        data[mask] = (sums[mask] / counts[mask].astype(np.float32))
        data_for_offset = data

        epoch_boundary_times = [epoch_t0 + i * step_sec for i in range(n_original)]

    else:
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")

        # Remove annotations immediately after reading
        raw.set_annotations(mne.Annotations([], [], []))

        info = raw.info
        all_bads = set(info.get("bads", [])) | bads_from_tsv
        exclude_list = list(all_bads) if exclude_bads else []
        picks = mne.pick_types(info, eeg=True, meg=False, eog=False, ecg=False, emg=False, stim=False,
                              misc=False, exclude=exclude_list)
        ch_names = [info["ch_names"][i] for i in picks]
        n_ch = len(ch_names)
        if n_ch == 0:
            raise RuntimeError("No EEG channels selected (check exclude_bads / channel types).")

        n_times = raw.n_times
        decim = max(1, int(np.ceil(n_times / max_points)))

        data = raw.get_data(picks=picks).astype(np.float32)[:, ::decim]
        data_for_offset = data  # keep unscaled for spacing/offset computation
        data = data * 5.0      # scale only the raw signal strength
        time = raw.times[::decim].astype(np.float64)
        epoch_boundary_times = None

    # Robust offset from per-channel robust peak-to-peak (1st..99th percentile)
    q = np.nanpercentile(data_for_offset, [1, 99], axis=1)
    ptp_robust = (q[1] - q[0]).astype(np.float64)
    base = float(np.nanpercentile(ptp_robust, 75)) if np.any(np.isfinite(ptp_robust)) else 1.0
    if not np.isfinite(base) or base <= 0:
        base = 1.0
    offset = offset_multiplier * base
    print(f"[overview] Channel spacing (offset): {offset:.3e} V  = {offset*1e6:.2f} µV")

    # Figure sizing
    height_in = max(6.0, n_ch * height_per_channel_in)
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    palette = list(plt.get_cmap("tab10").colors)

    y_positions = -np.arange(n_ch, dtype=np.float64) * offset
    for i in range(n_ch):
        color = palette[i % len(palette)]
        ax.plot(time, data[i, :] + y_positions[i], lw=line_width, color=color)

    # Channel labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ch_names, fontsize=label_fontsize)

    # Epoch boundary lines (epochs only)
    if epoch_boundary_times is not None:
        for x in epoch_boundary_times:
            ax.axvline(x, linestyle=":", linewidth=0.6, color="k", alpha=0.25)

    ax.set_xlabel("Time (s)")
    ax.set_title(f"sub-{subject} task-{task} ({'epochs' if epo else 'raw'}) | {fif_path.name}", fontsize=10)

    # Tight-ish layout but keep performance reasonable
    fig.subplots_adjust(left=0.18, right=0.995, top=0.98, bottom=0.04)

    # No x-padding; start exactly at x=0 (or at the earliest time if negative)
    left = 0.0 if time[0] >= 0 else float(time[0])
    ax.set_xlim(left=left, right=float(time[-1]))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(15.0))
    ax.margins(x=0)


    if out_path is None:
        run_tag = f"_run-{run}" if run is not None else ""
        kind = "epo" if epo else "raw"
        out_path = processed_dir / f"sub-{subject}_task-{task}{run_tag}_{kind}_overview.png"
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    # Examples
    png1 = plot_processed_eeg_overview(
        subject="NDARJA830BYV",
        task="freeView",
        processed_dir="processed",
        epo=False,
        exclude_bads=True,
        dpi=200,
        width_in=80,
        line_width=0.1,
    )
    print("Wrote:", png1)

    png2 = plot_processed_eeg_overview(
        subject="NDARJA830BYV",
        task="freeView",
        processed_dir="processed",
        epo=True,
        exclude_bads=True,
        dpi=270,
        width_in=80,
        line_width=0.1,
    )
    print("Wrote:", png2)
