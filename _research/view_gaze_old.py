import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider


def load_idf(path):
    """
    Parse an SMI IDF-converted text file.

    - Reads metadata (Calibration Area) for screen size.
    - Finds the header line starting with 'Time'.
    - Reads only data lines (skips MSG lines).
    """
    path = Path(path)
    screen_w = None
    screen_h = None

    header = None
    data_lines = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # Grab screen size from Calibration Area
            if line.startswith("## Calibration Area"):
                parts = line.split("\t")
                if len(parts) >= 3:
                    try:
                        screen_w = int(parts[1])
                        screen_h = int(parts[2])
                    except ValueError:
                        pass
                continue

            # Until we find the data header, just keep scanning
            if header is None:
                if line.startswith("Time\t"):
                    header = line.split("\t")
                continue

            # After header: actual data lines
            if not line.strip():
                continue

            parts = line.split("\t")

            # Skip MSG lines
            if len(parts) > 1 and parts[1] == "MSG":
                continue

            data_lines.append(parts)

    if header is None:
        raise ValueError("Could not find header line starting with 'Time'.")

    # Normalize number of columns to match header
    n_cols = len(header)
    normalized = []
    for parts in data_lines:
        if len(parts) < n_cols:
            parts = parts + [""] * (n_cols - len(parts))
        elif len(parts) > n_cols:
            parts = parts[:n_cols]
        normalized.append(parts)

    df = pd.DataFrame(normalized, columns=header)

    # Convert everything except 'Type' to numeric where possible
    for col in df.columns:
        if col == "Type":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, screen_w, screen_h


def add_gaze_columns(df, validity_threshold=1):
    """
    Add binocular gaze columns (GazeX, GazeY) computed from
    L/R POR X/Y and their validity.

    - If both eyes have validity <= threshold: use their average.
    - Else if only left is valid: use left.
    - Else if only right is valid: use right.
    - Else: NaN.
    """
    # Column names from your header
    lx = df["L POR X [px]"]
    ly = df["L POR Y [px]"]
    rx = df["R POR X [px]"]
    ry = df["R POR Y [px]"]
    lv = df["L Validity"]
    rv = df["R Validity"]

    l_good = (lv <= validity_threshold) & lx.notna() & ly.notna()
    r_good = (rv <= validity_threshold) & rx.notna() & ry.notna()
    both_good = l_good & r_good

    gaze_x = np.where(
        both_good,
        (lx + rx) / 2.0,
        np.where(l_good, lx, np.where(r_good, rx, np.nan)),
    )
    gaze_y = np.where(
        both_good,
        (ly + ry) / 2.0,
        np.where(l_good, ly, np.where(r_good, ry, np.nan)),
    )

    df = df.copy()
    df["GazeX"] = gaze_x
    df["GazeY"] = gaze_y
    return df


def compute_canvas_limits(df, screen_w, screen_h, p_low=0.025, p_high=0.975):
    """
    Compute axis limits that contain approximately p_high - p_low
    of all gaze points, and ensure that the entire screen rectangle
    [0, screen_w] x [0, screen_h] is visible within those limits.
    """
    finite = df[["GazeX", "GazeY"]].dropna()
    if finite.empty:
        raise ValueError("No valid gaze samples (GazeX/GazeY) found.")

    x_low, x_high = finite["GazeX"].quantile([p_low, p_high])
    y_low, y_high = finite["GazeY"].quantile([p_low, p_high])

    # Add 5% margin
    margin_x = 0.05 * (x_high - x_low)
    margin_y = 0.05 * (y_high - y_low)

    xmin = min(x_low - margin_x, 0.0)
    xmax = max(x_high + margin_x, float(screen_w) if screen_w is not None else x_high + margin_x)
    ymin = min(y_low - margin_y, 0.0)
    ymax = max(y_high + margin_y, float(screen_h) if screen_h is not None else y_high + margin_y)

    return xmin, xmax, ymin, ymax


def create_gaze_viewer(df, screen_w, screen_h):
    if screen_w is None or screen_h is None:
        screen_w = float(df["GazeX"].max() - df["GazeX"].min())
        screen_h = float(df["GazeY"].max() - df["GazeY"].min())

    xmin, xmax, ymin, ymax = compute_canvas_limits(df, screen_w, screen_h)

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.subplots_adjust(bottom=0.25)

    screen_rect = Rectangle(
        (0, 0),
        screen_w,
        screen_h,
        fill=False,
        linewidth=1.5,
    )
    ax.add_patch(screen_rect)

    (gaze_point,) = ax.plot([], [], marker="o", markersize=8)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)  # invert y
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X [px]")
    ax.set_ylabel("Y [px]")
    ax.set_title("Gaze position over time")

    time_text = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
    slider = Slider(
        ax=ax_slider,
        label="Sample",
        valmin=0,
        valmax=len(df) - 1,
        valinit=0,
        valstep=1,
    )

    def update(_):
        i = int(slider.val)
        row = df.iloc[i]
        gx, gy = row["GazeX"], row["GazeY"]

        if math.isnan(gx) or math.isnan(gy):
            # no valid gaze
            gaze_point.set_data([], [])
        else:
            # set position (must be sequences)
            gaze_point.set_data([gx], [gy])

            # check if gaze is within the screen rectangle
            if 0 <= gx <= screen_w and 0 <= gy <= screen_h:
                # on-screen: normal color (e.g. blue)
                gaze_point.set_color("blue")
            else:
                # off-screen: red
                gaze_point.set_color("red")

        time_text.set_text(f"t = {row['Time']}  (sample {i})")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)

    return fig, ax, slider



if __name__ == "__main__":
    # 1) Change this to your actual file path
    path_to_file = "sub-NDARAB678VYW_task-symbolSearch_et.txt"

    # 2) Load and parse the file
    df_raw, screen_w_px, screen_h_px = load_idf(path_to_file)

    # 3) Add binocular gaze columns
    df = add_gaze_columns(df_raw, validity_threshold=1)

    print(df)

    # 4) Create viewer and show
    fig, ax, slider = create_gaze_viewer(df, screen_w_px, screen_h_px)
    plt.show()
