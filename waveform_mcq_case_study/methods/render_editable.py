#!/usr/bin/env python3
"""Editable, relative-path ECG renderer for the review bundle.

The defaults reproduce a compact 4x3 layout plus a lead-II rhythm strip.
Change CLI flags or the STYLE dictionary below to adapt the paper figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
LAYOUT = (
    (("I", 0.0), ("AVR", 2.5), ("V1", 5.0), ("V4", 7.5)),
    (("II", 0.0), ("AVL", 2.5), ("V2", 5.0), ("V5", 7.5)),
    (("III", 0.0), ("AVF", 2.5), ("V3", 5.0), ("V6", 7.5)),
)

STYLE = {
    "paper_background": "#FFFFFF",
    "grid_background": "#FFFDFD",
    "minor_grid": "#F7DEDE",
    "major_grid": "#E9A7A7",
    "trace": "#111111",
    "title": "#17253A",
    "perturbed_title": "#9B2F2F",
}


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as data:
        signal = np.asarray(data["signal"], dtype=np.float64)
        fs = int(data["fs"])
        names = [str(value).upper() for value in data["lead_names"].tolist()]
    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError(f"expected N x 12 signal, got {signal.shape}")
    return signal, fs, names


def configure_axis(axis, seconds: float, y_limit: float, grid_alpha: float):
    axis.set_facecolor(STYLE["grid_background"])
    axis.set_xlim(0.0, seconds)
    axis.set_ylim(-y_limit, y_limit)
    axis.set_xticks(np.arange(0.0, seconds + 1e-8, 0.2))
    axis.set_xticks(np.arange(0.0, seconds + 1e-8, 0.04), minor=True)
    axis.set_yticks(np.arange(-y_limit, y_limit + 1e-8, 0.5))
    axis.set_yticks(np.arange(-y_limit, y_limit + 1e-8, 0.1), minor=True)
    axis.grid(which="minor", color=STYLE["minor_grid"], linewidth=0.22, alpha=grid_alpha)
    axis.grid(which="major", color=STYLE["major_grid"], linewidth=0.48, alpha=grid_alpha)
    axis.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in axis.spines.values():
        spine.set_color(STYLE["major_grid"])
        spine.set_linewidth(0.45)


def plot_segment(axis, values, fs, label, seconds, y_limit, line_width, grid_alpha):
    configure_axis(axis, seconds, y_limit, grid_alpha)
    time = np.arange(len(values)) / fs
    axis.plot(time, values, color=STYLE["trace"], linewidth=line_width,
              solid_joinstyle="round", solid_capstyle="round")
    axis.text(0.015, 0.87, label, transform=axis.transAxes,
              fontsize=8.5, fontweight="bold", color=STYLE["trace"])


def draw_ecg(figure, slot, signal, fs, names, title, title_color, args):
    by_name = {name: index for index, name in enumerate(names)}
    sub = slot.subgridspec(
        5 if args.rhythm else 4,
        4,
        height_ratios=([0.18, 1, 1, 1, 0.82] if args.rhythm else [0.18, 1, 1, 1]),
        hspace=args.hspace,
        wspace=args.wspace,
    )
    title_axis = figure.add_subplot(sub[0, :])
    title_axis.set_axis_off()
    title_axis.text(0.0, 0.5, title, ha="left", va="center",
                    fontsize=args.title_size, fontweight="bold", color=title_color)
    for row_index, row in enumerate(LAYOUT):
        for column_index, (lead, start_seconds) in enumerate(row):
            axis = figure.add_subplot(sub[row_index + 1, column_index])
            start = int(round(start_seconds * fs))
            stop = min(len(signal), start + int(round(args.segment_seconds * fs)))
            plot_segment(axis, signal[start:stop, by_name[lead]], fs, lead,
                         args.segment_seconds, args.y_limit, args.line_width, args.grid_alpha)
            axis.set_box_aspect(args.lead_box_aspect)
    if args.rhythm:
        rhythm_axis = figure.add_subplot(sub[4, :])
        stop = min(len(signal), int(round(args.rhythm_seconds * fs)))
        plot_segment(rhythm_axis, signal[:stop, by_name["II"]], fs, "II rhythm",
                     args.rhythm_seconds, args.y_limit, args.line_width, args.grid_alpha)
        rhythm_axis.set_box_aspect(args.rhythm_box_aspect)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--perturbed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="output path without extension")
    parser.add_argument("--title", default="ECG waveform counterfactual")
    parser.add_argument("--change-label", default="Targeted waveform edit")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fig-width", type=float, default=13.0)
    parser.add_argument("--fig-height", type=float, default=4.6)
    parser.add_argument("--segment-seconds", type=float, default=2.5)
    parser.add_argument("--rhythm-seconds", type=float, default=10.0)
    parser.add_argument("--y-limit", type=float, default=2.0)
    parser.add_argument("--line-width", type=float, default=0.78)
    parser.add_argument("--grid-alpha", type=float, default=1.0)
    parser.add_argument("--hspace", type=float, default=0.065)
    parser.add_argument("--wspace", type=float, default=0.032)
    parser.add_argument("--lead-box-aspect", type=float, default=0.64)
    parser.add_argument("--rhythm-box-aspect", type=float, default=0.145)
    parser.add_argument("--title-size", type=float, default=10.5)
    parser.add_argument("--no-rhythm", dest="rhythm", action="store_false")
    parser.set_defaults(rhythm=True)
    args = parser.parse_args()

    original, fs_original, names_original = load_npz(args.original)
    perturbed, fs_perturbed, names_perturbed = load_npz(args.perturbed)
    if fs_original != fs_perturbed or names_original != names_perturbed:
        raise ValueError("original and perturbed metadata differ")

    figure = plt.figure(figsize=(args.fig_width, args.fig_height), facecolor=STYLE["paper_background"])
    outer = figure.add_gridspec(1, 2, wspace=0.035)
    draw_ecg(figure, outer[0], original, fs_original, names_original,
             f"(a) Original · {args.title}", STYLE["title"], args)
    draw_ecg(figure, outer[1], perturbed, fs_original, names_original,
             f"(b) Perturbed · {args.change_label}", STYLE["perturbed_title"], args)
    figure.text(0.985, 0.99, "25 mm/s   10 mm/mV", ha="right", va="top",
                fontsize=8.0, color="#505A67")
    figure.subplots_adjust(left=0.018, right=0.992, top=0.965, bottom=0.02)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
