#!/usr/bin/env python3
"""Render two historical five-option waveform MCQ+CoT examples."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parent
COLORS = {
    "ink": "#17253A",
    "muted": "#627083",
    "blue": "#2457A6",
    "blue_bg": "#EAF1FB",
    "green": "#137A3E",
    "green_bg": "#E7F6EC",
    "gray_bg": "#F5F7FA",
    "line": "#D7DEE7",
}


def read_rows():
    rows = []
    with (ROOT / "selected_complex_cases_2.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    order = {"wfcmcq_08_281": 0, "wfcmcq_13_253": 1}
    return sorted(rows, key=lambda row: order[row["id"]])


def image_for(row):
    if row["source_id"] == "wf_08_00439":
        return ROOT / "images/first_degree_av_block_ecg439.png"
    if row["source_id"] == "wf_13_00434":
        return ROOT / "images/right_bundle_branch_block_ecg434.png"
    raise KeyError(row["source_id"])


def crop_grid(path):
    image = Image.open(path).convert("RGB")
    rgb = np.asarray(image).astype(np.int16)
    red = (rgb[:, :, 0] > 185) & (rgb[:, :, 0] > rgb[:, :, 1] + 12) & (rgb[:, :, 0] > rgb[:, :, 2] + 12)
    ys, xs = np.where(red)
    if len(xs):
        pad = 8
        image = image.crop((max(0, xs.min()-pad), max(0, ys.min()-pad), min(image.width, xs.max()+pad+1), min(image.height, ys.max()+pad+1)))
    image.thumbnail((2500, 1800), Image.Resampling.LANCZOS)
    return image


def parse_question(text):
    lines = [line.strip() for line in text.replace("<image>", "").splitlines() if line.strip()]
    stem, options = [], []
    for line in lines:
        match = re.match(r"^([A-E])\.\s*(.+)$", line)
        if match:
            options.append((match.group(1), match.group(2)))
        elif not options and not line.lower().startswith("please reason"):
            stem.append(line.replace("Question: ", "", 1))
    return " ".join(stem), options


def box(ax, y, height, face, edge):
    ax.add_patch(FancyBboxPatch((0, y), 0.99, height, transform=ax.transAxes,
                                boxstyle="round,pad=0.006,rounding_size=0.015",
                                facecolor=face, edgecolor=edge, linewidth=1.0, clip_on=False))


def draw_text(ax, row, index):
    ax.set_axis_off()
    stem, options = parse_question(row["messages"][0]["content"])
    titles = ["First-degree AV block: integrated interpretation", "RBBB: mechanism localization"]
    ax.text(0, 0.985, f"Case {index+1}  ·  Existing complex five-option MCQ", transform=ax.transAxes,
            va="top", fontsize=11.0, fontweight="bold", color=COLORS["blue"])
    ax.text(0, 0.925, titles[index], transform=ax.transAxes, va="top",
            fontsize=9.7, fontweight="semibold", color=COLORS["ink"])
    ax.text(0, 0.873, f"{row['source_id']}  ·  competency: {row['competency']}", transform=ax.transAxes,
            va="top", fontsize=7.4, color=COLORS["blue"],
            bbox=dict(boxstyle="round,pad=0.24", facecolor=COLORS["blue_bg"], edgecolor="none"))
    ax.text(0, 0.815, textwrap.fill(stem, 84), transform=ax.transAxes,
            va="top", fontsize=7.2, linespacing=1.13, color=COLORS["ink"])

    top, height, gap = 0.675, 0.096, 0.006
    for item, (letter, option) in enumerate(options):
        y = top - item * (height + gap) - height
        correct = letter == row["gold_letter"]
        box(ax, y, height, COLORS["green_bg"] if correct else COLORS["gray_bg"],
            COLORS["green"] if correct else COLORS["line"])
        prefix = "✓" if correct else " "
        ax.text(0.016, y + height/2, f"{prefix}  {letter}.  {textwrap.fill(option, 84)}",
                transform=ax.transAxes, va="center", fontsize=5.65, linespacing=1.02,
                color=COLORS["green"] if correct else COLORS["ink"],
                fontweight="semibold" if correct else "normal")

    length = len(row["messages"][1]["content"])
    y = top - 5 * (height + gap) - 0.012
    ax.text(0, y, f"Qwen3.8 thinking CoT: PASS  ·  boxed {row['gold_letter']}  ·  {length:,} characters",
            transform=ax.transAxes, va="top", fontsize=7.5, color=COLORS["green"],
            bbox=dict(boxstyle="round,pad=0.24", facecolor=COLORS["green_bg"], edgecolor="none"))
    ax.text(0, y-0.064,
            "Full CoT is retained in selected_complex_cases_2.jsonl; the figure shows the unmodified model-input question.",
            transform=ax.transAxes, va="top", fontsize=6.6, color=COLORS["muted"])
    ax.text(0, 0.01, "Historical closed-loop waveform source; automated label, not clinician-adjudicated",
            transform=ax.transAxes, va="bottom", fontsize=6.3, color=COLORS["muted"])


def main():
    rows = read_rows()
    figure = plt.figure(figsize=(13.5, 10.5), facecolor="white")
    grid = figure.add_gridspec(2, 2, width_ratios=(1.18, 1.0), hspace=0.055, wspace=0.035)
    figure.subplots_adjust(left=0.018, right=0.985, top=0.935, bottom=0.02)
    figure.suptitle("Existing complex MCQ + CoT examples for the same waveform mothers",
                    x=0.02, y=0.988, ha="left", va="top", fontsize=14.8,
                    fontweight="bold", color=COLORS["ink"])
    figure.text(0.98, 0.985, "Five-option · mechanism-heavy · Qwen3.8 thinking",
                ha="right", va="top", fontsize=8.0, color=COLORS["muted"])
    for index, row in enumerate(rows):
        image_ax = figure.add_subplot(grid[index, 0])
        image_ax.imshow(crop_grid(image_for(row)))
        image_ax.set_axis_off()
        image_ax.set_title("Exact historical model-input ECG", loc="left", pad=3,
                           fontsize=8.0, color=COLORS["muted"], fontweight="semibold")
        draw_text(figure.add_subplot(grid[index, 1]), row, index)
        if index == 0:
            figure.add_artist(plt.Line2D([0.018, 0.985], [0.502, 0.502], transform=figure.transFigure,
                                         color=COLORS["line"], linewidth=1.0))
    figure.savefig(ROOT / "paper_complex_mcq_cot_2cases.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    figure.savefig(ROOT / "paper_complex_mcq_cot_2cases.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)


if __name__ == "__main__":
    main()
