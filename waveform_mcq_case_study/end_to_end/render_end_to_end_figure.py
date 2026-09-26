#!/usr/bin/env python3
"""Compact paper figure: waveform edit -> MCQ -> Qwen3.8 CoT."""

from __future__ import annotations

import json
import hashlib
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image


ROOT = Path(
    "<output_root>/train/ecg/waveform_perturb/"
    "paper_examples/two_case_perturbation_20260924"
)
OUT = ROOT / "end_to_end"

CASES = [
    {
        "folder": "case_01_pr_prolongation",
        "id": "paper_cf_439_pr_prolongation_morphology_preserving",
        "label": "Case 1",
        "title": "Morphology-preserving PR prolongation",
        "metric": "PR 130 → 298 ms  ·  HR 74.6 bpm preserved",
        "edit": "P and QRS–T copied exactly; PQ extended and TP shortened within each RR cycle.",
        "label_flip": "Original A  →  Counterfactual C",
        "gold": "C",
        "options": [
            ("A", "HR 74.6 · PR 130 · QRS 96 · QTc 393.7 → no 1° AV block"),
            ("B", "HR 48.0 · PR 298 · QRS 96 · QTc 393.7 → bradycardia + 1° AV block"),
            ("C", "HR 74.6 · PR 298 · QRS 96 · QTc 393.7 → first-degree AV block"),
            ("D", "HR 74.6 · PR 298 · QRS 130 · QTc 393.7 → 1° AV block + IVCD"),
        ],
        "cot_summary": "Qwen3.8 re-measured normal rate, prolonged PR and narrow QRS; excluded A/B/D.",
    },
    {
        "folder": "case_02_rate_slowing",
        "id": "paper_cf_774_rr_extension_morphology_preserving",
        "label": "Case 2",
        "title": "Morphology-preserving sinus-rate slowing",
        "metric": "HR 72.3 → 47.9 bpm  ·  PR/QRS preserved",
        "edit": "P–QRS–T copied exactly; only the post-T/pre-P isoelectric segment was extended.",
        "label_flip": "Original A  →  Counterfactual B",
        "gold": "B",
        "options": [
            ("A", "HR 72.3 · PR 150 · QRS 48 · QTc 348.5 → normal rate"),
            ("B", "HR 47.9 · PR 150 · QRS 48 · QTc 348.5 → sinus bradycardia"),
            ("C", "HR 47.9 · PR 220 · QRS 48 · QTc 348.5 → bradycardia + 1° AV block"),
            ("D", "HR 47.9 · PR 150 · QRS 48 · QTc 490 → bradycardia + long QT"),
        ],
        "cot_summary": "Qwen3.8 measured RR ≈1.25 s, normal PR/QRS/QTc and selected sinus bradycardia.",
    },
]

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


def load_cot_outputs() -> dict[str, dict]:
    outputs = {}
    with (OUT / "qwen38_cot_raw.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            outputs[row["id"]] = row
    return outputs


def thumbnail(path: Path, max_size=(3000, 1800)) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def rounded_box(ax, y: float, height: float, face: str, edge: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0.0, y),
            0.99,
            height,
            transform=ax.transAxes,
            boxstyle="round,pad=0.006,rounding_size=0.016",
            linewidth=1.0,
            facecolor=face,
            edgecolor=edge,
            clip_on=False,
        )
    )


def draw_case_text(ax, case: dict, cot: dict) -> None:
    ax.set_axis_off()
    ax.text(0, 0.985, f"{case['label']}  ·  {case['title']}", transform=ax.transAxes,
            va="top", fontsize=10.8, fontweight="bold", color=COLORS["blue"])
    ax.text(0, 0.917, case["metric"], transform=ax.transAxes, va="top",
            fontsize=8.5, fontweight="semibold", color=COLORS["ink"])
    ax.text(0, 0.862, case["edit"], transform=ax.transAxes, va="top",
            fontsize=7.1, linespacing=1.18, color=COLORS["muted"])
    ax.text(0, 0.79, "Synthetic measurement MCQ", transform=ax.transAxes, va="top",
            fontsize=8.2, fontweight="bold", color=COLORS["ink"])

    top = 0.745
    height = 0.088
    gap = 0.010
    for index, (letter, option) in enumerate(case["options"]):
        y = top - index * (height + gap) - height
        is_gold = letter == case["gold"]
        face = COLORS["green_bg"] if is_gold else COLORS["gray_bg"]
        edge = COLORS["green"] if is_gold else COLORS["line"]
        rounded_box(ax, y, height, face, edge)
        marker = "✓" if is_gold else " "
        text = textwrap.fill(option, width=64)
        ax.text(0.018, y + height / 2, f"{marker}  {letter}.  {text}", transform=ax.transAxes,
                va="center", fontsize=6.85, linespacing=1.1,
                color=COLORS["green"] if is_gold else COLORS["ink"],
                fontweight="semibold" if is_gold else "normal")

    y = top - 4 * (height + gap) - 0.012
    ax.text(0, y, case["label_flip"], transform=ax.transAxes, va="top",
            fontsize=8.2, fontweight="bold", color=COLORS["blue"])
    ax.text(0, y - 0.052, case["cot_summary"], transform=ax.transAxes, va="top",
            fontsize=7.1, linespacing=1.18, color=COLORS["ink"])
    chars = len(cot["assistant_target"])
    ax.text(0, y - 0.116,
            f"Qwen3.8-max thinking: PASS  ·  boxed {cot['pred_letter']}  ·  {chars:,} assistant characters",
            transform=ax.transAxes, va="top", fontsize=6.9, color=COLORS["green"],
            bbox=dict(boxstyle="round,pad=0.24", facecolor=COLORS["green_bg"], edgecolor="none"))
    ax.text(0, 0.012,
            "signal edit → re-measure → MCQ → multimodal CoT → strict boxed-answer gate",
            transform=ax.transAxes, va="bottom", fontsize=6.6, color=COLORS["muted"])


def main() -> None:
    outputs = load_cot_outputs()
    figure = plt.figure(figsize=(13.4, 8.3), facecolor="white")
    grid = figure.add_gridspec(2, 2, width_ratios=(1.62, 1.0), hspace=0.07, wspace=0.035)
    figure.subplots_adjust(left=0.018, right=0.985, top=0.925, bottom=0.02)
    figure.suptitle("From waveform perturbation to a trainable multimodal CoT example",
                    x=0.02, y=0.988, ha="left", va="top", fontsize=15.0,
                    fontweight="bold", color=COLORS["ink"])
    figure.text(0.98, 0.985, "Two closed-loop counterfactuals", ha="right", va="top",
                fontsize=8.3, color=COLORS["muted"])

    for index, case in enumerate(CASES):
        cot = outputs[case["id"]]
        if not cot.get("match"):
            raise RuntimeError(f"CoT does not match gold: {case['id']}")
        image_ax = figure.add_subplot(grid[index, 0])
        image_ax.imshow(thumbnail(ROOT / case["folder"] / "comparison.png"))
        image_ax.set_axis_off()
        text_ax = figure.add_subplot(grid[index, 1])
        draw_case_text(text_ax, case, cot)
        if index == 0:
            figure.add_artist(
                plt.Line2D([0.018, 0.985], [0.503, 0.503], transform=figure.transFigure,
                           color=COLORS["line"], linewidth=1.0)
            )

    figure.text(0.02, 0.004,
                "Automated, closed-loop labels; not clinician-adjudicated. ECG panels are rendered from the saved 500 Hz waveforms.",
                ha="left", va="bottom", fontsize=6.5, color=COLORS["muted"])
    figure.savefig(OUT / "paper_end_to_end_waveform_mcq_cot.png", dpi=300,
                   bbox_inches="tight", pad_inches=0.05)
    figure.savefig(OUT / "paper_end_to_end_waveform_mcq_cot.pdf",
                   bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)

    validation = {
        "samples": 2,
        "model": "qwen3.8-max",
        "thinking": True,
        "all_predictions_match_gold": all(outputs[case["id"]].get("match") for case in CASES),
        "all_reasoning_nonempty": all(outputs[case["id"]].get("reasoning_nonempty") for case in CASES),
        "results": [
            {
                "id": case["id"],
                "gold": case["gold"],
                "prediction": outputs[case["id"]].get("pred_letter"),
                "assistant_characters": len(outputs[case["id"]].get("assistant_target") or ""),
            }
            for case in CASES
        ],
    }
    (OUT / "PIPELINE_VALIDATION.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json" or "__pycache__" in path.parts:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": digest})
    (OUT / "MANIFEST.json").write_text(
        json.dumps({"files": manifest}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
