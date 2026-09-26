#!/usr/bin/env python3
"""Build two publication-ready, auditable ECG perturbation examples.

CPU-only; source records are read-only and every output stays below this
directory. Standalone ECG images contain no diagnostic label or answer.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wfdb
from matplotlib.gridspec import GridSpecFromSubplotSpec
from PIL import Image


OUT = Path(__file__).resolve().parent
TONGZHOU = Path("<output_root>").resolve()
PTB = Path(
    "<ptbxl_download>/"
    "physionet.org/files/ptb-xl/1.0.3"
)
QA_DIR = TONGZHOU / "train/ecg/waveform_perturb/data/cycles/loop_001/ptbxl_plus_qa"
SOURCE_CATALOG = (
    TONGZHOU
    / "train/data/ecg/runs/waveform_complex_mcq_sources_v1_20260920/"
    "waveform_sources_575.jsonl"
)
ANALYSIS_CODE = OUT / "code_snapshot"
sys.path.insert(0, str(ANALYSIS_CODE))
from analyze import analyze  # noqa: E402
from load_fiducials import delineate_lead, detect_rpeaks  # noqa: E402


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LAYOUT = [
    [("I", 0.0), ("aVR", 2.5), ("V1", 5.0), ("V4", 7.5)],
    [("II", 0.0), ("aVL", 2.5), ("V2", 5.0), ("V5", 7.5)],
    [("III", 0.0), ("aVF", 2.5), ("V3", 5.0), ("V6", 7.5)],
]
PNG_DPI = 600
Y_LIMIT_MV = 2.0

CASES = [
    {
        "folder": "case_01_pr_prolongation",
        "case_label": "Case 1",
        "question_id": "distill_q_439",
        "ecg_id": 439,
        "operator": "pr_prolongation_morphology_preserving",
        "target_pr_ms": 300.0,
        "synthetic_gold": "C",
        "expected_original_gold": "A",
        "catalog_target": "1stAVB",
        "paper_title": "PR-interval prolongation",
    },
    {
        "folder": "case_02_rate_slowing",
        "case_label": "Case 2",
        "question_id": "distill_q_774",
        "ecg_id": 774,
        "operator": "rr_extension_morphology_preserving",
        "target_hr_bpm": 48.0,
        "synthetic_gold": "B",
        "expected_original_gold": "A",
        "catalog_target": "SinusBrady",
        "paper_title": "Heart-rate reduction",
    },
]


def assert_write_scope(path: Path) -> None:
    resolved = path.resolve()
    if resolved != TONGZHOU and TONGZHOU not in resolved.parents:
        raise RuntimeError(f"refusing write outside tongzhou: {resolved}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    assert_write_scope(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    assert_write_scope(path)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def source_hashes(record: Path) -> dict[str, str]:
    return {"hea": sha256(record.with_suffix(".hea")), "dat": sha256(record.with_suffix(".dat"))}


def normalize_signal(signal: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    index = {str(name).upper(): i for i, name in enumerate(names)}
    missing = [lead for lead in LEADS if lead.upper() not in index]
    if missing:
        raise ValueError(f"missing standard leads: {missing}")
    ordered = np.asarray(signal, dtype=np.float64)[:, [index[lead.upper()] for lead in LEADS]]
    if ordered.ndim != 2 or ordered.shape[1] != 12:
        raise ValueError(f"unexpected signal shape: {ordered.shape}")
    if not np.all(np.isfinite(ordered)):
        raise ValueError("signal contains NaN or infinite values")
    return ordered, LEADS.copy()


def resample_rows(values: np.ndarray, length: int) -> np.ndarray:
    length = int(length)
    if length < 1:
        return np.empty((0, values.shape[1]), dtype=np.float64)
    if len(values) == 0:
        return np.zeros((length, values.shape[1]), dtype=np.float64)
    if len(values) == 1:
        return np.repeat(values, length, axis=0)
    if len(values) == length:
        return values.copy()
    old = np.linspace(0.0, 1.0, len(values))
    new = np.linspace(0.0, 1.0, length)
    return np.stack([np.interp(new, old, values[:, col]) for col in range(values.shape[1])], axis=1)


def measured(signal: np.ndarray, fs: int, names: list[str], fields: dict[str, Any]) -> tuple[dict, np.ndarray]:
    peaks = detect_rpeaks(signal, fs, sig_name=names)
    margin = int(round(0.3 * fs))
    peaks = np.asarray([p for p in peaks if margin <= int(p) <= len(signal) - margin], dtype=int)
    if len(peaks) < 3:
        raise RuntimeError(f"only {len(peaks)} usable R peaks")
    local_fields = dict(fields)
    local_fields["fs"] = fs
    local_fields["sig_name"] = names
    return analyze(signal, local_fields, fs, peaks), peaks


def prolong_pr_preserve_morphology(
    signal: np.ndarray, fs: int, names: list[str], target_pr_ms: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Move intact QRS-T later; resample only isoelectric PQ and TP segments."""
    peaks = detect_rpeaks(signal, fs, sig_name=names)
    lead_ii = [name.upper() for name in names].index("II")
    beats = delineate_lead(signal[:, lead_ii], peaks, fs)
    output = signal.copy()
    target_samples = int(round(target_pr_ms * fs / 1000.0))
    edits: list[dict[str, Any]] = []

    for index, beat in enumerate(beats[:-1]):
        start = int(beat["p_on"])
        p_off = int(beat["p_off"])
        qrs_on = int(beat["qrs_on"])
        t_off = int(beat["t_off"])
        next_p = int(beats[index + 1]["p_on"])
        if not (0 <= start < p_off < qrs_on < t_off < next_p <= len(signal)):
            continue
        p_wave = signal[start : p_off + 1]
        pq_gap = signal[p_off + 1 : qrs_on]
        qrs_t = signal[qrs_on : t_off + 1]
        tp_rest = signal[t_off + 1 : next_p]
        new_pq_length = target_samples - len(p_wave)
        total_length = next_p - start
        new_tp_length = total_length - len(p_wave) - new_pq_length - len(qrs_t)
        if new_pq_length < int(0.02 * fs) or new_tp_length < int(0.02 * fs):
            continue
        rebuilt = np.concatenate(
            [p_wave, resample_rows(pq_gap, new_pq_length), qrs_t, resample_rows(tp_rest, new_tp_length)], axis=0
        )
        if len(rebuilt) != total_length:
            raise AssertionError((len(rebuilt), total_length))
        output[start:next_p] = rebuilt
        new_qrs_on = start + len(p_wave) + new_pq_length
        copied = output[new_qrs_on : new_qrs_on + len(qrs_t)]
        edits.append(
            {
                "beat_index": index,
                "source_bounds_samples": {"p_on": start, "qrs_on": qrs_on, "t_off": t_off, "next_p_on": next_p},
                "output_bounds_samples": {"p_on": start, "qrs_on": new_qrs_on, "t_off": new_qrs_on + len(qrs_t) - 1},
                "constructed_pr_ms": round((new_qrs_on - start) * 1000.0 / fs, 3),
                "qrs_t_copy_max_abs_error_mV": float(np.max(np.abs(copied - qrs_t))),
            }
        )
    if len(edits) < 5:
        raise RuntimeError(f"too few safely edited beats: {len(edits)}")
    return output, {
        "name": "pr_prolongation_morphology_preserving",
        "target_pr_ms": target_pr_ms,
        "edited_beats": len(edits),
        "method": "P and QRS-T copied exactly; PQ gap lengthened and TP rest shortened within each RR cycle",
        "edits": edits,
    }


def slow_rate_preserve_morphology(
    signal: np.ndarray, fs: int, names: list[str], target_hr_bpm: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Lower rate by extending only TP rest while copying P-QRS-T exactly."""
    peaks = detect_rpeaks(signal, fs, sig_name=names)
    lead_ii = [name.upper() for name in names].index("II")
    beats = delineate_lead(signal[:, lead_ii], peaks, fs)
    target_period = int(round(60.0 * fs / target_hr_bpm))
    if len(beats) < 5:
        raise RuntimeError("too few beats for rate resynthesis")
    first_index = 1
    first_p = int(beats[first_index]["p_on"])
    prefix_start = max(0, first_p - int(round(0.25 * fs)))
    prefix = signal[prefix_start:first_p].copy()
    cycle_templates: list[tuple[int, np.ndarray, int]] = []
    for index in range(first_index, len(beats) - 1):
        beat = beats[index]
        start = int(beat["p_on"])
        t_off = int(beat["t_off"])
        next_p = int(beats[index + 1]["p_on"])
        if not (0 <= start < t_off < next_p <= len(signal)):
            continue
        active = signal[start : t_off + 1]
        tp_rest = signal[t_off + 1 : next_p]
        new_tp_length = target_period - len(active)
        if new_tp_length < int(0.04 * fs):
            raise RuntimeError("target rate leaves too little TP rest")
        cycle = np.concatenate([active, resample_rows(tp_rest, new_tp_length)], axis=0)
        if len(cycle) != target_period:
            raise AssertionError((len(cycle), target_period))
        cycle_templates.append((index, cycle, len(active)))
    if not cycle_templates:
        raise RuntimeError("no usable beat cycles")

    pieces = [prefix]
    output_cursor = len(prefix)
    edits: list[dict[str, Any]] = []
    cycle_index = 0
    while output_cursor < len(signal):
        source_beat_index, cycle, active_length = cycle_templates[cycle_index % len(cycle_templates)]
        visible = cycle[: len(signal) - output_cursor]
        pieces.append(visible)
        copied_length = min(active_length, len(visible))
        source_start = int(beats[source_beat_index]["p_on"])
        source_active = signal[source_start : source_start + copied_length]
        edits.append(
            {
                "source_beat_index": source_beat_index,
                "reused_cycle": cycle_index >= len(cycle_templates),
                "visible_samples": len(visible),
                "full_cycle_visible": len(visible) == len(cycle),
                "output_bounds_samples": {
                    "p_on": output_cursor,
                    "visible_end": output_cursor + len(visible) - 1,
                },
                "constructed_cycle_ms": round(target_period * 1000.0 / fs, 3),
                "visible_p_qrs_t_copy_max_abs_error_mV": float(
                    np.max(np.abs(visible[:copied_length] - source_active))
                ),
            }
        )
        output_cursor += len(visible)
        cycle_index += 1
    output = np.concatenate(pieces, axis=0).copy()
    if len(output) != len(signal):
        raise AssertionError((len(output), len(signal)))
    return output, {
        "name": "rr_extension_morphology_preserving",
        "target_hr_bpm": target_hr_bpm,
        "target_rr_ms": round(target_period * 1000.0 / fs, 3),
        "output_cycles_used": len(edits),
        "full_output_cycles": sum(int(edit["full_cycle_visible"]) for edit in edits),
        "method": "P-QRS-T copied exactly; only the post-T/pre-P isoelectric segment is extended",
        "edits": edits,
    }


def configure_axis(axis: plt.Axes, seconds: float, y_limit: float = Y_LIMIT_MV) -> None:
    axis.set_facecolor("#fffdfd")
    axis.set_xlim(0.0, seconds)
    axis.set_ylim(-y_limit, y_limit)
    axis.set_xticks(np.arange(0.0, seconds + 1e-8, 0.2))
    axis.set_xticks(np.arange(0.0, seconds + 1e-8, 0.04), minor=True)
    axis.set_yticks(np.arange(-y_limit, y_limit + 1e-8, 0.5))
    axis.set_yticks(np.arange(-y_limit, y_limit + 1e-8, 0.1), minor=True)
    axis.grid(which="minor", color="#f7dede", linewidth=0.22)
    axis.grid(which="major", color="#e9a7a7", linewidth=0.45)
    axis.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in axis.spines.values():
        spine.set_color("#e9a7a7")
        spine.set_linewidth(0.45)


def plot_segment(axis: plt.Axes, values: np.ndarray, fs: int, label: str, seconds: float) -> None:
    configure_axis(axis, seconds)
    time = np.arange(len(values)) / fs
    axis.plot(time, values, color="#111111", linewidth=0.82, solid_joinstyle="round", solid_capstyle="round")
    axis.text(0.018, 0.87, label, transform=axis.transAxes, fontsize=10.5, weight="bold", color="#111111")


def draw_full_ecg(figure: plt.Figure, signal: np.ndarray, fs: int, names: list[str], title: str) -> None:
    by_name = {name.upper(): index for index, name in enumerate(names)}
    grid = figure.add_gridspec(4, 4, hspace=0.07, wspace=0.035)
    for row_index, row in enumerate(LAYOUT):
        for column_index, (lead, start_seconds) in enumerate(row):
            axis = figure.add_subplot(grid[row_index, column_index])
            start = int(round(start_seconds * fs))
            stop = start + int(round(2.5 * fs))
            plot_segment(axis, signal[start:stop, by_name[lead.upper()]], fs, lead, 2.5)
            axis.set_box_aspect(0.64)
    rhythm = figure.add_subplot(grid[3, :])
    plot_segment(rhythm, signal[: int(10 * fs), by_name["II"]], fs, "II rhythm", 10.0)
    rhythm.set_box_aspect(0.16)
    figure.suptitle(title, x=0.02, y=0.987, ha="left", fontsize=14, weight="bold", color="#161616")
    figure.text(0.985, 0.982, "25 mm/s   10 mm/mV", ha="right", va="top", fontsize=10, color="#333333")
    figure.subplots_adjust(left=0.018, right=0.992, bottom=0.024, top=0.955)


def save_figure(figure: plt.Figure, png: Path, pdf: Path, title: str) -> None:
    assert_write_scope(png)
    assert_write_scope(pdf)
    metadata = {"Title": title, "Author": "Reproducible ECG perturbation pipeline", "Subject": "ECG waveform example"}
    figure.savefig(png, dpi=PNG_DPI, facecolor="white", metadata=metadata)
    figure.savefig(pdf, facecolor="white", metadata=metadata)
    plt.close(figure)
    with Image.open(png) as image:
        image.convert("RGB").save(png, format="PNG", dpi=(PNG_DPI, PNG_DPI), optimize=True)


def render_standalone(signal: np.ndarray, fs: int, names: list[str], title: str, png: Path, pdf: Path) -> None:
    figure = plt.figure(figsize=(16.0, 10.0), facecolor="white")
    draw_full_ecg(figure, signal, fs, names, title)
    save_figure(figure, png, pdf, title)


def draw_compact_card(
    figure: plt.Figure, slot: Any, signal: np.ndarray, fs: int, names: list[str], title: str, title_color: str = "#202020"
) -> None:
    by_name = {name.upper(): index for index, name in enumerate(names)}
    grid = GridSpecFromSubplotSpec(4, 4, subplot_spec=slot, height_ratios=[0.22, 1, 1, 1], hspace=0.065, wspace=0.035)
    title_axis = figure.add_subplot(grid[0, :])
    title_axis.axis("off")
    title_axis.text(0.0, 0.48, title, ha="left", va="center", fontsize=11.5, weight="bold", color=title_color)
    for row_index, row in enumerate(LAYOUT):
        for column_index, (lead, start_seconds) in enumerate(row):
            axis = figure.add_subplot(grid[row_index + 1, column_index])
            start = int(round(start_seconds * fs))
            stop = start + int(round(2.5 * fs))
            plot_segment(axis, signal[start:stop, by_name[lead.upper()]], fs, lead, 2.5)
            axis.set_box_aspect(0.64)


def metric_subset(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in ["HR_bpm", "PR_ms", "QRS_ms", "QT_ms", "QTc_ms"]}


def render_case_comparison(case: dict[str, Any], case_dir: Path) -> None:
    original = case["signal_original"]
    perturbed = case["signal_perturbed"]
    fs = case["fs"]
    names = case["lead_names"]
    base = case["metrics_original"]
    after = case["metrics_perturbed"]
    figure = plt.figure(figsize=(16.0, 9.2), facecolor="white")
    outer = figure.add_gridspec(2, 2, height_ratios=[4.9, 1.25], hspace=0.13, wspace=0.055)
    draw_compact_card(figure, outer[0, 0], original, fs, names, "(a) Original waveform")
    draw_compact_card(figure, outer[0, 1], perturbed, fs, names, "(b) Perturbed waveform", "#8b2f2f")
    lead_ii = [name.upper() for name in names].index("II")
    if case["operator"] == "pr_prolongation_morphology_preserving":
        # Show the beat whose constructed source PR is closest to the reported
        # record-level median, so the detail label agrees with the audit table.
        edit = min(
            case["operator_log"]["edits"],
            key=lambda item: abs(
                (item["source_bounds_samples"]["qrs_on"] - item["source_bounds_samples"]["p_on"])
                * 1000.0
                / fs
                - base["PR_ms"]
            ),
        )
        p_on = edit["source_bounds_samples"]["p_on"]
        old_qrs = edit["source_bounds_samples"]["qrs_on"]
        new_qrs = edit["output_bounds_samples"]["qrs_on"]
        left = max(0, p_on - int(0.12 * fs))
        right = min(len(original), new_qrs + int(0.52 * fs))
        items = [(original, old_qrs, "Original PR", "#333333"), (perturbed, new_qrs, "Perturbed PR", "#a43838")]
        for col, (sig, qrs, label, color) in enumerate(items):
            axis = figure.add_subplot(outer[1, col])
            time = (np.arange(left, right) - p_on) / fs
            axis.plot(time, sig[left:right, lead_ii], color="#111111", linewidth=1.15)
            axis.axvline(0.0, color="#2766a5", linewidth=1.0, linestyle="--")
            axis.axvline((qrs - p_on) / fs, color=color, linewidth=1.0, linestyle="--")
            axis.axvspan(0.0, (qrs - p_on) / fs, color=color, alpha=0.10)
            axis.text(0.01, 0.88, f"{label}: {(qrs-p_on)*1000/fs:.0f} ms", transform=axis.transAxes, fontsize=10, weight="bold")
            axis.set_xlim(time[0], time[-1])
            axis.set_ylim(-0.45, 1.75)
            axis.grid(color="#eadada", linewidth=0.35)
            axis.set_xlabel("Time from P onset (s)", fontsize=9)
            axis.set_ylabel("Lead II (mV)", fontsize=9)
    else:
        items = [(original, base, "Original rhythm", "#333333"), (perturbed, after, "Perturbed rhythm", "#a43838")]
        for col, (sig, metrics, label, color) in enumerate(items):
            axis = figure.add_subplot(outer[1, col])
            duration = 5.0
            n = int(duration * fs)
            axis.plot(np.arange(n) / fs, sig[:n, lead_ii], color="#111111", linewidth=1.0)
            peaks = detect_rpeaks(sig[:n], fs, sig_name=names)
            for peak in [int(p) for p in peaks if int(0.3 * fs) < p < n - int(0.3 * fs)]:
                axis.axvline(peak / fs, color=color, alpha=0.32, linewidth=0.7)
            axis.text(0.01, 0.88, f"{label}: HR {metrics['HR_bpm']:.1f} bpm", transform=axis.transAxes, fontsize=10, weight="bold")
            axis.set_xlim(0, duration)
            axis.set_ylim(-0.45, 1.75)
            axis.grid(color="#eadada", linewidth=0.35)
            axis.set_xlabel("Time (s)", fontsize=9)
            axis.set_ylabel("Lead II (mV)", fontsize=9)
    figure.suptitle(f"{case['case_label']} — {case['paper_title']}", x=0.02, y=0.994, ha="left", fontsize=15, weight="bold")
    figure.text(0.985, 0.988, "Waveform source: PTB-XL · train split", ha="right", va="top", fontsize=9, color="#555555")
    figure.subplots_adjust(left=0.035, right=0.99, bottom=0.065, top=0.955)
    save_figure(figure, case_dir / "comparison.png", case_dir / "comparison.pdf", f"{case['case_label']} comparison")


def render_final_panel(cases: list[dict[str, Any]]) -> None:
    figure = plt.figure(figsize=(16.0, 10.6), facecolor="white")
    outer = figure.add_gridspec(2, 2, hspace=0.09, wspace=0.055)
    for row, case in enumerate(cases):
        if case["operator"] == "pr_prolongation_morphology_preserving":
            delta = f"PR {case['metrics_original']['PR_ms']:.0f} → {case['metrics_perturbed']['PR_ms']:.0f} ms"
        else:
            delta = f"HR {case['metrics_original']['HR_bpm']:.1f} → {case['metrics_perturbed']['HR_bpm']:.1f} bpm"
        draw_compact_card(figure, outer[row, 0], case["signal_original"], case["fs"], case["lead_names"], f"{case['case_label']}a  Original")
        draw_compact_card(figure, outer[row, 1], case["signal_perturbed"], case["fs"], case["lead_names"], f"{case['case_label']}b  Perturbed · {delta}", "#8b2f2f")
    figure.suptitle("Morphology-preserving ECG counterfactuals", x=0.02, y=0.995, ha="left", fontsize=16, weight="bold")
    figure.text(0.985, 0.991, "25 mm/s   10 mm/mV", ha="right", va="top", fontsize=10)
    figure.subplots_adjust(left=0.02, right=0.992, bottom=0.02, top=0.962)
    save_figure(figure, OUT / "paper_panel_2cases.png", OUT / "paper_panel_2cases.pdf", "Two ECG perturbation cases")


def image_info(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        return {"width_px": image.width, "height_px": image.height, "mode": image.mode, "dpi": list(image.info.get("dpi", (None, None)))}


def question_markdown(title: str, image_name: str, question: str, gold: str, answer_text: str, status: str) -> str:
    clean = question.replace("<image>\n", "", 1)
    return f"""# {title}

![ECG]({image_name})

{clean}

## Answer key

- Gold: `{gold}`
- Answer: {answer_text}
- Label status: `{status}`
"""


def build_case(
    config: dict[str, Any],
    questions: dict[str, dict[str, Any]],
    answers: dict[str, dict[str, Any]],
    provenance: dict[str, dict[str, Any]],
    catalog: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    case_dir = OUT / config["folder"]
    assert_write_scope(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    question = questions[config["question_id"]]
    answer = answers[config["question_id"]]
    prov = provenance[config["question_id"]]
    if answer["gold"] != config["expected_original_gold"]:
        raise RuntimeError(f"unexpected source gold for {config['question_id']}")
    if int(prov["ecg_id"]) != config["ecg_id"] or prov["split"] != "train":
        raise RuntimeError(f"source provenance mismatch: {config['question_id']}")

    record = Path(prov["source_waveform"])
    hashes_before = source_hashes(record)
    signal, fields = wfdb.rdsamp(str(record))
    signal, names = normalize_signal(signal, list(fields["sig_name"]))
    fs = int(fields["fs"])
    required = int(round(10.0 * fs))
    if len(signal) < required:
        raise RuntimeError(f"source shorter than 10 s: {len(signal)/fs:.3f}")
    signal = signal[:required].copy()

    if config["operator"] == "pr_prolongation_morphology_preserving":
        perturbed, operator_log = prolong_pr_preserve_morphology(signal, fs, names, config["target_pr_ms"])
    elif config["operator"] == "rr_extension_morphology_preserving":
        perturbed, operator_log = slow_rate_preserve_morphology(signal, fs, names, config["target_hr_bpm"])
    else:
        raise ValueError(config["operator"])

    metrics_original, peaks_original = measured(signal, fs, names, fields)
    metrics_perturbed, peaks_perturbed = measured(perturbed, fs, names, fields)
    hashes_after = source_hashes(record)
    if hashes_before != hashes_after:
        raise RuntimeError(f"source changed while reading: {record}")

    base_normal = 60.0 <= metrics_original["HR_bpm"] <= 100.0 and metrics_original["QRS_ms"] < 120.0 and 120.0 <= metrics_original["PR_ms"] <= 200.0
    if config["operator"] == "pr_prolongation_morphology_preserving":
        target_pass = metrics_perturbed["PR_ms"] > 200.0
        protected_pass = (
            abs(metrics_perturbed["HR_bpm"] - metrics_original["HR_bpm"]) <= 3.0
            and abs(metrics_perturbed["QRS_ms"] - metrics_original["QRS_ms"]) <= 15.0
            and abs(metrics_perturbed["QT_ms"] - metrics_original["QT_ms"]) <= 20.0
        )
    else:
        target_pass = metrics_perturbed["HR_bpm"] < 60.0
        protected_pass = (
            abs(metrics_perturbed["PR_ms"] - metrics_original["PR_ms"]) <= 10.0
            and abs(metrics_perturbed["QRS_ms"] - metrics_original["QRS_ms"]) <= 10.0
            and abs(metrics_perturbed["QT_ms"] - metrics_original["QT_ms"]) <= 20.0
        )
    engineering = {
        "shape_preserved": list(perturbed.shape) == list(signal.shape),
        "finite": bool(np.all(np.isfinite(perturbed))),
        "max_abs_mV": round(float(np.max(np.abs(perturbed))), 6),
        "clipped_fraction_at_3_5mV": round(float(np.mean(np.abs(perturbed) > 3.5)), 9),
        "mean_abs_change_mV": round(float(np.mean(np.abs(perturbed - signal))), 9),
        "max_abs_change_mV": round(float(np.max(np.abs(perturbed - signal))), 6),
    }
    all_gates = bool(base_normal and target_pass and protected_pass and all([engineering["shape_preserved"], engineering["finite"], engineering["clipped_fraction_at_3_5mV"] == 0.0]))
    if not all_gates:
        raise RuntimeError(
            f"validation failed for {config['folder']}: base={base_normal}, target={target_pass}, protected={protected_pass}, engineering={engineering}"
        )

    assert_write_scope(case_dir / "signal_original.npz")
    np.savez_compressed(case_dir / "signal_original.npz", signal=signal, fs=fs, lead_names=np.asarray(names))
    np.savez_compressed(case_dir / "signal_perturbed.npz", signal=perturbed, fs=fs, lead_names=np.asarray(names))
    source_image = Path(prov["source_image"])
    copied_source_image = case_dir / "source_original.png"
    shutil.copy2(source_image, copied_source_image)
    render_standalone(signal, fs, names, f"{config['case_label']} · Original waveform", case_dir / "original.png", case_dir / "original.pdf")
    render_standalone(perturbed, fs, names, f"{config['case_label']} · Perturbed waveform", case_dir / "perturbed.png", case_dir / "perturbed.pdf")

    final_instruction = "Please reason step by step, and put your final answer within \\boxed{}."
    if config["operator"] == "pr_prolongation_morphology_preserving":
        hr = metrics_perturbed["HR_bpm"]
        pr = metrics_perturbed["PR_ms"]
        qrs = metrics_perturbed["QRS_ms"]
        qtc = metrics_perturbed["QTc_ms"]
        synthetic_question = (
            "<image>\n"
            "Question: 按本数据构建流程的自动闭环测量口径，看这张12导联ECG，下列哪项参数分析与结论是正确的?\n"
            "Options:\n"
            f"A: HR {hr:.1f} bpm，PR {metrics_original['PR_ms']:.1f} ms，QRS {qrs:.1f} ms，QTc {qtc:.1f} ms → 结论: 未见一度房室传导阻滞\n"
            f"B: HR 48.0 bpm，PR {pr:.1f} ms，QRS {qrs:.1f} ms，QTc {qtc:.1f} ms → 结论: 窦性心动过缓合并一度房室传导阻滞\n"
            f"C: HR {hr:.1f} bpm，PR {pr:.1f} ms，QRS {qrs:.1f} ms，QTc {qtc:.1f} ms → 结论: 一度房室传导阻滞\n"
            f"D: HR {hr:.1f} bpm，PR {pr:.1f} ms，QRS 130.0 ms，QTc {qtc:.1f} ms → 结论: 一度房室传导阻滞合并室内传导延迟\n"
            f"{final_instruction}"
        )
        synthetic_answer = (
            f"HR {hr:.1f} bpm，PR {pr:.1f} ms，QRS {qrs:.1f} ms，QTc {qtc:.1f} ms "
            "→ 结论: 一度房室传导阻滞"
        )
        synthesis_note = "沿用原题四选一结构，依据扰动后闭环复测值重写选项；金标由 A 翻转为 C。"
    else:
        hr = metrics_perturbed["HR_bpm"]
        pr = metrics_perturbed["PR_ms"]
        qrs = metrics_perturbed["QRS_ms"]
        qtc = metrics_perturbed["QTc_ms"]
        synthetic_question = (
            "<image>\n"
            "Question: 按本数据构建流程的自动闭环测量口径，看这张12导联ECG，下列哪项参数分析与结论是正确的?\n"
            "Options:\n"
            f"A: HR {metrics_original['HR_bpm']:.1f} bpm，PR {pr:.1f} ms，QRS {qrs:.1f} ms，QTc {qtc:.1f} ms → 结论: 正常心率\n"
            f"B: HR {hr:.1f} bpm，PR {pr:.1f} ms，QRS {qrs:.1f} ms，QTc {qtc:.1f} ms → 结论: 窦性心动过缓\n"
            f"C: HR {hr:.1f} bpm，PR 220.0 ms，QRS {qrs:.1f} ms，QTc {qtc:.1f} ms → 结论: 窦性心动过缓合并一度房室传导阻滞\n"
            f"D: HR {hr:.1f} bpm，PR {pr:.1f} ms，QRS {qrs:.1f} ms，QTc 490.0 ms → 结论: 窦性心动过缓合并 QT 间期延长\n"
            f"{final_instruction}"
        )
        synthetic_answer = (
            f"HR {hr:.1f} bpm，PR {pr:.1f} ms，QRS {qrs:.1f} ms，QTc {qtc:.1f} ms "
            "→ 结论: 窦性心动过缓"
        )
        synthesis_note = "沿用原题四选一结构，依据扰动后闭环复测值重写选项；金标由 A 翻转为 B。"

    original_payload = {
        "id": config["question_id"],
        "question": question["question"],
        "images": ["source_original.png"],
        "gold": answer["gold"],
        "answer_text": answer["answer_text"],
        "answer_diagnosis": answer["answer_diagnosis"],
        "label_status": "dataset_provided_annotation",
        "provenance": prov,
    }
    synthetic_payload = {
        "id": f"paper_cf_{config['ecg_id']}_{config['operator']}",
        "parent_question_id": config["question_id"],
        "question": synthetic_question,
        "images": ["perturbed.png"],
        "gold": config["synthetic_gold"],
        "answer_text": synthetic_answer,
        "synthesis_note": synthesis_note,
        "label_status": "algorithmically_constructed_and_closed_loop_verified_not_clinician_adjudicated",
    }
    write_json(case_dir / "original_question.json", original_payload)
    write_json(case_dir / "synthesized_question.json", synthetic_payload)
    write_text(case_dir / "original_question.md", question_markdown("原始题", "source_original.png", question["question"], answer["gold"], answer["answer_text"], "dataset_provided_annotation"))
    write_text(case_dir / "synthesized_question.md", question_markdown("合成题", "perturbed.png", synthetic_question, config["synthetic_gold"], synthetic_answer, "algorithmically_constructed_and_closed_loop_verified_not_clinician_adjudicated"))

    source_catalog_row = catalog[(str(config["ecg_id"]), config["catalog_target"])]
    base_catalog_row = catalog[(str(config["ecg_id"]), "Normal")]
    audit = {
        "case_id": config["folder"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "PTB-XL 1.0.3 / PTB-XL-Plus-derived QA",
            "split": prov["split"],
            "strat_fold": prov["strat_fold"],
            "ecg_id": config["ecg_id"],
            "patient_id": prov["patient_id"],
            "waveform_record": str(record),
            "waveform_sha256": hashes_before,
            "source_image": str(source_image),
            "source_image_sha256": sha256(source_image),
            "question_id": config["question_id"],
        },
        "selection_evidence": {"normal_catalog_entry": base_catalog_row, "prior_closed_loop_target_entry": source_catalog_row},
        "operator": operator_log,
        "measurements": {
            "method": "existing deterministic ECG analyzer, rerun after perturbation with R-peak redetection",
            "analyzer": str(ANALYSIS_CODE / "analyze.py"),
            "analyzer_sha256": sha256(ANALYSIS_CODE / "analyze.py"),
            "fiducial_detector": str(ANALYSIS_CODE / "load_fiducials.py"),
            "fiducial_detector_sha256": sha256(ANALYSIS_CODE / "load_fiducials.py"),
            "original": metric_subset(metrics_original),
            "perturbed": metric_subset(metrics_perturbed),
            "detected_r_peaks_original": [int(value) for value in peaks_original],
            "detected_r_peaks_perturbed": [int(value) for value in peaks_perturbed],
        },
        "validation_gates": {
            "source_is_train_split": prov["split"] == "train",
            "source_hash_stable_during_read": hashes_before == hashes_after,
            "original_normal_parameter_gate": bool(base_normal),
            "target_threshold_pass": bool(target_pass),
            "protected_parameter_tolerance_pass": bool(protected_pass),
            "engineering": engineering,
            "all_automated_gates_passed": all_gates,
        },
        "rendering": {
            "layout": "standard 4x3 2.5-second segments plus 10-second lead-II rhythm strip",
            "calibration": "25 mm/s, 10 mm/mV",
            "y_range_mV": [-Y_LIMIT_MV, Y_LIMIT_MV],
            "png_dpi": PNG_DPI,
            "standalone_images_do_not_contain_diagnosis_or_answer": True,
        },
        "clinical_status": (
            "Illustrative algorithmic counterfactual. The target parameter and engineering checks pass, "
            "but this is not a clinician-adjudicated diagnostic gold record."
        ),
    }
    write_json(case_dir / "audit.json", audit)
    case_runtime = dict(config)
    case_runtime.update(
        {
            "case_dir": case_dir,
            "signal_original": signal,
            "signal_perturbed": perturbed,
            "fs": fs,
            "lead_names": names,
            "metrics_original": metric_subset(metrics_original),
            "metrics_perturbed": metric_subset(metrics_perturbed),
            "operator_log": operator_log,
            "audit": audit,
            "original_payload": original_payload,
            "synthetic_payload": synthetic_payload,
        }
    )
    render_case_comparison(case_runtime, case_dir)
    return case_runtime


def build_readme(cases: list[dict[str, Any]]) -> None:
    rows = []
    for case in cases:
        before = case["metrics_original"]
        after = case["metrics_perturbed"]
        rows.append(
            f"| {case['case_label']} | ECG {case['ecg_id']} | {case['paper_title']} | "
            f"HR {before['HR_bpm']:.1f}→{after['HR_bpm']:.1f} bpm; "
            f"PR {before['PR_ms']:.0f}→{after['PR_ms']:.0f} ms; "
            f"QRS {before['QRS_ms']:.0f}→{after['QRS_ms']:.0f} ms | PASS |"
        )
    readme = f"""# 两个 ECG 人工扰动论文示例

本目录包含两个独立、可复现的“原始题 → 合成题”案例。所有新文件仅写入
`{OUT}`，外部源数据只读。

| 案例 | 源记录 | 定向扰动 | 闭环复测 | 状态 |
|---|---:|---|---|---|
{chr(10).join(rows)}

## 论文直接使用

- `paper_panel_2cases.pdf`：推荐，矢量双案例组合图。
- `paper_panel_2cases.png`：600 dpi 无损位图。
- 每个 `case_*` 内：`source_original.png` 是原题自带原图；`original.*` 是同一原始波形的新渲染；`perturbed.*` 是加扰后新渲染；`comparison.*` 是紧凑对照图。
- `original_question.*` 与 `synthesized_question.*` 分别保存原题和合成题；`audit.json` 保存逐项溯源、参数、哈希和门禁。
- `signal_original.npz` 与 `signal_perturbed.npz` 保存 500 Hz、12 导联的可复算波形。
- `ENVIRONMENT.json` 与 `MANIFEST.json` 保存软件版本和全部文件校验和。
- `code_snapshot/` 固化本次闭环复测所用的分析器与 fiducial 检测器。

## 口径

单图采用标准 4×3 的 2.5 秒导联布局，并附 10 秒 II 导联节律条；网格为
0.04/0.20 秒与 0.1/0.5 mV，标注 25 mm/s、10 mm/mV。模型输入用的
`original.png`/`perturbed.png` 不含诊断或答案，避免标签泄漏。

Case 1 只重采样 PQ 与 TP 等电段，P 波和 QRS–T 样本原样复制；Case 2 只延长
TP 等电段，P–QRS–T 样本原样复制。两例均重新检测 R 峰并复测参数，详见
各自 `audit.json`。

## 重要边界

原题答案属于数据集既有标注；合成题答案是“已知算子 + 自动闭环测量”得到的
反事实标签，不等同于心电专家复核。若论文文字要把合成波形作为临床诊断金标，
仍应增加人工心电审核。当前材料适合展示数据构造方法与参数级反事实。

## 复现

```bash
python {OUT / 'render_paper_examples.py'}
```

脚本为 CPU-only，不占用 GPU。`MANIFEST.json` 记录除自身外全部产物的 SHA-256。
"""
    write_text(OUT / "README.md", readme)
    caption = """# Figure caption / 图注

中文：两个形态保持的 ECG 反事实示例。上排（Case 1）通过延长 PQ 等电段构造 PR 间期延长，同时复制原始 P 波与 QRS–T 形态；下排（Case 2）通过延长 TP 等电段降低心率，同时复制原始 P–QRS–T 形态。左列为原始波形，右列为扰动后波形。所有波形来自训练划分，纸速 25 mm/s，增益 10 mm/mV。

English: Two morphology-preserving ECG counterfactuals. In Case 1, the PQ isoelectric segment is extended to prolong the PR interval while the original P wave and QRS–T morphology are copied unchanged. In Case 2, the TP isoelectric segment is extended to reduce heart rate while the original P–QRS–T morphology is copied unchanged. Original waveforms are shown on the left and perturbed waveforms on the right. All source records are from the training split; paper speed, 25 mm/s; gain, 10 mm/mV.
"""
    write_text(OUT / "caption_zh_en.md", caption)
    validation = [
        "# Validation summary", "",
        "| Case | Source split | Source hash stable | Target gate | Protected metrics | Finite | Clipping | Result |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for case in cases:
        gates = case["audit"]["validation_gates"]
        eng = gates["engineering"]
        validation.append(
            f"| {case['case_label']} | train | yes | pass | pass | yes | {eng['clipped_fraction_at_3_5mV']:.6f} | **PASS** |"
        )
    validation += ["", "自动门禁覆盖来源划分、源文件哈希稳定、波形有限值/形状/削波、目标阈值及非目标参数容差。", "医学诊断金标仍需心电专家人工复核。"]
    write_text(OUT / "VALIDATION.md", "\n".join(validation))


def build_manifest(cases: list[dict[str, Any]]) -> None:
    all_questions = []
    for case in cases:
        original = dict(case["original_payload"])
        original["images"] = [f"{case['folder']}/source_original.png"]
        synthetic = dict(case["synthetic_payload"])
        synthetic["images"] = [f"{case['folder']}/perturbed.png"]
        all_questions.extend([original, synthetic])
    assert_write_scope(OUT / "questions_all.jsonl")
    with (OUT / "questions_all.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_questions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    files = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        entry: dict[str, Any] = {"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        if path.suffix.lower() == ".png":
            entry["image"] = image_info(path)
        files.append(entry)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(OUT),
        "cases": len(cases),
        "source_policy": "PTB-XL records and source question images were read-only; every output is under tongzhou",
        "rendering": {"png_dpi": PNG_DPI, "pdf": "vector", "layout": "4x3 plus rhythm strip for standalone ECGs"},
        "validation": "all automated gates passed",
        "clinical_label_status": "algorithmic counterfactual; not clinician-adjudicated",
        "manifest_excludes_itself": True,
        "files": files,
    }
    write_json(OUT / "MANIFEST.json", manifest)


def build_environment() -> None:
    packages = {}
    for name in ["numpy", "matplotlib", "wfdb", "Pillow", "scipy"]:
        packages[name] = importlib.metadata.version(name)
    write_json(
        OUT / "ENVIRONMENT.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
            "gpu_used": False,
        },
    )


def main() -> None:
    assert_write_scope(OUT)
    questions = {row["id"]: row for row in read_jsonl(QA_DIR / "questions.jsonl")}
    answers = {row["id"]: row for row in read_jsonl(QA_DIR / "answers.jsonl")}
    provenance = {row["id"]: row for row in read_jsonl(QA_DIR / "provenance.jsonl")}
    catalog_rows = read_jsonl(SOURCE_CATALOG)
    catalog = {(str(row["mother_ecg_id"]), row["target"]): row for row in catalog_rows}
    cases = [build_case(config, questions, answers, provenance, catalog) for config in CASES]
    render_final_panel(cases)
    build_readme(cases)
    build_environment()
    build_manifest(cases)
    print(json.dumps({
        "output": str(OUT),
        "cases": len(cases),
        "all_gates_passed": all(case["audit"]["validation_gates"]["all_automated_gates_passed"] for case in cases),
        "panel_png": str(OUT / "paper_panel_2cases.png"),
        "panel_pdf": str(OUT / "paper_panel_2cases.pdf"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
