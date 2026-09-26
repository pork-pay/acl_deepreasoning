#!/usr/bin/env python3
"""Self-contained CLI for the two morphology-preserving waveform edits.

Examples (run from the bundle root):

  python methods/waveform_synthesis.py \
    --operator pr --input waveforms/case_01_pr_prolongation/signal_original.npz \
    --output editable_outputs/case01_pr300.npz --target 300

  python methods/waveform_synthesis.py \
    --operator rate --input waveforms/case_02_rate_slowing/signal_original.npz \
    --output editable_outputs/case02_hr48.npz --target 48
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE_ROOT / "code_snapshot"))
from load_fiducials import detect_rpeaks, delineate_lead  # noqa: E402


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
    return np.stack(
        [np.interp(new, old, values[:, column]) for column in range(values.shape[1])],
        axis=1,
    )


def prolong_pr(signal: np.ndarray, fs: int, names: list[str], target_pr_ms: float):
    """Copy P and QRS–T exactly; lengthen PQ and shorten TP within each RR."""
    peaks = detect_rpeaks(signal, fs, sig_name=names)
    lead_ii = [name.upper() for name in names].index("II")
    beats = delineate_lead(signal[:, lead_ii], peaks, fs)
    output = signal.copy()
    target_samples = int(round(target_pr_ms * fs / 1000.0))
    edits = []

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
            [
                p_wave,
                resample_rows(pq_gap, new_pq_length),
                qrs_t,
                resample_rows(tp_rest, new_tp_length),
            ],
            axis=0,
        )
        if len(rebuilt) != total_length:
            raise AssertionError((len(rebuilt), total_length))
        output[start:next_p] = rebuilt
        new_qrs_on = start + len(p_wave) + new_pq_length
        copied = output[new_qrs_on : new_qrs_on + len(qrs_t)]
        edits.append(
            {
                "beat_index": index,
                "source_bounds_samples": {
                    "p_on": start,
                    "qrs_on": qrs_on,
                    "t_off": t_off,
                    "next_p_on": next_p,
                },
                "output_bounds_samples": {
                    "p_on": start,
                    "qrs_on": new_qrs_on,
                    "t_off": new_qrs_on + len(qrs_t) - 1,
                },
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
        "method": "P and QRS-T copied exactly; PQ lengthened and TP shortened within each RR cycle",
        "edits": edits,
    }


def slow_rate(signal: np.ndarray, fs: int, names: list[str], target_hr_bpm: float):
    """Copy P–QRS–T exactly; extend only post-T/pre-P TP rest."""
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
    templates = []
    for index in range(first_index, len(beats) - 1):
        start = int(beats[index]["p_on"])
        t_off = int(beats[index]["t_off"])
        next_p = int(beats[index + 1]["p_on"])
        if not (0 <= start < t_off < next_p <= len(signal)):
            continue
        active = signal[start : t_off + 1]
        tp_rest = signal[t_off + 1 : next_p]
        new_tp_length = target_period - len(active)
        if new_tp_length < int(0.04 * fs):
            raise RuntimeError("target rate leaves too little TP rest")
        cycle = np.concatenate([active, resample_rows(tp_rest, new_tp_length)], axis=0)
        templates.append((index, cycle, len(active)))
    if not templates:
        raise RuntimeError("no usable beat cycles")

    pieces = [prefix]
    cursor = len(prefix)
    edits = []
    cycle_index = 0
    while cursor < len(signal):
        source_index, cycle, active_length = templates[cycle_index % len(templates)]
        visible = cycle[: len(signal) - cursor]
        pieces.append(visible)
        copied_length = min(active_length, len(visible))
        source_start = int(beats[source_index]["p_on"])
        source_active = signal[source_start : source_start + copied_length]
        edits.append(
            {
                "source_beat_index": source_index,
                "reused_cycle": cycle_index >= len(templates),
                "visible_samples": len(visible),
                "full_cycle_visible": len(visible) == len(cycle),
                "output_bounds_samples": {
                    "p_on": cursor,
                    "visible_end": cursor + len(visible) - 1,
                },
                "constructed_cycle_ms": round(target_period * 1000.0 / fs, 3),
                "visible_p_qrs_t_copy_max_abs_error_mV": float(
                    np.max(np.abs(visible[:copied_length] - source_active))
                ),
            }
        )
        cursor += len(visible)
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
        "method": "P-QRS-T copied exactly; only post-T/pre-P TP rest extended",
        "edits": edits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=("pr", "rate"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=float, required=True, help="PR ms or heart rate bpm")
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as data:
        signal = np.asarray(data["signal"], dtype=np.float64)
        fs = int(data["fs"])
        names = [str(value) for value in data["lead_names"].tolist()]
    if signal.shape != (5000, 12):
        raise ValueError(f"expected (5000, 12), got {signal.shape}")

    if args.operator == "pr":
        output, audit = prolong_pr(signal, fs, names, args.target)
    else:
        output, audit = slow_rate(signal, fs, names, args.target)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, signal=output, fs=fs, lead_names=np.asarray(names))
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "audit": str(audit_path), "shape": list(output.shape)}, indent=2))


if __name__ == "__main__":
    main()
