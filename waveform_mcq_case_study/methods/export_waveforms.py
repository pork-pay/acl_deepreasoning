#!/usr/bin/env python3
"""Export bundle NPZ waveforms to gzip-compressed CSV with time and lead names."""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        signal = np.asarray(data["signal"], dtype=np.float64)
        fs = int(data["fs"])
        names = [str(value) for value in data["lead_names"].tolist()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", *names])
        for index, values in enumerate(signal):
            writer.writerow([f"{index / fs:.6f}", *[f"{value:.8f}" for value in values]])
    print(f"rows={len(signal)} fs={fs} leads={len(names)} output={args.output}")


if __name__ == "__main__":
    main()
