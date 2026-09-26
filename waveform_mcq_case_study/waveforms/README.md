# Included waveform representations

- `source_wfdb/*.hea + *.dat`: original PTB-XL 500 Hz record, copied byte-for-byte.
- `signal_original.npz`: normalized 10-second 12-lead source array used by the synthesis code.
- `signal_perturbed.npz`: saved counterfactual array used for the MCQ image.
- `signal_*.csv.gz`: portable, compressed tabular export with time in seconds and all 12 leads in mV.

The NPZ form is recommended for exact reconstruction because it preserves floating-point values and metadata without decimal text rounding.
