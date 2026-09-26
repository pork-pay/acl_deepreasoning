# Waveform → MCQ → CoT case-study review bundle

This folder is self-contained for review.

## Start here

1. `end_to_end/paper_end_to_end_waveform_mcq_cot.pdf` — recommended paper figure.
2. `end_to_end/paper_end_to_end_waveform_mcq_cot.png` — PNG preview.
3. `end_to_end/END_TO_END_WAVEFORM_MCQ_COT_PIPELINE.md` — detailed construction process, prompts, API payload, gates, and commands.
4. `end_to_end/waveform_mcq_cot_sft_2.jsonl` — two final trainable SFT examples.
5. `end_to_end/qwen38_cot_raw.jsonl` — complete Qwen3.8 reasoning and final outputs.
6. `EDITING_GUIDE.md` — local waveform regeneration and figure-editing instructions.

## Case contents

- `case_01_pr_prolongation/`: original ECG, PR-prolonged ECG, comparison figure, signals, original/synthetic MCQ, and audit.
- `case_02_rate_slowing/`: original ECG, rate-slowed ECG, comparison figure, signals, original/synthetic MCQ, and audit.
- `paper_panel_2cases.{pdf,png}`: waveform-only original-versus-perturbed figure.
- `questions_all.jsonl`: two original and two synthetic MCQs.

## Validation status

- Both waveform edits pass the automated target and protected-metric gates.
- Both Qwen3.8 thinking outputs match their strict boxed gold labels.
- Source traces come from PTB-XL training folds.
- Labels are algorithmically constructed and closed-loop verified, but are not clinician-adjudicated.

Use `BUNDLE_SHA256SUMS.txt` to verify copied files.

## Editable assets

- `waveforms/`: original WFDB records plus original/perturbed NPZ and compressed CSV.
- `methods/waveform_synthesis.py`: standalone PR-prolongation and rate-slowing operators.
- `methods/render_editable.py`: relative-path, parameterized ECG renderer.
- `editable_outputs/`: exact waveform reproductions and example custom renders.
