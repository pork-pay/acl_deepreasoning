# acl_deepreasoning

Test push from training environment via ssh-over-443.

## sample/
One example from the `medxpert-ecg` split (MedXpertQA-MM cardiovascular subset, ECG-vision-filtered):
- `MM-19.json` — question text + gold answer (id `medxpert-ecg-MM-19`, 5-choice A–E)
- `MM-19.jpeg` — the ECG image (embedded in the source parquet)

## hint_inject_samples/
`单图、良性平片` (chest X-ray + elbow radiograph) samples for the **truncate-correct-CoT @ ~30% → gpt-5.6 self-check hint → 35B continue** method. Two single-image samples, each **triple-verified** (teacher == 35B continuation == gpt-5.6 oracle). See [`hint_inject_samples/README.md`](hint_inject_samples/README.md).

## waveform_mcq_case_study/
Reproducible two-case **ECG waveform perturbation → closed-loop remeasurement → MCQ → CoT** study (PTB-XL source records, read-only):
- Case 1 — PR-interval prolongation (PR 130→298 ms); Case 2 — heart-rate reduction (HR 72.3→47.9 bpm). Both closed-loop PASS.
- Each case: original/perturbed/comparison figures (PDF+PNG), original & synthesized questions, audit (parametrized provenance + gates), 500 Hz 12-lead signals (.npz / WFDB).
- `end_to_end/` — the full waveform→MCQ→CoT pipeline figure + the two actual Qwen3.8 thinking-CoT SFT samples.
- `paper_panel_2cases.{pdf,png}` — the two-case figure for the paper.
- Internal absolute paths were redacted for the public repo; `MANIFEST.json`/`BUNDLE_SHA256SUMS.txt` were recomputed so integrity still verifies (`sha256sum -c BUNDLE_SHA256SUMS.txt`). See [`waveform_mcq_case_study/README.md`](waveform_mcq_case_study/README.md) + [`waveform_mcq_case_study/REDACTION_NOTICE.md`](waveform_mcq_case_study/REDACTION_NOTICE.md).

## complex_waveform_mcq_cot/
Two non-simplified full **ECG multi-option MCQ + complete distilled Qwen3.8 thinking-CoT** cases from validated historical waveform perturbations:
- Case 1 — First-degree AV block (5 options, **gold A**, full 26,441-char CoT, integrated interpretation).
- Case 2 — Right bundle branch block (5 options, **gold C**, full 19,614-char CoT, mechanism localization).
- `MCQ_COT_PROMPTS_REVIEW.md` — each exact image + exact MCQ prompt + complete accepted CoT (the place to start).
- `selected_complex_cases_2.jsonl` — the two original SFT rows (unshortened). `paper_complex_mcq_cot_2cases.{pdf,png}` — compact figure.
- Internal image paths in the JSONL were redacted for the public repo; `SHA256SUMS.txt` was recomputed (`sha256sum -c` 8/8 OK). See [`complex_waveform_mcq_cot/README.md`](complex_waveform_mcq_cot/README.md) + [`complex_waveform_mcq_cot/REDACTION_NOTICE.md`](complex_waveform_mcq_cot/REDACTION_NOTICE.md).
