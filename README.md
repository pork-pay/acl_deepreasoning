# acl_deepreasoning

Test push from training environment via ssh-over-443.

## sample/
One example from the `medxpert-ecg` split (MedXpertQA-MM cardiovascular subset, ECG-vision-filtered):
- `MM-19.json` — question text + gold answer (id `medxpert-ecg-MM-19`, 5-choice A–E)
- `MM-19.jpeg` — the ECG image (embedded in the source parquet)

## hint_inject_samples/
`单图、良性平片` (chest X-ray + elbow radiograph) samples for the **truncate-correct-CoT @ ~30% → gpt-5.6 self-check hint → 35B continue** method. Two single-image samples, each **triple-verified** (teacher == 35B continuation == gpt-5.6 oracle). See [`hint_inject_samples/README.md`](hint_inject_samples/README.md).
