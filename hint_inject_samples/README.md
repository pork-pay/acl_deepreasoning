# Hint-Injection (Truncate-and-Continue) Paper Samples

Demonstration samples for the method used in this paper: take a **correct**
multimodal medical chain-of-thought (CoT), truncate it at a ~30% semantic
boundary (chosen by an LLM), have an LLM write **one operational self-check hint**
grounded only in the visible prefix (no answer leakage, mid-thought unfinished
end), then continue generation from `prefix + hint` so the model completes the
reasoning and reaches the correct answer.

```
<think>
[teacher-prefix ... ~30% of units]
[gpt-5.6 hint: name the decisive evidence/ distinction to re-verify, grounded in prefix only]
[35B continuation ... completes reasoning, no restating of hint leakage]
</think>
Most appropriate answer: \boxed{X}.
```

## Correctness gating (the bundle has no usable official gold)
The source bundle's questions are **rephrased** relative to the official
MedXpertQA test set, and its images use a different naming scheme, so the
official label cannot be joined. "Correct" is therefore defined as triple
agreement:

- **teacher** boxed answer (the truncated correct CoT, from qwen-3.8)
- **35B continuation** answer (continue_final_message from `prefix + hint`)
- **gpt-5.6-sol** independent re-answer (oracle, answers the question fresh, no CoT prefix)

A sample is included only if all three agree.

## Samples (single image, benign plain film)
| folder | modality | vignette | ans | cut |
|---|---|---|---|---|
| 选题08_胸片_ansA | chest X-ray (portable AP, post-op) | bilateral lower-lung opacities, most likely cause | A | 27% (unit 10/37) |
| 选题31_骨折_ansC | elbow radiograph | fall on outstretched hand, most likely diagnosis | C | 28% (unit 10/35) |

Each folder contains:
- `原题.txt` / `题干.txt` — full question + options
- `续写加hint版.txt` — final stitched assistant turn (prefix → hint → 35B continuation → boxed answer)
- `meta.txt` — teacher / 35B-cont / gpt-5.6 oracle answers, cut unit, hint, lengths
- the image (`jpg`/`png`)

Both samples are **triple-verified** (teacher == 35B continuation == gpt-5.6 oracle).

---

两个论文样本(单图、良性平片):截断teacher CoT@~30% → gpt-5.6 在前缀可见范围写一条操作性自检 hint → 35B 以 assistant 前缀续写到正确答案;经 teacher == 35B续写 == gpt-5.6独立重答 三一致判定为医学正确。
