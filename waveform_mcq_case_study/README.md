# 两个 ECG 人工扰动论文示例

本目录包含两个独立、可复现的“原始题 → 合成题”案例。所有新文件仅写入
`<output_root>/train/ecg/waveform_perturb/paper_examples/two_case_perturbation_20260924`，外部源数据只读。

| 案例 | 源记录 | 定向扰动 | 闭环复测 | 状态 |
|---|---:|---|---|---|
| Case 1 | ECG 439 | PR-interval prolongation | HR 74.6→74.6 bpm; PR 130→298 ms; QRS 88→96 ms | PASS |
| Case 2 | ECG 774 | Heart-rate reduction | HR 72.3→47.9 bpm; PR 148→150 ms; QRS 51→48 ms | PASS |

## 论文直接使用

- `paper_panel_2cases.pdf`：推荐，矢量双案例组合图。
- `paper_panel_2cases.png`：600 dpi 无损位图。
- 每个 `case_*` 内：`source_original.png` 是原题自带原图；`original.*` 是同一原始波形的新渲染；`perturbed.*` 是加扰后新渲染；`comparison.*` 是紧凑对照图。
- `original_question.*` 与 `synthesized_question.*` 分别保存原题和合成题；`audit.json` 保存逐项溯源、参数、哈希和门禁。
- `signal_original.npz` 与 `signal_perturbed.npz` 保存 500 Hz、12 导联的可复算波形。
- `ENVIRONMENT.json` 与 `MANIFEST.json` 保存软件版本和全部文件校验和。
- `code_snapshot/` 固化本次闭环复测所用的分析器与 fiducial 检测器。
- `end_to_end/paper_end_to_end_waveform_mcq_cot.{pdf,png}`：波形扰动→复测→MCQ→CoT 的论文案例图。
- `end_to_end/END_TO_END_WAVEFORM_MCQ_COT_PIPELINE.md`：含实际提示词、请求参数、门禁和复现命令的完整流程。
- `end_to_end/waveform_mcq_cot_sft_2.jsonl`：两条实际 Qwen3.8 thinking CoT SFT 样本。

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
python <output_root>/train/ecg/waveform_perturb/paper_examples/two_case_perturbation_20260924/render_paper_examples.py
```

脚本为 CPU-only，不占用 GPU。`MANIFEST.json` 记录除自身外全部产物的 SHA-256。
