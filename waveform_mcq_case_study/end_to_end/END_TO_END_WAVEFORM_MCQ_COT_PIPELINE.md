# 波形扰动 → MCQ → CoT SFT：可复现端到端案例

本文记录两条实际案例的完整构造链，而不是抽象方案：

1. PTB-XL 原始 12 导联波形；
2. 形态保持的定向波形扰动；
3. 扰动后重新检测、测量和闭环验收；
4. 根据实测值确定性构造四选一 MCQ；
5. Qwen3.8-max 看扰动后 ECG，开启 thinking 生成 CoT；
6. 严格检查最终 `\boxed{}` 与算法金标一致，装配成 SFT。

全部写入 `<output_root>/`。原始 PTB-XL 数据只读，两个源记录均来自训练划分，不使用测试集。

## 1. 两个案例

| 案例 | 源 ECG | strat_fold | 扰动 | 原始实测 | 扰动后实测 | 原题→合成题金标 |
|---|---:|---:|---|---|---|---|
| Case 1 | 439 | 7 | 延长 PQ 等电段 | HR 74.6、PR 130、QRS 88、QTc 400.4 | HR 74.6、PR 298、QRS 96、QTc 393.7 | A→C |
| Case 2 | 774 | 5 | 延长 TP 等电段 | HR 72.3、PR 148、QRS 51、QTc 420.4 | HR 47.9、PR 150、QRS 48、QTc 348.5 | A→B |

单位：HR 为 bpm，其余间期为 ms。合成标签均为自动闭环标签，尚未经过心电专家人工终审。

## 2. 阶段一：源波形选择

源记录来自 PTB-XL 1.0.3 的 500 Hz 波形：

- ECG 439：`records500/00000/00439_hr`，训练折 `strat_fold=7`；
- ECG 774：`records500/00000/00774_hr`，训练折 `strat_fold=5`。

每条截取固定 10 秒，得到 `(5000, 12)` 数组，导联顺序统一为：

```text
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

选择门槛：原始波形必须处于正常参数范围，且已有原题、来源哈希和训练划分证明：

```text
60 <= HR <= 100
QRS < 120 ms
120 <= PR <= 200 ms
source split == train
```

关键文件：

- `case_01_pr_prolongation/audit.json`
- `case_02_rate_slowing/audit.json`
- `render_paper_examples.py`

## 3. 阶段二：形态保持的波形扰动

### 3.1 Case 1：PR 延长

目标是把 PR 延长至约 300 ms，同时维持心率和 QRS–T 形态。

每个安全心搏按 lead II 的 fiducial 边界拆成：

```text
[P wave] [PQ isoelectric gap] [QRS–T] [TP rest]
```

然后执行：

```text
P wave             原样复制
PQ isoelectric gap 重采样并延长
QRS–T              原样复制
TP rest            相应缩短，保证该 RR 周期总长度不变
```

核心约束：

- 目标 PR：300 ms；
- 至少安全编辑 5 个心搏，本例实际编辑 10 个；
- 每个心搏内 `QRS–T copy max abs error = 0 mV`；
- RR 周期总长度不变，因此心率基本不变。

最终复测：

```text
HR   74.6 → 74.6 bpm
PR    130 → 298 ms
QRS    88 → 96 ms
QT    359 → 353 ms
QTc 400.4 → 393.7 ms
```

验收条件：

```text
PR_perturbed > 200 ms
|HR_perturbed - HR_original| <= 3 bpm
|QRS_perturbed - QRS_original| <= 15 ms
|QT_perturbed - QT_original| <= 20 ms
```

### 3.2 Case 2：减慢心率

目标是把心率降至约 48 bpm，同时完整保留每个 P–QRS–T 活动段。

每个周期拆成：

```text
[P–QRS–T active segment] [post-T / pre-P TP rest]
```

然后执行：

```text
P–QRS–T active segment  原样复制
TP rest                 延长到目标 RR 周期
target RR               60 / 48 = 1.25 s = 625 samples @ 500 Hz
```

本例输出 8 个可见周期，其中 7 个完整周期。每个可见 P–QRS–T 段与母本逐样本一致，最大复制误差为 0 mV。

最终复测：

```text
HR   72.3 → 47.9 bpm
PR    148 → 150 ms
QRS    51 → 48 ms
QT    383 → 390 ms
QTc 420.4 → 348.5 ms
```

验收条件：

```text
HR_perturbed < 60 bpm
|PR_perturbed - PR_original| <= 10 ms
|QRS_perturbed - QRS_original| <= 10 ms
|QT_perturbed - QT_original| <= 20 ms
```

### 3.3 两例共同工程门槛

```text
shape == (5000, 12)
all values finite
fraction(|amplitude| > 3.5 mV) == 0
source file SHA-256 unchanged
target gate == pass
protected metrics gate == pass
```

两例自动门禁均为 `PASS`。

## 4. 阶段三：重新测量与渲染

扰动后不沿用“设计目标值”，而是重新执行：

```text
R-peak redetection
→ beat delineation
→ HR / PR / QRS / QT / QTc recomputation
→ target and protected-metric checks
```

渲染采用：

- 标准 4×3、每格 2.5 秒；
- 附 10 秒 lead-II rhythm strip；
- 25 mm/s、10 mm/mV；
- 固定纵轴 ±2 mV；
- PNG 600 dpi；
- PDF 中波形、文字和网格为矢量；
- 图内不写诊断、金标或扰动类型，避免答案泄漏。

每例保留：

```text
source_original.png   数据库原图
original.png/.pdf     重新渲染的原始波形
perturbed.png/.pdf    扰动后波形
comparison.png/.pdf   原始与扰动对照
signal_original.npz
signal_perturbed.npz
audit.json
```

## 5. 阶段四：MCQ 构造

### 5.1 这两个案例没有让 LLM 自由编题

MCQ 由扰动后实测参数确定性生成。这样可以避免：

- LLM 把目标设计值当成实测值；
- 非目标间期被错误沿用；
- 多个选项同时成立；
- 题面泄露“这是合成波形”。

构造规则：

1. 正确选项使用扰动后复测值；
2. 每个干扰项只改变一个诊断维度；
3. 保持四个选项格式对称；
4. 正确答案位置与原题不同，用于形成反事实翻盘；
5. 题面末尾严格使用训练集已有指令。

### 5.2 公共提示词模板

```text
<image>
Question: 按本数据构建流程的自动闭环测量口径，看这张12导联ECG，下列哪项参数分析与结论是正确的?
Options:
A: {option_A}
B: {option_B}
C: {option_C}
D: {option_D}
Please reason step by step, and put your final answer within \boxed{}.
```

这里的 `{option_*}` 由程序填充，不由模型生成。

### 5.3 Case 1 的完整 MCQ

```text
<image>
Question: 按本数据构建流程的自动闭环测量口径，看这张12导联ECG，下列哪项参数分析与结论是正确的?
Options:
A: HR 74.6 bpm，PR 130.0 ms，QRS 96.0 ms，QTc 393.7 ms → 结论: 未见一度房室传导阻滞
B: HR 48.0 bpm，PR 298.0 ms，QRS 96.0 ms，QTc 393.7 ms → 结论: 窦性心动过缓合并一度房室传导阻滞
C: HR 74.6 bpm，PR 298.0 ms，QRS 96.0 ms，QTc 393.7 ms → 结论: 一度房室传导阻滞
D: HR 74.6 bpm，PR 298.0 ms，QRS 130.0 ms，QTc 393.7 ms → 结论: 一度房室传导阻滞合并室内传导延迟
Please reason step by step, and put your final answer within \boxed{}.
```

金标为 `C`：A 使用原始 PR；B 同时伪造心动过缓；D 同时伪造 QRS 增宽。

### 5.4 Case 2 的完整 MCQ

```text
<image>
Question: 按本数据构建流程的自动闭环测量口径，看这张12导联ECG，下列哪项参数分析与结论是正确的?
Options:
A: HR 72.3 bpm，PR 150.0 ms，QRS 48.0 ms，QTc 348.5 ms → 结论: 正常心率
B: HR 47.9 bpm，PR 150.0 ms，QRS 48.0 ms，QTc 348.5 ms → 结论: 窦性心动过缓
C: HR 47.9 bpm，PR 220.0 ms，QRS 48.0 ms，QTc 348.5 ms → 结论: 窦性心动过缓合并一度房室传导阻滞
D: HR 47.9 bpm，PR 150.0 ms，QRS 48.0 ms，QTc 490.0 ms → 结论: 窦性心动过缓合并 QT 间期延长
Please reason step by step, and put your final answer within \boxed{}.
```

金标为 `B`：A 沿用原始心率；C 伪造 PR 延长；D 伪造 QTc 延长。

## 6. 阶段五：Qwen3.8-max 合成 CoT

### 6.1 实际模型输入

没有额外 system prompt。实际请求只有一个多模态 user message：

```json
{
  "model": "qwen3.8-max",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "<上面的完整 MCQ>"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }
  ],
  "max_tokens": 16384,
  "temperature": 0.0,
  "enable_thinking": true,
  "chat_template_kwargs": {"enable_thinking": true},
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

图片处理参数：最长边 2048 px，JPEG quality 90。演示仅两题，因此并发 2；生产批量通常并发 64。

### 6.2 为什么必须开启 thinking

thinking 开启后：

- `reasoning_content`：模型内部完整推理；
- `content`：最终简洁解释与 `\boxed{X}`；
- SFT assistant：`reasoning_content + "\n\n" + content`。

若关闭 thinking，历史实测平均 CoT 只有约 2703 字符，明显短于参考 37K；开启并拼接后历史大池均值约 9967 字符。

### 6.3 严格接收门槛

一条 CoT 只有同时满足以下条件才进入 SFT：

```text
API response successful
reasoning_content non-empty
final content contains a strict boxed A–D letter
last boxed letter in complete assistant target == gold
last boxed letter in final content == gold
```

必须取最后一个 `\boxed{}`，因为 reasoning 中可能出现试探答案。

### 6.4 本次两条实际结果

| 案例 | gold | Qwen预测 | reasoning 字符 | final content 字符 | 完整 assistant 字符 | 结果 |
|---|---|---|---:|---:|---:|---|
| PR 延长 | C | C | 8185 | 658 | 8845 | PASS |
| 心率减慢 | B | B | 5247 | 685 | 5934 | PASS |

Case 1 最终内容：

```text
Looking at the rhythm strip: the R-R intervals correspond to a rate of about 75 bpm ...
the QRS complexes are narrow ... Therefore ... first-degree AV block ...
\boxed{C}
```

Case 2 最终内容：

```text
The R-R intervals span roughly 6+ large boxes ... giving a rate of approximately 48 bpm ...
PR about 150 ms ... QRS narrow ... QTc approximately 350 ms ...
\boxed{B}
```

完整未截断 CoT 见 `waveform_mcq_cot_sft_2.jsonl`。

## 7. 最终 SFT 格式

```json
{
  "id": "paper_cf_439_pr_prolongation_morphology_preserving",
  "messages": [
    {"role": "user", "content": "<image>\n...完整MCQ..."},
    {"role": "assistant", "content": "...reasoning_content...\n\n...final content...\n\\boxed{C}"}
  ],
  "images": [".../case_01_pr_prolongation/perturbed.png"],
  "gold_code": "1stAVB",
  "gold_letter": "C",
  "target": "1stAVB",
  "competency": "measurement_criterion",
  "provenance": {
    "waveform_counterfactual": true,
    "cot_model": "qwen3.8-max",
    "thinking_requested": true,
    "reasoning_nonempty": true,
    "teacher_match": true
  }
}
```

## 8. 可复现命令

### 8.1 重建波形、题目与论文图

```bash
python <output_root>/train/ecg/waveform_perturb/paper_examples/\
two_case_perturbation_20260924/render_paper_examples.py
```

### 8.2 生成 Qwen3.8 CoT

密钥仅通过环境变量传入，不写文件：

```bash
export MATRIXLLM_KEY='<your-key>'

python <output_root>/train/data/ecg/scripts/run_waveform_mcq_model.py \
  --model qwen3.8-max \
  --input ./cot_teacher_input.jsonl \
  --output ./qwen38_cot_raw.jsonl \
  --failures ./qwen38_cot_failures.jsonl \
  --concurrency 2 \
  --attempts 5 \
  --timeout-seconds 1200 \
  --max-tokens 16384 \
  --image-max-dim 2048 \
  --jpeg-quality 90 \
  --thinking \
  --stream
```

生产规模可把 `--concurrency` 调为 64，但需考虑网关限流。

## 9. 大规模生产版的差异

两条论文案例为了证明因果闭环，MCQ 直接由复测数字生成。完整 23 类生产池还会：

1. 每类选择 25 个闭环通过的波形源；
2. 每个源生成 20 个能力槽位，共 500 MCQ/类；
3. 覆盖整体判读、形态、测量标准、机制定位、鉴别、动态行为和风险；
4. 干扰项来自显式混淆类，不跨任务随意抽取；
5. GPT‑5.6-sol 与 Gemini‑3.8-flash 先做独立答题过滤；
6. Qwen3.8-max thinking 再生成 CoT；
7. 仅保留图存在、标签一致、推理非空、最终 boxed 正确的样本。

生产版 MCQ 公共模板：

```text
<image>
Question: {intro} {competency-specific stem}
Options:
A. {statement_A}
B. {statement_B}
C. {statement_C}
D. {statement_D}
E. {statement_E}
Please reason step by step, and put your final answer within \boxed{}.
```

反馈驱动对比题模板：

```text
<image>
A standard-calibration 12-lead ECG is shown. Use only visible ECG evidence;
do not infer unprovided history or laboratory values.
{one of three contrastive stems}
A. {diagnosis-and-evidence statement}
B. {diagnosis-and-evidence statement}
C. {diagnosis-and-evidence statement}
D. {diagnosis-and-evidence statement}
Please reason step by step, explicitly compare the closest alternatives,
and put your final answer within \boxed{}.
```

## 10. 关键边界

- 波形改变的“目标参数正确”不等于临床诊断已由专家确认；
- 题目和 CoT 不进入测试折；
- 不把设计值直接当实测值；每次扰动后必须重检测、重计算；
- 不在图标题中写疾病名或答案；
- 如果非目标参数超出容差，整条丢弃而不是修改答案掩盖；
- CoT 答错时丢弃，不强行把结尾字母改成金标；
- 论文使用时应将标签表述为 `algorithmically constructed and closed-loop verified`，并披露尚未 clinician-adjudicated。

## 11. 文件索引

- 波形与 MCQ 主目录：`../`
- 两条 CoT 请求：`cot_teacher_input.jsonl`
- 原始 Qwen 输出：`qwen38_cot_raw.jsonl`
- 可训练 SFT：`waveform_mcq_cot_sft_2.jsonl`
- 波形生成脚本：`../render_paper_examples.py`
- 通用 Qwen 批推器：`<output_root>/train/data/ecg/scripts/run_waveform_mcq_model.py`
- 大规模 MCQ builder：`<output_root>/train/data/ecg/scripts/build_waveform_complex_mcq_pool.py`
- 大规模 SFT assembler：`<output_root>/train/data/ecg/scripts/assemble_waveform_complex_qwen_sft.py`
