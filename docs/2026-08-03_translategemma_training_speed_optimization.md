# Reducing TranslateGemma 12B Fine-Tuning Time

## Current Baseline

- GPU: **1 × NVIDIA H100 NVL**
- Method: **QLoRA, 4-bit NF4**
- Sequence length: **2048**
- Micro-batch size: **10**
- Gradient accumulation: **5**
- Effective batch size: **50**
- Measured throughput: approximately **1.2 samples/second**
- Estimated time for 3 million training samples:
  - **1 epoch:** approximately 27–30 days
  - **3 epochs:** approximately 80–90 days

## Practical Target

A realistic target for one H100 is:

> **Train one epoch over 3 million records in about one week.**

Training three complete epochs in one week will usually require multiple H100 GPUs.

## Recommended Optimizations

### 1. Start With One Epoch

Use one full epoch first instead of committing to three epochs.

```yaml
epochs: 1
```

Evaluate after the first epoch using:

- Validation loss
- COMET or MetricX
- Scientific probe-set performance
- Number, unit, formula, and acronym preservation
- Terminology accuracy

Continue training only if the evaluation still improves.

---

### 2. Enable Sequence Packing

Packing combines multiple short examples into full token blocks and reduces wasted padding.

Recommended approach:

```python
SFTConfig(
    max_length=2048,
    packing=True,
    packing_strategy="bfd",
)
```

Measure **non-padding tokens per second**, not only samples per second.

Potential improvement:

```text
Approximately 1.5× to 4×, depending on sequence-length distribution.
```

---

### 3. Use FlashAttention 2

Load the model with:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": 0},
)
```

Install it with:

```bash
pip install flash-attn --no-build-isolation
```

FlashAttention is most effective when combined with packing or padding-free batches.

---

### 4. Analyze Sequence Lengths

Calculate token-length percentiles before training:

```text
P50, P75, P90, P95, P99
```

If most examples are shorter than 1024 tokens, use two training groups:

```text
Stage 1: Most records with max_length=1024
Stage 2: Long records with max_length=2048
```

Example:

```yaml
max_length: 1024
```

Do not process all records at 2048 tokens when only a small fraction requires that length.

---

### 5. Group Examples by Length

When packing is unavailable, enable length-based batching:

```python
TrainingArguments(
    group_by_length=True,
    length_column_name="length",
)
```

This reduces padding by placing similarly sized examples in the same batch.

---

### 6. Compare QLoRA With BF16 LoRA

QLoRA reduces memory usage but may not provide the highest throughput on an H100 because of quantization and dequantization overhead.

Benchmark both configurations:

```text
A. QLoRA 4-bit + packing + FlashAttention
B. BF16 LoRA + packing + FlashAttention
```

Example BF16 loading:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": 0},
)
```

Select the configuration with the highest **real tokens per second** while maintaining stable loss.

---

### 7. Reevaluate Gradient Checkpointing

Gradient checkpointing saves memory but increases computation.

First enable packing and FlashAttention. Then test whether checkpointing can be disabled safely:

```yaml
gradient_checkpointing: false
```

Only keep it disabled if peak VRAM remains safely below the GPU limit.

---

### 8. Pre-tokenize the Dataset

Tokenize the complete dataset once and save it to fast local storage.

```python
tokenized = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    num_proc=32,
    remove_columns=dataset.column_names,
)

tokenized.save_to_disk("/fast-nvme/tokenized_train")
```

Load it during training:

```python
dataset = load_from_disk("/fast-nvme/tokenized_train")
```

This improves startup time, experiment repetition, and CPU/I/O efficiency.

---

### 9. Use Fast Local Storage

Store the following on local NVMe storage:

- Tokenized datasets
- Checkpoints
- Hugging Face cache
- Temporary files

Avoid reading millions of records repeatedly from slow network storage.

---

### 10. Use Multiple GPUs for Three Epochs

For three epochs in approximately one week, use multiple H100 GPUs.

Recommended setup:

```text
4 × H100
+ sequence packing
+ FlashAttention 2
+ length-based dataset buckets
```

Launch example:

```bash
accelerate launch   --multi_gpu   --num_processes 4   train.py   --config config.yaml
```

Effective batch size must include the number of GPUs:

```text
effective_batch =
per_device_batch × gradient_accumulation × number_of_gpus
```

Example:

```text
2 × 6 × 4 = 48
```

---

## Benchmark Before the Full Run

Use a fixed subset of 20,000–50,000 records and run each configuration for at least 200 optimizer steps.

| Run | Packing | Attention | Max Length | Precision |
|---|---|---|---:|---|
| A | No | Current | 2048 | 4-bit QLoRA |
| B | Yes | FlashAttention 2 | 2048 | 4-bit QLoRA |
| C | Yes | FlashAttention 2 | 1024 | 4-bit QLoRA |
| D | Yes | FlashAttention 2 | 1024 | BF16 LoRA |

Record:

- Real tokens per second
- Non-padding tokens per second
- Samples per second
- Peak VRAM
- Training loss
- Validation loss
- Translation quality after an equal number of target tokens

## Recommended One-H100 Plan

```text
1. Train for one epoch.
2. Enable sequence packing.
3. Enable FlashAttention 2.
4. Use max_length=1024 for most records.
5. Train long records separately at max_length=2048.
6. Compare QLoRA against BF16 LoRA.
7. Keep gradient checkpointing only if required.
8. Pre-tokenize and store data on local NVMe.
```

## Expected Outcome

With one H100, the combination of:

```text
one epoch
+ packing
+ FlashAttention 2
+ shorter sequence buckets
+ optimized precision
```

may reduce training time from roughly one month per epoch to approximately one week.

For three complete epochs in one week, plan for approximately **4–8 H100 GPUs**, depending on measured token throughput and dataset length distribution.
