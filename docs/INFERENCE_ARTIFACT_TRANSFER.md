# TranslateGemma Inference Artifact Transfer & Staging Protocol

Status: Active Reference
Date: 2026-08-16

## 1. Overview

This document specifies the artifact boundary between the **fine-tune machine** and the **serving machine**.

- **Fine-Tune Machine**: Performs training, evaluation, and adapter weight merging into the base model.
- **Serving Machine**: Hosts vLLM and the FastAPI Gateway. Operates strictly offline without access to training data, training code, or external Hugging Face downloads.

## 2. Release Directory Layout

A valid merged model release directory contains the following immutable files:

```text
exports/<release-id>/
├── config.json                        # Gemma model architecture configuration
├── generation_config.json             # Generation parameters with eos_token_id: [1, 106]
├── model-00001-of-0000X.safetensors   # BF16 merged weight shards
├── model.safetensors.index.json       # Weight index (if sharded)
├── tokenizer.json                     # Fast tokenizer definition
├── tokenizer.model                    # SentencePiece model binary (if present)
├── tokenizer_config.json              # Tokenizer settings & chat template
├── special_tokens_map.json            # Special token mappings
├── preprocessor_config.json           # Processor configuration
├── merge_manifest.json                # Release provenance, commit hashes, package versions
└── SHA256SUMS                         # Cryptographic checksums of all release files
```

## 3. Step-by-Step Staging and Transfer Workflow

### Step 3.1: Generate Merged Release on Fine-Tune Host
```bash
uv run python scripts/merge_lora_adapter.py \
    --base-model google/translategemma-12b-it \
    --adapter checkpoints/sft-translategemma-12b-it \
    --output-dir exports/tg-12b-merged-20260816 \
    --release-id tg-12b-merged-20260816 \
    --dtype bfloat16
```

### Step 3.2: Verify Checkpoint Parity and Integrity
Before transferring, run the local verification suite:
```bash
# Verify checksums and configuration
python scripts/verify_model_export.py --model-dir exports/tg-12b-merged-20260816

# Verify translation parity and stop behavior
uv run python scripts/verify_merged_checkpoint.py \
    --base-model google/translategemma-12b-it \
    --adapter checkpoints/sft-translategemma-12b-it \
    --merged-model exports/tg-12b-merged-20260816 \
    --output-report reports/merge_verification_20260816.json
```

### Step 3.3: Transfer Artifact to Serving Machine
Transfer the release directory to the target serving machine using `rsync` or secure object storage:

```bash
# Example rsync to serving host
rsync -avP --checksum \
    exports/tg-12b-merged-20260816/ \
    deploy@serving-host:/opt/models/translategemma/releases/tg-12b-merged-20260816/
```

### Step 3.4: Validate Checksums on Serving Host
On the serving host, verify the transferred directory before launching containers:
```bash
python scripts/verify_model_export.py \
    --model-dir /opt/models/translategemma/releases/tg-12b-merged-20260816
```

### Step 3.5: Update Symlink for Zero-Downtime Release Activation
On the serving host, point the active model symlink to the new verified release:
```bash
ln -sfn /opt/models/translategemma/releases/tg-12b-merged-20260816 /opt/models/translategemma/current
```
Mount `/opt/models/translategemma/current` read-only (`:ro`) into the vLLM container.
