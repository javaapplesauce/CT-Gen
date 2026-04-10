# CT-Gen FSDP: Multi-GPU Training Pipeline

A fully sharded data-parallel (FSDP) training pipeline for **CT-Gen**, a 3D medical vision-language model that generates structured radiological reports from CT volumes.

## Architecture

```
                         frozen, float32                    trainable
  ┌──────────────┐      ┌──────────────┐      ┌───────────────────────┐
  │   CT Volume   │      │    CTViT     │      │   CTGenAggregator     │
  │ (1,40,480,480)│─────▶│  (CT-CLIP v2)│─────▶│  AdaptiveAvgPool3d    │
  └──────────────┘      └──────────────┘      │  (4,8,8) → 256 tokens │
                         2304 tokens           │  MLP: 512→2048→5376   │
                         dim=512               └───────────┬───────────┘
                                                           │ 256 tokens, dim=5376
                                                           ▼
                                               ┌───────────────────────┐
                    ┌─────────────────────────▶│  MedGemma 27B         │
                    │  text tokens              │  (bf16, LoRA in S2)   │
                    │                           └───────────┬───────────┘
                    │                                       │
             ┌──────┴──────┐                     ┌──────────▼──────────┐
             │  Tokenizer  │                     │  Radiological Report │
             └─────────────┘                     └─────────────────────┘
```

### Visual token pipeline

1. **CTViT** encodes a 40-slice CT volume into 2304 tokens (4 temporal x 24 x 24 spatial), each dim=512.
2. **CTGenAggregator** pools these down to 256 tokens via `AdaptiveAvgPool3d((4,8,8))` and projects 512 → 5376 through a two-layer MLP with GELU activation.
3. The 256 visual tokens are wrapped with `<|visual_start|>` and `<|visual_end|>` special tokens, prepended to the text sequence, and fed to the LLM as `inputs_embeds`.

### Training stages

| | Stage 1: Cross-Modal Alignment | Stage 2: Instruction Tuning |
|---|---|---|
| **Goal** | Teach the projector to translate 3D CT features into LLM-compatible tokens | Fine-tune the full system to follow instructions and output structured reports |
| **Trainable** | CTGenAggregator only | CTGenAggregator + LLM (LoRA adapters) |
| **Frozen** | CTViT + LLM | CTViT |
| **Data** | Volume + raw report text | Volume + instruction-formatted report |
| **Loss target** | Next-token prediction on report tokens | Next-token prediction on assistant response only |
| **Label masking** | -100 for visual prefix, -100 for padding | -100 for visual prefix + instruction, -100 for padding |
| **Epochs** | 10 | 5 |
| **LR** | 2e-4 | 1e-4 (projector), 1e-5 (LoRA) |

## Files

| File | Purpose | When to run |
|------|---------|-------------|
| `conf/config.yaml` | All configuration (Hydra) — paths, model, training, data, eval | Edit before training |
| `setup.sh` | Install Python/system dependencies, clone CT-CLIP, write accelerate config | Once per machine |
| `download_and_preprocess.py` | Download CT-CLIP weights + CT-RATE volumes via aria2c, preprocess NIfTI → float16 `.pt` tensors | Once per machine |
| `train.py` | Multi-GPU FSDP training (Stage 1 + Stage 2) with held-out test split | Via `accelerate launch` |
| `evaluate.py` | Generate reports on test set, compute BLEU/ROUGE/METEOR/BERTScore | After training |
| `inference.py` | Single-GPU inference demo — loads best checkpoint, generates a report | After training |
| `run.sh` | End-to-end launcher: download → preprocess → train | Convenience wrapper |

## Quick start (Lambda Labs)

```bash
# 0. Accept gated model/dataset terms on HuggingFace:
#    - https://huggingface.co/google/medgemma-27b-text-it  (MedGemma)
#    - https://huggingface.co/datasets/ibrahimhamamci/CT-RATE  (CT-RATE)

# 1. Rent an 8x A100 SXM4 80GB instance on Lambda Labs (recommended for 27B)
#    Optionally attach a persistent filesystem for data/checkpoints

# 2. SSH in
ssh ubuntu@<instance-ip>

# 3. Clone your repo
git clone <your-repo-url> ~/CT-Gen
cd ~/CT-Gen/scripts/ct-gen-fsdp

# 4. Set credentials and base directory
export HF_TOKEN=hf_...                              # huggingface.co/settings/tokens
export WANDB_API_KEY=...                             # wandb.ai/authorize
export CTGEN_BASE=/home/ubuntu/persistent/ctgen      # persistent FS (recommended)
# Or for ephemeral storage:
# export CTGEN_BASE=/home/ubuntu/ctgen

# 5. Run
chmod +x setup.sh run.sh
./setup.sh        # installs deps (~2 min)
./run.sh          # downloads data, preprocesses, trains both stages

# 6. Evaluate on held-out test set
python evaluate.py
```

### Persistent filesystem

Lambda Labs lets you attach a persistent filesystem that survives instance termination. Point `CTGEN_BASE` to it so preprocessed data, checkpoints, and model weights persist across sessions. The persistent FS is typically mounted at `/home/ubuntu/persistent` — create a subdirectory for CT-Gen:

```bash
export CTGEN_BASE=/home/ubuntu/persistent/ctgen
```

This is especially valuable because:
- Preprocessed `.pt` tensors (~370 GB for the full dataset) don't need to be re-downloaded/re-processed
- Checkpoints survive instance preemptions or intentional shutdowns
- You can stop/restart instances without losing progress

### Running steps individually

```bash
# Download + preprocess only (single process)
python download_and_preprocess.py

# Train only (multi-GPU)
accelerate launch \
    --config_file $CTGEN_BASE/accel/fsdp_config.yaml \
    train.py

# Evaluate on held-out test set (after training)
python evaluate.py
python evaluate.py --limit 20        # quick check on 20 samples

# Inference on a single volume (after training)
python inference.py
python inference.py --volume $CTGEN_BASE/data/ct_rate_pt/valid_42_a_1.pt
```

## Configuration

All configuration lives in **`conf/config.yaml`** and is managed via [Hydra](https://hydra.cc/). Any value can be overridden from the command line:

```bash
# Override hyperparameters
accelerate launch ... train.py stage1.lr=3e-4 stage1.epochs=20

# Override data settings
python download_and_preprocess.py data.volume_limit=500 data.chunk_size=20

# Override model
accelerate launch ... train.py model.llm_id=meta-llama/Meta-Llama-3-8B-Instruct model.llm_hidden_dim=4096
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CTGEN_BASE` | `/home/ubuntu/ctgen` | Root directory for all data, models, and checkpoints (use persistent FS path) |
| `HF_TOKEN` | *(required)* | HuggingFace token with access to CT-RATE (gated dataset) and MedGemma (gated model) |
| `WANDB_API_KEY` | *(optional)* | Weights & Biases API key for experiment tracking |
| `WANDB_PROJECT` | `ct-gen-vlm` | W&B project name |

### Config structure (`conf/config.yaml`)

```yaml
paths:          # base_dir, ct_clip_weights, pt_dir, raw_dir, ckpt_dir
model:          # llm_id, llm_hidden_dim, visual_token_dim, lora_*
stage1:         # epochs, lr, batch_size, accum_steps, log_every
stage2:         # epochs, lr, batch_size, accum_steps, log_every
data:           # test_split, val_split, volume_limit, chunk_size, ...
eval:           # checkpoint, max_tokens
inference:      # max_tokens
instruction:    # prompt text
tokens:         # visual_start, visual_end
wandb:          # project
```

Key defaults:

| Key | Default | Notes |
|-----|---------|-------|
| `model.llm_id` | `google/medgemma-27b-text-it` | Gated — accept terms on HuggingFace |
| `model.llm_hidden_dim` | `5376` | Must match the LLM |
| `model.lora_r` | `16` | LoRA rank |
| `stage1.epochs` / `stage2.epochs` | `10` / `5` | |
| `stage1.lr` / `stage2.lr` | `2e-4` / `1e-4` | Stage 2 LoRA LR = `lr * 0.1` |
| `stage1.accum_steps` | `4` | Effective batch = `accum * batch * num_gpus` |
| `data.test_split` | `0.05` | Held-out test fraction |
| `data.val_split` | `0.1` | Validation fraction |
| `data.volume_limit` | `null` (all) | Set to e.g. `500` for a test run |
| `data.chunk_size` | `50` | Volumes per download-preprocess chunk |

### Switching LLMs

Override from the command line, or edit `conf/config.yaml`:

```bash
accelerate launch ... train.py \
    model.llm_id=meta-llama/Meta-Llama-3-8B-Instruct \
    model.llm_hidden_dim=4096
```

Also update `fsdp_transformer_layer_cls_to_wrap` in the accelerate config (`setup.sh`):

| LLM | `llm_hidden_dim` | FSDP wrap class |
|-----|-------------------|-----------------|
| `google/medgemma-27b-text-it` (default) | 5376 | `Gemma3DecoderLayer` |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 4096 | `LlamaDecoderLayer` |
| `microsoft/Phi-3-mini-4k-instruct` | 3072 | `Phi3DecoderLayer` |

## FSDP details

### Why FSDP over DDP?

Standard DDP replicates the full model on every GPU. MedGemma 27B in bf16 is ~54 GB — it doesn't fit on a single A100 80GB. FSDP shards params, gradients, and optimizer state across GPUs:

| Component | DDP (per GPU) | FSDP (4 GPUs, per GPU) | FSDP (8 GPUs, per GPU) |
|-----------|--------------|------------------------|------------------------|
| Model params | 54 GB | 13.5 GB | 6.75 GB |
| Optimizer state | 108 GB | 27 GB | 13.5 GB |
| Gradients | 54 GB | 13.5 GB | 6.75 GB |
| **Total** | **216 GB** | **54 GB** | **27 GB** |

With 4 GPUs this leaves ~26 GB per A100 for activations and CTViT — tight but workable thanks to gradient checkpointing. With 8 GPUs you get ~53 GB free per GPU — comfortable, and you can increase batch size or accumulation steps for faster convergence. **8x A100 is recommended for MedGemma 27B.**

### FSDP configuration

| Setting | Value | Reason |
|---------|-------|--------|
| Sharding | `FULL_SHARD` | Shard params + gradients + optimizer state |
| Backward prefetch | `BACKWARD_PRE` | Prefetch next layer's params during backward — hides communication |
| Forward prefetch | `true` | Prefetch next layer's params during forward |
| Auto-wrap | `Gemma3DecoderLayer` | Each transformer layer is an FSDP unit — balances memory and communication |
| `use_orig_params` | `true` | Required for mixed parameter groups (projector vs LoRA LR) |
| Mixed precision | `bf16` | A100 tensor cores; avoids fp16 overflow on dense medical features |

### Why bf16 LoRA instead of QLoRA?

QLoRA (4-bit NF4 quantization + LoRA) doesn't work with FSDP. Quantized parameters are stored in a packed format that FSDP cannot shard across GPUs. Since 4-8x A100 80GB provides 320-640 GB total VRAM, there is no need for quantization — the full bf16 model fits comfortably with FSDP sharding.

## Data pipeline

### CT-RATE dataset

The pipeline downloads from the [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) HuggingFace dataset (gated — you must accept terms at the HF page before your token will work).

**Important:** `valid_metadata.csv` is not publicly accessible. The pipeline uses `validation_reports.csv` exclusively for both volume IDs and report text.

### Streaming chunked pipeline

The naive approach — download all 21 TB, then preprocess — requires 21 TB of temporary disk. The streaming pipeline solves this:

```
for each chunk of 50 volumes:
    1. aria2c downloads ~50 GB of raw NIfTIs  (64 concurrent × 16 streams)
    2. Multiprocessing pool preprocesses all 50 in parallel
    3. Raw NIfTIs are deleted immediately
    4. Next chunk begins
```

**Peak disk usage:** ~50 GB (one chunk of raw NIfTIs) + final .pt files, instead of 21 TB.

### Preprocessing

Raw NIfTI volumes (~1 GB each) are preprocessed once into float16 PyTorch tensors (~17.6 MB each):

```
Raw NIfTI → Reorient (RAS) → Resample (1.5mm x 1.5mm x 2.0mm)
          → Normalize (HU [-1000, 400] → [0, 1])
          → Resize (480 x 480 x 40 slices)
          → Permute to (C, Slices, H, W)
          → Cast to float16
          → Save as .pt
```

| Scale | Raw NIfTI | Peak raw disk (streaming) | Final preprocessed .pt |
|-------|-----------|---------------------------|----------------------|
| 1 volume | ~1 GB | ~1 GB | ~17.6 MB |
| 500 volumes | ~500 GB | ~50 GB | ~8.8 GB |
| 21,000 volumes (full dataset) | ~21 TB | **~50 GB** | ~370 GB |

Preprocessing is parallelized across all available CPU cores (auto-detected, capped at 32 workers).

### Download optimization

Downloads use `aria2c` with aggressive parallelism:

- 64 concurrent file downloads
- 16 TCP streams per file
- `--file-allocation=none` (skip pre-zeroing disk)
- `--disk-cache=512M` (buffer writes)

CT-CLIP weights and the reports CSV use `hf_transfer` (HuggingFace's Rust-based downloader, 3-5x faster than Python requests).

### Resumability

The pipeline is fully resumable. If interrupted:
- Already-preprocessed `.pt` files are detected and skipped
- Raw NIfTIs from a partial chunk are detected and preprocessed without re-downloading
- Just re-run `python download_and_preprocess.py` to continue where you left off

## W&B tracking

Each stage creates a separate W&B run with the following metrics:

| Metric | Stage 1 | Stage 2 |
|--------|---------|---------|
| Training loss | `s1/train_loss` | `s2/train_loss` |
| Validation loss | `s1/val_loss` | `s2/val_loss` |
| Learning rate | `s1/lr` | `s2/lr` |
| Epoch | `s1/epoch` | `s2/epoch` |
| Global step | `s1/step` | `s2/step` |
| Best val loss | `s1/best_val_loss` | `s2/best_val_loss` |

Run configs (hyperparameters, GPU count, effective batch size) are logged automatically.

## Evaluation

### Data split

Training automatically holds out 5% of volumes as a **test set** (deterministic, seed=42). The test volume names are saved to `$CTGEN_BASE/ckpts/test_volumes.json` so evaluation can be run independently after training. The remaining 95% is split 90/10 into train/val for each stage.

| Split | Fraction | Purpose |
|-------|----------|---------|
| Train | ~85.5% | Model training |
| Val | ~9.5% | Checkpoint selection (best val loss) |
| Test | 5% | Held-out evaluation (never seen during training) |

### Running evaluation

```bash
# Full evaluation on test set
python evaluate.py

# Quick check on 20 samples
python evaluate.py --limit 20

# Evaluate a specific checkpoint
python evaluate.py --checkpoint stage2/final

# Custom output path
python evaluate.py --output results/eval_v1.json
```

### Metrics

| Metric | What it measures |
|--------|-----------------|
| **BLEU-1** | Unigram precision — word overlap |
| **BLEU-4** | 4-gram precision — phrase-level fluency |
| **METEOR** | Unigram recall + synonyms + stemming |
| **ROUGE-1** | Unigram recall (word-level coverage) |
| **ROUGE-2** | Bigram recall |
| **ROUGE-L** | Longest common subsequence (structural similarity) |
| **BERTScore P** | Contextual embedding precision |
| **BERTScore R** | Contextual embedding recall |
| **BERTScore F1** | Harmonic mean of BERTScore P and R |

Results are saved to `$CTGEN_BASE/ckpts/eval_results.json` with per-sample generated/reference pairs for qualitative review.

## Checkpoints

Checkpoints are saved via `accelerator.save_state()` (FSDP-aware, handles sharded state):

```
$CTGEN_BASE/ckpts/
├── test_volumes.json   # held-out test volume names
├── eval_results.json   # evaluation metrics + per-sample outputs
├── stage1/
│   ├── best/           # lowest validation loss
│   └── final/          # end of Stage 1
└── stage2/
    ├── best/           # lowest validation loss
    └── final/          # end of Stage 2
```

Stage 2 automatically loads the best Stage 1 projector checkpoint before starting.

## Directory layout (after training)

```
$CTGEN_BASE/                    # e.g. /home/ubuntu/persistent/ctgen
├── CT-CLIP/                    # cloned CT-CLIP repo (for CTViT)
├── accel/
│   └── fsdp_config.yaml        # accelerate FSDP config
├── models/
│   └── CT-CLIP_v2.pt           # pretrained CTViT weights
├── data/
│   ├── ct_rate_raw/            # raw NIfTIs (auto-deleted by streaming pipeline)
│   │   └── dataset/
│   │       └── radiology_text_reports/
│   │           └── validation_reports.csv
│   └── ct_rate_pt/             # preprocessed float16 tensors
│       ├── valid_1_a_1.pt
│       ├── valid_1_a_2.pt
│       └── ...
└── ckpts/
    ├── test_volumes.json       # held-out test volume names
    ├── eval_results.json       # evaluation metrics + per-sample outputs
    ├── stage1/best/
    ├── stage1/final/
    ├── stage2/best/
    └── stage2/final/
```
