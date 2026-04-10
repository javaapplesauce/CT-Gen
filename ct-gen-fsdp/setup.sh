#!/usr/bin/env bash
# ============================================================
# CT-Gen FSDP — One-time machine setup for Lambda Labs
# Run once after SSH-ing into the instance:
#   chmod +x setup.sh && ./setup.sh
#
# Lambda Labs instances run Ubuntu as user 'ubuntu'.
# Set CTGEN_BASE to your persistent filesystem mount, e.g.:
#   export CTGEN_BASE=/home/ubuntu/persistent/ctgen
# ============================================================
set -euo pipefail

BASE_DIR="${CTGEN_BASE:-/home/ubuntu/ctgen}"
echo "Base directory: $BASE_DIR"
mkdir -p "$BASE_DIR"

# ── System deps ───────────────────────────────────────────────
sudo apt-get update -qq && sudo apt-get install -y -qq aria2 > /dev/null 2>&1 || true

# ── Python deps ───────────────────────────────────────────────
pip install -q \
    hydra-core \
    omegaconf \
    "monai[nibabel]" \
    "transformers>=4.52.0" \
    "huggingface_hub>=0.26" \
    hf_transfer \
    "peft>=0.13" \
    "accelerate>=1.1" \
    sentencepiece \
    protobuf \
    wandb \
    aria2p \
    pandas \
    tqdm \
    nltk \
    rouge-score \
    bert-score

# ── CT-CLIP (for CTViT architecture) ─────────────────────────
CT_CLIP_DIR="$BASE_DIR/CT-CLIP"
if [ ! -d "$CT_CLIP_DIR" ]; then
    echo "Cloning CT-CLIP..."
    git clone https://github.com/ibrahimethemhamamci/CT-CLIP.git "$CT_CLIP_DIR"
else
    echo "CT-CLIP already cloned."
fi
pip install -q -e "$CT_CLIP_DIR/transformer_maskgit"
pip install -q -e "$CT_CLIP_DIR/CT_CLIP"

# ── Write accelerate config ──────────────────────────────────
ACCEL_DIR="$BASE_DIR/accel"
mkdir -p "$ACCEL_DIR"
cat > "$ACCEL_DIR/fsdp_config.yaml" << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: Gemma3DecoderLayer
  fsdp_use_orig_params: true
  fsdp_cpu_ram_efficient_loading: true
mixed_precision: bf16
num_machines: 1
num_processes: 8
main_training_function: main
EOF
echo "Accelerate config: $ACCEL_DIR/fsdp_config.yaml"

# ── Verify GPU setup ─────────────────────────────────────────
echo ""
echo "=== GPU Info ==="
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
# Update accelerate config to match detected GPU count
if [ "$NUM_GPUS" -gt 0 ] 2>/dev/null; then
    sed -i "s/^num_processes:.*/num_processes: $NUM_GPUS/" "$ACCEL_DIR/fsdp_config.yaml"
    echo "Set num_processes=$NUM_GPUS in accelerate config"
fi
echo ""
echo "Setup complete. Next steps:"
echo "  1. export CTGEN_BASE=/home/ubuntu/persistent/ctgen  # your persistent FS"
echo "  2. export HF_TOKEN=hf_..."
echo "  3. export WANDB_API_KEY=..."
echo "  4. python download_and_preprocess.py"
echo "  5. accelerate launch --config_file $ACCEL_DIR/fsdp_config.yaml train.py"
