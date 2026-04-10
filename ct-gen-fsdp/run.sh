#!/usr/bin/env bash
# ============================================================
# CT-Gen FSDP — Full pipeline launcher for Lambda Labs
#
# Usage:
#   # First time:
#   chmod +x setup.sh run.sh
#   ./setup.sh
#   export CTGEN_BASE=/home/ubuntu/persistent/ctgen
#   export HF_TOKEN=hf_...
#   export WANDB_API_KEY=...
#   ./run.sh
#
#   # Resume (data already downloaded):
#   ./run.sh
# ============================================================
set -euo pipefail

export CTGEN_BASE="${CTGEN_BASE:-/home/ubuntu/ctgen}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── NCCL tuning for single-node multi-GPU (A100 SXM4 NVLink) ─
export NCCL_P2P_DISABLE=0            # enable NVLink peer-to-peer
export NCCL_IB_DISABLE=1             # no InfiniBand on single node
# Let NCCL auto-detect the socket interface.  Pinning to 'lo' (loopback)
# causes "no usable network interface" errors on Lambda/RunPod where the
# host NIC is eth0 or ens5.  Auto-detection picks the right one.
unset NCCL_SOCKET_IFNAME
export NCCL_DEBUG=WARN               # set to INFO for debugging

# ── hf_transfer for fast downloads ────────────────────────────
export HF_HUB_ENABLE_HF_TRANSFER=1

# ── Step 1: Download + Preprocess (single process) ────────────
echo "=== Step 1: Download & Preprocess ==="
python "$SCRIPT_DIR/download_and_preprocess.py"

# ── Step 2: Multi-GPU training via accelerate ─────────────────
echo ""
echo "=== Step 2: Training (FSDP) ==="

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "Detected $NUM_GPUS GPUs"

ACCEL_CONFIG="$CTGEN_BASE/accel/fsdp_config.yaml"

if [ -f "$ACCEL_CONFIG" ]; then
    # Update num_processes in config to match actual GPU count
    sed -i "s/^num_processes:.*/num_processes: $NUM_GPUS/" "$ACCEL_CONFIG"
    accelerate launch --config_file "$ACCEL_CONFIG" "$SCRIPT_DIR/train.py"
else
    # Fallback: pass args directly
    accelerate launch \
        --num_processes="$NUM_GPUS" \
        --mixed_precision=bf16 \
        "$SCRIPT_DIR/train.py"
fi

echo ""
echo "=== Training complete ==="
echo "Checkpoints: $CTGEN_BASE/ckpts/"
