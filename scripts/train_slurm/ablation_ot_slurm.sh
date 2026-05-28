#!/usr/bin/env bash
#SBATCH --job-name=abl_ot
#SBATCH -p YOUR_SLURM_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -G 8
#SBATCH --mem=0
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

cd "${REPO_DIR:-/path/to/W-Flow}"
source "${ENV_ACTIVATE:-/path/to/venv/bin/activate}"

# TorchInductor emits GCC/Clang-style flags. nvhpc's nvc++ cannot compile
# these kernels, so force GNU compilers for JIT C++ compilation.
if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
  export CC="$(command -v gcc)"
  export CXX="$(command -v g++)"
  export TORCHINDUCTOR_CXX="$CXX"
fi

NGPU=${NGPU:-${SLURM_GPUS_ON_NODE:-8}}
MASTER_PORT=${MASTER_PORT:-6667}

CONFIG=${CONFIG:-configs/gen/ablation_ot_1node.yaml}
EXP_NAME=${EXP_NAME:-ablation_ot_1node}

WORKDIR=${WORKDIR:-/path/to/workdir/${EXP_NAME}}
WANDB_PROJECT=${WANDB_PROJECT:-YOUR_WANDB_PROJECT}
WANDB_NAME=${WANDB_NAME:-$EXP_NAME}

echo "============================================================"
echo "  Ablation W-Flow Training  "
echo "============================================================"
echo "  Job ID:        ${SLURM_JOB_ID:-N/A}"
echo "  Node list:     ${SLURM_JOB_NODELIST:-N/A}"
echo "  GPUs:          $NGPU"
echo "  CPUs/task:     ${SLURM_CPUS_PER_TASK:-N/A}"
echo "  Config:        $CONFIG"
echo "  Workdir:       $WORKDIR"
echo "  Master port:   $MASTER_PORT"
echo "  wandb project: $WANDB_PROJECT"
echo "  wandb run:     $WANDB_NAME"
echo "  CC:            ${CC:-<unset>}"
echo "  CXX:           ${CXX:-<unset>}"
echo "============================================================"

mkdir -p "$WORKDIR"

export NCCL_DEBUG=WARN
export DRIFT_COMPILE=1
export WANDB_PROJECT
export WANDB_NAME
export HF_HOME=${HF_HOME:-/path/to/huggingface_cache}

echo "[diag] nvidia-smi -L"
GPU_ENUM="$(nvidia-smi -L 2>&1 || true)"
echo "$GPU_ENUM"

echo "[diag] CPU affinity"
python - <<'PY'
import os

try:
    affinity = sorted(os.sched_getaffinity(0))
    print(f"cpu_affinity_count={len(affinity)}")
except Exception as exc:
    print(f"cpu_affinity_count=unavailable ({exc})")
print(f"os_cpu_count={os.cpu_count()}")
PY

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NGPU" \
  --master_port="$MASTER_PORT" \
  train.py \
  --config "$CONFIG" \
  --workdir "$WORKDIR"

echo "finished!"
