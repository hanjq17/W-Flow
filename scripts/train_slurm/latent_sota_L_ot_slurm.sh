#!/usr/bin/env bash
#SBATCH --job-name=latent_sota_L
#SBATCH -p YOUR_SLURM_PARTITION
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH --time=8:00:00
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

export HF_HOME=${HF_HOME:-/path/to/huggingface_cache}
mkdir -p "$HF_HOME"

CONFIG=${CONFIG:-configs/gen/latent_sota_L_ot_8node.yaml}
EXP_NAME=${EXP_NAME:-latent_sota_L_ot_8node}

WORKDIR=${WORKDIR:-/path/to/workdir/$EXP_NAME}
WANDB_PROJECT=${WANDB_PROJECT:-YOUR_WANDB_PROJECT}
WANDB_NAME=${WANDB_NAME:-$EXP_NAME}

MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${SLURM_JOB_NUM_NODES}
NGPU=${NGPU:-8}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

mkdir -p "$WORKDIR"

export DRIFT_COMPILE=1
export NCCL_DEBUG=WARN
export WANDB_PROJECT
export WANDB_NAME
export WANDB_RESUME=${WANDB_RESUME:-never}

echo "============================================================"
echo "  W-Flow-XL Training  "
echo "============================================================"
echo "  Job ID:        ${SLURM_JOB_ID:-N/A}"
echo "  Node list:     ${SLURM_JOB_NODELIST:-N/A}"
echo "  Nodes:         $NNODES"
echo "  GPUs/node:     $NGPU"
echo "  CPUs/task:     ${SLURM_CPUS_PER_TASK:-N/A}"
echo "  Master addr:   $MASTER_ADDR"
echo "  Master port:   $MASTER_PORT"
echo "  Config:        $CONFIG"
echo "  Workdir:       $WORKDIR"
echo "  HF_HOME:       $HF_HOME"
echo "  wandb project: $WANDB_PROJECT"
echo "  wandb run:     $WANDB_NAME"
echo "  CC:            ${CC:-<unset>}"
echo "  CXX:           ${CXX:-<unset>}"
echo "============================================================"

which python
python -V

unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

srun --export=ALL --ntasks="$NNODES" --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" --kill-on-bad-exit=1 bash -lc '
  cd "${REPO_DIR:-/path/to/W-Flow}"
  source "${ENV_ACTIVATE:-/path/to/venv/bin/activate}"
  export HF_HOME=${HF_HOME:-/path/to/huggingface_cache}
  export NODE_RANK=$SLURM_NODEID
  export CC='"${CC:-}"'
  export CXX='"${CXX:-}"'
  export TORCHINDUCTOR_CXX='"${TORCHINDUCTOR_CXX:-}"'

  echo "host=$(hostname) node_rank=$NODE_RANK"
  GPU_ENUM="$(nvidia-smi -L 2>&1 || true)"
  echo "$GPU_ENUM"

  python -c "import os; print('cpu_affinity_count='+str(len(os.sched_getaffinity(0)))); print('os_cpu_count='+str(os.cpu_count()))" || true

  torchrun \
    --nnodes='"$NNODES"' \
    --nproc_per_node='"$NGPU"' \
    --node_rank="$NODE_RANK" \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    train.py \
      --config "'"$CONFIG"'" \
      --workdir "'"$WORKDIR"'"
'

echo "finished!"
