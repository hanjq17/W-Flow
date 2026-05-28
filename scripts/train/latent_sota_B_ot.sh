set -euo pipefail

############ Single Node Training (Uncomment below) ############
# CONFIG=configs/gen/latent_sota_B_ot_1node.yaml
# EXP_NAME=latent_sota_B_ot_1node
# WORKDIR=/path/to/workdir/$EXP_NAME
# WANDB_PROJECT=YOUR_WANDB_PROJECT
# WANDB_NAME=$EXP_NAME

# NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
# NGPU=${NGPU:-1}
# MASTER_PORT=${MASTER_PORT:-6667}

# DRIFT_COMPILE=1 \
# NCCL_DEBUG=WARN \
# WANDB_PROJECT=$WANDB_PROJECT \
# WANDB_NAME=$WANDB_NAME \
# torchrun \
#     --nproc_per_node="$NGPU" \
#     --master_port="$MASTER_PORT" \
#     train.py \
#     --config "$CONFIG" \
#     --workdir "$WORKDIR"

############ Multi Node Training ############
CONFIG=configs/gen/latent_sota_B_ot_8node.yaml
EXP_NAME=latent_sota_B_ot_8node
WORKDIR=/path/to/workdir/$EXP_NAME
WANDB_PROJECT=YOUR_WANDB_PROJECT
WANDB_NAME=$EXP_NAME

# Set multi-node training environment variables
NGPU=${NGPU:-8}
NNODES=${NNODES:-8}
NODE_RANK=${NODE_RANK:?Set NODE_RANK, e.g. 0..7}
MASTER_ADDR=${MASTER_ADDR:?Set MASTER_ADDR to rank-0 node IP/host}
MASTER_PORT=${MASTER_PORT:-6667}

DRIFT_COMPILE=1 \
NCCL_DEBUG=WARN \
WANDB_PROJECT=$WANDB_PROJECT \
WANDB_NAME=$WANDB_NAME \
torchrun \
    --nproc_per_node="$NGPU" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config "$CONFIG" \
    --workdir "$WORKDIR"

echo "finished!"