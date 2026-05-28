#!/bin/bash

set -euo pipefail

NGPU=8

WFLOW_HF_ROOT="/path/to/wflow_hf_root"
EXP_NAME=latent_sota_B_ot
STEPNUM='00200000'

CKPT=$WFLOW_HF_ROOT/checkpoints/$EXP_NAME/state_$STEPNUM.pt
CONFIG=configs/gen/latent_sota_B_ot_8node.yaml
cfg=1.19

WORKDIR=runs/$EXP_NAME/ckpt_$STEPNUM
OUTDIR=results/$EXP_NAME/ckpt_$STEPNUM

mkdir -p "$WORKDIR"
mkdir -p "$OUTDIR"

NCCL_DEBUG=WARN \
torchrun --nproc_per_node="$NGPU" --master_port=6667 \
    inference_ours.py evaluate \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    --cfg-scale "$cfg" \
    --num-samples 50000 \
    --gen-bsz 64 \
    --workdir "$WORKDIR/work_cfg${cfg}" \
    --json-out "$OUTDIR/results_cfg${cfg}.json"
echo ""
cat "$OUTDIR/results_cfg${cfg}.json"
