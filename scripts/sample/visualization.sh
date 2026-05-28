#!/bin/bash

WFLOW_HF_ROOT="/path/to/wflow_hf_root"
EXP_NAME=latent_sota_XL_ot
STEPNUM='00180000'

CKPT=$WFLOW_HF_ROOT/checkpoints/$EXP_NAME/state_$STEPNUM.pt
CONFIG=configs/gen/latent_sota_XL_ot_8node.yaml
cfg=3.0

python inference_ours.py sample \
  --ckpt "$CKPT" \
  --config "$CONFIG" \
  --cfg-scale "$cfg" \
  --class-ids "207,829,387,386,360,817,95,927,263,698,483,88" \
  --num-rows 3 \
  --seed 42 \
  --save-path my_grid.png

