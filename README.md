<div align=center>

# W-Flow: One-Step Generative Modeling via<br>Wasserstein Gradient Flows

Jiaqi Han* $^1$, Puheng Li* $^1$, Qiushan Guo $^2$,

Renyuan Xu† $^1$, Stefano Ermon† $^1$, Emmanuel J. Candès† $^1$

**$^1$ Stanford University**   **$^2$ ByteDance**

*Equal Contribution  † Equal Advising

<p>
<a href='https://arxiv.org/abs/2605.11755'><img src='https://img.shields.io/static/v1?&logo=arxiv&label=Paper&message=Arxiv:W-Flow&color=B31B1B'></a>
<a href='https://hanjq17.github.io/W-Flow/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href="https://huggingface.co/jiaqihan99/W-Flow"><img src="https://img.shields.io/badge/HuggingFace-W--Flow-yellow.svg" alt="HuggingFace" /></a>
  <a href="https://colab.research.google.com/github/hanjq17/W-Flow/blob/main/notebooks/visualization_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" /></a>
</p>

</div>

Official PyTorch codebase for **One-Step Generative Modeling via Wasserstein Gradient Flows**.
W-Flow trains a generator to map a simple reference distribution to the data distribution in a single forward pass.
The training dynamics are guided by a Wasserstein gradient flow of the Sinkhorn divergence, yielding globally coordinated optimal-transport updates and 1-NFE ImageNet generation.

<p align="center">
  <img src="assets/wflow-samples1_cropped.png" width="90%" alt="W-Flow ImageNet samples" />
</p>

## 📚 Table of Contents

<!-- - [✨ Highlights](#highlights)
- [🤗 Pretrained Checkpoints](#pretrained-checkpoints)
- [🚀 Quick Start](#quick-start)
- [🛠 Environment Setup](#environment-setup)
- [⚙️ Path Configuration](#path-configuration)
- [🗂️ Dataset and Latent Cache](#dataset-and-latent-cache)
- [🖼️ Sampling](#sampling)
- [📊 FID Evaluation](#fid-evaluation)
- [🏋️ Training](#training)
- [📦 Checkpoint Format](#checkpoint-format)
- [📁 Repository Layout](#repository-layout)
- [📌 Citation](#citation) -->
- [Highlights](#highlights)
- [Pretrained Checkpoints](#pretrained-checkpoints)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Path Configuration](#path-configuration)
- [Dataset and Latent Cache](#dataset-and-latent-cache)
- [Sampling](#sampling)
- [FID Evaluation](#fid-evaluation)
- [Training](#training)
- [Checkpoint Format](#checkpoint-format)
- [Repository Layout](#repository-layout)
- [Citation](#citation)
<!-- - [Contact](#contact) -->

## ✨ Highlights

- **One-step sampling:** W-Flow generates ImageNet 256×256 samples with one neural network evaluation.
- **Strong ImageNet performance:** W-Flow achieves 1.52 FID with B/2, 1.35 FID with L/2, and 1.29 FID with XL/2 on class-conditional ImageNet 256×256 generation.
- **Wasserstein gradient flow dynamics:** Generated particles are updated by the steepest descent direction of the Sinkhorn divergence.
- **Debiased OT updates:** The implementation uses generated-to-real and generated-to-generated Sinkhorn barycentric maps, including the two-batch self-transport estimator used in the paper.
- **Training from scratch:** The generator is not distilled from a multi-step teacher.
- **PyTorch release:** This repository contains the PyTorch training, sampling, and FID evaluation code for the W-Flow ImageNet latent models.

## 🤗 Pretrained Checkpoints

Pretrained W-Flow checkpoints are hosted at [`jiaqihan99/W-Flow`](https://huggingface.co/jiaqihan99/W-Flow).
The expected local layout after download is:

```text
WFLOW_HF_ROOT/
└── checkpoints/
    ├── latent_ablation_ot/
    │   └── state_00030000.pt
    ├── latent_sota_B_ot/
    │   └── state_00200000.pt
    ├── latent_sota_L_ot/
    │   └── state_00200000.pt
    └── latent_sota_XL_ot/
        └── state_00180000.pt
```

| Model | Config | Checkpoint | CFG $w+1$ | Paper FID | Paper IS |
| --- | --- | --- | ---: | ---: | ---: |
| W-Flow B/2 | `configs/gen/latent_sota_B_ot_8node.yaml` | `checkpoints/latent_sota_B_ot/state_00200000.pt` | 1.19 | 1.52 | 271.8 |
| W-Flow L/2 | `configs/gen/latent_sota_L_ot_8node.yaml` | `checkpoints/latent_sota_L_ot/state_00200000.pt` | 1.14 | 1.35 | 272.5 |
| W-Flow XL/2 | `configs/gen/latent_sota_XL_ot_8node.yaml` | `checkpoints/latent_sota_XL_ot/state_00180000.pt` | 1.09 | 1.29 | 265.4 |
| W-Flow ablation | `configs/gen/ablation_ot_1node.yaml` | `checkpoints/latent_ablation_ot/state_00030000.pt` | 1.5 | 7.08 | - |

Feature extractors for training are loaded from the original Drifting model release at [`Goodeat/drifting`](https://huggingface.co/Goodeat/drifting), e.g. `hf://mae_latent_640`.
The latent VAE is [`stabilityai/sd-vae-ft-mse`](https://huggingface.co/stabilityai/sd-vae-ft-mse).

**Note:** The "CFG scale" specified throughout this codebase corresponds to $(w + 1)$ in our paper, where $w\geq 0$ is the actual CFG scale.

## 🚀 Quick Start

The easiest way to generate samples is the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hanjq17/W-Flow/blob/main/notebooks/visualization_demo.ipynb)

Default notebook settings:

```python
CODE_REPO_URL = "https://github.com/hanjq17/W-Flow"
CODE_REPO_REF = "main"

WFLOW_HF_REPO_ID = "jiaqihan99/W-Flow"
WFLOW_HF_ROOT = "/content/wflow_hf_root"
DOWNLOAD_WFLOW_CHECKPOINTS = True

VAE_HF_REPO_ID = "stabilityai/sd-vae-ft-mse"
VAE_HF_PATH = "/content/sdvae_hf_root"
DOWNLOAD_VAE_WEIGHTS = True
```

The notebook downloads both the W-Flow checkpoint repo and the SD-VAE decoder. For local sampling after downloading checkpoints, make sure `VAE_HF_PATH` in `utils/env.py` points to a local copy of `stabilityai/sd-vae-ft-mse`:

```bash
export WFLOW_HF_ROOT=/path/to/wflow_hf_root

python inference_ours.py sample \
  --ckpt "$WFLOW_HF_ROOT/checkpoints/latent_sota_XL_ot/state_00180000.pt" \
  --config configs/gen/latent_sota_XL_ot_8node.yaml \
  --cfg-scale 3.0 \
  --class-ids "207,829,387,386,360,817,95,927,263,698,483,88" \
  --num-rows 3 \
  --seed 42 \
  --save-path my_grid.png
```

<p align="center">
  <img src="assets/my_grid.png" width="80%" alt="W-Flow visualization grid" />
</p>

## 🛠 Environment Setup

Create an environment and install dependencies:

```bash
conda create -n wflow python=3.10 -y
conda activate wflow
pip install -r requirements.txt
```

The helper script is equivalent:

```bash
bash install_env.sh
```

For FID evaluation, `torch-fidelity` downloads the Inception feature extractor into `TORCH_HUB_DIR`.

## ⚙️ Path Configuration

Before training or evaluation, edit `utils/env.py`:

```python
HF_REPO_ID = "Goodeat/drifting"
HF_ROOT = "/path/to/drifting_hf_root"
VAE_HF_PATH = "/path/to/sdvae_hf_root"
TORCH_HUB_DIR = "/path/to/torch_hub"

WFLOW_HF_REPO_ID = "jiaqihan99/W-Flow"
WFLOW_HF_ROOT = "/path/to/wflow_hf_root"

IMAGENET_PATH = "/path/to/imagenet-1k"
IMAGENET_CACHE_PATH = "/path/to/imagenet256-latents-sdvae"
IMAGENET_FID_NPZ = "/path/to/jit_in256_stats.npz"
IMAGENET_PR_NPZ = "/path/to/imagenet_val_prc_arr0.npz"
```

Then download model assets:

```bash
python misc/download_pretrained.py
```

This downloads:

- W-Flow checkpoints from `WFLOW_HF_REPO_ID` into `WFLOW_HF_ROOT`.
- SD-VAE weights into `VAE_HF_PATH`.
- MAE feature extractors from `Goodeat/drifting` into `HF_ROOT`.
- The torch-fidelity Inception network into `TORCH_HUB_DIR`.


## 🗂️ Dataset and Latent Cache

Download ImageNet-1k and arrange it as:

```text
imagenet-1k/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

Build the SD-VAE latent cache:

```bash
python -m dataset.latent \
  --data-path /path/to/imagenet-1k \
  --target-path /path/to/imagenet256-latents-sdvae \
  --local-batch-size 128 \
  --num-workers 8 \
  --pin-memory
```

The cache builder writes memory-mapped NumPy files:

```text
IMAGENET_CACHE_PATH/
├── train_moments.npy
├── train_moments_flip.npy
├── train_targets.npy
├── val_moments.npy
├── val_moments_flip.npy
└── val_targets.npy
```

Set `IMAGENET_CACHE_PATH` in `utils/env.py` to this directory before training.

**Note:** `prepare.sh` collects the asset download and latent-cache build commands in one place. After setting the paths above and downloading ImageNet, you can run:

```bash
bash prepare.sh
```


## 🖼️ Sampling

Use `inference_ours.py sample` for single-GPU or CPU preview grids:

```bash
python inference_ours.py sample \
  --ckpt /path/to/state_00180000.pt \
  --config configs/gen/latent_sota_XL_ot_8node.yaml \
  --cfg-scale 3.0 \
  --class-ids "207,829,387,386,360,817,95,927,263,698,483,88" \
  --num-rows 3 \
  --seed 42 \
  --save-path my_grid.png
```

You can also edit and run:

```bash
bash scripts/sample/visualization.sh
```

## 📊 FID Evaluation

FID/IS evaluation generates 50K images with `torchrun` and computes metrics with torch-fidelity.
Set `WFLOW_HF_ROOT` inside the script or replace it with your local path, then run:

```bash
bash scripts/eval_fid/latent_sota_B_ot.sh
bash scripts/eval_fid/latent_sota_L_ot.sh
bash scripts/eval_fid/latent_sota_XL_ot.sh
bash scripts/eval_fid/ablation_ot.sh
```

Equivalent direct command:

```bash
torchrun --nproc_per_node=8 inference_ours.py evaluate \
  --ckpt "$WFLOW_HF_ROOT/checkpoints/latent_sota_XL_ot/state_00180000.pt" \
  --config configs/gen/latent_sota_XL_ot_8node.yaml \
  --cfg-scale 1.09 \
  --num-samples 50000 \
  --gen-bsz 64 \
  --fid-ref /path/to/jit_in256_stats.npz \
  --workdir runs/latent_sota_XL_ot/ckpt_00180000 \
  --json-out results/latent_sota_XL_ot/ckpt_00180000/results.json
```

`IMAGENET_FID_NPZ` should be set in `utils/env.py` and point to the ImageNet 256×256 FID statistics. The released W-Flow checkpoint repo provides `stats/jit_in256_stats.npz`, copied from the JiT FID statistics.

## 🏋️ Training

This release trains class-conditional ImageNet 256×256 generators in SD-VAE latent space with DiT-style architectures and pretrained latent-MAE feature encoders.

Single-node scripts:

```bash
bash scripts/train/ablation_ot.sh
```

The SOTA scripts contain both a commented single-node block and an active multi-node block:

```bash
bash scripts/train/latent_sota_B_ot.sh
bash scripts/train/latent_sota_L_ot.sh
bash scripts/train/latent_sota_XL_ot.sh
```

For multi-node training, launch the same script on each node and set:

```bash
export NGPU=8
export NNODES=8
export NODE_RANK=<0..7>
export MASTER_ADDR=<rank-0-host>
export MASTER_PORT=6667
```

You can also call the trainer directly:

```bash
torchrun --nproc_per_node=8 train.py \
  --config configs/gen/latent_sota_B_ot_1node.yaml \
  --workdir /path/to/workdir/latent_sota_B_ot_1node
```

- The training scripts set `DRIFT_COMPILE=1` by default to enable `torch.compile` for the generator and feature/loss computations when available in order to speedup the training once compiled. If compilation causes long startup time or compatibility issues on your machine, disable it with `DRIFT_COMPILE=0`.

- During training, checkpoints are written under `<workdir>/checkpoints/`, and periodic FID preview evaluation is controlled by `train.eval_per_step` and `train.cfg_list` in the config. Note that the FID metrics logged during training are just for reference; refer to "📊 FID Evaluation" for computing the precise FID metrics.

- The config field `train.ot_mode` selects the loss implementation: `"debiased"` uses the W-Flow OT loss, while `"none"` uses the original drifting loss in `drift_loss.py`, which could serve as an unofficial PyTorch implementation of drifting models. Note that we never report results obtained by this implementation of drifting model in our paper; we always cite the results reported in their original paper.


Slurm launch examples are also available:

```bash
sbatch scripts/train_slurm/ablation_ot_slurm.sh
sbatch scripts/train_slurm/latent_sota_B_ot_slurm.sh
sbatch scripts/train_slurm/latent_sota_L_ot_slurm.sh
sbatch scripts/train_slurm/latent_sota_XL_ot_slurm.sh
```

## 📦 Checkpoint Format

Training checkpoints are PyTorch files:

```text
<workdir>/
├── checkpoints/
│   ├── state_00002000.pt
│   └── ...
└── params_ema/
    ├── ema_params.pt
    └── metadata.json
```

`state_*.pt` contains:

- `step`
- `model`
- `ema_model`
- `optimizer`
- `ema_decay`

`inference_ours.py` loads `ema_model` from `state_*.pt` by default and falls back to `model` if EMA weights are not present.

## 📁 Repository Layout

```text
configs/gen/              # W-Flow ImageNet configs
dataset/                  # ImageNet and latent-cache datasets
models/                   # DiT generator and MAE feature encoder
scripts/train/            # Training launch scripts
scripts/train_slurm/      # Slurm training launch scripts
scripts/eval_fid/         # FID evaluation scripts
scripts/sample/           # Visualization script
utils/                    # Checkpointing, distributed, FID, logging, setup utilities
notebooks/                # Interactive visualization notebook
```

## 📌 Citation

```bibtex
@article{han2026one,
  title={One-Step Generative Modeling via Wasserstein Gradient Flows},
  author={Han, Jiaqi and Li, Puheng and Guo, Qiushan and Xu, Renyuan and Ermon, Stefano and Cand{\`e}s, Emmanuel J},
  journal={arXiv preprint arXiv:2605.11755},
  year={2026}
}
```

## 📬 Contact

For questions about the paper or codebase, please contact Jiaqi Han (`jiaqihan@stanford.edu`) and Puheng Li (`puhengli@stanford.edu`).

## 🗒️ Acknowledgments

This PyTorch implementation builds on the [Drifting Models code](https://github.com/lambertae/drifting) in JAX and the pretrained feature extractors from [`Goodeat/drifting`](https://huggingface.co/Goodeat/drifting). The evaluation code is largely based on [Pixel Mean Flows](https://github.com/Lyy-iiis/pMF/tree/torch).
We sincerely thank the authors for open-sourcing their codebase.
