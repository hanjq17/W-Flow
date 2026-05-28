########### misc/download_pretrained.py ###########
# Download pretrained W-Flow checkpoint and SD-VAE weights for inference
# Download InceptionNet for FID eval
# Download pretrained MAE weights for training

# 1. Specify the path to the pretrained W-Flow and SD-VAE weights:
# "WFLOW_HF_ROOT" and "VAE_HF_PATH" in utils/env.py
# 2. Specify the root path to InceptionNet for FID calculation:
# "TORCH_HUB_DIR" in utils/env.py
# 3. Specify the path to the pretrained MAE weights:
# "HF_ROOT" in utils/env.py
# 4. Run the following code to download them all:
python misc/download_pretrained.py

########### dataset/latent.py ###########
# 1. Download imagenet-1k and specify the path "IMAGENET_PATH" in utils/env.py
# 2. Specify the path to the latent cache "IMAGENET_CACHE_PATH" in utils/env.py
# 3. Tokenize the imagenet-1k dataset and save the latent cache:
python -m dataset.latent

########### Additional FID Calculation Related ###########
# 1. Download ImageNet precomputed statistics and InceptionNet for evaluation
# Specify the path to the precomputed statistics:
# "IMAGENET_FID_NPZ" and "IMAGENET_PR_NPZ" in utils/env.py
# IMAGENET_FID_NPZ is provided in our huggingface repo as:
# "stats/jit_in256_stats.npz", which is directly copied from JIT:
# https://github.com/LTH14/JiT/blob/main/fid_stats/jit_in256_stats.npz
