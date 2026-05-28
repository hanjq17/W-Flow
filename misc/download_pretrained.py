from huggingface_hub import snapshot_download
from utils.env import (
    HF_REPO_ID, HF_ROOT, VAE_HF_PATH, TORCH_HUB_DIR, WFLOW_HF_REPO_ID, WFLOW_HF_ROOT,
)
import os
import torch


##################### Inference Related #####################
"""
Download Pretrained VAE
"""
repo_id = "stabilityai/sd-vae-ft-mse"
snapshot_download(
    repo_id=repo_id,
    local_dir=VAE_HF_PATH,
)

print(f"Downloaded pretrained VAE weights from {repo_id} to {VAE_HF_PATH}")

"""
Download Pretrained W-Flow
"""
snapshot_download(
    repo_id=WFLOW_HF_REPO_ID,
    local_dir=WFLOW_HF_ROOT,
)

print(f"Downloaded pretrained W-Flow weights from {WFLOW_HF_REPO_ID} to {WFLOW_HF_ROOT}")


##################### FID Calculation Related #####################
"""
Download Pretrained InceptionNet for FID calculation
"""
# Match utils/fid_util.py and utils/fidelity_wrapper.py
os.makedirs(TORCH_HUB_DIR, exist_ok=True)
torch.hub.set_dir(TORCH_HUB_DIR)

from torch_fidelity.utils import create_feature_extractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Match current repo:
# - feature extractor: inception-v3-compat
# - FID layer: 2048
# - ISC layer: logits_unbiased
fe = create_feature_extractor(
    "inception-v3-compat",
    ["2048", "logits_unbiased"],
    cuda=(device.type == "cuda"),
)
fe.eval()

# torch-fidelity inception-v3-compat expects NCHW uint8 images in [0, 255].
x = torch.zeros(1, 3, 256, 256, dtype=torch.uint8, device=device)
with torch.no_grad():
    features_2048, logits = fe(x)
print("Downloaded/loaded torch-fidelity Inception successfully.")
print("torch hub dir:", torch.hub.get_dir())
print("FID feature shape:", tuple(features_2048.shape))
print("logits shape:", tuple(logits.shape))

print(f"Downloaded pretrained InceptionNet for FID calculation to {TORCH_HUB_DIR}")


##################### Training Related #####################
"""
Download Pretrained MAE from Drifting
"""
snapshot_download(
    repo_id=HF_REPO_ID,
    local_dir=HF_ROOT,
)

print(f"Downloaded pretrained MAE weights from {HF_REPO_ID} to {HF_ROOT}")
