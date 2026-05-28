"""Latent cache dataset and cache builder for ImageNet release workflows.

Cache format (6 memory-mapped numpy files):
    cache_root/
        train_moments.npy       # (N, 32, 32, 4) float32
        train_moments_flip.npy  # (N, 32, 32, 4) float32
        train_targets.npy       # (N,) int64
        val_moments.npy
        val_moments_flip.npy
        val_targets.npy

Build:
    python -m dataset.latent --data-path /path/to/imagenet --target-path /path/to/cache

The dataset reads via np.load(mmap_mode='r') for zero-copy random access.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from utils.env import IMAGENET_CACHE_PATH, IMAGENET_PATH


# ---------------------------------------------------------------------------
# Dataset (reader)
# ---------------------------------------------------------------------------

class LatentDataset(Dataset):
    """Memory-mapped latent cache dataset.

    Accepts either:
      - ``root`` pointing to the cache directory + ``split`` name, OR
      - ``root`` pointing to ``{cache_root}/{split}`` (legacy-compatible call).

    The files must be named ``{split}_moments.npy``, etc.  When ``root``
    already contains files with the split prefix we use them directly;
    otherwise we look one level up.
    """

    def __init__(self, root: str, split: str | None = None):
        root = str(root)

        if split is not None:
            base, prefix = root, split
        else:
            base = os.path.dirname(root)
            prefix = os.path.basename(root)

        def _path(name: str) -> str:
            return os.path.join(base, f"{prefix}_{name}.npy")

        if not os.path.isfile(_path("moments")):
            base = root
            prefix = "train" if "train" in root else "val"

        self.moments: np.ndarray = np.load(_path("moments"), mmap_mode="r")
        self.moments_flip: np.ndarray = np.load(_path("moments_flip"), mmap_mode="r")
        self.targets: np.ndarray = np.load(_path("targets"), mmap_mode="r")
        assert len(self.moments) == len(self.targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        if torch.rand(1).item() < 0.5:
            m = np.array(self.moments[index])
        else:
            m = np.array(self.moments_flip[index])
        return m, int(self.targets[index])


# ---------------------------------------------------------------------------
# Cache builder helpers
# ---------------------------------------------------------------------------

def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def _center_crop_256(image: Image.Image) -> Image.Image:
    return center_crop_arr(image, 256)


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def create_cached_dataset(
    local_batch_size: int,
    target_path: str,
    data_path: str,
    *,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
) -> None:
    """Encode ImageNet images into VAE latents and write memory-mapped .npy files.

    Runs on a single GPU.  For multi-GPU encoding, launch separate processes
    that each handle a different split or shard and concatenate afterwards.
    """
    from dataset.vae import vae_enc_decode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encode_fn, _ = vae_enc_decode(replicate_params=False)

    Path(target_path).mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Lambda(_center_crop_256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    for split in ("train", "val"):
        split_dir = os.path.join(data_path, split)
        if not os.path.isdir(split_dir):
            print(f"Skipping {split}: {split_dir} does not exist")
            continue

        ds = datasets.ImageFolder(split_dir, transform=transform)
        n_samples = len(ds)
        print(f"[{split}] {n_samples} images")

        latent_shape = (32, 32, 4)

        moments_path = os.path.join(target_path, f"{split}_moments.npy")
        flip_path = os.path.join(target_path, f"{split}_moments_flip.npy")
        targets_path = os.path.join(target_path, f"{split}_targets.npy")

        mm_moments = np.lib.format.open_memmap(
            moments_path, mode="w+", dtype=np.float32,
            shape=(n_samples, *latent_shape),
        )
        mm_flip = np.lib.format.open_memmap(
            flip_path, mode="w+", dtype=np.float32,
            shape=(n_samples, *latent_shape),
        )
        mm_targets = np.lib.format.open_memmap(
            targets_path, mode="w+", dtype=np.int64,
            shape=(n_samples,),
        )

        loader = DataLoader(
            ds,
            batch_size=local_batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=(prefetch_factor if num_workers > 0 else None),
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )

        write_idx = 0
        for step, (images, labels) in tqdm(
            enumerate(loader), total=len(loader), desc=f"encode:{split}",
        ):
            images = images.to(device)

            with torch.no_grad():
                # Use identical RNG state for both normal and flipped encode
                # to match JAX's functional RNG semantics (same noise for both).
                rng = torch.Generator(device=device)
                rng.manual_seed(step)
                latents = encode_fn(images, rng=rng).detach().cpu().numpy()

                rng_flip = torch.Generator(device=device)
                rng_flip.manual_seed(step)
                latents_flip = encode_fn(
                    torch.flip(images, dims=(3,)), rng=rng_flip
                ).detach().cpu().numpy()

            bs = latents.shape[0]
            end_idx = write_idx + bs
            mm_moments[write_idx:end_idx] = latents
            mm_flip[write_idx:end_idx] = latents_flip
            mm_targets[write_idx:end_idx] = labels.numpy().astype(np.int64)
            write_idx = end_idx

        mm_moments.flush()
        mm_flip.flush()
        mm_targets.flush()
        del mm_moments, mm_flip, mm_targets

        print(f"[{split}] wrote {write_idx} samples to {target_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ImageNet latent cache (memory-mapped .npy files)."
    )
    parser.add_argument(
        "--data-path", default=IMAGENET_PATH,
        help="ImageNet root containing train/ and val/.",
    )
    parser.add_argument(
        "--target-path", default=IMAGENET_CACHE_PATH or "latent_cache",
        help="Output directory for .npy cache files.",
    )
    parser.add_argument(
        "--local-batch-size", type=int, default=128,
        help="Encoding batch size.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=8,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=2,
        help="DataLoader prefetch factor when num_workers > 0.",
    )
    parser.add_argument(
        "--pin-memory", action="store_true",
        help="Enable DataLoader pin_memory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    create_cached_dataset(
        local_batch_size=args.local_batch_size,
        target_path=args.target_path,
        data_path=args.data_path,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
    )


if __name__ == "__main__":
    main()
