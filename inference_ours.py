from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dataset.dataset import get_postprocess_fn
from models.generator import DitGen
from utils.dist_util import barrier, init_distributed, process_count, process_index
from utils.env import IMAGENET_FID_NPZ
from utils.misc import load_config, run_init

run_init()


def _print0(*args, **kwargs):
    if process_index() == 0:
        print(*args, **kwargs)


def _local_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _load_model(ckpt_path: str, config_path: str):
    """Load EMA model from a training checkpoint."""
    config = load_config(config_path)
    model_cfg = dict(config.model)
    if "num_classes" not in model_cfg and hasattr(config, "dataset"):
        model_cfg["num_classes"] = int(config.dataset.get("num_classes", 1000))

    model = DitGen(**model_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    step = ckpt.get("step", -1)

    ema_sd = ckpt.get("ema_model")
    if ema_sd is None:
        _print0("WARNING: no ema_model in checkpoint, falling back to model weights")
        ema_sd = ckpt.get("model", ckpt)

    missing, unexpected = model.load_state_dict(ema_sd, strict=False)
    if missing:
        _print0(f"WARNING: missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        _print0(f"WARNING: unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    device = _local_device()
    model = model.to(device).eval()

    postprocess_fn = get_postprocess_fn(
        use_aug=False,
        use_latent=bool(config.dataset.get("use_latent", False)),
        use_cache=bool(config.dataset.get("use_cache", False)),
    )

    _print0(f"Loaded EMA model from step {step} ({ckpt_path})")
    _print0(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, postprocess_fn, step, device


# ---------------------------------------------------------------------------
# Generation (multi-GPU aware)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_and_save(
    model, postprocess_fn, save_folder: str,
    *, num_samples: int, device_batch_size: int,
    cfg_scale: float, seed: int, device: torch.device,
):
    world_size = process_count()
    local_rank = process_index()

    if local_rank == 0:
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder, exist_ok=True)
    barrier()

    num_classes = getattr(model, "num_classes", 1000)
    assert num_samples % num_classes == 0, (
        f"num_samples ({num_samples}) must be divisible by num_classes ({num_classes})"
    )

    labels_all = np.arange(num_classes).repeat(num_samples // num_classes)
    pad = world_size * device_batch_size
    labels_all = np.concatenate([labels_all, np.zeros(pad, dtype=labels_all.dtype)])

    num_steps = (num_samples + world_size * device_batch_size - 1) // (
        world_size * device_batch_size
    )

    pbar = tqdm(range(num_steps), desc="Generating", disable=(local_rank != 0))
    for step_i in pbar:
        global_start = step_i * world_size * device_batch_size
        rank_start = global_start + local_rank * device_batch_size
        rank_end = rank_start + device_batch_size

        batch_labels = torch.from_numpy(
            labels_all[rank_start:rank_end]
        ).long().to(device)

        sample_indices = rank_start + torch.arange(device_batch_size)
        rng = torch.Generator(device=device)
        rng.manual_seed(seed ^ int(sample_indices[0].item()))

        latent_samples = model(
            c=batch_labels, cfg_scale=cfg_scale,
            deterministic=True, train=False, rng=rng,
        )["samples"]

        pixel_images = postprocess_fn(latent_samples)
        pixel_np = pixel_images.detach().cpu().float().numpy()
        pixel_np = np.clip(pixel_np, 0.0, 1.0)

        for b in range(device_batch_size):
            img_id = int(sample_indices[b].item())
            if img_id >= num_samples:
                break
            img_hwc = (pixel_np[b].transpose(1, 2, 0) * 255).round().astype(np.uint8)
            Image.fromarray(img_hwc).save(
                os.path.join(save_folder, f"{img_id:05d}.png")
            )

    barrier()


# ---------------------------------------------------------------------------
# Sample mode -- preview grid
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_sample(
    model, postprocess_fn, *,
    class_ids: list[int], cfg_scale: float,
    seed: int, num_rows: int, save_path: str, device: torch.device,
):
    labels = torch.tensor(class_ids, dtype=torch.long, device=device)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    latent_samples = model(
        c=labels, cfg_scale=cfg_scale,
        deterministic=True, train=False, rng=rng,
    )["samples"]

    pixel_images = postprocess_fn(latent_samples)
    imgs = pixel_images.detach().cpu().float().numpy()
    imgs = np.clip(imgs, 0.0, 1.0)
    imgs = (imgs.transpose(0, 2, 3, 1) * 255).round().astype(np.uint8)

    n = len(imgs)
    num_cols = (n + num_rows - 1) // num_rows
    h, w, c = imgs.shape[1], imgs.shape[2], imgs.shape[3]

    grid = np.zeros((num_rows * h, num_cols * w, c), dtype=np.uint8)
    for idx in range(n):
        r, col = divmod(idx, num_cols)
        grid[r * h : (r + 1) * h, col * w : (col + 1) * w, :] = imgs[idx]

    out = Path(save_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out)
    print(f"Saved {num_rows}x{num_cols} grid ({n} images) to {out}")


# ---------------------------------------------------------------------------
# Evaluate mode -- FID / ISC
# ---------------------------------------------------------------------------

def run_eval(
    model, postprocess_fn, ckpt_path: str, ckpt_step: int,
    workdir: str, *, num_samples: int, cfg_scale: float,
    gen_bsz: int, fid_ref: str, seed: int,
    keep_samples: bool, device: torch.device,
) -> dict | None:
    calculate_metrics = None
    if process_index() == 0:
        from utils.fidelity_wrapper import calculate_metrics

    save_folder = os.path.join(workdir, "fid_outputs")

    t0 = time.time()
    generate_and_save(
        model, postprocess_fn, save_folder,
        num_samples=num_samples, device_batch_size=gen_bsz,
        cfg_scale=cfg_scale, seed=seed, device=device,
    )
    gen_time = time.time() - t0
    _print0(f"Generation done in {gen_time:.1f}s")

    result = None
    if process_index() == 0:
        _print0("Computing metrics via torch-fidelity (inception-v3-compat) ...")
        metrics_dict = calculate_metrics(
            input1=save_folder, input2=fid_ref,
            cuda=True, isc=True, fid=True, kid=False, prc=False, verbose=True,
        )

        fid = metrics_dict.get("frechet_inception_distance")
        isc_mean = metrics_dict.get("inception_score_mean")
        isc_std = metrics_dict.get("inception_score_std")
        _print0(f"FID: {fid}")
        _print0(f"Inception Score: {isc_mean} +/- {isc_std}")

        result = {
            "ckpt": ckpt_path,
            "step": ckpt_step,
            "cfg_scale": cfg_scale,
            "fid": fid,
            "isc_mean": isc_mean,
            "isc_std": isc_std,
            "gen_time": gen_time,
        }

        if not keep_samples:
            shutil.rmtree(save_folder)

    barrier()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference from our training checkpoints (state_*.pt)."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--ckpt", required=True,
        help="Path to state_*.pt checkpoint file.",
    )
    shared.add_argument(
        "--config", required=True,
        help="Path to training config YAML (for model architecture).",
    )
    shared.add_argument("--cfg-scale", type=float, default=1.0)
    shared.add_argument("--seed", type=int, default=0)
    shared.add_argument("--workdir", default="runs/infer_ours")

    sp = sub.add_parser("sample", parents=[shared], help="Generate a preview grid.")
    sp.add_argument(
        "--class-ids", type=str,
        default="207,360,387,974,88,979,417,279",
    )
    sp.add_argument("--num-rows", type=int, default=2)
    sp.add_argument("--save-path", type=str, default="")

    ep = sub.add_parser("evaluate", parents=[shared], help="Generate 50k images and compute FID.")
    ep.add_argument("--num-samples", type=int, default=50000)
    ep.add_argument("--gen-bsz", type=int, default=64)
    ep.add_argument("--fid-ref", type=str, default=IMAGENET_FID_NPZ)
    ep.add_argument("--json-out", type=str, default="")
    ep.add_argument("--keep-samples", action="store_true")

    return parser


def main() -> None:
    init_distributed()
    args = build_parser().parse_args()

    if process_index() == 0:
        os.makedirs(args.workdir, exist_ok=True)
    barrier()

    model, postprocess_fn, ckpt_step, device = _load_model(args.ckpt, args.config)

    if args.mode == "sample":
        if process_index() == 0:
            class_ids = [int(x.strip()) for x in args.class_ids.split(",") if x.strip()]
            save_path = args.save_path or os.path.join(args.workdir, "sample_grid.png")
            run_sample(
                model, postprocess_fn,
                class_ids=class_ids, cfg_scale=args.cfg_scale,
                seed=args.seed, num_rows=args.num_rows,
                save_path=save_path, device=device,
            )
        barrier()

    elif args.mode == "evaluate":
        result = run_eval(
            model, postprocess_fn, args.ckpt, ckpt_step,
            args.workdir,
            num_samples=args.num_samples, cfg_scale=args.cfg_scale,
            gen_bsz=args.gen_bsz, fid_ref=args.fid_ref,
            seed=args.seed, keep_samples=args.keep_samples,
            device=device,
        )
        if result is not None:
            print(json.dumps(result, indent=2))
            if args.json_out:
                out = Path(args.json_out).resolve()
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
