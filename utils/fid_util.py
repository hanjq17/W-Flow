from __future__ import annotations

import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
from scipy import linalg

from dataset.dataset import epoch0_sampler
from utils.dist_util import process_allgather, process_count, process_index
from utils.env import IMAGENET_FID_NPZ, IMAGENET_PR_NPZ, TORCH_HUB_DIR
from utils.logging import is_rank_zero
from tqdm import tqdm

if os.path.isdir(TORCH_HUB_DIR) or not os.path.exists(TORCH_HUB_DIR):
    os.makedirs(TORCH_HUB_DIR, exist_ok=True)
    torch.hub.set_dir(TORCH_HUB_DIR)

INCEPTION_NET = None
_DATASET_STATS = {
    "imagenet256": IMAGENET_FID_NPZ,
}
_PR_REF_PATH = IMAGENET_PR_NPZ


def _canonical_dataset_name(name: str) -> str:
    n = name.lower()
    if "imagenet256" in n:
        return "imagenet256"
    raise ValueError(f"Only ImageNet is supported now, got: {name}")


class _InceptionWrap:
    """Wraps torch_fidelity's TF-compatible Inception V3 (inception-v3-compat).

    This MUST match the Inception used to compute the reference FID statistics.
    Using torchvision's Inception V3 instead would give wrong FID values.
    """

    def __init__(self, device: torch.device):
        self.device = device
        from torch_fidelity.utils import create_feature_extractor
        self.fe = create_feature_extractor(
            "inception-v3-compat",
            ["2048", "logits_unbiased"],
            cuda=(device.type == "cuda"),
        )
        self.fe.eval()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 3, H, W) float tensor in [0, 255]. Resize handled internally."""
        out = self.fe(x)
        return out[0], out[1]


def _get_inception() -> _InceptionWrap:
    global INCEPTION_NET
    if INCEPTION_NET is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        INCEPTION_NET = _InceptionWrap(device)
    return INCEPTION_NET


def _to_uint8(samples):
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=0.0)
    return (samples * 255).clip(0, 255).astype(np.uint8)


def _extract_inception_features(
    samples_uint8: np.ndarray,
    *,
    compute_logits: bool = False,
    batch_size: int = 200,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Run TF-compatible Inception on uint8 images, return (features_f64, logits_or_None)."""
    net = _get_inception()
    if samples_uint8.ndim == 4 and samples_uint8.shape[-1] != 3:
        samples_uint8 = samples_uint8.transpose(0, 2, 3, 1)

    feats_list = []
    logits_list = []
    for i in range(0, len(samples_uint8), batch_size):
        x = torch.from_numpy(samples_uint8[i : i + batch_size]).to(net.device)
        x = x.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW, uint8 [0, 255]
        feats, logits = net(x)
        feats_list.append(feats.detach().cpu().numpy())
        if compute_logits:
            logits_list.append(logits.detach().cpu().numpy())

    features = np.concatenate(feats_list, axis=0).astype(np.float64)
    logits = np.concatenate(logits_list, axis=0) if logits_list else None
    return features, logits


def _compute_inception_score(logits, splits=10):
    rng = np.random.RandomState(2020)
    logits = logits[rng.permutation(logits.shape[0]), :]
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy().astype(np.float64)

    n = probs.shape[0]
    split_size = n // splits
    probs = probs[: split_size * splits]
    scores = []
    for i in range(splits):
        part = probs[i * split_size : (i + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    scores = np.asarray(scores, dtype=np.float64)
    return float(np.mean(scores)), float(np.std(scores))


def _load_ref_stats(dataset_name: str):
    canon = _canonical_dataset_name(dataset_name)
    path = _DATASET_STATS[canon]
    data = np.load(path)
    if "ref_mu" in data:
        return {"mu": data["ref_mu"], "sigma": data["ref_sigma"]}
    return {"mu": data["mu"], "sigma": data["sigma"]}


def _compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def _pairwise_distances(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    norm_u = np.sum(np.square(U), axis=1, keepdims=True)
    norm_v = np.sum(np.square(V), axis=1, keepdims=True).T
    return np.maximum(norm_u - 2 * np.matmul(U, V.T) + norm_v, 0.0)


def _manifold_radii(features: np.ndarray, k: int = 3, row_batch_size: int = 10000, col_batch_size: int = 10000) -> np.ndarray:
    num_images = len(features)
    radii = np.zeros([num_images, 1], dtype=np.float64)
    distance_batch = np.zeros([row_batch_size, num_images], dtype=np.float64)
    seq = np.arange(k + 1, dtype=np.int32)

    for begin1 in range(0, num_images, row_batch_size):
        end1 = min(begin1 + row_batch_size, num_images)
        row_batch = features[begin1:end1]
        for begin2 in range(0, num_images, col_batch_size):
            end2 = min(begin2 + col_batch_size, num_images)
            col_batch = features[begin2:end2]
            distance_batch[0 : end1 - begin1, begin2:end2] = _pairwise_distances(row_batch, col_batch)
        part = np.partition(distance_batch[0 : end1 - begin1, :], kth=seq, axis=1)
        radii[begin1:end1, 0] = part[:, k]
    return radii


def _evaluate_manifold(features_ref: np.ndarray, radii_ref: np.ndarray, eval_features: np.ndarray, row_batch_size: int = 10000, col_batch_size: int = 10000) -> np.ndarray:
    num_eval = len(eval_features)
    num_ref = len(features_ref)
    distance_batch = np.zeros([row_batch_size, num_ref], dtype=np.float64)
    preds = np.zeros([num_eval, 1], dtype=np.int32)

    for begin1 in range(0, num_eval, row_batch_size):
        end1 = min(begin1 + row_batch_size, num_eval)
        f_batch = eval_features[begin1:end1]
        for begin2 in range(0, num_ref, col_batch_size):
            end2 = min(begin2 + col_batch_size, num_ref)
            r_batch = features_ref[begin2:end2]
            distance_batch[0 : end1 - begin1, begin2:end2] = _pairwise_distances(f_batch, r_batch)
        samples_in = distance_batch[0 : end1 - begin1, :, None] <= radii_ref
        preds[begin1:end1] = np.any(samples_in, axis=1).astype(np.int32)

    return preds


def _compute_precision_recall(features_real: np.ndarray, features_fake: np.ndarray, k: int = 3) -> Tuple[float, float]:
    features_real = np.asarray(features_real, dtype=np.float64)
    features_fake = np.asarray(features_fake, dtype=np.float64)
    radii_real = _manifold_radii(features_real, k=k)
    radii_fake = _manifold_radii(features_fake, k=k)

    fake_in_real = _evaluate_manifold(features_real, radii_real, features_fake)
    real_in_fake = _evaluate_manifold(features_fake, radii_fake, features_real)

    precision = float(np.mean(fake_in_real.astype(np.float64), axis=0)[0])
    recall = float(np.mean(real_in_fake.astype(np.float64), axis=0)[0])
    return precision, recall


def evaluate_fid(
    dataset_name,
    gen_func,
    gen_params,
    eval_loader,
    logger,
    num_samples=5000,
    log_folder="fid",
    log_prefix="gen_model",
    eval_prc_recall=False,
    eval_isc=True,
    eval_fid=True,
    rng_eval=None,
):
    """Distributed FID evaluation.

    All ranks generate from their eval_loader shard in parallel, extract
    Inception features locally, then all_gather features.  Rank 0 computes
    the final FID / IS / PRC metrics.
    """
    if rng_eval is None:
        rng_eval = 0

    start = time.time()
    world = process_count()
    rank = process_index()
    samples_per_rank = (num_samples + world - 1) // world

    # --- 1. Each rank generates from its eval_loader shard ---
    eval_iter = epoch0_sampler(eval_loader)
    all_samples = []
    cur = 0
    it = tqdm(enumerate(eval_iter), desc="FID gen") if is_rank_zero() else enumerate(eval_iter)
    for i, batch in it:
        gen_samples = gen_func(batch, **gen_params, rng=int(rng_eval) + i)

        if torch.is_tensor(gen_samples):
            local_samples = gen_samples.detach().float().cpu().numpy()
        else:
            local_samples = np.asarray(gen_samples)

        all_samples.append(_to_uint8(local_samples))
        cur += local_samples.shape[0]
        if cur >= samples_per_rank:
            break

    local_images = np.concatenate(all_samples, axis=0)[:samples_per_rank]

    # --- 2. Each rank extracts Inception features locally ---
    local_feats, local_logits = _extract_inception_features(
        local_images, compute_logits=eval_isc,
    )

    # --- 3. All_gather features (small: ~50 MB/rank for 6250 images) ---
    feats_t = torch.from_numpy(local_feats.astype(np.float32))
    all_feats = process_allgather(feats_t).numpy()[:num_samples].astype(np.float64)

    all_logits = None
    if eval_isc and local_logits is not None:
        logits_t = torch.from_numpy(local_logits.astype(np.float32))
        all_logits = process_allgather(logits_t).numpy()[:num_samples]

    all_prc_feats = None
    if eval_prc_recall:
        all_prc_feats = all_feats

    # --- 4. Rank 0 computes metrics ---
    metrics: Dict[str, float] = {}
    if is_rank_zero():
        ref = _load_ref_stats(dataset_name)

        mu = np.mean(all_feats, axis=0)
        sigma = np.cov(all_feats, rowvar=False)

        if eval_fid:
            metrics["fid"] = _compute_frechet_distance(ref["mu"], ref["sigma"], mu, sigma)
        if eval_isc and all_logits is not None:
            mean, std = _compute_inception_score(all_logits)
            metrics["isc_mean"] = mean
            metrics["isc_std"] = std
        if eval_prc_recall and all_prc_feats is not None:
            ref_images = np.load(_PR_REF_PATH)["arr_0"].astype(np.uint8)
            ref_feats, _ = _extract_inception_features(ref_images, compute_logits=False)
            precision, recall = _compute_precision_recall(ref_feats, all_prc_feats, k=3)
            metrics["precision"] = float(precision)
            metrics["recall"] = float(recall)

        metrics["fid_time"] = float(time.time() - start)
        logger.log_dict({f"{log_folder}/{log_prefix}_{k}": v for k, v in metrics.items()})
        logger.log_image(f"{log_folder}/{log_prefix}_viz", local_images[:64])

    return metrics
