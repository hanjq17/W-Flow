from __future__ import annotations

import argparse
import copy
import gc
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from einops import rearrange, repeat
from tqdm import tqdm

from dataset.dataset import get_postprocess_fn, infinite_sampler
from drift_loss import drift_loss
from drift_loss_ot import drift_loss_ot
from memory_bank import ArrayMemoryBank
from models.mae_model import build_activation_function
from utils.ckpt_util import restore_checkpoint, save_checkpoint, save_params_ema_artifact
from utils.dist_util import barrier, maybe_ddp_model, process_count, process_index, unwrap_ddp
from utils.env import HF_ROOT
from utils.fid_util import evaluate_fid
from utils.init_util import maybe_init_state_params
from utils.logging import is_rank_zero, log_for_0
from utils.misc import load_config, profile_func, run_init, seed_everything
from utils.model_builder import build_model_dict


run_init()


@dataclass
class TrainState:
    step: int
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    ema_model: torch.nn.Module
    ema_decay: float


def _generator_model_config(model) -> dict:
    model = unwrap_ddp(model)
    return {
        name: value
        for name, value in vars(model).items()
        if name not in {"parent", "name"} and not name.startswith("_")
    }


def _set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = float(lr)


@torch.no_grad()
def _update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, ema_decay: float):
    model = unwrap_ddp(model)
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(ema_decay).add_(p, alpha=(1.0 - ema_decay))
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b)


def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    return x


def train_step(
    state: TrainState,
    labels,
    samples,
    negative_samples,
    feature_params,
    feature_apply,
    learning_rate_fn: Any = None,
    cfg_min=1.0,
    cfg_max=4.0,
    neg_cfg_pw=1.0,
    no_cfg_frac=0.0,
    gen_per_label=8,
    activation_kwargs=dict(),
    loss_kwargs=dict(R_list=[0.02, 0.05, 0.2]),
    max_grad_norm=2.0,
    grad_accum_steps=1,
    device: torch.device = torch.device("cpu"),
    ot_mode: str = "none",
    ot_kwargs: dict | None = None,
    diverse_noise: bool = False,
):
    labels = torch.as_tensor(labels, device=device, dtype=torch.long)

    _inline_feat = int(os.environ.get("DRIFT_FEAT_CHUNK", "0")) > 0
    samples = torch.as_tensor(samples, device=device)
    negative_samples = torch.as_tensor(negative_samples, device=device)

    bsz = labels.shape[0]
    rng = torch.Generator(device=device)
    if diverse_noise:
        rng.manual_seed(int(state.step) * process_count() + process_index() + 1)
    else:
        rng.manual_seed(int(state.step) + 1)

    frac = torch.rand((bsz,), generator=rng, device=device)
    pw = 1.0 - float(neg_cfg_pw)
    if abs(pw) < 1e-6:
        cfg = torch.exp(torch.log(torch.tensor(float(cfg_min), device=device)) + frac * (torch.log(torch.tensor(float(cfg_max), device=device)) - torch.log(torch.tensor(float(cfg_min), device=device))))
    else:
        cfg = (cfg_min**pw + frac * (cfg_max**pw - cfg_min**pw)) ** (1.0 / pw)

    frac2 = torch.rand((bsz,), generator=rng, device=device)
    cfg = torch.where(frac2 < float(no_cfg_frac), torch.ones_like(cfg), cfg)

    n_pos = samples.shape[1]
    n_gen = int(gen_per_label)
    n_uncond = negative_samples.shape[1]

    uncond_w = (cfg - 1.0) * (n_gen - 1) / max(1, n_uncond)

    # --- real features (full batch, no grad) ---
    sg_features = None
    if not _inline_feat:
        neg_samples_input = torch.cat([samples, negative_samples], dim=1)
        neg_samples_input = rearrange(neg_samples_input, "b x h w c -> (b x) h w c")
        with torch.no_grad():
            sg_features = feature_apply(feature_params, neg_samples_input, **activation_kwargs)
            del neg_samples_input, samples, negative_samples
            sg_features = {k: rearrange(v, "(b x) f d -> b x f d", b=bsz, x=n_pos + n_uncond) for k, v in sg_features.items()}

    # --- learning rate ---
    lr = float(learning_rate_fn(state.step)) if learning_rate_fn is not None else 0.0
    _set_lr(state.optimizer, lr)

    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)

    # --- gradient accumulation loop ---
    grad_accum_steps = max(1, int(grad_accum_steps))
    chunk_size = (bsz + grad_accum_steps - 1) // grad_accum_steps
    actual_accum = (bsz + chunk_size - 1) // chunk_size

    total_loss_accum = torch.zeros((), device=device)
    total_info = {}

    for accum_idx in range(actual_accum):
        s = accum_idx * chunk_size
        e = min(s + chunk_size, bsz)
        if s >= bsz:
            break

        chunk_labels = labels[s:e]
        chunk_cfg = cfg[s:e]
        chunk_uncond_w = uncond_w[s:e]
        chunk_bsz = e - s

        if _inline_feat:
            _ci = torch.cat([samples[s:e], negative_samples[s:e]], dim=1)
            _ci = rearrange(_ci, "b x h w c -> (b x) h w c")
            with torch.no_grad():
                chunk_sg = feature_apply(feature_params, _ci, **activation_kwargs)
                del _ci
                chunk_sg = {k: rearrange(v, "(b x) f d -> b x f d", b=chunk_bsz, x=n_pos + n_uncond) for k, v in chunk_sg.items()}
        else:
            chunk_sg = {k: v[s:e] for k, v in sg_features.items()}

        input_labels = repeat(chunk_labels, "b -> (b g)", g=n_gen)
        input_cfg = repeat(chunk_cfg, "b -> (b g)", g=n_gen)

        use_no_sync = hasattr(state.model, "no_sync") and accum_idx < actual_accum - 1
        sync_ctx = state.model.no_sync() if use_no_sync else nullcontext()

        _use_ot = ot_mode == "debiased"
        _ot_kw = ot_kwargs or {}
        _use_new_cfg = _ot_kw.get("use_new_cfg", False)
        _resample_neg = _ot_kw.get("resample_neg", False) and _use_ot

        resamp_features = None
        if _resample_neg:
            with torch.no_grad():
                neg_samples = state.model(
                    c=input_labels,
                    cfg_scale=input_cfg,
                    deterministic=False,
                    train=False,
                    rng=rng,
                )["samples"]
                resamp_features = feature_apply(feature_params, neg_samples, **activation_kwargs)
                del neg_samples
                resamp_features = {k: rearrange(v, "(b g) f d -> b g f d", b=chunk_bsz, g=n_gen) for k, v in resamp_features.items()}

        with sync_ctx:
            gen_samples = state.model(
                c=input_labels,
                cfg_scale=input_cfg,
                deterministic=False,
                train=True,
                rng=rng,
            )["samples"]

            gen_features = feature_apply(feature_params, gen_samples, **activation_kwargs)
            gen_features = {k: rearrange(v, "(b g) f d -> b g f d", b=chunk_bsz, g=n_gen) for k, v in gen_features.items()}

            chunk_loss = torch.zeros((), device=device)

            for k in chunk_sg.keys():
                feature_pos = chunk_sg[k][:, :n_pos]
                feature_uncond = chunk_sg[k][:, n_pos:]
                feature_gen = gen_features[k]

                feature_pos = rearrange(feature_pos, "b x f d -> (b f) x d")
                feature_gen = rearrange(feature_gen, "b x f d -> (b f) x d")
                feature_uncond = rearrange(feature_uncond, "b x f d -> (b f) x d")

                if _resample_neg:
                    feature_neg_detached = rearrange(resamp_features[k], "b x f d -> (b f) x d")
                else:
                    feature_neg_detached = feature_gen.detach()

                Bf = feature_gen.shape[0]
                weight_neg = repeat(chunk_uncond_w, "b -> (b f) k", f=Bf // max(1, chunk_uncond_w.shape[0]), k=n_uncond)

                if _use_ot:
                    ot_loss_kwargs = dict(loss_kwargs)
                    ot_loss_kwargs["sinkhorn_num_iter"] = _ot_kw.get("sinkhorn_num_iter", 20)
                    ot_loss_kwargs["sinkhorn_stop_thr"] = _ot_kw.get("sinkhorn_stop_thr", 1e-4)
                    ot_loss_kwargs["disable_diag_mask"] = _ot_kw.get("disable_diag_mask", False)
                    ot_loss_kwargs["batch_sinkhorn"] = _ot_kw.get("batch_sinkhorn", False)
                    ot_loss_kwargs["use_quadratic_cost"] = _ot_kw.get("use_quadratic_cost", False)

                    if _use_new_cfg:
                        cfg_w = repeat(chunk_cfg - 1.0, "b -> (b f)", f=Bf // max(1, chunk_cfg.shape[0]))
                        loss_feat, info = drift_loss_ot(
                            gen=feature_gen,
                            fixed_pos=feature_pos,
                            fixed_neg=feature_neg_detached,
                            weight_gen=torch.ones_like(feature_gen[:, :, 0]),
                            weight_pos=torch.ones_like(feature_pos[:, :, 0]),
                            weight_neg=torch.ones_like(feature_neg_detached[:, :, 0]),
                            use_new_cfg=True,
                            fixed_uncond=feature_uncond,
                            weight_uncond=repeat(cfg_w, "b -> b 1"),
                            **ot_loss_kwargs,
                        )
                    else:
                        ot_neg = torch.cat([feature_neg_detached, feature_uncond], dim=1)
                        ot_neg_w = torch.cat([
                            torch.ones(Bf, n_gen, device=device),
                            weight_neg,
                        ], dim=1)
                        loss_feat, info = drift_loss_ot(
                            gen=feature_gen,
                            fixed_pos=feature_pos,
                            fixed_neg=ot_neg,
                            weight_gen=torch.ones_like(feature_gen[:, :, 0]),
                            weight_pos=torch.ones_like(feature_pos[:, :, 0]),
                            weight_neg=ot_neg_w,
                            **ot_loss_kwargs,
                        )
                else:
                    loss_feat, info = drift_loss(
                        gen=feature_gen,
                        fixed_pos=feature_pos,
                        fixed_neg=feature_uncond,
                        weight_gen=torch.ones_like(feature_gen[:, :, 0]),
                        weight_pos=torch.ones_like(feature_pos[:, :, 0]),
                        weight_neg=weight_neg,
                        **loss_kwargs,
                    )

                chunk_loss = chunk_loss + loss_feat.mean()
                for k2, v2 in info.items():
                    key = f"{k2}/{k}"
                    if key not in total_info:
                        total_info[key] = v2.detach() if torch.is_tensor(v2) else torch.tensor(float(v2), device=device)
                    else:
                        total_info[key] = total_info[key] + (v2.detach() if torch.is_tensor(v2) else torch.tensor(float(v2), device=device))

            chunk_loss = chunk_loss / max(1, len(chunk_sg))
            scaled_loss = chunk_loss / actual_accum
            scaled_loss.backward()

            total_loss_accum = total_loss_accum + chunk_loss.detach()

    g_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_grad_norm)
    state.optimizer.step()
    _update_ema(state.ema_model, state.model, state.ema_decay)

    metrics = {}
    for k, v in total_info.items():
        val = v / actual_accum
        metrics[k] = val.mean().detach() if torch.is_tensor(val) else torch.tensor(float(val), device=device)
    metrics["loss"] = total_loss_accum / actual_accum
    metrics["g_norm"] = torch.as_tensor(g_norm, device=device)
    metrics["lr"] = torch.tensor(lr, device=device)

    state.step += 1
    return state, metrics


@torch.no_grad()
def generate_step(batch, model, rng, postprocess_fn, cfg_scale=1.0, device: Optional[torch.device] = None):
    _, labels = batch
    labels = torch.as_tensor(labels, dtype=torch.long)
    if device is None:
        device = next(model.parameters()).device
    labels = labels.to(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(rng) + 1)

    latent_samples = model(
        c=labels,
        cfg_scale=cfg_scale,
        deterministic=True,
        train=False,
        rng=gen,
    )["samples"]
    return postprocess_fn(latent_samples)


def train_gen(
    model,
    optimizer,
    logger,
    eval_loader,
    train_loader,
    learning_rate_fn,
    preprocess_fn,
    postprocess_fn,
    dataset_name="imagenet256",
    train_batch_size=0,
    total_steps=100000,
    save_per_step=10000,
    eval_per_step=5000,
    eval_samples=50000,
    activation_fn=None,
    feature_params=None,
    ema_decay=0.999,
    seed=42,
    pos_per_sample=32,
    neg_per_sample=16,
    forward_dict=dict(
        gen_per_label=16,
        cfg_min=1.0,
        cfg_max=4.0,
        neg_cfg_pw=1.0,
        no_cfg_frac=0.0,
    ),
    positive_bank_size=64,
    negative_bank_size=512,
    cfg_list=(1.0,),
    activation_kwargs=dict(
        patch_mean_size=[2, 4],
        patch_std_size=[2, 4],
        use_std=True,
        use_mean=True,
        every_k_block=2,
    ),
    max_grad_norm=2.0,
    loss_kwargs=dict(R_list=(0.02, 0.05, 0.2)),
    keep_every=500000,
    keep_last=2,
    init_from="",
    push_per_step=0,
    push_at_resume=3000,
    grad_accum_steps=1,
    workdir="runs",
    ot_mode="none",
    ot_kwargs=None,
    diverse_noise=False,
):
    if isinstance(ema_decay, (list, tuple)):
        if len(ema_decay) != 1:
            raise ValueError(f"Expected a single ema_decay value, got {ema_decay}")
        ema_decay = float(ema_decay[0])
    else:
        ema_decay = float(ema_decay)

    if cfg_list is None:
        cfg_list = [1.0]
    elif isinstance(cfg_list, (list, tuple)):
        cfg_list = [float(cfg) for cfg in cfg_list]
    else:
        cfg_list = [float(cfg_list)]

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    seed_everything(int(seed) + process_index())

    model = model.to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    _compile = os.environ.get("DRIFT_COMPILE", "1") != "0"
    if _compile and hasattr(model, "model"):
        if getattr(model.model, "use_remat", False):
            import torch._dynamo.config as _dynamo_config
            _dynamo_config.optimize_ddp = False
            log_for_0("Disabled DDPOptimizer (use_remat + torch.compile)")
        log_for_0("Compiling inner generator (LightningDiT) with torch.compile ...")
        model.model = torch.compile(model.model, dynamic=True)

    model = maybe_ddp_model(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    opt = optimizer(model.parameters())

    state = TrainState(step=0, model=model, optimizer=opt, ema_model=ema_model, ema_decay=ema_decay)
    state = restore_checkpoint(state=state, workdir=workdir)
    if int(state.step) == 0 and init_from:
        log_for_0("Initializing generator params from init_from=%s", init_from)
        state = maybe_init_state_params(
            state,
            model_type="generator",
            init_from=init_from,
            hf_cache_dir=HF_ROOT,
        )

    assert feature_params is not None, "feature_params must be provided for feature extraction"

    log_for_0("Starting training loop (world_size=%d, grad_accum=%d)...", process_count(), grad_accum_steps)
    step = int(state.step)
    initial_step = step
    pbar = tqdm(range(step, total_steps), initial=step, total=total_steps) if is_rank_zero() else range(step, total_steps)
    memory_bank_positive = ArrayMemoryBank(num_classes=1000, max_size=positive_bank_size)
    memory_bank_negative = ArrayMemoryBank(num_classes=1, max_size=negative_bank_size)
    train_iter = infinite_sampler(train_loader, step)
    _ot_kw = dict(ot_kwargs) if ot_kwargs else {}

    for step in pbar:
        start_time = time.time()
        n_push = 0
        logger.set_step(step)

        goal = push_per_step
        if initial_step > 0 and step == initial_step:
            goal = push_at_resume * push_per_step
            log_for_0("pushing at resume: %d", goal)

        while True:
            batch = next(train_iter)
            processed_batch = preprocess_fn(batch)
            images = processed_batch["images"]
            labels = processed_batch["labels"]
            memory_bank_positive.add(images, labels)
            memory_bank_negative.add(images, labels * 0)
            n_push += int(images.shape[0])
            if n_push >= goal:
                break

        bsz_per_host = train_batch_size // max(1, process_count())
        if labels.shape[0] < bsz_per_host:
            raise ValueError(f"Labels shape {labels.shape[0]} < bsz_per_host {bsz_per_host}")

        perm = torch.randperm(labels.shape[0])[:bsz_per_host]
        labels_sel = labels[perm]

        positive_samples = memory_bank_positive.sample(labels_sel, n_samples=pos_per_sample)
        negative_samples = memory_bank_negative.sample(labels_sel * 0, n_samples=neg_per_sample)

        process_time = time.time() - start_time

        profile_metrics = {}
        if step == initial_step:
            profile_metrics = profile_func(
                lambda s, l, p, n, fp: train_step(
                    s, l, p, n, fp, activation_fn,
                    learning_rate_fn=learning_rate_fn,
                    activation_kwargs=activation_kwargs,
                    loss_kwargs=loss_kwargs,
                    max_grad_norm=max_grad_norm,
                    grad_accum_steps=grad_accum_steps,
                    device=device,
                    ot_mode=ot_mode,
                    ot_kwargs=_ot_kw,
                    diverse_noise=diverse_noise,
                    **forward_dict,
                ),
                (state, labels_sel, positive_samples, negative_samples, feature_params),
                name="train_step",
            )

        state, metrics = train_step(
            state,
            labels_sel,
            positive_samples,
            negative_samples,
            feature_params,
            activation_fn,
            learning_rate_fn=learning_rate_fn,
            activation_kwargs=activation_kwargs,
            loss_kwargs=loss_kwargs,
            max_grad_norm=max_grad_norm,
            grad_accum_steps=grad_accum_steps,
            device=device,
            ot_mode=ot_mode,
            ot_kwargs=_ot_kw,
            diverse_noise=diverse_noise,
            **forward_dict,
        )

        total_time = time.time() - start_time
        metrics["total_time"] = total_time
        metrics["process_time"] = process_time
        metrics["kimg"] = (step + 1) * positive_samples.shape[0] / 1000.0
        metrics["forward_kimg"] = (step + 1) * positive_samples.shape[0] / 1000.0 * forward_dict["gen_per_label"]
        metrics.update(profile_metrics)

        logger.log_dict(metrics)
        step += 1

        if step % save_per_step == 0 or step == total_steps:
            save_checkpoint(state, keep=keep_last, keep_every=keep_every, workdir=workdir)
            if is_rank_zero():
                save_params_ema_artifact(
                    state,
                    workdir=workdir,
                    kind="gen",
                    model_config=_generator_model_config(state.model),
                )

        if (step % eval_per_step == 0) or (step == 1) or (step == total_steps):
            torch.cuda.empty_cache()
            is_sanity = step == 1
            n_samples = 500 if is_sanity else eval_samples
            folder_prefix = "sanity" if is_sanity else "CFG"
            round_best_fid = float("inf")
            round_best_cfg = cfg_list[0]
            eval_cfg_list = cfg_list if not is_sanity else [cfg_list[0]]

            for eval_cfg in eval_cfg_list:
                result = evaluate_fid(
                    dataset_name=dataset_name,
                    gen_func=generate_step,
                    gen_params={"model": state.ema_model, "cfg_scale": eval_cfg, "postprocess_fn": postprocess_fn, "device": device},
                    eval_loader=eval_loader,
                    logger=logger,
                    num_samples=n_samples,
                    log_folder=f"{folder_prefix}{eval_cfg}",
                    log_prefix=f"EMA_{state.ema_decay:g}",
                    rng_eval=0,
                )
                fid_val = result.get("fid", float("inf"))
                if fid_val < round_best_fid:
                    round_best_fid = fid_val
                    round_best_cfg = eval_cfg
            if not is_sanity:
                log_for_0("best_fid=%.4f best_cfg=%.1f (step=%d)", round_best_fid, round_best_cfg, step)
                if is_rank_zero():
                    logger.log_dict({"best_fid": round_best_fid, "best_cfg": round_best_cfg})

    logger.finish()
    del model, eval_loader, train_loader, state
    gc.collect()


def main_gen(config, output_dir="runs"):
    if "logging" not in config:
        config.logging = {}
    config.logging.name = Path(output_dir).resolve().name

    from models.generator import DitGen

    # Fix seed here
    train_seed = int(config.train.get("seed", 42))
    seed_everything(train_seed)

    model_dict = build_model_dict(config, DitGen, workdir=output_dir)
    use_aug = bool(config.dataset.get("use_aug", False))
    use_latent = bool(config.dataset.get("use_latent", False))
    use_cache = bool(config.dataset.get("use_cache", False))
    postprocess_fn_noclip = get_postprocess_fn(
        use_aug=use_aug,
        use_latent=use_latent,
        use_cache=use_cache,
        has_clip=False,
    )
    feature_cfg = model_dict.feature
    mae_path = str(feature_cfg.get("mae_path", "")).strip()
    if not mae_path and bool(feature_cfg.get("use_mae", True)):
        load_dict = feature_cfg.get("load_dict", {})
        if str(load_dict.get("source", "hf")).strip().lower() == "local":
            mae_path = str(load_dict.get("path", "")).strip()
        else:
            model_name = str(load_dict.get("hf_model_name", "")).strip()
            if model_name:
                mae_path = f"hf://{model_name}"
    if bool(feature_cfg.get("use_mae", True)) and not mae_path:
        raise ValueError("feature.mae_path (or feature.load_dict.hf_model_name / feature.load_dict.path) is required when use_mae=true.")

    activation_fn, variables = build_activation_function(
        mae_path=mae_path,
        use_convnext=bool(feature_cfg.get("use_convnext", False)),
        convnext_bf16=bool(feature_cfg.get("convnext_bf16", False)),
        use_mae=bool(feature_cfg.get("use_mae", True)),
        postprocess_fn=postprocess_fn_noclip,
    )

    train_gen(
        model=model_dict.model,
        optimizer=model_dict.optimizer,
        logger=model_dict.logger,
        eval_loader=model_dict.eval_loader,
        train_loader=model_dict.train_loader,
        learning_rate_fn=model_dict.learning_rate_fn,
        preprocess_fn=model_dict.preprocess_fn,
        postprocess_fn=model_dict.postprocess_fn,
        dataset_name=model_dict.dataset_name,
        activation_fn=activation_fn,
        feature_params=variables,
        workdir=output_dir,
        **config.train,
    )


def main(args):
    run_init()
    config = load_config(args.config)
    main_gen(config, output_dir=args.workdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gen/ablation_ot_1node.yaml", help="Path to configuration file.")
    parser.add_argument("--workdir", type=str, default="runs", help="Local workdir root for checkpoints/logs.")
    args = parser.parse_args()
    args.output_dir = args.workdir

    main(args)
