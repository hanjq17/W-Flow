from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterable

import numpy as np
import torch


def flatten_tree(tree: Any, prefix: tuple[str, ...] = ()) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(tree, Mapping):
        for k, v in tree.items():
            out.update(flatten_tree(v, prefix + (str(k),)))
        return out
    if isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            out.update(flatten_tree(v, prefix + (str(i),)))
        return out
    out[".".join(prefix)] = tree
    return out


def _as_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.detach().cpu()
    arr = np.array(x, copy=True)
    return torch.from_numpy(arr)


def _convert_value_to_target(src: Any, target: torch.Tensor, key: str = "") -> torch.Tensor:
    src_t = _as_tensor(src)

    # JAX Dense kernel is (in_features, out_features); PyTorch Linear weight
    # is (out_features, in_features).  For non-square matrices the shape
    # mismatch triggers the generic transpose branch below, but square
    # matrices slip through the direct shape-match check.  Detect Linear
    # weights by their key suffix and always transpose them.
    if src_t.ndim == 2 and key.endswith(".linear.weight"):
        return src_t.t().contiguous().to(dtype=target.dtype)

    if tuple(src_t.shape) == tuple(target.shape):
        return src_t.to(dtype=target.dtype)

    if src_t.ndim == 2 and tuple(src_t.t().shape) == tuple(target.shape):
        return src_t.t().contiguous().to(dtype=target.dtype)

    if src_t.ndim == 4:
        hwio = src_t.permute(3, 2, 0, 1).contiguous()
        if tuple(hwio.shape) == tuple(target.shape):
            return hwio.to(dtype=target.dtype)

        ihwo = src_t.permute(2, 3, 0, 1).contiguous()
        if tuple(ihwo.shape) == tuple(target.shape):
            return ihwo.to(dtype=target.dtype)

    raise ValueError(f"shape mismatch src={tuple(src_t.shape)} target={tuple(target.shape)}")


def _normalize_common(k: str) -> str:
    k = k.replace("/", ".")
    if k.startswith("params."):
        k = k[len("params.") :]

    k = k.replace(".Dense_0.kernel", ".weight")
    k = k.replace(".Dense_0.bias", ".bias")
    k = k.replace(".kernel", ".weight")
    k = k.replace(".scale", ".weight")
    k = k.replace(".embedding", ".weight")
    return k


def jax_to_torch_key_mae(k: str) -> str:
    k = _normalize_common(k)
    k = re.sub(r"stages_(\d+)", r"stages.\1", k)
    k = re.sub(r"layers_(\d+)", r"\1", k)
    return k


def jax_to_torch_key_generator(k: str) -> str:
    k = _normalize_common(k)

    k = re.sub(r"^Embed_0\.weight$", "class_embed.weight", k)
    k = re.sub(r"^noise_embeds_(\d+)\.weight$", r"noise_embeds.\1.weight", k)

    # cfg embedder
    k = re.sub(r"^TimestepEmbedder_0\.TorchLinear_0\.weight$", "cfg_embedder.fc1.linear.weight", k)
    k = re.sub(r"^TimestepEmbedder_0\.TorchLinear_0\.bias$", "cfg_embedder.fc1.linear.bias", k)
    k = re.sub(r"^TimestepEmbedder_0\.TorchLinear_1\.weight$", "cfg_embedder.fc2.linear.weight", k)
    k = re.sub(r"^TimestepEmbedder_0\.TorchLinear_1\.bias$", "cfg_embedder.fc2.linear.bias", k)
    k = re.sub(r"^RMSNorm_0\.weight$", "cfg_norm.weight", k)

    k = re.sub(r"^LightningDiT_0\.pos_embed$", "model.pos_embed", k)
    k = re.sub(r"^LightningDiT_0\.cls_embed$", "model.cls_embed", k)
    k = re.sub(r"^LightningDiT_0\.TorchLinear_0\.weight$", "model.patch_embed.linear.weight", k)
    k = re.sub(r"^LightningDiT_0\.TorchLinear_0\.bias$", "model.patch_embed.linear.bias", k)
    k = re.sub(r"^LightningDiT_0\.TorchLinear_1\.weight$", "model.c_token_proj.linear.weight", k)
    k = re.sub(r"^LightningDiT_0\.TorchLinear_1\.bias$", "model.c_token_proj.linear.bias", k)

    k = re.sub(r"^LightningDiT_0\.blocks_(\d+)\.", r"model.blocks.\1.", k)
    k = re.sub(r"^model\.blocks\.(\d+)\.RMSNorm_0\.weight$", r"model.blocks.\1.norm1.weight", k)
    k = re.sub(r"^model\.blocks\.(\d+)\.RMSNorm_1\.weight$", r"model.blocks.\1.norm2.weight", k)

    k = re.sub(r"^model\.blocks\.(\d+)\.Attention_0\.TorchLinear_0\.weight$", r"model.blocks.\1.attn.qkv.linear.weight", k)
    k = re.sub(r"^model\.blocks\.(\d+)\.Attention_0\.TorchLinear_0\.bias$", r"model.blocks.\1.attn.qkv.linear.bias", k)
    k = re.sub(r"^model\.blocks\.(\d+)\.Attention_0\.TorchLinear_1\.weight$", r"model.blocks.\1.attn.proj.linear.weight", k)
    k = re.sub(r"^model\.blocks\.(\d+)\.Attention_0\.TorchLinear_1\.bias$", r"model.blocks.\1.attn.proj.linear.bias", k)
    k = re.sub(r"^model\.blocks\.(\d+)\.Attention_0\.q_norm\.weight$", r"model.blocks.\1.attn.q_norm.weight", k)
    k = re.sub(r"^model\.blocks\.(\d+)\.Attention_0\.q_norm\.bias$", r"model.blocks.\1.attn.q_norm.bias", k)
    k = re.sub(r"^model\.blocks\.(\d+)\.Attention_0\.k_norm\.weight$", r"model.blocks.\1.attn.k_norm.weight", k)
    k = re.sub(r"^model\.blocks\.(\d+)\.Attention_0\.k_norm\.bias$", r"model.blocks.\1.attn.k_norm.bias", k)

    k = re.sub(r"^model\.blocks\.(\d+)\.SwiGLUFFN_0\.TorchLinear_0\.weight$", r"model.blocks.\1.mlp.w1.linear.weight", k)
    k = re.sub(r"^model\.blocks\.(\d+)\.SwiGLUFFN_0\.TorchLinear_0\.bias$", r"model.blocks.\1.mlp.w1.linear.bias", k)
    k = re.sub(r"^model\.blocks\.(\d+)\.SwiGLUFFN_0\.TorchLinear_1\.weight$", r"model.blocks.\1.mlp.w3.linear.weight", k)
    k = re.sub(r"^model\.blocks\.(\d+)\.SwiGLUFFN_0\.TorchLinear_1\.bias$", r"model.blocks.\1.mlp.w3.linear.bias", k)
    k = re.sub(r"^model\.blocks\.(\d+)\.SwiGLUFFN_0\.TorchLinear_2\.weight$", r"model.blocks.\1.mlp.w2.linear.weight", k)
    k = re.sub(r"^model\.blocks\.(\d+)\.SwiGLUFFN_0\.TorchLinear_2\.bias$", r"model.blocks.\1.mlp.w2.linear.bias", k)

    # this was the missing one
    k = re.sub(r"^model\.blocks\.(\d+)\.TorchLinear_0\.weight$", r"model.blocks.\1.adaLN_mod.1.linear.weight", k)
    k = re.sub(r"^model\.blocks\.(\d+)\.TorchLinear_0\.bias$", r"model.blocks.\1.adaLN_mod.1.linear.bias", k)
    
    # block adaLN mod
    k = re.sub(r"^model\.blocks\.(\d+)\.TorchLinear_2\.weight$", r"model.blocks.\1.adaLN_mod.1.linear.weight", k)
    k = re.sub(r"^model\.blocks\.(\d+)\.TorchLinear_2\.bias$", r"model.blocks.\1.adaLN_mod.1.linear.bias", k)

    # final layer
    k = re.sub(r"^LightningDiT_0\.FinalLayer_0\.RMSNorm_0\.weight$", "model.final_layer.norm_final.weight", k)
    k = re.sub(r"^LightningDiT_0\.FinalLayer_0\.TorchLinear_0\.weight$", "model.final_layer.adaLN_mod.1.linear.weight", k)
    k = re.sub(r"^LightningDiT_0\.FinalLayer_0\.TorchLinear_0\.bias$", "model.final_layer.adaLN_mod.1.linear.bias", k)
    k = re.sub(r"^LightningDiT_0\.FinalLayer_0\.TorchLinear_1\.weight$", "model.final_layer.linear.linear.weight", k)
    k = re.sub(r"^LightningDiT_0\.FinalLayer_0\.TorchLinear_1\.bias$", "model.final_layer.linear.linear.bias", k)

    return k


def convert_tree_to_state_dict(
    tree: Any,
    target_state: Dict[str, torch.Tensor],
    key_fn: Callable[[str], str],
) -> Dict[str, torch.Tensor]:
    flat = flatten_tree(tree)

    missing = []
    
    mapped: Dict[str, Any] = {}
    raw2new: Dict[str, str] = {}
    for k, v in flat.items():
        nk = key_fn(k)
        mapped[nk] = v
        raw2new[k] = nk

    out: Dict[str, torch.Tensor] = {}
    not_found: list[str] = []
    bad_shape: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    for tk, tv in target_state.items():
        if tk in mapped:
            try:
                out[tk] = _convert_value_to_target(mapped[tk], tv, key=tk)
                continue
            except Exception:
                pass

        out[tk] = tv
        missing.append(tk)

    if len(missing) > 0:
        short = ", ".join(missing[:10])
        raise ValueError(f"Unable to map {len(missing)} parameters. First keys: {short}")

    return out


def convert_mae_tree_to_state_dict(tree: Any, target_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return convert_tree_to_state_dict(tree, target_state, jax_to_torch_key_mae)


def convert_generator_tree_to_state_dict(tree: Any, target_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return convert_tree_to_state_dict(tree, target_state, jax_to_torch_key_generator)
