"""ConvNeXt feature model (PyTorch port)."""

from __future__ import annotations

import re
from functools import partial
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils.logging import log_for_0


class ConvNextLayerNorm(nn.Module):
    """LayerNorm on the last channel for NHWC tensors."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        old_dtype = x.dtype
        x = x.float()
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight * x + self.bias
        return x.to(dtype=old_dtype)


class ConvNextGRN(nn.Module):
    """Global Response Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, self.dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        old_dtype = x.dtype
        x = x.float()
        norm = torch.sum(x**2, dim=(1, 2), keepdim=True)
        gx = torch.sqrt(norm + self.eps)
        nx = gx / (torch.mean(gx, dim=-1, keepdim=True) + self.eps)
        out = self.gamma * (x * nx) + self.beta + x
        return out.to(dtype=old_dtype)


class ConvNextBlock(nn.Module):
    """ConvNeXtV2 residual block."""

    def __init__(self, dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dim = int(dim)
        self.dwconv = nn.Conv2d(self.dim, self.dim, kernel_size=7, padding=3, groups=self.dim, bias=True, dtype=dtype)
        self.norm = ConvNextLayerNorm(self.dim, eps=1e-6)
        self.pwconv1 = nn.Linear(self.dim, 4 * self.dim, bias=True, dtype=dtype)
        self.grn = ConvNextGRN(4 * self.dim)
        self.pwconv2 = nn.Linear(4 * self.dim, self.dim, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: NHWC
        residual = x
        y = x.permute(0, 3, 1, 2).contiguous()
        y = self.dwconv(y)
        y = y.permute(0, 2, 3, 1).contiguous()
        y = self.norm(y)
        y = self.pwconv1(y)
        y = F.gelu(y, approximate="none")
        y = self.grn(y)
        y = self.pwconv2(y)
        return residual + y


def safe_std(x, axis, eps=1e-6):
    x32 = x.float()
    mean = torch.mean(x32, dim=axis, keepdim=True)
    var = torch.mean((x32 - mean) ** 2, dim=axis, keepdim=False)
    var = torch.clamp(var, min=0.0)
    return torch.sqrt(var + eps)


class _Downsample0(nn.Module):
    def __init__(self, out_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.conv = nn.Conv2d(3, out_dim, kernel_size=4, stride=4, dtype=dtype)
        self.norm = ConvNextLayerNorm(out_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: NHWC
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return self.norm(x)


class _Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.norm = ConvNextLayerNorm(in_dim, eps=1e-6)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class ConvNextV2(nn.Module):
    """ConvNeXtV2 backbone with activation export."""

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        drop_path_rate: float = 0.0,
        head_init_scale: float = 1.0,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        dtype: torch.dtype = torch.float32,
    ):
        del in_chans, drop_path_rate, head_init_scale
        super().__init__()
        self.num_classes = int(num_classes)
        self.depths = tuple(int(d) for d in depths)
        self.dims = tuple(int(d) for d in dims)
        self.dtype = dtype

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(_Downsample0(self.dims[0], dtype=dtype))
        for i in range(3):
            self.downsample_layers.append(_Downsample(self.dims[i], self.dims[i + 1], dtype=dtype))

        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.ModuleList([ConvNextBlock(dim=self.dims[i], dtype=dtype) for _ in range(self.depths[i])]))

        self.norm = nn.LayerNorm(self.dims[-1], eps=1e-6)
        self.head = nn.Linear(self.dims[-1], self.num_classes, dtype=dtype)

    def get_activations(self, x: torch.Tensor) -> dict:
        # x: NHWC
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1).contiguous().to(dtype=self.dtype)
        feature_dict = {}

        def normalize(y):
            old_dtype = y.dtype
            y = y.float()
            y = (y - y.mean(dim=-1, keepdim=True)) / (y.std(dim=-1, keepdim=True, correction=0) + 1e-3)
            return y.to(dtype=old_dtype)

        for i in range(4):
            x = self.downsample_layers[i](x)
            for block in self.stages[i]:
                x = block(x)
            x_normed = normalize(x)
            if i > 0:
                feature_dict[f"convenxt_stage_{i}"] = rearrange(x_normed, "b h w c -> b (h w) c")
            feature_dict[f"convenxt_stage_{i}_mean"] = x_normed.mean(dim=(1, 2))[:, None, :]
            feature_dict[f"convenxt_stage_{i}_std"] = safe_std(rearrange(x_normed, "b h w c -> b (h w) c"), axis=1)[:, None, :]

        feature_dict["global_mean"] = self.norm(x.mean(dim=(1, 2)))[:, None, :]
        feature_dict["global_std"] = safe_std(rearrange(normalize(x), "b h w c -> b (h w) c"), axis=1)[:, None, :]
        return feature_dict

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1).contiguous().to(dtype=self.dtype)
        for i in range(4):
            x = self.downsample_layers[i](x)
            for block in self.stages[i]:
                x = block(x)
        return self.norm(x.mean(dim=(1, 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


ConvNextBase = partial(ConvNextV2, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
ConvNextTiny = partial(ConvNextV2, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])


def _map_hf_key_to_local(path: str) -> str:
    path = re.sub(r"classifier\.", "head.", path)
    path = re.sub(r"convnextv2\.encoder\.", "", path)
    path = re.sub(r"convnextv2\.embeddings\.patch_embeddings\.", "downsample_layers.0.conv.", path)
    path = re.sub(r"convnextv2\.embeddings\.layernorm\.", "downsample_layers.0.norm.", path)
    path = re.sub(r"stages\.([0-3])\.downsampling_layer\.0\.", lambda m: f"downsample_layers.{int(m.group(1))}.norm.", path)
    path = re.sub(r"stages\.([0-3])\.downsampling_layer\.1\.", lambda m: f"downsample_layers.{int(m.group(1))}.conv.", path)
    path = re.sub(r"stages\.([0-3])\.layers\.([0-9]+)\.", lambda m: f"stages.{m.group(1)}.{m.group(2)}.", path)
    path = re.sub(r"layernorm", "norm", path)
    path = re.sub(r"grn\.weight", "grn.gamma", path)
    path = re.sub(r"grn\.bias", "grn.beta", path)
    path = re.sub(r"convnextv2\.", "", path)
    return path


def load_convnext_torch_model(model_name: str = "base", use_bf16: bool = False):
    dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available()) else torch.float32
    if model_name == "base":
        model = ConvNextBase(dtype=dtype)
        model_load_name = "facebook/convnextv2-base-22k-224"

    elif model_name == "tiny":
        model = ConvNextTiny(dtype=dtype)
        model_load_name = "facebook/convnextv2-tiny-22k-224"
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    from transformers import ConvNextV2ForImageClassification

    model_pt = ConvNextV2ForImageClassification.from_pretrained(model_load_name).state_dict()
    mapped = {}
    for k, v in model_pt.items():
        nk = _map_hf_key_to_local(k)
        mapped[nk] = v

    target = model.state_dict()
    loaded = {}
    missing = []
    for k, t in target.items():
        if k in mapped and mapped[k].shape == t.shape:
            loaded[k] = mapped[k].to(dtype=t.dtype)
        else:
            missing.append(k)
            loaded[k] = t

    model.load_state_dict(loaded, strict=False)
    if missing:
        log_for_0("[ConvNeXt] missing keys while loading pretrained weights: %s", missing)
        raise ValueError(f"ConvNeXt pretrained weight loading has {len(missing)} missing keys: {missing}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, model.state_dict()
