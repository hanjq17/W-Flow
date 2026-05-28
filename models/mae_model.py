"""Self-contained PyTorch MAE-ResNet (ported from release JAX code)."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from utils.env import HF_REPO_ID, HF_ROOT

_COMPILE = os.environ.get("DRIFT_COMPILE", "1") != "0"


def _choose_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return max(g, 1)


class _BasicBlock(nn.Module):
    def __init__(
        self,
        filters: int,
        in_channels: Optional[int] = None,
        stride: int = 1,
        gn_max_groups: int = 32,
        dropout_prob: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.filters = int(filters)
        self.in_channels = int(in_channels) if in_channels is not None else None
        self.stride = int(stride)

        actual_in = self.in_channels if self.in_channels is not None else self.filters
        self.conv1 = nn.Conv2d(
            actual_in, self.filters,
            kernel_size=3, stride=self.stride, padding=1, bias=False, dtype=dtype,
        )
        self.gn1 = nn.GroupNorm(_choose_gn_groups(self.filters, gn_max_groups), self.filters, dtype=dtype)
        self.conv2 = nn.Conv2d(self.filters, self.filters, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype)
        self.gn2 = nn.GroupNorm(_choose_gn_groups(self.filters, gn_max_groups), self.filters, dtype=dtype)
        self.drop = nn.Dropout(dropout_prob)

        need_proj = (actual_in != self.filters) or (self.stride != 1)
        if need_proj:
            self.proj_conv = nn.Conv2d(
                actual_in, self.filters,
                kernel_size=1, stride=self.stride, bias=False, dtype=dtype,
            )
            self.proj_gn = nn.GroupNorm(_choose_gn_groups(self.filters, gn_max_groups), self.filters, dtype=dtype)
        else:
            self.proj_conv = None
            self.proj_gn = None

    def forward(self, x: torch.Tensor, *, train: bool) -> torch.Tensor:
        residual = x
        y = self.conv1(x)
        y = self.gn1(y)
        y = torch.relu(y)
        y = self.drop(y) if train else y
        y = self.conv2(y)
        y = self.gn2(y)

        if self.proj_conv is not None:
            residual = self.proj_conv(residual)
            residual = self.proj_gn(residual)

        return torch.relu(residual + y)


class _ResNetEncoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
        dropout_prob: float = 0.0,
        gn_max_groups: int = 32,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.base_channels = int(base_channels)
        self.layers = tuple(int(v) for v in layers)
        self.dtype = dtype

        self.conv1 = nn.Conv2d(3, self.base_channels, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype)
        self.gn1 = nn.GroupNorm(_choose_gn_groups(self.base_channels, gn_max_groups), self.base_channels, dtype=dtype)

        stages = []
        in_ch = self.base_channels
        for stage_idx, num_blocks in enumerate(self.layers):
            stride = 2 if stage_idx > 0 else 1
            out_ch = in_ch * 2 if stage_idx > 0 else in_ch
            blocks = []
            blocks.append(
                _BasicBlock(
                    out_ch,
                    in_channels=in_ch,
                    stride=stride,
                    dropout_prob=dropout_prob,
                    dtype=dtype,
                )
            )
            for _ in range(1, num_blocks):
                blocks.append(
                    _BasicBlock(
                        out_ch,
                        in_channels=out_ch,
                        stride=1,
                        dropout_prob=dropout_prob,
                        dtype=dtype,
                    )
                )
            stages.append(nn.ModuleList(blocks))
            setattr(self, f"layer{stage_idx + 1}_norm", nn.GroupNorm(_choose_gn_groups(out_ch, gn_max_groups), out_ch, dtype=dtype))
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor, *, train: bool, return_block_outputs: bool = False):
        # x: BHWC -> BCHW
        x = x.permute(0, 3, 1, 2).contiguous()

        feats: Dict[str, torch.Tensor] = {}
        block_outputs: Dict[str, List[torch.Tensor]] = {}

        x = self.conv1(x)
        x = self.gn1(x)
        x = torch.relu(x)
        feats["conv1"] = x

        for i, blocks in enumerate(self.stages):
            layer_name = f"layer{i + 1}"
            outs: List[torch.Tensor] = []
            for block in blocks:
                x = block(x, train=train)
                outs.append(x)
            block_outputs[layer_name] = outs
            norm_layer = getattr(self, f"{layer_name}_norm")
            x = norm_layer(x)
            feats[layer_name] = x

        if return_block_outputs:
            return feats, block_outputs
        return feats


class _ConvGNReLU(nn.Module):
    def __init__(self, in_channels: int, channels: int, kernel: int = 3, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=kernel, padding=kernel // 2, bias=False, dtype=dtype)
        self.gn = nn.GroupNorm(_choose_gn_groups(channels, 32), channels, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.gn(self.conv(x)))


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.concat_norm_fn = nn.GroupNorm(_choose_gn_groups(in_channels + skip_channels, 32), in_channels + skip_channels, dtype=dtype)
        self.proj = _ConvGNReLU(in_channels + skip_channels, out_channels, kernel=3, dtype=dtype)
        self.refine = _ConvGNReLU(out_channels, out_channels, kernel=3, dtype=dtype)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.concat_norm_fn(x)
        x = self.proj(x)
        x = self.refine(x)
        return x


class _UNetDecoder(nn.Module):
    def __init__(self, base_channels: int, out_channels: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        c1 = base_channels
        c2 = base_channels
        c3 = base_channels * 2
        c4 = base_channels * 4
        c5 = base_channels * 8

        self.bridge = _ConvGNReLU(c5, c5, dtype=dtype)
        self.up43 = _UpBlock(c5, c4, c4, dtype=dtype)
        self.up32 = _UpBlock(c4, c3, c3, dtype=dtype)
        self.up21 = _UpBlock(c3, c2, c2, dtype=dtype)
        self.up10 = _UpBlock(c2, c1, c1, dtype=dtype)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1, dtype=dtype)

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.bridge(feats["layer4"])
        x = self.up43(x, feats["layer3"])
        x = self.up32(x, feats["layer2"])
        x = self.up21(x, feats["layer1"])
        x = self.up10(x, feats["conv1"])
        return self.head(x)


def patch_input(x: torch.Tensor, input_patch_size: int) -> torch.Tensor:
    return rearrange(
        x,
        "b (h1 h2) (w1 w2) c -> b h1 w1 (h2 w2 c)",
        h2=input_patch_size,
        w2=input_patch_size,
    )


def make_patch_mask(x: torch.Tensor, rng: torch.Generator, mask_ratio: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    b, h, w, _ = x.shape
    nh, nw = h // patch_size, w // patch_size
    noise = torch.rand((b, nh, nw), dtype=x.dtype, device=x.device, generator=rng)
    mask = (noise < mask_ratio[:, None, None]).to(dtype=x.dtype)
    mask = torch.repeat_interleave(mask, patch_size, dim=1)
    mask = torch.repeat_interleave(mask, patch_size, dim=2)
    return mask[..., None]


def safe_std(x: torch.Tensor, axis, eps: float = 1e-6, keepdims: bool = False) -> torch.Tensor:
    x32 = x.float()
    mean = x32.mean(dim=axis, keepdim=True)
    var = ((x32 - mean) ** 2).mean(dim=axis, keepdim=keepdims)
    return torch.sqrt(torch.clamp(var, min=0.0) + eps)


class MAEResNetJAX(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        base_channels: int = 64,
        patch_size: int = 4,
        dropout_prob: float = 0.0,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
        use_bf16: bool = False,
        input_patch_size: int = 1,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.patch_size = int(patch_size)
        self.dropout_prob = float(dropout_prob)
        self.layers = tuple(int(v) for v in layers)
        self.use_bf16 = bool(use_bf16)
        self.input_patch_size = int(input_patch_size)

        self.dtype = torch.bfloat16 if (self.use_bf16 and torch.cuda.is_available()) else torch.float32

        enc_in_channels = self.in_channels * self.input_patch_size * self.input_patch_size
        self.encoder = _ResNetEncoder(
            base_channels=self.base_channels,
            layers=self.layers,
            dropout_prob=self.dropout_prob,
            dtype=self.dtype,
        )
        # override first conv in encoder to match channels after patch_input
        self.encoder.conv1 = nn.Conv2d(enc_in_channels, self.base_channels, kernel_size=3, stride=1, padding=1, bias=False, dtype=self.dtype)

        self.decoder = _UNetDecoder(
            base_channels=self.base_channels,
            out_channels=self.in_channels * self.input_patch_size * self.input_patch_size,
            dtype=self.dtype,
        )
        self.fc = nn.Linear(self.base_channels * 8, self.num_classes, dtype=self.dtype)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        *,
        lambda_cls: float = 0.0,
        mask_ratio_min: float = 0.75,
        mask_ratio_max: float = 0.75,
        train: bool = True,
        rng: Optional[torch.Generator] = None,
    ):
        x = x.to(dtype=self.dtype)
        x = patch_input(x, self.input_patch_size)

        if rng is None:
            rng = torch.Generator(device=x.device)
            rng.manual_seed(torch.randint(0, 2**31 - 1, (1,), device=x.device).item())

        b = x.shape[0]
        mask_ratio = torch.rand((b,), dtype=self.dtype, device=x.device, generator=rng) * (mask_ratio_max - mask_ratio_min) + mask_ratio_min
        mask = make_patch_mask(x, rng, mask_ratio, self.patch_size)
        x_in = x * (1.0 - mask)

        feats = self.encoder(x_in, train=train)
        top = feats["layer4"]
        pooled = top.mean(dim=(2, 3))
        logits = self.fc(pooled)
        recon = self.decoder(feats).permute(0, 2, 3, 1).contiguous()

        one_hot = torch.nn.functional.one_hot(labels.long(), num_classes=self.num_classes).to(dtype=self.dtype)
        cls_loss = -(one_hot * torch.nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1)

        mse = (recon - x) ** 2
        recon_loss = (mse * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-8)
        loss = float(lambda_cls) * cls_loss + (1.0 - float(lambda_cls)) * recon_loss

        metrics = {
            "loss": loss,
            "cls_loss": cls_loss,
            "recon_loss": recon_loss,
            "accuracy": (torch.argmax(logits, dim=-1) == labels).to(dtype=self.dtype),
            "mask_ratio": mask.mean(dim=(1, 2, 3)),
        }
        return loss, metrics

    def get_activations(
        self,
        x: torch.Tensor,
        *,
        patch_mean_size: Optional[List[int]] = None,
        patch_std_size: Optional[List[int]] = None,
        use_std: bool = True,
        use_mean: bool = True,
        every_k_block: float = 2,
    ) -> Dict[str, torch.Tensor]:
        patch_mean_size = patch_mean_size or [2, 4]
        patch_std_size = patch_std_size or [2, 4]

        x = x.to(dtype=self.dtype)
        x = patch_input(x, self.input_patch_size)

        need_blocks = isinstance(every_k_block, (int, float)) and not math.isinf(float(every_k_block)) and every_k_block >= 1
        if need_blocks:
            feats, block_outputs = self.encoder(x, train=False, return_block_outputs=True)
        else:
            feats = self.encoder(x, train=False)
            block_outputs = {}

        out: Dict[str, torch.Tensor] = {}
        out["norm_x"] = torch.sqrt((x**2).mean(dim=(1, 2)) + 1e-6)[:, None, :]

        def process_feat(name: str, feat: torch.Tensor) -> None:
            feat_hw = feat.permute(0, 2, 3, 1).contiguous()
            b, h, w, c = feat_hw.shape
            out[name] = rearrange(feat_hw, "b h w c -> b (h w) c")
            if use_mean:
                out[f"{name}_mean"] = feat_hw.mean(dim=(1, 2))[:, None, :]
            if use_std:
                out[f"{name}_std"] = safe_std(feat_hw, axis=(1, 2))[:, None, :]

            for size in patch_mean_size:
                if h % size == 0 and w % size == 0:
                    reshaped = rearrange(feat_hw, "b (h s1) (w s2) c -> b (h w) (s1 s2) c", s1=size, s2=size)
                    out[f"{name}_mean_{size}"] = reshaped.mean(dim=2)

            for size in patch_std_size:
                if h % size == 0 and w % size == 0:
                    reshaped = rearrange(feat_hw, "b (h s1) (w s2) c -> b (h w) (s1 s2) c", s1=size, s2=size)
                    out[f"{name}_std_{size}"] = safe_std(reshaped, axis=2)

        for name, feat in feats.items():
            process_feat(name, feat)

        if need_blocks:
            k = int(every_k_block)
            for i in range(1, 5):
                lname = f"layer{i}"
                blocks = block_outputs.get(lname, [])
                for blk_idx, feat_i in enumerate(blocks, start=1):
                    if blk_idx % k == 0:
                        process_feat(f"{lname}_blk{blk_idx}", feat_i)

        return out

    def dummy_input(self) -> Dict[str, Any]:
        p = self.input_patch_size
        return {
            "x": torch.zeros((1, 32 * p, 32 * p, self.in_channels), dtype=torch.float32),
            "labels": torch.zeros((1,), dtype=torch.long),
            "lambda_cls": 0.0,
            "mask_ratio_min": 0.75,
            "mask_ratio_max": 0.75,
            "train": False,
        }


def load_mae_hf(
    name: str,
    *,
    dir: str = HF_ROOT,
) -> Tuple[MAEResNetJAX, Any, Dict[str, Any]]:
    from models.hf import load_mae_torch

    return load_mae_torch(
        name,
        repo_id=HF_REPO_ID,
        prefix=None,
        output_root=dir,
    )


def _mae_from_metadata(metadata: Dict[str, Any]) -> MAEResNetJAX:
    model_config = dict(metadata.get("model_config", {}) or {})
    num_classes = int(model_config.pop("num_classes", 1000))
    return MAEResNetJAX(num_classes=num_classes, **model_config)


def build_feature_model_and_params(
    path: str = "",
    use_convnext: bool = False,
    convnext_bf16: bool = False,
):
    if use_convnext:
        from models.convnext import load_convnext_torch_model

        return load_convnext_torch_model(model_name="base", use_bf16=convnext_bf16)

    if not path:
        raise ValueError("`path` is required when use_convnext=False.")

    from utils.init_util import load_init_entry

    entry, metadata = load_init_entry("mae", path, hf_cache_dir=HF_ROOT)
    if not metadata:
        raise ValueError(f"MAE artifact is missing metadata required to rebuild the model: {path}")
    feature_model = _mae_from_metadata(metadata)
    missing, unexpected = feature_model.load_state_dict(entry, strict=False)
    if missing or unexpected:
        raise ValueError(f"Failed to load MAE weights cleanly. missing={missing[:8]} unexpected={unexpected[:8]}")
    feature_model.eval()
    for p in feature_model.parameters():
        p.requires_grad_(False)
    if _COMPILE:
        feature_model.encoder = torch.compile(feature_model.encoder, dynamic=True)
    return feature_model, feature_model.state_dict()


def build_activation_function(
    mae_path: str = "",
    use_convnext=False,
    convnext_bf16=False,
    use_mae=True,
    postprocess_fn=lambda x: x,
):
    variables = {}
    feature_model = None
    convnext_model = None

    if use_mae:
        feature_model, feature_params = build_feature_model_and_params(path=mae_path)
        variables["mae_params"] = feature_params

    if use_convnext:
        convnext_model, convnext_feature_params = build_feature_model_and_params(use_convnext=True, convnext_bf16=convnext_bf16)
        variables["convnext_params"] = convnext_feature_params

    def activation_fn(params, x, convnext_kwargs=dict(), has_scale=False, **kwargs):
        del params
        usual_feats = {}
        usual_feats["global"] = x.reshape(x.shape[0], 1, -1)
        if has_scale:
            usual_feats["norm_x"] = torch.sqrt((x**2).mean(dim=(1, 2)) + 1e-6)[:, None, :]

        if use_mae:
            feature_model.to(x.device)
            mae_feats = feature_model.get_activations(x, **kwargs)
            usual_feats = {**usual_feats, **mae_feats}

        if use_convnext:
            convnext_model.to(x.device)
            xx = postprocess_fn(x)
            xx = xx.permute(0, 2, 3, 1).contiguous().float()
            mean = torch.tensor([0.485, 0.456, 0.406], device=xx.device)
            std = torch.tensor([0.229, 0.224, 0.225], device=xx.device)
            xx = (xx - mean) / std
            convnext_feats = convnext_model.get_activations(xx, **convnext_kwargs)
            usual_feats = {**usual_feats, **convnext_feats}
        return usual_feats

    return activation_fn, variables
