from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint

from utils.env import HF_REPO_ID, HF_ROOT


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    embed_dim_half = embed_dim // 2
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim_half, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim_half, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _init_linear_like_torch(linear: nn.Linear, weight_init: str = "xavier_uniform", bias_init: str = "zeros"):
    if weight_init == "xavier_uniform":
        nn.init.xavier_uniform_(linear.weight)
    elif weight_init == "zeros":
        nn.init.zeros_(linear.weight)
    elif weight_init == "normal":
        nn.init.normal_(linear.weight, std=0.02)
    else:
        nn.init.xavier_uniform_(linear.weight)

    if linear.bias is not None:
        if bias_init == "zeros":
            nn.init.zeros_(linear.bias)
        else:
            nn.init.constant_(linear.bias, 0.0)


class TorchLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init: str = "xavier_uniform",
        bias_init: str = "zeros",
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        _init_linear_like_torch(self.linear, weight_init=weight_init, bias_init=bias_init)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor):
        return self.linear(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            normed = normed * self.weight
        return normed.to(dtype=input_dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, dtype: torch.dtype = torch.float32):
    B, N, H, D = q.shape
    half_dim = D // 2
    freqs = (1.0 / (10000 ** (torch.arange(0, half_dim, device=q.device, dtype=torch.float32) / half_dim))).to(dtype)
    t = torch.arange(N, device=q.device, dtype=dtype)
    freqs = torch.outer(t, freqs)
    emb = torch.cat([freqs, freqs], dim=-1)

    cos = torch.cos(emb)[None, :, None, :]
    sin = torch.sin(emb)[None, :, None, :]

    def rotate_half(x_):
        x1, x2 = x_[..., :half_dim], x_[..., half_dim:]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = TorchLinear(hidden_size, intermediate_size, bias=True)
        self.w3 = TorchLinear(hidden_size, intermediate_size, bias=True)
        self.w2 = TorchLinear(intermediate_size, hidden_size, bias=True)

    def forward(self, x):
        out = torch.nn.functional.silu(self.w1(x)) * self.w3(x)
        return self.w2(out)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_rmsnorm: bool = False,
        use_rope: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_fp32: bool = True,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.qk_norm = bool(qk_norm)
        self.use_rmsnorm = bool(use_rmsnorm)
        self.use_rope = bool(use_rope)
        self.attn_drop = float(attn_drop)
        self.proj_drop = float(proj_drop)
        self.attn_fp32 = bool(attn_fp32)

        self.qkv = TorchLinear(self.dim, self.dim * 3, bias=qkv_bias)
        if self.qk_norm:
            head_dim = self.dim // self.num_heads
            if self.use_rmsnorm:
                self.q_norm = RMSNorm(head_dim)
                self.k_norm = RMSNorm(head_dim)
            else:
                self.q_norm = nn.LayerNorm(head_dim, eps=1e-6, elementwise_affine=True)
                self.k_norm = nn.LayerNorm(head_dim, eps=1e-6, elementwise_affine=True)
        else:
            self.q_norm = None
            self.k_norm = None
        self.proj = TorchLinear(self.dim, self.dim, bias=True)
        self.attn_dropout = nn.Dropout(self.attn_drop)
        self.proj_dropout = nn.Dropout(self.proj_drop)

        if self.use_rope:
            head_dim = self.dim // self.num_heads
            half_dim = head_dim // 2
            freqs = 1.0 / (10000 ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
            t = torch.arange(max_seq_len, dtype=torch.float32)
            freqs = torch.outer(t, freqs)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.register_buffer("_rope_cos", torch.cos(emb)[None, :, None, :], persistent=False)
            self.register_buffer("_rope_sin", torch.sin(emb)[None, :, None, :], persistent=False)

    def forward(self, x: torch.Tensor, deterministic: bool = True, return_qk: bool = False):
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim)
        q, k, v = qkv[:, :, 0, :, :], qkv[:, :, 1, :, :], qkv[:, :, 2, :, :]

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            rope_dtype = torch.float32 if self.attn_fp32 else q.dtype
            cos = self._rope_cos[:, :N].to(dtype=rope_dtype)
            sin = self._rope_sin[:, :N].to(dtype=rope_dtype)
            half_dim = self.dim // self.num_heads // 2
            q1, q2 = q[..., :half_dim], q[..., half_dim:]
            k1, k2 = k[..., :half_dim], k[..., half_dim:]
            q = (q * cos) + (torch.cat([-q2, q1], dim=-1) * sin)
            k = (k * cos) + (torch.cat([-k2, k1], dim=-1) * sin)

        qk = (q, k) if return_qk else None

        if self.attn_fp32:
            q = q.float()
            k = k.float()
            v = v.float()

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        dp = self.attn_drop if not deterministic else 0.0

        if self.attn_fp32:
            with torch.amp.autocast(device_type=q.device.type, enabled=False):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dp)
        else:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dp)

        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        if self.proj_drop > 0.0 and not deterministic:
            out = self.proj_dropout(out)
        return out, qk


class StandardMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_dim: int):
        super().__init__()
        self.fc1 = TorchLinear(hidden_size, mlp_hidden_dim, bias=True)
        self.fc2 = TorchLinear(mlp_hidden_dim, hidden_size, bias=True)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.fc1(x), approximate="none")
        return self.fc2(h)


class LightningDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = False,
        use_swiglu: bool = False,
        use_rmsnorm: bool = False,
        cond_dim: Optional[int] = None,
        use_rope: bool = False,
        attn_fp32: bool = True,
        max_seq_len: int = 256,
    ):
        super().__init__()
        del cond_dim
        self.hidden_size = int(hidden_size)
        self.use_swiglu = bool(use_swiglu)
        self.use_rmsnorm = bool(use_rmsnorm)

        if self.use_rmsnorm:
            self.norm1 = RMSNorm(self.hidden_size)
            self.norm2 = RMSNorm(self.hidden_size)
        else:
            self.norm1 = nn.LayerNorm(self.hidden_size, eps=1e-6, elementwise_affine=False)
            self.norm2 = nn.LayerNorm(self.hidden_size, eps=1e-6, elementwise_affine=False)

        self.attn = Attention(
            dim=self.hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            use_rope=use_rope,
            attn_fp32=attn_fp32,
            max_seq_len=max_seq_len,
        )

        mlp_hidden_dim = int(self.hidden_size * float(mlp_ratio))
        if self.use_swiglu:
            hid_size = int(2 / 3 * mlp_hidden_dim)
            hid_size = (hid_size + 31) // 32 * 32
            self.mlp = SwiGLUFFN(self.hidden_size, hid_size)
        else:
            self.mlp = StandardMLP(self.hidden_size, mlp_hidden_dim)

        self.adaLN_mod = nn.Sequential(
            nn.SiLU(),
            TorchLinear(self.hidden_size, 6 * self.hidden_size, bias=True, weight_init="zeros", bias_init="zeros"),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, deterministic: bool = True):
        chunks = self.adaLN_mod(c.float()).to(dtype=x.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(chunks, 6, dim=1)

        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm, deterministic=deterministic)[0]

        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        use_rmsnorm: bool = False,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        del cond_dim
        self.hidden_size = int(hidden_size)
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)

        self.norm_final = RMSNorm(self.hidden_size) if use_rmsnorm else nn.LayerNorm(self.hidden_size, eps=1e-6, elementwise_affine=False)
        self.adaLN_mod = nn.Sequential(
            nn.SiLU(),
            TorchLinear(self.hidden_size, 2 * self.hidden_size, bias=True, weight_init="zeros", bias_init="zeros"),
        )
        self.linear = TorchLinear(
            self.hidden_size,
            self.patch_size * self.patch_size * self.out_channels,
            bias=True,
            weight_init="zeros",
            bias_init="zeros",
        )

    def forward(self, x, c):
        shift, scale = torch.chunk(self.adaLN_mod(c.float()).to(dtype=x.dtype), 2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class LightningDiT(nn.Module):
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 32,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        out_channels: int = 32,
        use_qknorm: bool = False,
        use_swiglu: bool = False,
        use_rope: bool = False,
        use_rmsnorm: bool = False,
        cond_dim: Optional[int] = None,
        n_cls_tokens: int = 0,
        attn_fp32: bool = True,
        use_remat: bool = False,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.out_channels = int(out_channels)
        self.use_qknorm = bool(use_qknorm)
        self.use_swiglu = bool(use_swiglu)
        self.use_rope = bool(use_rope)
        self.use_rmsnorm = bool(use_rmsnorm)
        self.cond_dim = cond_dim
        self.n_cls_tokens = int(n_cls_tokens)
        self.attn_fp32 = bool(attn_fp32)
        self.use_remat = bool(use_remat)

        patch_in_dim = self.patch_size * self.patch_size * self.in_channels
        self.patch_embed = TorchLinear(patch_in_dim, self.hidden_size, bias=True)

        target_grid = self.input_size // self.patch_size
        max_seq_len = target_grid * target_grid + self.n_cls_tokens

        self.blocks = nn.ModuleList(
            [
                LightningDiTBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    use_qknorm=self.use_qknorm,
                    use_swiglu=self.use_swiglu,
                    use_rmsnorm=self.use_rmsnorm,
                    cond_dim=self.cond_dim,
                    use_rope=self.use_rope,
                    attn_fp32=self.attn_fp32,
                    max_seq_len=max_seq_len,
                )
                for _ in range(self.depth)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
            use_rmsnorm=self.use_rmsnorm,
            cond_dim=self.cond_dim,
        )

        target_grid = self.input_size // self.patch_size
        num_patches = target_grid * target_grid
        pe = get_2d_sincos_pos_embed(self.hidden_size, target_grid)
        self.pos_embed = nn.Parameter(torch.as_tensor(pe, dtype=torch.float32).unsqueeze(0), requires_grad=True)
        if self.n_cls_tokens > 0:
            self.cls_embed = nn.Parameter(torch.randn(1, self.n_cls_tokens, self.hidden_size) * 0.02, requires_grad=True)
            if self.cond_dim is None:
                raise ValueError("cond_dim must be set when n_cls_tokens > 0")
            self.c_token_proj = TorchLinear(int(self.cond_dim), self.hidden_size, bias=True)
        else:
            self.register_parameter("cls_embed", None)
            self.c_token_proj = None

    def _patch_embed(self, x):
        B, H, W, C = x.shape
        p = self.patch_size
        target_grid = self.input_size // p
        effective_p = H // target_grid

        x = x.reshape(B, target_grid, effective_p, target_grid, effective_p, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B, target_grid * target_grid, effective_p * effective_p * C)
        return self.patch_embed(x)

    def forward(self, x: torch.Tensor, c: torch.Tensor, deterministic: bool = True):
        x = self._patch_embed(x)
        x = (x + self.pos_embed.to(device=x.device, dtype=x.dtype)).to(dtype=x.dtype)

        if self.n_cls_tokens > 0:
            c_tokens = self.c_token_proj(c)
            c_tokens = c_tokens.unsqueeze(1).repeat(1, self.n_cls_tokens, 1)
            c_tokens = c_tokens + self.cls_embed.to(device=x.device, dtype=x.dtype)
            x = torch.cat([c_tokens, x], dim=1)

        for blk in self.blocks:
            if self.use_remat and self.training:
                x = torch.utils.checkpoint.checkpoint(lambda _x, _c: blk(_x, _c, deterministic), x, c, use_reentrant=False)
            else:
                x = blk(x, c, deterministic)

        x = self.final_layer(x, c)

        if self.n_cls_tokens > 0:
            x = x[:, self.n_cls_tokens :, :]

        B = x.shape[0]
        p = self.patch_size
        out_size = self.input_size
        grid_h = out_size // p
        grid_w = out_size // p
        x = x.reshape(B, grid_h, grid_w, p, p, self.out_channels)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B, out_size, out_size, self.out_channels)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.fc1 = TorchLinear(self.frequency_embedding_size, self.hidden_size, bias=True, weight_init="normal")
        self.fc2 = TorchLinear(self.hidden_size, self.hidden_size, bias=True, weight_init="normal")

    def forward(self, t: torch.Tensor):
        t = t.to(dtype=torch.float32)
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        h = self.fc1(embedding)
        h = torch.nn.functional.silu(h)
        h = self.fc2(h)
        return h


class DitGen(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        num_classes: int = 1001,
        noise_classes: int = 0,
        noise_coords: int = 1,
        input_size: int = 32,
        in_channels: int = 3,
        n_cls_tokens: int = 0,
        patch_size: int = 2,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        out_channels: int = 3,
        use_qknorm: bool = False,
        use_swiglu: bool = False,
        use_rope: bool = False,
        use_rmsnorm: bool = False,
        use_bf16: bool = False,
        attn_fp32: bool = True,
        use_remat: bool = False,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.num_classes = int(num_classes)
        self.noise_classes = int(noise_classes)
        self.noise_coords = int(noise_coords)
        self.input_size = int(input_size)
        self.in_channels = int(in_channels)
        self.n_cls_tokens = int(n_cls_tokens)
        self.patch_size = int(patch_size)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.out_channels = int(out_channels)
        self.use_qknorm = bool(use_qknorm)
        self.use_swiglu = bool(use_swiglu)
        self.use_rope = bool(use_rope)
        self.use_rmsnorm = bool(use_rmsnorm)
        self.use_bf16 = bool(use_bf16)
        self.attn_fp32 = bool(attn_fp32)
        self.use_remat = bool(use_remat)

        self.class_embed = nn.Embedding(self.num_classes, self.cond_dim)
        nn.init.normal_(self.class_embed.weight, std=0.02)

        self.noise_embeds = nn.ModuleList()
        if self.noise_classes > 0:
            for _ in range(self.noise_coords):
                emb = nn.Embedding(self.noise_classes, self.cond_dim)
                nn.init.normal_(emb.weight, std=0.02)
                self.noise_embeds.append(emb)

        self.cfg_embedder = TimestepEmbedder(self.cond_dim)
        self.cfg_norm = RMSNorm(self.cond_dim)

        self.model = LightningDiT(
            input_size=self.input_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            hidden_size=self.hidden_size,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            out_channels=self.out_channels,
            use_qknorm=self.use_qknorm,
            use_swiglu=self.use_swiglu,
            use_rope=self.use_rope,
            use_rmsnorm=self.use_rmsnorm,
            cond_dim=self.cond_dim,
            n_cls_tokens=self.n_cls_tokens,
            attn_fp32=self.attn_fp32,
            use_remat=self.use_remat,
        )

    def dummy_input(self):
        return {
            "c": torch.ones(1, dtype=torch.long),
            "cfg_scale": 1.0,
            "temp": 1.0,
            "deterministic": True,
        }

    def rng_keys(self):
        return ["noise"]

    def generate_image(self, x, cond, deterministic=True):
        return self.model(x, cond, deterministic=deterministic)

    def c_cfg_noise_to_cond(self, c, cfg_scale, noise_labels):
        B = c.shape[0]
        cond = self.class_embed(c)
        if self.noise_classes > 0:
            for i in range(self.noise_coords):
                cond = cond + self.noise_embeds[i](noise_labels[:, i])

        if isinstance(cfg_scale, (float, int)):
            cfg_scale_t = torch.full((B,), float(cfg_scale), device=c.device, dtype=torch.float32)
        else:
            cfg_scale_t = torch.as_tensor(cfg_scale, device=c.device, dtype=torch.float32)
            if cfg_scale_t.ndim == 0:
                cfg_scale_t = cfg_scale_t[None].repeat(B)

        # keep timestep embedder path in fp32
        cfg_scale_t = self.cfg_embedder(cfg_scale_t.float())
        cfg_scale_t = self.cfg_norm(cfg_scale_t.float())

        # match cond dtype only at the end
        cfg_scale_t = cfg_scale_t.to(dtype=cond.dtype)
        cond = cond + cfg_scale_t * 0.02

        return cond

    def forward(self, c, cfg_scale=1.0, temp=1.0, deterministic=True, train=False, rng=None):
        del train
        B = c.shape[0]
        device = c.device
        if rng is None:
            rng = torch.Generator(device=device)
            rng.manual_seed(torch.randint(0, 2**31 - 1, (1,), device=device).item())

        x = torch.randn((B, self.input_size, self.input_size, self.in_channels), generator=rng, device=device)
        x = x * float(temp)

        if self.noise_classes > 0:
            noise_labels = torch.randint(
                low=0,
                high=max(1, self.noise_classes),
                size=(B, max(1, self.noise_coords)),
                generator=rng,
                device=device,
            )
        else:
            noise_labels = torch.zeros((B, max(1, self.noise_coords)), dtype=torch.long, device=device)

        _use_autocast = self.use_bf16 and device.type == "cuda"
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=_use_autocast):
            cond = self.c_cfg_noise_to_cond(c, cfg_scale, noise_labels)
            samples = self.generate_image(x, cond, deterministic=deterministic)

        return {
            "samples": samples,
            "noise": {
                "x": x,
                "noise_labels": noise_labels,
            },
        }


def build_generator_from_config(model_config: Dict[str, Any]) -> DitGen:
    return DitGen(**dict(model_config))


def load_hf(
    name: str,
    *,
    dir: str = HF_ROOT,
):
    from models.hf import load_generator_torch

    return load_generator_torch(
        name=name,
        repo_id=HF_REPO_ID,
        prefix=None,
        output_root=dir,
    )
