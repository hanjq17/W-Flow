from __future__ import annotations

from functools import partial
from typing import Any, Tuple

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from utils.env import VAE_HF_PATH

_vae_cache = {}


def _to_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device)
    return torch.as_tensor(x, device=device)


def vae_enc_decode(replicate_params: bool = True):
    del replicate_params
    cache_key = ("vae_enc_decode",)
    if cache_key in _vae_cache:
        return _vae_cache[cache_key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(VAE_HF_PATH).to(device)
    vae.eval()

    @torch.no_grad()
    def _encode_fn(images, rng=None):
        x = _to_tensor(images, device)
        if x.ndim != 4:
            raise ValueError(f"expected 4D tensor BCHW, got {tuple(x.shape)}")
        x = x.float()
        posterior = vae.encode(x).latent_dist
        if isinstance(rng, torch.Generator):
            latents = posterior.sample(generator=rng)
        elif isinstance(rng, int):
            g = torch.Generator(device=device)
            g.manual_seed(int(rng))
            latents = posterior.sample(generator=g)
        else:
            latents = posterior.sample()
        latents = latents * 0.18215
        return latents.permute(0, 2, 3, 1).contiguous()

    @torch.no_grad()
    def _decode_fn(latents):
        z = _to_tensor(latents, device)
        if z.ndim != 4:
            raise ValueError(f"expected 4D tensor BHWC, got {tuple(z.shape)}")
        z = z.permute(0, 3, 1, 2).contiguous().float()
        out = vae.decode(z / 0.18215).sample
        return out

    result = (partial(_encode_fn), partial(_decode_fn))
    _vae_cache[cache_key] = result
    return result
