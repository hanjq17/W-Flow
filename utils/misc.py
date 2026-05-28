from __future__ import annotations

import os
import random
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import yaml

from utils.dist_util import init_distributed


# adapted from https://github.com/NVlabs/edm
class EasyDict(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value):
        self[name] = value


def _dict_to_easydict(d):
    if not isinstance(d, dict):
        return d
    out = EasyDict()
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _dict_to_easydict(v)
        elif isinstance(v, list):
            out[k] = [_dict_to_easydict(i) for i in v]
        else:
            out[k] = v
    return out


def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return _dict_to_easydict(yaml.safe_load(f))


def prepare_rng(seed_or_gen: int | torch.Generator | None, tags=("params", "dropout")):
    del tags
    if isinstance(seed_or_gen, torch.Generator):
        return seed_or_gen
    gen = torch.Generator(device="cpu")
    if seed_or_gen is None:
        seed_or_gen = 0
    gen.manual_seed(int(seed_or_gen))
    return gen


_did_run_init = False


def run_init():
    global _did_run_init
    if _did_run_init:
        return
    init_distributed()
    _did_run_init = True


_rand_cache = {}


def ddp_rand_func(rand_type="normal", shard="ddp"):
    del shard
    key = rand_type
    if key in _rand_cache:
        return _rand_cache[key]

    def _normal(rng: torch.Generator, shape):
        return torch.randn(shape, generator=rng)

    def _uniform(rng: torch.Generator, shape):
        return torch.rand(shape, generator=rng)

    if rand_type == "normal":
        fn = _normal
    elif rand_type == "uniform":
        fn = _uniform
    else:
        raise ValueError(rand_type)
    _rand_cache[key] = fn
    return fn


def _profile_log(report: list[str], msg: str, *, console_print: bool) -> None:
    report.append(msg)
    if console_print:
        print(msg, flush=True)


def profile_func(
    target_fn: Callable,
    args: tuple,
    kwargs: Optional[Dict] = None,
    name: str = "Model",
    console_print: bool = True,
    hardware_peak_bw: float = 1600.0,
    actual_run: bool = False,
    n_loops: int = 10,
    print_hlo: bool = False,
):
    del hardware_peak_bw, print_hlo
    kwargs = kwargs or {}
    report: list[str] = []
    metrics: Dict[str, float] = {}

    _profile_log(report, f"[Profile] Inspecting '{name}'", console_print=console_print)

    if actual_run:
        with torch.no_grad():
            _ = target_fn(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_loops):
                _ = target_fn(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / max(1, n_loops)
        metrics["profile/Time_ms"] = dt * 1000.0
        _profile_log(report, f"[Profile] Runtime: time={metrics['profile/Time_ms']:.2f} ms", console_print=console_print)

    return metrics


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
