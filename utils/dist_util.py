from __future__ import annotations

import os
from contextlib import suppress
from typing import Any, Callable

import torch
import torch.distributed as dist


def dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def process_count() -> int:
    if dist_is_initialized():
        return int(dist.get_world_size())
    return 1


def process_index() -> int:
    if dist_is_initialized():
        return int(dist.get_rank())
    return 0


def local_device_count() -> int:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        return max(1, int(n))
    return 1


def device_count() -> int:
    return process_count() * local_device_count()


def barrier() -> None:
    if dist_is_initialized():
        dist.barrier()


def sync_global_devices(_: str = "") -> None:
    barrier()


def init_distributed() -> None:
    if dist_is_initialized():
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend=backend)


def to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(to_device(v, device) for v in x)
    return x


def tree_map(fn: Callable[[Any], Any], x: Any) -> Any:
    if isinstance(x, dict):
        return {k: tree_map(fn, v) for k, v in x.items()}
    if isinstance(x, list):
        return [tree_map(fn, v) for v in x]
    if isinstance(x, tuple):
        return tuple(tree_map(fn, v) for v in x)
    return fn(x)


def tree_map2(fn: Callable[[Any, Any], Any], a: Any, b: Any) -> Any:
    if isinstance(a, dict):
        return {k: tree_map2(fn, a[k], b[k]) for k in a}
    if isinstance(a, list):
        return [tree_map2(fn, x, y) for x, y in zip(a, b)]
    if isinstance(a, tuple):
        return tuple(tree_map2(fn, x, y) for x, y in zip(a, b))
    return fn(a, b)


def detach_clone_tree(x: Any) -> Any:
    return tree_map(lambda t: t.detach().clone() if torch.is_tensor(t) else t, x)


def _gather_tensor(x: torch.Tensor) -> torch.Tensor:
    if not dist_is_initialized():
        return x
    world = process_count()
    if world == 1:
        return x

    orig_device = x.device
    if x.device.type == "cpu" and torch.cuda.is_available():
        x = x.cuda()

    shape = torch.tensor(x.shape, device=x.device, dtype=torch.int64)
    all_shapes = [torch.zeros_like(shape) for _ in range(world)]
    dist.all_gather(all_shapes, shape)
    max_shape = torch.stack(all_shapes, dim=0).max(dim=0).values.tolist()

    pad_sizes = []
    for i in range(x.ndim - 1, -1, -1):
        pad_sizes.extend([0, int(max_shape[i] - x.shape[i])])
    padded = torch.nn.functional.pad(x, pad_sizes)

    gathered = [torch.zeros_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded)

    outs = []
    for g, s in zip(gathered, all_shapes):
        slicer = tuple(slice(0, int(v.item())) for v in s)
        outs.append(g[slicer])
    result = torch.cat(outs, dim=0)

    if orig_device.type == "cpu":
        result = result.cpu()
    return result


def process_allgather(x: Any, tiled: bool = True) -> Any:
    del tiled
    if torch.is_tensor(x):
        return _gather_tensor(x)
    if isinstance(x, dict):
        return {k: process_allgather(v, tiled=tiled) for k, v in x.items()}
    if isinstance(x, list):
        return [process_allgather(v, tiled=tiled) for v in x]
    if isinstance(x, tuple):
        return tuple(process_allgather(v, tiled=tiled) for v in x)
    return x


def maybe_ddp_model(model: torch.nn.Module, device_ids: list[int] | None = None) -> torch.nn.Module:
    if not dist_is_initialized():
        return model

    if torch.cuda.is_available():
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
    return torch.nn.parallel.DistributedDataParallel(model)


def unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def cleanup_distributed() -> None:
    with suppress(Exception):
        if dist_is_initialized():
            dist.destroy_process_group()
