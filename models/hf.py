"""Minimal Hugging Face helpers for Drift artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from utils.env import HF_ROOT
from utils.param_convert import convert_generator_tree_to_state_dict, convert_mae_tree_to_state_dict


def read_metadata(artifact_dir: Path) -> Dict[str, Any]:
    return json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))


def load_jax_ema_params(artifact_dir: Path) -> Any:
    msgpack_path = artifact_dir / "ema_params.msgpack"
    if msgpack_path.is_file():
        try:
            from flax import serialization

            return serialization.msgpack_restore(msgpack_path.read_bytes())
        except Exception as e:  # pragma: no cover - fallback path
            raise RuntimeError(f"Failed to read JAX msgpack params: {msgpack_path}") from e

    pt_path = artifact_dir / "ema_params.pt"
    if pt_path.is_file():
        return torch.load(pt_path, map_location="cpu", weights_only=False)

    raise FileNotFoundError(f"No ema_params.msgpack or ema_params.pt found in {artifact_dir}")


def _download_artifact(
    *,
    repo_id: str,
    kind: str,
    backend: str,
    model_id: str,
    output_root: str,
    prefix: Optional[str],
) -> Path:
    from huggingface_hub import snapshot_download

    local_root = Path(output_root).resolve() / "models" / kind / backend / model_id
    local_root.mkdir(parents=True, exist_ok=True)
    root = f"models/{kind}/{backend}/{model_id}"
    path_in_repo = f"{prefix.strip('/')}/{root}" if prefix else root

    # snapshot_download(
    #     repo_id=repo_id,
    #     repo_type="model",
    #     allow_patterns=[f"{path_in_repo}/*"],
    #     local_dir=str(local_root),
    # )
    nested = local_root / path_in_repo
    return nested if nested.exists() else local_root


def _load_torch_or_convert(
    model: torch.nn.Module,
    raw_params: Any,
    converter,
) -> Dict[str, torch.Tensor]:
    target = model.state_dict()

    if isinstance(raw_params, dict):
        is_flat = all(isinstance(k, str) for k in raw_params.keys())
        if is_flat and set(raw_params.keys()) == set(target.keys()):
            out = {k: torch.as_tensor(v).to(dtype=target[k].dtype) for k, v in raw_params.items()}
            return out

    return converter(raw_params, target)


def load_mae_torch(
    name: str,
    *,
    repo_id: str,
    prefix: Optional[str] = None,
    output_root: str = HF_ROOT,
) -> Tuple[Any, Any, Dict[str, Any]]:
    artifact_dir = _download_artifact(
        repo_id=repo_id,
        kind="mae",
        backend="jax",
        model_id=name,
        output_root=output_root,
        prefix=prefix,
    )
    metadata = read_metadata(artifact_dir)

    from models.mae_model import _mae_from_metadata

    module = _mae_from_metadata(metadata)
    raw_params = load_jax_ema_params(artifact_dir)
    state_dict = _load_torch_or_convert(module, raw_params, convert_mae_tree_to_state_dict)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"Failed to load MAE weights cleanly. missing={missing[:8]} unexpected={unexpected[:8]}")
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module, module.state_dict(), metadata


def load_generator_torch(
    name: str,
    *,
    repo_id: str,
    prefix: Optional[str] = None,
    output_root: str = HF_ROOT,
) -> Tuple[Any, Any, Dict[str, Any]]:
    artifact_dir = _download_artifact(
        repo_id=repo_id,
        kind="gen",
        backend="jax",
        model_id=name,
        output_root=output_root,
        prefix=prefix,
    )
    metadata = read_metadata(artifact_dir)

    model_cfg = dict(metadata.get("model_config", {}) or {})
    if not model_cfg:
        raise ValueError(
            f"Generator artifact is missing metadata.model_config and cannot be restored: {name}"
        )

    from models.generator import build_generator_from_config

    module = build_generator_from_config(model_cfg)
    raw_params = load_jax_ema_params(artifact_dir)
    state_dict = _load_torch_or_convert(module, raw_params, convert_generator_tree_to_state_dict)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"Failed to load generator weights cleanly. missing={missing[:8]} unexpected={unexpected[:8]}")
    module.eval()
    return module, module.state_dict(), metadata
