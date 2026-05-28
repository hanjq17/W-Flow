from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from models.hf import load_generator_torch, load_jax_ema_params, load_mae_torch, read_metadata
from utils.dist_util import unwrap_ddp
from utils.env import HF_ROOT
from utils.param_convert import convert_generator_tree_to_state_dict, convert_mae_tree_to_state_dict


def resolve_artifact_dir(path: str) -> Path:
    base = Path(path).resolve()
    params_ema_dir = base / "params_ema"
    ckpt_dir = base / "checkpoints"
    if params_ema_dir.is_dir():
        return params_ema_dir
    if ckpt_dir.is_dir():
        return ckpt_dir
    return base


def put_like(tree: Any, template: Any) -> Any:
    if isinstance(template, dict) and isinstance(tree, dict):
        out = {}
        for k, v in tree.items():
            if k in template and torch.is_tensor(template[k]):
                out[k] = torch.as_tensor(v, dtype=template[k].dtype)
            else:
                out[k] = v
        return out
    if torch.is_tensor(template):
        return torch.as_tensor(tree, dtype=template.dtype)
    return tree


def _load_local_init_entry(path: str) -> Tuple[Any, Dict[str, Any]]:
    artifact_dir = resolve_artifact_dir(path)
    metadata_path = artifact_dir / "metadata.json"
    params_path = artifact_dir / "ema_params.msgpack"
    params_pt_path = artifact_dir / "ema_params.pt"
    legacy_meta_path = artifact_dir / "ema_model.metadata.json"
    legacy_params_path = artifact_dir / "ema_model.msgpack"

    if metadata_path.is_file() and (params_path.is_file() or params_pt_path.is_file()):
        return load_jax_ema_params(artifact_dir), read_metadata(artifact_dir)
    if params_path.is_file() or params_pt_path.is_file():
        return load_jax_ema_params(artifact_dir), {}

    if legacy_meta_path.is_file() and legacy_params_path.is_file():
        metadata = json.loads(legacy_meta_path.read_text(encoding="utf-8"))
        from flax import serialization

        params = serialization.msgpack_restore(legacy_params_path.read_bytes())
        return params, metadata
    if legacy_params_path.is_file():
        from flax import serialization

        params = serialization.msgpack_restore(legacy_params_path.read_bytes())
        return params, {}

    # New PyTorch checkpoint fallback
    state_path = artifact_dir / "state.pt"
    model_path = artifact_dir / "model.pt"
    if state_path.is_file():
        restored = torch.load(state_path, map_location="cpu", weights_only=False)
        if isinstance(restored, dict) and "model" in restored:
            return restored["model"], {}
    if model_path.is_file():
        return torch.load(model_path, map_location="cpu", weights_only=False), {}

    raise ValueError(
        "Local init_from must be an artifact or checkpoint dir with params: "
        f"{artifact_dir}"
    )


def load_init_entry(
    model_type: str,
    init_from: str,
    *,
    hf_cache_dir: str = HF_ROOT,
) -> Tuple[Any, Dict[str, Any]]:
    if not init_from:
        raise ValueError("`init_from` is empty.")

    if not init_from.startswith("hf://"):
        return _load_local_init_entry(init_from)

    model_name = init_from[len("hf://") :].strip()
    if not model_name:
        raise ValueError("Invalid HF init_from path, expected `hf://<name>`.")

    if model_type == "mae":
        _, params, metadata = load_mae_torch(
            model_name,
            repo_id="Goodeat/drifting",
            output_root=hf_cache_dir,
        )
        return params, metadata

    if model_type == "generator":
        _, params, metadata = load_generator_torch(
            model_name,
            repo_id="Goodeat/drifting",
            output_root=hf_cache_dir,
        )
        return params, metadata

    raise ValueError(f"Unsupported model_type={model_type!r}, expected 'mae' or 'generator'.")


def _convert_entry_for_model(model_type: str, entry: Any, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    target = model.state_dict()

    if isinstance(entry, dict):
        is_flat = all(isinstance(k, str) for k in entry.keys())
        if is_flat and set(entry.keys()) == set(target.keys()):
            return {k: torch.as_tensor(v).to(dtype=target[k].dtype) for k, v in entry.items()}

    if model_type == "mae":
        return convert_mae_tree_to_state_dict(entry, target)
    if model_type == "generator":
        return convert_generator_tree_to_state_dict(entry, target)
    raise ValueError(model_type)


def maybe_init_state_params(
    state: Any,
    *,
    model_type: str,
    init_from: str,
    hf_cache_dir: str = HF_ROOT,
) -> Any:
    if not init_from:
        return state

    raw_model = unwrap_ddp(state.model)
    loaded_params, _ = load_init_entry(model_type, init_from, hf_cache_dir=hf_cache_dir)
    converted = _convert_entry_for_model(model_type, loaded_params, raw_model)
    missing, unexpected = raw_model.load_state_dict(converted, strict=False)
    if missing or unexpected:
        raise ValueError(f"Failed to load model params. missing={missing[:8]} unexpected={unexpected[:8]}")

    if getattr(state, "ema_model", None) is not None:
        unwrap_ddp(state.ema_model).load_state_dict(raw_model.state_dict(), strict=True)
    return state


def load_generator_model_and_params(
    init_from: str,
    *,
    hf_cache_dir: str = HF_ROOT,
) -> Tuple[Any, Any, Dict[str, Any]]:
    if not init_from:
        raise ValueError("`init_from` is empty.")

    if init_from.startswith("hf://"):
        model_name = init_from[len("hf://") :].strip()
        model, params, metadata = load_generator_torch(
            model_name,
            repo_id="Goodeat/drifting",
            output_root=hf_cache_dir,
        )
        return model, params, metadata

    entry, metadata = _load_local_init_entry(init_from)
    model_cfg = dict(metadata.get("model_config", {}) or {})
    if not model_cfg:
        raise ValueError(
            f"missing metadata.model_config: local artifact at {Path(init_from).resolve()} "
            "cannot be restored without model_config in metadata.json"
        )
    from models.generator import build_generator_from_config

    model = build_generator_from_config(model_cfg)
    state_dict = _convert_entry_for_model("generator", entry, model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"Failed to load generator weights. missing={missing[:8]} unexpected={unexpected[:8]}")
    model.eval()
    return model, model.state_dict(), metadata
