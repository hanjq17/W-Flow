from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from utils.dist_util import barrier, unwrap_ddp
from utils.logging import is_rank_zero, log_for_0


def _to_python_int(x) -> int:
    if torch.is_tensor(x):
        return int(x.detach().cpu().reshape(-1)[0].item())
    return int(x)


def _output_root(workdir: Optional[str] = None) -> Path:
    if workdir:
        return Path(workdir).resolve()
    return Path("runs").resolve()


def _job_ckpt_dir(workdir: Optional[str] = None) -> Path:
    return _output_root(workdir) / "checkpoints"


def _list_ckpts(ckpt_dir: Path) -> list[Path]:
    return sorted(ckpt_dir.glob("state_*.pt"))


def _extract_step(path: Path) -> int:
    stem = path.stem
    return int(stem.split("_")[-1])


def restore_checkpoint(step=None, state=None, workdir: Optional[str] = None):
    ckpt_dir = _job_ckpt_dir(workdir=workdir)
    if not ckpt_dir.exists():
        log_for_0("No local checkpoint dir at %s", str(ckpt_dir))
        return state

    if step is not None:
        step = int(step)

    ckpts = _list_ckpts(ckpt_dir)
    if not ckpts:
        return state

    if step is None:
        target_path = ckpts[-1]
    else:
        cand = [p for p in ckpts if _extract_step(p) == step]
        if not cand:
            return state
        target_path = cand[-1]

    payload = torch.load(target_path, map_location="cpu", weights_only=False)
    if state is None:
        return payload

    unwrap_ddp(state.model).load_state_dict(payload["model"], strict=True)
    if (
        getattr(state, "ema_model", None) is not None
        and "ema_model" in payload
        and payload["ema_model"] is not None
    ):
        unwrap_ddp(state.ema_model).load_state_dict(payload["ema_model"], strict=True)
    state.optimizer.load_state_dict(payload["optimizer"])
    state.step = int(payload.get("step", 0))
    state.ema_decay = float(payload.get("ema_decay", getattr(state, "ema_decay", 0.999)))
    return state


def save_checkpoint(state, keep=2, keep_every=None, workdir: Optional[str] = None):
    barrier()
    if not is_rank_zero():
        barrier()
        return

    step = _to_python_int(state.step)
    ckpt_dir = _job_ckpt_dir(workdir=workdir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"state_{step:08d}.pt"

    payload = {
        "step": step,
        "model": {k: v.detach().cpu() for k, v in unwrap_ddp(state.model).state_dict().items()},
        "ema_model": {k: v.detach().cpu() for k, v in unwrap_ddp(state.ema_model).state_dict().items()} if getattr(state, "ema_model", None) is not None else None,
        "optimizer": state.optimizer.state_dict(),
        "ema_decay": float(getattr(state, "ema_decay", 0.999)),
    }
    torch.save(payload, path)
    log_for_0("Saving checkpoint step %d to %s", step, str(path))

    ckpts = _list_ckpts(ckpt_dir)
    if keep is not None and keep > 0 and len(ckpts) > keep:
        protected = set()
        if keep_every:
            for p in ckpts:
                s = _extract_step(p)
                if s % int(keep_every) == 0:
                    protected.add(p)
        removable = [p for p in ckpts[:-keep] if p not in protected]
        for p in removable:
            p.unlink(missing_ok=True)

    barrier()


def save_params_ema_artifact(
    state: Any,
    *,
    workdir: Optional[str] = None,
    kind: str,
    model_config: Optional[Dict[str, Any]] = None,
) -> Path:
    step = _to_python_int(state.step)
    ema_decay = float(getattr(state, "ema_decay"))

    out_dir = _output_root(workdir) / "params_ema"
    out_dir.mkdir(parents=True, exist_ok=True)

    ema_state = (
        {k: v.detach().cpu() for k, v in unwrap_ddp(state.ema_model).state_dict().items()}
        if getattr(state, "ema_model", None) is not None
        else {k: v.detach().cpu() for k, v in unwrap_ddp(state.model).state_dict().items()}
    )
    torch.save(ema_state, out_dir / "ema_params.pt")

    metadata = {
        "format": "torch.pt",
        "kind": kind,
        "backend": "torch",
        "ema_decay": ema_decay,
        "step": step,
        "path": "params_ema/ema_params.pt",
        "model_config": dict(model_config or {}),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    log_for_0("Saved EMA params artifact step %d to %s", step, str(out_dir))
    return out_dir
