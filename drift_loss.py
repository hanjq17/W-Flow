from __future__ import annotations

import os
from typing import Dict, Iterable, Tuple

import torch

_COMPILE = os.environ.get("DRIFT_COMPILE", "1") != "0"


def cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2.0 * xydot
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


@torch.no_grad()
def _compute_drift_field_impl(
    old_gen: torch.Tensor,
    fixed_neg: torch.Tensor,
    fixed_pos: torch.Tensor,
    weight_gen: torch.Tensor,
    weight_neg: torch.Tensor,
    weight_pos: torch.Tensor,
    R_list: Tuple[float, ...],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-tensor drift field computation, compilable (no dicts or strings).

    All shape-dependent quantities (C_g, C_n, S) are derived from the input
    tensor shapes so ``torch.compile(dynamic=True)`` tracks them as symbolic
    integers instead of specialising on each concrete value.
    """
    C_g = old_gen.shape[1]
    S = old_gen.shape[2]

    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    dist = cdist(old_gen, targets)
    weighted_dist = dist * targets_w[:, None, :]
    scale = weighted_dist.mean() / (targets_w.mean() + 1e-8)

    scale_inputs = torch.clamp(scale / (S ** 0.5), min=1e-3)
    old_gen_scaled = old_gen / scale_inputs
    targets_scaled = targets / scale_inputs

    dist_normed = dist / torch.clamp(scale, min=1e-3)

    mask_val = 100.0
    C_total = targets.shape[1]
    diag_mask = torch.eye(C_g, dtype=torch.float32, device=old_gen.device)
    block_mask = torch.nn.functional.pad(diag_mask, (0, C_total - C_g, 0, 0)).unsqueeze(0)
    dist_normed = dist_normed + block_mask * mask_val

    force_across_R = torch.zeros_like(old_gen_scaled)
    f_norms = torch.empty(len(R_list), device=old_gen.device)

    split_idx = C_g + fixed_neg.shape[1]

    for i, R in enumerate(R_list):
        logits = -dist_normed / float(R)
        affinity = torch.softmax(logits, dim=-1)
        aff_transpose = torch.softmax(logits, dim=-2)
        affinity = torch.sqrt(torch.clamp(affinity * aff_transpose, min=1e-6))
        affinity = affinity * targets_w[:, None, :]

        aff_neg = affinity[:, :, :split_idx]
        aff_pos = affinity[:, :, split_idx:]

        sum_pos = torch.sum(aff_pos, dim=-1, keepdim=True)
        r_coeff_neg = -aff_neg * sum_pos
        sum_neg = torch.sum(aff_neg, dim=-1, keepdim=True)
        r_coeff_pos = aff_pos * sum_neg

        R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)
        total_force_R = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)

        total_coeffs = R_coeff.sum(dim=-1)
        total_force_R = total_force_R - total_coeffs[..., None] * old_gen_scaled
        f_norm_val = (total_force_R ** 2).mean()
        f_norms[i] = f_norm_val

        force_scale = torch.sqrt(torch.clamp(f_norm_val, min=1e-8))
        force_across_R = force_across_R + total_force_R / force_scale

    goal_scaled = old_gen_scaled + force_across_R
    return goal_scaled, scale_inputs, scale, f_norms


if _COMPILE:
    _compute_drift_field = torch.compile(_compute_drift_field_impl, dynamic=True)
else:
    _compute_drift_field = _compute_drift_field_impl


def drift_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    R_list: Iterable[float] = (0.02, 0.05, 0.2),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if fixed_neg is None:
        fixed_neg = torch.zeros_like(gen[:, :0, :])

    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])
    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()

    R_tuple = tuple(float(r) for r in R_list)
    goal_scaled, scale_inputs, scale, f_norms = _compute_drift_field(
        old_gen, fixed_neg, fixed_pos, weight_gen, weight_neg, weight_pos, R_tuple,
    )

    info: Dict[str, torch.Tensor] = {"scale": scale.mean()}
    for i, R in enumerate(R_tuple):
        info[f"loss_{R}"] = f_norms[i].mean()

    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = torch.mean(diff ** 2, dim=(-1, -2))
    return loss, info
