"""Debiased entropic OT drifting loss.

Drop-in alternative to ``drift_loss`` that replaces softmax-based affinities
with Sinkhorn optimal-transport barycentric maps.

Calling convention mirrors ``drift_loss``:
    loss, info = drift_loss_ot(gen, fixed_pos, fixed_neg, ...)
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

_COMPILE = os.environ.get("DRIFT_COMPILE", "1") != "0"


# ---------------------------------------------------------------------------
# Sinkhorn OT plan (balanced, log-domain)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sinkhorn_plan_log_impl(
    X: torch.Tensor,
    Y: torch.Tensor,
    reg: torch.Tensor,
    diag_mask: bool = False,
    num_iter: int = 50,
    stop_thr: float = 1e-4,
    target_weights: torch.Tensor | None = None,
    use_quadratic_cost: bool = False,
) -> torch.Tensor:
    """Balanced entropic OT plan in log domain, batched over dim-0.

    Args:
        X: source points   [B, N, D]
        Y: target points   [B, M, D]
        reg: scalar tensor, entropic regularisation (> 0)
        diag_mask: if True, mask the first N diagonal entries of the cost
        num_iter: fixed number of Sinkhorn iterations
        stop_thr: unused (kept for API compat)
        target_weights: optional non-uniform target marginal [B, M]
        use_quadratic_cost: if True, use c(x,y) = ||x-y||^2 / 2 instead of ||x-y||

    Returns:
        Pi: transport plan  [B, N, M]
    """
    device, dtype = X.device, X.dtype
    B, N, _ = X.shape
    M = Y.shape[1]

    C = torch.cdist(X, Y)  # [B, N, M]
    if use_quadratic_cost:
        C = 0.5 * C * C

    if diag_mask:
        diag = torch.arange(min(N, M), device=device)
        C[:, diag, diag] = C[:, diag, diag] + 1e6

    logK = -C / reg  # [B, N, M]

    log_a = torch.full((B, N), -math.log(N), device=device, dtype=dtype)
    if target_weights is None:
        log_b = torch.full((B, M), -math.log(M), device=device, dtype=dtype)
    else:
        b = target_weights.to(device=device, dtype=dtype)
        b = b / b.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        log_b = torch.log(b.clamp_min(1e-30))

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    n = max(int(num_iter), 1)
    # MMD: n = 0, i.e., unnormalized kernel
    # KL: n = 0.5, i.e., only compute: log_u = log_a - torch.logsumexp(logK + log_v[:, None, :], dim=-1)
    for _ in range(n):
        log_u = log_a - torch.logsumexp(logK + log_v[:, None, :], dim=-1)
        log_v = log_b - torch.logsumexp(logK.transpose(1, 2) + log_u[:, None, :], dim=-1)

    return torch.exp(log_u[:, :, None] + logK + log_v[:, None, :])


if _COMPILE:
    _sinkhorn_plan_log = torch.compile(_sinkhorn_plan_log_impl, dynamic=True)
else:
    _sinkhorn_plan_log = _sinkhorn_plan_log_impl


# ---------------------------------------------------------------------------
# Sinkhorn OT plan — batched: per-element reg [B] and diag_mask [B]
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sinkhorn_batched_impl(
    X: torch.Tensor,
    Y: torch.Tensor,
    reg: torch.Tensor,
    diag_mask: torch.Tensor,
    num_iter: int,
    target_weights: torch.Tensor,
    use_quadratic_cost: bool = False,
) -> torch.Tensor:
    """Sinkhorn OT plan with per-element regularisation and diagonal mask.

    Batches multiple independent transport problems (different reg values,
    different masking) into a single kernel launch.

    Args:
        X: source points          [B, N, D]
        Y: target points          [B, M, D]
        reg: per-element reg      [B]
        diag_mask: per-element    [B] bool
        num_iter: fixed Sinkhorn iterations
        target_weights:           [B, M]  (always required — no None branch)
        use_quadratic_cost: if True, use c(x,y) = ||x-y||^2 / 2 instead of ||x-y||

    Returns:
        Pi: transport plan  [B, N, M]
    """
    device, dtype = X.device, X.dtype
    B, N, _ = X.shape
    M = Y.shape[1]

    C = torch.cdist(X, Y)  # [B, N, M]
    if use_quadratic_cost:
        C = 0.5 * C * C

    eye = torch.eye(N, M, device=device, dtype=dtype)
    C = C + eye[None] * diag_mask.to(dtype=dtype)[:, None, None] * 1e6

    logK = -C / reg[:, None, None]  # [B, N, M]

    log_a = torch.full((B, N), -math.log(N), device=device, dtype=dtype)
    b = target_weights.to(dtype=dtype)
    b = b / b.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    log_b = torch.log(b.clamp_min(1e-30))

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    n = max(int(num_iter), 1)
    for _ in range(n):
        log_u = log_a - torch.logsumexp(logK + log_v[:, None, :], dim=-1)
        log_v = log_b - torch.logsumexp(logK.transpose(1, 2) + log_u[:, None, :], dim=-1)

    return torch.exp(log_u[:, :, None] + logK + log_v[:, None, :])


if _COMPILE:
    _sinkhorn_batched = torch.compile(_sinkhorn_batched_impl, dynamic=True)
else:
    _sinkhorn_batched = _sinkhorn_batched_impl


# ---------------------------------------------------------------------------
# Barycentric map (non-batched, kept for backward compat / standalone use)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _barycentric_map(
    z: torch.Tensor,
    support: torch.Tensor,
    eps: torch.Tensor,
    num_iter: int = 50,
    stop_thr: float = 1e-4,
    diag_mask: bool = False,
    target_weights: torch.Tensor | None = None,
    use_quadratic_cost: bool = False,
) -> torch.Tensor:
    """OT barycentric projection: T(z) = (Pi @ support) / row_mass.

    All inputs are detached internally -- no gradients flow through OT.
    """
    Pi = _sinkhorn_plan_log(
        z.detach(),
        support.detach(),
        reg=eps,
        diag_mask=diag_mask,
        num_iter=int(num_iter),
        stop_thr=float(stop_thr),
        target_weights=target_weights,
        use_quadratic_cost=use_quadratic_cost,
    )
    row_mass = Pi.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.bmm(Pi, support.detach()) / row_mass


# ---------------------------------------------------------------------------
# Debiased OT velocity field
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_V_debiased(
    z: torch.Tensor,
    w_pos: torch.Tensor,
    w_neg: torch.Tensor,
    eps: torch.Tensor,
    num_iter: int = 50,
    stop_thr: float = 1e-4,
    neg_diag_mask: bool = True,
    neg_target_weights: torch.Tensor | None = None,
    cfg_weight: torch.Tensor | None = None,
    w_uncond: torch.Tensor | None = None,
    use_quadratic_cost: bool = False,
) -> torch.Tensor:
    """Debiased OT velocity: V = T_pq - T_qneg  (+ optional CFG term).

    Args:
        z:       generated features  [B, N, D]
        w_pos:   positive features   [B, P, D]
        w_neg:   negative features   [B, Q, D]
        eps:     scalar tensor, entropic regularisation
        neg_diag_mask: mask self-transport in z->neg map
        neg_target_weights: optional weights for neg marginal [B, Q]
        cfg_weight: per-batch CFG strength [B] for new-CFG mode
        w_uncond:   unconditional features [B, U, D] for new-CFG mode
        use_quadratic_cost: if True, use quadratic OT cost

    Returns:
        V: velocity field  [B, N, D]
    """
    bary_kw = dict(eps=eps, num_iter=num_iter, stop_thr=stop_thr,
                   use_quadratic_cost=use_quadratic_cost)

    T_pq = _barycentric_map(z, w_pos, diag_mask=False, **bary_kw)
    T_qneg = _barycentric_map(
        z, w_neg,
        diag_mask=neg_diag_mask,
        target_weights=neg_target_weights,
        **bary_kw,
    )

    V = T_pq - T_qneg

    if cfg_weight is not None and w_uncond is not None:
        T_quncond = _barycentric_map(z, w_uncond, diag_mask=False, **bary_kw)
        V = V + cfg_weight.view(-1, 1, 1) * (T_pq - T_quncond)

    return V


# ---------------------------------------------------------------------------
# Batched velocity computation across all R values (optimised path)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_V_all_R(
    z: torch.Tensor,
    w_pos: torch.Tensor,
    w_neg: torch.Tensor,
    R_list: Tuple[float, ...],
    reg_scale: float,
    num_iter: int,
    neg_diag_mask: bool,
    neg_target_weights: torch.Tensor | None,
    cfg_weight: torch.Tensor | None,
    w_uncond: torch.Tensor | None,
    use_quadratic_cost: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute aggregated velocity field for all R values in one Sinkhorn call.

    Batches every (R_value, map_type) combination along dim-0 with per-element
    ``reg`` and ``diag_mask``, replacing 6-9 sequential Sinkhorn invocations
    with a single one.

    Args:
        reg_scale: sqrt(S) for L2 cost, S for quadratic cost.

    Returns:
        V_agg:   [B, N, D]  aggregated (normalised) velocity
        f_norms: [num_R]    per-R velocity norms for logging
    """
    device, dtype = z.device, z.dtype
    B, N, D = z.shape
    num_R = len(R_list)
    use_cfg = cfg_weight is not None and w_uncond is not None
    maps_per_R = 3 if use_cfg else 2
    total_maps = num_R * maps_per_R

    P, Q = w_pos.shape[1], w_neg.shape[1]
    U = w_uncond.shape[1] if use_cfg else 0
    max_M = max(P, Q, U) if use_cfg else max(P, Q)

    # --- Pad targets and weights to uniform max_M ---
    # Padded target *positions* are filled with a large sentinel (not zeros) so
    # that cdist produces a huge cost and Sinkhorn routes no mass there, even
    # with a very sharp kernel (small reg).  Padded *weights* remain zero.
    _PAD_VAL = 1e4

    def _pad(t: torch.Tensor, tw: torch.Tensor | None, M_orig: int):
        if tw is None:
            tw = torch.ones(B, M_orig, device=device, dtype=dtype)
        if M_orig == max_M:
            return t, tw
        pad_size = max_M - M_orig
        t_padded = torch.cat([t, t.new_full((B, pad_size, D), _PAD_VAL)], dim=1)
        tw_padded = F.pad(tw, (0, pad_size))
        return t_padded, tw_padded

    pos_p, pos_tw = _pad(w_pos, None, P)
    neg_p, neg_tw = _pad(w_neg, neg_target_weights, Q)

    if use_cfg:
        unc_p, unc_tw = _pad(w_uncond, None, U)
        Y_per_R = torch.cat([pos_p, neg_p, unc_p], dim=0)
        W_per_R = torch.cat([pos_tw, neg_tw, unc_tw], dim=0)
    else:
        Y_per_R = torch.cat([pos_p, neg_p], dim=0)
        W_per_R = torch.cat([pos_tw, neg_tw], dim=0)

    # --- Stack across all R values ---
    X_batch = z.repeat(total_maps, 1, 1)        # [total*B, N, D]
    Y_batch = Y_per_R.repeat(num_R, 1, 1)       # [total*B, max_M, D]
    W_batch = W_per_R.repeat(num_R, 1)           # [total*B, max_M]

    # Per-element diag_mask: [pos=F, neg=?, uncond=F] repeated per R
    dm_false = torch.zeros(B, device=device, dtype=torch.bool)
    dm_neg = torch.full((B,), neg_diag_mask, device=device, dtype=torch.bool)
    if use_cfg:
        dm_per_R = torch.cat([dm_false, dm_neg, dm_false])
    else:
        dm_per_R = torch.cat([dm_false, dm_neg])
    DM_batch = dm_per_R.repeat(num_R)

    # Per-element reg: constant within one R, different across R
    eps_per_R = torch.tensor(
        [float(R) * reg_scale for R in R_list], device=device, dtype=dtype,
    )
    REG_batch = eps_per_R.repeat_interleave(maps_per_R * B)

    # --- Single Sinkhorn call ---
    Pi = _sinkhorn_batched(
        X_batch, Y_batch, REG_batch, DM_batch, num_iter, W_batch,
        use_quadratic_cost=use_quadratic_cost,
    )

    # --- Barycentric maps: T = (Pi @ Y) / row_mass ---
    row_mass = Pi.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    T_all = torch.bmm(Pi, Y_batch) / row_mass   # [total*B, N, D]
    T_all = T_all.view(num_R, maps_per_R, B, N, D)

    T_pq = T_all[:, 0]                           # [num_R, B, N, D]
    T_qneg = T_all[:, 1]                         # [num_R, B, N, D]

    V_raw = T_pq - T_qneg
    if use_cfg:
        T_quncond = T_all[:, 2]
        V_raw = V_raw + cfg_weight[None, :, None, None] * (T_pq - T_quncond)

    # Per-R normalisation and sum
    f_norms = (V_raw ** 2).mean(dim=(1, 2, 3))               # [num_R]
    force_scales = torch.sqrt(f_norms.clamp(min=1e-8))       # [num_R]
    V_agg = (V_raw / force_scales[:, None, None, None]).sum(dim=0)

    return V_agg, f_norms


# ---------------------------------------------------------------------------
# Public entry point  (mirrors drift_loss signature)
# ---------------------------------------------------------------------------

def drift_loss_ot(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    R_list: Iterable[float] = (0.02, 0.05, 0.2),
    sinkhorn_num_iter: int = 20,
    sinkhorn_stop_thr: float = 1e-4,
    use_new_cfg: bool = False,
    fixed_uncond: torch.Tensor | None = None,
    weight_uncond: torch.Tensor | None = None,
    disable_diag_mask: bool = False,
    batch_sinkhorn: bool = False,
    use_quadratic_cost: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Debiased entropic-OT drifting loss.

    Same return contract as ``drift_loss``:
        loss  [B]  per-sample MSE
        info  dict of scalar metrics

    When *use_new_cfg=False* (default), unconditional negatives are expected to
    already be part of ``fixed_neg`` with appropriate ``weight_neg`` -- exactly
    the same as the original ``drift_loss``.

    When *use_new_cfg=True*, unconditional samples are passed separately via
    ``fixed_uncond`` / ``weight_uncond`` and a third barycentric map is used.

    When *batch_sinkhorn=True*, all R values and map types are batched into a
    single Sinkhorn call (6-9x fewer kernel launches).  Default *False*
    preserves the original sequential behaviour.
    """
    B, C_g, S = gen.shape

    if fixed_neg is None:
        fixed_neg = torch.zeros_like(gen[:, :0, :])

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()

    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])
    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()

    # -- feature-scale normalisation (same logic as drift_loss) --
    with torch.no_grad():
        info: Dict[str, torch.Tensor] = {}

        targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
        targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

        dist = torch.cdist(old_gen, targets)
        if use_quadratic_cost:
            weighted_dist_sq = (dist * dist) * targets_w[:, None, :]
            scale = (weighted_dist_sq.mean() / (targets_w.mean() + 1e-8)).sqrt()
            del weighted_dist_sq
        else:
            weighted_dist = dist * targets_w[:, None, :]
            scale = weighted_dist.mean() / (targets_w.mean() + 1e-8)
            del weighted_dist
        info["scale"] = scale
        del targets, targets_w, dist

        scale_inputs = torch.clamp(scale / (S ** 0.5), min=1e-3)

    old_gen_scaled = old_gen / scale_inputs
    pos_scaled = fixed_pos.detach() / scale_inputs
    neg_scaled = fixed_neg.detach() / scale_inputs
    del old_gen, fixed_pos, fixed_neg, weight_gen, weight_pos

    # -- prepare new-CFG components --
    cfg_weight_per_sample = None
    uncond_scaled = None
    if use_new_cfg and fixed_uncond is not None:
        fixed_uncond = fixed_uncond.float()
        uncond_scaled = fixed_uncond.detach() / scale_inputs
        del fixed_uncond
        if weight_uncond is not None:
            cfg_weight_per_sample = weight_uncond.float()[:, 0]
        neg_target_weights = None
    else:
        neg_target_weights = weight_neg

    # -- accumulate velocity across R values --
    # Reg scales with cost magnitude: sqrt(S) for L2, S for quadratic
    reg_scale = float(S) if use_quadratic_cost else math.sqrt(S)
    R_tuple = tuple(float(r) for r in R_list)
    with torch.no_grad():
        if batch_sinkhorn:
            V_agg, f_norms = _compute_V_all_R(
                old_gen_scaled,
                pos_scaled,
                neg_scaled,
                R_list=R_tuple,
                reg_scale=reg_scale,
                num_iter=sinkhorn_num_iter,
                neg_diag_mask=(not disable_diag_mask),
                neg_target_weights=neg_target_weights,
                cfg_weight=cfg_weight_per_sample,
                w_uncond=uncond_scaled,
                use_quadratic_cost=use_quadratic_cost,
            )
            for i, R in enumerate(R_tuple):
                info[f"loss_{R}"] = f_norms[i]
        else:
            V_agg = torch.zeros_like(old_gen_scaled)
            for R in R_tuple:
                eps_eff = torch.tensor(
                    float(R) * reg_scale,
                    device=old_gen_scaled.device, dtype=old_gen_scaled.dtype,
                )
                V_raw = _compute_V_debiased(
                    old_gen_scaled,
                    pos_scaled,
                    neg_scaled,
                    eps=eps_eff,
                    num_iter=sinkhorn_num_iter,
                    stop_thr=sinkhorn_stop_thr,
                    neg_diag_mask=(not disable_diag_mask),
                    neg_target_weights=neg_target_weights,
                    cfg_weight=cfg_weight_per_sample,
                    w_uncond=uncond_scaled,
                    use_quadratic_cost=use_quadratic_cost,
                )
                f_norm = (V_raw ** 2).mean()
                info[f"loss_{R}"] = f_norm
                force_scale = torch.sqrt(torch.clamp(f_norm, min=1e-8))
                V_agg = V_agg + V_raw / force_scale

        goal_scaled = old_gen_scaled + V_agg

    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = torch.mean(diff ** 2, dim=(-1, -2))
    info = {k: v.mean() for k, v in info.items()}
    return loss, info
