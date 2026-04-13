# Copyright (c) OpenMMLab. All rights reserved.
# Triton fused dequant+matmul kernels for TurboQuant on-the-fly inference.
#
# Based on:
#   https://github.com/cksac/turboquant-model/blob/main/src/turboquant_model/triton_kernels.py
#
# These kernels avoid materializing the full dequantized weight matrix by
# fusing 4-bit unpack → codebook lookup → matmul → norm rescale in a single
# kernel launch.  The group loop runs *inside* the kernel so that:
#   1. Only one kernel launch is needed regardless of the number of groups.
#   2. The partial sums across groups are accumulated in registers and written
#      to HBM only once.
#
# Key optimisations:
#   - Autotune over (BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages).
#   - 16-entry codebook (64 bytes) stays in L1 / registers after first load.
#   - TF32 tensor-core MMA via allow_tf32=True on Ampere+ / Ada / Hopper.
#   - Pre-scaled norms (norms / scale) computed once on the host.
#   - Software pipelining via num_stages in the autotune configs.
"""Triton fused dequant + matmul kernels for TurboQuant."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from lmdeploy.pytorch.kernels.cuda.turbo_quant import hadamard_rotate

# ---------------------------------------------------------------------------
# Autotune configurations
# ---------------------------------------------------------------------------
_AUTOTUNE_CONFIGS = [
    # ---- decode-like (B = 1 .. 4) ----
    triton.Config(
        {'BLOCK_B': 1, 'BLOCK_N': 64, 'BLOCK_K': 64},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {'BLOCK_B': 1, 'BLOCK_N': 128, 'BLOCK_K': 64},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {'BLOCK_B': 4, 'BLOCK_N': 64, 'BLOCK_K': 64},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {'BLOCK_B': 4, 'BLOCK_N': 128, 'BLOCK_K': 64},
        num_warps=4, num_stages=3,
    ),
    # ---- small-batch prefill ----
    triton.Config(
        {'BLOCK_B': 16, 'BLOCK_N': 64, 'BLOCK_K': 64},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {'BLOCK_B': 16, 'BLOCK_N': 64, 'BLOCK_K': 128},
        num_warps=8, num_stages=3,
    ),
    # ---- large-batch prefill ----
    triton.Config(
        {'BLOCK_B': 32, 'BLOCK_N': 64, 'BLOCK_K': 64},
        num_warps=8, num_stages=3,
    ),
    triton.Config(
        {'BLOCK_B': 32, 'BLOCK_N': 64, 'BLOCK_K': 128},
        num_warps=8, num_stages=4,
    ),
    triton.Config(
        {'BLOCK_B': 64, 'BLOCK_N': 64, 'BLOCK_K': 128},
        num_warps=8, num_stages=4,
    ),
]


# ---------------------------------------------------------------------------
# Fused kernel — all groups processed in one launch
# ---------------------------------------------------------------------------
@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=['B', 'N', 'FULL_K'])
@triton.jit
def _turboquant_fused_grouped_matmul_kernel(
    # --- inputs ---
    input_ptr,       # (B, FULL_K) float32 — pre-rotated activations
    indices_ptr,     # (N, FULL_K // 2) uint8 — packed 4-bit indices
    codebook_ptr,    # (16,) float32
    norms_ptr,       # (N, N_GROUPS) float32 — pre-scaled norms
    # --- output ---
    output_ptr,      # (B, N) float32
    # --- scalar dims ---
    B,               # batch (flattened B*seq)
    N,               # out_features
    FULL_K,          # in_features
    PACKED_K,        # FULL_K // 2
    GROUP_SIZE: tl.constexpr,
    N_GROUPS: tl.constexpr,
    # --- tile sizes (from autotune) ---
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused 4-bit dequant + matmul with all groups in one kernel.

    output[b, n] = sum_g  norms_scaled[n, g] * sum_k  input_rot[b, g*GS + k] * codebook[idx[n, g*GS + k]]
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)   # (BLOCK_B,)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # (BLOCK_N,)
    mask_b = rb < B
    mask_n = rn < N

    # Final accumulator across *all* groups — lives in registers.
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    for g in range(N_GROUPS):
        g_start = g * GROUP_SIZE

        # Per-group accumulator
        group_acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

        # Iterate over K-tiles within this group
        for k_off in range(0, GROUP_SIZE, BLOCK_K):
            rk_local = k_off + tl.arange(0, BLOCK_K)   # offset within group
            mask_k = rk_local < GROUP_SIZE
            rk_global = g_start + rk_local              # global K index

            # ---- Load input tile: (BLOCK_B, BLOCK_K) ----
            inp_off = rb[:, None] * FULL_K + rk_global[None, :]
            inp_mask = mask_b[:, None] & mask_k[None, :]
            inp_tile = tl.load(input_ptr + inp_off, mask=inp_mask, other=0.0)

            # ---- Unpack 4-bit indices: (BLOCK_N, BLOCK_K) ----
            byte_col = rk_global // 2
            is_high = (rk_global % 2) == 1
            byte_off = rn[:, None] * PACKED_K + byte_col[None, :]
            w_mask = mask_n[:, None] & mask_k[None, :]

            packed = tl.load(
                indices_ptr + byte_off, mask=w_mask, other=0
            ).to(tl.uint8)
            lo = packed & 0x0F
            hi = (packed >> 4) & 0x0F
            idx = tl.where(is_high[None, :], hi, lo)

            # ---- Codebook gather (16 entries, 64 B → L1/regs) ----
            w_quant = tl.load(
                codebook_ptr + idx.to(tl.int32), mask=w_mask, other=0.0
            )

            # ---- TF32 tensor-core dot ----
            group_acc += tl.dot(
                inp_tile.to(tl.float32),
                tl.trans(w_quant.to(tl.float32)),
                allow_tf32=True,
            )

        # Scale by pre-scaled group norms and accumulate — still in registers
        norm_off = rn * N_GROUPS + g
        norms_g = tl.load(norms_ptr + norm_off, mask=mask_n, other=0.0)
        acc += group_acc * norms_g[None, :]

    # ---- Write output to HBM once ----
    out_off = rb[:, None] * N + rn[None, :]
    out_mask = mask_b[:, None] & mask_n[None, :]
    tl.store(
        output_ptr + out_off,
        acc.to(output_ptr.dtype.element_ty),
        mask=out_mask,
    )


# ---------------------------------------------------------------------------
# Python wrapper — single-launch grouped fused matmul
# ---------------------------------------------------------------------------
def triton_fused_dequant_matmul(
    x_rot: torch.Tensor,
    indices_packed: torch.Tensor,
    codebook: torch.Tensor,
    norms: torch.Tensor,
    in_features: int,
    group_size: int,
    sigma_scale: float | None = None,
) -> torch.Tensor:
    """Fused 4-bit dequant + matmul (all groups, single launch).

    Args:
        x_rot: Pre-rotated activations, ``(B, K)`` float32.
        indices_packed: Packed 4-bit weight indices, ``(N, K//2)`` uint8.
        codebook: Lloyd-Max centroids, ``(n_levels,)`` float32.
        norms: Per-row / per-group weight norms, ``(N, n_groups)`` float32.
        in_features: Full input dimension K.
        group_size: Group size used during quantisation.
        sigma_scale: Override for ``sigma * sqrt(group_size)``; if *None*
            computed as ``sqrt(group_size) / sqrt(in_features)``.

    Returns:
        ``(B, N)`` float32 output tensor.
    """
    B = x_rot.shape[0]
    N, PACKED_K = indices_packed.shape
    n_groups = norms.shape[1]

    # Pre-scale norms on host: norms_scaled[n, g] = norms[n, g] / scale
    if sigma_scale is None:
        sigma = 1.0 / math.sqrt(in_features)
        sigma_scale = sigma * math.sqrt(group_size)

    norms_scaled = (norms / sigma_scale).contiguous()

    output = torch.empty(B, N, dtype=torch.float32, device=x_rot.device)

    def grid(META):
        return (
            triton.cdiv(B, META['BLOCK_B']),
            triton.cdiv(N, META['BLOCK_N']),
        )

    _turboquant_fused_grouped_matmul_kernel[grid](
        x_rot.contiguous(),
        indices_packed,
        codebook,
        norms_scaled,
        output,
        B, N, in_features, PACKED_K,
        GROUP_SIZE=group_size,
        N_GROUPS=n_groups,
    )

    return output


# ---------------------------------------------------------------------------
# High-level forward: hadamard rotate + fused kernel + bias / cast
# ---------------------------------------------------------------------------
def turboquant_fused_forward(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    codebook: torch.Tensor,
    weight_norms: torch.Tensor,
    in_features: int,
    out_features: int,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full fused forward: hadamard-rotate input → grouped fused matmul → bias.

    Replaces the two-step path (``_dequantize_weight`` + ``torch.matmul``)
    with a single Triton launch that never materialises the ``(N, K)`` float
    weight matrix.

    Mathematical equivalence (per group *g*)::

        W_g = diag(norms_g) · H_inv · diag(1/scale) · C[idx_g]

    so::

        x @ W_g^T = (x @ H_inv^T) @ (C[idx_g] / scale)^T · diag(norms_g)
                   = x_rot @ C[idx_g]^T · (norms_g / scale)

    where ``x_rot = hadamard_rotate(x)`` is computed once per group and shared
    across output rows, and ``norms_g / scale`` is pre-computed on the host.

    Args:
        x: Input activations, ``(B, seq, K)`` or ``(B_flat, K)``.
        weight_packed: Packed 4-bit indices, ``(N, K//2)`` uint8.
        codebook: Lloyd-Max centroids, ``(16,)`` float32.
        weight_norms: Per-row per-group norms, ``(N, n_groups)`` float32.
        in_features: K.
        out_features: N.
        group_size: Quantisation group size.
        bias: Optional bias, ``(N,)`` float32.

    Returns:
        Output tensor in the same dtype as *x* (or bfloat16 if *x* is
        float32), with shape ``(*batch_dims, N)``.
    """
    orig_shape = x.shape
    orig_dtype = x.dtype

    # Flatten to 2-D
    if x.dim() == 3:
        B_flat = x.shape[0] * x.shape[1]
        x_2d = x.reshape(B_flat, in_features)
    elif x.dim() == 2:
        B_flat = x.shape[0]
        x_2d = x
    else:
        raise ValueError(f'Expected 2-D or 3-D input, got {x.dim()}-D')

    x_f32 = x_2d.float()
    n_groups = in_features // group_size

    # Per-group Hadamard rotation:
    # (B, K) → (B, n_groups, group_size) → rotate last dim → (B, K)
    x_grouped = x_f32.reshape(B_flat, n_groups, group_size)
    x_rot = hadamard_rotate(x_grouped).reshape(B_flat, in_features).contiguous()

    # Fused dequant + matmul (single kernel launch for all groups)
    output = triton_fused_dequant_matmul(
        x_rot=x_rot,
        indices_packed=weight_packed,
        codebook=codebook,
        norms=weight_norms,
        in_features=in_features,
        group_size=group_size,
    )

    # Bias
    if bias is not None:
        output = output + bias.float()

    # Cast back to input dtype
    out_dtype = orig_dtype if orig_dtype != torch.float32 else torch.bfloat16
    output = output.to(out_dtype)

    # Restore batch dimensions
    if len(orig_shape) == 3:
        output = output.reshape(orig_shape[0], orig_shape[1], out_features)

    return output
