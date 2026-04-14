"""Common test utilities for TurboQuant (quant_policy=QuantPolicy.TURBO_QUANT)
kernel tests.

This module contains shared helper functions for testing TurboQuant quantization,
which is used by quant_policy=QuantPolicy.TURBO_QUANT (K=QJL4, V=2bit mixed precision).

TurboQuant is a quantization method that:
- Uses Lloyd-Max algorithm for optimal quantization
- Applies Hadamard rotation for better distribution
- Stores only L2 norms (not scales/zeros) for dequantization
"""

import math

import torch

from lmdeploy.pytorch.kernels.cuda.turbo_quant import (
    get_lloyd_max_codebook,
    hadamard_rotate,
    hadamard_rotate_inv,
)


def _div_up(a, b):
    """Integer division with rounding up."""
    return (a + b - 1) // b


def _unpack_indices(packed: torch.Tensor, nbits: int, original_dim: int) -> torch.Tensor:
    """Unpack bit-packed indices back to integer tensor."""
    if nbits == 2:
        i0 = (packed & 0x03)
        i1 = ((packed >> 2) & 0x03)
        i2 = ((packed >> 4) & 0x03)
        i3 = ((packed >> 6) & 0x03)
        indices = torch.cat([i0, i1, i2, i3], dim=-1)
    elif nbits == 4:
        # Unpack 2 nibbles per byte: low nibble and high nibble
        i0 = (packed & 0x0F)
        i1 = ((packed >> 4) & 0x0F)
        indices = torch.cat([i0, i1], dim=-1)
    else:
        indices = packed

    # Trim to original dimension
    return indices[..., :original_dim].long()


def _unpack_qjl4_nibbles(packed: torch.Tensor, original_dim: int):
    """Unpack 4bit qjl nibbles into:
    - idx3: [0, 7]
    - bit1: [0, 1]
    """
    nib = _unpack_indices(packed, 4, original_dim)
    idx3 = nib & 0x7
    bit1 = (nib >> 3) & 0x1
    return idx3.long(), bit1.long()


def quant_turboquant_mse(kv: torch.Tensor, nbits: int):
    """TurboQuant MSE quantization (without QJL).

    Args:
        kv: input tensor of shape (..., head_dim)
        nbits: number of bits (only 2 supported)

    Returns:
        q_kv: bit-packed indices (uint8)
        norms: L2 norms for dequantization, shape (...,)
    """
    head_dim = kv.shape[-1]
    device = str(kv.device)

    # Get Lloyd-Max codebook
    _, boundaries = get_lloyd_max_codebook(head_dim, nbits, device=device)

    # Compute L2 norms
    norms = kv.float().norm(dim=-1, keepdim=True)

    # Normalize to unit sphere
    kv_unit = kv.float() / (norms + 1e-10)
    y = hadamard_rotate(kv_unit)

    # Quantize: find nearest centroid via searchsorted
    indices = torch.searchsorted(boundaries, y.contiguous())
    indices = indices.clamp(0, 2 ** nbits - 1)

    # Bit-pack indices (2-bit: 4 values per byte)
    if nbits == 2:
        q_kv1, q_kv2, q_kv3, q_kv4 = indices.split(indices.shape[-1] // 4, -1)
        q_kv = q_kv1 + q_kv2 * 4 + q_kv3 * 16 + q_kv4 * 64
    else:
        q_kv = indices

    return q_kv.to(torch.uint8), norms.squeeze(-1)


def quant_turboquant_qjl4(kv: torch.Tensor):
    """TurboQuant QJL4 quantization for K: 3bit MSE + 1bit QJL.

    Returns:
        q_kv: packed uint8 tensor, shape (..., head_dim // 2)
        meta: (..., 2)
              meta[..., 0] = mse_norm
              meta[..., 1] = qjl_norm
    """
    head_dim = kv.shape[-1]
    device = str(kv.device)

    # Get Lloyd-Max codebook (3-bit)
    centroids, boundaries = get_lloyd_max_codebook(head_dim, 3, device=device)

    # Compute MSE norm
    mse_norm = kv.float().norm(dim=-1, keepdim=True)
    kv_unit = kv.float() / (mse_norm + 1e-10)

    # Apply hadamard rotation
    y = hadamard_rotate(kv_unit)

    # Quantize: find nearest centroid
    idx3 = torch.searchsorted(boundaries, y.contiguous()).clamp(0, 7).long()
    c = centroids[idx3]

    # Compute QJL residual
    residual = y - c
    qjl_bit = (residual >= 0).long()
    qjl_norm = residual.norm(dim=-1, keepdim=True) / math.sqrt(head_dim)

    # Pack nibble: low 3 bits = MSE index, high 1 bit = QJL sign
    nibble = idx3 | (qjl_bit << 3)
    q1, q2 = nibble.split(nibble.shape[-1] // 2, dim=-1)
    q_kv = (q1 + (q2 << 4)).to(torch.uint8)

    meta = torch.cat([mse_norm, qjl_norm], dim=-1)
    return q_kv, meta


def dequantize_turboquant_mse(q_kv: torch.Tensor, norms: torch.Tensor, nbits: int):
    """TurboQuant MSE dequantization (without QJL).

    Args:
        q_kv: bit-packed indices (uint8)
        norms: L2 norms for rescaling, shape (...,)
        nbits: number of bits (only 2 supported)

    Returns:
        reconstructed kv tensor in original domain
    """
    # First dequantize to rotate domain
    y_hat = dequantize_turboquant_mse_rot(q_kv, norms, nbits)
    # Then inverse rotate to original domain
    x_hat = hadamard_rotate_inv(y_hat)
    return x_hat


def dequantize_turboquant_mse_rot(q_kv: torch.Tensor, norms: torch.Tensor, nbits: int):
    """TurboQuant MSE dequantization to ROTATE domain (no inverse rotation).

    Args:
        q_kv: bit-packed indices (uint8)
        norms: L2 norms for rescaling, shape (...,)
        nbits: number of bits (only 2 supported)

    Returns:
        reconstructed kv tensor in rotate domain
    """
    # Infer head_dim from packed shape
    if nbits == 2:
        head_dim = q_kv.shape[-1] * 4
    else:
        head_dim = q_kv.shape[-1]

    device = str(q_kv.device)

    # Get Lloyd-Max codebook
    centroids, _ = get_lloyd_max_codebook(head_dim, nbits, device=device)

    # Unpack indices
    indices = _unpack_indices(q_kv, nbits, head_dim)

    # Look up centroids
    y_hat = centroids[indices]

    # Rescale by norms (in rotate domain, no inverse rotation)
    y_hat = y_hat * norms.unsqueeze(-1)

    return y_hat


def dequantize_turboquant_qjl4(q_kv: torch.Tensor, meta: torch.Tensor):
    """Dequantize TurboQuant QJL4 to original domain."""
    # First dequantize to rotate domain
    y_hat = dequantize_turboquant_qjl4_rot(q_kv, meta)
    # Then inverse rotate to original domain
    x_hat = hadamard_rotate_inv(y_hat)
    return x_hat


def dequantize_turboquant_qjl4_rot(q_kv: torch.Tensor, meta: torch.Tensor):
    """Dequantize TurboQuant QJL4 to ROTATE domain (no inverse rotation)."""
    head_dim = q_kv.shape[-1] * 2
    device = str(q_kv.device)

    # Get Lloyd-Max codebook (3-bit)
    centroids, _ = get_lloyd_max_codebook(head_dim, 3, device=device)

    # Unpack nibbles
    idx3, bit1 = _unpack_qjl4_nibbles(q_kv, head_dim)
    sign = bit1.float() * 2.0 - 1.0

    # Get meta values
    mse_norm = meta[..., 0]
    qjl_norm = meta[..., 1]

    # Reconstruct in rotate domain (no inverse rotation)
    y_hat = centroids[idx3] + qjl_norm.unsqueeze(-1) * sign
    y_hat = y_hat * mse_norm.unsqueeze(-1)

    return y_hat


def compute_metrics(a: torch.Tensor, b: torch.Tensor):
    """Compute similarity metrics between two tensors.

    Args:
        a, b: tensors to compare

    Returns:
        dict with 'cosine', 'nmse', 'snr_db' keys
    """
    import math

    a_flat = a.flatten()
    b_flat = b.flatten()
    cosine = torch.cosine_similarity(a_flat, b_flat, dim=0).item()
    mse = ((a - b) ** 2).mean().item()
    nmse = mse / (b ** 2).mean().item()
    signal = (b ** 2).mean().item()
    noise = ((a - b) ** 2).mean().item()
    snr_db = 10 * math.log10(signal / (noise + 1e-10))
    return {'cosine': cosine, 'nmse': nmse, 'snr_db': snr_db}


# ---------------------------------------------------------------------------
# Fused kernel reference implementations (for turbo_quant_fused.py tests)
# ---------------------------------------------------------------------------


def unpack_4bit_indices(packed: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Unpack 4-bit indices from packed uint8 tensor (sequential layout).

    Within each group of packed bytes:
        lo nibble → first half of group
        hi nibble → second half of group
        concat(lo, hi) → original column order

    Args:
        packed: (N, K//2) uint8 tensor
        group_size: quantization group size

    Returns:
        (N, K) int64 tensor of indices in original order
    """
    N, PACKED_K = packed.shape
    K = PACKED_K * 2
    packed_gs = group_size // 2

    chunks = []
    for pg_start in range(0, PACKED_K, packed_gs):
        block = packed[:, pg_start: pg_start + packed_gs]
        lo = (block & 0x0F).long()
        hi = ((block >> 4) & 0x0F).long()
        chunks.append(lo)   # first half
        chunks.append(hi)   # second half
    return torch.cat(chunks, dim=1)[:, :K]

def reference_fused_matmul(
    x_rot: torch.Tensor,
    indices_packed: torch.Tensor,
    codebook: torch.Tensor,
    norms: torch.Tensor,
    group_size: int,
    sigma_scale: float | None = None,
) -> torch.Tensor:
    """Reference implementation of fused dequant+matmul (sequential layout).

    Args:
        x_rot: (B, K) pre-rotated activations
        indices_packed: (N, K//2) packed 4-bit indices (sequential layout)
        codebook: (16,) float32 Lloyd-Max centroids
        norms: (N, n_groups) float32 per-row per-group norms
        group_size: quantization group size
        sigma_scale: sigma scaling factor (if None, computed from K and group_size)

    Returns:
        (B, N) float32 output
    """
    B, K = x_rot.shape
    N = indices_packed.shape[0]
    n_groups = norms.shape[1]

    if sigma_scale is None:
        sigma = 1.0 / math.sqrt(K)
        sigma_scale = sigma * math.sqrt(group_size)

    norms_scaled = norms / sigma_scale
    indices = unpack_4bit_indices(indices_packed, group_size)
    W_rot = codebook[indices].float()

    acc = torch.zeros(B, N, dtype=torch.float32, device=x_rot.device)
    for g in range(n_groups):
        g_start = g * group_size
        g_end = g_start + group_size
        x_g = x_rot[:, g_start:g_end]
        W_g = W_rot[:, g_start:g_end]
        acc += torch.matmul(x_g, W_g.T) * norms_scaled[:, g]
    return acc

def reference_turboquant_forward(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    codebook: torch.Tensor,
    weight_norms: torch.Tensor,
    in_features: int,
    out_features: int,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference implementation of full turboquant forward (sequential layout).

    Args:
        x: (B, seq, K) or (B_flat, K) input activations
        weight_packed: (N, K//2) packed 4-bit indices (sequential layout)
        codebook: (16,) float32 Lloyd-Max centroids
        weight_norms: (N, n_groups) float32 per-row per-group norms
        in_features: K
        out_features: N
        group_size: quantization group size
        bias: optional (N,) bias

    Returns:
        Output tensor in same dtype as input
    """
    orig_shape = x.shape
    orig_dtype = x.dtype

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

    x_grouped = x_f32.reshape(B_flat, n_groups, group_size)
    x_rot = hadamard_rotate(x_grouped).reshape(B_flat, in_features)

    output = reference_fused_matmul(
        x_rot=x_rot,
        indices_packed=weight_packed,
        codebook=codebook,
        norms=weight_norms,
        group_size=group_size,
    )

    if bias is not None:
        output = output + bias.float()

    out_dtype = orig_dtype if orig_dtype != torch.float32 else torch.bfloat16
    output = output.to(out_dtype)

    if len(orig_shape) == 3:
        output = output.reshape(orig_shape[0], orig_shape[1], out_features)

    return output

def make_weight_packed(N: int, K: int, group_size: int = 128) -> tuple:
    """Create packed 4-bit weight indices and norms for testing.

    Uses SEQUENTIAL layout: within each group, first half → lo nibble,
    second half → hi nibble.

    Args:
        N: number of output features
        K: input features dimension
        group_size: quantization group size

    Returns:
        weight_packed: (N, K//2) uint8 (sequential layout)
        weight_norms: (N, n_groups) float32
    """
    n_groups = K // group_size
    half = group_size // 2
    indices = torch.randint(0, 16, (N, K), device='cuda')

    # Sequential pack: first half → lo, second half → hi
    chunks = []
    for g_start in range(0, K, group_size):
        lo = indices[:, g_start: g_start + half].to(torch.uint8)
        hi = indices[:, g_start + half: g_start + group_size].to(torch.uint8)
        chunks.append(lo | (hi << 4))
    weight_packed = torch.cat(chunks, dim=1)

    weight_norms = torch.rand(N, n_groups, device='cuda') + 0.1
    return weight_packed, weight_norms
