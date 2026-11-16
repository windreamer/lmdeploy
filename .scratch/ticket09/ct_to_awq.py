# Ticket 09 deliverable 2: compressed-tensors -> AutoAWQ layout converter.
#
# ct (pack-quantized W4A16 g128 sym), as produced by gen_tiny_ct.py:
#   weight_packed     int32 [N, K/8]   8 codes/int32, LSB-first, along K
#   weight_scale      fp16  [N, G]
#   weight_zero_point int8  [N, G]   constant 8 (symmetric)
#
# AutoAWQ layout (matches lmdeploy AWQFormat.accepts and the awq CUDA kernels):
#   qweight int32 [K, N/8] — packing along the OUTPUT dim: each int32 holds 8
#     consecutive output-channel codes for ONE input row k. Nibble order within
#     the int32 is AWQ's permutation [0,4,1,5,2,6,3,7]: unpacking nibbles
#     x0..x7 and stacking [x0,x4,x1,x5,x2,x6,x3,x7] yields sequential outputs
#     (lmdeploy weight_format.py:_unpack_awq_gemm). Packing is the inverse.
#   scales fp16 [G, N]  (K dim rows, sequential)
#   qzeros int32 [G, N/8] — every nibble = 8 (symmetric: no zero_point ->
#     AutoAWQ convention pads 0x88888888), each int32 likewise AWQ-ordered
#     (all nibbles equal, so permutation is a no-op on the value).
#
# dequant reference both sides: w = (code - 8) * scale_group.

from __future__ import annotations

import torch

AWQ_NIBBLE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def unpack_u4_lsb(packed: torch.Tensor) -> torch.Tensor:
    """Int32 [..., M] -> uint8 [..., M, 8], LSB-first nibbles."""
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32) * 4
    return ((packed[..., None] >> shifts) & 15).to(torch.uint8)


def ct_to_awq(weight_packed: torch.Tensor,
              weight_scale: torch.Tensor,
              weight_zero_point: torch.Tensor | None = None):
    """Convert ct pack-quantized tensors to AutoAWQ layout.

    Returns (qweight int32 [K, N/8], scales fp16 [G, N], qzeros int32 [G, N/8]).
    """
    n, k8 = weight_packed.shape
    k = k8 * 8
    g = weight_scale.shape[1]
    assert k == weight_scale.shape[1] * 128, (k, weight_scale.shape)

    assert (weight_zero_point == 8).all(), 'ct symmetric zero point must be 8'
    codes = unpack_u4_lsb(weight_packed)               # [N, K/8, 8] uint8, sequential along K
    codes = codes.reshape(n, k)                        # [N, K], code = q + 8

    # transpose to [K, N] view, pack 8 consecutive outputs per int32 with AWQ order
    codes_t = codes.t().contiguous()                   # [K, N]
    assert n % 8 == 0
    groups8 = codes_t.view(k, n // 8, 8)               # sequential output index within nibble
    qweight = torch.zeros(k, n // 8, dtype=torch.int32)
    for seq_j in range(8):
        nibble_pos = AWQ_NIBBLE_ORDER[seq_j]           # where seq_j lands in the int32
        qweight |= groups8[..., seq_j].to(torch.int32) << (4 * nibble_pos)

    scales = weight_scale.t().contiguous()             # [G, N]
    qzeros = torch.full((g, n // 8), 0x88888888, dtype=torch.int64).to(torch.int32)
    return qweight, scales, qzeros


def awq_dequant(qweight: torch.Tensor, scales: torch.Tensor,
                qzeros: torch.Tensor) -> torch.Tensor:
    """Reference dequant of AutoAWQ tensors back to fp weight [N, K]."""
    k, n8 = qweight.shape
    n = n8 * 8
    nibbles = unpack_u4_lsb(qweight)                       # [K, N/8, 8]
    w = torch.empty(k, n, dtype=torch.uint8)
    order = torch.tensor(AWQ_NIBBLE_ORDER)
    # replicate lmdeploy _unpack_awq_gemm: sequential j = nibble at position order[j]
    w.view(k, n // 8, 8)[:] = nibbles[..., order]
    g_count, group = scales.shape
    w = w.view(g_count, k // g_count, n)                       # [G, group, N], groups along K
    zero = unpack_u4_lsb(qzeros)                               # [G, N/8, 8]
    zero = zero.reshape(g_count, n).float()
    return ((w.float() - zero[:, None]) * scales.float()[:, None]).reshape(-1, n).t()


def convert_layer(ct_dict: dict):
    """Full layer conversion + roundtrip sanity: dequant path == ct dequant
    path."""
    packed, scale, zp = ct_dict['weight_packed'], ct_dict['weight_scale'], ct_dict['weight_zero_point']
    n = packed.shape[0]
    k = packed.shape[1] * 8
    # ct reference dequant
    codes = unpack_u4_lsb(packed).reshape(n, k).float() - 8
    w_ct = (codes * scale.repeat_interleave(k // scale.shape[1], dim=1).float())

    qw, sc, qz = ct_to_awq(packed, scale, zp)
    w_awq = awq_dequant(qw, sc, qz)
    max_abs = (w_ct - w_awq).abs().max().item()
    assert max_abs == 0.0, f"roundtrip mismatch: {max_abs}"
    assert qw.shape == (k, n // 8) and sc.shape == (scale.shape[1], n) \
        and qz.shape == (scale.shape[1], n // 8), (qw.shape, sc.shape, qz.shape)
    return {'qweight': qw, 'scales': sc, 'qzeros': qz, 'bias': ct_dict['bias']}, max_abs


if __name__ == '__main__':
    import glob
    import os
    d = os.path.join(os.path.dirname(__file__), 'tiny_ct_model')
    worst = 0.0
    for f in sorted(glob.glob(os.path.join(d, '*.pt'))):
        ct = torch.load(f)
        awq, err = convert_layer(ct)
        worst = max(worst, err)
        print(f"{os.path.basename(f)}: qweight{tuple(awq['qweight'].shape)} "
              f"scales{tuple(awq['scales'].shape)} qzeros{tuple(awq['qzeros'].shape)} roundtrip_err={err}")
    assert worst == 0.0
    print('ct -> AutoAWQ conversion roundtrip exact on all layers')
