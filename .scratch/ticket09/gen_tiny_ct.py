# Ticket 09 deliverable 1: tiny compressed-tensors W4A16 g128 sym generator.
#
# Real compressed-tensors pack-quantized semantics (per compressed-tensors docs,
# `pack-quantized` scheme, W4A16, num_bits=4, symmetric, group_size=128):
#   - weight is quantized per-group along the INPUT dim (groups of 128 along K)
#   - symmetric: scale = amax / (2**(bits-1) - 1) = amax / 7
#     q = clamp(round(w / scale), -8, 7)
#   - stored zero point = 2**(bits-1) = 8 (constant for symmetric; stored as
#     `weight_zero_point`, NOT packed)
#   - `weight_packed`: 8 consecutive 4-bit codes (q + 8, i.e. 0..15) packed into
#     one int32, LSB-first: element j of the 8 occupies bits [4j, 4j+4).
#     8 elements per int32, no wasted bits; shape [out_features, in_features/8],
#     packing along the last (input) dim.
#   - `weight_scale`: [out_features, groups], activation dtype (fp16 here)
#
# Geometry: Llama-3-8B real dims (K=4096 / 14336; N=4096/14336/1024/2048/28672),
# shapes preserved, layers cut to tiny count. Weights are random *quantizable*
# data (gaussian, per-group outliers so quantization is not degenerate).
#
# Zero downloads: everything from config + torch.randn.

from __future__ import annotations

import argparse
import json
import os

import torch

torch.manual_seed(9)

# Llama-3-8B linear geometry (input_dim -> [output_dims])
GEOMETRY = {
    'qkv_proj':  (4096, [2048 + 1024 + 1024]),   # fused qkv
    'o_proj':    (4096, [4096]),
    'gate_up':   (4096, [28672]),                # fused gate_up (intermediate 14336)
    'down_proj': (14336, [4096]),
}
GROUP = 128
BITS = 4
NUM_LAYERS = 2
DTYPE = torch.float16


def quantize_ct(w: torch.Tensor, group: int = GROUP, bits: int = BITS):
    """Quantize fp weight [out, in] with real compressed-tensors W4A16 sym
    g-simantics.

    Returns (weight_packed int32 [out, in/8], weight_scale fp16 [out, groups], weight_zero_point int8 [out, groups]).
    """
    out_f, in_f = w.shape
    assert in_f % group == 0, (in_f, group)
    g = in_f // group
    w = w.float().view(out_f, g, group)
    amax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / (2 ** (bits - 1) - 1)                     # amax / 7
    q = torch.clamp(torch.round(w / scale), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    q_u4 = (q + 2 ** (bits - 1)).to(torch.int32)             # 0..15, [out, g, group]

    # pack-quantized: 8 codes per int32, LSB-first, along last dim
    assert group % 8 == 0
    q_u4 = q_u4.view(out_f, g, group // 8, 8)
    packed = torch.zeros(out_f, g, group // 8, dtype=torch.int32)
    for j in range(8):  # element j -> bits 4j..4j+4 (LSB first)
        packed |= q_u4[..., j] << (4 * j)
    packed = packed.view(out_f, in_f // 8)

    scale = scale.squeeze(-1).to(DTYPE)                      # [out, g]
    zp = torch.full((out_f, g), 2 ** (bits - 1), dtype=torch.int8)
    return packed, scale, zp


def gen_tiny_ct(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    config = {
        'base_config': 'meta-llama/Meta-Llama-3-8B (geometry only, cut to 2 layers)',
        'num_layers': NUM_LAYERS,
        'group_size': GROUP,
        'bits': BITS,
        'strategy': 'symmetric, groupwise along input dim',
        'ct_format': 'pack-quantized',
        'dtype': str(DTYPE),
        'layers': {},
    }
    for li in range(NUM_LAYERS):
        for name, (k, outs) in GEOMETRY.items():
            for oi, n in enumerate(outs):
                w = torch.randn(n, k, dtype=torch.float16)
                # per-group outliers -> non-degenerate quantization scales
                g = k // GROUP
                w.view(n, g, GROUP)[:, :, 0] *= 3.0
                packed, scale, zp = quantize_ct(w)
                key = f"model.layers.{li}.self_attn.{name}" if name != 'down_proj' \
                    else f"model.layers.{li}.mlp.down_proj"
                if name == 'gate_up':
                    key = f"model.layers.{li}.mlp.gate_up_proj"
                torch.save(
                    {
                        'weight_packed': packed,
                        'weight_scale': scale,
                        'weight_zero_point': zp,
                        'bias': torch.randn(n, dtype=torch.float16) * 0.01,
                    },
                    os.path.join(out_dir, f"{key}.pt"),
                )
                config['layers'][key] = {'out_features': n, 'in_features': k}
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"tiny ct model written to {out_dir}: "
          f"{len(config['layers'])} linear layers, {NUM_LAYERS} layers, "
          f"W{BITS}A16 g{GROUP} sym pack-quantized")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='.scratch/ticket09/tiny_ct_model')
    args = ap.parse_args()
    gen_tiny_ct(args.out)
