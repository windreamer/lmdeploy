# Copyright (c) OpenMMLab. All rights reserved.
"""Tiny compressed-tensors W4A16 test checkpoint generator.

The generator synthesizes a tiny ct checkpoint directory from config-only
geometry. Nothing downloads. All tensors come from ``torch.randn``, pushed
through the real quantization in ``lmdeploy.turbomind.ct_repack``. The same
seed reproduces byte-identical shards.

Geometry presets mirror the entries in ``tests/turbomind/linear/models.yaml``.
Keep the two lists in sync. Every preset line must keep the input dim a
multiple of 128. The output dim must stay a multiple of 8.

A generated directory holds one ``config.json`` plus one ``.pt`` shard per
linear, keyed ``model.layers.<i>.<name>``. The keys are placeholders with no
loader behind them.
"""

from __future__ import annotations

import json
import os

import torch

from lmdeploy.turbomind.ct_repack import GROUP_SIZE, WEIGHT_DTYPE, quantize_ct, validate_ct_geometry

# (name, out_features, in_features) per model, copied from models.yaml.
PRESETS = {
    # Llama-3-8B.
    'llama3_8b': (
        ('gate_up', 28672, 4096),
        ('down', 4096, 14336),
        ('qkv', 6144, 4096),
        ('o', 4096, 4096),
    ),
    # GLM-4-9B.
    'glm4_9b': (
        ('gate_up', 27392, 4096),
        ('down', 4096, 13696),
        ('qkv', 4608, 4096),
        ('o', 4096, 4096),
    ),
    # GLM-5.2. Dense blocks.
    'glm52': (
        ('dense_gate_up', 24576, 6144),
        ('dense_down', 6144, 12288),
        # MoE blocks.
        ('gate_up', 4096, 6144),
        ('down', 6144, 2048),
        ('shared_gate_up', 4096, 6144),
        ('shared_down', 6144, 2048),
        # MLA blocks.
        ('mla_q_a', 2048, 6144),
        ('mla_q_b', 36864, 2048),
        ('mla_kv_a', 576, 6144),
        ('mla_o', 6144, 36864),
        # Indexer blocks.
        ('index_q', 4096, 2048),
        ('index_k', 128, 6144),
        ('index_weight', 32, 6144),
    ),
}


def shard_key(layer: int, name: str) -> str:
    return f'model.layers.{layer}.{name}'


def select_lines(preset: str, only=None):
    """Resolve a preset to its linear list, optionally filtered by name."""

    lines = PRESETS[preset]
    if only is None:
        return lines
    wanted = set(only)
    kept = tuple(item for item in lines if item[0] in wanted)
    missing = wanted - {item[0] for item in kept}
    if missing:
        raise ValueError(f'unknown linear names for preset {preset}: {sorted(missing)}')
    return kept


def gen_tiny_ct(out_dir: str,
                preset: str = 'glm52',
                num_layers: int = 2,
                seed: int = 9,
                only=None) -> dict:
    """Write a tiny ct checkpoint directory. Returns the written config.

    ``only`` restricts the preset to the given linear names. Random data
    carries a per-group outlier on the first group element. This keeps the
    quantization scales away from zero.
    """
    lines = select_lines(preset, only)
    validate_ct_geometry(lines)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    layers = {}
    for layer in range(num_layers):
        for name, n, k in lines:
            w = torch.randn(n, k, dtype=WEIGHT_DTYPE)
            w.view(n, k // GROUP_SIZE, GROUP_SIZE)[:, :, 0] *= 3.0
            packed, scale, zero = quantize_ct(w)
            bias = torch.randn(n, dtype=WEIGHT_DTYPE) * 0.01
            key = shard_key(layer, name)
            torch.save(
                {
                    'weight_packed': packed,
                    'weight_scale': scale,
                    'weight_zero_point': zero,
                    'bias': bias,
                },
                os.path.join(out_dir, f'{key}.pt'),
            )
            layers[key] = {'out_features': n, 'in_features': k}
    config = {
        'format': 'compressed-tensors',
        'quant_config': {
            'scheme': 'W4A16',
            'bits': 4,
            'group_size': GROUP_SIZE,
            'symmetric': True,
            'pack_format': 'pack-quantized',
        },
        'activation_dtype': 'float16',
        'preset': preset,
        'num_layers': num_layers,
        'seed': seed,
        'layers': layers,
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    return config


def load_shard(model_dir: str, key: str) -> dict:
    """Load one shard of a generated (or converted) directory by key."""
    return torch.load(os.path.join(model_dir, f'{key}.pt'), map_location='cpu', weights_only=True)
