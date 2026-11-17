# Copyright (c) OpenMMLab. All rights reserved.
"""Seeded tiny checkpoints for the ticket B3-06 rung suites.

``gen_tiny_dense_ct``: dense llama with ``head_dim=64``. The dense tiny
generator in ``tests/turbomind/tiny_ct.py`` carries ``head_dim=32``, which
the TurboMind int8-KV geometry probe rejects (head dims 64/128/192/256);
this variant satisfies the probe so the int8-KV rung can actually enable
the TM block layout.

The MoE rung reuses the tiny Mixtral ct pair built by the ticket 02 golden
gate (``_make_tiny_moe_pair``); the glm4_moe architecture attempt was
dropped because of a shared-expert broadcast bug on the single-GPU path
(recorded in the MoE suite report).

Conventions inherited from tiny_ct: every input dim is a multiple of
GROUP_SIZE, every output dim a multiple of 128 (AWQ loader chunk
alignment), a per-group outlier on the first group element keeps the
quantization scales away from zero, and the seed reproduces byte-identical
weights. Nothing downloads; the tokenizer files come from the local AWQ
checkpoint.
"""

import json
import os
import shutil

import torch
from safetensors.torch import save_file

from lmdeploy.turbomind.ct_repack import GROUP_SIZE, WEIGHT_DTYPE, quantize_ct

# Dense geometry for the int8-KV rung. head_dim 64 passes the TurboMind
# int8-KV head-dim probe.
DENSE_HIDDEN = 128
DENSE_INTER = 256
DENSE_NUM_LAYERS = 2
DENSE_HEADS = 2
DENSE_KV_HEADS = 2
DENSE_HEAD_DIM = 64

_TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')


def _vocab_size_from_tokenizer(tokenizer_src: str) -> int:
    """Read the largest vocab id plus one from the tokenizer files."""
    with open(os.path.join(tokenizer_src, 'tokenizer.json'), encoding='utf-8') as f:
        tokenizer_json = json.load(f)
    size = len(tokenizer_json['model']['vocab'])
    for added in tokenizer_json.get('added_tokens', ()):
        size = max(size, added['id'] + 1)
    return size


def _copy_tokenizer(out_dir: str, tokenizer_src: str):
    for name in _TOKENIZER_FILES:
        src = os.path.join(tokenizer_src, name)
        if not os.path.isfile(src):
            raise FileNotFoundError(f'tokenizer file missing: {src}')
        shutil.copy(src, os.path.join(out_dir, name))


def _quantized(tensors: dict, name: str, n: int, k: int):
    """Append one ct group-128 triplet for one linear."""
    w = torch.randn(n, k, dtype=WEIGHT_DTYPE)
    w.view(n, k // GROUP_SIZE, GROUP_SIZE)[:, :, 0] *= 3.0
    packed, scale, zero = quantize_ct(w)
    tensors[f'{name}.weight_packed'] = packed
    tensors[f'{name}.weight_scale'] = scale
    tensors[f'{name}.weight_zero_point'] = zero


def _ct_quantization_config() -> dict:
    return {
        'quant_method': 'compressed-tensors',
        'format': 'pack-quantized',
        'quantization_status': 'compressed',
        'config_groups': {
            'group_0': {
                'targets': ['Linear'],
                'input_activations': None,
                'output_activations': None,
                'weights': {
                    'num_bits': 4,
                    'group_size': 128,
                    'strategy': 'group',
                    'symmetric': True,
                    'dynamic': False,
                    'type': 'int',
                },
            }
        },
        'ignore': [],
    }


def gen_tiny_dense_ct(out_dir: str, tokenizer_src: str, seed: int = 11, num_layers: int = DENSE_NUM_LAYERS) -> dict:
    """Write the tiny dense ct checkpoint with head_dim 64.

    Returns the config.
    """
    os.makedirs(out_dir, exist_ok=True)
    _copy_tokenizer(out_dir, tokenizer_src)
    torch.manual_seed(seed)
    hidden = DENSE_HIDDEN
    vocab = _vocab_size_from_tokenizer(tokenizer_src)

    tensors: dict[str, torch.Tensor] = {
        'model.embed_tokens.weight': torch.randn(vocab, hidden, dtype=WEIGHT_DTYPE) * 0.02,
    }
    for layer in range(num_layers):
        prefix = f'model.layers.{layer}'
        _quantized(tensors, f'{prefix}.self_attn.q_proj', DENSE_HEADS * DENSE_HEAD_DIM, hidden)
        _quantized(tensors, f'{prefix}.self_attn.k_proj', DENSE_KV_HEADS * DENSE_HEAD_DIM, hidden)
        _quantized(tensors, f'{prefix}.self_attn.v_proj', DENSE_KV_HEADS * DENSE_HEAD_DIM, hidden)
        _quantized(tensors, f'{prefix}.self_attn.o_proj', hidden, DENSE_HEADS * DENSE_HEAD_DIM)
        _quantized(tensors, f'{prefix}.mlp.gate_proj', DENSE_INTER, hidden)
        _quantized(tensors, f'{prefix}.mlp.up_proj', DENSE_INTER, hidden)
        _quantized(tensors, f'{prefix}.mlp.down_proj', hidden, DENSE_INTER)
        tensors[f'{prefix}.input_layernorm.weight'] = torch.ones(hidden, dtype=WEIGHT_DTYPE)
        tensors[f'{prefix}.post_attention_layernorm.weight'] = torch.ones(hidden, dtype=WEIGHT_DTYPE)
    tensors['model.norm.weight'] = torch.ones(hidden, dtype=WEIGHT_DTYPE)
    tensors['lm_head.weight'] = torch.randn(vocab, hidden, dtype=WEIGHT_DTYPE) * 0.02
    save_file(tensors, os.path.join(out_dir, 'model.safetensors'))

    config = {
        'architectures': ['LlamaForCausalLM'],
        'model_type': 'llama',
        'torch_dtype': 'float16',
        'hidden_size': hidden,
        'intermediate_size': DENSE_INTER,
        'num_hidden_layers': num_layers,
        'num_attention_heads': DENSE_HEADS,
        'num_key_value_heads': DENSE_KV_HEADS,
        'head_dim': DENSE_HEAD_DIM,
        'vocab_size': vocab,
        'max_position_embeddings': 512,
        'rms_norm_eps': 1e-5,
        'rope_theta': 10000.0,
        'tie_word_embeddings': False,
        'attention_bias': False,
        'quantization_config': _ct_quantization_config(),
    }
    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    return config
