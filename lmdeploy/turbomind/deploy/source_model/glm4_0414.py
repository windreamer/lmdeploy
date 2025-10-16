# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class Glm4_0414Reader(LlamaReader):
    """Glm4_0414Reader."""

    def _ffn(self, i: int, kind: str):
        if not kind:
            return self.filter(self.ffn_pattern)

        prefix = f'{self.attn_layer_prefix}.{i}.mlp'
        gate_up = self.params[f'{prefix}.gate_up_proj.{kind}']
        mid = gate_up.shape[0] // 2
        gate = self.transform(gate_up[:mid], kind)
        up = self.transform(gate_up[mid:], kind)
        down = self.transform(self.params[f'{prefix}.down_proj.{kind}'], kind)

        return (gate, down, up)

    def extra_norm(self, i: int) -> Dict[str, Any]:
        post_self_attn_layernorm_weight = self.transform(
            self.params[f'{self.attn_layer_prefix}.{i}.post_self_attn_layernorm.weight'], 'weight')
        post_mlp_layernorm_weight = self.transform(
            self.params[f'{self.attn_layer_prefix}.{i}.post_mlp_layernorm.weight'], 'weight')

        return {
            'post_self_attn_layernorm.weight': post_self_attn_layernorm_weight,
            'post_mlp_layernorm.weight': post_mlp_layernorm_weight,
        }


@INPUT_MODELS.register_module(name='glm4-0414')
class Glm4_0414Model(LlamaModel):
    """Glm4-0414 model in hf format."""

    Reader = Glm4_0414Reader

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)

    def model_info(self):
        cfg = self.model_config
        model_info = super().model_info()
        model_info['post_self_attn_norm'] = True
        model_info['post_mlp_norm'] = True
        model_info['attention_bias'] = cfg.get('attention_bias', False)
        model_info['partial_rotary_factor'] = cfg.get('partial_rotary_factor')
        return model_info
