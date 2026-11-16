# Copyright (c) OpenMMLab. All rights reserved.
"""Build request for the TurboMind GEMM linear route."""

from dataclasses import dataclass

import torch

from .awq_modules import LinearW4A16Impl
from .base import BuildSpec


@dataclass(frozen=True)
class LinearTmGemmBuildSpec(BuildSpec[LinearW4A16Impl]):
    """Immutable requirements for a W4A16 linear that prefers the TurboMind
    GEMM kernel.

    The spec type is intentionally separate from ``LinearW4A16BuildSpec``:
    ``BuildSpec`` objects carry immutable requirements, and the release
    policy below is TurboMind specific. Backends that do not know the type
    fall back through ``LinearW4A16Impl`` semantics, so the produced
    operator stays interchangeable with the AWQ Triton implementation.
    """

    in_features: int
    out_features: int
    w_bit: int
    group_size: int
    bias: bool
    output_dtype: torch.dtype | None
    # Spill the packed GPU torch weights to an exact CPU copy after packing
    # and free the GPU storage. Hot updates restore from the copy and re-pack.
    # Modules that consume ``get_unquantized_weight`` (MLA absorption such as
    # kv_b_proj) must be built with this disabled; see SPEC.md, weight
    # release decision and its MLA exemption clause.
    release_torch_weights: bool = True
    # ``'silu'`` packs gate/up with ``Linear.fuse_weight`` so the kernel
    # applies the SiLU activation in its epilogue. Fused projections reject
    # bias because the epilogue activation leaves no place for a per-half
    # bias add.
    fusion_type: str | None = None
