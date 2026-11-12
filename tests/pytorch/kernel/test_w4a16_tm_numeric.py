# Copyright (c) OpenMMLab. All rights reserved.
"""Numeric comparison between the TurboMind and triton W4A16 AWQ linear
implementations, plus gate-driven fallback assertions at the builder seam.

Runs on any CUDA device the TurboMind gate accepts (majors {8, 9}); other
devices skip the TM comparisons.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')

GROUP_SIZE = 128


AWQ_GEMM_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _pack_awq_gemm(values: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 into the AWQ-gemm int32 layout.

    ``values`` hold logical int4 codes; nibble ``j`` of each int32 stores the
    logical column ``AWQ_GEMM_ORDER.index(j)`` of its 8-wide group, matching
    the interleave the triton kernel and the TurboMind AWQFormat both decode.
    """
    codes = (values.to(torch.int32) + 8) % 16
    grouped = codes.unflatten(-1, (-1, 8))
    packed = torch.empty_like(grouped)
    for nibble in range(8):
        packed[..., nibble] = grouped[..., AWQ_GEMM_ORDER.index(nibble)]
    out = torch.zeros(packed.shape[:-1], dtype=torch.int32)
    for nibble in range(8):
        out |= packed[..., nibble] << (4 * nibble)
    return out


def _make_awq_weights(in_features: int, out_features: int, group_size: int, seed: int = 0):
    """Return (qweight, scales, qzeros, dequantized reference weight)."""
    generator = torch.Generator(device='cpu').manual_seed(seed)
    scales = (torch.randn((in_features // group_size, out_features), generator=generator) * 0.05 + 1.0).half()
    q4 = torch.randint(-8, 8, (in_features, out_features), generator=generator)
    z4 = torch.randint(-8, 8, (in_features // group_size, out_features), generator=generator)
    qweight = _pack_awq_gemm(q4)
    qzeros = _pack_awq_gemm(z4)

    z = z4.repeat_interleave(group_size, dim=0)
    weight = (q4 - z).float() * scales.float().repeat_interleave(group_size, dim=0)
    return qweight.cuda(), scales.cuda(), qzeros.cuda(), weight.cuda()


def _quality(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual = actual.float().flatten()
    expected = expected.float().flatten()
    nrmse = (actual - expected).norm() / (expected.norm() + 1e-6)
    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=0)
    return float(nrmse), float(cosine)


def _run_tm(x, qweight, scales, qzeros, group_size=GROUP_SIZE, bias=None):
    from lmdeploy.pytorch.backends.cuda.turbomind_awq_modules import TurbomindAwqLinearW4A16Impl

    impl = TurbomindAwqLinearW4A16Impl(x.size(-1), scales.size(1), 4, group_size)
    impl.update_weights(qweight, scales, qzeros, bias)
    try:
        return impl.forward(x, qweight, scales, qzeros, bias)
    finally:
        impl._release()


def _run_triton(x, qweight, scales, qzeros, group_size=GROUP_SIZE):
    from lmdeploy.pytorch.backends.cuda.awq_modules import AwqLinearW4A16Impl

    impl = AwqLinearW4A16Impl(x.size(-1), scales.size(1), 4, group_size)
    return impl.forward(x, qweight, scales, qzeros)


ALIGNED_SHAPES = [
    (256, 256),
    (512, 384),
    (2048, 1024),
]


@pytest.mark.parametrize('out_features,in_features', ALIGNED_SHAPES)
@pytest.mark.parametrize('batch', [1, 7])
@torch.inference_mode()
def test_tm_w4a16_matches_triton(out_features, in_features, batch):
    qweight, scales, qzeros, weight = _make_awq_weights(in_features, out_features, GROUP_SIZE)
    x = torch.randn((batch, in_features), dtype=torch.float16, device='cuda')

    reference = (x.float() @ weight).half()
    triton_out = _run_triton(x, qweight, scales, qzeros)
    tm_out = _run_tm(x, qweight, scales, qzeros)

    assert tm_out.shape == triton_out.shape
    tm_nrmse, tm_cos = _quality(tm_out, triton_out)
    assert tm_cos >= 0.999, f'TM vs triton cosine {tm_cos}'
    tr_nrmse, tr_cos = _quality(triton_out, reference)
    assert tr_cos >= 0.99, f'triton vs reference cosine {tr_cos}'
    tmr_nrmse, tmr_cos = _quality(tm_out, reference)
    assert tmr_cos >= 0.99, f'TM vs reference cosine {tmr_cos}'


@pytest.mark.parametrize('out_features,in_features', ALIGNED_SHAPES)
@torch.inference_mode()
def test_tm_w4a16_matches_triton_with_bias(out_features, in_features):
    qweight, scales, qzeros, _ = _make_awq_weights(in_features, out_features, GROUP_SIZE)
    bias = torch.randn((out_features, ), dtype=torch.float16, device='cuda')
    x = torch.randn((3, in_features), dtype=torch.float16, device='cuda')

    triton_out = _run_triton(x, qweight, scales, qzeros) + bias
    tm_out = _run_tm(x, qweight, scales, qzeros, bias=bias)

    _, cos = _quality(tm_out, triton_out)
    assert cos >= 0.999


@torch.inference_mode()
def test_triton_group32_matches_reference_and_tm_gate_refuses():
    """group 32 stays on triton (gate refuses); triton still numerically
    sound against the dequantized reference."""
    group_size = 32
    in_features, out_features = 256, 128
    qweight, scales, qzeros, weight = _make_awq_weights(in_features, out_features, group_size)
    x = torch.randn((4, in_features), dtype=torch.float16, device='cuda')

    from lmdeploy.pytorch.backends.cuda.awq_modules import AwqLinearW4A16Impl

    impl = AwqLinearW4A16Impl(in_features, out_features, 4, group_size)
    out = impl.forward(x, qweight, scales, qzeros)
    reference = (x.float() @ weight).half()
    _, cos = _quality(out, reference)
    assert cos >= 0.99

    from lmdeploy.pytorch.backends.cuda import awq_modules

    reason = awq_modules._w4a16_tm_feasibility_reason(in_features, out_features, 4, group_size, torch.float16)
    assert reason is not None and 'group_size' in reason


@torch.inference_mode()
def test_unaligned_shapes_fall_back_to_awq_via_builder():
    """K % 128 != 0 must resolve to the awq implementation at the builder
    seam even on a capability with a factory-table row."""
    from lmdeploy.pytorch.backends.awq_modules import LinearW4A16BuildSpec
    from lmdeploy.pytorch.backends.cuda import awq_modules

    capability = torch.cuda.get_device_capability()
    if capability[0] not in (8, 9):
        pytest.skip(f'capability {capability} has no w4a16 factory row')

    for in_features, out_features in [(130, 256), (256, 33)]:
        spec = LinearW4A16BuildSpec(in_features=in_features,
                                    out_features=out_features,
                                    w_bit=4,
                                    group_size=GROUP_SIZE,
                                    bias=False,
                                    output_dtype=torch.float16)
        impl = awq_modules._build_linear_w4a16(spec)
        assert type(impl) is awq_modules.AwqLinearW4A16Impl


@torch.inference_mode()
def test_auto_uses_turbomind_on_supported_capability():
    """On gate-passing capabilities with a factory row, default auto builds
    the TM implementation end to end and matches triton."""
    from lmdeploy.pytorch.backends.awq_modules import LinearW4A16BuildSpec
    from lmdeploy.pytorch.backends.cuda import awq_modules

    capability = torch.cuda.get_device_capability()
    if capability[0] not in (8, 9):
        pytest.skip(f'capability {capability} has no w4a16 factory row')

    in_features, out_features = 512, 256
    spec = LinearW4A16BuildSpec(in_features=in_features,
                                out_features=out_features,
                                w_bit=4,
                                group_size=GROUP_SIZE,
                                bias=False,
                                output_dtype=torch.float16)
    impl = awq_modules._build_linear_w4a16(spec)
    from lmdeploy.pytorch.backends.cuda.turbomind_awq_modules import TurbomindAwqLinearW4A16Impl

    assert type(impl) is TurbomindAwqLinearW4A16Impl

    qweight, scales, qzeros, _ = _make_awq_weights(in_features, out_features, GROUP_SIZE)
    x = torch.randn((5, in_features), dtype=torch.float16, device='cuda')
    impl.update_weights(qweight, scales, qzeros)
    try:
        tm_out = impl.forward(x, qweight, scales, qzeros)
    finally:
        impl._release()
    triton_out = _run_triton(x, qweight, scales, qzeros)
    _, cos = _quality(tm_out, triton_out)
    assert cos >= 0.999
