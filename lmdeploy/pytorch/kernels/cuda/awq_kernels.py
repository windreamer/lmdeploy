# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
from triton import language as tl


@triton.jit
def _dequant_s4_to_f16x2(weight, shift: tl.constexpr, is_top: tl.constexpr):

    immLut: tl.constexpr = (0xf0 & 0xcc) | 0xaa
    BOTTOM_MASK: tl.constexpr = 0x000f000f
    TOP_MASK: tl.constexpr = 0x00f000f0
    I4s_TO_F16s_MAGIC_NUM: tl.constexpr = 0x64006400
    FP16_TOP_MAGIC_NUM: tl.constexpr = 0x64006400
    ONE_SIXTEENTH: tl.constexpr = 0x2c002c00
    NEG_64: tl.constexpr = 0xd400d400

    if shift:
        weight = weight >> 8

    if is_top:
        return tl.inline_asm_elementwise("""{
        .reg .b32 tmp;
        lop3.b32 tmp, $2, $3, $4, $5;
        fma.rn.f16x2 tmp, tmp, $6, $7;
        mov.b32 {$0, $1}, tmp;
    }""",
                                         '=h,=h,r,n,n,n,r,r',
                                         args=[weight, TOP_MASK, I4s_TO_F16s_MAGIC_NUM, immLut, ONE_SIXTEENTH, NEG_64],
                                         dtype=(tl.float16, tl.float16),
                                         is_pure=True,
                                         pack=1)
    else:
        return tl.inline_asm_elementwise("""{
        .reg .b32 tmp;
        lop3.b32 tmp, $2, $3, $4, $5;
        sub.f16x2 tmp, tmp, $6;
        mov.b32 {$0, $1}, tmp;
    }""",
                                         '=h,=h,r,n,n,n,r',
                                         args=[weight, BOTTOM_MASK, I4s_TO_F16s_MAGIC_NUM, immLut, FP16_TOP_MAGIC_NUM],
                                         dtype=(tl.float16, tl.float16),
                                         is_pure=True,
                                         pack=1)


@triton.jit
def _unpack_weight(weight):
    """Unpack weight."""
    # broadcast and shift
    width: tl.constexpr = 8
    BLOCK_SIZE_K: tl.constexpr = weight.shape[0]
    BLOCK_SIZE_QN: tl.constexpr = weight.shape[1]
    BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_QN * width

    w0, w1 = _dequant_s4_to_f16x2(weight, False, False)
    w2, w3 = _dequant_s4_to_f16x2(weight, False, True)
    w4, w5 = _dequant_s4_to_f16x2(weight, True, False)
    w6, w7 = _dequant_s4_to_f16x2(weight, True, True)

    w04 = tl.join(w0, w4)
    w15 = tl.join(w1, w5)
    w26 = tl.join(w2, w6)
    w37 = tl.join(w3, w7)
    w0246 = tl.join(w04, w26)
    w1357 = tl.join(w15, w37)
    weight = tl.join(w0246, w1357)

    return weight.reshape(BLOCK_SIZE_K, BLOCK_SIZE_N)

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_N': 64,  'GROUP_SIZE_M': 8, 'SPLIT_K': 1, 'NUM_STAGES': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1, 'NUM_STAGES': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1, 'NUM_STAGES': 3}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1, 'NUM_STAGES': 3}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1, 'NUM_STAGES': 5}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1, 'NUM_STAGES': 3}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1, 'NUM_STAGES': 5}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 64,  'GROUP_SIZE_M': 8, 'SPLIT_K': 2, 'NUM_STAGES': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 2, 'NUM_STAGES': 3}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256, 'GROUP_SIZE_M': 8, 'SPLIT_K': 2, 'NUM_STAGES': 3}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 64,  'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'NUM_STAGES': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'NUM_STAGES': 3}, num_warps=4),
    ]

_MAX_STAGES = None
_MIN_STAGES = None

def _init_stages():
    global _MAX_STAGES, _MIN_STAGES
    if _MAX_STAGES is not None:
        return
    props = torch.cuda.get_device_properties(0)
    if props.major >= 9:                        # Hopper
        _MIN_STAGES = 3
        _MAX_STAGES = 5
    elif props.major == 8 and props.minor == 0: # A100
        _MIN_STAGES = 3
        _MAX_STAGES = 5
    elif props.major == 8:                      # Ada/A10
        _MIN_STAGES = 2
        _MAX_STAGES = 3
    else:                                       # Volta/Turing
        _MIN_STAGES = 2
        _MAX_STAGES = 2


def _awq_config_pruner(configs, nargs, **kwargs):
    _init_stages()
    n = nargs['N']
    k = nargs['K']
    num_groups = k // 128

    used = set()
    for config in configs:
        bsn = config.kwargs['BLOCK_SIZE_N']
        sk = config.kwargs['SPLIT_K']
        ns = config.kwargs['NUM_STAGES']
        nw = config.num_warps

        if bsn > n:
            continue
        if ns > _MAX_STAGES:
            continue
        if ns < _MIN_STAGES:
            continue
        if sk > 1 and num_groups // sk < 4:
            continue

        key = (bsn, sk, ns, nw)
        if key in used:
            continue
        used.add(key)
        yield config



@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['N', 'K'],
    reset_to_zero=['c_ptr'],
    prune_configs_by={'early_config_prune': _awq_config_pruner},
)
@triton.jit
def awq_linear_kernel(
        a_ptr,
        qw_ptr,
        s_ptr,
        qz_ptr,
        c_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_am,
        stride_ak: tl.constexpr,  #
        stride_wk: tl.constexpr,
        stride_wn: tl.constexpr,  #
        stride_sk: tl.constexpr,
        stride_sn: tl.constexpr,  #
        stride_zk: tl.constexpr,
        stride_zn: tl.constexpr,  #
        stride_cm,
        stride_cn: tl.constexpr,
        # Meta-parameters
        SPLIT_K: tl.constexpr,
        NUM_STAGES: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.

    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    kid = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    BLOCK_SIZE_QN: tl.constexpr = BLOCK_SIZE_N // 8
    offs_wn = pid_n * BLOCK_SIZE_QN + tl.arange(0, BLOCK_SIZE_QN)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    qw_ptrs = qw_ptr + (offs_k[:, None] * stride_wk + offs_wn[None, :] * stride_wn)
    s_ptrs = s_ptr + offs_bn * stride_sn
    qz_ptrs = qz_ptr + offs_wn * stride_zn

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    k_start = kid
    k_last = K // BLOCK_SIZE_K

    # prefetch
    a_ptrs += k_start * BLOCK_SIZE_K * stride_ak
    qw_ptrs += k_start * BLOCK_SIZE_K * stride_wk
    s_ptrs += k_start * stride_sk
    qz_ptrs += k_start * stride_zk
    qw = tl.load(qw_ptrs)
    qz = tl.load(qz_ptrs)[None, :]
    s = tl.load(s_ptrs)[None, :]
    qw_ptrs += SPLIT_K * BLOCK_SIZE_K * stride_wk
    s_ptrs += SPLIT_K * stride_sk
    qz_ptrs += SPLIT_K * stride_zk

    for k in tl.range(k_start, k_last, SPLIT_K, num_stages=NUM_STAGES):

        # unpack b
        z = _unpack_weight(qz)
        w = _unpack_weight(qw)
        b = (w - z) * s

        # load a
        a = tl.load(a_ptrs)

        # load next q
        mask = k + SPLIT_K < k_last
        qz = tl.load(qz_ptrs, mask=mask)[None, :]
        s = tl.load(s_ptrs, mask=mask)[None, :]
        qw = tl.load(qw_ptrs, mask=mask)

        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, acc=accumulator)

        # Advance the ptrs to the next K block.
        a_ptrs += SPLIT_K * BLOCK_SIZE_K * stride_ak
        qw_ptrs += SPLIT_K * BLOCK_SIZE_K * stride_wk
        s_ptrs += SPLIT_K * stride_sk
        qz_ptrs += SPLIT_K * stride_zk

    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if SPLIT_K > 1:
        tl.atomic_add(c_ptrs, c, mask=c_mask, sem='relaxed', scope='gpu')
    else:
        tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def _unpack_npack_fast(packed, meta_dtype):
    """Unpack [BLOCK_K, BLOCK_QN] int32 → [BLOCK_K, BLOCK_QN*8] fp16."""
    BLOCK_K: tl.constexpr = packed.shape[0]
    BLOCK_QN: tl.constexpr = packed.shape[1]
    BLOCK_N: tl.constexpr = BLOCK_QN * 8
    shifts = tl.arange(0, 8) * 4
    w = (packed[:, :, None] >> shifts[None, None, :]) & 0xF
    w = w.reshape(BLOCK_K, BLOCK_N)
    if meta_dtype == tl.float16:
        return w.to(tl.float16)
    else:
        return w.to(tl.bfloat16)


_GEMV_CFGS = [
    (128, 32, 2),
    (256, 32, 2),
    (512, 32, 4),
]


def _gemv_configs():
    return [
        triton.Config(
            {'BLOCK_N': c[0], 'BLOCK_K': c[1]},
            num_warps=c[2],
            num_stages=1,
            pre_hook=lambda nargs: nargs['c_ptr'].zero_(),
        )
        for c in _GEMV_CFGS
    ]


@triton.autotune(
    configs=_gemv_configs(),
    key=['N', 'K', 'group_size'],
)
@triton.jit
def awq_gemv_kernel(
    a_ptr, qw_ptr, s_ptr, qz_ptr, c_ptr,
    N: tl.constexpr, K: tl.constexpr, group_size: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_wk: tl.constexpr, stride_wn: tl.constexpr,
    stride_sk: tl.constexpr, stride_sn: tl.constexpr,
    stride_zk: tl.constexpr, stride_zn: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """GEMV for BS=1 with packed weights AND packed zeros."""
    BLOCK_QN: tl.constexpr = BLOCK_N // 8

    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1) * 2

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_qn = pid_n * BLOCK_QN + tl.arange(0, BLOCK_QN)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # metadata: scales are [G, N] fp16, qzeros are [G, N//8] int32
    g = pid_k * BLOCK_K // group_size
    scales = tl.load(
        s_ptr + g * stride_sk + offs_n * stride_sn,
        mask=offs_n < N, other=0.0,
    ).to(tl.float16)

    qz_packed = tl.load(
        qz_ptr + g * stride_zk + offs_qn * stride_zn,
        mask=offs_qn < (N // 8), other=0,
    )
    zeros = _unpack_npack_fast(qz_packed.reshape(1, BLOCK_QN), tl.float16).reshape(BLOCK_N)

    # ── First K-block ──
    a = tl.load(a_ptr + offs_k * stride_ak, mask=offs_k < K, other=0.0).to(tl.float32)
    qw = tl.load(
        qw_ptr + offs_k[:, None] * stride_wk + offs_qn[None, :] * stride_wn,
        mask=(offs_k[:, None] < K) & (offs_qn[None, :] < (N // 8)), other=0,
    )
    w = _unpack_npack_fast(qw, tl.float16)
    w = ((w - zeros[None, :]) * scales[None, :]).to(tl.float32)
    acc = tl.sum(a[:, None] * w, axis=0)

    # ── Second K-block ──
    offs_k2 = offs_k + BLOCK_K
    a2 = tl.load(a_ptr + offs_k2 * stride_ak, mask=offs_k2 < K, other=0.0).to(tl.float32)
    qw2 = tl.load(
        qw_ptr + offs_k2[:, None] * stride_wk + offs_qn[None, :] * stride_wn,
        mask=(offs_k2[:, None] < K) & (offs_qn[None, :] < (N // 8)), other=0,
    )
    w2 = _unpack_npack_fast(qw2, tl.float16)
    w2 = ((w2 - zeros[None, :]) * scales[None, :]).to(tl.float32)
    acc += tl.sum(a2[:, None] * w2, axis=0)

    # ── Store ──
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_N), BLOCK_N)
    tl.atomic_add(c_ptr + offs_cn * stride_cn, acc.to(tl.float16),
                  mask=offs_cn < N, sem='relaxed')


def _get_block_size_m(M):
    if M <= 16:
        return 16
    elif M <= 64:
        return 64
    else:
        return 128


def awq_linear(x, qweight, scales, qzeros):
    M = x.size(0)
    K = qweight.size(0)
    N = scales.size(1)
    group_size = K // scales.size(0)

    # BS=1: GEMV
    if M == 1:
        out = torch.zeros(1, N, dtype=x.dtype, device=x.device)
        def grid(meta):
            return (
                    triton.cdiv(N, meta['BLOCK_N']),
                    triton.cdiv(K, meta['BLOCK_K'] * 2),
                )
        awq_gemv_kernel[grid](
            x, qweight, scales, qzeros, out,
            N, K, group_size,
            x.stride(1),
            qweight.stride(0), qweight.stride(1),
            scales.stride(0), scales.stride(1),
            qzeros.stride(0), qzeros.stride(1),
            out.stride(1),
        )
        return out

    # BS>1: GEMM
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            META['SPLIT_K'],
        )

    out = torch.zeros(M, N, dtype=x.dtype, device=x.device)

    BLOCK_SIZE_M = _get_block_size_m(M)

    awq_linear_kernel[grid](
        x, qweight, scales, qzeros, out,
        M, N, K,
        stride_am=x.stride(0), stride_ak=x.stride(1),
        stride_wk=qweight.stride(0), stride_wn=qweight.stride(1),
        stride_sk=scales.stride(0), stride_sn=scales.stride(1),
        stride_zk=qzeros.stride(0), stride_zn=qzeros.stride(1),
        stride_cm=out.stride(0), stride_cn=out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=group_size,
    )
    return out
