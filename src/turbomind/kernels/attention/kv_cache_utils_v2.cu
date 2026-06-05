// Copyright (c) OpenMMLab. All rights reserved.

#include <type_traits>

#include "src/turbomind/kernels/attention/block.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/attention/rotary_embedding.h"
#include "src/turbomind/kernels/attention/turbo_quant_qjl4.h"
#include "src/turbomind/kernels/attention/turbo_quant_v2.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/thread_map.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

using cutlass::FastDivmod;

template<class KvQuant, int CTA_S, int HeadDim, int WarpCnt, class T, class BlockLayout>
__global__ void __launch_bounds__(128) ProcessKV_v2(char**          blocks,
                                                    const T*        k,
                                                    const T*        v,
                                                    const T*        k_bias,
                                                    const T*        v_bias,
                                                    const int*      cu_q_len,
                                                    const int*      cu_k_len,
                                                    const int*      cu_block_num,
                                                    RopeKernelParam rope_param,
                                                    int64_t         stride_b,
                                                    int64_t         stride_c,
                                                    int64_t         stride_h,
                                                    int64_t         stride_s,
                                                    int             layer_id,
                                                    int             cp_rank,
                                                    FastDivmod      cp_size,
                                                    BlockLayout     block_layout)
{
    using Trait = attention::KvQuantTrait<KvQuant, T>;
    using TK    = typename Trait::StorageK;
    using TV    = typename Trait::StorageV;

    constexpr bool kQuantKV = Trait::kQuantKV;

    constexpr int kVecSize = sizeof(uint4) / sizeof(T);

    using Vec = Array<T, kVecSize>;
    using Map = RakedThreadMap<HeadDim, CTA_S, kVecSize, WarpCnt>;

    constexpr int ITER_C = Map::kIterC;
    constexpr int ITER_S = Map::kIterS;

    constexpr bool HAS_V = !(typename BlockLayout::Config{}.is_share_kv());

    const int token_idx = blockIdx.x * CTA_S;  // local offset into `input_length`
    const int head_idx  = blockIdx.y;
    const int batch_idx = blockIdx.z;

    const int qi_beg = cu_q_len[batch_idx];
    const int qi_end = cu_q_len[batch_idx + 1];
    const int q_len  = qi_end - qi_beg;

    const int k_len       = cu_k_len[batch_idx + 1] - cu_k_len[batch_idx];
    const int history_len = k_len - q_len;

    if (qi_beg + token_idx >= qi_end) {  // empty tile
        return;
    }

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int2 offset = Map::get_offset(warp_id, lane_id);

    Vec __align__(16) vec_K[ITER_S][ITER_C];
    Vec __align__(16) vec_V[ITER_S][ITER_C];

    Vec bias_V[ITER_C];
    Vec bias_K[ITER_C];

    if (k_bias) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            Ldg(bias_K[c], &k_bias[head_idx * HeadDim + di]);
        }
    }
    if (v_bias && HAS_V) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            Ldg(bias_V[c], &v_bias[head_idx * HeadDim + di]);
        }
    }

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int     qi = offset.y + s * Map::kDeltaS + token_idx;  // sequence local
            const int     di = offset.x + c * Map::kDeltaC;
            const int64_t index =
                (batch_idx * stride_b + qi_beg * stride_c + qi * stride_s + head_idx * stride_h) * HeadDim + di;
            if (qi < q_len) {
                Ldg(vec_K[s][c], &k[index]);
                if constexpr (HAS_V) {
                    Ldg(vec_V[s][c], &v[index]);
                }
            }
            else {
                // Zero-initialize unused slots to prevent NaN propagation through warp reductions
                PRAGMA_UNROLL
                for (int j = 0; j < kVecSize; ++j) {
                    vec_K[s][c][j] = (T)0.f;
                    if constexpr (HAS_V) {
                        vec_V[s][c][j] = (T)0.f;
                    }
                }
            }
        }
    }

    if (k_bias) {
        using namespace ops;
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                vec_K[s][c] = vec_K[s][c] + bias_K[c];
            }
        }
    }
    if (v_bias && HAS_V) {
        using namespace ops;
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                vec_V[s][c] = vec_V[s][c] + bias_V[c];
            }
        }
    }

    if (rope_param.type != RopeType::kNull) {
        FastRoPE rope(rope_param, batch_idx, std::integral_constant<int, kVecSize>{});
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            rope.init(di);
            PRAGMA_UNROLL
            for (int s = 0; s < ITER_S; ++s) {
                const int ti = history_len + offset.y + s * Map::kDeltaS + token_idx;  // sequence local
                rope.apply(vec_K[s][c], ti);
            }
        }
    }

    Array<T, 2> param_K[ITER_S];
    Array<T, 2> param_V[ITER_S];

    // Output arrays for quantized data (used by both affine and TurboQuant paths)
    Array<TK, kVecSize> out_K[ITER_S][ITER_C];
    // For 16× packed sub-byte V (uint2_t), out_V is unused in the TurboQuant Hadamard path.
    // Declare with appropriate size to satisfy Array<N%16==0> constraint.
    static constexpr int kVOutSize =
        (Trait::kBitsV < 8) ? (sizeof(typename detail::__storage_of<TV>::type) * 8 / Trait::kBitsV) : kVecSize;
    Array<TV, kVOutSize> out_V[ITER_S][ITER_C];

    blocks += cu_block_num[batch_idx];

    block::Head<T, KvQuant, BlockLayout> block_head{block_layout, layer_id, head_idx};

    if constexpr (kQuantKV) {
        if constexpr (Trait::kHadamardRotate) {
            // TurboQuant: Hadamard rotate + codebook quantize
            // Step 1: Hadamard rotate K/V (per-token, in-place on vec_K/V)
            PRAGMA_UNROLL
            for (int s = 0; s < ITER_S; ++s) {
                PRAGMA_UNROLL
                for (int c = 0; c < ITER_C; ++c) {
                    // Register butterfly
                    Array<float, kVecSize> fval_k;
                    PRAGMA_UNROLL
                    for (int j = 0; j < kVecSize; ++j) {
                        fval_k[j] = (float)vec_K[s][c][j];
                    }
                    // Local butterfly (log2(kVecSize)=3 stages)
                    PRAGMA_UNROLL
                    for (int stg = 0; stg < 3; ++stg) {
                        const int stride = 1 << stg;
                        PRAGMA_UNROLL
                        for (int j = 0; j < kVecSize; ++j) {
                            if (j & stride) {
                                float a            = fval_k[j - stride];
                                float b            = fval_k[j];
                                fval_k[j - stride] = a + b;
                                fval_k[j]          = a - b;
                            }
                        }
                    }
                    // Shuffle butterfly
                    PRAGMA_UNROLL
                    for (int stg = 0; stg < (HeadDim >= 256 ? 5 : HeadDim >= 128 ? 4 : 3); ++stg) {
                        const int   mask = 1 << stg;
                        const float sign = (lane_id & mask) ? -1.f : 1.f;
                        PRAGMA_UNROLL
                        for (int j = 0; j < kVecSize; ++j) {
                            float other = __shfl_xor_sync(uint32_t(-1), fval_k[j], mask);
                            fval_k[j]   = sign * fval_k[j] + other;
                        }
                    }
                    // Scale + store back
                    const float had_scale = 1.0f / sqrtf((float)HeadDim);
                    PRAGMA_UNROLL
                    for (int j = 0; j < kVecSize; ++j) {
                        vec_K[s][c][j] = (T)(fval_k[j] * had_scale);
                    }
                    // Same for V
                    if constexpr (HAS_V) {
                        Array<float, kVecSize> fval_v;
                        PRAGMA_UNROLL
                        for (int j = 0; j < kVecSize; ++j) {
                            fval_v[j] = (float)vec_V[s][c][j];
                        }
                        PRAGMA_UNROLL
                        for (int stg = 0; stg < 3; ++stg) {
                            const int stride = 1 << stg;
                            PRAGMA_UNROLL
                            for (int j = 0; j < kVecSize; ++j) {
                                if (j & stride) {
                                    float a            = fval_v[j - stride];
                                    float b            = fval_v[j];
                                    fval_v[j - stride] = a + b;
                                    fval_v[j]          = a - b;
                                }
                            }
                        }
                        PRAGMA_UNROLL
                        for (int stg = 0; stg < (HeadDim >= 256 ? 5 : HeadDim >= 128 ? 4 : 3); ++stg) {
                            const int   mask = 1 << stg;
                            const float sign = (lane_id & mask) ? -1.f : 1.f;
                            PRAGMA_UNROLL
                            for (int j = 0; j < kVecSize; ++j) {
                                float other = __shfl_xor_sync(uint32_t(-1), fval_v[j], mask);
                                fval_v[j]   = sign * fval_v[j] + other;
                            }
                        }
                        PRAGMA_UNROLL
                        for (int j = 0; j < kVecSize; ++j) {
                            vec_V[s][c][j] = (T)(fval_v[j] * had_scale);
                        }
                    }
                }
            }
            // Step 2: L2 norms + quantize + store to block cache (per-token)
            PRAGMA_UNROLL
            for (int s = 0; s < ITER_S; ++s) {
                float sum_k_sq = 0.f, sum_v_sq = 0.f, residual_sq = 0.f;
                PRAGMA_UNROLL
                for (int c = 0; c < ITER_C; ++c) {
                    PRAGMA_UNROLL
                    for (int j = 0; j < kVecSize; ++j) {
                        float kv = (float)vec_K[s][c][j];
                        sum_k_sq += kv * kv;
                        if constexpr (HAS_V) {
                            float vv = (float)vec_V[s][c][j];
                            sum_v_sq += vv * vv;
                        }
                    }
                }
                // Warp reduction: only across C-dimension lanes (kWarpThreadC),
                // NOT across S-dimension (different tokens share the same warp).
                // Using kWarpThreadC as limit avoids cross-token contamination.
                PRAGMA_UNROLL
                for (int mask = Map::kWarpThreadC / 2; mask >= 1; mask /= 2) {
                    sum_k_sq += __shfl_xor_sync(uint32_t(-1), sum_k_sq, mask);
                    sum_v_sq += __shfl_xor_sync(uint32_t(-1), sum_v_sq, mask);
                }
                float mse_norm     = sqrtf(sum_k_sq);
                float v_norm       = sqrtf(sum_v_sq);
                float inv_mse_norm = (mse_norm > 0.f) ? 1.f / mse_norm : 0.f;
                float inv_v_norm   = (v_norm > 0.f) ? 1.f / v_norm : 0.f;
                float sigma        = 1.f / sqrtf((float)HeadDim);
                // QJL4 quantize K + accumulate residual
                uint32_t packed_k_arr[ITER_C];
                PRAGMA_UNROLL
                for (int c = 0; c < ITER_C; ++c) {
                    uint32_t packed = 0;
                    PRAGMA_UNROLL
                    for (int j = 0; j < kVecSize; ++j) {
                        float y        = (float)vec_K[s][c][j] * inv_mse_norm;
                        int   idx3     = attention::quantize_k4_mse(y, sigma);
                        float centroid = attention::dequantize_k4_mse(idx3, sigma);
                        float res      = y - centroid;
                        int   sign_bit = (res >= 0.f) ? 1 : 0;
                        residual_sq += res * res;
                        packed |= ((uint32_t)(idx3 | (sign_bit << 3)) << (j * 4));
                    }
                    packed_k_arr[c] = packed;
                }
                PRAGMA_UNROLL
                for (int mask = Map::kWarpThreadC / 2; mask >= 1; mask /= 2) {
                    residual_sq += __shfl_xor_sync(uint32_t(-1), residual_sq, mask);
                }
                float qjl_norm = sqrtf(residual_sq / (float)HeadDim);
                // 2-bit MSE quantize V
                uint16_t packed_v_arr[ITER_C];
                if constexpr (HAS_V) {
                    PRAGMA_UNROLL
                    for (int c = 0; c < ITER_C; ++c) {
                        uint16_t pv = 0;
                        PRAGMA_UNROLL
                        for (int j = 0; j < kVecSize; ++j) {
                            float u    = (float)vec_V[s][c][j] * inv_v_norm;
                            int   idx2 = attention::quantize_v2(u, sigma);
                            pv |= ((uint16_t)idx2 << (j * 2));
                        }
                        packed_v_arr[c] = pv;
                    }
                }
                // Warp shuffle for 16× packed V: combine two threads' 8-value packs into one uint32_t word.
                // Even C-dim threads (lane_id & 1 == 0) hold the lower 8 values; odd threads hold the upper 8.
                // After shuffle, only even threads write the combined word to avoid data races.
                uint32_t combined_v_arr[ITER_C];
                if constexpr (HAS_V && Trait::kBitsV < 8) {
                    PRAGMA_UNROLL
                    for (int c = 0; c < ITER_C; ++c) {
                        uint32_t neighbor = __shfl_xor_sync(uint32_t(-1), (uint32_t)packed_v_arr[c], 1);
                        if (lane_id & 1) {
                            combined_v_arr[c] = ((uint32_t)packed_v_arr[c] << 16) | (neighbor & 0xFFFF);
                        }
                        else {
                            combined_v_arr[c] = (neighbor << 16) | (uint32_t)packed_v_arr[c];
                        }
                    }
                }
                param_K[s][0] = (T)mse_norm;
                param_K[s][1] = (T)qjl_norm;
                param_V[s][0] = (T)v_norm;
                param_V[s][1] = (T)0;
                // Store packed data + params to block cache
                const int qi = offset.y + s * Map::kDeltaS + token_idx;
                const int ti = history_len + qi;
                int       lti, lti_rank;
                lti = cp_size.divmod(lti_rank, ti);
                if (qi < q_len && lti_rank == cp_rank) {
                    block_head.with((char**)blocks, lti, [&](auto k_cache, auto v_cache, T* k_param, T* v_param) {
                        PRAGMA_UNROLL
                        for (int c = 0; c < ITER_C; ++c) {
                            int di = offset.x + c * Map::kDeltaC;
                            Store(&k_cache[di], (const Array<TK, kVecSize>&)packed_k_arr[c]);
                            if constexpr (HAS_V) {
                                if constexpr (Trait::kBitsV < 8) {
                                    // 16× packing: only even C-dim threads write the combined word
                                    if ((lane_id & 1) == 0) {
                                        Store(&v_cache[di], (const Array<TV, 16>&)combined_v_arr[c]);
                                    }
                                }
                                else {
                                    Store(&v_cache[di], (const Array<TV, kVecSize>&)packed_v_arr[c]);
                                }
                            }
                        }
                        if (offset.x == 0) {
                            Store(k_param, param_K[s]);
                            if constexpr (HAS_V) {
                                Store(v_param, param_V[s]);
                            }
                        }
                    });
                }
            }
            return;
        }
        else {
            // Affine quantization path (kQuantKV but no Hadamard)
            PRAGMA_UNROLL
            for (int s = 0; s < ITER_S; ++s) {
                ConvertKvCache<T, typename Trait::StorageK> conv_K{param_K[s][0], param_K[s][1]};
                ConvertKvCache<T, typename Trait::StorageV> conv_V{param_V[s][0], param_V[s][1]};
                PRAGMA_UNROLL
                for (int c = 0; c < ITER_C; ++c) {
                    out_K[s][c] = conv_K(vec_K[s][c]);
                    if constexpr (HAS_V) {
                        out_V[s][c] = conv_V(vec_V[s][c]);
                    }
                }
            }
        }
    }
    else {
        // No quantization (kQuantKV == false): identity conversion
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                out_K[s][c] = (Array<TK, kVecSize>&)vec_K[s][c];
                if constexpr (HAS_V) {
                    out_V[s][c] = (Array<TV, kVecSize>&)vec_V[s][c];
                }
            }
        }
    }

    int local_ti, local_ti_rank;

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        const int qi = offset.y + s * Map::kDeltaS + token_idx;  // local offset into `input_length`
        const int ti = history_len + qi;                         // timestep
        local_ti     = cp_size.divmod(local_ti_rank, ti);
        if (qi < q_len && local_ti_rank == cp_rank) {
            block_head.with((char**)blocks, local_ti, [&](auto k_cache, auto v_cache, T* k_param, T* v_param) {
                PRAGMA_UNROLL
                for (int c = 0; c < ITER_C; ++c) {
                    int di = offset.x + c * Map::kDeltaC;
                    if constexpr (!Trait::kHadamardRotate) {
                        Store(&k_cache[di], out_K[s][c]);
                        if constexpr (HAS_V) {
                            Store(&v_cache[di], out_V[s][c]);
                        }
                    }
                }
                if constexpr (kQuantKV && !Trait::kHadamardRotate) {
                    if (offset.x == 0) {
                        StoreQuantParam<TK>(k_param, param_K[s]);
                        if constexpr (HAS_V) {
                            StoreQuantParam<TV>(v_param, param_V[s]);
                        }
                    }
                }
            });
        }
    }
}

template<class T>
void invokeProcessKV_v2(char**                 blocks,
                        const T*               k,
                        const T*               v,
                        const T*               k_bias,
                        const T*               v_bias,
                        const int*             cu_q_len,
                        const int*             cu_k_len,
                        const int*             cu_block_num,
                        const RopeKernelParam& rope_param,
                        int64_t                stride_b,
                        int64_t                stride_c,
                        int64_t                stride_h,
                        int64_t                stride_s,
                        int                    block_seq_len,
                        int                    layer_id,
                        int                    cp_rank,
                        FastDivmod             cp_size,
                        int                    max_q_len,
                        int                    head_num,
                        int                    head_dim,
                        int                    batch_size,
                        int                    quant_policy,
                        cudaStream_t           stream)
{

    auto invoke = [&](auto kv_quant, const auto dim) {
        using KvQuantT = decltype(kv_quant);

        constexpr int  kHeadDim = dim;
        constexpr bool kShareKV = kHeadDim == 576;

        constexpr int WARPS = 4;
        constexpr int CTA_S = kShareKV ? 32 : 64;

        int  block = WARPS * WARP_SIZE;
        dim3 grid(cdiv(max_q_len, CTA_S), head_num, batch_size);

        TM_CHECK_EQ(head_dim, kHeadDim);

        block::Layout block_layout{block::Config<T, KvQuantT, kHeadDim, kShareKV>{head_num, block_seq_len}};

        ProcessKV_v2<KvQuantT, CTA_S, kHeadDim, WARPS><<<grid, block, 0, stream>>>(blocks,
                                                                                   k,
                                                                                   v,
                                                                                   k_bias,
                                                                                   v_bias,
                                                                                   cu_q_len,
                                                                                   cu_k_len,
                                                                                   cu_block_num,
                                                                                   rope_param,
                                                                                   stride_b,
                                                                                   stride_c,
                                                                                   stride_h,
                                                                                   stride_s,
                                                                                   layer_id,
                                                                                   cp_rank,
                                                                                   cp_size,
                                                                                   block_layout);
    };

    auto dispatch = [&](auto kv_quant) {
        if (0) {}
        else if (head_dim == 64) {
            return invoke(kv_quant, std::integral_constant<int, 64>{});
        }
        else if (head_dim == 128) {
            return invoke(kv_quant, std::integral_constant<int, 128>{});
        }
        else if (head_dim == 192) {
            return invoke(kv_quant, std::integral_constant<int, 192>{});
        }
        else if (head_dim == 256) {
            return invoke(kv_quant, std::integral_constant<int, 256>{});
        }
        else if (head_dim == 576) {
            return invoke(kv_quant, std::integral_constant<int, 576>{});
        }
        TM_UNREACHABLE;
    };

    if (quant_policy == 42) {
        dispatch(attention::KvQuantTurbo{});
    }
    else if (quant_policy & QuantPolicy::kCacheKVInt8) {
        dispatch(attention::KvQuantInt8{});
    }
    else if (quant_policy & QuantPolicy::kCacheKVInt4) {
        dispatch(attention::KvQuantInt4{});
    }
    else {
        dispatch(attention::KvQuantNone{});
    }

    TM_CUDA_CHECK(cudaGetLastError());
}

#define INSTANTIATE_invokeProcessKV_v2(type)                                                                           \
    template void invokeProcessKV_v2(char**                 blocks,                                                    \
                                     const type*            k,                                                         \
                                     const type*            v,                                                         \
                                     const type*            k_bias,                                                    \
                                     const type*            v_bias,                                                    \
                                     const int*             cu_q_len,                                                  \
                                     const int*             cu_k_len,                                                  \
                                     const int*             cu_block_num,                                              \
                                     const RopeKernelParam& rope_param,                                                \
                                     int64_t                stride_b,                                                  \
                                     int64_t                stride_c,                                                  \
                                     int64_t                stride_h,                                                  \
                                     int64_t                stride_s,                                                  \
                                     int                    block_seq_len,                                             \
                                     int                    layer_id,                                                  \
                                     int                    cp_rank,                                                   \
                                     FastDivmod             cp_size,                                                   \
                                     int                    max_q_len,                                                 \
                                     int                    head_num,                                                  \
                                     int                    head_dim,                                                  \
                                     int                    batch_size,                                                \
                                     int                    quant_policy,                                              \
                                     cudaStream_t           stream);

INSTANTIATE_invokeProcessKV_v2(half);
#if ENABLE_BF16
INSTANTIATE_invokeProcessKV_v2(nv_bfloat16);
#endif

template<int CTA_S, int HeadDim, int WarpCnt, class T, class KvQuant, class BlockLayout>
__global__ void __launch_bounds__(128) flattenKV_v2(T*              k,
                                                    T*              v,
                                                    char**          blocks,
                                                    const int*      cu_k_len,
                                                    const int*      cu_block_num,
                                                    RopeKernelParam rope_param,
                                                    int64_t         stride_b,
                                                    int64_t         stride_c,
                                                    int64_t         stride_h,
                                                    int64_t         stride_s,
                                                    int             layer_id,
                                                    int             cp_rank,
                                                    FastDivmod      cp_size,
                                                    BlockLayout     block_layout)
{
    using Trait = attention::KvQuantTrait<KvQuant, T>;
    using TK    = typename Trait::StorageK;
    using TV    = typename Trait::StorageV;

    constexpr bool kQuantKV = Trait::kQuantKV;

    constexpr int kVecSize = sizeof(uint4) / sizeof(T);

    using Map = RakedThreadMap<HeadDim, CTA_S, kVecSize, WarpCnt>;

    constexpr int ITER_C = Map::kIterC;
    constexpr int ITER_S = Map::kIterS;

    constexpr bool HAS_V = !(typename BlockLayout::Config{}.is_share_kv());

    const int token_idx = blockIdx.x * CTA_S;
    const int head_idx  = blockIdx.y;
    const int batch_idx = blockIdx.z;

    const int ti_0   = cu_k_len[0];
    const int ti_beg = cu_k_len[batch_idx] - ti_0;
    const int ti_end = cu_k_len[batch_idx + 1] - ti_0;

    const int seq_len = ti_end - ti_beg;

    if (ti_beg + token_idx >= ti_end) {  // empty tile
        return;
    }

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int2 offset = Map::get_offset(warp_id, lane_id);

    Array<TK, kVecSize> __align__(16) vec_K[ITER_S][ITER_C];
    // For sub-byte V types with 16× packing, each read covers 16 elements (one uint32_t word).
    // For other types, kVecSize elements per read.
    static constexpr int kVVecSize =
        (Trait::kBitsV < 8) ? (sizeof(typename detail::__storage_of<TV>::type) * 8 / Trait::kBitsV) : kVecSize;
    Array<TV, kVVecSize> __align__(16) vec_V[ITER_S][ITER_C];

    Array<T, kVecSize> __align__(16) out_K[ITER_S][ITER_C];
    Array<T, kVecSize> __align__(16) out_V[ITER_S][ITER_C];

    // Zero-initialize output arrays (TurboQuant path may skip slots)
    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            PRAGMA_UNROLL
            for (int j = 0; j < kVecSize; ++j) {
                out_K[s][c][j] = (T)0.f;
                out_V[s][c][j] = (T)0.f;
            }
        }
    }

    blocks += cu_block_num[batch_idx];

    block::Head<T, KvQuant, BlockLayout> block_head{block_layout, layer_id, head_idx};

    Array<T, 2> param_K[ITER_S];
    Array<T, 2> param_V[ITER_S];

    int local_ti, local_ti_rank;

    // TurboQuant read path: read packed data from block cache, dequantize, inv-Hadamard
    if constexpr (Trait::kHadamardRotate) {
        const float sigma          = 1.f / sqrtf((float)HeadDim);
        const float had_scale      = 1.0f / sqrtf((float)HeadDim);
        const int   kShuffleStages = (HeadDim >= 256) ? 5 : (HeadDim >= 128) ? 4 : 3;

        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            const int si = offset.y + s * Map::kDeltaS + token_idx;
            local_ti     = cp_size.divmod(local_ti_rank, si);
            if (si < seq_len && local_ti_rank == cp_rank) {
                block_head.with((char**)blocks, local_ti, [&](auto k_cache, auto v_cache, T* k_param, T* v_param) {
                    // Read params
                    Ldg(param_K[s], k_param);
                    if constexpr (HAS_V) {
                        Ldg(param_V[s], v_param);
                    }
                    float mse_norm = (float)param_K[s][0];
                    float qjl_norm = (float)param_K[s][1];
                    float v_norm   = (float)param_V[s][0];

                    PRAGMA_UNROLL
                    for (int c = 0; c < ITER_C; ++c) {
                        int di = offset.x + c * Map::kDeltaC;
                        // Read K packed data
                        Ldg(vec_K[s][c], &k_cache[di]);
                        const uint32_t packed_k = reinterpret_cast<const uint32_t&>(vec_K[s][c]);
                        // Dequantize K: QJL4 codebook → rotated domain
                        PRAGMA_UNROLL
                        for (int j = 0; j < kVecSize; ++j) {
                            uint8_t nibble   = (packed_k >> (j * 4)) & 0xF;
                            int     idx3     = nibble & 0x7;
                            int     sign_bit = (nibble >> 3) & 0x1;
                            float   centroid = attention::dequantize_k4_mse(idx3, sigma);
                            float   sign_val = sign_bit * 2.f - 1.f;
                            out_K[s][c][j]   = (T)(mse_norm * (centroid + qjl_norm * sign_val));
                        }
                        // Read + dequantize V: 2-bit codebook
                        if constexpr (HAS_V) {
                            // 16× packing: read full uint32_t word, extract relevant 8-value half
                            Ldg(vec_V[s][c], &v_cache[di]);
                            const uint32_t packed_v_word = reinterpret_cast<const uint32_t&>(vec_V[s][c]);
                            const uint16_t packed_v =
                                (lane_id & 1) ? (uint16_t)(packed_v_word >> 16) : (uint16_t)(packed_v_word & 0xFFFF);
                            PRAGMA_UNROLL
                            for (int j = 0; j < kVecSize; ++j) {
                                int   idx2     = (packed_v >> (j * 2)) & 0x3;
                                float centroid = attention::dequantize_v2(idx2, sigma);
                                out_V[s][c][j] = (T)(v_norm * centroid);
                            }
                        }
                    }
                });
            }
        }
        // Inverse Hadamard on dequantized K/V
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                // K
                Array<float, kVecSize> fval_k;
                PRAGMA_UNROLL
                for (int j = 0; j < kVecSize; ++j) {
                    fval_k[j] = (float)out_K[s][c][j];
                }
                PRAGMA_UNROLL
                for (int stg = 0; stg < 3; ++stg) {
                    const int stride = 1 << stg;
                    PRAGMA_UNROLL
                    for (int j = 0; j < kVecSize; ++j) {
                        if (j & stride) {
                            float a = fval_k[j - stride], b = fval_k[j];
                            fval_k[j - stride] = a + b;
                            fval_k[j]          = a - b;
                        }
                    }
                }
                PRAGMA_UNROLL
                for (int stg = 0; stg < kShuffleStages; ++stg) {
                    const int   mask = 1 << stg;
                    const float sign = (lane_id & mask) ? -1.f : 1.f;
                    PRAGMA_UNROLL
                    for (int j = 0; j < kVecSize; ++j) {
                        float other = __shfl_xor_sync(uint32_t(-1), fval_k[j], mask);
                        fval_k[j]   = sign * fval_k[j] + other;
                    }
                }
                PRAGMA_UNROLL
                for (int j = 0; j < kVecSize; ++j) {
                    out_K[s][c][j] = (T)(fval_k[j] * had_scale);
                }
                // V
                if constexpr (HAS_V) {
                    Array<float, kVecSize> fval_v;
                    PRAGMA_UNROLL
                    for (int j = 0; j < kVecSize; ++j) {
                        fval_v[j] = (float)out_V[s][c][j];
                    }
                    PRAGMA_UNROLL
                    for (int stg = 0; stg < 3; ++stg) {
                        const int stride = 1 << stg;
                        PRAGMA_UNROLL
                        for (int j = 0; j < kVecSize; ++j) {
                            if (j & stride) {
                                float a = fval_v[j - stride], b = fval_v[j];
                                fval_v[j - stride] = a + b;
                                fval_v[j]          = a - b;
                            }
                        }
                    }
                    PRAGMA_UNROLL
                    for (int stg = 0; stg < kShuffleStages; ++stg) {
                        const int   mask = 1 << stg;
                        const float sign = (lane_id & mask) ? -1.f : 1.f;
                        PRAGMA_UNROLL
                        for (int j = 0; j < kVecSize; ++j) {
                            float other = __shfl_xor_sync(uint32_t(-1), fval_v[j], mask);
                            fval_v[j]   = sign * fval_v[j] + other;
                        }
                    }
                    PRAGMA_UNROLL
                    for (int j = 0; j < kVecSize; ++j) {
                        out_V[s][c][j] = (T)(fval_v[j] * had_scale);
                    }
                }
            }
        }
    }
    else {
        // Standard read + affine dequantization (original path)
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            const int si = offset.y + s * Map::kDeltaS + token_idx;
            local_ti     = cp_size.divmod(local_ti_rank, si);
            if (si < seq_len && local_ti_rank == cp_rank) {
                block_head.with((char**)blocks, local_ti, [&](auto k_cache, auto v_cache, T* k_param, T* v_param) {
                    PRAGMA_UNROLL
                    for (int c = 0; c < ITER_C; ++c) {
                        int di = offset.x + c * Map::kDeltaC;
                        Ldg(vec_K[s][c], &k_cache[di]);
                        if constexpr (HAS_V) {
                            Ldg(vec_V[s][c], &v_cache[di]);
                        }
                    }
                    if constexpr (kQuantKV) {
                        Ldg(param_K[s], k_param);
                        if constexpr (HAS_V) {
                            Ldg(param_V[s], v_param);
                        }
                    }
                });
            }
        }
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            ConvertKvCache<typename Trait::StorageK, T> conv_K{param_K[s][0], param_K[s][1]};
            ConvertKvCache<typename Trait::StorageV, T> conv_V{param_V[s][0], param_V[s][1]};
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                out_K[s][c] = conv_K(vec_K[s][c]);
                if constexpr (HAS_V) {
                    out_V[s][c] = conv_V(vec_V[s][c]);
                }
            }
        }
    }

    if (rope_param.type != RopeType::kNull) {
        FastRoPE rope(rope_param, batch_idx, std::integral_constant<int, kVecSize>{});
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            rope.init(di);
            PRAGMA_UNROLL
            for (int s = 0; s < ITER_S; ++s) {
                const int ti = offset.y + s * Map::kDeltaS + token_idx;  // sequence local
                rope.apply(out_K[s][c], ti);
            }
        }
    }

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int si = offset.y + s * Map::kDeltaS + token_idx;
            const int di = offset.x + c * Map::kDeltaC;
            local_ti     = cp_size.divmod(local_ti_rank, si);
            if (si < seq_len && local_ti_rank == cp_rank) {
                const int64_t index =
                    (batch_idx * stride_b + ti_beg * stride_c + local_ti * stride_s + head_idx * stride_h) * HeadDim
                    + di;
                Store(&k[index], out_K[s][c]);
                if constexpr (HAS_V) {
                    Store(&v[index], out_V[s][c]);
                }
            }
        }
    }
}

template<class T>
void invokeFlattenKV_v2(T*                     k,
                        T*                     v,
                        char**                 blocks,
                        const int*             cu_k_len,
                        const int*             cu_block_num,
                        const RopeKernelParam& rope_param,
                        int64_t                stride_b,
                        int64_t                stride_c,
                        int64_t                stride_h,
                        int64_t                stride_s,
                        int                    block_seq_len,
                        int                    layer_id,
                        int                    cp_rank,
                        FastDivmod             cp_size,
                        int                    max_seq_len,
                        int                    head_num,
                        int                    head_dim,
                        int                    batch_size,
                        int                    quant_policy,
                        cudaStream_t           stream)
{

    auto invoke = [&](auto kv_quant, const auto dim) {
        using KvQuantT = decltype(kv_quant);
        using TK       = typename attention::KvQuantTrait<KvQuantT, T>::StorageK;
        using TV       = typename attention::KvQuantTrait<KvQuantT, T>::StorageV;

        constexpr int  kHeadDim = dim;
        constexpr bool kShareKV = kHeadDim == 576;

        constexpr int kWarpCnt = 4;
        constexpr int CTA_S    = kShareKV ? 32 : 64;

        constexpr int block = kWarpCnt * WARP_SIZE;
        const dim3    grid((max_seq_len + CTA_S - 1) / CTA_S, head_num, batch_size);

        TM_CHECK_EQ(head_dim, kHeadDim);

        block::Layout block_layout{block::Config<T, KvQuantT, kHeadDim, kShareKV>{head_num, block_seq_len}};

        flattenKV_v2<CTA_S, kHeadDim, kWarpCnt, T, KvQuantT><<<grid, block, 0, stream>>>(k,
                                                                                         v,
                                                                                         blocks,
                                                                                         cu_k_len,
                                                                                         cu_block_num,
                                                                                         rope_param,
                                                                                         stride_b,
                                                                                         stride_c,
                                                                                         stride_h,
                                                                                         stride_s,
                                                                                         layer_id,
                                                                                         cp_rank,
                                                                                         cp_size,
                                                                                         block_layout);
    };

    auto dispatch = [&](auto kv_quant) {
        if (0) {}
        else if (head_dim == 64) {
            return invoke(kv_quant, std::integral_constant<int, 64>{});
        }
        else if (head_dim == 128) {
            return invoke(kv_quant, std::integral_constant<int, 128>{});
        }
        else if (head_dim == 192) {
            return invoke(kv_quant, std::integral_constant<int, 192>{});
        }
        else if (head_dim == 256) {
            return invoke(kv_quant, std::integral_constant<int, 256>{});
        }
        else if (head_dim == 576) {
            return invoke(kv_quant, std::integral_constant<int, 576>{});
        }
        TM_UNREACHABLE;
    };

    if (quant_policy == 42) {
        dispatch(attention::KvQuantTurbo{});
    }
    else if (quant_policy & QuantPolicy::kCacheKVInt8) {
        dispatch(attention::KvQuantInt8{});
    }
    else if (quant_policy & QuantPolicy::kCacheKVInt4) {
        dispatch(attention::KvQuantInt4{});
    }
    else {
        dispatch(attention::KvQuantNone{});
    }

    TM_CUDA_CHECK(cudaGetLastError());
}

#define INSTANTIATE_invokeFlattenKV_v2(type)                                                                           \
    template void invokeFlattenKV_v2(type*                  k,                                                         \
                                     type*                  v,                                                         \
                                     char**                 blocks,                                                    \
                                     const int*             cu_k_len,                                                  \
                                     const int*             cu_block_num,                                              \
                                     const RopeKernelParam& rope_param,                                                \
                                     int64_t                stride_b,                                                  \
                                     int64_t                stride_c,                                                  \
                                     int64_t                stride_h,                                                  \
                                     int64_t                stride_s,                                                  \
                                     int                    block_seq_len,                                             \
                                     int                    layer_id,                                                  \
                                     int                    cp_rank,                                                   \
                                     FastDivmod             cp_size,                                                   \
                                     int                    max_seq_len,                                               \
                                     int                    head_num,                                                  \
                                     int                    head_dim,                                                  \
                                     int                    batch_size,                                                \
                                     int                    quant_policy,                                              \
                                     cudaStream_t           stream);

INSTANTIATE_invokeFlattenKV_v2(half);
#if ENABLE_BF16
INSTANTIATE_invokeFlattenKV_v2(nv_bfloat16);
#endif

}  // namespace turbomind
