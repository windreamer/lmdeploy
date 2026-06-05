// Copyright (c) OpenMMLab. All rights reserved.

#include "cutlass/fast_math.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/thread_map.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <type_traits>

namespace turbomind::attention {

// ---------------------------------------------------------------------------
// Fused Hadamard butterfly on shared memory
// ---------------------------------------------------------------------------
// Applies orthogonal Hadamard transform to s_out[0][0..HeadDim-1] in-place.
// H is symmetric and Q = H/sqrt(d) is orthogonal, so Q^{-1} = Q (same transform).
//
// Butterfly: log2(HeadDim) stages, each reads pairs (i, i^stride), writes (a+b, a-b).
// Final normalization: multiply by 1/sqrt(d).
// ---------------------------------------------------------------------------

template<int HeadDim, int WarpCnt>
__device__ void hadamard_smem_butterfly(float* s_out_row)
{
    constexpr int kLogDim = static_cast<int>(log2(HeadDim));
    const float kScale = 1.0f / sqrtf((float)HeadDim);
    constexpr int kThreads = WarpCnt * WARP_SIZE;

    const int tid = threadIdx.x;

    PRAGMA_UNROLL
    for (int s = 0; s < kLogDim; ++s) {
        const int stride = 1 << s;
        // Each thread processes a contiguous slice of indices
        constexpr int kPerThread = (HeadDim + kThreads - 1) / kThreads;
        PRAGMA_UNROLL
        for (int p = 0; p < kPerThread; ++p) {
            int i = tid + p * kThreads;
            if (i < HeadDim && !(i & stride)) {
                int j = i ^ stride;
                float a = s_out_row[i];
                float b = s_out_row[j];
                s_out_row[i] = a + b;
                s_out_row[j] = a - b;
            }
        }
        __syncthreads();
    }

    // Apply normalization
    constexpr int kPerThreadN = (HeadDim + kThreads - 1) / kThreads;
    PRAGMA_UNROLL
    for (int p = 0; p < kPerThreadN; ++p) {
        int i = tid + p * kThreads;
        if (i < HeadDim) {
            s_out_row[i] *= kScale;
        }
    }
    __syncthreads();
}

template<int CP, int CTA_K, int HeadDim, int WarpCnt, bool First, bool ApplyHadamard, class T>
__global__ void reduce(T*         out,
                       float*     partial_ML,
                       float*     partial_O,
                       const int* split_cnt_,
                       int        max_split_cnt,
                       int        query_num,
                       int        head_num,
                       float      exp_scale,
                       int        cp_rank,
                       int        stride_k,
                       int        offset_k)
{
    __shared__ float s_out[WarpCnt][HeadDim];
    __shared__ float s_ML[WarpCnt][2];
    __shared__ float s_scale[CTA_K];

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int head_idx  = ReduceCtaMap::head_idx();
    const int query_idx = ReduceCtaMap::query_idx();
    const int chunk_idx = ReduceCtaMap::split_idx();

    offset_k *= chunk_idx;
    const int split_cnt = (split_cnt_ != nullptr) ? split_cnt_[query_idx] : 1;
    if (offset_k >= split_cnt) {  // out of bound
        return;
    }

    // merge cp and k for the first time and merge k thereafter.
    constexpr int kCpUb     = First ? CP : 1;
    constexpr int kWarpIter = First ? (CP + WarpCnt - 1) / WarpCnt : 1;
    float         ML[kWarpIter][2];

    // frag_M of this cp_rank and lane
    float frag_M = -std::numeric_limits<float>::infinity();

    const int offset_r = cp_rank * query_num * head_num * max_split_cnt * 2;
    const int offset_m = First ? 0 : offset_r;
    const int warp_m   = First ? cp_rank % WarpCnt : 0;

    PRAGMA_UNROLL
    for (int i = 0; i < kWarpIter; ++i) {
        int        cp_i = warp_id + i * WarpCnt;
        int        ki   = lane_id * stride_k + offset_k;
        const bool mask = cp_i < kCpUb && ki < split_cnt;  // cp, q, h, k, 2
        const int  index =
            offset_m + ((cp_i * query_num * head_num + (query_idx * head_num + head_idx)) * max_split_cnt + ki) * 2;

        Array<float, 2> temp_ML = {-std::numeric_limits<float>::infinity(), 0.f};
        if (mask) {
            Load(temp_ML, &partial_ML[index]);
        }
        Store(&ML[i][0], temp_ML);

        frag_M = (mask && warp_m == warp_id) ? ML[i][0] : frag_M;
    }

    float block_M = -std::numeric_limits<float>::infinity();
    float block_L = 0.f;
    PRAGMA_UNROLL
    for (int i = 0; i < kWarpIter; ++i) {
        block_M = fmaxf(block_M, ML[i][0]);
    }

    PRAGMA_UNROLL
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        block_M = fmaxf(block_M, __shfl_xor_sync(uint32_t(-1), block_M, mask));
    }

    PRAGMA_UNROLL
    for (int i = 0; i < kWarpIter; ++i) {
        block_L += (ML[i][0] == -std::numeric_limits<float>::infinity()) ?
                       0.0f :
                       exp2f((ML[i][0] - block_M) * exp_scale) * ML[i][1];
    }

    PRAGMA_UNROLL
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        block_L += __shfl_xor_sync(uint32_t(-1), block_L, mask);
    }

    if constexpr (First && CP > 1) {
        if (lane_id == 0) {
            Store(&s_ML[warp_id][0], Array<float, 2>{block_M, block_L});
        }
        __syncthreads();

        if (warp_id == 0 && lane_id == 0) {
            PRAGMA_UNROLL
            for (int i = 0; i < WarpCnt; ++i) {
                block_M = fmaxf(block_M, s_ML[i][0]);
            }

            block_L = 0.f;
            PRAGMA_UNROLL
            for (int i = 0; i < WarpCnt; ++i) {
                block_L += exp2f((s_ML[i][0] - block_M) * exp_scale) * s_ML[i][1];
            }

            Store(&s_ML[0][0], Array<float, 2>{block_M, block_L});
        }
        __syncthreads();

        block_M = s_ML[0][0];
        block_L = s_ML[0][1];
    }

    if (gridDim.z > 1 && warp_id == 0) {
        int        ki    = lane_id * stride_k + offset_k;
        const bool mask  = ki < split_cnt;  // q, h, k, 2
        const int  index = offset_r + ((query_idx * head_num + head_idx) * max_split_cnt + ki) * 2;
        if (mask) {
            Store(&partial_ML[index], Array<float, 2>{block_M, block_L});
        }
    }

    if (warp_id == warp_m) {
        const float divisor = gridDim.z == 1 ? block_L : 1.0f;
        s_scale[lane_id] =
            frag_M == -std::numeric_limits<float>::infinity() ? 0.0f : exp2f((frag_M - block_M) * exp_scale) / divisor;
    }

    __syncthreads();

    // HeadDim / WARP_SIZE
    // 128     -> 4
    // 64, 192 -> 2
    constexpr int kVecSize = HeadDim % 128 == 0 ? 4 : 2;

    using Map = RakedThreadMap<HeadDim, WarpCnt, kVecSize, WarpCnt, WARP_SIZE>;
    static_assert(Map::kIterS == 1);

    constexpr int C = Map::kIterC;

    using Vec = Array<float, kVecSize>;

    Vec accu_O[C]{};
    Vec frag_O[C];

    const int2 d = Map::get_offset(warp_id, lane_id);

    auto for_each = [&](auto fn) {
        const int ki = d.y;
        PRAGMA_UNROLL
        for (int c = 0; c < C; ++c) {
            const int di = d.x + c * Map::kDeltaC;
            fn(c, ki, di);
        }
    };

    PRAGMA_UNROLL
    for (int k = 0; k < CTA_K; k += WarpCnt) {
        for_each([&](int c, int ki, int di) {
            using namespace ops;
            ki += k;
            const int  split_idx = offset_k + stride_k * ki;
            const bool mask      = split_idx < split_cnt;
            const int  index     = (query_idx * head_num + head_idx) * max_split_cnt + split_idx;
            const int  offset    = index * HeadDim + di;
            if (mask) {
                Load(frag_O[c], &partial_O[offset]);
                accu_O[c] = accu_O[c] + frag_O[c] * s_scale[ki];
            }
        });
    }

    for_each([&](int c, int ki, int di) {
        Store(&s_out[ki][di], accu_O[c]);  //
    });

    PRAGMA_UNROLL
    for (int w = WarpCnt / 2; w > 0; w /= 2) {
        __syncthreads();
        for_each([&](int c, int ki, int di) {
            using namespace ops;
            if (ki < w) {
                (Vec&)s_out[ki][di] = (Vec&)s_out[ki][di] + (Vec&)s_out[w + ki][di];
            }
        });
    }

    for_each([&](int c, int ki, int di) {
        if (ki == 0) {
            if (gridDim.z == 1) {
                // Final output — apply Hadamard if TurboQuant before writing to gmem
                if constexpr (ApplyHadamard) {
                    // Only one thread needs to trigger the smem butterfly (all threads participate)
                    if (c == 0 && di == 0) {
                        hadamard_smem_butterfly<HeadDim, WarpCnt>(s_out[0]);
                    }
                    // No syncthreads needed here — hadamard_smem_butterfly ends with one
                    // Now read the Hadamard-transformed result from smem
                    const int offset = (query_idx * head_num + head_idx) * HeadDim + di;
                    Store(&out[offset], cast<T>((Vec&)s_out[ki][di]));
                }
                else {
                    const int offset = (query_idx * head_num + head_idx) * HeadDim + di;
                    Store(&out[offset], cast<T>((Vec&)s_out[ki][di]));
                }
            }
            else {
                const int offset = ((query_idx * head_num + head_idx) * max_split_cnt + offset_k) * HeadDim + di;
                Store(&partial_O[offset], (Vec&)s_out[ki][di]);
            }
        }
    });
}

template<int HeadDim, class T>
void invokeReduceV3(T*           out,
                    float*       partial_ML,
                    float*       partial_O,
                    const int*   split_cnt,
                    int          partial_len,
                    int          max_split_cnt,
                    int          cp_size,
                    int          cp_rank,
                    int          query_num,
                    int          head_num,
                    float        exp_scale,
                    bool         apply_hadamard,
                    cudaStream_t stream)
{
    constexpr int CTA_K = 32;  // warp size

    constexpr int    kWarpCnt  = 4;
    constexpr size_t kSmemSize = sizeof(float) * (kWarpCnt * HeadDim + kWarpCnt * 2 + CTA_K);
    static_assert(kSmemSize < (48 << 10), "shared memory usage exceeds 48KB per block");

    partial_ML -= cp_rank * query_num * head_num * partial_len * 2;  // begin address of cp_rank0

    auto invoke = [&](auto cp, auto is_first, auto do_hadamard, int stride_k) {
        const dim3 block = kWarpCnt * WARP_SIZE;
        const dim3 grid  = ReduceCtaMap::get_grid_shape(query_num, head_num, max_split_cnt, CTA_K);

        reduce<cp, CTA_K, HeadDim, kWarpCnt, is_first, do_hadamard><<<grid, block, kSmemSize, stream>>>(  //
            out,
            partial_ML,
            partial_O,
            split_cnt,
            partial_len,
            query_num,
            head_num,
            exp_scale,
            cp_rank,
            stride_k,
            stride_k * CTA_K);
    };

    auto dispatch_cp = [&](int stride_k, auto is_first) {
        // Dispatch on apply_hadamard (runtime bool → compile-time constant)
        auto dispatch_hadamard = [&](auto do_hadamard) {
            switch (cp_size) {
#define LAUNCH_INVOKE(n)                                                                                               \
    case n:                                                                                                            \
        invoke(std::integral_constant<int, n>{}, is_first, do_hadamard, stride_k);                                     \
        break;
                LAUNCH_INVOKE(1);
                LAUNCH_INVOKE(2);
                LAUNCH_INVOKE(4);
                LAUNCH_INVOKE(8);
                LAUNCH_INVOKE(16);
                LAUNCH_INVOKE(32);
                default:
                    TM_CHECK(false) << "reduce does not support cp_size = " << cp_size;
#undef LAUNCH_INVOKE
            }
        };
        if (apply_hadamard) {
            dispatch_hadamard(std::true_type{});
        }
        else {
            dispatch_hadamard(std::false_type{});
        }
    };

    int stride_k = 1;

    dispatch_cp(stride_k, std::true_type{});
    while (max_split_cnt > CTA_K) {
        max_split_cnt = (max_split_cnt + CTA_K - 1) / CTA_K;
        stride_k *= CTA_K;
        dispatch_cp(stride_k, std::false_type{});
    }

    TM_CUDA_CHECK(cudaGetLastError());
}

#define INSTANTIATE_invokeReduceV3(dim, type)                                                                          \
    template void invokeReduceV3<dim>(type * out,                                                                      \
                                      float*       partial_ML,                                                         \
                                      float*       partial_O,                                                          \
                                      const int*   split_cnt,                                                          \
                                      int          partial_len,                                                        \
                                      int          max_split_cnt,                                                      \
                                      int          cp_size,                                                            \
                                      int          cp_rank,                                                            \
                                      int          query_num,                                                          \
                                      int          head_num,                                                           \
                                      float        exp_scale,                                                          \
                                      bool         apply_hadamard,                                                     \
                                      cudaStream_t stream);

INSTANTIATE_invokeReduceV3(64, half);
INSTANTIATE_invokeReduceV3(128, half);
INSTANTIATE_invokeReduceV3(192, half);
INSTANTIATE_invokeReduceV3(256, half);
INSTANTIATE_invokeReduceV3(576, half);

#if ENABLE_BF16
INSTANTIATE_invokeReduceV3(64, nv_bfloat16);
INSTANTIATE_invokeReduceV3(128, nv_bfloat16);
INSTANTIATE_invokeReduceV3(192, nv_bfloat16);
INSTANTIATE_invokeReduceV3(256, nv_bfloat16);
INSTANTIATE_invokeReduceV3(576, nv_bfloat16);
#endif

}  // namespace turbomind::attention
