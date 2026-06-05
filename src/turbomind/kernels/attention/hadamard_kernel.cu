// Copyright (c) OpenMMLab. All rights reserved.
//
// This file contains code derived from fast-hadamard-transform by Tri Dao.
// Original copyright notice:
//
// ******************************************************************************
// * Copyright (c) 2023, Tri Dao.
// ******************************************************************************
//
// Algorithm: radix-2 butterfly decomposition of Sylvester Hadamard matrix.
//   H[i,j] = (-1)^{popcount(i & j)}
//   Q = H / sqrt(d) is orthogonal, so Q^{-1} = Q^T = Q (self-inverse).
//   Forward and inverse use the same kernel with scale = 1/sqrt(d).

#include "hadamard_kernel.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cassert>

namespace turbomind {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Type helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int BYTES>
struct BytesToType {
};

template<>
struct BytesToType<16> {
    using Type = uint4;
    static_assert(sizeof(Type) == 16);
};
template<>
struct BytesToType<8> {
    using Type = uint64_t;
    static_assert(sizeof(Type) == 8);
};
template<>
struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4);
};
template<>
struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Compile-time log2
////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int cilog2(int val)
{
    return val > 0 ? 1 + cilog2(val >> 1) : -1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Warp all-reduce via XOR shuffle
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(T const& x, T const& y)
    {
        return x + y;
    }
};

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator& op)
    {
        constexpr int OFFSET = THREADS / 2;
        x                    = op(x, __shfl_xor_sync(0xffffffff, x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

template<>
struct Allreduce<2> {
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator& op)
    {
        x = op(x, __shfl_xor_sync(0xffffffff, x, 1));
        return x;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Butterfly primitives
////////////////////////////////////////////////////////////////////////////////////////////////////

// Thread-level butterfly on kNElts values per chunk, kNChunks chunks.
// Each thread holds [kNChunks][kNElts] values and does kLogN stages of
// pairwise (a+b, a-b) at stride 1,2,4,...,2^(kLogN-1).
template<int kLogN, int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread(float x[kNChunks][1 << kLogN])
{
    constexpr int N = 1 << kLogN;
#pragma unroll
    for (int i = 0; i < kLogN; ++i) {
        const int stride = 1 << i;
#pragma unroll
        for (int j = 0; j < N / 2; ++j) {
            const int lo  = j & (stride - 1);
            const int idx = (j - lo) * 2 + lo;
#pragma unroll
            for (int c = 0; c < kNChunks; ++c) {
                const float a      = x[c][idx];
                const float b      = x[c][idx + stride];
                x[c][idx]          = a + b;
                x[c][idx + stride] = a - b;
            }
        }
    }
}

// Warp-level butterfly using __shfl_xor_sync.
// step ranges from kStepStart to kLogWarpSize-1.
// At each step, lanes at distance 2^step exchange and combine.
template<int kLogWarpSize, int kStepStart, int kNChunks, int kNItems>
__device__ __forceinline__ void hadamard_mult_warp(float x[kNChunks][kNItems])
{
    constexpr int N       = 1 << kLogWarpSize;
    int           lane_id = threadIdx.x % N;
#pragma unroll
    for (int step = kStepStart; step < kLogWarpSize; ++step) {
        const int   lane_mask = 1 << step;
        const float sign      = (lane_id & lane_mask) ? -1.f : 1.f;
#pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float x_val_other = __shfl_xor_sync(0xffffffff, x[c][i], lane_mask);
                x[c][i]           = sign * x[c][i] + x_val_other;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Shared memory exchange for cross-warp butterfly
////////////////////////////////////////////////////////////////////////////////////////////////////

// Pre=true: exchange before warp butterfly (transpose warp layout)
// Pre=false: exchange after warp butterfly (transpose back)
template<int kNChunks, int kChunksPerExchange, int kNElts, int kWarpSize, int kNWarps, bool Pre, typename vec_t>
__device__ __forceinline__ void exchange_smem(float x_vals[kNChunks][kNElts], vec_t* smem)
{
    constexpr int kNThreads        = kWarpSize * kNWarps;
    constexpr int kNExchangePerVec = kNElts / (sizeof(vec_t) / sizeof(float));
    const int     warp_id          = threadIdx.x / kWarpSize;
    const int     lane_id          = threadIdx.x % kWarpSize;
    const int     row_t            = threadIdx.x % kNWarps;
    const int     col_t            = threadIdx.x / kNWarps;

#pragma unroll
    for (int c0 = 0; c0 < kNChunks / kChunksPerExchange; ++c0) {
        __syncthreads();
#pragma unroll
        for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
#pragma unroll
            for (int r = 0; r < kNExchangePerVec; ++r) {
                smem[(c1 * kNExchangePerVec + r) * kNThreads
                     + (Pre ? warp_id * kWarpSize + (lane_id ^ warp_id) : row_t * kWarpSize + (col_t ^ row_t))] =
                    reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1])[r];
            }
        }
        __syncthreads();
#pragma unroll
        for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
#pragma unroll
            for (int r = 0; r < kNExchangePerVec; ++r) {
                reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1])[r] =
                    smem[(c1 * kNExchangePerVec + r) * kNThreads
                         + (Pre ? row_t * kWarpSize + (col_t ^ row_t) : warp_id * kWarpSize + (lane_id ^ warp_id))];
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Vectorized load/store
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kNChunks, int kNElts, typename input_t>
__device__ __forceinline__ void load_input(input_t* x, float x_vals[kNChunks][kNElts], int dim)
{
    using vec_t                           = typename BytesToType<sizeof(input_t) * kNElts>::Type;
    input_t x_vals_load[kNChunks][kNElts] = {};
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        if ((c * blockDim.x + threadIdx.x) * kNElts < dim) {
            reinterpret_cast<vec_t*>(x_vals_load)[c] = reinterpret_cast<const vec_t*>(x)[c * blockDim.x + threadIdx.x];
        }
    }
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            x_vals[c][i] = float(x_vals_load[c][i]);
        }
    }
}

template<int kNChunks, int kNElts, typename output_t>
__device__ __forceinline__ void store_output(output_t* out, float out_vals[kNChunks][kNElts], int dim, float scale)
{
    using vec_t = typename BytesToType<sizeof(output_t) * kNElts>::Type;
    output_t out_vals_store[kNChunks][kNElts];
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            out_vals_store[c][i] = output_t(out_vals[c][i] * scale);
        }
    }
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        if ((c * blockDim.x + threadIdx.x) * kNElts < dim) {
            reinterpret_cast<vec_t*>(out)[c * blockDim.x + threadIdx.x] =
                reinterpret_cast<const vec_t*>(out_vals_store)[c];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel traits
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kNThreads_, int kLogN_, typename input_t_>
struct HadamardKernelTraits {
    using input_t                  = input_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kLogN     = kLogN_;
    static constexpr int N         = 1 << kLogN;
    static constexpr int kNBytes   = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts   = kNBytes == 4 ? 4 : 8;  // fp32: 4, fp16/bf16: 8
    static constexpr int kNChunks = N / (kNElts * kNThreads);

    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;

    static constexpr int kNExchangePerVec  = sizeof(float) / sizeof(input_t);
    static constexpr int kSmemExchangeSize = std::min(N * 4, 32 * 1024);
    static constexpr int kNExchangeRounds  = N * 4 / kSmemExchangeSize;
    static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
    static constexpr int kSmemSize = kSmemExchangeSize;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Main kernel
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads) void hadamard_transform_kernel(HadamardParams params)
{
    constexpr int kNThreads        = Ktraits::kNThreads;
    constexpr int kNElts           = Ktraits::kNElts;
    constexpr int kNExchangePerVec = Ktraits::kNExchangePerVec;
    constexpr int kNChunks         = Ktraits::kNChunks;
    using input_t                  = typename Ktraits::input_t;
    using vec_t                    = typename Ktraits::vec_t;

    constexpr int kLogNElts = cilog2(kNElts);
    static_assert((1 << kLogNElts) == kNElts);
    constexpr int kWarpSize    = std::min(kNThreads, 32);
    constexpr int kLogWarpSize = cilog2(kWarpSize);
    static_assert((1 << kLogWarpSize) == kWarpSize);
    constexpr int kNWarps    = kNThreads / kWarpSize;
    constexpr int kLogNWarps = cilog2(kNWarps);
    static_assert((1 << kLogNWarps) == kNWarps || kNWarps == 1);

    constexpr int kChunksPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNExchangePerVec * kNThreads);
    static_assert(kChunksPerExchange * sizeof(vec_t) * kNExchangePerVec * kNThreads == Ktraits::kSmemExchangeSize);

    extern __shared__ char smem_[];
    vec_t*                 smem_exchange = reinterpret_cast<vec_t*>(smem_);

    const int batch_id = blockIdx.x;
    input_t*  x        = reinterpret_cast<input_t*>(params.x_ptr) + batch_id * params.stride;
    input_t*  out      = reinterpret_cast<input_t*>(params.out_ptr) + batch_id * params.stride;

    float x_vals[kNChunks][kNElts];
    load_input<kNChunks, kNElts>(x, x_vals, params.dim);

    // Stage 1: thread-level butterfly on kNElts values
    hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);

    // Stage 2: warp-level butterfly using shuffle
    hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(x_vals);

    // Stage 3: cross-warp exchange via shared memory + warp butterfly
    if constexpr (kNWarps > 1) {
        exchange_smem<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, true, vec_t>(x_vals, smem_exchange);
        hadamard_mult_warp<kLogNWarps, 0, kNChunks, kNElts>(x_vals);
        exchange_smem<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, false, vec_t>(x_vals, smem_exchange);
    }

    // Stage 4: cross-chunk butterfly (needed when kNChunks > 1, e.g. fp32 with small dim)
    if constexpr (kNChunks > 1) {
        float x_vals_transposed[kNElts][kNChunks];
#pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                x_vals_transposed[i][c] = x_vals[c][i];
            }
        }
        constexpr int kLogNChunks = cilog2(kNChunks);
        static_assert((1 << kLogNChunks) == kNChunks, "kNChunks must be a power of 2");
        hadamard_mult_thread<kLogNChunks, kNElts>(x_vals_transposed);
#pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                x_vals[c][i] = x_vals_transposed[i][c];
            }
        }
    }

    store_output<kNChunks, kNElts>(out, x_vals, params.dim, params.scale);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Launch helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kNThreads, int kLogN, typename input_t>
void hadamard_transform_launch(HadamardParams& params, cudaStream_t stream)
{
    using Ktraits           = HadamardKernelTraits<kNThreads, kLogN, input_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    auto          kernel    = &hadamard_transform_kernel<Ktraits>;
    if constexpr (kSmemSize >= 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
    }
    kernel<<<params.batch, Ktraits::kNThreads, kSmemSize, stream>>>(params);
}

template<typename input_t>
void hadamard_transform_dispatch(HadamardParams& params, cudaStream_t stream)
{
    // Dispatch by log_dim (power-of-2 dimension)
    // Thread count scales with dimension to keep registers and shared memory bounded.
    switch (params.log_dim) {
        case 3:
            hadamard_transform_launch<1, 3, input_t>(params, stream);
            break;  // dim=8
        case 4:
            hadamard_transform_launch<2, 4, input_t>(params, stream);
            break;  // dim=16
        case 5:
            hadamard_transform_launch<4, 5, input_t>(params, stream);
            break;  // dim=32
        case 6:
            hadamard_transform_launch<8, 6, input_t>(params, stream);
            break;  // dim=64
        case 7:
            hadamard_transform_launch<16, 7, input_t>(params, stream);
            break;  // dim=128
        case 8:
            hadamard_transform_launch<32, 8, input_t>(params, stream);
            break;  // dim=256
        case 9:
            hadamard_transform_launch<32, 9, input_t>(params, stream);
            break;  // dim=512
        case 10:
            hadamard_transform_launch<128, 10, input_t>(params, stream);
            break;  // dim=1024
        case 11:
            hadamard_transform_launch<256, 11, input_t>(params, stream);
            break;  // dim=2048
        case 12:
            hadamard_transform_launch<256, 12, input_t>(params, stream);
            break;  // dim=4096
        default:
            assert(false && "Unsupported Hadamard dimension: must be in [8, 4096] and power of 2");
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Public API
////////////////////////////////////////////////////////////////////////////////////////////////////

void hadamard_transform_fp16(HadamardParams& params, cudaStream_t stream)
{
    hadamard_transform_dispatch<half>(params, stream);
}

void hadamard_transform_bf16(HadamardParams& params, cudaStream_t stream)
{
    hadamard_transform_dispatch<nv_bfloat16>(params, stream);
}

void hadamard_transform_fp32(HadamardParams& params, cudaStream_t stream)
{
    hadamard_transform_dispatch<float>(params, stream);
}

}  // namespace turbomind
