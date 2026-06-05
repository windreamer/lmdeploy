// Copyright (c) OpenMMLab. All rights reserved.
//
// Standalone test for hadamard_register_butterfly (Fusion 1).
// Verifies that the register+shuffle butterfly produces the same result
// as the standalone hadamard_transform kernel.

#include "hadamard_kernel.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <thrust/universal_vector.h>

using namespace turbomind;

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef PRAGMA_UNROLL
#define PRAGMA_UNROLL _Pragma("unroll")
#endif

// Minimal Array<T, N> matching turbomind's definition
template<typename T, int N>
struct Array {
    T data[N];
    __host__ __device__ T&       operator[](int i) { return data[i]; }
    __host__ __device__ const T& operator[](int i) const { return data[i]; }
};

// Compile-time integer log2
template<int N>
struct Log2 {
    static constexpr int value = (N >= 256) ? 8 : (N >= 128) ? 7 : (N >= 64) ? 6 : (N >= 32) ? 5 : 4;
};

// Replicate the exact function from attention_universal.h
// Key: all butterfly math done in float (matches standalone kernel precision)
template<typename T, int kAccessC, int HeadDim>
__device__ void hadamard_register_butterfly(Array<T, kAccessC>& val)
{
    constexpr int kLocalStages   = (kAccessC >= 8) ? 3 : (kAccessC >= 4) ? 2 : 1;
    constexpr int kTotalStages   = Log2<HeadDim>::value;
    constexpr int kShuffleStages = kTotalStages - kLocalStages;

    // Convert to float for numerical accuracy (matches standalone kernel)
    Array<float, kAccessC> fval;
    PRAGMA_UNROLL
    for (int j = 0; j < kAccessC; ++j) {
        fval[j] = (float)val[j];
    }

    // Local butterfly stages (register-only, in float)
    PRAGMA_UNROLL
    for (int s = 0; s < kLocalStages; ++s) {
        const int stride = 1 << s;
        PRAGMA_UNROLL
        for (int j = 0; j < kAccessC; ++j) {
            if (j & stride) {
                float a = fval[j - stride];
                float b = fval[j];
                fval[j - stride] = a + b;
                fval[j]          = a - b;
            }
        }
    }

    // Shuffle butterfly stages — same pattern as hadamard_kernel.cu:
    //   sign = (lane_id & mask) ? -1 : +1
    //   x = sign * x + x_other
    const int lane_id = threadIdx.x % WARP_SIZE;
    PRAGMA_UNROLL
    for (int s = 0; s < kShuffleStages; ++s) {
        const int mask = 1 << s;
        const float sign = (lane_id & mask) ? -1.f : 1.f;
        PRAGMA_UNROLL
        for (int j = 0; j < kAccessC; ++j) {
            float other = __shfl_xor_sync(uint32_t(-1), fval[j], mask);
            fval[j] = sign * fval[j] + other;
        }
    }

    // Normalization + convert back to T
    const float scale = 1.0f / sqrtf((float)HeadDim);
    PRAGMA_UNROLL
    for (int j = 0; j < kAccessC; ++j) {
        val[j] = (T)(fval[j] * scale);
    }
}

// -----------------------------------------------------------------------
// Test kernel: simulates the attention kernel's Q load + Hadamard path
// -----------------------------------------------------------------------

// ThreadMapQ parameters for head_dim=128, CTA_H=8
// kWarpThreadC = 16, kAccessC = 8, kIterC = 1, kIterS = 1
template<typename T, int kAccessC, int HeadDim, int kWarpThreadC>
__global__ void test_register_butterfly_kernel(const T* input, T* output, int num_heads)
{
    // Simulate ThreadMapQ layout: C_part = lane_id % kWarpThreadC
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int C_part  = lane_id % kWarpThreadC;
    const int S_part  = lane_id / kWarpThreadC;
    const int warp_id = threadIdx.x / WARP_SIZE;

    // Each thread holds kAccessC elements at offset C_part * kAccessC
    const int di = C_part * kAccessC;

    // Process one head per warp (simplified from actual kernel)
    const int head_idx = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id;
    if (head_idx >= num_heads) return;

    Array<T, kAccessC> val{};
    // Load from gmem
    const T* head_ptr = input + head_idx * HeadDim;
    PRAGMA_UNROLL
    for (int j = 0; j < kAccessC; ++j) {
        if (di + j < HeadDim) {
            val[j] = head_ptr[di + j];
        }
    }

    // Apply Hadamard
    hadamard_register_butterfly<T, kAccessC, HeadDim>(val);

    // Store to gmem
    T* out_ptr = output + head_idx * HeadDim;
    PRAGMA_UNROLL
    for (int j = 0; j < kAccessC; ++j) {
        if (di + j < HeadDim) {
            out_ptr[di + j] = val[j];
        }
    }
}

// -----------------------------------------------------------------------
// Reference: CPU brute-force Hadamard
// -----------------------------------------------------------------------
void hadamard_reference(const float* input, float* output, int dim)
{
    const float scale = 1.0f / std::sqrt((float)dim);
    for (int i = 0; i < dim; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < dim; ++j) {
            int parity = __builtin_popcount(i & j) & 1;
            sum += (parity == 0 ? 1.0f : -1.0f) * input[j];
        }
        output[i] = sum * scale;
    }
}

// -----------------------------------------------------------------------
// Test: compare register butterfly vs standalone kernel vs CPU reference
// -----------------------------------------------------------------------
template<typename T, int HeadDim>
void test_register_butterfly(int num_heads = 8)
{
    constexpr int kAccessC      = 8;
    constexpr int kWarpThreadC = HeadDim / kAccessC;
    constexpr int kWarpCnt     = 4;

    std::cout << "Testing hadamard_register_butterfly dim=" << HeadDim
              << " type=" << (sizeof(T) == 4 ? "fp32" : (sizeof(T) == 2 ? "fp16" : "bf16")) << std::endl;

    std::mt19937                          rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    float max_err_vs_ref    = 0;
    float max_err_vs_kernel = 0;
    int   total_outliers    = 0;

    for (int t = 0; t < 4; ++t) {
        const int n = num_heads * HeadDim;

        std::vector<float> host_input(n);
        for (int i = 0; i < n; ++i) host_input[i] = dist(rng);

        thrust::universal_vector<T> d_input(n);
        thrust::universal_vector<T> d_butterfly_out(n);
        for (int i = 0; i < n; ++i) d_input[i] = T(host_input[i]);

        dim3 block(kWarpCnt * WARP_SIZE);
        dim3 grid((num_heads + kWarpCnt - 1) / kWarpCnt);
        test_register_butterfly_kernel<T, kAccessC, HeadDim, kWarpThreadC>
            <<<grid, block>>>(thrust::raw_pointer_cast(d_input.data()),
                              thrust::raw_pointer_cast(d_butterfly_out.data()),
                              num_heads);

        thrust::universal_vector<T> d_kernel_out(n);
        HadamardParams hp{};
        hp.batch   = num_heads;
        hp.dim     = HeadDim;
        hp.log_dim = Log2<HeadDim>::value;
        hp.stride  = HeadDim;
        hp.scale   = 1.0f / std::sqrt((float)HeadDim);
        hp.x_ptr   = thrust::raw_pointer_cast(d_input.data());
        hp.out_ptr = thrust::raw_pointer_cast(d_kernel_out.data());
        hadamard_transform<T>(hp, 0);

        cudaDeviceSynchronize();

        std::vector<T> butterfly_out(n);
        std::vector<T> kernel_out(n);
        thrust::copy(d_butterfly_out.begin(), d_butterfly_out.end(), butterfly_out.begin());
        thrust::copy(d_kernel_out.begin(), d_kernel_out.end(), kernel_out.begin());

        for (int h = 0; h < num_heads; ++h) {
            std::vector<float> ref_output(HeadDim);
            hadamard_reference(host_input.data() + h * HeadDim, ref_output.data(), HeadDim);

            for (int i = 0; i < HeadDim; ++i) {
                float bf_val  = float(butterfly_out[h * HeadDim + i]);
                float kn_val  = float(kernel_out[h * HeadDim + i]);
                float ref_val = ref_output[i];

                max_err_vs_ref    = std::max(max_err_vs_ref, std::abs(bf_val - ref_val));
                max_err_vs_kernel = std::max(max_err_vs_kernel, std::abs(bf_val - kn_val));
                if (std::abs(bf_val - kn_val) > 1e-5f) ++total_outliers;
            }
        }
    }

    std::cout << "  register butterfly vs CPU ref:  max_err=" << max_err_vs_ref << std::endl;
    std::cout << "  register butterfly vs kernel:   max_err=" << max_err_vs_kernel
              << " outliers=" << total_outliers << std::endl;

    if (max_err_vs_ref > 1e-2f) { std::cerr << "FAIL" << std::endl; std::exit(1); }
    if (total_outliers > 0) { std::cerr << "FAIL" << std::endl; std::exit(1); }
    std::cout << "  PASS" << std::endl;
}

int main()
{
    test_register_butterfly<half, 64>(8);
    test_register_butterfly<half, 128>(8);
    test_register_butterfly<half, 256>(4);

    test_register_butterfly<float, 64>(8);
    test_register_butterfly<float, 128>(8);

    std::cout << "All hadamard_register_butterfly tests passed!" << std::endl;
    return 0;
}