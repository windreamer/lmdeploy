// Copyright (c) OpenMMLab. All rights reserved.
//
// Standalone test for hadamard_smem_butterfly (reduce kernel fusion).
// Verifies that the smem butterfly produces the same result as the
// standalone hadamard_transform kernel.

#include "hadamard_kernel.h"
#include <cmath>
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

// -----------------------------------------------------------------------
// Device kernel: copy input to smem, apply butterfly, copy to output
// -----------------------------------------------------------------------

namespace test_detail {

// Compile-time integer log2
template<int N>
struct Log2 {
    static constexpr int value = (N >= 256) ? 8 : (N >= 128) ? 7 : (N >= 64) ? 6 : (N >= 32) ? 5 : (N >= 16) ? 4 : (N >= 8) ? 3 : 2;
};

template<int HeadDim, int WarpCnt>
__device__ void hadamard_smem_butterfly(float* s_out_row)
{
    constexpr int kLogDim = Log2<HeadDim>::value;
    const float   kScale  = 1.0f / sqrtf((float)HeadDim);
    constexpr int kThreads = WarpCnt * WARP_SIZE;

    const int tid = threadIdx.x;

    PRAGMA_UNROLL
    for (int s = 0; s < kLogDim; ++s) {
        const int stride = 1 << s;
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

template<int HeadDim, int WarpCnt>
__global__ void test_butterfly_kernel(const float* input, float* output)
{
    __shared__ float s_out[1][HeadDim];

    const int tid = threadIdx.x;
    constexpr int kThreads = WarpCnt * WARP_SIZE;

    // Load input to smem
    constexpr int kPerThread = (HeadDim + kThreads - 1) / kThreads;
    PRAGMA_UNROLL
    for (int p = 0; p < kPerThread; ++p) {
        int i = tid + p * kThreads;
        if (i < HeadDim) {
            s_out[0][i] = input[i];
        }
    }
    __syncthreads();

    // Apply butterfly
    hadamard_smem_butterfly<HeadDim, WarpCnt>(s_out[0]);

    // Read result from smem
    PRAGMA_UNROLL
    for (int p = 0; p < kPerThread; ++p) {
        int i = tid + p * kThreads;
        if (i < HeadDim) {
            output[i] = s_out[0][i];
        }
    }
}

}  // namespace test_detail

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
// Test: compare smem butterfly vs standalone kernel vs CPU reference
// -----------------------------------------------------------------------
void test_smem_butterfly(int head_dim, int num_tests = 8)
{
    constexpr int kWarpCnt = 4;  // matches reduce kernel

    std::cout << "Testing hadamard_smem_butterfly dim=" << head_dim << std::endl;

    std::mt19937                          rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    float max_err_butterfly_vs_ref    = 0;
    float max_err_butterfly_vs_kernel = 0;
    int   total_outliers              = 0;

    for (int t = 0; t < num_tests; ++t) {
        const int n = head_dim;

        // Random input
        std::vector<float> host_input(n);
        for (int i = 0; i < n; ++i) {
            host_input[i] = dist(rng);
        }

        // --- GPU: smem butterfly ---
        thrust::universal_vector<float> d_input(n);
        thrust::universal_vector<float> d_butterfly_out(n);
        thrust::copy(host_input.begin(), host_input.end(), d_input.begin());

        dim3 block(kWarpCnt * WARP_SIZE);
        dim3 grid(1);
        switch (head_dim) {
        case 64:
            test_detail::test_butterfly_kernel<64, kWarpCnt><<<grid, block>>>(
                thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_butterfly_out.data()));
            break;
        case 128:
            test_detail::test_butterfly_kernel<128, kWarpCnt><<<grid, block>>>(
                thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_butterfly_out.data()));
            break;
        case 256:
            test_detail::test_butterfly_kernel<256, kWarpCnt><<<grid, block>>>(
                thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_butterfly_out.data()));
            break;
        default:
            std::cerr << "Unsupported head_dim=" << head_dim << std::endl;
            return;
        }

        // --- GPU: standalone hadamard_transform ---
        thrust::universal_vector<float> d_kernel_out(n);
        HadamardParams hp{};
        hp.batch   = 1;
        hp.dim     = head_dim;
        hp.log_dim = static_cast<int>(round(log2(head_dim)));
        hp.stride  = head_dim;
        hp.scale   = 1.0f / std::sqrt((float)head_dim);
        hp.x_ptr   = thrust::raw_pointer_cast(d_input.data());
        hp.out_ptr = thrust::raw_pointer_cast(d_kernel_out.data());
        hadamard_transform_fp32(hp, 0);

        cudaDeviceSynchronize();

        // --- CPU reference ---
        std::vector<float> ref_output(n);
        hadamard_reference(host_input.data(), ref_output.data(), head_dim);

        // --- Compare butterfly vs reference ---
        std::vector<float> butterfly_out(n);
        thrust::copy(d_butterfly_out.begin(), d_butterfly_out.end(), butterfly_out.begin());

        std::vector<float> kernel_out(n);
        thrust::copy(d_kernel_out.begin(), d_kernel_out.end(), kernel_out.begin());

        for (int i = 0; i < n; ++i) {
            float err_ref = std::abs(butterfly_out[i] - ref_output[i]);
            max_err_butterfly_vs_ref = std::max(max_err_butterfly_vs_ref, err_ref);

            float err_kernel = std::abs(butterfly_out[i] - kernel_out[i]);
            max_err_butterfly_vs_kernel = std::max(max_err_butterfly_vs_kernel, err_kernel);

            // Butterfly vs kernel: should be bit-exact or very close
            if (err_kernel > 1e-5f) {
                ++total_outliers;
            }
        }
    }

    std::cout << "  butterfly vs CPU ref:  max_err=" << max_err_butterfly_vs_ref << std::endl;
    std::cout << "  butterfly vs kernel:   max_err=" << max_err_butterfly_vs_kernel
              << " outliers=" << total_outliers << std::endl;

    if (max_err_butterfly_vs_ref > 1e-3f) {
        std::cerr << "FAIL: butterfly vs reference error too large" << std::endl;
        std::exit(1);
    }
    if (total_outliers > 0) {
        std::cerr << "FAIL: butterfly vs standalone kernel mismatch" << std::endl;
        std::exit(1);
    }
    std::cout << "  PASS" << std::endl;
}

// -----------------------------------------------------------------------
// Self-inverse test: H(H(x)) = x
// -----------------------------------------------------------------------
void test_self_inverse(int head_dim)
{
    constexpr int kWarpCnt = 4;

    std::cout << "Testing self-inverse dim=" << head_dim << std::endl;

    const int n = head_dim;
    std::mt19937                          rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> host_input(n);
    for (int i = 0; i < n; ++i) {
        host_input[i] = dist(rng);
    }

    thrust::universal_vector<float> d_input(n);
    thrust::universal_vector<float> d_after1(n);
    thrust::universal_vector<float> d_after2(n);
    thrust::copy(host_input.begin(), host_input.end(), d_input.begin());

    // Apply butterfly once
    dim3 block(kWarpCnt * WARP_SIZE);
    dim3 grid(1);
    auto launch = [&](auto dim_constant, auto in, auto out) {
        test_detail::test_butterfly_kernel<decltype(dim_constant)::value, kWarpCnt>
            <<<grid, block>>>(in, out);
    };
    if (head_dim == 64) {
        launch(std::integral_constant<int, 64>{},
               thrust::raw_pointer_cast(d_input.data()),
               thrust::raw_pointer_cast(d_after1.data()));
        launch(std::integral_constant<int, 64>{},
               thrust::raw_pointer_cast(d_after1.data()),
               thrust::raw_pointer_cast(d_after2.data()));
    }
    else if (head_dim == 128) {
        launch(std::integral_constant<int, 128>{},
               thrust::raw_pointer_cast(d_input.data()),
               thrust::raw_pointer_cast(d_after1.data()));
        launch(std::integral_constant<int, 128>{},
               thrust::raw_pointer_cast(d_after1.data()),
               thrust::raw_pointer_cast(d_after2.data()));
    }
    else if (head_dim == 256) {
        launch(std::integral_constant<int, 256>{},
               thrust::raw_pointer_cast(d_input.data()),
               thrust::raw_pointer_cast(d_after1.data()));
        launch(std::integral_constant<int, 256>{},
               thrust::raw_pointer_cast(d_after1.data()),
               thrust::raw_pointer_cast(d_after2.data()));
    }

    cudaDeviceSynchronize();

    // After 2 transforms, should recover original
    std::vector<float> roundtrip(n);
    thrust::copy(d_after2.begin(), d_after2.end(), roundtrip.begin());

    float max_err = 0;
    for (int i = 0; i < n; ++i) {
        float err = std::abs(roundtrip[i] - host_input[i]);
        max_err = std::max(max_err, err);
    }

    std::cout << "  roundtrip max_err=" << max_err << std::endl;
    if (max_err > 1e-4f) {
        std::cerr << "FAIL: self-inverse test error too large" << std::endl;
        std::exit(1);
    }
    std::cout << "  PASS" << std::endl;
}

int main()
{
    // Test all supported head dims
    test_smem_butterfly(64);
    test_smem_butterfly(128);
    test_smem_butterfly(256);

    // Self-inverse: H(H(x)) = x (orthogonality)
    test_self_inverse(64);
    test_self_inverse(128);
    test_self_inverse(256);

    std::cout << "All hadamard_smem_butterfly tests passed!" << std::endl;
    return 0;
}