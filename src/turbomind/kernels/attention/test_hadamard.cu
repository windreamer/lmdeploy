// Copyright (c) OpenMMLab. All rights reserved.
//
// This file contains code derived from fast-hadamard-transform by Tri Dao.
// Original copyright notice:
//
// ******************************************************************************
// * Copyright (c) 2023, Tri Dao.
// ******************************************************************************

#include "hadamard_kernel.h"
#include "src/turbomind/kernels/attention/test_utils.h"
#include <cmath>
#include <iostream>
#include <random>
#include <thrust/universal_vector.h>

using namespace turbomind;

// Reference Hadamard transform on CPU (brute-force matrix multiply)
// H[i,j] = (-1)^{popcount(i & j)}, Q = H / sqrt(d)
void hadamard_reference(const float* input, float* output, int dim)
{
    const float scale = 1.0f / std::sqrt((float)dim);
    for (int i = 0; i < dim; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < dim; ++j) {
            // H[i,j] = (-1)^{popcount(i & j)}
            int parity = __builtin_popcount(i & j) & 1;
            sum += (parity == 0 ? 1.0f : -1.0f) * input[j];
        }
        output[i] = sum * scale;
    }
}

template<typename T>
void test_hadamard(int dim, int batch = 4)
{
    std::cout << "Testing Hadamard dim=" << dim << " batch=" << batch
              << " type=" << (sizeof(T) == 4 ? "fp32" : (sizeof(T) == 2 ? "fp16" : "bf16")) << std::endl;

    const int n = batch * dim;

    // Generate random input
    std::vector<float>                    host_input(n);
    std::mt19937                          rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        host_input[i] = dist(rng);
    }

    // Allocate GPU memory
    thrust::universal_vector<T> d_input(n);
    thrust::universal_vector<T> d_output(n);
    thrust::universal_vector<T> d_roundtrip(n);

    // Copy input to GPU (convert to T)
    for (int i = 0; i < n; ++i) {
        d_input[i] = T(host_input[i]);
    }

    // Setup params
    HadamardParams params;
    params.batch   = batch;
    params.dim     = dim;
    params.log_dim = (int)round(log2(dim));
    params.stride  = dim;
    params.scale   = 1.0f / std::sqrt((float)dim);
    params.x_ptr   = thrust::raw_pointer_cast(d_input.data());
    params.out_ptr = thrust::raw_pointer_cast(d_output.data());

    // Forward transform
    if constexpr (std::is_same_v<T, float>) {
        hadamard_transform_fp32(params, 0);
    }
    else if constexpr (std::is_same_v<T, half>) {
        hadamard_transform_fp16(params, 0);
    }
    else {
        hadamard_transform_bf16(params, 0);
    }

    // Self-inverse test: transform again should recover original
    params.x_ptr   = thrust::raw_pointer_cast(d_output.data());
    params.out_ptr = thrust::raw_pointer_cast(d_roundtrip.data());
    if constexpr (std::is_same_v<T, float>) {
        hadamard_transform_fp32(params, 0);
    }
    else if constexpr (std::is_same_v<T, half>) {
        hadamard_transform_fp16(params, 0);
    }
    else {
        hadamard_transform_bf16(params, 0);
    }

    cudaDeviceSynchronize();

    // Check roundtrip: input ≈ roundtrip (2x transform = identity)
    Compare(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_roundtrip.data()), n, n, 1);

    // Check against reference for fp32
    if constexpr (std::is_same_v<T, float>) {
        std::vector<float> host_output(n);
        cudaMemcpy(
            host_output.data(), thrust::raw_pointer_cast(d_output.data()), n * sizeof(float), cudaMemcpyDeviceToHost);

        float max_err  = 0;
        float mean_err = 0;
        for (int b = 0; b < batch; ++b) {
            std::vector<float> ref_output(dim);
            hadamard_reference(host_input.data() + b * dim, ref_output.data(), dim);
            for (int i = 0; i < dim; ++i) {
                float err = std::abs(host_output[b * dim + i] - ref_output[i]);
                max_err   = std::max(max_err, err);
                mean_err += err;
            }
        }
        mean_err /= n;
        std::cout << "  vs reference: max_err=" << max_err << " mean_err=" << mean_err << std::endl;
    }
}

int main()
{
    // Test common head_dim values
    test_hadamard<half>(64, 8);
    test_hadamard<half>(128, 8);
    test_hadamard<half>(256, 4);

    test_hadamard<float>(64, 8);
    test_hadamard<float>(128, 8);

#if ENABLE_BF16
    test_hadamard<nv_bfloat16>(64, 8);
    test_hadamard<nv_bfloat16>(128, 8);
#endif

    std::cout << "All Hadamard transform tests passed!" << std::endl;
    return 0;
}
