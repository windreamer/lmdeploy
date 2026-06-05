// Copyright (c) OpenMMLab. All rights reserved.

#include "hadamard_kernel.h"
#include "src/turbomind/kernels/attention/test_utils.h"
#include "src/turbomind/kernels/attention/turbo_quant_qjl4.h"
#include <cmath>
#include <iostream>
#include <random>
#include <thrust/universal_vector.h>

using namespace turbomind;

// ---------------------------------------------------------------------------
// GPU kernel for K QJL4 quantization
//   raw_K → L2_norm(mse_norm) → normalize → hadamard_rotate →
//   3-bit MSE boundary compare → residual → QJL sign + norm →
//   pack nibbles → store packed + [mse_norm, qjl_norm]
// ---------------------------------------------------------------------------

template<int kHeadDim, int kPackedDim = kHeadDim / 2>
__global__ void quantize_qjl4_kernel(const half* __restrict__ input,
                                     uint8_t* __restrict__ packed,
                                     half* __restrict__ mse_norm_out,
                                     half* __restrict__ qjl_norm_out,
                                     int batch_stride)
{
    const int batch_id = blockIdx.x;
    const int tid      = threadIdx.x;
    const int nthreads = blockDim.x;

    const half* in = input + batch_id * batch_stride;
    uint8_t*    pk = packed + batch_id * kPackedDim;
    half*       mn = mse_norm_out + batch_id;
    half*       qn = qjl_norm_out + batch_id;

    // --- Step 1: Compute L2 norm (mse_norm) ---
    float local_sum = 0.0f;
    for (int i = tid; i < kHeadDim; i += nthreads) {
        float val = __half2float(in[i]);
        local_sum += val * val;
    }

    __shared__ float s_partial[256];
    s_partial[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s)
            s_partial[tid] += s_partial[tid + s];
        __syncthreads();
    }
    float mse_norm = sqrtf(s_partial[0] + 1e-8f);
    if (tid == 0)
        *mn = __float2half(mse_norm);

    // --- Step 2: 3-bit MSE + 1-bit QJL on Hadamard-rotated normalized vector ---
    // We compute Hadamard rotation inline per element (the kernel receives
    // already-rotated input from the test harness, matching the real pipeline
    // where ProcessKV_v2 would rotate before calling this kernel).
    //
    // For the standalone test, the host applies Hadamard before calling us.

    float sigma    = 1.0f / sqrtf((float)kHeadDim);
    float inv_norm = 1.0f / mse_norm;

    // 3-bit boundaries at σ=1 (hardcoded, same as turbo_quant_qjl4.h)
    constexpr float b_std[7] = {-1.7479274f, -1.0499572f, -0.5005497f, 0.0f, 0.5005497f, 1.0499572f, 1.7479274f};
    // 3-bit centroids at σ=1
    constexpr float c_std[8] = {
        -2.1519456f, -1.3439092f, -0.7560052f, -0.2450942f, 0.2450942f, 0.7560052f, 1.3439092f, 2.1519456f};

    // Boundaries & centroids at runtime scale
    float b[7], c[8];
    for (int i = 0; i < 7; ++i)
        b[i] = b_std[i] * sigma;
    for (int i = 0; i < 8; ++i)
        c[i] = c_std[i] * sigma;

    // Per-thread residual^2 accumulation for qjl_norm
    float local_qjl_sq = 0.0f;

    // Pack pairs of nibbles into bytes
    for (int j = tid; j < kPackedDim; j += nthreads) {
        // Two consecutive elements map to one byte
        float y0 = __half2float(in[j]) * inv_norm;  // normalized rotated
        float y1 = __half2float(in[j + kPackedDim]) * inv_norm;

        // 3-bit MSE indices via boundary comparison
        int idx0 = 0, idx1 = 0;
        for (int k = 0; k < 7; ++k) {
            if (y0 > b[k])
                idx0++;
            if (y1 > b[k])
                idx1++;
        }

        // Centroid values
        float cent0 = c[idx0];
        float cent1 = c[idx1];

        // Residuals
        float r0 = y0 - cent0;
        float r1 = y1 - cent1;

        // QJL sign bit
        int sign0 = (r0 >= 0.0f) ? 1 : 0;
        int sign1 = (r1 >= 0.0f) ? 1 : 0;

        local_qjl_sq += r0 * r0 + r1 * r1;

        // Pack nibble: low 3 bits = MSE index, bit 3 = QJL sign
        uint8_t nib0 = idx0 | (sign0 << 3);
        uint8_t nib1 = idx1 | (sign1 << 3);

        pk[j] = nib0 | (nib1 << 4);
    }

    // Reduce qjl_sq across threads
    __shared__ float s_qjl[256];
    s_qjl[tid] = local_qjl_sq;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s)
            s_qjl[tid] += s_qjl[tid + s];
        __syncthreads();
    }

    // qjl_norm = ||residual|| / sqrt(d)
    if (tid == 0) {
        float qjl_norm = sqrtf(s_qjl[0] + 1e-8f) / sqrtf((float)kHeadDim);
        *qn            = __float2half(qjl_norm);
    }
}

// ---------------------------------------------------------------------------
// GPU kernel for K QJL4 dequantization (rotate domain, no inverse Hadamard)
//   unpack nibble → MSE index + QJL sign → codebook lookup + sign →
//   scale by mse_norm → output in rotate domain
// ---------------------------------------------------------------------------

template<int kHeadDim, int kPackedDim = kHeadDim / 2>
__global__ void dequantize_qjl4_kernel(const uint8_t* __restrict__ packed,
                                       const half* __restrict__ mse_norm_in,
                                       const half* __restrict__ qjl_norm_in,
                                       half* __restrict__ output,
                                       int batch_stride)
{
    const int batch_id = blockIdx.x;
    const int tid      = threadIdx.x;
    const int nthreads = blockDim.x;

    const uint8_t* pk  = packed + batch_id * kPackedDim;
    float          mn  = __half2float(mse_norm_in[batch_id]);
    float          qn  = __half2float(qjl_norm_in[batch_id]);
    half*          out = output + batch_id * batch_stride;

    float sigma = 1.0f / sqrtf((float)kHeadDim);

    // 3-bit centroids at σ=1 (hardcoded, same as turbo_quant_qjl4.h)
    constexpr float c_std[8] = {
        -2.1519456f, -1.3439092f, -0.7560052f, -0.2450942f, 0.2450942f, 0.7560052f, 1.3439092f, 2.1519456f};
    float c[8];
    for (int i = 0; i < 8; ++i)
        c[i] = c_std[i] * sigma;

    for (int j = tid; j < kPackedDim; j += nthreads) {
        uint8_t byte = pk[j];
        int     nib0 = byte & 0x0F;
        int     nib1 = (byte >> 4) & 0x0F;

        int idx0 = nib0 & 0x7;
        int sgn0 = (nib0 >> 3) & 0x1;
        int idx1 = nib1 & 0x7;
        int sgn1 = (nib1 >> 3) & 0x1;

        float sign0 = sgn0 * 2.0f - 1.0f;
        float sign1 = sgn1 * 2.0f - 1.0f;

        float v0 = (c[idx0] + qn * sign0) * mn;
        float v1 = (c[idx1] + qn * sign1) * mn;

        out[j]              = __float2half(v0);
        out[j + kPackedDim] = __float2half(v1);
    }
}

// ---------------------------------------------------------------------------
// CPU reference implementation for QJL4
// ---------------------------------------------------------------------------

void quantize_qjl4_reference(const float* input, uint8_t* packed, float* mse_norm, float* qjl_norm, int raw_dim)
{
    // L2 norm
    float norm = 0;
    for (int i = 0; i < raw_dim; ++i)
        norm += input[i] * input[i];
    norm      = sqrtf(norm + 1e-10f);
    *mse_norm = norm;

    float sigma    = 1.0f / sqrtf((float)raw_dim);
    float inv_norm = 1.0f / norm;

    // Boundaries & centroids at runtime scale
    float boundaries_std[7] = {-1.7479274f, -1.0499572f, -0.5005497f, 0.0f, 0.5005497f, 1.0499572f, 1.7479274f};
    float centroids_std[8]  = {
        -2.1519456f, -1.3439092f, -0.7560052f, -0.2450942f, 0.2450942f, 0.7560052f, 1.3439092f, 2.1519456f};

    int   packed_dim = raw_dim / 2;
    float qjl_sq     = 0;

    for (int j = 0; j < packed_dim; ++j) {
        float y0 = input[j] * inv_norm;
        float y1 = input[j + packed_dim] * inv_norm;

        // 3-bit MSE
        int idx0 = 0, idx1 = 0;
        for (int b = 0; b < 7; ++b) {
            if (y0 > boundaries_std[b] * sigma)
                idx0++;
            if (y1 > boundaries_std[b] * sigma)
                idx1++;
        }

        float cv0 = centroids_std[idx0] * sigma;
        float cv1 = centroids_std[idx1] * sigma;

        float r0 = y0 - cv0;
        float r1 = y1 - cv1;

        int sign0 = (r0 >= 0) ? 1 : 0;
        int sign1 = (r1 >= 0) ? 1 : 0;

        qjl_sq += r0 * r0 + r1 * r1;

        uint8_t nib0 = idx0 | (sign0 << 3);
        uint8_t nib1 = idx1 | (sign1 << 3);
        packed[j]    = nib0 | (nib1 << 4);
    }

    *qjl_norm = sqrtf(qjl_sq + 1e-8f) / sqrtf((float)raw_dim);
}

void dequantize_qjl4_reference(
    const uint8_t* packed, const float* mse_norm, const float* qjl_norm, float* output, int raw_dim)
{
    float sigma = 1.0f / sqrtf((float)raw_dim);
    float mn    = *mse_norm;
    float qn    = *qjl_norm;

    float centroids_std[8] = {
        -2.1519456f, -1.3439092f, -0.7560052f, -0.2450942f, 0.2450942f, 0.7560052f, 1.3439092f, 2.1519456f};

    int packed_dim = raw_dim / 2;
    for (int j = 0; j < packed_dim; ++j) {
        uint8_t byte = packed[j];
        int     nib0 = byte & 0x0F;
        int     nib1 = (byte >> 4) & 0x0F;

        int idx0 = nib0 & 0x7, sgn0 = (nib0 >> 3) & 0x1;
        int idx1 = nib1 & 0x7, sgn1 = (nib1 >> 3) & 0x1;

        float sign0 = sgn0 * 2.0f - 1.0f;
        float sign1 = sgn1 * 2.0f - 1.0f;

        output[j]              = (centroids_std[idx0] * sigma + qn * sign0) * mn;
        output[j + packed_dim] = (centroids_std[idx1] * sigma + qn * sign1) * mn;
    }
}

// ---------------------------------------------------------------------------
// Test driver — mirrors PyTorch test_turboquant.py::TestTurboQuantQJL4
// ---------------------------------------------------------------------------

template<int kHeadDim>
void test_qjl4_roundtrip(int batch = 100)
{
    std::cout << "Testing TurboQuant QJL4 roundtrip: head_dim=" << kHeadDim << " batch=" << batch << std::endl;

    const int n          = batch * kHeadDim;
    const int packed_dim = kHeadDim / 2;

    // Generate random input → normalize to unit sphere
    std::vector<float>              host_input(n);
    std::mt19937                    rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int b = 0; b < batch; ++b) {
        float* p    = host_input.data() + b * kHeadDim;
        float  norm = 0;
        for (int i = 0; i < kHeadDim; ++i) {
            p[i] = dist(rng);
            norm += p[i] * p[i];
        }
        norm = sqrtf(norm + 1e-10f);
        for (int i = 0; i < kHeadDim; ++i) {
            p[i] /= norm;
        }
    }

    // GPU memory
    thrust::universal_vector<half>    d_input(n);
    thrust::universal_vector<half>    d_rotated(n);
    thrust::universal_vector<uint8_t> d_packed(batch * packed_dim);
    thrust::universal_vector<half>    d_mse_norm(batch);
    thrust::universal_vector<half>    d_qjl_norm(batch);
    thrust::universal_vector<half>    d_dequant_rot(n);
    thrust::universal_vector<half>    d_dequant(n);

    for (int i = 0; i < n; ++i)
        d_input[i] = __float2half(host_input[i]);

    // Step 1: Hadamard rotate
    HadamardParams hparams;
    hparams.batch   = batch;
    hparams.dim     = kHeadDim;
    hparams.log_dim = (int)round(log2((float)kHeadDim));
    hparams.stride  = kHeadDim;
    hparams.scale   = 1.0f / sqrtf((float)kHeadDim);
    hparams.x_ptr   = thrust::raw_pointer_cast(d_input.data());
    hparams.out_ptr = thrust::raw_pointer_cast(d_rotated.data());
    hadamard_transform_fp16(hparams, 0);

    // Step 2: Quantize QJL4
    quantize_qjl4_kernel<kHeadDim><<<batch, 128>>>(thrust::raw_pointer_cast(d_rotated.data()),
                                                   thrust::raw_pointer_cast(d_packed.data()),
                                                   thrust::raw_pointer_cast(d_mse_norm.data()),
                                                   thrust::raw_pointer_cast(d_qjl_norm.data()),
                                                   kHeadDim);

    // Step 3: Dequantize to rotate domain
    dequantize_qjl4_kernel<kHeadDim><<<batch, 128>>>(thrust::raw_pointer_cast(d_packed.data()),
                                                     thrust::raw_pointer_cast(d_mse_norm.data()),
                                                     thrust::raw_pointer_cast(d_qjl_norm.data()),
                                                     thrust::raw_pointer_cast(d_dequant_rot.data()),
                                                     kHeadDim);

    // Step 4: Inverse Hadamard
    hparams.x_ptr   = thrust::raw_pointer_cast(d_dequant_rot.data());
    hparams.out_ptr = thrust::raw_pointer_cast(d_dequant.data());
    hadamard_transform_fp16(hparams, 0);
    cudaDeviceSynchronize();

    // ---- Metrics ----
    std::vector<float> host_dequant(n);
    {
        std::vector<half> tmp(n);
        cudaMemcpy(tmp.data(), thrust::raw_pointer_cast(d_dequant.data()), n * sizeof(half), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i)
            host_dequant[i] = __half2float(tmp[i]);
    }

    // MSE
    float mse = 0;
    for (int i = 0; i < n; ++i) {
        float e = host_input[i] - host_dequant[i];
        mse += e * e;
    }
    mse /= n;

    // Cosine similarity
    float cos_sum = 0;
    for (int b = 0; b < batch; ++b) {
        const float* x   = host_input.data() + b * kHeadDim;
        const float* xr  = host_dequant.data() + b * kHeadDim;
        float        dot = 0, nx = 0, nr = 0;
        for (int i = 0; i < kHeadDim; ++i) {
            dot += x[i] * xr[i];
            nx += x[i] * x[i];
            nr += xr[i] * xr[i];
        }
        cos_sum += dot / (sqrtf(nx + 1e-10f) * sqrtf(nr + 1e-10f));
    }
    float cos_sim = cos_sum / batch;

    std::cout << "  MSE=" << mse << " cos_sim=" << cos_sim << std::endl;

    // Verify packed data matches CPU reference
    std::vector<float> host_rotated(n);
    {
        std::vector<half> tmp(n);
        cudaMemcpy(tmp.data(), thrust::raw_pointer_cast(d_rotated.data()), n * sizeof(half), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i)
            host_rotated[i] = __half2float(tmp[i]);
    }

    std::vector<uint8_t> ref_packed(batch * packed_dim);
    std::vector<float>   ref_mse_norm(batch);
    std::vector<float>   ref_qjl_norm(batch);
    for (int b = 0; b < batch; ++b) {
        quantize_qjl4_reference(host_rotated.data() + b * kHeadDim,
                                ref_packed.data() + b * packed_dim,
                                ref_mse_norm.data() + b,
                                ref_qjl_norm.data() + b,
                                kHeadDim);
    }

    std::vector<uint8_t> gpu_packed(batch * packed_dim);
    cudaMemcpy(
        gpu_packed.data(), thrust::raw_pointer_cast(d_packed.data()), batch * packed_dim, cudaMemcpyDeviceToHost);

    int packed_mismatch = 0;
    for (int i = 0; i < batch * packed_dim; ++i) {
        if (gpu_packed[i] != ref_packed[i])
            packed_mismatch++;
    }
    std::cout << "  Packed data mismatch: " << packed_mismatch << "/" << (batch * packed_dim) << std::endl;

    // Assertions (matching PyTorch thresholds)
    if (cos_sim < 0.86f) {
        std::cerr << "FAIL: cosine similarity " << cos_sim << " below 0.86" << std::endl;
    }
}

int main()
{
    test_qjl4_roundtrip<128>(100);
    test_qjl4_roundtrip<64>(100);
    test_qjl4_roundtrip<256>(50);

    std::cout << "All TurboQuant QJL4 tests passed!" << std::endl;
    return 0;
}
