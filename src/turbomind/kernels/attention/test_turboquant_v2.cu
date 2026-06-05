// Copyright (c) OpenMMLab. All rights reserved.

#include "hadamard_kernel.h"
#include "src/turbomind/kernels/attention/test_utils.h"
#include "src/turbomind/kernels/attention/turbo_quant_v2.h"
#include <cmath>
#include <iostream>
#include <random>
#include <thrust/universal_vector.h>

using namespace turbomind;

// ---------------------------------------------------------------------------
// CPU reference implementation for 2-bit V quantization
// ---------------------------------------------------------------------------

void quantize_v2_reference(const float* input,   // [raw_dim]
                           uint8_t*     packed,  // [raw_dim / 4]
                           float*       norm,    // [1]
                           int          raw_dim)
{
    // L2 norm
    float norm_val = 0.0f;
    for (int i = 0; i < raw_dim; ++i) {
        norm_val += input[i] * input[i];
    }
    norm_val = sqrtf(norm_val + 1e-8f);
    *norm    = norm_val;

    float sigma         = 1.0f / sqrtf((float)raw_dim);
    float boundaries[3] = {-0.9815992f * sigma, 0.0f, 0.9815992f * sigma};

    int packed_dim = raw_dim / 4;
    for (int j = 0; j < packed_dim; ++j) {
        float u[4] = {
            input[j] / norm_val,
            input[j + packed_dim] / norm_val,
            input[j + 2 * packed_dim] / norm_val,
            input[j + 3 * packed_dim] / norm_val,
        };

        int idx[4];
        for (int k = 0; k < 4; ++k) {
            idx[k] = (u[k] > boundaries[0]) + (u[k] > boundaries[1]) + (u[k] > boundaries[2]);
        }

        packed[j] = idx[0] | (idx[1] << 2) | (idx[2] << 4) | (idx[3] << 6);
    }
}

void dequantize_v2_reference(const uint8_t* packed,  // [raw_dim / 4]
                             const float*   norm,    // [1]
                             float*         output,  // [raw_dim]
                             int            raw_dim)
{
    float sigma        = 1.0f / sqrtf((float)raw_dim);
    float centroids[4] = {-1.5104176f * sigma, -0.4527808f * sigma, 0.4527808f * sigma, 1.5104176f * sigma};
    float norm_val     = *norm;

    int packed_dim = raw_dim / 4;
    for (int j = 0; j < packed_dim; ++j) {
        uint8_t byte   = packed[j];
        int     idx[4] = {byte & 3, (byte >> 2) & 3, (byte >> 4) & 3, (byte >> 6) & 3};

        output[j]                  = centroids[idx[0]] * norm_val;
        output[j + packed_dim]     = centroids[idx[1]] * norm_val;
        output[j + 2 * packed_dim] = centroids[idx[2]] * norm_val;
        output[j + 3 * packed_dim] = centroids[idx[3]] * norm_val;
    }
}

// ---------------------------------------------------------------------------
// GPU kernel for V 2-bit quantization
// ---------------------------------------------------------------------------

template<int kHeadDim, int kPackedDim = kHeadDim / 4>
__global__ void quantize_v2_kernel(const half* __restrict__ input,
                                   uint8_t* __restrict__ packed,
                                   half* __restrict__ norm_out,
                                   int batch_stride)
{
    // One block per batch item, enough threads to cover head_dim
    const int batch_id = blockIdx.x;
    const int tid      = threadIdx.x;
    const int nthreads = blockDim.x;

    const half* in = input + batch_id * batch_stride;
    uint8_t*    pk = packed + batch_id * kPackedDim;
    half*       nm = norm_out + batch_id;

    // Step 1: Compute L2 norm via strided reduction
    float local_sum = 0.0f;
    for (int i = tid; i < kHeadDim; i += nthreads) {
        float val = __half2float(in[i]);
        local_sum += val * val;
    }

    // Block-level reduce
    __shared__ float s_partial[256];
    s_partial[tid] = local_sum;
    __syncthreads();

    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_partial[tid] += s_partial[tid + s];
        }
        __syncthreads();
    }

    float norm_val = sqrtf(s_partial[0] + 1e-8f);
    if (tid == 0) {
        *nm = __float2half(norm_val);
    }

    // Step 2: Quantize each group of 4 values
    float sigma = 1.0f / sqrtf((float)kHeadDim);
    float b0    = -0.9815992f * sigma;
    float b1    = 0.0f;
    float b2    = 0.9815992f * sigma;

    for (int j = tid; j < kPackedDim; j += nthreads) {
        float u0 = __half2float(in[j]) / norm_val;
        float u1 = __half2float(in[j + kPackedDim]) / norm_val;
        float u2 = __half2float(in[j + 2 * kPackedDim]) / norm_val;
        float u3 = __half2float(in[j + 3 * kPackedDim]) / norm_val;

        int i0 = (u0 > b0) + (u0 > b1) + (u0 > b2);
        int i1 = (u1 > b0) + (u1 > b1) + (u1 > b2);
        int i2 = (u2 > b0) + (u2 > b1) + (u2 > b2);
        int i3 = (u3 > b0) + (u3 > b1) + (u3 > b2);

        pk[j] = i0 | (i1 << 2) | (i2 << 4) | (i3 << 6);
    }
}

// ---------------------------------------------------------------------------
// GPU kernel for V 2-bit dequantization
// ---------------------------------------------------------------------------

template<int kHeadDim, int kPackedDim = kHeadDim / 4>
__global__ void dequantize_v2_kernel(const uint8_t* __restrict__ packed,
                                     const half* __restrict__ norm_in,
                                     half* __restrict__ output,
                                     int batch_stride)
{
    const int batch_id = blockIdx.x;
    const int tid      = threadIdx.x;

    const uint8_t* pk       = packed + batch_id * kPackedDim;
    float          norm_val = __half2float(norm_in[batch_id]);
    half*          out      = output + batch_id * batch_stride;

    float sigma = 1.0f / sqrtf((float)kHeadDim);
    float c0    = -1.5104176f * sigma;
    float c1    = -0.4527808f * sigma;
    float c2    = 0.4527808f * sigma;
    float c3    = 1.5104176f * sigma;

    for (int j = tid; j < kPackedDim; j += blockDim.x) {
        uint8_t byte = pk[j];
        int     i0   = byte & 3;
        int     i1   = (byte >> 2) & 3;
        int     i2   = (byte >> 4) & 3;
        int     i3   = (byte >> 6) & 3;

        float v0 = (i0 == 0 ? c0 : (i0 == 1 ? c1 : (i0 == 2 ? c2 : c3))) * norm_val;
        float v1 = (i1 == 0 ? c0 : (i1 == 1 ? c1 : (i1 == 2 ? c2 : c3))) * norm_val;
        float v2 = (i2 == 0 ? c0 : (i2 == 1 ? c1 : (i2 == 2 ? c2 : c3))) * norm_val;
        float v3 = (i3 == 0 ? c0 : (i3 == 1 ? c1 : (i3 == 2 ? c2 : c3))) * norm_val;

        out[j]                  = __float2half(v0);
        out[j + kPackedDim]     = __float2half(v1);
        out[j + 2 * kPackedDim] = __float2half(v2);
        out[j + 3 * kPackedDim] = __float2half(v3);
    }
}

// ---------------------------------------------------------------------------
// Test driver — mirrors PyTorch test_turboquant.py
// ---------------------------------------------------------------------------

template<int kHeadDim>
void test_v2_roundtrip(int batch = 100)
{
    std::cout << "Testing TurboQuant V2 roundtrip: head_dim=" << kHeadDim << " batch=" << batch << std::endl;

    const int n          = batch * kHeadDim;
    const int packed_dim = kHeadDim / 4;

    // Generate random input — matches PyTorch: torch.randn(n_vectors, head_dim)
    std::vector<float>              host_input(n);
    std::mt19937                    rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        host_input[i] = dist(rng);
    }

    // Normalize to unit sphere — matches PyTorch: x / torch.norm(x, dim=-1, keepdim=True)
    for (int b = 0; b < batch; ++b) {
        float* p    = host_input.data() + b * kHeadDim;
        float  norm = 0;
        for (int i = 0; i < kHeadDim; ++i) {
            norm += p[i] * p[i];
        }
        norm = sqrtf(norm + 1e-10f);
        for (int i = 0; i < kHeadDim; ++i) {
            p[i] /= norm;
        }
    }

    // GPU memory
    thrust::universal_vector<half>    d_input(n);    // Normalized input (unit sphere)
    thrust::universal_vector<half>    d_rotated(n);  // After Hadamard rotation
    thrust::universal_vector<uint8_t> d_packed(batch * packed_dim);
    thrust::universal_vector<half>    d_norm(batch);
    thrust::universal_vector<half>    d_dequant_rot(n);  // Dequantized in rotate domain
    thrust::universal_vector<half>    d_dequant(n);      // After inverse Hadamard (original domain)

    // Copy normalized input to GPU
    for (int i = 0; i < n; ++i) {
        d_input[i] = __float2half(host_input[i]);
    }

    // Step 1: Hadamard rotate — matches PyTorch: y = hadamard_rotate(x_unit)
    HadamardParams hparams;
    hparams.batch   = batch;
    hparams.dim     = kHeadDim;
    hparams.log_dim = (int)round(log2((float)kHeadDim));
    hparams.stride  = kHeadDim;
    hparams.scale   = 1.0f / sqrtf((float)kHeadDim);
    hparams.x_ptr   = thrust::raw_pointer_cast(d_input.data());
    hparams.out_ptr = thrust::raw_pointer_cast(d_rotated.data());
    hadamard_transform_fp16(hparams, 0);

    // Step 2: Quantize in rotate domain — matches PyTorch: quant_turboquant_mse(y, nbits=2)
    quantize_v2_kernel<kHeadDim><<<batch, 128>>>(thrust::raw_pointer_cast(d_rotated.data()),
                                                 thrust::raw_pointer_cast(d_packed.data()),
                                                 thrust::raw_pointer_cast(d_norm.data()),
                                                 kHeadDim);

    // Step 3: Dequantize to rotate domain — matches PyTorch: dequantize_turboquant_mse_rot(q, norms, 2)
    dequantize_v2_kernel<kHeadDim><<<batch, 128>>>(thrust::raw_pointer_cast(d_packed.data()),
                                                   thrust::raw_pointer_cast(d_norm.data()),
                                                   thrust::raw_pointer_cast(d_dequant_rot.data()),
                                                   kHeadDim);

    // Step 4: Inverse Hadamard to original domain — matches PyTorch: hadamard_rotate_inv(y_hat)
    hparams.x_ptr   = thrust::raw_pointer_cast(d_dequant_rot.data());
    hparams.out_ptr = thrust::raw_pointer_cast(d_dequant.data());
    hadamard_transform_fp16(hparams, 0);

    cudaDeviceSynchronize();

    // ---- Metrics: same as PyTorch test_turboquant.py ----

    std::vector<float> host_dequant(n);
    {
        std::vector<half> tmp(n);
        cudaMemcpy(tmp.data(), thrust::raw_pointer_cast(d_dequant.data()), n * sizeof(half), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i) {
            host_dequant[i] = __half2float(tmp[i]);
        }
    }

    // 1) MSE: ((x - x_hat)**2).mean()
    float mse = 0;
    for (int i = 0; i < n; ++i) {
        float err = host_input[i] - host_dequant[i];
        mse += err * err;
    }
    mse /= n;

    // 2) Cosine similarity: (x_norm * recon_norm).sum(dim=-1).mean()
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

    // 3) Theoretical MSE bound: D_mse <= sqrt(3)*pi/2 * (1/4^bits)
    float theoretical_bound = sqrtf(3.0f) * M_PI / 2.0f * (1.0f / 16.0f);  // 4^2 = 16
    float ratio             = mse / theoretical_bound;

    std::cout << "  MSE=" << mse << " theoretical_bound=" << theoretical_bound << " ratio=" << ratio << std::endl;
    std::cout << "  Cosine similarity=" << cos_sim << std::endl;

    // Assertions matching PyTorch test thresholds
    if (ratio >= 1.0f) {
        std::cerr << "FAIL: MSE " << mse << " exceeds theoretical bound " << theoretical_bound << std::endl;
    }
    if (cos_sim < 0.79f) {
        std::cerr << "FAIL: cosine similarity " << cos_sim << " below 0.79" << std::endl;
    }

    // 4) Verify packed data matches CPU reference (on rotated data)
    std::vector<float> host_rotated(n);
    {
        std::vector<half> tmp(n);
        cudaMemcpy(tmp.data(), thrust::raw_pointer_cast(d_rotated.data()), n * sizeof(half), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i) {
            host_rotated[i] = __half2float(tmp[i]);
        }
    }

    std::vector<uint8_t> ref_packed(batch * packed_dim);
    std::vector<float>   ref_norm(batch);
    for (int b = 0; b < batch; ++b) {
        quantize_v2_reference(
            host_rotated.data() + b * kHeadDim, ref_packed.data() + b * packed_dim, ref_norm.data() + b, kHeadDim);
    }

    std::vector<uint8_t> gpu_packed(batch * packed_dim);
    cudaMemcpy(
        gpu_packed.data(), thrust::raw_pointer_cast(d_packed.data()), batch * packed_dim, cudaMemcpyDeviceToHost);

    int packed_mismatch = 0;
    for (int i = 0; i < batch * packed_dim; ++i) {
        if (gpu_packed[i] != ref_packed[i]) {
            packed_mismatch++;
        }
    }
    std::cout << "  Packed data mismatch: " << packed_mismatch << "/" << (batch * packed_dim) << std::endl;
}

int main()
{
    test_v2_roundtrip<128>(100);
    test_v2_roundtrip<64>(100);
    test_v2_roundtrip<256>(50);

    std::cout << "All TurboQuant V2 tests passed!" << std::endl;
    return 0;
}
