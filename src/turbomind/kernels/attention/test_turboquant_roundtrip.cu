// Copyright (c) OpenMMLab. All rights reserved.
//
// Test: TurboQuant full write→read roundtrip
//
// Level 1: ProcessKV write path (Hadamard + quantize → packed data + norms)
// Level 3: Full roundtrip (write → read → compare with CPU reference)
//
// Self-contained: no dependency on array_ops.h / data_type.h

#include <cassert>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_fp16.h>

#include "src/turbomind/kernels/attention/turbo_quant_qjl4.h"
#include "src/turbomind/kernels/attention/turbo_quant_v2.h"

using namespace turbomind;

// Minimal Array<T, N> for this test
template<typename T, int N>
struct Array {
    T data[N];
    __device__ __host__ T& operator[](int i) { return data[i]; }
    __device__ __host__ const T& operator[](int i) const { return data[i]; }
};

#define PRAGMA_UNROLL _Pragma("unroll")

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const int kHeadDim = 128;
static const int kVecSize = 8;

// ---------------------------------------------------------------------------
// GPU kernel: ProcessKV write path (TurboQuant)
// Replicates the logic from kv_cache_utils_v2.cu
// ---------------------------------------------------------------------------

__global__ void turboquant_write_kernel(
    const half* __restrict__ input_k,   // [head_dim]
    const half* __restrict__ input_v,   // [head_dim]
    uint32_t*   __restrict__ packed_k,  // [head_dim/8] (8 nibbles per uint32)
    uint16_t*   __restrict__ packed_v,  // [head_dim/8] (8 x 2bit per uint16)
    half*       __restrict__ param_k,   // [2] = {mse_norm, qjl_norm}
    half*       __restrict__ param_v,   // [2] = {v_norm, 0}
    int head_dim)
{
    // Simulate: 1 warp, ThreadMapQ-like layout
    // lane_id 0..15 each holds kVecSize=8 consecutive elements
    const int lane_id = threadIdx.x % 32;
    const int kAccessC = kVecSize;
    const int kWarpThreadC = 16;  // 128 / kAccessC
    if (lane_id >= kWarpThreadC) return;

    const int di = lane_id * kAccessC;  // offset into head_dim

    // Load K/V
    Array<half, kVecSize> vec_k, vec_v;
    PRAGMA_UNROLL
    for (int j = 0; j < kVecSize; ++j) {
        vec_k[j] = input_k[di + j];
        vec_v[j] = input_v[di + j];
    }

    // 1. Hadamard rotate K (register butterfly, float intermediate)
    Array<float, kVecSize> fval_k;
    PRAGMA_UNROLL
    for (int j = 0; j < kVecSize; ++j) fval_k[j] = (float)vec_k[j];
    // Local butterfly (3 stages)
    PRAGMA_UNROLL
    for (int s = 0; s < 3; ++s) {
        const int stride = 1 << s;
        PRAGMA_UNROLL
        for (int j = 0; j < kVecSize; ++j) {
            if (j & stride) {
                float a = fval_k[j - stride], b = fval_k[j];
                fval_k[j - stride] = a + b;
                fval_k[j] = a - b;
            }
        }
    }
    // Shuffle butterfly (4 stages for head_dim=128)
    PRAGMA_UNROLL
    for (int s = 0; s < 4; ++s) {
        const int mask = 1 << s;
        const float sign = (lane_id & mask) ? -1.f : 1.f;
        PRAGMA_UNROLL
        for (int j = 0; j < kVecSize; ++j) {
            float other = __shfl_xor_sync(0xffffffff, fval_k[j], mask);
            fval_k[j] = sign * fval_k[j] + other;
        }
    }
    const float had_scale = 1.0f / sqrtf((float)head_dim);
    PRAGMA_UNROLL
    for (int j = 0; j < kVecSize; ++j) vec_k[j] = (half)(fval_k[j] * had_scale);

    // 1b. Hadamard rotate V (same)
    Array<float, kVecSize> fval_v;
    PRAGMA_UNROLL
    for (int j = 0; j < kVecSize; ++j) fval_v[j] = (float)vec_v[j];
    PRAGMA_UNROLL
    for (int s = 0; s < 3; ++s) {
        const int stride = 1 << s;
        PRAGMA_UNROLL
        for (int j = 0; j < kVecSize; ++j) {
            if (j & stride) {
                float a = fval_v[j - stride], b = fval_v[j];
                fval_v[j - stride] = a + b;
                fval_v[j] = a - b;
            }
        }
    }
    PRAGMA_UNROLL
    for (int s = 0; s < 4; ++s) {
        const int mask = 1 << s;
        const float sign = (lane_id & mask) ? -1.f : 1.f;
        PRAGMA_UNROLL
        for (int j = 0; j < kVecSize; ++j) {
            float other = __shfl_xor_sync(0xffffffff, fval_v[j], mask);
            fval_v[j] = sign * fval_v[j] + other;
        }
    }
    PRAGMA_UNROLL
    for (int j = 0; j < kVecSize; ++j) vec_v[j] = (half)(fval_v[j] * had_scale);

    // 2. L2 norms (warp reduce)
    float sum_k_sq = 0.f, sum_v_sq = 0.f;
    PRAGMA_UNROLL
    for (int j = 0; j < kVecSize; ++j) {
        float kv = (float)vec_k[j];
        sum_k_sq += kv * kv;
        float vv = (float)vec_v[j];
        sum_v_sq += vv * vv;
    }
    PRAGMA_UNROLL
    for (int mask = 16; mask >= 1; mask /= 2) {
        sum_k_sq += __shfl_xor_sync(0xffffffff, sum_k_sq, mask);
        sum_v_sq += __shfl_xor_sync(0xffffffff, sum_v_sq, mask);
    }
    const float mse_norm = sqrtf(sum_k_sq);
    const float v_norm   = sqrtf(sum_v_sq);
    const float inv_mse_norm = (mse_norm > 0.f) ? 1.f / mse_norm : 0.f;
    const float inv_v_norm   = (v_norm > 0.f) ? 1.f / v_norm : 0.f;

    // 3. QJL4 quantize K
    const float sigma = 1.f / sqrtf((float)head_dim);
    float residual_sq = 0.f;
    uint32_t pk = 0;
    PRAGMA_UNROLL
    for (int j = 0; j < kVecSize; ++j) {
        float y = (float)vec_k[j] * inv_mse_norm;
        int idx3 = attention::quantize_k4_mse(y, sigma);
        float centroid = attention::dequantize_k4_mse(idx3, sigma);
        float res = y - centroid;
        int sign_bit = (res >= 0.f) ? 1 : 0;
        residual_sq += res * res;
        uint8_t nibble = (uint8_t)(idx3 | (sign_bit << 3));
        pk |= ((uint32_t)nibble << (j * 4));
    }
    PRAGMA_UNROLL
    for (int mask = 16; mask >= 1; mask /= 2) {
        residual_sq += __shfl_xor_sync(0xffffffff, residual_sq, mask);
    }
    const float qjl_norm = sqrtf(residual_sq / (float)head_dim);

    // 4. 2-bit quantize V
    uint16_t pv = 0;
    PRAGMA_UNROLL
    for (int j = 0; j < kVecSize; ++j) {
        float u = (float)vec_v[j] * inv_v_norm;
        int idx2 = attention::quantize_v2(u, sigma);
        pv |= ((uint16_t)idx2 << (j * 2));
    }

    // Store results
    if (lane_id < kWarpThreadC) {
        packed_k[lane_id] = pk;
        packed_v[lane_id] = pv;
    }
    if (lane_id == 0) {
        param_k[0] = (half)mse_norm;
        param_k[1] = (half)qjl_norm;
        param_v[0] = (half)v_norm;
        param_v[1] = (half)0;
    }
}

// ---------------------------------------------------------------------------
// GPU kernel: FlattenKV read path (TurboQuant)
// Replicates the logic from kv_cache_utils_v2.cu
// ---------------------------------------------------------------------------

__global__ void turboquant_read_kernel(
    const uint32_t* __restrict__ packed_k,
    const uint16_t* __restrict__ packed_v,
    const half*    __restrict__ param_k,   // {mse_norm, qjl_norm}
    const half*    __restrict__ param_v,   // {v_norm, 0}
    half*          __restrict__ output_k,  // [head_dim]
    half*          __restrict__ output_v,  // [head_dim]
    int head_dim)
{
    const int lane_id = threadIdx.x % 32;
    const int kAccessC = kVecSize;
    const int kWarpThreadC = 16;
    if (lane_id >= kWarpThreadC) return;

    const int di = lane_id * kAccessC;
    const float sigma = 1.f / sqrtf((float)head_dim);
    const float had_scale = 1.0f / sqrtf((float)head_dim);

    float mse_norm = 0, qjl_norm = 0, v_norm = 0;
    if (lane_id == 0) {
        mse_norm = (float)param_k[0];
        qjl_norm = (float)param_k[1];
        v_norm   = (float)param_v[0];
    }
    mse_norm = __shfl_sync(0xffffffff, mse_norm, 0);
    qjl_norm = __shfl_sync(0xffffffff, qjl_norm, 0);
    v_norm   = __shfl_sync(0xffffffff, v_norm, 0);

    // 1. Dequantize K: QJL4
    uint32_t pk = packed_k[lane_id];
    Array<half, kVecSize> out_k;
    PRAGMA_UNROLL
    for (int j = 0; j < kVecSize; ++j) {
        uint8_t nibble = (pk >> (j * 4)) & 0xF;
        int idx3     = nibble & 0x7;
        int sign_bit = (nibble >> 3) & 0x1;
        float centroid = attention::dequantize_k4_mse(idx3, sigma);
        float sign_val = sign_bit * 2.f - 1.f;
        out_k[j] = (half)(mse_norm * (centroid + qjl_norm * sign_val));
    }

    // 2. Dequantize V: 2-bit
    uint16_t pv = packed_v[lane_id];
    Array<half, kVecSize> out_v;
    PRAGMA_UNROLL
    for (int j = 0; j < kVecSize; ++j) {
        int idx2 = (pv >> (j * 2)) & 0x3;
        float centroid = attention::dequantize_v2(idx2, sigma);
        out_v[j] = (half)(v_norm * centroid);
    }

    // 3. Inverse Hadamard on K
    Array<float, kVecSize> fval_k;
    PRAGMA_UNROLL
    for (int j = 0; j < kVecSize; ++j) fval_k[j] = (float)out_k[j];
    PRAGMA_UNROLL
    for (int s = 0; s < 3; ++s) {
        const int stride = 1 << s;
        PRAGMA_UNROLL
        for (int j = 0; j < kVecSize; ++j) {
            if (j & stride) {
                float a = fval_k[j - stride], b = fval_k[j];
                fval_k[j - stride] = a + b;
                fval_k[j] = a - b;
            }
        }
    }
    PRAGMA_UNROLL
    for (int s = 0; s < 4; ++s) {
        const int mask = 1 << s;
        const float sign = (lane_id & mask) ? -1.f : 1.f;
        PRAGMA_UNROLL
        for (int j = 0; j < kVecSize; ++j) {
            float other = __shfl_xor_sync(0xffffffff, fval_k[j], mask);
            fval_k[j] = sign * fval_k[j] + other;
        }
    }
    PRAGMA_UNROLL
    for (int j = 0; j < kVecSize; ++j) out_k[j] = (half)(fval_k[j] * had_scale);

    // 4. Inverse Hadamard on V
    Array<float, kVecSize> fval_v;
    PRAGMA_UNROLL
    for (int j = 0; j < kVecSize; ++j) fval_v[j] = (float)out_v[j];
    PRAGMA_UNROLL
    for (int s = 0; s < 3; ++s) {
        const int stride = 1 << s;
        PRAGMA_UNROLL
        for (int j = 0; j < kVecSize; ++j) {
            if (j & stride) {
                float a = fval_v[j - stride], b = fval_v[j];
                fval_v[j - stride] = a + b;
                fval_v[j] = a - b;
            }
        }
    }
    PRAGMA_UNROLL
    for (int s = 0; s < 4; ++s) {
        const int mask = 1 << s;
        const float sign = (lane_id & mask) ? -1.f : 1.f;
        PRAGMA_UNROLL
        for (int j = 0; j < kVecSize; ++j) {
            float other = __shfl_xor_sync(0xffffffff, fval_v[j], mask);
            fval_v[j] = sign * fval_v[j] + other;
        }
    }
    PRAGMA_UNROLL
    for (int j = 0; j < kVecSize; ++j) out_v[j] = (half)(fval_v[j] * had_scale);

    // Store results
    PRAGMA_UNROLL
    for (int j = 0; j < kVecSize; ++j) {
        output_k[di + j] = out_k[j];
        output_v[di + j] = out_v[j];
    }
}

// ---------------------------------------------------------------------------
// CPU reference: exact same algorithm as GPU but in float64
// ---------------------------------------------------------------------------

static void cpu_hadamard(double* x, int n) {
    // Iterative Walsh-Hadamard transform
    for (int stride = 1; stride < n; stride *= 2) {
        for (int i = 0; i < n; i += 2 * stride) {
            for (int j = 0; j < stride; ++j) {
                double a = x[i + j];
                double b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
        }
    }
    double scale = 1.0 / sqrt((double)n);
    for (int i = 0; i < n; ++i) x[i] *= scale;
}

static void cpu_quantize_qjl4(const double* rotated_k, int head_dim,
                               uint8_t* packed_nibbles, double* mse_norm_out, double* qjl_norm_out) {
    const double sigma = 1.0 / sqrt((double)head_dim);
    double mse_norm = 0;
    for (int i = 0; i < head_dim; ++i) mse_norm += rotated_k[i] * rotated_k[i];
    mse_norm = sqrt(mse_norm);
    *mse_norm_out = mse_norm;
    double inv_mse = (mse_norm > 0) ? 1.0 / mse_norm : 0;

    double residual_sq = 0;
    for (int i = 0; i < head_dim; ++i) {
        double y = rotated_k[i] * inv_mse;
        // 3-bit MSE boundary compare
        int idx3 = 0;
        const double b[] = {-1.7479274, -1.0499572, -0.5005497, 0.0, 0.5005497, 1.0499572, 1.7479274};
        for (int k = 0; k < 7; ++k) if (y > b[k] * sigma) idx3++;
        double centroids[] = {-2.1519456, -1.3439092, -0.7560052, -0.2450942, 0.2450942, 0.7560052, 1.3439092, 2.1519456};
        double centroid = centroids[idx3] * sigma;
        double residual = y - centroid;
        int sign_bit = (residual >= 0) ? 1 : 0;
        residual_sq += residual * residual;
        packed_nibbles[i] = (uint8_t)(idx3 | (sign_bit << 3));
    }
    *qjl_norm_out = sqrt(residual_sq / (double)head_dim);
}

static void cpu_quantize_v2(const double* rotated_v, int head_dim,
                             int* indices, double* v_norm_out) {
    const double sigma = 1.0 / sqrt((double)head_dim);
    double vnorm = 0;
    for (int i = 0; i < head_dim; ++i) vnorm += rotated_v[i] * rotated_v[i];
    vnorm = sqrt(vnorm);
    *v_norm_out = vnorm;
    double inv_vnorm = (vnorm > 0) ? 1.0 / vnorm : 0;
    const double b[] = {-0.9815992, 0.0, 0.9815992};
    for (int i = 0; i < head_dim; ++i) {
        double u = rotated_v[i] * inv_vnorm;
        int idx2 = 0;
        for (int k = 0; k < 3; ++k) if (u > b[k] * sigma) idx2++;
        indices[i] = idx2;
    }
}

static void cpu_dequantize_qjl4(const uint8_t* nibbles, double mse_norm, double qjl_norm,
                                 int head_dim, double* out) {
    const double sigma = 1.0 / sqrt((double)head_dim);
    const double centroids[] = {-2.1519456, -1.3439092, -0.7560052, -0.2450942,
                                 0.2450942,  0.7560052,  1.3439092,  2.1519456};
    for (int i = 0; i < head_dim; ++i) {
        int idx3 = nibbles[i] & 0x7;
        int sign_bit = (nibbles[i] >> 3) & 0x1;
        double centroid = centroids[idx3] * sigma;
        double sign_val = sign_bit * 2.0 - 1.0;
        out[i] = mse_norm * (centroid + qjl_norm * sign_val);
    }
}

static void cpu_dequantize_v2(const int* indices, double v_norm, int head_dim, double* out) {
    const double sigma = 1.0 / sqrt((double)head_dim);
    const double centroids[] = {-1.5104176, -0.4527808, 0.4527808, 1.5104176};
    for (int i = 0; i < head_dim; ++i) {
        out[i] = v_norm * centroids[indices[i]] * sigma;
    }
}

// ---------------------------------------------------------------------------
// Main test
// ---------------------------------------------------------------------------

int main() {
    const int N = kHeadDim;
    const int N_PACKED_K = N / 8;  // 8 nibbles per uint32
    const int N_PACKED_V = N / 8;  // 8 x 2bit per uint16

    // Allocate host memory
    std::vector<float> h_input_k(N), h_input_v(N);
    std::vector<float> h_output_k(N), h_output_v(N);
    std::vector<uint32_t> h_packed_k(N_PACKED_K);
    std::vector<uint16_t> h_packed_v(N_PACKED_V);
    float h_param_k[2], h_param_v[2];

    // Generate random input
    srand(42);
    for (int i = 0; i < N; ++i) {
        h_input_k[i] = (float)(rand() % 2000 - 1000) / 100.0f;
        h_input_v[i] = (float)(rand() % 2000 - 1000) / 100.0f;
    }

    // --- CPU reference ---
    std::vector<double> cpu_rotated_k(N), cpu_rotated_v(N);
    for (int i = 0; i < N; ++i) {
        cpu_rotated_k[i] = h_input_k[i];
        cpu_rotated_v[i] = h_input_v[i];
    }
    cpu_hadamard(cpu_rotated_k.data(), N);
    cpu_hadamard(cpu_rotated_v.data(), N);

    // CPU quantize
    std::vector<uint8_t> cpu_nibbles(N);
    double cpu_mse_norm, cpu_qjl_norm;
    cpu_quantize_qjl4(cpu_rotated_k.data(), N, cpu_nibbles.data(), &cpu_mse_norm, &cpu_qjl_norm);

    std::vector<int> cpu_v2_indices(N);
    double cpu_v_norm;
    cpu_quantize_v2(cpu_rotated_v.data(), N, cpu_v2_indices.data(), &cpu_v_norm);

    // CPU dequantize → inv Hadamard → final output
    std::vector<double> cpu_dequant_k(N), cpu_dequant_v(N);
    cpu_dequantize_qjl4(cpu_nibbles.data(), cpu_mse_norm, cpu_qjl_norm, N, cpu_dequant_k.data());
    cpu_dequantize_v2(cpu_v2_indices.data(), cpu_v_norm, N, cpu_dequant_v.data());
    cpu_hadamard(cpu_dequant_k.data(), N);
    cpu_hadamard(cpu_dequant_v.data(), N);

    // --- GPU write path ---
    half *d_input_k, *d_input_v;
    uint32_t *d_packed_k;
    uint16_t *d_packed_v;
    half *d_param_k, *d_param_v;
    cudaMalloc(&d_input_k, N * sizeof(half));
    cudaMalloc(&d_input_v, N * sizeof(half));
    cudaMalloc(&d_packed_k, N_PACKED_K * sizeof(uint32_t));
    cudaMalloc(&d_packed_v, N_PACKED_V * sizeof(uint16_t));
    cudaMalloc(&d_param_k, 2 * sizeof(half));
    cudaMalloc(&d_param_v, 2 * sizeof(half));

    // Upload input as half (stored as uint16_t bits)
    std::vector<uint16_t> h_bits_k(N), h_bits_v(N);
    for (int i = 0; i < N; ++i) {
        half h = __float2half(h_input_k[i]);
        memcpy(&h_bits_k[i], &h, sizeof(uint16_t));
        h = __float2half(h_input_v[i]);
        memcpy(&h_bits_v[i], &h, sizeof(uint16_t));
    }
    cudaMemcpy(d_input_k, h_bits_k.data(), N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_v, h_bits_v.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    // Launch write kernel (1 warp = 32 threads, but only 16 active)
    turboquant_write_kernel<<<1, 32>>>(d_input_k, d_input_v, d_packed_k, d_packed_v,
                                       d_param_k, d_param_v, kHeadDim);
    cudaDeviceSynchronize();

    // Download write results
    cudaMemcpy(h_packed_k.data(), d_packed_k, N_PACKED_K * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_packed_v.data(), d_packed_v, N_PACKED_V * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_param_k, d_param_k, 2 * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_param_v, d_param_v, 2 * sizeof(half), cudaMemcpyDeviceToHost);

    // Debug: dump raw param bits
    uint16_t raw_pk[2], raw_pv[2];
    memcpy(raw_pk, h_param_k, 4);
    memcpy(raw_pv, h_param_v, 4);
    printf("Raw param_k bits: 0x%04x 0x%04x\n", raw_pk[0], raw_pk[1]);
    printf("Raw param_v bits: 0x%04x 0x%04x\n", raw_pv[0], raw_pv[1]);

    // --- Verify write path against CPU reference ---
    // Params are half values — read via uint16_t bits
    uint16_t pk_bits[2], pv_bits[2];
    memcpy(pk_bits, h_param_k, 4);
    memcpy(pv_bits, h_param_v, 4);
    half h_mse_norm, h_qjl_norm, h_v_norm;
    memcpy(&h_mse_norm, &pk_bits[0], sizeof(half));
    memcpy(&h_qjl_norm, &pk_bits[1], sizeof(half));
    memcpy(&h_v_norm, &pv_bits[0], sizeof(half));
    float gpu_mse_norm = __half2float(h_mse_norm);
    float gpu_qjl_norm = __half2float(h_qjl_norm);
    float gpu_v_norm   = __half2float(h_v_norm);

    printf("=== Level 1: Write path verification ===\n");
    printf("K norms: CPU mse=%.6f qjl=%.6f | GPU mse=%.6f qjl=%.6f\n",
           cpu_mse_norm, cpu_qjl_norm, gpu_mse_norm, gpu_qjl_norm);
    printf("V norm:  CPU=%.6f | GPU=%.6f\n", cpu_v_norm, gpu_v_norm);

    bool norm_ok = fabs(cpu_mse_norm - gpu_mse_norm) / (cpu_mse_norm + 1e-6) < 0.01
                && fabs(cpu_qjl_norm - gpu_qjl_norm) / (cpu_qjl_norm + 1e-6) < 0.05
                && fabs(cpu_v_norm - gpu_v_norm) / (cpu_v_norm + 1e-6) < 0.01;

    // Verify packed K data (nibble-level)
    int k_mismatch = 0;
    for (int i = 0; i < N; ++i) {
        int word = i / 8;
        int slot = i % 8;
        uint8_t gpu_nibble = (h_packed_k[word] >> (slot * 4)) & 0xF;
        if (gpu_nibble != cpu_nibbles[i]) k_mismatch++;
    }
    printf("K packed data: %d/%d nibble mismatches vs CPU\n", k_mismatch, N);

    // Verify packed V data (2-bit level)
    int v_mismatch = 0;
    for (int i = 0; i < N; ++i) {
        int word = i / 8;
        int slot = i % 8;
        int gpu_idx2 = (h_packed_v[word] >> (slot * 2)) & 0x3;
        if (gpu_idx2 != cpu_v2_indices[i]) v_mismatch++;
    }
    printf("V packed data: %d/%d index mismatches vs CPU\n", v_mismatch, N);

    bool write_ok = norm_ok && k_mismatch == 0 && v_mismatch == 0;
    printf("Level 1: %s\n\n", write_ok ? "PASS" : "FAIL");

    // --- GPU read path ---
    half *d_output_k, *d_output_v;
    cudaMalloc(&d_output_k, N * sizeof(half));
    cudaMalloc(&d_output_v, N * sizeof(half));

    turboquant_read_kernel<<<1, 32>>>(d_packed_k, d_packed_v, d_param_k, d_param_v,
                                      d_output_k, d_output_v, kHeadDim);
    cudaDeviceSynchronize();

    // Download read results
    std::vector<uint16_t> h_bits_out_k(N), h_bits_out_v(N);
    cudaMemcpy(h_bits_out_k.data(), d_output_k, N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bits_out_v.data(), d_output_v, N * sizeof(half), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        half hk; memcpy(&hk, &h_bits_out_k[i], sizeof(uint16_t));
        half hv; memcpy(&hv, &h_bits_out_v[i], sizeof(uint16_t));
        h_output_k[i] = __half2float(hk);
        h_output_v[i] = __half2float(hv);
    }

    // --- Verify roundtrip ---
    printf("=== Level 3: Roundtrip verification ===\n");
    float max_err_k = 0, max_err_v = 0;
    float sum_sq_orig_k = 0, sum_sq_orig_v = 0;
    float sum_sq_err_k = 0, sum_sq_err_v = 0;
    for (int i = 0; i < N; ++i) {
        float ek = fabs(cpu_dequant_k[i] - h_output_k[i]);
        float ev = fabs(cpu_dequant_v[i] - h_output_v[i]);
        if (ek > max_err_k) max_err_k = ek;
        if (ev > max_err_v) max_err_v = ev;
        sum_sq_err_k += (cpu_dequant_k[i] - h_output_k[i]) * (cpu_dequant_k[i] - h_output_k[i]);
        sum_sq_err_v += (cpu_dequant_v[i] - h_output_v[i]) * (cpu_dequant_v[i] - h_output_v[i]);
        sum_sq_orig_k += cpu_dequant_k[i] * cpu_dequant_k[i];
        sum_sq_orig_v += cpu_dequant_v[i] * cpu_dequant_v[i];
    }
    printf("K: max_err=%.6f  cos_sim=%.6f\n", max_err_k,
           sum_sq_orig_k > 0 ? sum_sq_orig_k / (sum_sq_orig_k + sum_sq_err_k) : 1.0f);
    printf("V: max_err=%.6f  cos_sim=%.6f\n", max_err_v,
           sum_sq_orig_v > 0 ? sum_sq_orig_v / (sum_sq_orig_v + sum_sq_err_v) : 1.0f);

    // Also check vs original input (quantization quality)
    float max_quant_err_k = 0, max_quant_err_v = 0;
    double sum_sq_orig_input_k = 0, sum_sq_quant_err_k = 0;
    double sum_sq_orig_input_v = 0, sum_sq_quant_err_v = 0;
    for (int i = 0; i < N; ++i) {
        double err_k = fabs(h_input_k[i] - cpu_dequant_k[i]);
        double err_v = fabs(h_input_v[i] - cpu_dequant_v[i]);
        if (err_k > max_quant_err_k) max_quant_err_k = err_k;
        if (err_v > max_quant_err_v) max_quant_err_v = err_v;
        sum_sq_orig_input_k += (double)h_input_k[i] * h_input_k[i];
        sum_sq_orig_input_v += (double)h_input_v[i] * h_input_v[i];
        sum_sq_quant_err_k += err_k * err_k;
        sum_sq_quant_err_v += err_v * err_v;
    }
    printf("K quantization quality: max_err=%.4f  cos_sim=%.6f\n", max_quant_err_k,
           sum_sq_orig_input_k > 0 ? sum_sq_orig_input_k / (sum_sq_orig_input_k + sum_sq_quant_err_k) : 1.0);
    printf("V quantization quality: max_err=%.4f  cos_sim=%.6f\n", max_quant_err_v,
           sum_sq_orig_input_v > 0 ? sum_sq_orig_input_v / (sum_sq_orig_input_v + sum_sq_quant_err_v) : 1.0);

    bool roundtrip_ok = max_err_k < 0.01 && max_err_v < 0.01;
    printf("Level 3: %s\n", roundtrip_ok ? "PASS" : "FAIL");

    // Cleanup
    cudaFree(d_input_k); cudaFree(d_input_v);
    cudaFree(d_packed_k); cudaFree(d_packed_v);
    cudaFree(d_param_k); cudaFree(d_param_v);
    cudaFree(d_output_k); cudaFree(d_output_v);

    return (write_ok && roundtrip_ok) ? 0 : 1;
}