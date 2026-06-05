// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace turbomind::attention {

// ---------------------------------------------------------------------------
// TurboQuant dequantizer policies for KvQuantTrait::DequantK / DequantV
// ---------------------------------------------------------------------------

// QJL4 K dequantizer: 3-bit Lloyd-Max codebook + 1-bit QJL sign
//   val = nibble [0..15], bits[2:0]=MSE index, bit[3]=QJL sign
//   p0 = mse_norm, p1 = qjl_norm
//   result = (centroids_3bit[idx3] * sigma + qjl_norm * sign) * mse_norm

template<typename T>
struct TurboDequantK;

template<>
struct TurboDequantK<half> {
    __device__ static half apply(half val, half mse_norm, half qjl_norm, int head_dim)
    {
        constexpr float c_std[8] = {
            -2.1519456f, -1.3439092f, -0.7560052f, -0.2450942f, 0.2450942f, 0.7560052f, 1.3439092f, 2.1519456f};
        float sigma    = 1.0f / sqrtf((float)head_dim);
        int   nibble   = (int)__half2float(val);
        int   idx3     = nibble & 0x7;
        int   sign_bit = (nibble >> 3) & 0x1;
        float centroid = c_std[idx3] * sigma;
        float sign_val = sign_bit * 2.0f - 1.0f;
        float y_hat    = centroid + __half2float(qjl_norm) * sign_val;
        return __float2half(y_hat * __half2float(mse_norm));
    }
};

template<>
struct TurboDequantK<nv_bfloat16> {
    __device__ static nv_bfloat16 apply(nv_bfloat16 val, nv_bfloat16 mse_norm, nv_bfloat16 qjl_norm, int head_dim)
    {
        constexpr float c_std[8] = {
            -2.1519456f, -1.3439092f, -0.7560052f, -0.2450942f, 0.2450942f, 0.7560052f, 1.3439092f, 2.1519456f};
        float sigma    = 1.0f / sqrtf((float)head_dim);
        int   nibble   = (int)__bfloat162float(val);
        int   idx3     = nibble & 0x7;
        int   sign_bit = (nibble >> 3) & 0x1;
        float centroid = c_std[idx3] * sigma;
        float sign_val = sign_bit * 2.0f - 1.0f;
        float y_hat    = centroid + __bfloat162float(qjl_norm) * sign_val;
        return __float2bfloat16(y_hat * __bfloat162float(mse_norm));
    }
};

template<>
struct TurboDequantK<float> {
    __device__ static float apply(float val, float mse_norm, float qjl_norm, int head_dim)
    {
        constexpr float c_std[8] = {
            -2.1519456f, -1.3439092f, -0.7560052f, -0.2450942f, 0.2450942f, 0.7560052f, 1.3439092f, 2.1519456f};
        float sigma    = 1.0f / sqrtf((float)head_dim);
        int   nibble   = (int)val;
        int   idx3     = nibble & 0x7;
        int   sign_bit = (nibble >> 3) & 0x1;
        float centroid = c_std[idx3] * sigma;
        float sign_val = sign_bit * 2.0f - 1.0f;
        return (centroid + qjl_norm * sign_val) * mse_norm;
    }
};

// 2-bit V dequantizer: Lloyd-Max codebook lookup
//   val = 2-bit index [0..3], p0 = norm, p1 = unused

template<typename T>
struct TurboDequantV;

template<>
struct TurboDequantV<half> {
    __device__ static half apply(half val, half norm, half, int head_dim)
    {
        constexpr float c_std[4] = {-1.5104176f, -0.4527808f, 0.4527808f, 1.5104176f};
        float           sigma    = 1.0f / sqrtf((float)head_dim);
        int             idx      = (int)__half2float(val);
        float           y_hat    = c_std[idx] * sigma;
        return __float2half(y_hat * __half2float(norm));
    }
};

template<>
struct TurboDequantV<nv_bfloat16> {
    __device__ static nv_bfloat16 apply(nv_bfloat16 val, nv_bfloat16 norm, nv_bfloat16, int head_dim)
    {
        constexpr float c_std[4] = {-1.5104176f, -0.4527808f, 0.4527808f, 1.5104176f};
        float           sigma    = 1.0f / sqrtf((float)head_dim);
        int             idx      = (int)__bfloat162float(val);
        float           y_hat    = c_std[idx] * sigma;
        return __float2bfloat16(y_hat * __bfloat162float(norm));
    }
};

template<>
struct TurboDequantV<float> {
    __device__ static float apply(float val, float norm, float, int head_dim)
    {
        constexpr float c_std[4] = {-1.5104176f, -0.4527808f, 0.4527808f, 1.5104176f};
        float           sigma    = 1.0f / sqrtf((float)head_dim);
        int             idx      = (int)val;
        float           y_hat    = c_std[idx] * sigma;
        return y_hat * norm;
    }
};

}  // namespace turbomind::attention
