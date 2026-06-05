// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cmath>

namespace turbomind::attention {

// ---------------------------------------------------------------------------
// Lloyd-Max codebook for 3-bit K cache quantization (TurboQuant QJL4)
// ---------------------------------------------------------------------------
// QJL4 = 3-bit Lloyd-Max MSE index + 1-bit QJL residual sign, packed as 4-bit nibble.
//
// NOTE: __constant__ arrays cannot be indexed with runtime variables in
// cross-compilation-unit scenarios. The inline helpers below use literal
// index access. For convenience in kernels, use the kK4CentroidStd_N /
// kK4BoundaryStd_N named constants directly.

// 3-bit centroids at σ=1 (individual named constants for portable kernel use)
__device__ __constant__ static const float kK4CentroidStd_0 = -2.1519456f;
__device__ __constant__ static const float kK4CentroidStd_1 = -1.3439092f;
__device__ __constant__ static const float kK4CentroidStd_2 = -0.7560052f;
__device__ __constant__ static const float kK4CentroidStd_3 = -0.2450942f;
__device__ __constant__ static const float kK4CentroidStd_4 = 0.2450942f;
__device__ __constant__ static const float kK4CentroidStd_5 = 0.7560052f;
__device__ __constant__ static const float kK4CentroidStd_6 = 1.3439092f;
__device__ __constant__ static const float kK4CentroidStd_7 = 2.1519456f;

// 3-bit boundaries at σ=1 (individual named constants)
__device__ __constant__ static const float kK4BoundaryStd_0 = -1.7479274f;
__device__ __constant__ static const float kK4BoundaryStd_1 = -1.0499572f;
__device__ __constant__ static const float kK4BoundaryStd_2 = -0.5005497f;
__device__ __constant__ static const float kK4BoundaryStd_3 = 0.0000000f;
__device__ __constant__ static const float kK4BoundaryStd_4 = 0.5005497f;
__device__ __constant__ static const float kK4BoundaryStd_5 = 1.0499572f;
__device__ __constant__ static const float kK4BoundaryStd_6 = 1.7479274f;

// ---------------------------------------------------------------------------
// Inline device helpers for QJL4 quantization / dequantization
// ---------------------------------------------------------------------------

/// Find nearest 3-bit centroid index via boundary comparison.
/// y = Hadamard-rotated normalized value, sigma = 1/sqrt(d)
/// Returns index in [0, 7]
__device__ __forceinline__ int quantize_k4_mse(float y, float sigma)
{
    return (y > kK4BoundaryStd_0 * sigma) + (y > kK4BoundaryStd_1 * sigma) + (y > kK4BoundaryStd_2 * sigma)
           + (y > kK4BoundaryStd_3 * sigma) + (y > kK4BoundaryStd_4 * sigma) + (y > kK4BoundaryStd_5 * sigma)
           + (y > kK4BoundaryStd_6 * sigma);
}

/// Look up 3-bit centroid value by index (scalar version).
__device__ __forceinline__ float dequantize_k4_mse(int idx, float sigma)
{
    float v = (idx == 0) ? kK4CentroidStd_0 :
              (idx == 1) ? kK4CentroidStd_1 :
              (idx == 2) ? kK4CentroidStd_2 :
              (idx == 3) ? kK4CentroidStd_3 :
              (idx == 4) ? kK4CentroidStd_4 :
              (idx == 5) ? kK4CentroidStd_5 :
              (idx == 6) ? kK4CentroidStd_6 :
                           kK4CentroidStd_7;
    return v * sigma;
}

/// Pack two QJL4 nibbles (idx3 | sign<<3) into one byte.
/// nibble_lo occupies bits [3:0], nibble_hi bits [7:4]
__device__ __forceinline__ uint8_t pack_k4(int idx3_lo, int sign_lo, int idx3_hi, int sign_hi)
{
    uint8_t nib_lo = static_cast<uint8_t>(idx3_lo | (sign_lo << 3));
    uint8_t nib_hi = static_cast<uint8_t>(idx3_hi | (sign_hi << 3));
    return nib_lo | (nib_hi << 4);
}

/// Unpack QJL4 nibble from a byte at the given half position (0=low, 1=high).
/// Returns (idx3, sign_bit)
__device__ __forceinline__ void unpack_k4(uint8_t packed, int half, int& idx3, int& sign_bit)
{
    uint8_t nibble = (packed >> (half * 4)) & 0x0F;
    idx3           = nibble & 0x7;
    sign_bit       = (nibble >> 3) & 0x1;
}

}  // namespace turbomind::attention
