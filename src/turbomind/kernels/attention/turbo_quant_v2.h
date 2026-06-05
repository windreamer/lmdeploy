// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cmath>

namespace turbomind::attention {

// ---------------------------------------------------------------------------
// Lloyd-Max codebook for 2-bit V cache quantization (TurboQuant)
// ---------------------------------------------------------------------------
// These are the optimal centroids and boundaries for a unit-variance (σ=1)
// Gaussian source, computed offline via the Lloyd-Max algorithm.
//
// Runtime usage:
//   float sigma = 1.0f / sqrtf((float)head_dim);
//   float centroids[4] = {kV2CentroidsStd[0] * sigma, ...};
//   float boundaries[3] = {kV2BoundariesStd[0] * sigma, ...};
//
// Quantization: idx = (u > b[0]) + (u > b[1]) + (u > b[2])
// Dequantization: val = centroids[idx] * sigma * norm

// 2-bit centroids at σ=1: [-1.5104176, -0.4527808, 0.4527808, 1.5104176]
__device__ __constant__ static const float kV2CentroidsStd[4] = {
    -1.5104176f,
    -0.4527808f,
    0.4527808f,
    1.5104176f,
};

// 2-bit boundaries at σ=1: [-0.9815992, 0.0, 0.9815992]
__device__ __constant__ static const float kV2BoundariesStd[3] = {
    -0.9815992f,
    0.0f,
    0.9815992f,
};

// ---------------------------------------------------------------------------
// Inline device helpers for 2-bit quantization / dequantization
// ---------------------------------------------------------------------------

/// Find nearest 2-bit centroid index via boundary comparison.
/// u = normalized value (on unit sphere), sigma = 1/sqrt(d)
/// Returns index in [0, 3]
__device__ __forceinline__ int quantize_v2(float u, float sigma)
{
    const float b0 = kV2BoundariesStd[0] * sigma;
    const float b1 = kV2BoundariesStd[1] * sigma;
    const float b2 = kV2BoundariesStd[2] * sigma;
    return (u > b0) + (u > b1) + (u > b2);
}

/// Look up 2-bit centroid value by index.
/// Returns centroid * sigma (still normalized, caller multiplies by norm)
__device__ __forceinline__ float dequantize_v2(int idx, float sigma)
{
    return kV2CentroidsStd[idx] * sigma;
}

/// Pack four 2-bit indices into one byte.
/// idx0 occupies bits [1:0], idx1 bits [3:2], idx2 bits [5:4], idx3 bits [7:6]
__device__ __forceinline__ uint8_t pack_v2(int idx0, int idx1, int idx2, int idx3)
{
    return static_cast<uint8_t>(idx0 | (idx1 << 2) | (idx2 << 4) | (idx3 << 6));
}

/// Unpack 2-bit index from a packed byte at the given group position (0-3).
__device__ __forceinline__ int unpack_v2(uint8_t packed, int group)
{
    return (packed >> (group * 2)) & 0x3;
}

}  // namespace turbomind::attention
