// Copyright (c) OpenMMLab. All rights reserved.
//
// This file contains code derived from fast-hadamard-transform by Tri Dao.
// Original copyright notice:
//
// ******************************************************************************
// * Copyright (c) 2023, Tri Dao.
// ******************************************************************************

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace turbomind {

struct HadamardParams {
    int     batch;
    int     dim;
    int     log_dim;  // log2(dim), must be integer (dim must be power of 2)
    int64_t stride;   // stride between consecutive vectors (in elements)
    float   scale;    // normalization factor: 1/sqrt(dim) for orthogonal Hadamard
    void*   x_ptr;    // input  [batch, dim], stride between rows
    void*   out_ptr;  // output [batch, dim], stride between rows
};

// Apply Walsh-Hadamard transform: y = scale * H @ x
// H is the Sylvester Hadamard matrix, scale = 1/sqrt(dim) makes it orthogonal.
// Since H is symmetric and Q = H/sqrt(d) is orthogonal, Q^{-1} = Q^T = Q,
// so the same function serves as both forward and inverse transform.
//
// Supported types: float, half, nv_bfloat16
// dim must be a power of 2 in [8, 4096]

void hadamard_transform_fp16(HadamardParams& params, cudaStream_t stream);
void hadamard_transform_bf16(HadamardParams& params, cudaStream_t stream);
void hadamard_transform_fp32(HadamardParams& params, cudaStream_t stream);

}  // namespace turbomind
