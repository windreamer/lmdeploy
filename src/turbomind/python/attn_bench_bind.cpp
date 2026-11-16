// Copyright (c) OpenMMLab. All rights reserved.

// Minimal pybind surface for kernel-level decode-attention benchmarks.
// Exposes one fixed-GQA decode entry that fills `AttentionParams` the same way
// `UnifiedAttentionLayer` does and runs `dispatchDecoding` in a timed loop.
// The block pool is caller-owned raw device memory (TM per-block layout); the
// host builds the `char**` pointer array here, mirroring
// `unified_attention_layer.cc` `Setup()`.

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <vector>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/attention/attention_params.h"
#include "src/turbomind/kernels/attention/decoding.h"
#include "src/turbomind/models/llama/llama_rope.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace turbomind::python_attn_bench {
namespace {

template<class T>
std::vector<float> RunDecodeBench(const core::Tensor& q,
                                  core::Tensor&       out,
                                  char**              block_ptrs,
                                  const int*          cu_block_nums,
                                  int                 num_heads,
                                  int                 num_kv_heads,
                                  int                 head_dim,
                                  int                 batch_size,
                                  int                 block_len,
                                  int                 max_k_len,
                                  int                 quant_policy,
                                  int                 iters,
                                  int                 warmup,
                                  cudaStream_t        stream)
{
    AttentionParams<T> params{};

    // Mirror the production layout: one fused [B, (H + 2Hkv) * D] qkv row per
    // token; Q is copied in, K/V pointers stay valid but are never dereferenced
    // by the decoding path (KV comes from the block pool).
    core::Tensor qkv{{batch_size, (num_heads + 2 * num_kv_heads) * head_dim}, q.dtype(), q.device()};
    TM_CUDA_CHECK(cudaMemcpy2DAsync(qkv.raw_data(),
                                    qkv.stride(0) * sizeof(T),
                                    q.raw_data(),
                                    num_heads * head_dim * sizeof(T),
                                    num_heads * head_dim * sizeof(T),
                                    batch_size,
                                    cudaMemcpyDeviceToDevice,
                                    stream));

    params.out    = (T*)out.raw_data();
    params.q      = (T*)qkv.raw_data();
    params.k      = params.q + num_heads * head_dim;
    params.v      = params.k + num_kv_heads * head_dim;
    params.stride = (num_heads + 2 * num_kv_heads) * head_dim;

    params.batch_size = batch_size;
    params.token_num  = batch_size;  // decode: one query token per sequence
    params.max_q_len  = 1;
    params.max_k_len  = max_k_len;

    params.block_iter_params = BlockIteratorParams{block_ptrs,  //
                                                   cu_block_nums,
                                                   0,  // cache_block_offset
                                                   block_len};

    params.finished           = nullptr;
    params.cu_q_len           = nullptr;
    params.cu_k_len           = nullptr;
    params.readonly_block_num = nullptr;
    params.token_mask         = nullptr;

    params.num_heads     = num_heads;
    params.num_kv_heads  = num_kv_heads;
    params.size_per_head = head_dim;
    params.inv_sqrt_dh   = (float)std::log2(expf(1.f)) / std::sqrt((float)head_dim);

    params.window_size = 256 << 20;

    params.rope_param      = RopeKernelParam{};
    params.rope_param.type = RopeType::kNull;  // Q is pre-rotated by the caller

    // Per-sequence lengths are constants in this benchmark; build them once on
    // device, outside the timed region.
    core::Tensor_<bool> finished{{batch_size}, kDEVICE};
    core::Tensor_<int>  cu_q_len{{batch_size + 1}, kDEVICE};
    core::Tensor_<int>  cu_k_len{{batch_size + 1}, kDEVICE};
    {
        std::vector<char> f(batch_size, 0);
        std::vector<int>  qo(batch_size + 1), ko(batch_size + 1);
        for (int i = 0; i <= batch_size; ++i) {
            qo[i] = i;
            ko[i] = i * max_k_len;
        }
        TM_CUDA_CHECK(cudaMemcpyAsync(finished.raw_data(), f.data(), f.size(), cudaMemcpyHostToDevice, stream));
        TM_CUDA_CHECK(
            cudaMemcpyAsync(cu_q_len.raw_data(), qo.data(), qo.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
        TM_CUDA_CHECK(
            cudaMemcpyAsync(cu_k_len.raw_data(), ko.data(), ko.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    }

    // Split-K workspace, sized like `UnifiedAttentionLayer` (kMaxWorkspaceTokens budget).
    constexpr int        kMaxSplitK          = 128;
    constexpr int        kMaxWorkspaceTokens = 4096;
    const int            max_split_k = std::min(std::max(1, kMaxWorkspaceTokens / params.token_num), kMaxSplitK);
    core::Tensor_<int>   split_cnt{{params.token_num}, kDEVICE};
    core::Tensor_<float> partial_O{{params.token_num, num_heads, max_split_k, head_dim}, kDEVICE};
    core::Tensor_<float> partial_ML{{params.token_num, num_heads, max_split_k, 2}, kDEVICE};
    TM_CUDA_CHECK(cudaMemsetAsync(split_cnt.raw_data(), 0, sizeof(int) * params.token_num, stream));

    params.cu_q_len     = cu_q_len.data();
    params.cu_k_len     = cu_k_len.data();
    params.finished     = finished.data();
    params.split_cnt    = split_cnt.data();
    params.partial_ML   = partial_ML.data();
    params.partial_O    = partial_O.data();
    params.max_split_k  = max_split_k;
    params.arch         = getSMVersion();
    params.stream       = stream;
    params.quant_policy = quant_policy;

    std::vector<cudaEvent_t> beg(iters), end(iters);
    for (int i = 0; i < iters; ++i) {
        TM_CUDA_CHECK(cudaEventCreate(&beg[i]));
        TM_CUDA_CHECK(cudaEventCreate(&end[i]));
    }

    for (int i = 0; i < warmup; ++i) {
        dispatchDecoding<T>(params);
    }
    TM_CUDA_CHECK(cudaStreamSynchronize(stream));
    TM_CUDA_CHECK(cudaGetLastError());

    for (int i = 0; i < iters; ++i) {
        TM_CUDA_CHECK(cudaEventRecord(beg[i], stream));
        dispatchDecoding<T>(params);
        TM_CUDA_CHECK(cudaEventRecord(end[i], stream));
    }
    TM_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> ms(iters);
    for (int i = 0; i < iters; ++i) {
        TM_CUDA_CHECK(cudaEventElapsedTime(&ms[i], beg[i], end[i]));
        TM_CUDA_CHECK(cudaEventDestroy(beg[i]));
        TM_CUDA_CHECK(cudaEventDestroy(end[i]));
    }
    return ms;
}

py::list AttnDecodeBench(const std::shared_ptr<core::Tensor>& q,
                         std::shared_ptr<core::Tensor>        out,
                         std::uintptr_t                       block_ptrs_addr,
                         std::uintptr_t                       cu_block_nums_addr,
                         int                                  num_heads,
                         int                                  num_kv_heads,
                         int                                  head_dim,
                         int                                  batch_size,
                         int                                  block_len,
                         int                                  max_k_len,
                         int                                  kv_quant,
                         int                                  iters,
                         int                                  warmup,
                         std::uintptr_t                       stream_addr)
{
    TM_CHECK_EQ(q->shape(0), batch_size) << "q rows must equal batch_size";
    TM_CHECK_EQ(q->size(), (size_t)batch_size * num_heads * head_dim) << "q must be [B, H*D] contiguous";

    auto         stream = stream_addr ? (cudaStream_t)stream_addr : core::Context::stream().handle();
    core::Tensor out_t  = out ? *out : core::Tensor{{batch_size, num_heads * head_dim}, q->dtype(), q->device()};

    char**     block_ptrs    = (char**)block_ptrs_addr;
    const int* cu_block_nums = (const int*)cu_block_nums_addr;

    int quant_policy = 0;
    if (kv_quant == 8) {
        quant_policy = QuantPolicy::kCacheKVInt8;
    }
    else if (kv_quant == 4) {
        quant_policy = QuantPolicy::kCacheKVInt4;
    }
    TM_CHECK(kv_quant == 0 || kv_quant == 4 || kv_quant == 8) << "kv_quant must be one of 0, 4, 8";

    std::vector<float> ms;
    {
        py::gil_scoped_release release;
        if (q->dtype() == kBfloat16) {
            ms = RunDecodeBench<nv_bfloat16>(*q,
                                             out_t,
                                             block_ptrs,
                                             cu_block_nums,
                                             num_heads,
                                             num_kv_heads,
                                             head_dim,
                                             batch_size,
                                             block_len,
                                             max_k_len,
                                             quant_policy,
                                             iters,
                                             warmup,
                                             stream);
        }
        else if (q->dtype() == kFloat16) {
            ms = RunDecodeBench<half>(*q,
                                      out_t,
                                      block_ptrs,
                                      cu_block_nums,
                                      num_heads,
                                      num_kv_heads,
                                      head_dim,
                                      batch_size,
                                      block_len,
                                      max_k_len,
                                      quant_policy,
                                      iters,
                                      warmup,
                                      stream);
        }
        else {
            TM_CHECK(0) << "q dtype must be bf16 or fp16, got: " << to_string(q->dtype());
        }
    }

    py::list ret;
    for (float v : ms) {
        ret.append(v);
    }
    return ret;
}

}  // namespace

void bind_attn_bench(py::module_& m)
{
    m.def("attn_decode_bench",
          &AttnDecodeBench,
          "q"_a,
          "out"_a = py::none(),
          "block_ptrs_addr"_a,
          "cu_block_nums_addr"_a,
          "num_heads"_a,
          "num_kv_heads"_a,
          "head_dim"_a,
          "batch_size"_a,
          "block_len"_a,
          "max_k_len"_a,
          "kv_quant"_a   = 0,
          "iters"_a      = 100,
          "warmup"_a     = 20,
          "stream_ptr"_a = std::uintptr_t{0},
          "Timed decode-attention dispatch; returns per-iteration ms.");
}

}  // namespace turbomind::python_attn_bench
