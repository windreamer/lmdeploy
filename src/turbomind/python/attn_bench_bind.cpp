// Copyright (c) OpenMMLab. All rights reserved.

// Minimal pybind surface for kernel-level decode-attention benchmarks plus
// the production entries used by the PyTorch-engine int8-KV graft:
// - `attn_decode_bench`: timed decode dispatch (benchmark only).
// - `tm_attn_decode`:    single-shot decode dispatch on caller-owned blocks.
// - `tm_process_kv`:     quantizing KV append (`invokeProcessKV_v2`, null
//                        rope) into caller-owned TM blocks.
// The block pool is caller-owned raw device memory (TM per-block layout); the
// host builds the `char**` pointer array here, mirroring
// `unified_attention_layer.cc` `Setup()`. Production entries push a
// `ContextGuard` over the caller's stream so core allocations work without an
// ambient TurboMind context.

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <vector>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/stream.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/attention/attention_params.h"
#include "src/turbomind/kernels/attention/decoding.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/models/llama/llama_rope.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace turbomind::python_attn_bench {
namespace {

// Workspace tensors and dispatch params for one decode launch. The members
// must outlive the kernel: `params` references their raw storage.
template<class T>
struct DecodePlan {
    AttentionParams<T>   params{};
    core::Tensor         qkv;
    core::Tensor_<bool>  finished;
    core::Tensor_<int>   cu_q_len;
    core::Tensor_<int>   cu_k_len;
    core::Tensor_<int>   readonly_block_num;
    core::Tensor_<int>   split_cnt;
    core::Tensor_<float> partial_O;
    core::Tensor_<float> partial_ML;
};

// Deterministic byte layout of the graph-path decode workspace. Every plan
// buffer lives at a fixed offset inside one caller-owned persistent buffer,
// so a capture-time dispatch records **zero** allocation nodes (the eager
// path's per-call `cudaMallocAsync` plan buffers showed up as 8 free / 0
// alloc nodes and made replay consume freed memory). Offsets are 256-byte
// aligned and identical for both `tm_decode_workspace_bytes` (Python-side
// sizing) and `TmAttnDecodeGraph` (dispatch), by construction.
struct DecodeWorkspaceLayout {
    size_t qkv;
    size_t finished;
    size_t readonly_block_num;
    size_t split_cnt;
    size_t partial_O;
    size_t partial_ML;
    size_t total;

    static constexpr size_t Align(size_t x)
    {
        return (x + 255) & ~(size_t)255;
    }

    // `elem_size` is sizeof(q dtype); `max_split_k` follows the eager path's
    // `kMaxWorkspaceTokens` budget exactly.
    static int MaxSplitK(int batch_size)
    {
        constexpr int kMaxSplitK          = 128;
        constexpr int kMaxWorkspaceTokens = 4096;
        return std::min(std::max(1, kMaxWorkspaceTokens / batch_size), kMaxSplitK);
    }

    static DecodeWorkspaceLayout Make(int batch_size, int num_heads, int num_kv_heads, int head_dim, size_t elem_size)
    {
        DecodeWorkspaceLayout l{};
        const int             max_split_k = MaxSplitK(batch_size);
        const size_t          row         = (size_t)(num_heads + 2 * num_kv_heads) * head_dim * elem_size;
        l.qkv                             = 0;
        l.finished                        = Align(l.qkv + row * batch_size);
        l.readonly_block_num              = Align(l.finished + batch_size * sizeof(bool));
        l.split_cnt                       = Align(l.readonly_block_num + batch_size * sizeof(int));
        l.partial_O                       = Align(l.split_cnt + batch_size * sizeof(int));
        l.partial_ML = Align(l.partial_O + (size_t)batch_size * num_heads * max_split_k * head_dim * sizeof(float));
        l.total      = Align(l.partial_ML + (size_t)batch_size * num_heads * max_split_k * 2 * sizeof(float));
        return l;
    }
};

// Fill `AttentionParams` the same way `UnifiedAttentionLayer` does (decode
// path, `CreateParams`). `q`/`k`/`v` are [B, H*D] / [B, Hkv*D] /
// [B, Hkv*D] contiguous and must hold the caller's real values: the decoding
// kernel (`attention_universal.h`, `kProcessKV`) re-stores the K/V rows
// (rope'd, quantized) into the block pool at `ti = history_len` every
// dispatch, so garbage in `k`/`v` is quantized and written into the pool's
// last valid position. `readonly_block_num` is a valid zero buffer, mirroring
// the production decode convention (`Scheduler::PlanContinue` sets it to 0).
// Null `cu_q_len_arg` / `cu_k_len_arg` fall back to the benchmark's constant
// per-sequence lengths.
template<class T>
DecodePlan<T> BuildDecodePlan(const core::Tensor& q,
                              const core::Tensor& k,
                              const core::Tensor& v,
                              core::Tensor&       out,
                              char**              block_ptrs,
                              const int*          cu_block_nums,
                              const int*          cu_q_len_arg,
                              const int*          cu_k_len_arg,
                              int                 num_heads,
                              int                 num_kv_heads,
                              int                 head_dim,
                              int                 batch_size,
                              int                 block_len,
                              int                 max_k_len,
                              int                 quant_policy,
                              cudaStream_t        stream)
{
    DecodePlan<T>       plan;
    AttentionParams<T>& params = plan.params;

    // Mirror the production layout: one fused [B, (H + 2Hkv) * D] qkv row per
    // token, with Q, K and V all populated from caller data (production points
    // `params.k/v` into the real fused qkv GEMM output,
    // `unified_attention_layer.cc` CreateParams). The K/V sections are
    // dereferenced and re-stored into the block pool by the decoding kernel.
    {
        TM_CHECK_EQ(k.size(), (size_t)batch_size * num_kv_heads * head_dim) << "k must be [B, Hkv*D] contiguous";
        TM_CHECK_EQ(v.size(), (size_t)batch_size * num_kv_heads * head_dim) << "v must be [B, Hkv*D] contiguous";
        TM_CHECK_EQ(k.dtype(), q.dtype()) << "k dtype must match q";
        TM_CHECK_EQ(v.dtype(), q.dtype()) << "v dtype must match q";

        plan.qkv = core::Tensor{{batch_size, (num_heads + 2 * num_kv_heads) * head_dim}, q.dtype(), q.device()};
        const size_t q_bytes   = num_heads * head_dim * sizeof(T);
        const size_t kv_bytes  = num_kv_heads * head_dim * sizeof(T);
        const size_t row_bytes = (num_heads + 2 * num_kv_heads) * head_dim * sizeof(T);
        char*        qkv_base  = (char*)plan.qkv.raw_data();
        TM_CUDA_CHECK(cudaMemcpy2DAsync(
            qkv_base, row_bytes, q.raw_data(), q_bytes, q_bytes, batch_size, cudaMemcpyDeviceToDevice, stream));
        TM_CUDA_CHECK(cudaMemcpy2DAsync(qkv_base + q_bytes,
                                        row_bytes,
                                        k.raw_data(),
                                        kv_bytes,
                                        kv_bytes,
                                        batch_size,
                                        cudaMemcpyDeviceToDevice,
                                        stream));
        TM_CUDA_CHECK(cudaMemcpy2DAsync(qkv_base + q_bytes + kv_bytes,
                                        row_bytes,
                                        v.raw_data(),
                                        kv_bytes,
                                        kv_bytes,
                                        batch_size,
                                        cudaMemcpyDeviceToDevice,
                                        stream));
    }

    params.out    = (T*)out.raw_data();
    params.q      = (T*)plan.qkv.raw_data();
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

    params.finished   = nullptr;
    params.cu_q_len   = nullptr;
    params.cu_k_len   = nullptr;
    params.token_mask = nullptr;

    params.num_heads     = num_heads;
    params.num_kv_heads  = num_kv_heads;
    params.size_per_head = head_dim;
    params.inv_sqrt_dh   = (float)std::log2(expf(1.f)) / std::sqrt((float)head_dim);

    params.window_size = 256 << 20;

    params.rope_param      = RopeKernelParam{};
    params.rope_param.type = RopeType::kNull;  // Q is pre-rotated by the caller

    // Per-sequence lengths are either caller-provided device arrays or
    // benchmark constants built once on device.
    plan.finished           = core::Tensor_<bool>{{batch_size}, kDEVICE};
    plan.cu_q_len           = core::Tensor_<int>{{batch_size + 1}, kDEVICE};
    plan.cu_k_len           = core::Tensor_<int>{{batch_size + 1}, kDEVICE};
    plan.readonly_block_num = core::Tensor_<int>{{batch_size}, kDEVICE};
    {
        std::vector<char> f(batch_size, 0);
        std::vector<int>  qo(batch_size + 1), ko(batch_size + 1);
        for (int i = 0; i <= batch_size; ++i) {
            qo[i] = i;
            ko[i] = i * max_k_len;
        }
        // Per-call semantics: this eager entry is never captured; the stack
        // H2D below is legal because the stack outlives the dispatch. The
        // CUDA-graph-safe entry (`TmAttnDecodeGraph`/`DispatchDecodeGraph`)
        // carries the capture contract with memsets instead.
        TM_CUDA_CHECK(cudaMemcpyAsync(plan.finished.raw_data(), f.data(), f.size(), cudaMemcpyHostToDevice, stream));
        // Decode always appends past the read-only prefix (production decode
        // sets `readonly_block_num = 0`); a valid zero buffer keeps that
        // contract explicit instead of relying on a null-pointer special case.
        TM_CUDA_CHECK(cudaMemsetAsync(plan.readonly_block_num.raw_data(), 0, sizeof(int) * batch_size, stream));
        if (cu_q_len_arg == nullptr) {
            TM_CUDA_CHECK(cudaMemcpyAsync(
                plan.cu_q_len.raw_data(), qo.data(), qo.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
        }
        else {
            TM_CUDA_CHECK(cudaMemcpyAsync(
                plan.cu_q_len.raw_data(), cu_q_len_arg, qo.size() * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        }
        if (cu_k_len_arg == nullptr) {
            TM_CUDA_CHECK(cudaMemcpyAsync(
                plan.cu_k_len.raw_data(), ko.data(), ko.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
        }
        else {
            TM_CUDA_CHECK(cudaMemcpyAsync(
                plan.cu_k_len.raw_data(), cu_k_len_arg, ko.size() * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        }
    }

    // Split-K workspace, sized like `UnifiedAttentionLayer` (kMaxWorkspaceTokens budget).
    constexpr int kMaxSplitK          = 128;
    constexpr int kMaxWorkspaceTokens = 4096;
    const int     max_split_k         = std::min(std::max(1, kMaxWorkspaceTokens / params.token_num), kMaxSplitK);
    plan.split_cnt                    = core::Tensor_<int>{{params.token_num}, kDEVICE};
    plan.partial_O  = core::Tensor_<float>{{params.token_num, num_heads, max_split_k, head_dim}, kDEVICE};
    plan.partial_ML = core::Tensor_<float>{{params.token_num, num_heads, max_split_k, 2}, kDEVICE};
    TM_CUDA_CHECK(cudaMemsetAsync(plan.split_cnt.raw_data(), 0, sizeof(int) * params.token_num, stream));

    params.cu_q_len           = plan.cu_q_len.data();
    params.cu_k_len           = plan.cu_k_len.data();
    params.readonly_block_num = plan.readonly_block_num.data();
    params.finished           = plan.finished.data();
    params.split_cnt          = plan.split_cnt.data();
    params.partial_ML         = plan.partial_ML.data();
    params.partial_O          = plan.partial_O.data();
    params.max_split_k        = max_split_k;
    params.arch               = getSMVersion();
    params.stream             = stream;
    params.quant_policy       = quant_policy;

    return plan;
}

template<class T>
std::vector<float> RunDecodeBench(const core::Tensor& q,
                                  const core::Tensor& k,
                                  const core::Tensor& v,
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
    DecodePlan<T> plan = BuildDecodePlan<T>(q,
                                            k,
                                            v,
                                            out,
                                            block_ptrs,
                                            cu_block_nums,
                                            nullptr,
                                            nullptr,
                                            num_heads,
                                            num_kv_heads,
                                            head_dim,
                                            batch_size,
                                            block_len,
                                            max_k_len,
                                            quant_policy,
                                            stream);

    std::vector<cudaEvent_t> beg(iters), end(iters);
    for (int i = 0; i < iters; ++i) {
        TM_CUDA_CHECK(cudaEventCreate(&beg[i]));
        TM_CUDA_CHECK(cudaEventCreate(&end[i]));
    }

    for (int i = 0; i < warmup; ++i) {
        dispatchDecoding<T>(plan.params);
    }
    TM_CUDA_CHECK(cudaStreamSynchronize(stream));
    TM_CUDA_CHECK(cudaGetLastError());

    for (int i = 0; i < iters; ++i) {
        TM_CUDA_CHECK(cudaEventRecord(beg[i], stream));
        dispatchDecoding<T>(plan.params);
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

// CUDA-graph-safe single-shot decode. Contract with the Python caller (all
// buffers allocated and filled OUTSIDE/INSIDE capture but always persistent
// across replays):
// - `q`/`k`/`v`: persistent device rows whose CONTENT is rewritten per step
//   (captured D2D memcpy2D nodes re-read their current contents on replay);
// - `cu_q_len`/`cu_k_len`: caller-owned device int32[B+1] cumulative
//   lengths; the params point at them directly, so per-sequence lengths are
//   read by the kernel at run time (this is what makes prefix growth and
//   per-seq masking representable inside a graph — defect ③ of the audit);
// - `workspace`: caller-owned persistent byte buffer laid out per
//   `DecodeWorkspaceLayout`; the dispatch performs no allocation and no
//   free, so capture records zero alloc/free nodes (defect ②);
// - no host-stack sources are captured: `finished`/`readonly_block_num`/
//   `split_cnt` are captured memset nodes and `cu_*_len` are caller buffers
//   (defect ①).
// `max_k_len` is the CAPACITY bound frozen at capture (grid/split schedule
// is derived from it host-side); correctness for shorter sequences comes
// from the device `cu_k_len`, which the kernel consumes per sequence
// (`attention_universal.h` reads `cu_k_len` at run time).
template<class T>
void DispatchDecodeGraph(const core::Tensor& q,
                         const core::Tensor& k,
                         const core::Tensor& v,
                         core::Tensor&       out,
                         char**              block_ptrs,
                         const int*          cu_block_nums,
                         const int*          cu_q_len,
                         const int*          cu_k_len,
                         char*               workspace,
                         int                 num_heads,
                         int                 num_kv_heads,
                         int                 head_dim,
                         int                 batch_size,
                         int                 block_len,
                         int                 max_k_len,
                         int                 quant_policy,
                         cudaStream_t        stream)
{
    TM_CHECK_EQ(k.size(), (size_t)batch_size * num_kv_heads * head_dim) << "k must be [B, Hkv*D] contiguous";
    TM_CHECK_EQ(v.size(), (size_t)batch_size * num_kv_heads * head_dim) << "v must be [B, Hkv*D] contiguous";
    TM_CHECK_EQ(k.dtype(), q.dtype()) << "k dtype must match q";
    TM_CHECK_EQ(v.dtype(), q.dtype()) << "v dtype must match q";
    TM_CHECK_NE(cu_q_len, nullptr) << "device cu_q_len is required";
    TM_CHECK_NE(cu_k_len, nullptr) << "device cu_k_len is required";
    TM_CHECK_NE(workspace, nullptr) << "persistent workspace is required";

    const auto layout = DecodeWorkspaceLayout::Make(batch_size, num_heads, num_kv_heads, head_dim, sizeof(T));

    AttentionParams<T> params{};

    // Fused [B, (H + 2Hkv) * D] qkv row per token, filled by captured D2D
    // memcpy2D nodes from the caller's persistent q/k/v rows.
    const size_t q_bytes   = num_heads * head_dim * sizeof(T);
    const size_t kv_bytes  = num_kv_heads * head_dim * sizeof(T);
    const size_t row_bytes = (num_heads + 2 * num_kv_heads) * head_dim * sizeof(T);
    char*        qkv_base  = workspace + layout.qkv;
    {
        TM_CUDA_CHECK(cudaMemcpy2DAsync(
            qkv_base, row_bytes, q.raw_data(), q_bytes, q_bytes, batch_size, cudaMemcpyDeviceToDevice, stream));
        TM_CUDA_CHECK(cudaMemcpy2DAsync(qkv_base + q_bytes,
                                        row_bytes,
                                        k.raw_data(),
                                        kv_bytes,
                                        kv_bytes,
                                        batch_size,
                                        cudaMemcpyDeviceToDevice,
                                        stream));
        TM_CUDA_CHECK(cudaMemcpy2DAsync(qkv_base + q_bytes + kv_bytes,
                                        row_bytes,
                                        v.raw_data(),
                                        kv_bytes,
                                        kv_bytes,
                                        batch_size,
                                        cudaMemcpyDeviceToDevice,
                                        stream));
    }

    params.out    = (T*)out.raw_data();
    params.q      = (T*)qkv_base;
    params.k      = params.q + num_heads * head_dim;
    params.v      = params.k + num_kv_heads * head_dim;
    params.stride = (num_heads + 2 * num_kv_heads) * head_dim;

    params.batch_size = batch_size;
    params.token_num  = batch_size;
    params.max_q_len  = 1;
    params.max_k_len  = max_k_len;

    params.block_iter_params = BlockIteratorParams{block_ptrs, cu_block_nums, 0, block_len};

    bool* finished_ptr        = (bool*)(workspace + layout.finished);
    int*  readonly_ptr        = (int*)(workspace + layout.readonly_block_num);
    params.finished           = finished_ptr;
    params.cu_q_len           = cu_q_len;
    params.cu_k_len           = cu_k_len;
    params.token_mask         = nullptr;
    params.readonly_block_num = readonly_ptr;

    TM_CUDA_CHECK(cudaMemsetAsync(finished_ptr, 0, sizeof(bool) * batch_size, stream));
    TM_CUDA_CHECK(cudaMemsetAsync(readonly_ptr, 0, sizeof(int) * batch_size, stream));

    params.num_heads     = num_heads;
    params.num_kv_heads  = num_kv_heads;
    params.size_per_head = head_dim;
    params.inv_sqrt_dh   = (float)std::log2(expf(1.f)) / std::sqrt((float)head_dim);
    params.window_size   = 256 << 20;

    params.rope_param      = RopeKernelParam{};
    params.rope_param.type = RopeType::kNull;

    const int max_split_k = DecodeWorkspaceLayout::MaxSplitK(batch_size);
    params.split_cnt      = (int*)(workspace + layout.split_cnt);
    params.partial_O      = (float*)(workspace + layout.partial_O);
    params.partial_ML     = (float*)(workspace + layout.partial_ML);
    params.max_split_k    = max_split_k;
    TM_CUDA_CHECK(cudaMemsetAsync(params.split_cnt, 0, sizeof(int) * batch_size, stream));

    params.arch         = getSMVersion();
    params.stream       = stream;
    params.quant_policy = quant_policy;

    dispatchDecoding<T>(params);
}

size_t TmDecodeWorkspaceBytes(int batch_size, int num_heads, int num_kv_heads, int head_dim, size_t elem_size)
{
    TM_CHECK(batch_size > 0) << "batch_size must be positive";
    return DecodeWorkspaceLayout::Make(batch_size, num_heads, num_kv_heads, head_dim, elem_size).total;
}

void TmAttnDecodeGraph(const std::shared_ptr<core::Tensor>& q,
                       const std::shared_ptr<core::Tensor>& k,
                       const std::shared_ptr<core::Tensor>& v,
                       std::shared_ptr<core::Tensor>        out,
                       std::uintptr_t                       block_ptrs_addr,
                       std::uintptr_t                       cu_block_nums_addr,
                       std::uintptr_t                       cu_q_len_addr,
                       std::uintptr_t                       cu_k_len_addr,
                       std::uintptr_t                       workspace_addr,
                       int                                  num_heads,
                       int                                  num_kv_heads,
                       int                                  head_dim,
                       int                                  batch_size,
                       int                                  block_len,
                       int                                  max_k_len,
                       int                                  kv_quant,
                       std::uintptr_t                       stream_addr)
{
    TM_CHECK_EQ(q->shape(0), batch_size) << "q rows must equal batch_size";
    TM_CHECK_EQ(q->size(), (size_t)batch_size * num_heads * head_dim) << "q must be [B, H*D] contiguous";
    TM_CHECK(kv_quant == 0 || kv_quant == 4 || kv_quant == 8) << "kv_quant must be one of 0, 4, 8";
    TM_CHECK(out != nullptr) << "out tensor is required";

    auto       stream        = (cudaStream_t)stream_addr;
    char**     block_ptrs    = (char**)block_ptrs_addr;
    const int* cu_block_nums = (const int*)cu_block_nums_addr;
    const int* cu_q_len      = (const int*)cu_q_len_addr;
    const int* cu_k_len      = (const int*)cu_k_len_addr;
    char*      workspace     = (char*)workspace_addr;

    int quant_policy = 0;
    if (kv_quant == 8) {
        quant_policy = QuantPolicy::kCacheKVInt8;
    }
    else if (kv_quant == 4) {
        quant_policy = QuantPolicy::kCacheKVInt4;
    }

    // No allocations below: safe under CUDA graph capture.
    py::gil_scoped_release release;
    if (q->dtype() == kBfloat16) {
        DispatchDecodeGraph<nv_bfloat16>(*q,
                                         *k,
                                         *v,
                                         *out,
                                         block_ptrs,
                                         cu_block_nums,
                                         cu_q_len,
                                         cu_k_len,
                                         workspace,
                                         num_heads,
                                         num_kv_heads,
                                         head_dim,
                                         batch_size,
                                         block_len,
                                         max_k_len,
                                         quant_policy,
                                         stream);
    }
    else if (q->dtype() == kFloat16) {
        DispatchDecodeGraph<half>(*q,
                                  *k,
                                  *v,
                                  *out,
                                  block_ptrs,
                                  cu_block_nums,
                                  cu_q_len,
                                  cu_k_len,
                                  workspace,
                                  num_heads,
                                  num_kv_heads,
                                  head_dim,
                                  batch_size,
                                  block_len,
                                  max_k_len,
                                  quant_policy,
                                  stream);
    }
    else {
        TM_CHECK(0) << "q dtype must be bf16 or fp16, got: " << to_string(q->dtype());
    }
    TM_CUDA_CHECK(cudaGetLastError());
}

py::list AttnDecodeBench(const std::shared_ptr<core::Tensor>& q,
                         const std::shared_ptr<core::Tensor>& k,
                         const std::shared_ptr<core::Tensor>& v,
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
                                             *k,
                                             *v,
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
                                      *k,
                                      *v,
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

// Production single-shot decode on caller-owned blocks. Runs one dispatch
// with caller-provided per-sequence lengths; no events, no sync. Not usable
// under CUDA graph capture (core allocations + TM_CHECK host logic); callers
// fall back to the PyTorch read path when capturing.
void TmAttnDecode(const std::shared_ptr<core::Tensor>& q,
                  const std::shared_ptr<core::Tensor>& k,
                  const std::shared_ptr<core::Tensor>& v,
                  std::shared_ptr<core::Tensor>        out,
                  std::uintptr_t                       block_ptrs_addr,
                  std::uintptr_t                       cu_block_nums_addr,
                  std::uintptr_t                       cu_q_len_addr,
                  std::uintptr_t                       cu_k_len_addr,
                  int                                  num_heads,
                  int                                  num_kv_heads,
                  int                                  head_dim,
                  int                                  batch_size,
                  int                                  block_len,
                  int                                  max_k_len,
                  int                                  kv_quant,
                  std::uintptr_t                       stream_addr)
{
    TM_CHECK_EQ(q->shape(0), batch_size) << "q rows must equal batch_size";
    TM_CHECK_EQ(q->size(), (size_t)batch_size * num_heads * head_dim) << "q must be [B, H*D] contiguous";
    TM_CHECK(kv_quant == 0 || kv_quant == 4 || kv_quant == 8) << "kv_quant must be one of 0, 4, 8";
    TM_CHECK(out != nullptr) << "out tensor is required";

    auto       stream        = (cudaStream_t)stream_addr;
    char**     block_ptrs    = (char**)block_ptrs_addr;
    const int* cu_block_nums = (const int*)cu_block_nums_addr;
    // Optional caller-owned device cumulative lengths ([B+1] int32). Null
    // falls back to the benchmark's constant per-sequence lengths (old
    // behavior); non-null feeds real per-sequence lengths to the plan.
    const int* cu_q_len = (const int*)cu_q_len_addr;
    const int* cu_k_len = (const int*)cu_k_len_addr;

    int quant_policy = 0;
    if (kv_quant == 8) {
        quant_policy = QuantPolicy::kCacheKVInt8;
    }
    else if (kv_quant == 4) {
        quant_policy = QuantPolicy::kCacheKVInt4;
    }

    // Core allocations below need a context; borrow the caller's stream and
    // the default device pool (same recipe as `create_device_context`).
    core::ContextGuard ctx_guard{
        core::Stream::borrow(stream), core::Allocator(kCPU), core::Allocator(core::Stream::borrow(stream), true)};

    py::gil_scoped_release release;
    if (q->dtype() == kBfloat16) {
        DecodePlan<nv_bfloat16> plan = BuildDecodePlan<nv_bfloat16>(*q,
                                                                    *k,
                                                                    *v,
                                                                    *out,
                                                                    block_ptrs,
                                                                    cu_block_nums,
                                                                    cu_q_len,
                                                                    cu_k_len,
                                                                    num_heads,
                                                                    num_kv_heads,
                                                                    head_dim,
                                                                    batch_size,
                                                                    block_len,
                                                                    max_k_len,
                                                                    quant_policy,
                                                                    stream);
        dispatchDecoding<nv_bfloat16>(plan.params);
    }
    else if (q->dtype() == kFloat16) {
        DecodePlan<half> plan = BuildDecodePlan<half>(*q,
                                                      *k,
                                                      *v,
                                                      *out,
                                                      block_ptrs,
                                                      cu_block_nums,
                                                      cu_q_len,
                                                      cu_k_len,
                                                      num_heads,
                                                      num_kv_heads,
                                                      head_dim,
                                                      batch_size,
                                                      block_len,
                                                      max_k_len,
                                                      quant_policy,
                                                      stream);
        dispatchDecoding<half>(plan.params);
    }
    else {
        TM_CHECK(0) << "q dtype must be bf16 or fp16, got: " << to_string(q->dtype());
    }
    TM_CUDA_CHECK(cudaGetLastError());
}

// Production quantizing KV append (`invokeProcessKV_v2`) into caller-owned
// blocks. `k`/`v` hold the NEW tokens only ([sum_q, Hkv, D] token-major rows,
// head stride in units of head_dim via `stride_h`); rope is null because the
// caller pre-rotates. `cu_q_len`/`cu_k_len` are cumulative device int32
// [B+1]; `cu_block_base[batch]` indexes the sequence-major `char**` list.
// NOTE: the block byte layout produced here follows `block::Layout`
// (per-head interleaved K/V), which the attention impl must agree on.
void TmProcessKV(const std::shared_ptr<core::Tensor>& k,
                 const std::shared_ptr<core::Tensor>& v,
                 std::uintptr_t                       block_ptrs_addr,
                 std::uintptr_t                       cu_q_len_addr,
                 std::uintptr_t                       cu_k_len_addr,
                 std::uintptr_t                       cu_block_base_addr,
                 int64_t                              stride_c,
                 int64_t                              stride_s,
                 int64_t                              stride_h,
                 int                                  max_q_len,
                 int                                  num_kv_heads,
                 int                                  head_dim,
                 int                                  batch_size,
                 int                                  block_len,
                 int                                  kv_quant,
                 std::uintptr_t                       stream_addr)
{
    TM_CHECK(kv_quant == 8) << "tm_process_kv supports int8 KV cache only";
    TM_CHECK_EQ(k->dtype(), v->dtype()) << "k/v dtypes must match";
    TM_CHECK_EQ(k->stride(k->ndim() - 1), 1) << "k head-dim must be contiguous";

    auto       stream = (cudaStream_t)stream_addr;
    char**     blocks = (char**)block_ptrs_addr;
    const int* cu_q   = (const int*)cu_q_len_addr;
    const int* cu_k   = (const int*)cu_k_len_addr;
    const int* cu_blk = (const int*)cu_block_base_addr;

    int             quant_policy = QuantPolicy::kCacheKVInt8;
    RopeKernelParam rope{};
    rope.type = RopeType::kNull;

    core::ContextGuard ctx_guard{
        core::Stream::borrow(stream), core::Allocator(kCPU), core::Allocator(core::Stream::borrow(stream), true)};

    py::gil_scoped_release release;
    if (k->dtype() == kBfloat16) {
        invokeProcessKV_v2<nv_bfloat16>(blocks,
                                        (const nv_bfloat16*)k->raw_data(),
                                        (const nv_bfloat16*)v->raw_data(),
                                        nullptr,
                                        nullptr,
                                        cu_q,
                                        cu_k,
                                        cu_blk,
                                        nullptr,
                                        rope,
                                        0,  // stride_b: batch base comes from cu_q_len
                                        stride_c,
                                        stride_h,
                                        stride_s,
                                        block_len,
                                        0,  // cache_block_offset
                                        0,  // cp_rank
                                        cutlass::FastDivmod(1),
                                        max_q_len,
                                        num_kv_heads,
                                        head_dim,
                                        batch_size,
                                        quant_policy,
                                        stream);
    }
    else if (k->dtype() == kFloat16) {
        invokeProcessKV_v2<half>(blocks,
                                 (const half*)k->raw_data(),
                                 (const half*)v->raw_data(),
                                 nullptr,
                                 nullptr,
                                 cu_q,
                                 cu_k,
                                 cu_blk,
                                 nullptr,
                                 rope,
                                 0,
                                 stride_c,
                                 stride_h,
                                 stride_s,
                                 block_len,
                                 0,
                                 0,
                                 cutlass::FastDivmod(1),
                                 max_q_len,
                                 num_kv_heads,
                                 head_dim,
                                 batch_size,
                                 quant_policy,
                                 stream);
    }
    else {
        TM_CHECK(0) << "k dtype must be bf16 or fp16, got: " << to_string(k->dtype());
    }
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void bind_attn_bench(py::module_& m)
{
    m.def("attn_decode_bench",
          &AttnDecodeBench,
          "q"_a,
          "k"_a,
          "v"_a,
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
          "Timed decode-attention dispatch; returns per-iteration ms. `k`/`v` "
          "[B, Hkv*D] are re-stored (quantized) into the pool's last valid "
          "position each dispatch, mirroring production decode; pass the real "
          "K/V rows of that position.");

    m.def("tm_attn_decode",
          &TmAttnDecode,
          "q"_a,
          "k"_a,
          "v"_a,
          "out"_a,
          "block_ptrs_addr"_a,
          "cu_block_nums_addr"_a,
          "cu_q_len_addr"_a = std::uintptr_t{0},
          "cu_k_len_addr"_a = std::uintptr_t{0},
          "num_heads"_a,
          "num_kv_heads"_a,
          "head_dim"_a,
          "batch_size"_a,
          "block_len"_a,
          "max_k_len"_a,
          "kv_quant"_a   = 0,
          "stream_ptr"_a = std::uintptr_t{0},
          "Single-shot decode-attention dispatch into `out`. `k`/`v` "
          "[B, Hkv*D] must hold the real K/V rows of the newest token "
          "(ti = k_len - 1): the kernel re-stores them (rope'd, quantized) "
          "into the pool, mirroring production decode. Optional "
          "`cu_q_len_addr`/`cu_k_len_addr` are device int32[B+1] cumulative "
          "per-sequence lengths; null (default) keeps the benchmark's "
          "constant-length baking (equal-length batches only).");

    m.def("tm_decode_workspace_bytes",
          &TmDecodeWorkspaceBytes,
          "batch_size"_a,
          "num_heads"_a,
          "num_kv_heads"_a,
          "head_dim"_a,
          "elem_size"_a,
          "Persistent workspace byte count for `tm_attn_decode_graph`.");

    m.def("tm_attn_decode_graph",
          &TmAttnDecodeGraph,
          "q"_a,
          "k"_a,
          "v"_a,
          "out"_a,
          "block_ptrs_addr"_a,
          "cu_block_nums_addr"_a,
          "cu_q_len_addr"_a,
          "cu_k_len_addr"_a,
          "workspace_addr"_a,
          "num_heads"_a,
          "num_kv_heads"_a,
          "head_dim"_a,
          "batch_size"_a,
          "block_len"_a,
          "max_k_len"_a,
          "kv_quant"_a   = 0,
          "stream_ptr"_a = std::uintptr_t{0},
          "CUDA-graph-safe decode dispatch: all plan state lives in "
          "caller-owned persistent buffers (device cu_q_len/cu_k_len, "
          "workspace); no allocation, no host-stack H2D. Capture records "
          "only kernel/memcpy/memset nodes; replay follows per-step buffer "
          "content. `max_k_len` is the frozen capacity bound.");

    m.def("tm_process_kv",
          &TmProcessKV,
          "k"_a,
          "v"_a,
          "block_ptrs_addr"_a,
          "cu_q_len_addr"_a,
          "cu_k_len_addr"_a,
          "cu_block_base_addr"_a,
          "stride_c"_a,
          "stride_s"_a,
          "stride_h"_a,
          "max_q_len"_a,
          "num_kv_heads"_a,
          "head_dim"_a,
          "batch_size"_a,
          "block_len"_a,
          "kv_quant"_a   = 8,
          "stream_ptr"_a = std::uintptr_t{0},
          "Quantizing KV append (invokeProcessKV_v2, null rope) into caller-owned blocks.");
}

}  // namespace turbomind::python_attn_bench
