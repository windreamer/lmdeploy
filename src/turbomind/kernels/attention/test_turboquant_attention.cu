// TurboQuant end-to-end test — forked from test_attention.cu
// Compile: separate binary with KV_TURBO=1, DECODING=1

#define KV_TURBO 1
#define DECODING 1
#define SINK 0

#include "attention.h"
#include "block.h"
#include "decoding.h"
#include "kv_cache_utils_v2.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/attention/attention_params.h"
#include "src/turbomind/kernels/attention/reference.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "test_utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>
#include <utility>

using namespace turbomind;

template<class T, class Tkv>
struct Config {
    int head_dim_;
    int head_num_;
    int block_len_;

    TM_HOST_DEVICE constexpr int t_bits() const
    {
        if constexpr (std::is_same_v<T, Tkv>) return 0;
        else return bitsof<T>;
    }

    TM_HOST_DEVICE constexpr int q_bits() const { return bitsof<Tkv>; }
    TM_HOST_DEVICE constexpr int v_bits() const { return 2; }  // TurboQuant V=2bit
    TM_HOST_DEVICE constexpr int k_param_count() const { return 2; }
    TM_HOST_DEVICE constexpr int v_param_count() const { return 2; }
    TM_HOST_DEVICE constexpr int head_dim() const { return head_dim_; }
    TM_HOST_DEVICE int head_num() const { return head_num_; }
    TM_HOST_DEVICE constexpr int block_len() const { return block_len_; }
    TM_HOST_DEVICE constexpr bool is_share_kv() const { return false; }
};

template<class T>
int test_turboquant_decode()
{
    AttentionParams<T> params{};

    // Small decode scenario for quick verification
    constexpr size_t kHeadDim    = 128;
    constexpr size_t kHeadNum    = 8;
    constexpr size_t KvHeadNum   = kHeadNum;  // MHA for simplicity
    constexpr size_t kBatchSize  = 4;
    constexpr size_t kInputLen   = 1;
    constexpr size_t kSequenceLen = 256;
    constexpr int    kBlockSz    = 64;
    constexpr int    kMaxSplitK  = 1;
    constexpr int    kTestIter   = 1;
    constexpr float  kRoPEBase  = 10000.f;
    constexpr int    kRoPEDim   = kHeadDim / 2;
    constexpr size_t kContextLen = kSequenceLen + kInputLen;
    constexpr size_t kTokenNum  = kBatchSize * kInputLen;

    using Tkv = uint4_t;
    constexpr int kQuantPolicy = QuantPolicy::kCacheKVTurbo;

    Config<T, Tkv> config{(int)kHeadDim, (int)KvHeadNum, (int)kBlockSz};
    block::Layout layout{config};

    std::cout << "=== TurboQuant Decode End-to-End Test ===\n";
    std::cout << "head_dim=" << kHeadDim << " kv_heads=" << KvHeadNum
              << " batch=" << kBatchSize << " seq_len=" << kSequenceLen
              << " quant_policy=" << kQuantPolicy << "\n";
    std::cout << "K token data: " << layout.k_token_data_size() << " bytes\n";
    std::cout << "V token data: " << layout.v_token_data_size() << " bytes\n";
    std::cout << "Layer size: " << layout.layer_size() << " bytes\n";

    RNG rng{};

    // FP16 K/V cache for reference
    thrust::universal_vector<T> k_cache(kBatchSize * KvHeadNum * kContextLen * kHeadDim);
    thrust::universal_vector<T> v_cache(kBatchSize * KvHeadNum * kContextLen * kHeadDim);
    rng.GenerateNormal(k_cache.data().get(), k_cache.size(), 1.f, 0.f);
    rng.GenerateNormal(v_cache.data().get(), v_cache.size(), 1.f, 0.f);

    // Apply RoPE to K cache
    TM_SCOPE_CALL(invokeApplyRotaryEmbedding(
        k_cache.data().get(), kContextLen, KvHeadNum, kHeadDim, kRoPEBase, kRoPEDim, kBatchSize));

    // Keep reference copy
    thrust::universal_vector<T> k_cache_ref = k_cache;
    thrust::universal_vector<T> v_cache_ref = v_cache;

    // Build block cache from FP16 K/V
    const size_t n_blocks = (kContextLen + kBlockSz - 1) / kBlockSz;
    thrust::universal_vector<char> blocks(kBatchSize * n_blocks * layout.block_size(1));
    thrust::universal_vector<char*> k_ptrs(kBatchSize * n_blocks + 1);

    // Simple linear block allocation (no shuffling for test)
    for (size_t i = 0; i < kBatchSize * n_blocks; ++i) {
        k_ptrs[i] = blocks.data().get() + i * layout.block_size(1);
    }

    thrust::universal_vector<int> cu_seq_lens(kBatchSize + 1);
    thrust::universal_vector<int> cu_block_cnts(kBatchSize + 1);
    for (size_t i = 0; i <= kBatchSize; ++i) {
        cu_seq_lens[i] = i * kContextLen;
    }
    std::vector<int> n_blocks_vec(kBatchSize + 1, n_blocks);
    thrust::universal_vector<int> cu_block_cnts_h(n_blocks_vec.size());
    std::exclusive_scan(n_blocks_vec.begin(), n_blocks_vec.end(), cu_block_cnts_h.begin(), 0);
    cu_block_cnts = cu_block_cnts_h;

    // Interleave K/V for ProcessKV_v2
    const size_t kHSD = KvHeadNum * kContextLen * kHeadDim;
    thrust::universal_vector<T> kv_cache(k_cache.size() * 2);
    {
        auto k_src = k_cache.begin();
        auto v_src = v_cache.begin();
        auto dst = kv_cache.begin();
        for (size_t i = 0; i < kBatchSize; ++i) {
            dst = thrust::copy_n(k_src, kHSD, dst);
            dst = thrust::copy_n(v_src, kHSD, dst);
            k_src += kHSD;
            v_src += kHSD;
        }
    }

    // ProcessKV_v2: FP16 → TurboQuant block cache
    TM_SCOPE_CALL(invokeProcessKV_v2(k_ptrs.data().get(),
                                     kv_cache.data().get(),
                                     kv_cache.data().get() + KvHeadNum * kContextLen * kHeadDim,
                                     (T*)nullptr,
                                     (T*)nullptr,
                                     cu_seq_lens.data().get(),
                                     cu_seq_lens.data().get(),
                                     cu_block_cnts.data().get(),
                                     RopeKernelParam{},
                                     2 * KvHeadNum * kContextLen,
                                     0,
                                     (int)kContextLen,
                                     1,
                                     (int)kBlockSz,
                                     0, 0, 1,
                                     (int)kContextLen,
                                     (int)KvHeadNum,
                                     (int)kHeadDim,
                                     (int)kBatchSize,
                                     kQuantPolicy));

    if (auto err = cudaGetLastError(); err != cudaSuccess) {
        std::cout << "ProcessKV_v2 FAILED: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // FlattenKV_v2: TurboQuant block cache → FP16 (for reference comparison)
    // Use interleaved buffer [B, 2, H, S, D] as required by stride_b
    thrust::universal_vector<T> kv_cache_2(k_cache.size() * 2);

    TM_SCOPE_CALL(invokeFlattenKV_v2(kv_cache_2.data().get(),
                                     kv_cache_2.data().get() + KvHeadNum * kContextLen * kHeadDim,
                                     k_ptrs.data().get(),
                                     cu_seq_lens.data().get(),
                                     cu_block_cnts.data().get(),
                                     RopeKernelParam{},
                                     2 * KvHeadNum * kContextLen,
                                     0,
                                     (int)kContextLen,
                                     1,
                                     (int)kBlockSz,
                                     0, 0, 1,
                                     (int)kContextLen,
                                     (int)KvHeadNum,
                                     (int)kHeadDim,
                                     (int)kBatchSize,
                                     kQuantPolicy));

    cudaDeviceSynchronize();

    if (auto err = cudaGetLastError(); err != cudaSuccess) {
        std::cout << "FlattenKV_v2 FAILED: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // Compare ProcessKV→FlattenKV roundtrip vs original FP16
    std::cout << "\n--- ProcessKV→FlattenKV Roundtrip vs FP16 ---\n";
    T* k_out = kv_cache_2.data().get();
    T* v_out = kv_cache_2.data().get() + KvHeadNum * kContextLen * kHeadDim;
    Compare(k_out, k_cache_ref.data().get(),
            kContextLen * kHeadDim, kContextLen * kHeadDim,
            kBatchSize * KvHeadNum, 0);
    Compare(v_out, v_cache_ref.data().get(),
            kContextLen * kHeadDim, kContextLen * kHeadDim,
            kBatchSize * KvHeadNum);

    // Now test decode attention
    thrust::universal_vector<T> qkv(kBatchSize * kInputLen * (kHeadNum + KvHeadNum * 2) * kHeadDim);
    thrust::universal_vector<T> output(kBatchSize * kInputLen * kHeadNum * kHeadDim);
    thrust::universal_vector<T> output_ref(kBatchSize * kInputLen * kHeadNum * kHeadDim);
    rng.GenerateNormal(qkv.data().get(), qkv.size(), 1.f, 0.f);

    thrust::universal_vector<bool> finished(kBatchSize);
    thrust::universal_vector<int> sequence_length(kBatchSize);
    thrust::universal_vector<int> input_length(kBatchSize);
    thrust::universal_vector<int> context_length(kBatchSize);
    thrust::universal_vector<float> rope_base(kBatchSize);
    thrust::universal_vector<int> cu_seqlens(kBatchSize + 1);
    thrust::universal_vector<int> cu_kv_lens(kBatchSize + 1);

    thrust::device_vector<float> partial_ML(kTokenNum * kHeadNum * kMaxSplitK * 2);
    thrust::device_vector<float> partial_O(kTokenNum * kHeadNum * kMaxSplitK * kHeadDim);
    thrust::device_vector<int> split_cnt(kTokenNum);

    for (size_t i = 0; i <= kBatchSize; ++i) {
        cu_seqlens[i] = i * kInputLen;
        cu_kv_lens[i] = i * kContextLen;
    }
    for (size_t i = 0; i < kBatchSize; ++i) {
        input_length[i] = kInputLen;
        sequence_length[i] = kSequenceLen;
        context_length[i] = kContextLen;
        rope_base[i] = kRoPEBase;
    }

    // FP16 reference attention
    Reference<T> reference({});
    reference.Reshape(kInputLen, kContextLen, kHeadNum, kHeadDim, KvHeadNum, kBatchSize, 128 << 20);

    // Use k_cache_ref/v_cache_ref (FP16) for reference
    thrust::universal_vector<void*> k_cache_ref_ptrs(kBatchSize);
    thrust::universal_vector<void*> v_cache_ref_ptrs(kBatchSize);
    for (size_t i = 0; i < kBatchSize; ++i) {
        k_cache_ref_ptrs[i] = k_cache_ref.data().get() + i * k_cache_ref.size() / kBatchSize;
        v_cache_ref_ptrs[i] = v_cache_ref.data().get() + i * v_cache_ref.size() / kBatchSize;
    }

    reference.Execute(output_ref.data().get(),
                      k_cache_ref.data().get(),
                      v_cache_ref.data().get(),
                      qkv.data().get(),
                      (T*)nullptr,
                      nullptr,
                      kRoPEBase,
                      kRoPEDim);

    cudaDeviceSynchronize();

    // Set up decode attention params
    params.out = output.data().get();
    params.q   = qkv.data().get();
    params.k   = params.q + kHeadNum * kHeadDim;
    params.v   = params.k + KvHeadNum * kHeadDim;
    params.stride = (kHeadNum + 2 * KvHeadNum) * kHeadDim;

    params.token_num  = kTokenNum;
    params.batch_size = kBatchSize;
    params.max_q_len  = kInputLen;
    params.max_k_len  = kContextLen;

    params.block_iter_params = BlockIteratorParams{k_ptrs.data().get(),
                                                   cu_block_cnts.data().get(),
                                                   0,
                                                   (int)kBlockSz};

    params.quant_policy = kQuantPolicy;

    params.finished   = finished.data().get();
    params.rope_theta = rope_base.data().get();
    params.cu_q_len   = cu_seqlens.data().get();
    params.cu_k_len   = cu_kv_lens.data().get();

    params.num_heads     = kHeadNum;
    params.num_kv_heads  = KvHeadNum;
    params.size_per_head = kHeadDim;
    params.window_size   = 128 << 20;
    params.inv_sqrt_dh   = 1.f / std::sqrt((float)kHeadDim);

    float scale_factor = -std::log2f(kRoPEBase) / kRoPEDim;
    params.rope_param  = RopeKernelParam{RopeType::kDefault, nullptr, kRoPEDim, scale_factor, 1.f};

    params.split_cnt  = split_cnt.data().get();
    params.partial_ML = partial_ML.data().get();
    params.partial_O  = partial_O.data().get();
    params.max_split_k = kMaxSplitK;
    params.arch        = getSMVersion();

    // Launch TurboQuant decode attention
    dispatchDecoding<T>(params);

    cudaDeviceSynchronize();

    if (auto err = cudaGetLastError(); err != cudaSuccess) {
        std::cout << "dispatchDecoding FAILED: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // Compare TurboQuant decode output vs FP16 reference
    std::cout << "\n--- TurboQuant Decode vs FP16 Reference ---\n";
    Compare(output.data().get(), output_ref.data().get(),
            kHeadNum * kHeadDim, kHeadNum * kHeadDim,
            kBatchSize * kInputLen, 0);

    return 0;
}

int main()
{
    return test_turboquant_decode<half>();
}
