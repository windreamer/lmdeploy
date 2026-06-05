// Copyright (c) OpenMMLab. All rights reserved.

#include "decoding.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/attention/hadamard_kernel.h"
#include "src/turbomind/kernels/attention/registry.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

// Apply Hadamard rotation to Q for TurboQuant decode path.
// Rotates each head's vector independently (head_dim must be power of 2).
template<class T>
static void rotateQForTurboQuant(const AttentionParams<T>& params)
{
    const int head_dim  = params.size_per_head;
    const int num_heads = params.num_heads;

    HadamardParams hp{};
    hp.batch   = params.batch_size * num_heads;
    hp.dim     = head_dim;
    hp.log_dim = static_cast<int>(log2(head_dim));
    hp.stride  = head_dim;
    hp.scale   = 1.0f / sqrtf((float)head_dim);
    hp.x_ptr   = params.q;
    hp.out_ptr  = params.q;  // in-place

    hadamard_transform<T>(hp, params.stream);
}

// Apply inverse Hadamard to attention output for TurboQuant decode path.
template<class T>
static void invRotateOutputForTurboQuant(const AttentionParams<T>& params)
{
    const int head_dim  = params.size_per_head;
    const int num_heads = params.num_heads;

    HadamardParams hp{};
    hp.batch   = params.batch_size * num_heads;
    hp.dim     = head_dim;
    hp.log_dim = static_cast<int>(log2(head_dim));
    hp.stride  = head_dim;
    hp.scale   = 1.0f / sqrtf((float)head_dim);
    hp.x_ptr   = params.out;
    hp.out_ptr  = params.out;  // in-place

    hadamard_transform<T>(hp, params.stream);
}

template<class T>
void dispatchDecoding(const AttentionParams<T>& params)
{
    using namespace attention;

    const bool is_kv_int8     = params.quant_policy & QuantPolicy::kCacheKVInt8;
    const bool is_kv_int4     = params.quant_policy & QuantPolicy::kCacheKVInt4;
    const bool is_turbo_quant = params.quant_policy == 42;
    const int  query_group_sz = params.num_heads / params.num_kv_heads;

    TM_CHECK(!(is_kv_int4 && is_kv_int8));

    // TurboQuant: rotate Q before attention (H is orthogonal, same transform for forward/inverse)
    if (is_turbo_quant) {
        rotateQForTurboQuant(params);
    }

    int kv_quant = is_turbo_quant ? 42 : (is_kv_int4 ? 4 : (is_kv_int8 ? 8 : 0));

    AttnDesc desc{};
    desc.mode           = AttnDesc::kDecoding;
    desc.head_dim       = params.size_per_head;
    desc.data_type      = data_type_v<T>;
    desc.kv_quant       = kv_quant;
    desc.query_group_sz = query_group_sz;

    auto& reg    = Registry::instance();
    auto* kernel = reg.Find(desc);

    TM_CHECK(kernel) << "No decoding kernel found: " + to_string(desc);

    TM_SCOPE_CALL(kernel->Launch(&params, reg.sm_count()));

    // TurboQuant: inverse-rotate output after attention + split-K reduction
    if (is_turbo_quant) {
        invRotateOutputForTurboQuant(params);
    }
}

template void dispatchDecoding(const AttentionParams<half>& params);
#if ENABLE_BF16
template void dispatchDecoding(const AttentionParams<nv_bfloat16>& params);
#endif

}  // namespace turbomind
