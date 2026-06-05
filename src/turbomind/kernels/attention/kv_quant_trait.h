// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/sub_byte_ptr.h"
#include <type_traits>

namespace turbomind::attention {

// ---------------------------------------------------------------------------
// KV quantization scheme tags
// ---------------------------------------------------------------------------

/// No quantization: KV stored in compute type T
struct KvQuantNone {
};

/// INT8 quantization: KV stored as uint8_t, affine (scale + zero)
struct KvQuantInt8 {
};

/// INT4 quantization: KV stored as uint4_t, affine (scale + zero)
struct KvQuantInt4 {
};

/// TurboQuant: K=4bit QJL4 (codebook), V=2bit MSE (codebook), Hadamard rotation
struct KvQuantTurbo {
};

// ---------------------------------------------------------------------------
// KvQuantTrait<Tag> — central type trait for KV cache quantization
// ---------------------------------------------------------------------------
// Provides all derived types and constants needed by the attention kernel
// pipeline.  Each tag maps to a full set of properties for K and V
// independently, paving the way for asymmetric K/V quantization (e.g.
// TurboQuant: K=4bit QJL4, V=2bit MSE).
//
// Trait members:
//   StorageK / StorageV   — raw KV cache element type
//   PointerK / PointerV   — pointer type for accessing KV data (T* or SubBytePtr)
//   kBitsK / kBitsV       — bit-width of StorageK / StorageV
//   kQuantKV              — whether quantization is active (K or V differs from T)
//   kParamCountK / kParamCountV — # of param values per token per head
//   kv_quant              — runtime integer descriptor for kernel dispatch
//   kHadamardRotate       — whether Hadamard rotation is applied
//   EncoderK / EncoderV   — store-path converter (T → StorageK/V)
//   DecoderK / DecoderV   — load-path converter (StorageK/V → T)
// ---------------------------------------------------------------------------

template<class KvQuant, class T>
struct KvQuantTrait;

// ---- KvQuantNone ----

template<class T>
struct KvQuantTrait<KvQuantNone, T> {
    using StorageK = T;
    using StorageV = T;

    using PointerK = T*;
    using PointerV = T*;

    static constexpr int kBitsK = bitsof<T>;
    static constexpr int kBitsV = bitsof<T>;

    static constexpr bool kQuantKV = false;

    static constexpr int kParamCountK = 0;
    static constexpr int kParamCountV = 0;

    static constexpr int kv_quant = 0;

    static constexpr bool kHadamardRotate = false;
};

// ---- KvQuantInt8 ----

template<class T>
struct KvQuantTrait<KvQuantInt8, T> {
    using StorageK = uint8_t;
    using StorageV = uint8_t;

    using PointerK = uint8_t*;
    using PointerV = uint8_t*;

    static constexpr int kBitsK = 8;
    static constexpr int kBitsV = 8;

    static constexpr bool kQuantKV = true;

    static constexpr int kParamCountK = 2;
    static constexpr int kParamCountV = 2;

    static constexpr int kv_quant = 8;

    static constexpr bool kHadamardRotate = false;
};

// ---- KvQuantInt4 ----

template<class T>
struct KvQuantTrait<KvQuantInt4, T> {
    using StorageK = uint4_t;
    using StorageV = uint4_t;

    using PointerK = SubBytePtr<uint4_t>;
    using PointerV = SubBytePtr<uint4_t>;

    static constexpr int kBitsK = 4;
    static constexpr int kBitsV = 4;

    static constexpr bool kQuantKV = true;

    static constexpr int kParamCountK = 2;
    static constexpr int kParamCountV = 2;

    static constexpr int kv_quant = 4;

    static constexpr bool kHadamardRotate = false;
};

// ---- KvQuantTurbo (TurboQuant: K=4bit QJL4, V=2bit MSE) ----

template<class T>
struct KvQuantTrait<KvQuantTurbo, T> {
    // K: 4-bit QJL4 — 3-bit Lloyd-Max index + 1-bit QJL sign, packed as nibble
    using StorageK = uint4_t;
    // V: 2-bit Lloyd-Max MSE, 4 values packed per uint8_t
    using StorageV = uint2_t;

    using PointerK = SubBytePtr<uint4_t>;
    using PointerV = SubBytePtr<uint2_t>;

    static constexpr int kBitsK = 4;
    static constexpr int kBitsV = 2;

    static constexpr bool kQuantKV = true;

    // K: [mse_norm, qjl_norm], V: [norm]
    static constexpr int kParamCountK = 2;
    static constexpr int kParamCountV = 1;

    static constexpr int kv_quant = 42;

    static constexpr bool kHadamardRotate = true;
};

// ---------------------------------------------------------------------------
// Helper: select KvQuant tag from runtime quant_policy value
// ---------------------------------------------------------------------------

template<int quant_policy>
struct KvQuantFromPolicy {
    using type =
        std::conditional_t<quant_policy == 42,
                           KvQuantTurbo,
                           std::conditional_t<quant_policy & 0x04,
                                              KvQuantInt4,
                                              std::conditional_t<quant_policy & 0x08, KvQuantInt8, KvQuantNone>>>;
};

template<int quant_policy>
using KvQuantFromPolicy_t = typename KvQuantFromPolicy<quant_policy>::type;

}  // namespace turbomind::attention
