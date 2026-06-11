// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/sub_byte_ptr.h"
#include <type_traits>

namespace turbomind::attention {

struct KvQuantNone {
};
struct KvQuantInt8 {
};
struct KvQuantInt4 {
};

template<typename T>
struct AffineDequant {
    template<int N, typename TK>
    __device__ static Array<T, N> convert(const Array<TK, N>& raw, T p0, T p1);

    __device__ static T apply(T val, T scale, T zero, int /*head_dim*/)
    {
        return val * scale + zero;
    }
};

template<typename KvQuant, typename T>
struct KvQuantTrait;

template<typename T>
struct KvQuantTrait<KvQuantNone, T> {
    using StorageK = T;
    using StorageV = T;
    using PointerK = T*;
    using PointerV = T*;
    using DequantK = AffineDequant<T>;
    using DequantV = AffineDequant<T>;

    static constexpr int  kBitsK       = bitsof<T>;
    static constexpr int  kBitsV       = bitsof<T>;
    static constexpr bool kQuantKV     = false;
    static constexpr int  kParamCountK = 0;
    static constexpr int  kParamCountV = 0;
    static constexpr int  kv_quant     = 0;
};

template<typename T>
struct KvQuantTrait<KvQuantInt8, T> {
    using StorageK = uint8_t;
    using StorageV = uint8_t;
    using PointerK = uint8_t*;
    using PointerV = uint8_t*;
    using DequantK = AffineDequant<T>;
    using DequantV = AffineDequant<T>;

    static constexpr int  kBitsK       = 8;
    static constexpr int  kBitsV       = 8;
    static constexpr bool kQuantKV     = true;
    static constexpr int  kParamCountK = 2;
    static constexpr int  kParamCountV = 2;
    static constexpr int  kv_quant     = 8;
};

template<typename T>
struct KvQuantTrait<KvQuantInt4, T> {
    using StorageK = uint4_t;
    using StorageV = uint4_t;
    using PointerK = SubBytePtr<uint4_t>;
    using PointerV = SubBytePtr<uint4_t>;
    using DequantK = AffineDequant<T>;
    using DequantV = AffineDequant<T>;

    static constexpr int  kBitsK       = 4;
    static constexpr int  kBitsV       = 4;
    static constexpr bool kQuantKV     = true;
    static constexpr int  kParamCountK = 2;
    static constexpr int  kParamCountV = 2;
    static constexpr int  kv_quant     = 4;
};

template<int quant_policy>
struct KvQuantFromPolicy {
    using type = std::conditional_t<(quant_policy & 0x04),
                                    KvQuantInt4,
                                    std::conditional_t<(quant_policy & 0x08), KvQuantInt8, KvQuantNone>>;
};

template<int quant_policy>
using KvQuantFromPolicy_t = typename KvQuantFromPolicy<quant_policy>::type;

}  // namespace turbomind::attention
