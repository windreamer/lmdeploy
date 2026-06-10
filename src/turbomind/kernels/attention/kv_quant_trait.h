// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

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

template<typename KvQuant, typename T>
struct KvQuantTrait;

template<typename T>
struct KvQuantTrait<KvQuantNone, T> {
    using Storage = T;
    using Pointer = T*;

    static constexpr int  kBits       = bitsof<T>;
    static constexpr bool kQuantKV    = false;
    static constexpr int  kParamCount = 0;
    static constexpr int  kv_quant    = 0;
};

template<typename T>
struct KvQuantTrait<KvQuantInt8, T> {
    using Storage = uint8_t;
    using Pointer = uint8_t*;

    static constexpr int  kBits       = 8;
    static constexpr bool kQuantKV    = true;
    static constexpr int  kParamCount = 2;
    static constexpr int  kv_quant    = 8;
};

template<typename T>
struct KvQuantTrait<KvQuantInt4, T> {
    using Storage = uint4_t;
    using Pointer = SubBytePtr<uint4_t>;

    static constexpr int  kBits       = 4;
    static constexpr bool kQuantKV    = true;
    static constexpr int  kParamCount = 2;
    static constexpr int  kv_quant    = 4;
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
