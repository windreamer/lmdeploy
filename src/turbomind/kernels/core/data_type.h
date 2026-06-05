// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_fp16.h>
#if ENABLE_BF16
#include <cuda_bf16.h>
#endif

#include <cstdint>

#include "src/turbomind/core/data_type.h"

namespace turbomind {

namespace detail {

// Physical storage types for sub-byte logical types.
// Each packs multiple logical values into a single machine word via the `x` member.
// The type of `x` is the physical storage word — used by SubBytePtr for word-aligned addressing.
struct __uint4_t {
    uint32_t x;  // 8× packing: 8 four-bit values per uint32_t word
};

struct __uint2_t {
    uint32_t x;  // 16× packing: 16 two-bit values per uint32_t word
};

// Trait: deduce the physical storage word type for a (possibly sub-byte) logical type.
// Default: the type itself (non-sub-byte types).
// Specializations: auto-derive from the `x` member of the corresponding __storage struct,
//   so changing __uintX_t::x automatically updates the storage word type.
template<class T>
struct __storage_of {
    using type = T;
};

template<>
struct __storage_of<uint4_t> {
    using type = decltype(__uint4_t::x);
};

template<>
struct __storage_of<uint2_t> {
    using type = decltype(__uint2_t::x);
};

}  // namespace detail

template<class T, class SFINAE = void>
struct get_pointer_type_t {
    using type = T*;
};

template<class T>
using get_pointer_type = typename get_pointer_type_t<T>::type;

}  // namespace turbomind
