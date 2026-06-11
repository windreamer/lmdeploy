// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/attention/kv_quant_trait.h"
#include "src/turbomind/kernels/attention/quantization.h"

namespace turbomind::attention {

template<typename T>
template<int N, typename TK>
__device__ Array<T, N> AffineDequant<T>::convert(const Array<TK, N>& raw, T p0, T p1)
{
    if constexpr (std::is_same_v<TK, T>) {
        return raw;
    }
    else {
        auto        mid = ConvertKvCache<TK, T>::convert(raw);
        Array<T, N> result;
        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            result[i] = mid[i] * p0 + p1;
        }
        return result;
    }
}

}  // namespace turbomind::attention
