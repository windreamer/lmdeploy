// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/data_type.h"

namespace turbomind {

template<class T>
struct SubBytePtr {

    // Physical storage word type for word-aligned addressing
    using Storage                        = typename detail::__storage_of<T>::type;
    static constexpr int kBitsPerStorage = sizeof(Storage) * 8;
    static constexpr int kValuesPerWord  = kBitsPerStorage / (int)bitsof<T>;
    static constexpr int kBytesPerWord   = sizeof(Storage);

    constexpr SubBytePtr() = default;

    constexpr __host__ __device__ explicit SubBytePtr(T* ptr): ptr_((char*)ptr) {}

    constexpr __host__ __device__ SubBytePtr(char* ptr): ptr_(ptr) {}

    // Word-aligned addressing: element i maps to the storage word containing it.
    // For uint2_t: elements 0-15 → word 0, elements 16-31 → word 1, etc.
    // For uint4_t: elements 0-7 → word 0, elements 8-15 → word 1, etc.
    __host__ __device__ T& operator[](int i)
    {
        return *reinterpret_cast<T*>(ptr_ + (i / kValuesPerWord) * kBytesPerWord);
    }

    __host__ __device__ const T& operator[](int i) const
    {
        return *reinterpret_cast<const T*>(ptr_ + (i / kValuesPerWord) * kBytesPerWord);
    }

    // Word-aligned pointer arithmetic: advance by n logical elements.
    // NOTE: n is rounded DOWN to the nearest whole word boundary.
    // SubBytePtr + 1 does NOT move the pointer (1 < kValuesPerWord).
    // All existing callers pass n that is a multiple of kValuesPerWord.
    friend __host__ __device__ SubBytePtr operator+(const SubBytePtr a, int n)
    {
        return SubBytePtr{a.ptr_ + (n / kValuesPerWord) * kBytesPerWord};
    }

    friend __host__ __device__ SubBytePtr operator+(int n, const SubBytePtr a)
    {
        return a + n;
    }

    friend __host__ __device__ bool operator==(const SubBytePtr& a, const SubBytePtr& b)
    {
        return a.ptr_ == b.ptr_;
    }

    __host__ __device__ explicit operator T*() const
    {
        return (T*)ptr_;
    }

    char* ptr_;
};

template<class T>
struct get_pointer_type_t<T, std::enable_if_t<bitsof<T> % 8 != 0>> {
    using type = SubBytePtr<T>;
};

}  // namespace turbomind
