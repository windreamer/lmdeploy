// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdint>
#include <memory>
#include <optional>

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/activation_ops.h"
#include "src/turbomind/kernels/norm/rms_norm.h"

namespace py = pybind11;

namespace turbomind::python_norm_act {
namespace {

using core::Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

const TensorPtr& TensorArg(const TensorPtr& p, const char* name)
{
    if (!p) {
        throw py::value_error(std::string(name) + " is null");
    }
    return p;
}

const int* ValidTokensPtr(const std::optional<TensorPtr>& valid_tokens)
{
    return valid_tokens ? static_cast<const int*>((*valid_tokens)->raw_data()) : nullptr;
}

}  // namespace

void bind_norm_act(py::module_& m)
{
    // out <- rmsnorm(x) * w. x/out: [num, dims], w: [dims].
    m.def("rms_norm",
          [](const TensorPtr& out, const TensorPtr& x, const TensorPtr& w, float eps, bool zero_centered, std::uintptr_t stream_ptr) {
              invokeRMSNorm(*TensorArg(out, "out"),
                            *TensorArg(x, "x"),
                            *TensorArg(w, "w"),
                            eps,
                            zero_centered,
                            reinterpret_cast<cudaStream_t>(stream_ptr));
          },
          "out"_a,
          "x"_a,
          "w"_a,
          "eps"_a,
          "zero_centered"_a = false,
          "stream_ptr"_a);

    // In-place qk rmsnorm over a packed qkv buffer: qkv [token, (q + k + v) * head_dim, head_dim],
    // normalized per q/k head with the respective per-head-dim weights. v is left untouched.
    m.def("qk_rms_norm",
          [](const TensorPtr& qkv,
             const TensorPtr& q_weight,
             const TensorPtr& k_weight,
             int              q_head_num,
             int              k_head_num,
             float            eps,
             bool             zero_centered,
             std::uintptr_t   stream_ptr) {
              invokeQkRMSNorm(*TensorArg(qkv, "qkv"),
                              *TensorArg(q_weight, "q_weight"),
                              *TensorArg(k_weight, "k_weight"),
                              q_head_num,
                              k_head_num,
                              eps,
                              zero_centered,
                              reinterpret_cast<cudaStream_t>(stream_ptr));
          },
          "qkv"_a,
          "q_weight"_a,
          "k_weight"_a,
          "q_head_num"_a,
          "k_head_num"_a,
          "eps"_a,
          "zero_centered"_a = false,
          "stream_ptr"_a);

    // In-place fused add-residual rmsnorm, matching the triton add_rms_norm
    // semantics: residual <- x + residual (+ bias); hidden_states <- norm(residual) * w.
    // Both tensors are [num, dims] and updated in place.
    m.def("residual_rms_norm",
          [](const TensorPtr&             hidden_states,
             const TensorPtr&             residual,
             const TensorPtr&             w,
             const std::optional<TensorPtr>& bias,
             float                        eps,
             bool                         zero_centered,
             std::uintptr_t               stream_ptr) {
              invokeResidualBiasRMSNorm(TensorArg(hidden_states, "hidden_states")->raw_data(),
                                        TensorArg(residual, "residual")->raw_data(),
                                        TensorArg(w, "w")->raw_data(),
                                        bias ? (*bias)->raw_data() : nullptr,
                                        hidden_states->dtype(),
                                        static_cast<int>(hidden_states->shape(-1)),
                                        static_cast<int>(hidden_states->size() / hidden_states->shape(-1)),
                                        eps,
                                        zero_centered,
                                        reinterpret_cast<cudaStream_t>(stream_ptr));
          },
          "hidden_states"_a,
          "residual"_a,
          "w"_a,
          "bias"_a = std::nullopt,
          "eps"_a,
          "zero_centered"_a = false,
          "stream_ptr"_a);

    // In-place gated silu activation: gate <- silu(gate) * up, gate/up: [num, dim].
    // `valid_tokens` optionally points to a device-side int tensor holding the
    // number of valid (unpadded) tokens in padded buffers.
    m.def("silu_and_mul",
          [](const TensorPtr& gate, const TensorPtr& up, const std::optional<TensorPtr>& valid_tokens, std::uintptr_t stream_ptr) {
              Tensor& gate_ref      = *TensorArg(gate, "gate");
              const int* valid_ptr  = ValidTokensPtr(valid_tokens);
              Activation(gate_ref,
                         *TensorArg(up, "up"),
                         ActivationType::kSilu,
                         valid_ptr,
                         reinterpret_cast<cudaStream_t>(stream_ptr));
          },
          "gate"_a,
          "up"_a,
          "valid_tokens"_a = std::nullopt,
          "stream_ptr"_a);
}

}  // namespace turbomind::python_norm_act
