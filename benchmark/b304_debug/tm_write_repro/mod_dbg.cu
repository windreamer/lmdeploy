// Minimal pybind wrapper around the instrumented ProcessKV_v2 (no ContextGuard,
// no from_dlpack): isolates the TM binding machinery from the kernel itself.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include "kernel_dbg_inc.cu"

void my_process_kv(torch::Tensor k, torch::Tensor v,
                   int64_t ptrs_addr, int64_t cuq_addr, int64_t cuk_addr, int64_t cub_addr,
                   int64_t stride_c, int64_t stride_s, int64_t stride_h,
                   int64_t max_q_len, int64_t num_kv_heads, int64_t head_dim,
                   int64_t batch_size, int64_t block_len, int64_t stream_addr)
{
    turbomind::RopeKernelParam rope{};
    rope.type = turbomind::RopeType::kNull;
    turbomind::invokeProcessKV_v2<nv_bfloat16>(
        (char**)ptrs_addr, (const nv_bfloat16*)k.data_ptr(), (const nv_bfloat16*)v.data_ptr(),
        nullptr, nullptr, (const int*)cuq_addr, (const int*)cuk_addr, (const int*)cub_addr, nullptr,
        rope, 0, stride_c, stride_h, stride_s, (int)block_len, 0, 0, cutlass::FastDivmod(1),
        (int)max_q_len, (int)num_kv_heads, (int)head_dim, (int)batch_size,
        turbomind::QuantPolicy::kCacheKVInt8, (cudaStream_t)stream_addr);
}

PYBIND11_MODULE(mod_dbg, m) { m.def("my_process_kv", &my_process_kv); }
