// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/attention/kv_quant_trait.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/sub_byte_ptr.h"
#include <iostream>
#include <type_traits>

namespace turbomind {

namespace block {

template<class T, class KvQuant, int HeadDim, bool ShareKV = false>
struct Config {
    using Trait = attention::KvQuantTrait<KvQuant, T>;

    int head_num_;
    int block_len_;

    TM_HOST_DEVICE constexpr int t_bits() const
    {
        if constexpr (Trait::kQuantKV) {
            return bitsof<T>;
        }
        else {
            return 0;
        }
    }

    TM_HOST_DEVICE constexpr int k_bits() const
    {
        return Trait::kBitsK;
    }

    TM_HOST_DEVICE constexpr int v_bits() const
    {
        return Trait::kBitsV;
    }

    // Legacy alias
    TM_HOST_DEVICE constexpr int q_bits() const
    {
        return k_bits();
    }

    TM_HOST_DEVICE constexpr int k_param_count() const
    {
        return Trait::kParamCountK;
    }

    TM_HOST_DEVICE constexpr int v_param_count() const
    {
        return Trait::kParamCountV;
    }

    TM_HOST_DEVICE constexpr int head_dim() const
    {
        return HeadDim;
    }

    TM_HOST_DEVICE int head_num() const
    {
        return head_num_;
    }

    TM_HOST_DEVICE constexpr int block_len() const
    {
        return block_len_;
    }

    TM_HOST_DEVICE constexpr bool is_share_kv() const
    {
        return ShareKV;
    }
};

// Layout -> LayerId -> HeadId -> Timestep -> [Block] -> (k_data, v_data, k_param, v_param)

template<class T, class KvQuant, class Layout>
class Head {
public:
    using Trait = attention::KvQuantTrait<KvQuant, T>;
    using TK    = typename Trait::StorageK;
    using TV    = typename Trait::StorageV;

    TM_HOST_DEVICE Head(Layout layout, int layer_id, int head_id):
        layout_{layout}, layer_id_{layer_id}, head_id_{head_id}
    {
    }

    TM_HOST_DEVICE auto k_data(char* block, int ti) const
    {
        if constexpr (Trait::kBitsK < 8) {
            return SubBytePtr<TK>{block + layout_.k_data(layer_id_, head_id_, ti)};
        }
        else {
            return reinterpret_cast<TK*>(block + layout_.k_data(layer_id_, head_id_, ti));
        }
    }

    TM_HOST_DEVICE auto v_data(char* block, int ti) const
    {
        if constexpr (Trait::kBitsV < 8) {
            return typename Trait::PointerV{block + layout_.v_data(layer_id_, head_id_, ti)};
        }
        else {
            return reinterpret_cast<TV*>(block + layout_.v_data(layer_id_, head_id_, ti));
        }
    }

    TM_HOST_DEVICE T* k_param(char* block, int ti) const
    {
        return reinterpret_cast<T*>(block + layout_.k_param(layer_id_, head_id_, ti));
    }

    TM_HOST_DEVICE T* v_param(char* block, int ti) const
    {
        return reinterpret_cast<T*>(block + layout_.v_param(layer_id_, head_id_, ti));
    }

    TM_HOST_DEVICE void get_block_coord(int seq_ti, int& block_idx, int& block_ti) const
    {
        block_idx = seq_ti / block_len();
        block_ti  = seq_ti % block_len();
    }

    TM_HOST_DEVICE auto block_len() const
    {
        return layout_.config().block_len();
    }

    template<class Func>
    TM_HOST_DEVICE auto with(char** block_ptrs, int ti, Func&& func) const
    {
        int block_id;
        int block_ti;
        get_block_coord(ti, block_id, block_ti);

        char* block = block_ptrs[block_id];

        return ((Func &&) func)(
            k_data(block, block_ti), v_data(block, block_ti), k_param(block, block_ti), v_param(block, block_ti));
    }

private:
    Layout layout_;

    int layer_id_;
    int head_id_;
};

// L(H2SDQ+H2S2T)
template<class Config_>
struct Layout {

    using Config = Config_;

    Config config_;

    // This trivial ctor is defined for CTAD
    TM_HOST_DEVICE Layout(Config config): config_{config} {}

    TM_HOST_DEVICE const Config& config() const
    {
        return config_;
    }

    TM_HOST_DEVICE constexpr bool is_share_kv() const
    {
        return config().is_share_kv();
    }

    TM_HOST_DEVICE constexpr int kv_num() const
    {
        return is_share_kv() ? 1 : 2;
    }

    // ---- Token data/param sizes ----
    // For asymmetric K/V, use k_* and v_* variants.
    // Legacy accessors delegate to k_* for backward compat.

    TM_HOST_DEVICE int k_token_data_size() const
    {
        return config().k_bits() * config().head_dim() / 8;
    }

    TM_HOST_DEVICE int v_token_data_size() const
    {
        return config().v_bits() * config().head_dim() / 8;
    }

    TM_HOST_DEVICE int k_token_param_size() const
    {
        return config().t_bits() * config().k_param_count() / 8;
    }

    TM_HOST_DEVICE int v_token_param_size() const
    {
        return config().t_bits() * config().v_param_count() / 8;
    }

    TM_HOST_DEVICE int token_data_size() const
    {
        return k_token_data_size();
    }

    TM_HOST_DEVICE int token_param_size() const
    {
        return k_token_param_size();
    }

    TM_HOST_DEVICE int k_head_data_size() const
    {
        return config().block_len() * k_token_data_size();
    }

    TM_HOST_DEVICE int v_head_data_size() const
    {
        return is_share_kv() ? 0 : config().block_len() * v_token_data_size();
    }

    TM_HOST_DEVICE int k_head_param_size() const
    {
        return config().block_len() * k_token_param_size();
    }

    TM_HOST_DEVICE int v_head_param_size() const
    {
        return is_share_kv() ? 0 : config().block_len() * v_token_param_size();
    }

    TM_HOST_DEVICE int head_data_size() const
    {
        return k_head_data_size();
    }

    TM_HOST_DEVICE int head_param_size() const
    {
        return k_head_param_size();
    }

    TM_HOST_DEVICE int layer_size() const
    {
        return config().head_num() * (k_head_data_size() + v_head_data_size())
               + config().head_num() * (k_head_param_size() + v_head_param_size());
    }

    TM_HOST_DEVICE int block_size(int layer_num) const
    {
        return layer_size() * layer_num;
    }

    TM_HOST_DEVICE int k_data(int layer, int head, int token) const
    {
        return layer_data(layer) + head_data(head) + k_token_data(token);
    }

    TM_HOST_DEVICE int v_data(int layer, int head, int token) const
    {
        return layer_data(layer) + head_data(head) + k_head_data_size() + v_token_data(token);
    }

    TM_HOST_DEVICE int k_param(int layer, int head, int token) const
    {
        return layer_param(layer) + head_param(head) + k_token_param(token);
    }

    TM_HOST_DEVICE int v_param(int layer, int head, int token) const
    {
        return layer_param(layer) + head_param(head) + k_head_param_size() + v_token_param(token);
    }

    TM_HOST_DEVICE int layer_data(int layer) const
    {
        return layer * layer_size();
    }

    TM_HOST_DEVICE int layer_param(int layer) const
    {
        return layer_data(layer) + config_.head_num() * (k_head_data_size() + v_head_data_size());
    }

    TM_HOST_DEVICE int head_data(int head) const
    {
        return head * (k_head_data_size() + v_head_data_size());
    }

    TM_HOST_DEVICE int head_param(int head) const
    {
        return head * (k_head_param_size() + v_head_param_size());
    }

    TM_HOST_DEVICE int k_token_data(int ti) const
    {
        return ti * k_token_data_size();
    }

    TM_HOST_DEVICE int v_token_data(int ti) const
    {
        return ti * v_token_data_size();
    }

    TM_HOST_DEVICE int k_token_param(int ti) const
    {
        return ti * k_token_param_size();
    }

    TM_HOST_DEVICE int v_token_param(int ti) const
    {
        return ti * v_token_param_size();
    }

    // Legacy aliases (K-only, for backward compat with symmetric K/V)
    TM_HOST_DEVICE int token_data(int ti) const
    {
        return k_token_data(ti);
    }

    TM_HOST_DEVICE int token_param(int ti) const
    {
        return k_token_param(ti);
    }
};

template<class Config>
void dump(const Layout<Config>& layout)
{
    std::cout << "head_dim: " << layout.config().head_dim() << "\n";
    std::cout << "head_num: " << layout.config().head_num() << "\n";
    std::cout << "block_len: " << layout.config().block_len() << "\n";
    std::cout << "k_bits: " << layout.config().k_bits() << "\n";
    std::cout << "v_bits: " << layout.config().v_bits() << "\n";
    std::cout << "t_bits: " << layout.config().t_bits() << "\n";
    std::cout << "k_token_data_size: " << layout.k_token_data_size() << "\n";
    std::cout << "v_token_data_size: " << layout.v_token_data_size() << "\n";
    std::cout << "k_token_param_size: " << layout.k_token_param_size() << "\n";
    std::cout << "v_token_param_size: " << layout.v_token_param_size() << "\n";
    std::cout << "k_head_data_size: " << layout.k_head_data_size() << "\n";
    std::cout << "v_head_data_size: " << layout.v_head_data_size() << "\n";
    std::cout << "k_head_param_size: " << layout.k_head_param_size() << "\n";
    std::cout << "v_head_param_size: " << layout.v_head_param_size() << "\n";
    std::cout << "layer_size: " << layout.layer_size() << "\n";
}

}  // namespace block

}  // namespace turbomind
