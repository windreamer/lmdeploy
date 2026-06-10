// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/arch.h"
#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/block_iterator.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_simt.h"
#include "src/turbomind/kernels/attention/mainloop.h"
#include "src/turbomind/kernels/attention/mainloop_sm80.h"
#include "src/turbomind/kernels/attention/registrar.h"

namespace turbomind::attention {

constexpr int kHeadDim = 192;
constexpr int kCTA_S   = 64;
constexpr int kWARP_S  = 16;
constexpr int kStages  = 3;
constexpr int kQh      = 1;

// HeadDim=192 uses SIMT+kStages for all TK (incl. uint8_t), kQh=1 only
template<class T, class KvQuant>
using KT = AttentionUniversal<
    arch::Sm80,
    Mainloop<Sm80_CpAsync<kStages>, Impl<MMA_SIMT, T, KvQuant, kQh, 1, kCTA_S, kQh, 1, kWARP_S, kHeadDim, kStages>>,
    GetBlockIterFactory<T, KvQuant, kCTA_S, kHeadDim>,
    DecodingCtaMap>;

namespace {
Registrar reg([](Collector& c) {
    c.add<KT<half, KvQuantNone>>();
    c.add<KT<half, KvQuantInt8>>();

#if ENABLE_BF16
    c.add<KT<nv_bfloat16, KvQuantNone>>();
    c.add<KT<nv_bfloat16, KvQuantInt8>>();
#endif
});
}  // namespace

}  // namespace turbomind::attention
