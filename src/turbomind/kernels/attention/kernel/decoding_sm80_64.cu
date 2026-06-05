// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/arch.h"
#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/block_iterator.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_81616.h"
#include "src/turbomind/kernels/attention/impl_simt.h"
#include "src/turbomind/kernels/attention/mainloop.h"
#include "src/turbomind/kernels/attention/mainloop_sm80.h"
#include "src/turbomind/kernels/attention/registrar.h"

namespace turbomind::attention {

constexpr int kHeadDim = 64;
constexpr int kCTA_S   = 64;
constexpr int kWARP_S  = 16;

template<class Mainloop_, class CacheIter>
using KT = AttentionUniversal<arch::Sm80, Mainloop_, CacheIter, DecodingCtaMap>;

// T==Tkv, Qh<=2: SIMT, stages=3
template<class T, int Qh>
using Decoding_SIMT =
    KT<Mainloop<Sm80_CpAsync<3>, Impl<MMA_SIMT, T, KvQuantNone, Qh, 1, kCTA_S, Qh, 1, kWARP_S, kHeadDim, 3>>,
       GetBlockIterFactory<T, KvQuantNone, kCTA_S, kHeadDim>>;

// Qh>2: MMA_81616; Stages=3 for T==Tkv, Stages=5 for quant Tkv
// Qh = (Qh_+7)/8*8: Qh_=3..8→Qh=8, Qh_=9→Qh=16
template<class T, class KvQuant, int Qh, int Stages>
using Decoding_MMA =
    KT<Mainloop<Sm80_CpAsync<Stages>, Impl<MMA_81616, T, KvQuant, Qh, 1, kCTA_S, Qh, 1, kWARP_S, kHeadDim, Stages>>,
       GetBlockIterFactory<T, KvQuant, kCTA_S, kHeadDim>>;

namespace {
Registrar reg([](Collector& c) {
    c.add<Decoding_SIMT<half, 1>>();
    c.add<Decoding_SIMT<half, 2>>();
    c.add<Decoding_MMA<half, KvQuantNone, 8, 3>>();
    c.add<Decoding_MMA<half, KvQuantNone, 16, 3>>();
    c.add<Decoding_MMA<half, KvQuantInt8, 8, 5>>();
    c.add<Decoding_MMA<half, KvQuantInt8, 16, 5>>();
    c.add<Decoding_MMA<half, KvQuantInt4, 8, 5>>();
    c.add<Decoding_MMA<half, KvQuantInt4, 16, 5>>();
    c.add<Decoding_MMA<half, KvQuantTurbo, 16, 5>>();

#if ENABLE_BF16
    c.add<Decoding_SIMT<nv_bfloat16, 1>>();
    c.add<Decoding_SIMT<nv_bfloat16, 2>>();
    c.add<Decoding_MMA<nv_bfloat16, KvQuantNone, 8, 3>>();
    c.add<Decoding_MMA<nv_bfloat16, KvQuantNone, 16, 3>>();
    c.add<Decoding_MMA<nv_bfloat16, KvQuantInt8, 8, 5>>();
    c.add<Decoding_MMA<nv_bfloat16, KvQuantInt8, 16, 5>>();
    c.add<Decoding_MMA<nv_bfloat16, KvQuantInt4, 8, 5>>();
    c.add<Decoding_MMA<nv_bfloat16, KvQuantInt4, 16, 5>>();
    c.add<Decoding_MMA<nv_bfloat16, KvQuantTurbo, 16, 5>>();
#endif
});
}  // namespace

}  // namespace turbomind::attention
