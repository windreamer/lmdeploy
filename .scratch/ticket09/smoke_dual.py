# Ticket 09 deliverable 3: dual-path smoke on tiny ct model.
#
# Same repacked AutoAWQ tensors (from ct_to_awq) fed to:
#   path A: lmdeploy pytorch AWQ Triton kernel  (AwqLinearW4A16Impl, impl level)
#   path B: TurboMind GEMM channel              (TurbomindAwqLinearW4A16Impl)
# Compare logits: KL(A||B) on softmax, argmax agreement, max abs diff,
# plus each path vs a float dequant reference.
#
# Note on scope: impl-level instantiation (no engine / no OpsBackend), which is
# the agreed fallback. Backend-builder gate `_turbomind_support_reason` refuses
# non-sm90, so on this RTX 3090 (sm86) we bypass it deliberately; whether the
# TM GEMM registry actually serves u4 g128 on sm86 is exactly what we probe.

from __future__ import annotations

import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from ct_to_awq import convert_layer  # noqa: E402

torch.manual_seed(9)
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'tiny_ct_model')
torch.backends.cuda.matmul.allow_tf32 = False


def softmax_kl(a: torch.Tensor, b: torch.Tensor) -> float:
    """KL(P||Q), P from path A logits, Q from path B, over last dim."""
    p = torch.log_softmax(a.float(), dim=-1)
    q = torch.log_softmax(b.float(), dim=-1)
    return (p.exp() * (p - q)).sum(-1).mean().item()


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def dequant_ref(ct: dict) -> torch.Tensor:
    packed, scale = ct['weight_packed'], ct['weight_scale']
    n, k = packed.shape[0], packed.shape[1] * 8
    shifts = torch.arange(8, dtype=torch.int32) * 4
    codes = ((packed[..., None] >> shifts) & 15).reshape(n, k).float() - 8
    return codes * scale.repeat_interleave(k // scale.shape[1], dim=1).float()


def main():
    from lmdeploy import turbomind
    if not turbomind.is_available():
        sys.exit('TurboMind native modules not built')

    from lmdeploy.pytorch.backends.cuda.awq_modules import AwqLinearW4A16Impl
    from lmdeploy.pytorch.backends.cuda.turbomind_awq_modules import TurbomindAwqLinearW4A16Impl

    print(f"device: {torch.cuda.get_device_name(0)} sm{torch.cuda.get_device_capability()}")
    files = sorted(glob.glob(os.path.join(MODEL_DIR, '*.pt')))
    rows = []
    for f in files:
        name = os.path.basename(f)[:-3]
        ct = torch.load(f)
        awq, _ = convert_layer(ct)
        n, k = ct['weight_packed'].shape[0], ct['weight_packed'].shape[1] * 8

        qw = awq['qweight'].cuda()
        sc = awq['scales'].cuda()
        qz = awq['qzeros'].cuda()
        bias = ct['bias'].cuda()
        m = 4
        x = torch.randn(m, k, dtype=torch.float16, device='cuda')

        ref = torch.nn.functional.linear(x, dequant_ref(ct).half().cuda()) + bias

        impl_awq = AwqLinearW4A16Impl(k, n, 4, 128)
        out_awq = impl_awq.forward(x, qw, sc, qz, bias=bias).squeeze(0)

        impl_tm = TurbomindAwqLinearW4A16Impl(k, n, 4, 128)
        try:
            impl_tm.update_weights(qw, sc, qz)
        except Exception as e:
            print(f"[{name}] TM update_weights FAILED: {type(e).__name__}: {e}")
            continue
        with torch.no_grad():
            out_tm = impl_tm.forward(x, qw, sc, qz, bias=bias).squeeze(0)

        kl = softmax_kl(out_awq, out_tm)
        agree = (out_awq.argmax(-1) == out_tm.argmax(-1)).float().mean().item()
        maxabs = (out_awq.float() - out_tm.float()).abs().max().item()
        rows.append((name, k, n, kl, agree, maxabs,
                     rel_err(out_awq, ref), rel_err(out_tm, ref)))
        print(f"[{name}] K={k} N={n} m={m}: KL(A||B)={kl:.3e} argmax_agree={agree:.3f} "
              f"maxabs={maxabs:.3e} relerr(A,ref)={rows[-1][-2]:.2e} relerr(B,ref)={rows[-1][-1]:.2e}")

    print('\nsummary (worst over layers):')
    worst_kl = max(r[3] for r in rows)
    worst_agree = min(r[4] for r in rows)
    print(f"  max KL = {worst_kl:.3e}, min argmax agreement = {worst_agree:.3f}")
    json.dump(
        [{'layer': r[0], 'k': r[1], 'n': r[2], 'kl': r[3], 'argmax_agree': r[4],
          'max_abs': r[5], 'rel_err_awq_vs_ref': r[6], 'rel_err_tm_vs_ref': r[7]}
         for r in rows],
        open(os.path.join(os.path.dirname(__file__), 'smoke_dual_results.json'), 'w'),
        indent=2)


if __name__ == '__main__':
    main()
