# B3-06 Acceptance Report — Harness Rungs for MoE / int8-KV / TP=2

Status: **closed with findings** (2026-11-17, device: RTX 3090 24 GiB,
branch `graft/w4a16-core`, commit in the JSON reports).

Scope adjustments applied: the **GDN rung is out** (batch 3 rolled GDN
back, see ROADMAP two-stage decision; `fla/` assets stay). Everything else
of the ticket is delivered.

## 1. Placement decisions (放置决策)

| deliverable                              | location                                                                         | why                                                                                                                                            |
| ---------------------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| MoE / int8-KV / TP=2 rungs               | `benchmark/w4a16_acceptance.py`, `--suite {gemm,moe,int8kv,tp2}`                 | one harness, one decode-timing core; the ticket 05 matrix stays `--suite gemm` with unchanged semantics (regression re-run: tiny preset green) |
| tiny dense checkpoint with `head_dim=64` | `benchmark/tiny_b306_checkpoints.py::gen_tiny_dense_ct`                          | the ticket 03 tiny carries `head_dim=32`, which the TM int8-KV geometry probe rejects; 64 passes                                               |
| tiny MoE checkpoint                      | built by the ticket 02 golden gate (`_make_tiny_moe_pair`), reused not rewritten | proven ct Mixtral pair; the glm4_moe architecture attempt is recorded as a source gap (below)                                                  |
| reports                                  | `reports/w4a16/b306_{moe,int8kv,tp2}/report.{json,md}`                           | sibling of the ticket 05 `reports/w4a16/` output; JSON for machines, MD for review                                                             |
| SPEC-B3 / ROADMAP closure                | section 5 below                                                                  | `.scratch/` is outside this ticket's write scope, so the closure text is delivered ready to paste                                              |

## 2. Measured data (实测)

### MoE rung (`b306_moe`, full rung set, green)

Tiny ct Mixtral (8 experts, top-2, group 128), decode 32 tokens, 2 timed
passes + warmup, seq 256:

| batch | triton fused_moe tok/s | TM grouped W4A16 tok/s | TM/Triton |
| ----- | ---------------------- | ---------------------- | --------- |
| 1     | 801.5                  | 743.7                  | 0.93      |
| 8     | 794.9                  | 868.3                  | 1.09      |
| 32    | 827.6                  | 804.5                  | 0.97      |

S1 numeric gate: **PASS** — worst truncated KL 1.21e-03 ≤ 1.0e-02
(`MOE_TINY_KL_THRESHOLD`), top-1 agreement recorded in the report. The KL
carries the full int4 error because the reference is the unquantized twin
(same judge convention as the ticket 02 golden gate).

Smoke tier (b=1, 8 tokens, 0 timed passes) also green: worst KL 1.21e-03.

### int8-KV rung (`b306_int8kv`, off-route green, on-route blocked by source defects)

Stock int8 KV path (knob off), same rung geometry:

| batch | stock int8 KV tok/s |
| ----- | ------------------- |
| 1     | 811.9               |
| 8     | 799.9               |
| 32    | 768.0               |

The TM int8-KV on-route (`LMDEPLOY_TM_ATTN_INT8_KV=on`) cannot complete
engine start on this tree. Four source defects, all reproduced and
documented in `b306_int8kv/report.md` (notes section):

1. `backends/cuda/attention/int8_kv.py` imports `from ..tm_kv import ...`
   from the `backends.cuda.attention` package; that resolves to the
   nonexistent `backends.cuda.tm_kv` → `ModuleNotFoundError` at engine
   start.
2. `TmInt8KVAttentionImpl._forward_tm_decode` calls
   `tm.tm_attn_decode(q, out, ptrs, cu)`; the binding
   (`src/turbomind/python/attn_bench_bind.cpp`) requires
   `(q, k_rows, v_rows, out, block_ptrs, cu_block_nums)` with `[B, Hkv*D]`
   newest-token rows.
3. The capture guard (`torch.cuda.is_current_stream_capturing()`) returns
   False during graph capture, so the TM decode path runs inside capture
   and hits defect 2 there.
4. With 1–3 shimmed in the harness, engine start still fails in the
   capture phase (`cudaErrorStreamCaptureInvalidated`): the non-TM
   fallback branch under the TM int8 pool views is not capture-safe.

The KL ≤ 2e-3 numeric gate therefore stays **unmeasured**; the rung
pipeline (routes, rungs, gate, report) is ready and reruns unblocked once
the source defects are fixed. `lmdeploy/` is outside this ticket's write
scope; the harness carries two documented diagnostic shims (module alias +
class-level decode re-bind) that reach the capture phase.

### TP=2 (`b306_tp2`, pipeline-level ready, declared)

12 tp=2 rungs exist in the h200 preset matrix; all 12 auto-skip on this
single-GPU host via `rung_skip` with an explicit reason; `RouteEngine`
accepts `tp=2` on both engine configs. Measured tp=2 numbers are an H200
item (`--preset h200`).

## 3. Leftovers (遗留)

1. **int8-KV source defects 1–4 above** — owner: `lmdeploy/pytorch/backends/cuda/attention/` + `tm_kv.py`. Rerun:
   `python benchmark/w4a16_acceptance.py --suite int8kv` after the fix; the numeric gate (KL ≤ 2e-3) and the throughput comparison complete then.
2. **TP=2 measured rungs** — H200 host, `--preset h200 --tps 1,2`.
3. **Upstream CI job definitions** — CI batch, as planned in SPEC-B3.
4. **GLM MoE family (glm4_moe) TP=1 serving bug** — `Glm4MoE.forward`
   broadcasts `shared_states` `[1,T,H]` into `out_states` `[T,H]` on the
   dp=1/tp=1 path; any GLM-4.5-family MoE model fails single-GPU decode.
   Discovered while attempting the glm52 MoE geometry on the glm4_moe
   architecture. Owner: `lmdeploy/pytorch/models/glm4_moe.py`.
5. **SPEC-B3/ROADMAP sync** — section 5 below; `.scratch/` is read-only
   for this ticket.

## 4. Reuse ledger

- decode timing core, rung/skip logic, report skeleton: ticket 05 harness (this file's host, untouched paths).
- numeric helpers (`top20_of`, `truncated_kl`) and gate thresholds: `tests/pytorch/engine/w4a16_golden.py` + the MoE golden gate constants.
- tiny MoE checkpoint pair: ticket 02 golden-gate builder.
- tiny dense checkpoint: new (`head_dim=64` variant, tiny_ct conventions).
- GDN: no rung, per scope adjustment; no GDN assets touched.

## 5. SPEC-B3 / ROADMAP closure text (paste-ready)

For `.scratch/tm-gemm-graft/SPEC-B3.md` (add a "Status (2026-11-17, B3-06)" section):

- MoE grouped (B3-02): implemented and linked — golden gate
  `tests/pytorch/engine/test_w4a16_moe_golden_gate.py`; harness rung
  `benchmark/w4a16_acceptance.py --suite moe`; measured green (numeric
  gate PASS, TM/Triton 0.93/1.09/0.97 at b=1/8/32 on the tiny pair).
- int8-KV (B3-04): implemented and linked — bridge probes B3-01, kernel
  contract `tests/pytorch/kernel/test_tm_int8_kv.py`; harness rung
  `--suite int8kv`; off-route green, on-route blocked by four source
  defects in the attention integration (documented in
  `benchmark/b3_06_report.md` §2); numeric gate deferred to the source fix.
- TP=2: pipeline ready, measured numbers deferred to H200 (`--suite tp2`
  declaration + `--preset h200`).
- GDN: out of batch 3 per the 2026-11-17 rollback; the two-stage ROADMAP
  entry stands.

For `.scratch/tm-gemm-graft/ROADMAP.md` (packed-KV row, 批次 column note):
"B3-06 harness rung ready; on-route blocked by attention-integration
source defects (see benchmark/b3_06_report.md §2); H200 复测随 CI 批."
