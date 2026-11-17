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

### int8-KV rung (`b306_int8kv`, full rung set, green on the fixed source)

Stock int8 KV (knob off) vs the TM int8 block layout (knob on), decode
32 tokens, 2 timed passes + warmup, seq 256. The B3-04 owner fixed the
four attention-integration source defects this ticket had found and
documented (mis-targeted `..tm_kv` import, decode call missing the
newest-token K/V rows, capture guard letting the TM path run inside graph
capture, non-capture-safe fallback); the harness shims are retired — a
source-state probe now only reports fixed/PRESENT, so the on-route
numbers below come from the real path:

| batch | stock int8 KV tok/s | TM int8 layout tok/s | TM/stock |
|---|---|---|---|
| 1 | 797.6 | 63.7 | 0.08 |
| 8 | 797.5 | 492.5 | 0.64 |
| 32 | 749.5 | 920.7 | 1.23 |

Numeric gate: **PASS** — worst truncated KL 3.50e-06 ≤ 2e-3, top-1
agreement 1.00. No threshold amendment needed; the measured distribution
sits three orders below the bar, so `INT8_KV_KL_THRESHOLD = 2e-3` stays.

Report items, not gates: the TM layout loses at b=1/8 on this card (the
TM decode kernel dispatches eager — the capture guard keeps graph capture
on the Triton fallback — so b=1 carries per-step dispatch overhead), and
wins at b=32. Batch-2 kernel-level data (TM 1.34–1.86× at production
depths) remains the anchor; the H200 matrix with graph-captured decode is
where the production verdict lands.

### TP=2 (`b306_tp2`, pipeline-level ready, declared)

12 tp=2 rungs exist in the h200 preset matrix; all 12 auto-skip on this
single-GPU host via `rung_skip` with an explicit reason; `RouteEngine`
accepts `tp=2` on both engine configs. Measured tp=2 numbers are an H200
item (`--preset h200`).

## 3. Leftovers (遗留)

1. **int8-KV source defects** — resolved: the B3-04 owner fixed all four
   in `lmdeploy/pytorch/backends/cuda/attention/int8_kv.py` + `tm_kv.py`;
   the clean-source run measures the numeric gate PASS (KL 3.50e-06). The
   harness shims are retired; `_tm_int8_kv_source_state` keeps a
   fixed/PRESENT probe so a stale checkout fails loudly instead of
   silently masking the real path.
2. **TP=2 measured rungs** — H200 host, `--preset h200 --tps 1,2`.
3. **Upstream CI job definitions** — CI batch, as planned in SPEC-B3.
4. **GLM MoE family (glm4_moe) TP=1 serving bug** — `Glm4MoE.forward`
   broadcasts `shared_states` `[1,T,H]` into `out_states` `[T,H]` on the
   dp=1/tp=1 path; any GLM-4.5-family MoE model fails single-GPU decode.
   Discovered while attempting the glm52 MoE geometry on the glm4_moe
   architecture. Owner: `lmdeploy/pytorch/models/glm4_moe.py`.
5. **int8-KV TM-layout decode at small batch** — report item, no gate:
   eager TM dispatch loses to the graph-captured Triton fallback at
   b=1/8 on this card; production verdict needs the H200 matrix with
   graph-captured TM decode (capture-safety of the TM decode kernel is
   the open item).
6. **SPEC-B3/ROADMAP sync** — section 5 below; `.scratch/` is read-only
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
  `--suite int8kv`; measured green on the fixed source (numeric gate
  PASS, worst KL 3.50e-06 ≤ 2e-3; TM/stock 0.08/0.64/1.23 at b=1/8/32
  eager-decode, H200 graph-captured matrix is the production verdict).
- TP=2: pipeline ready, measured numbers deferred to H200 (`--suite tp2`
  declaration + `--preset h200`).
- GDN: out of batch 3 per the 2026-11-17 rollback; the two-stage ROADMAP
  entry stands.

For `.scratch/tm-gemm-graft/ROADMAP.md` (packed-KV row, 批次 column note):
"B3-06 harness rung 实测绿(修复后真实路径,KL 3.50e-06;TM/stock 0.08/0.64/1.23 @ b=1/8/32 eager decode);TM decode 图捕获适配为遗留项;H200 复测随 CI 批."
