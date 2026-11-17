# B3 GDN zero-copy direct-read (option B) — landing report

Branch `graft/w4a16-core`. Re-introduces the rolled-back TurboMind GDN decode
graft (commit `44e0165e`) in bridge-less form: the transposed `kVK` recurrent
kernel variant (landed `385468de`) reads the PyTorch indexed pool rows
directly through `state_ptrs`. Env restored to default `auto`.

## Form

`lmdeploy/pytorch/backends/cuda/tm_gated_delta_rule.py`

- No gather/scatter/transpose kernels, no compact TM-format state, no
  scatter-back. Decode points `state_ptrs[seq]` at the destination pool row
  (`[heads, V, K]` per row = exactly the per-head `[V, K]` slab the `kVK`
  variant expects), one head group covering all heads; states never leave
  the pool, so checkpoint/PD/eviction storage is untouched.
- One fused pack kernel per decode step: fills the persistent q/k/v/g/beta
  step buffers (f32 g/beta, optional q/k l2 normalization), aligns each
  destination slab with its source in pool layout (checkpoint rebinding
  copies once, fresh history zero-fills, the common destination==source case
  touches nothing), and fills the byte-address pointer array plus per-row
  `finished` flags (invalid rows clamp to slot 0 with the store off).
  Plans, buffers and dlpack handles are cached per workload signature; a
  step is prepare + one pack launch + one `delta_rule_run`.
- Hard-constraint probe unchanged from the rolled-back version (head_dim 128,
  f16/bf16 input, f32-or-input state, planner acceptance, width==1);
  chunked prefill stays fla; `LMDEPLOY_GDN_BACKEND=on` raises on decline.
- `LMDEPLOY_GDN_BACKEND` rejoined the unified `LMDEPLOY_TM_KERNELS` matrix as
  class `gdn` (legacy env), default `auto`.

## Verification (14 tests, `tests/pytorch/kernel/test_tm_gdn_decode.py`)

- S1 parity vs the fla production decode: state max_abs `0.000000` every run;
  outputs `0.000000` in most runs with occasional last-ulp drift traced to
  the fla tilelang kernel's cross-process code generation (bounded `< 2e-3`,
  typically `<= 6e-5`). Rebound-slot (checkpoint) and padded/invalid-row
  semantics verified against fla and slot-integrity checks.
- Sanitizer triple (memcheck / racecheck / initcheck) over mixed
  same-slot + rebound + fresh-history + invalid decode at B=1..8, hv 8..64:
  0 errors, 0 hazards each.
- Regression: `test_tm_gdn_decode.py` (14) + `test_tm_kernels_env.py` (62) +
  `test_compact_backend.py` (39) all pass; the compact-backend routing test
  was updated for the new auto default (non-indexed specs still bind fla).

## Net gain vs the bridge version (44e0165e)

| B (hv=32) | bridge full path | zero-copy eager | zero-copy graph replay |
| --------- | ---------------- | --------------- | ---------------------- |
| 1         | −83%             | −17%            | **+8%**                |
| 8         | —                | −18%            | **+75%**               |
| 32        | −148%            | −12%            | **+87%**               |

Eager single-step numbers are host-enqueue-bound (both paths ~0.11–0.13 ms
python per step); production decode runs under piecewise CUDA graphs, so the
replay row is the hot-path number. There the adapter cost is gone entirely
and the pure kernel wins (batch-2: 0.013 vs 0.047 ms pure kernel) show
through: B=32 is 7.6× faster end-to-end.

At B=1 the gain is small (+8%) but positive: the fla fused-graph path is far
stronger than its standalone kernel suggested (0.0112 ms), so the batch-3
"+50% at B=1" projection does not materialize; the route still never loses.

Verdict: option B lands with default `auto` — positive net gain on every
graph-replayed batch, zero-copy storage semantics, and probe-gated fallback
for everything the kernel cannot serve (head_dim != 128, width > 1, plan
declines).
