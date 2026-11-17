# B3-04 debug notes — ATen KV writer multi-batch investigation (2026-11-17)

Verdict: `fill_kv_cache_tm` (lmdeploy/pytorch/backends/tm_kv.py) is **correct**.
The three red tests decomposed into two test-side bugs (fixed) and one
kernel-side defect (xfailed, owned by src/turbomind).

## Evidence index (this folder)

- `probe_aten_placement.py` — first red loop; exact per-token placement map
  (constant tokens => byte-exact readback). All packed-feed cases green.
- `probe_matrix.py` / `probe_matrix2.py` — geometry/dtype matrix; first
  version had a wrong-layer readback (superseded by matrix2).
- `probe_decisive.py` — dtype isolation (fp32 == bf16 => not dtype) +
  single-token snapshot diff.
- `probe_onestep.py` — byte-diff + per-batch value check of one T=4 append:
  writes confined to blocks {3,7,11,15}, codes/params byte-exact.
- `probe_stepdiff.py` — 64 sequential steps, all confined to expected slots.
- `probe_reconcile.py` — contradiction resolver (right slots, wrong values
  when fed b-major rows).
- `probe_provenance.py` — provenance of a "wrong" slot: contained kt\[3, p0\]
  = flat b-major row 768 with 1.0 code match => the TEST fed b-major rows
  where the packed contract requires batch-major truncation.
- `sanitizer_entry_b304.py` + `memcheck.log` / `initcheck.log` — sanitizer
  trio on the fixed writer path (multi-batch, both writers, both readers).

## Findings

1. Test bug A (roundtrip, aten): `k.reshape(-1, H, D)[:B*P]` is b-major
   (row = b*KLEN + p); the writer contract (and triton `fill_kv_cache`) is
   packed batch order (row = b*q_len + i). For B=1 the two coincide — why the
   single-batch case was green. Fix: `k[:, :P].reshape(...)` (truncate, then
   flatten). Regression: `test_aten_writer_multi_batch_packed_placement`
   (exact per-slot; mutation-red-proofed via a temporary batch_ids shift).
2. Test bug B (aten): `q_start_loc = zeros` made the valid mask (correctly)
   drop all rows >= 192 — batches 1..3 never written; the KL/max_abs gates
   could not see a zero-filled batch (vacuous green). Fix: exclusive-cumsum
   start loc + explicit pool-content asserts.
3. Kernel-side defect (mixed-chain; xfailed, NOT the ATen writer): every
   `tm_attn_decode` call zero-writes the last valid position's KV storage
   (codes + params, all heads, K and V; ~2.1 KB per sequence block), even
   for aligned klen and production-convention arguments. Mechanism:
   `AttentionUniversal` hardwires `kProcessKV = true` for `DecodingCtaMap`
   (attention_universal.h:59); the write-back branch is guarded only by
   `readonly_block_num`, which `BuildDecodePlan`
   (src/turbomind/python/attn_bench_bind.cpp) leaves `nullptr`; the binding
   also wires `params.k/v` into a Q-only qkv workspace, so the "converted"
   values stored are zeros, and initcheck shows the same branch reading the
   uninitialized workspace rows (2048 hits in Prologue/Ldg). Bench-binding
   contract violation is proven; production impact surface is being
   characterized separately (kernel-owner ticket).

## Sanitizer trio (fixed writer path, this entry)

- memcheck: no invalid global accesses (0). 4 host-side
  `cudaErrorInvalidValue` hits from TM's kernel Registry attribute probing
  (registry.cu:29) at first decode — pre-existing, not memory errors.
- racecheck: 0 hazards (bridge writer -> ATen writer -> readers, one stream).
- initcheck: 2048 uninitialized-global reads, all inside the TM decode
  kernel's kProcessKV workspace reads (same defect as item 3); ATen writer
  and torch ops clean.

## \[DEBUG-b304b\] Root cause: decode tail round-trip mismatch (mixed-chain)

- Symptom: tail_err 2.3-2.6 vs 0.05 budget; decode-appended slots corrupted.
- Hypotheses ranked: (1) kernel quant convention mismatch, (2) ti off-by-one,
  (3) writer placement. All falsified: decode's own store at ti = history_len
  writes batch 0 correctly (dequant err ~0.005, kernel-quant rounding only).
- Decisive probe (probe_decisive2.py, probe_pos3.py): the kernel wrote batch 1's
  token into batch 1 block 2 slot 0 (pos 128), batch 2 into pos 64, batch 3 into
  pos 0 — exactly ptrs\[cu_block_base\[b\] + ti // 64\] with cu_block_base\[b\] = 3b.
- Root cause (test helper, not kernel/writer): `_tm_decode` passed
  `cu = _cu([klen // BLOCK_LEN])` (cumulative per-seq block counts, stride 3)
  as cu_block_nums, but `tm_block_pointer_table` returns a row-major table with
  C = KLEN // BLOCK_LEN = 4 columns whose bases are `[0, 4, 8, 12]`. Every
  batch b > 0 had its block pointers shifted by b, so its history_len store
  (and reads) landed in the wrong physical blocks. Batch 0 (base 0) was always
  right, masking single-batch tests.
- Fix: tests/pytorch/kernel/test_tm_int8_kv.py `_tm_decode` now builds cu from
  the table's column count (`_cu([C] * BATCH)`), matching the bases the table
  returns. Kernel and writer untouched.
- Result: mixed-chain green (byte-level tail round-trip within 0.005-0.01
  kernel-quant noise), full suite 8/8 passed.
