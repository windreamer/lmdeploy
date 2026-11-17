# KL Threshold Amendment — W4A16 GEMM Acceptance (Ticket 05)

Status: **DRAFT — backfill after the first engine-level T1 run.**

The ticket 05 decision fixes the numeric anchor pair (AWQ W4A16 main anchor,
HF bf16 secondary anchor) and defers the hard KL threshold: the first
engine-level run measures the tm-vs-awq first-token KL distribution, and the
threshold is then chosen from that distribution and backfilled. This template
holds the placeholder until then.

## 1. Where the gate lives (existing harness, do not rewrite)

| asset                                               | role                                                                   |
| --------------------------------------------------- | ---------------------------------------------------------------------- |
| `tests/pytorch/engine/w4a16_golden.py`              | gate library: prompt set, `run_engine`, `gate_against`, `KL_THRESHOLD` |
| `tests/pytorch/engine/test_w4a16_golden_gate.py`    | gate cases: tiny ct (fast) and real model (`slow`)                     |
| `tests/pytorch/engine/data/awq_ref_llama31_8b.json` | frozen external reference, dumped on the AWQ Triton route              |

The gate carries two numbers per prompt: top-1 agreement of the first token
over the full vocabulary, and truncated KL restricted to the reference top-20
support. Self-consistent parity never proves semantics; the frozen real-model
reference is the external judge.

## 2. First-run distribution (backfill)

Run the gate on the T1 target model and record the distribution:

```bash
# Real model gate against the frozen reference:
pytest tests/pytorch/engine/test_w4a16_golden_gate.py -m 'gpu and slow' -k real_model -v

# Tiny ct gate (fast dev loop):
pytest tests/pytorch/engine/test_w4a16_golden_gate.py -m gpu -k tiny_ct -v
```

| field                | value                  |
| -------------------- | ---------------------- |
| run date             | `<YYYY-MM-DD>`         |
| model                | `<model name>`         |
| num prompts          | `<N>` (golden set: 32) |
| top1_agreement       | `<float>`              |
| greedy_agreement     | `<float>`              |
| worst truncated KL   | `<float>`              |
| median per-prompt KL | `<float>`              |
| p90 per-prompt KL    | `<float>`              |

## 3. Chosen threshold (backfill)

| field              | value                                                |
| ------------------ | ---------------------------------------------------- |
| new `KL_THRESHOLD` | `<float>`                                            |
| margin rule        | `<e.g. one order above the measured worst KL>`       |
| decision record    | `HISTORY/<YYYY-MM>.md <date> KL threshold amendment` |

Current placeholder in `w4a16_golden.py`: `KL_THRESHOLD = 2e-3`, set one
order above the ticket 09 impl-level data point (worst 1.9e-4).

## 4. Backfill targets, in order

1. `tests/pytorch/engine/w4a16_golden.py` — set `KL_THRESHOLD`.
2. `.scratch/tm-gemm-graft/SPEC.md` — replace the "KL 阈值按 T1 首跑分布定后回填"
   wording in Implementation Decisions with the fixed number (scratch planning
   note, not shipped code).
3. `HISTORY/<YYYY-MM>.md` — one decision entry with the measured distribution.
4. Re-run the gate command from section 2 to confirm the tightened threshold
   holds on the reference pair.

## 5. B3-06 rung report template (MoE / int8-KV / TP=2)

The rung suites emit this shape automatically
(`benchmark/w4a16_acceptance.py --suite {moe,int8kv,tp2}`, output under
`reports/w4a16/b306_*/`). Keep the manual backfill fields aligned:

| field                | MoE rung                                   | int8-KV rung                                        | TP=2 rung                   |
| -------------------- | ------------------------------------------ | --------------------------------------------------- | --------------------------- |
| routes (ref -> cand) | triton fused_moe (fp16 twin) -> TM grouped | stock int8 KV -> TM int8 block layout               | matrix readiness, no timing |
| numeric gate         | KL ≤ `MOE_TINY_KL_THRESHOLD`, top-1 report | KL ≤ 2e-3, top-1 report                             | n/a (declaration)           |
| throughput table     | b ∈ {1, 8, 32}, TM/Triton per batch        | b ∈ {1, 8, 32}, on/off per batch                    | skip reasons per rung       |
| smoke tier           | b=1, 8 tokens, 0 timed passes              | same                                                | n/a                         |
| placement            | `reports/w4a16/b306_moe/`                  | `reports/w4a16/b306_int8kv/`                        | `reports/w4a16/b306_tp2/`   |
| known blockers           | none                                         | none on the fixed source; a fixed/PRESENT probe guards stale checkouts | single-GPU host; H200 item |

Amendment rule for the int8-KV threshold: same procedure as sections 2-4 —
first measured on/off distribution fixes the number, the constant
`INT8_KV_KL_THRESHOLD` in `benchmark/w4a16_acceptance.py` is the single
backfill target.
