# Copyright (c) OpenMMLab. All rights reserved.
"""One-shot acceptance harness for the W4A16 GEMM graft (ticket 05).

The harness runs the decode rung matrix of the acceptance ticket:

    decode batch {1, 2, 4, 8, 16, 32} x seq {2k, 32k} x TP {1, 2}

Three routes feed the report:

- ``awq_triton``: pytorch engine, ``LMDEPLOY_W4A16_GEMM_BACKEND=triton``.
  This is the main baseline. The hard gate compares against it.
- ``tm_graft``: pytorch engine, ``LMDEPLOY_W4A16_GEMM_BACKEND=turbomind``.
  This is the grafted route under acceptance.
- ``native_tm``: the C++ TurboMind engine. Upper-bound reference only.

Hard gate (the only gate): the graft route must not lose decode throughput
or per-step latency against ``awq_triton`` on any feasible rung. The ratio
against ``native_tm`` is a report item with target >= 0.90 and no gate.

Reuse: the numeric quality gate is not part of this script. It lives in the
ticket 03 assets under ``tests/pytorch/engine/`` (``w4a16_golden.py`` plus
the frozen reference ``data/awq_ref_llama31_8b.json``). The report links to
it. The engine load and the backend env switch mirror ``w4a16_golden.run_engine``.
The serving benchmarks in this directory measure dataset-driven serving
throughput; none isolates fixed-batch decode rungs, so this script carries
its own decode timing core on top of ``lmdeploy.pipeline``.

Outputs: ``report.json`` and ``report.md`` in ``--out-dir``.

Examples::

    python benchmark/w4a16_acceptance.py --preset smoke
    python benchmark/w4a16_acceptance.py --preset dev
    python benchmark/w4a16_acceptance.py --preset h200
    python benchmark/w4a16_acceptance.py --model-preset tiny
"""

import argparse
import datetime
import gc
import json
import os
import re
import statistics
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT / 'models' / 'llama-3.1-8b-awq'
GOLDEN_GATE_DIR = 'tests/pytorch/engine'
GOLDEN_GATE_TEST = f'{GOLDEN_GATE_DIR}/test_w4a16_golden_gate.py'
GOLDEN_GATE_LIB = f'{GOLDEN_GATE_DIR}/w4a16_golden.py'
GOLDEN_GATE_REF = f'{GOLDEN_GATE_DIR}/data/awq_ref_llama31_8b.json'
AMENDMENT_TEMPLATE = 'benchmark/w4a16_amendment_template.md'

# Rung geometry per preset. (seq_bucket, seq_len) pairs keep the report
# labels stable across presets.
SEQ_BUCKETS = {'2k': 2048, '32k': 32768}
BATCHES = [1, 2, 4, 8, 16, 32]
TM_RATIO_TARGET = 0.90  # report item, no gate
# Memory reserved for activations, CUDA graphs, and fragmentation.
WORKING_SET_BYTES = int(2.5e9)
# Weights occupy more device memory than the checkpoint files on disk.
WEIGHTS_EXPANSION = 1.15

PRESETS = {
    # Full acceptance matrix; memory-bound rungs self-mark on small cards.
    # 512 decode tokens keep the common decode window clear of the
    # admission skew at large batch x short seq rungs.
    'h200': dict(batches=BATCHES, seq_buckets=list(SEQ_BUCKETS), tps=[1, 2], decode_tokens=512, passes=3),
    # Same matrix; on the 3090 dev card the 32k x large-batch rungs fall out
    # through the KV feasibility estimate.
    'dev': dict(batches=BATCHES, seq_buckets=list(SEQ_BUCKETS), tps=[1], decode_tokens=512, passes=3),
    # Fast plumbing check of the whole harness path.
    'smoke': dict(batches=[1, 2], seq_buckets=['2k'], tps=[1], decode_tokens=32, passes=1),
    # Geometry-coverage rungs for the tiny ct model. The tiny checkpoint
    # caps positions at 512, so the seq rung shrinks accordingly.
    'tiny': dict(batches=[1, 2, 4], seq_buckets=['tiny'], tps=[1], decode_tokens=32, passes=2),
}
SEQ_BUCKET_LEN = dict(SEQ_BUCKETS, tiny=256)


@dataclass
class Rung:
    """One rung of the acceptance matrix."""
    tp: int
    seq_bucket: str
    seq_len: int
    batch: int


@dataclass
class RouteMetric:
    """Timed metrics of one route on one rung."""
    status: str = 'pending'  # ok | skipped | failed
    reason: str = ''
    throughput_tok_per_s: float | None = None
    step_ms: float | None = None
    tpot_ms: float | None = None
    first_token_s: float | None = None
    kv_tokens_needed: int | None = None
    kv_capacity_tokens: int | None = None
    passes: list = field(default_factory=list)

    def as_dict(self):
        return dict(status=self.status,
                    reason=self.reason,
                    throughput_tok_per_s=self.throughput_tok_per_s,
                    step_ms=self.step_ms,
                    tpot_ms=self.tpot_ms,
                    first_token_s=self.first_token_s,
                    kv_tokens_needed=self.kv_tokens_needed,
                    kv_capacity_tokens=self.kv_capacity_tokens,
                    passes=self.passes)


def hf_config(model_path: Path) -> dict:
    with open(model_path / 'config.json', encoding='utf-8') as f:
        return json.load(f)


def kv_bytes_per_token(cfg: dict) -> int:
    """Per-token KV bytes of one sequence in fp16/bf16."""
    head_dim = cfg.get('head_dim') or cfg['hidden_size'] // cfg['num_attention_heads']
    per_layer = 2 * cfg['num_key_value_heads'] * head_dim * 2
    return per_layer * cfg['num_hidden_layers']


def weights_bytes(model_path: Path) -> int:
    total = 0
    for p in model_path.iterdir():
        if p.suffix in ('.safetensors', '.bin'):
            total += p.stat().st_size
    return total


def gpu_free_bytes() -> int:
    import torch
    free, _total = torch.cuda.mem_get_info()
    return free


def git_commit() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              cwd=REPO_ROOT,
                              capture_output=True,
                              text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return 'unknown'


def estimate_capacity(model_path: Path, cfg: dict, cache_ratio: float) -> int:
    """Analytic KV token capacity estimate, valid before engine start."""
    kv_bpt = kv_bytes_per_token(cfg)
    avail = gpu_free_bytes() - int(weights_bytes(model_path) * WEIGHTS_EXPANSION) - WORKING_SET_BYTES
    avail = max(avail, 0)
    return int(avail * cache_ratio / kv_bpt)


@contextmanager
def w4a16_backend(choice: str):
    """Pin the W4A16 GEMM route for one engine lifecycle.

    The attribute patch covers the in-process executor. The environ copy
    covers executors that re-read the env in worker processes. The pattern
    mirrors ``tests/pytorch/engine/w4a16_golden.py::run_engine``.
    """
    import lmdeploy.pytorch.envs as envs
    original = envs.w4a16_gemm_backend
    original_env = os.environ.get('LMDEPLOY_W4A16_GEMM_BACKEND')
    envs.w4a16_gemm_backend = choice
    os.environ['LMDEPLOY_W4A16_GEMM_BACKEND'] = choice
    try:
        yield
    finally:
        envs.w4a16_gemm_backend = original
        if original_env is None:
            os.environ.pop('LMDEPLOY_W4A16_GEMM_BACKEND', None)
        else:
            os.environ['LMDEPLOY_W4A16_GEMM_BACKEND'] = original_env


class RouteEngine:
    """Engine lifecycle for one route: open, probe capacity, close."""

    def __init__(self, route: str, model_path: Path, session_len: int, max_batch: int, tp: int,
                 cache_ratio: float):
        self.route = route
        self.model_path = model_path
        self.session_len = session_len
        self.max_batch = max_batch
        self.tp = tp
        self.cache_ratio = cache_ratio
        self.pipe = None
        self.kv_capacity_tokens = None

    def open(self):
        from lmdeploy import pipeline
        from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig

        if self.route in ('awq_triton', 'tm_graft'):
            choice = 'triton' if self.route == 'awq_triton' else 'turbomind'
            kwargs = dict(session_len=self.session_len,
                          max_batch_size=self.max_batch,
                          cache_max_entry_count=self.cache_ratio,
                          enable_prefix_caching=False,
                          tp=self.tp)
            if self.tp == 1:
                # The in-process executor mirrors the ticket 03 gate setup.
                kwargs['distributed_executor_backend'] = 'uni'
            config = PytorchEngineConfig(**kwargs)
            with w4a16_backend(choice):
                self.pipe = pipeline(str(self.model_path), backend_config=config, log_level='WARNING')
            self._probe_pytorch_capacity()
        elif self.route == 'native_tm':
            config = TurbomindEngineConfig(session_len=self.session_len,
                                           max_batch_size=self.max_batch,
                                           cache_max_entry_count=self.cache_ratio,
                                           tp=self.tp,
                                           enable_prefix_caching=False)
            self.pipe = pipeline(str(self.model_path), backend_config=config, log_level='WARNING')
        else:
            raise ValueError(f'unknown route: {self.route}')
        return self

    def _probe_pytorch_capacity(self):
        try:
            cache_cfg = self.pipe.async_engine.engine.cache_config
            self.kv_capacity_tokens = cache_cfg.num_gpu_blocks * cache_cfg.block_size
        except Exception:
            # Keep the analytic estimate as the only capacity source.
            self.kv_capacity_tokens = None

    def close(self):
        if self.pipe is not None:
            self.pipe.close()
            self.pipe = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


_session_counter = [0]


def build_input_ids(tokenizer, seq_len: int, count: int, variant: int, vocab_size: int) -> list[list[int]]:
    """Deterministic token ids of exact length, distinct per request.

    Every request and every pass differ in a short token span at mid position. The differences block cross-request and
    cross-pass prompt cache reuse, so each rung pays its own prefill.
    """
    base_ids = tokenizer.encode(
        'The quick brown fox jumps over the lazy dog while the sun sets '
        'behind the mountains and the river keeps flowing to the sea.',
        add_special_tokens=False)
    ids = []
    for i in range(count):
        seq = base_ids * (seq_len // len(base_ids) + 1)
        seq = seq[:seq_len]
        v = variant * 131 + i * 17
        mid = seq_len // 2
        for k in range(4):
            # Keep ids in the plain text range. This stays clear of the
            # special-token block at the top of the vocabulary.
            seq[mid + k] = (v * (k + 3)) % (vocab_size - 2000) + 1000
        ids.append(seq)
    return ids


def time_decode(pipe, input_ids: list[list[int]], decode_tokens: int) -> dict:
    """Run one timed decode pass and return wave-timed metrics.

    All requests submit together. The scheduler can still admit prefills while early requests decode, so the numbers use
    the common window: it starts when the last request produced its first token and ends when the first request
    finished. Only token events inside that window count, so the pass measures the full concurrent batch, not admission
    stragglers.
    """
    from lmdeploy.messages import GenerationConfig

    engine = pipe.async_engine
    gen_config = GenerationConfig(max_new_tokens=decode_tokens,
                                  do_sample=False,
                                  ignore_eos=True,
                                  random_seed=0)
    stamps: list[tuple[float, int, int]] = []
    submit = time.perf_counter()

    async def _one(idx: int, ids: list[int]):
        _session_counter[0] += 1
        request = await engine.preprocess(None,
                                          input_ids=ids,
                                          session_id=_session_counter[0],
                                          gen_config=gen_config)
        async for out in engine.generate(request, stream_response=True):
            stamps.append((time.perf_counter(), idx, out.generate_token_len))

    async def _batch():
        import asyncio
        await asyncio.gather(*[_one(i, ids) for i, ids in enumerate(input_ids)])

    pipe._run(coro=_batch()).result()

    per_req: dict[int, list[tuple[float, int]]] = {}
    for ts, idx, gen_len in stamps:
        per_req.setdefault(idx, []).append((ts, gen_len))
    first_ts = {i: min(ts for ts, _ in v) for i, v in per_req.items()}
    last_ts = {i: max(ts for ts, _ in v) for i, v in per_req.items()}
    final_len = {i: max(g for _, g in v) for i, v in per_req.items()}
    t0 = max(first_ts.values())
    t1 = min(last_ts.values())
    if t1 - t0 <= 0:
        raise ValueError('no common decode window: a request finished before the last request started. '
                         'Raise --decode-tokens for this rung.')
    in_window = 0
    per_req_counts = []
    for i, v in per_req.items():
        n = sum(1 for ts, _ in v if t0 < ts <= t1)
        per_req_counts.append(n)
        in_window += n
    window = t1 - t0
    steps = statistics.median(per_req_counts)
    tpots = sorted((last_ts[i] - first_ts[i]) / max(final_len[i] - 1, 1) * 1000 for i in per_req)
    return dict(
        tokens=in_window,
        decode_span_s=window,
        throughput=in_window / window,
        step_ms=window / max(steps - 1, 1) * 1000 if steps > 1 else 0.0,
        tpot_ms=statistics.median(tpots),
        first_token_s=t0 - submit,
    )


def summarize_passes(raw: list[dict]) -> dict:
    """Median across timed passes; the first entry is the warmup."""
    kept = raw[1:] if len(raw) > 1 else raw
    return dict(
        throughput_tok_per_s=statistics.median(p['throughput'] for p in kept),
        step_ms=statistics.median(p['step_ms'] for p in kept),
        tpot_ms=statistics.median(p['tpot_ms'] for p in kept),
        first_token_s=statistics.median(p['first_token_s'] for p in kept),
        passes=raw,
    )


def run_rung_on_route(engine: RouteEngine, rung: Rung, tokenizer, vocab_size: int, decode_tokens: int,
                      passes: int, variant_seed: int, metric: RouteMetric):
    """Warm up and time one rung on an already open engine."""
    if rung.seq_len + decode_tokens > engine.session_len:
        metric.status = 'skipped'
        metric.reason = f'session_len {engine.session_len} below seq {rung.seq_len} + decode {decode_tokens}'
        return
    need_tokens = rung.batch * (rung.seq_len + decode_tokens)
    metric.kv_tokens_needed = need_tokens
    capacity = engine.kv_capacity_tokens
    if capacity is not None:
        metric.kv_capacity_tokens = capacity
        if need_tokens > capacity:
            metric.status = 'skipped'
            metric.reason = (f'KV need {need_tokens} tokens above engine capacity {capacity} tokens')
            return
    raw = []
    try:
        for p in range(passes + 1):
            ids = build_input_ids(tokenizer, rung.seq_len, rung.batch, variant_seed + p, vocab_size)
            raw.append(time_decode(engine.pipe, ids, decode_tokens))
    except Exception as e:  # noqa: BLE001  one bad rung must not kill the matrix
        metric.status = 'failed'
        metric.reason = f'{type(e).__name__}: {e}'
        return
    got = summarize_passes(raw)
    metric.status = 'ok'
    metric.throughput_tok_per_s = got['throughput_tok_per_s']
    metric.step_ms = got['step_ms']
    metric.tpot_ms = got['tpot_ms']
    metric.first_token_s = got['first_token_s']
    metric.passes = raw


def read_kl_threshold() -> float | None:
    path = REPO_ROOT / GOLDEN_GATE_LIB
    if not path.is_file():
        return None
    match = re.search(r'^KL_THRESHOLD\s*=\s*([0-9.eE+-]+)', path.read_text(encoding='utf-8'), re.M)
    return float(match.group(1)) if match else None


def rung_skip(rung: Rung, cfg: dict, decode_tokens: int, gpu_count: int, est_capacity: int, kv_bpt: int,
              cache_ratio: float) -> str | None:
    """Return the skip reason for a rung, or None when the rung is feasible.

    A rung whose KV need exceeds capacity would degrade to a smaller concurrent batch and report wrong numbers. The pre-
    start estimate keeps such rungs out of the timed set.
    """
    if rung.tp > gpu_count:
        return f'tp={rung.tp} needs {rung.tp} GPUs, device_count={gpu_count}'
    if rung.seq_len + decode_tokens > cfg.get('max_position_embeddings', 1 << 30):
        return (f'seq {rung.seq_len} + decode {decode_tokens} above model max_position_embeddings')
    need_tokens = rung.batch * (rung.seq_len + decode_tokens)
    if need_tokens > est_capacity:
        return (f'KV estimate: need {need_tokens} tokens, analytic capacity {est_capacity} tokens'
                f' (kv {kv_bpt // 1024} KiB/token, cache_ratio {cache_ratio})')
    return None


def render_markdown(report: dict) -> str:
    """Render the acceptance matrix in the ticket 05 layout."""
    env = report['environment']
    model = report['model']
    lines = []
    lines.append('# W4A16 GEMM Acceptance Report (ticket 05)')
    lines.append('')
    lines.append(f"- generated_at: {report['generated_at']}")
    lines.append(f"- model: `{model['path']}` ({model['quant']})")
    if model['geometry_only']:
        lines.append('- scope: **geometry coverage only.** The numbers prove harness plumbing on a seeded'
                     ' tiny checkpoint. They are not acceptance numbers.')
    else:
        lines.append('- scope: acceptance matrix on the real AWQ checkpoint.')
    lines.append(f"- device: {env['gpu']}, torch {env['torch']}, lmdeploy {env['lmdeploy']}, commit {env['commit']}")
    lines.append(f"- decode tokens per pass: {report['config']['decode_tokens']},"
                 f" timed passes: {report['config']['passes']} (median),"
                 f" cache_max_entry_count: {report['config']['cache_ratio']}")
    lines.append('')
    lines.append('## Rung matrix (decode)')
    lines.append('')
    lines.append(f'| TP | seq | batch | AWQ Triton tok/s | step ms | TM graft tok/s | step ms |'
                 f' graft/Triton | hard gate | native TM tok/s | graft/native (target >= {TM_RATIO_TARGET:.0%}) |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for rung in report['matrix']:
        tr = rung['routes'].get('awq_triton', {})
        tg = rung['routes'].get('tm_graft', {})
        tn = rung['routes'].get('native_tm', {})
        ratio = rung['ratios'].get('tm_graft_vs_awq_triton')
        ratio_native = rung['ratios'].get('tm_graft_vs_native_tm')
        gate = rung['gate']['verdict']
        lines.append('| {tp} | {sb} ({sl}) | {b} | {tr} | {trms} | {tg} | {tgms} | {r} | {g} | {tn} | {rn} |'.format(
            tp=rung['tp'],
            sb=rung['seq_bucket'],
            sl=rung['seq_len'],
            b=rung['batch'],
            tr=_num(tr.get('throughput_tok_per_s'), tr),
            trms=_num(tr.get('step_ms'), tr),
            tg=_num(tg.get('throughput_tok_per_s'), tg),
            tgms=_num(tg.get('step_ms'), tg),
            r=_num(ratio),
            g=gate,
            tn=_num(tn.get('throughput_tok_per_s'), tn),
            rn=_num(ratio_native),
        ))
    lines.append('')
    lines.append('## Hard gate (硬门槛): decode non-inferior to AWQ Triton')
    lines.append('')
    v = report['verdict']
    if v['hard_gate'] == 'N/A_GEOMETRY':
        lines.append('- verdict: **N/A (geometry run).** The tiny checkpoint carries no acceptance weight;'
                     ' per-rung ratios are plumbing evidence only.')
    else:
        lines.append(f"- verdict: **{v['hard_gate']}** over {v['gated_rungs']} gated rungs"
                     f" ({v['pass_rungs']} pass, {v['fail_rungs']} fail, {v['skipped_rungs']} skipped).")
    lines.append('- rule: `tm_graft throughput >= awq_triton throughput` per feasible rung; equal token counts'
                 ' make the step latency the same condition.')
    lines.append('')
    lines.append('## TM ratio report item (目标 >= 90%, no gate)')
    lines.append('')
    ratios = [r['ratios'].get('tm_graft_vs_native_tm') for r in report['matrix']]
    ratios = [x for x in ratios if x is not None]
    if ratios:
        lines.append(f"- min graft/native ratio over measured rungs: **{min(ratios):.3f}**;"
                     f" max: {max(ratios):.3f}.")
    else:
        lines.append('- no native TurboMind numbers in this run.')
    lines.append('')
    skipped = [(r['tp'], r['seq_bucket'], r['batch'], route, m['status'], m['reason']) for r in report['matrix']
               for route, m in r['routes'].items() if m['status'] in ('skipped', 'failed')]
    if skipped:
        lines.append('## Skipped and failed rungs')
        lines.append('')
        lines.append('| TP | seq | batch | route | status | reason |')
        lines.append('|---|---|---|---|---|---|')
        for tp, sb, b, route, status, reason in skipped:
            lines.append(f'| {tp} | {sb} | {b} | {route} | {status} | {reason} |')
        lines.append('')
    lines.append('## Quality gate (KL, numeric)')
    lines.append('')
    qg = report['quality_gate']
    lines.append(f"- golden gate: `{GOLDEN_GATE_TEST}` against the frozen reference `{GOLDEN_GATE_REF}`.")
    lines.append(f"- current engine-level KL threshold: `KL_THRESHOLD = {qg['kl_threshold']}` in `{GOLDEN_GATE_LIB}`.")
    lines.append(f"- threshold amendment procedure: `{AMENDMENT_TEMPLATE}`.")
    lines.append(f"- real-model gate command: `pytest {GOLDEN_GATE_TEST} -m 'gpu and slow' -k real_model -v`.")
    lines.append('')
    lines.append('## TP=2 note')
    lines.append('')
    lines.append(f"- {report['verdict']['tp2_note']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def _num(value, container=None):
    if value is None:
        if container is not None and container.get('status') != 'ok':
            return container.get('status', '-')
        return '-'
    if isinstance(value, float):
        if value >= 100:
            return f'{value:.1f}'
        if value >= 10:
            return f'{value:.2f}'
        return f'{value:.3f}'
    return str(value)


def main(
    preset: str = 'dev',
    model_preset: str = 'real',
    model_path: str | None = None,
    out_dir: str | None = None,
    batches: str = '',
    seq_buckets: str = '',
    tps: str = '1',
    decode_tokens: int | None = None,
    passes: int | None = None,
    cache_ratio: float = 0.85,
    routes: str = 'awq_triton,tm_graft,native_tm',
    noninferior_tolerance: float = 0.0,
):
    """Run the acceptance matrix and write report.json plus report.md.

    Args:
        preset: rung matrix preset (h200, dev, smoke, tiny).
        model_preset: 'real' uses the AWQ checkpoint; 'tiny' generates the
            seeded tiny ct checkpoint and marks the report as geometry
            coverage.
        model_path: explicit model directory; overrides ``model_preset``.
        out_dir: output directory; defaults to ``reports/w4a16``.
        batches: comma list overriding the preset batches.
        seq_buckets: comma list overriding the preset seq buckets.
        tps: comma list of TP degrees; degrees above the GPU count are
            skipped with a note.
        decode_tokens: generated tokens per pass.
        passes: timed passes per rung; one extra warmup pass always runs.
        cache_ratio: KV cache fraction of free device memory.
        routes: comma list of routes to run.
        noninferior_tolerance: gate slack; 0.0 is the strict ticket gate.
    """
    import torch

    preset_cfg = PRESETS[preset]
    if preset == 'tiny' and model_path is None:
        # The tiny rung set pairs with the tiny ct checkpoint.
        model_preset = 'tiny'
    batches_list = [int(x) for x in batches.split(',')] if batches else preset_cfg['batches']
    buckets_list = seq_buckets.split(',') if seq_buckets else preset_cfg['seq_buckets']
    tps_list = [int(x) for x in tps.split(',')] if tps else preset_cfg['tps']
    decode_tokens = decode_tokens or preset_cfg['decode_tokens']
    passes = passes or preset_cfg['passes']
    geometry_only = False

    if model_path:
        model_dir = Path(model_path).resolve()
    elif model_preset == 'tiny':
        model_dir = make_tiny_model()
        geometry_only = True
    else:
        model_dir = DEFAULT_MODEL_DIR.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f'model dir missing: {model_dir}')

    cfg = hf_config(model_dir)
    vocab_size = cfg['vocab_size']
    kv_bpt = kv_bytes_per_token(cfg)
    est_capacity = estimate_capacity(model_dir, cfg, cache_ratio)
    gpu_count = torch.cuda.device_count()

    seq_lens = {b: SEQ_BUCKET_LEN[b] for b in buckets_list}
    max_seq = max(seq_lens.values())
    session_len = max_seq + decode_tokens + 128
    rungs = [Rung(tp=tp, seq_bucket=b, seq_len=seq_lens[b], batch=bt)
             for tp in tps_list for b in buckets_list for bt in batches_list]

    out_path = Path(out_dir) if out_dir else REPO_ROOT / 'reports' / 'w4a16'
    out_path.mkdir(parents=True, exist_ok=True)

    route_list = [r.strip() for r in routes.split(',') if r.strip()]
    entries: dict[tuple, dict] = {}
    order = []
    for rung in rungs:
        key = (rung.tp, rung.seq_bucket, rung.batch)
        entries[key] = dict(tp=rung.tp, seq_bucket=rung.seq_bucket, seq_len=rung.seq_len, batch=rung.batch,
                            routes={}, ratios={}, gate=None)
        order.append(key)

    tokenizer = _get_tokenizer(model_dir)

    for tp in dict.fromkeys(r.tp for r in rungs):
        tp_rungs = [r for r in rungs if r.tp == tp]
        for route in route_list:
            # One engine serves every rung of one route and TP degree. The
            # session_len covers the longest rung, and the KV pool depends
            # on the cache fraction, not on the session length.
            engine = None
            engine_error = None
            for rung in tp_rungs:
                entry = entries[(rung.tp, rung.seq_bucket, rung.batch)]
                skip = rung_skip(rung, cfg, decode_tokens, gpu_count, est_capacity, kv_bpt, cache_ratio)
                if skip is not None:
                    entry['routes'][route] = RouteMetric(status='skipped', reason=skip).as_dict()
                    continue
                if engine is None and engine_error is None:
                    engine = RouteEngine(route, model_dir, session_len, max(batches_list), tp, cache_ratio)
                    try:
                        engine.open()
                    except Exception as e:  # noqa: BLE001
                        engine = None
                        engine_error = f'engine start failed: {type(e).__name__}: {e}'
                if engine is None:
                    entry['routes'][route] = RouteMetric(status='failed', reason=engine_error).as_dict()
                    continue
                metric = RouteMetric()
                run_rung_on_route(engine, rung, tokenizer, vocab_size, decode_tokens, passes,
                                  variant_seed=1000 + rung.seq_len + rung.batch, metric=metric)
                entry['routes'][route] = metric.as_dict()
            if engine is not None:
                engine.close()

    report_rungs = []
    for key in order:
        entry = entries[key]
        tr = entry['routes'].get('awq_triton', {})
        tg = entry['routes'].get('tm_graft', {})
        tn = entry['routes'].get('native_tm', {})
        if tr.get('status') == 'ok' and tg.get('status') == 'ok':
            ratio = tg['throughput_tok_per_s'] / tr['throughput_tok_per_s']
            entry['ratios']['tm_graft_vs_awq_triton'] = ratio
            if geometry_only:
                # A seeded tiny checkpoint carries no acceptance weight.
                # The ratio stays in the report as plumbing evidence.
                entry['gate'] = dict(name='decode_non_inferior_vs_awq_triton', verdict='info', ratio=ratio)
            else:
                ok = (ratio >= 1.0 - noninferior_tolerance
                      and tg['step_ms'] <= tr['step_ms'] / (1.0 - noninferior_tolerance))
                entry['gate'] = dict(name='decode_non_inferior_vs_awq_triton',
                                     verdict='pass' if ok else 'fail',
                                     ratio=ratio)
        else:
            reason = tg.get('reason') or tr.get('reason') or 'route not measured'
            entry['gate'] = dict(name='decode_non_inferior_vs_awq_triton', verdict='skipped', reason=reason)
        if tn.get('status') == 'ok' and tg.get('status') == 'ok':
            entry['ratios']['tm_graft_vs_native_tm'] = tg['throughput_tok_per_s'] / tn['throughput_tok_per_s']
        report_rungs.append(entry)

    gated = [r for r in report_rungs if r['gate']['verdict'] in ('pass', 'fail')]
    pass_n = sum(1 for r in gated if r['gate']['verdict'] == 'pass')
    fail_n = sum(1 for r in gated if r['gate']['verdict'] == 'fail')
    skipped = [dict(tp=r['tp'], seq_bucket=r['seq_bucket'], batch=r['batch'], reason=r['gate'].get('reason', ''))
               for r in report_rungs if r['gate']['verdict'] not in ('pass', 'fail', 'info')]
    if geometry_only:
        hard = 'N/A_GEOMETRY'
    elif fail_n:
        hard = 'FAIL'
    elif pass_n:
        hard = 'PASS'
    else:
        hard = 'INCONCLUSIVE'
    tp2_note = ('tp=2 rungs included in the matrix.' if any(r.tp > 1 for r in rungs) else
                'no tp=2 rung in this run; pass --tps 1,2 on a multi-GPU host.')
    if any(r.tp > gpu_count for r in rungs):
        tp2_note += f' rungs with tp above device_count={gpu_count} were skipped.'

    report = dict(
        schema='w4a16-acceptance/1',
        ticket='tm-gemm-graft/05',
        generated_at=datetime.datetime.now().isoformat(timespec='seconds'),
        model=dict(path=str(model_dir), quant='awq-gemm-w4a16' if not geometry_only else 'ct-w4a16-tiny',
                   geometry_only=geometry_only),
        environment=dict(gpu=torch.cuda.get_device_name(0), gpu_total_gb=round(_gpu_total_gb(), 1),
                         torch=torch.__version__, lmdeploy=_lmdeploy_version(), commit=git_commit(),
                         gpu_count=gpu_count),
        config=dict(preset=preset, batches=batches_list, seq_buckets=buckets_list, tps=tps_list,
                    decode_tokens=decode_tokens, passes=passes, cache_ratio=cache_ratio,
                    noninferior_tolerance=noninferior_tolerance,
                    session_len=session_len, kv_bytes_per_token=kv_bpt,
                    kv_capacity_estimate_tokens=est_capacity),
        matrix=report_rungs,
        verdict=dict(hard_gate=hard, gated_rungs=len(gated), pass_rungs=pass_n, fail_rungs=fail_n,
                     skipped_rungs=len(skipped), skipped=skipped, tp2_note=tp2_note),
        quality_gate=dict(harness=GOLDEN_GATE_LIB, reference=GOLDEN_GATE_REF, test=GOLDEN_GATE_TEST,
                          kl_threshold=read_kl_threshold(), amendment_template=AMENDMENT_TEMPLATE),
        reuse=dict(
            golden_gate='tests/pytorch/engine/w4a16_golden.py + test_w4a16_golden_gate.py (ticket 03), '
                        'referenced, not rewritten',
            golden_reference=GOLDEN_GATE_REF,
            env_switch_pattern='mirrors w4a16_golden.run_engine (attribute patch + environ copy)',
            engine_entry='lmdeploy.pipeline, the same engine entry as the ticket 03 gate',
            benchmark_dir='existing serving benchmarks measure dataset-driven serving throughput; '
                          'none isolates fixed-batch decode rungs, so this script adds its own decode timer',
        ),
    )

    json_path = out_path / 'report.json'
    md_path = out_path / 'report.md'
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    md_path.write_text(render_markdown(report), encoding='utf-8')
    print(f'[w4a16-acceptance] hard gate: {hard} ({pass_n} pass / {fail_n} fail / {len(skipped)} skipped)')
    print(f'[w4a16-acceptance] report: {json_path}')
    print(f'[w4a16-acceptance] report: {md_path}')
    return report


def _gpu_total_gb() -> float:
    import torch
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def _lmdeploy_version() -> str:
    import lmdeploy
    return getattr(lmdeploy, '__version__', 'unknown')


def _get_tokenizer(model_path: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)


def make_tiny_model() -> Path:
    """Generate the seeded tiny ct checkpoint under the report directory."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from tests.turbomind.tiny_ct import gen_tiny_ct_hf

    if not DEFAULT_MODEL_DIR.is_dir():
        raise FileNotFoundError(f'tokenizer source missing: {DEFAULT_MODEL_DIR}')
    out = REPO_ROOT / 'reports' / 'w4a16' / 'tiny_ct_seed9'
    gen_tiny_ct_hf(str(out), str(DEFAULT_MODEL_DIR), seed=9)
    return out


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', default='dev', choices=sorted(PRESETS), help='rung matrix preset')
    parser.add_argument('--model-preset', default='real', choices=['real', 'tiny'])
    parser.add_argument('--model-path', default=None, help='explicit model directory')
    parser.add_argument('--out-dir', default=None, help='output directory (default reports/w4a16)')
    parser.add_argument('--batches', default='', help='comma list, overrides preset')
    parser.add_argument('--seq-buckets', default='', help='comma list of bucket names, overrides preset')
    parser.add_argument('--tps', default='', help='comma list of TP degrees, overrides preset')
    parser.add_argument('--decode-tokens', type=int, default=None)
    parser.add_argument('--passes', type=int, default=None, help='timed passes; one warmup pass always runs')
    parser.add_argument('--cache-ratio', type=float, default=0.85)
    parser.add_argument('--routes', default='awq_triton,tm_graft,native_tm')
    parser.add_argument('--noninferior-tolerance', type=float, default=0.0)
    args = parser.parse_args()
    main(preset=args.preset,
         model_preset=args.model_preset,
         model_path=args.model_path,
         out_dir=args.out_dir,
         batches=args.batches,
         seq_buckets=args.seq_buckets,
         tps=args.tps,
         decode_tokens=args.decode_tokens,
         passes=args.passes,
         cache_ratio=args.cache_ratio,
         routes=args.routes,
         noninferior_tolerance=args.noninferior_tolerance)
