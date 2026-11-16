# Copyright (c) OpenMMLab. All rights reserved.
"""Golden sample gate and hot-update cases for the W4A16 GEMM routes.

Marks: ``gpu`` cases need a CUDA device; ``slow`` cases need the real
Llama-3.1-8B AWQ checkpoint under ``models/`` and stay out of the default
suite. The tiny cases run on seeded random weights with zero downloads
besides the tokenizer files copied from the real model directory.
"""

import json
import os

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REAL_MODEL_DIR = os.path.join(REPO_ROOT, 'models', 'llama-3.1-8b-awq')
REAL_REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'awq_ref_llama31_8b.json')
TOKENIZER_SRC = REAL_MODEL_DIR

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA'),
]


def _dump_tiny_reference(tmp_path, tiny_dir):
    """Dump the Triton-path reference for the tiny model in-process.

    The tiny weights are seeded, so this dump is reproducible; it still only proves self-consistent parity. The external
    judge for real semantics is the committed real-model reference.
    """
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from w4a16_golden import dump_reference

    out = os.path.join(str(tmp_path), 'awq_ref_tiny.json')
    return dump_reference(tiny_dir, out, backend='triton'), out


def _make_tiny_model(tmp_path, seed):
    from tests.turbomind.tiny_ct import gen_tiny_ct_hf

    if not os.path.isdir(TOKENIZER_SRC):
        pytest.skip(f'tokenizer source missing: {TOKENIZER_SRC}')
    tiny_dir = os.path.join(str(tmp_path), f'tiny_ct_seed{seed}')
    gen_tiny_ct_hf(tiny_dir, TOKENIZER_SRC, seed=seed)
    return tiny_dir


def _tm_gate(reference, tiny_dir, log_level: str = 'WARNING'):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from w4a16_golden import PROMPTS, gate_against, run_engine

    results = run_engine(tiny_dir, PROMPTS, backend=None, log_level=log_level)
    return gate_against(reference, results), results


def _ct_bucket(tiny_dir):
    """Collect the ct checkpoint tensors of one generated HF directory."""
    from safetensors.torch import load_file

    tensors = load_file(os.path.join(tiny_dir, 'model.safetensors'))
    suffixes = ('weight_packed', 'weight_scale', 'weight_zero_point')
    return [(name, tensor) for name, tensor in tensors.items() if name.endswith(suffixes)]


def _greedy_ids(tiny_dir, prompts):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from w4a16_golden import MAX_NEW_TOKENS

    from lmdeploy import pipeline
    from lmdeploy.messages import GenerationConfig, PytorchEngineConfig

    engine_config = PytorchEngineConfig(session_len=256,
                                        max_batch_size=8,
                                        cache_max_entry_count=0.1,
                                        enable_prefix_caching=False,
                                        distributed_executor_backend='uni')
    pipe = pipeline(tiny_dir, backend_config=engine_config, log_level='WARNING')
    try:
        gen_config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, ignore_eos=True, random_seed=0)
        responses = pipe(prompts, gen_config=gen_config, do_preprocess=False)
        model = pipe.async_engine.engine.executor.model_agent.patched_model.get_model()
        return [list(r.token_ids) for r in responses], model
    finally:
        pipe.close()


class TestTinyCtGoldenGate:
    """Engine e2e on the seeded tiny ct model: load-time conversion, TM
    route, spill, and the golden gate."""

    def test_tiny_ct_tm_route_matches_triton_reference(self, tmp_path):
        """Ct checkpoint -> canonical AWQ -> TM channel passes the gate."""
        import logging

        tiny_dir = _make_tiny_model(tmp_path, seed=9)
        reference, _ = _dump_tiny_reference(tmp_path, tiny_dir)

        # A direct handler on the lmdeploy logger: lmdeploy's own init may
        # lower root handler levels, which would silence caplog.
        logger = logging.getLogger('lmdeploy')
        records: list[str] = []

        class _Collector(logging.Handler):

            def emit(self, record):
                records.append(record.getMessage())

        collector = _Collector()
        logger.addHandler(collector)
        old_level = logger.level
        logger.setLevel(logging.INFO)
        try:
            gate, results = _tm_gate(reference, tiny_dir, log_level='INFO')
        finally:
            logger.removeHandler(collector)
            logger.setLevel(old_level)
        assert gate['top1_agreement'] == 1.0, gate
        assert gate['worst_truncated_kl'] <= gate['kl_threshold'], gate
        # Random weights carry no margin between top candidates, so a tiny
        # first-token KL can still flip a later greedy tie. The frozen
        # real-model reference carries the strict agreement check.
        assert gate['greedy_agreement'] >= 0.8, gate
        # Per-module routing choices are visible in the engine log.
        routing = [m for m in records if 'W4A16 GEMM routing' in m]
        assert any('-> turbomind' in m for m in routing)
        assert not any('-> awq triton' in m for m in routing), routing

    def test_tiny_ct_spills_packed_torch_weights(self, tmp_path):
        """The TM route keeps no raw torch weights on the GPU after load."""
        tiny_dir = _make_tiny_model(tmp_path, seed=9)
        _, model = _greedy_ids(tiny_dir, ['hello'])
        from lmdeploy.pytorch.nn.linear.tm_gemm import MergedTmGemmLinear, QKVTmGemmLinear, TmGemmLinear

        tm_modules = [
            mod for mod in model.modules() if isinstance(mod, (TmGemmLinear, MergedTmGemmLinear, QKVTmGemmLinear))
        ]
        assert len(tm_modules) == 8  # 2 layers x (qkv, o, gate_up, down)
        for mod in tm_modules:
            # Spilled: the raw torch storage is empty; the TM channel holds
            # the packed copy.
            assert mod.qweight.data.numel() == 0
            assert mod._spill is not None
            assert mod._spill.resident_bytes > 0
        resident = sum(p.data.numel() * p.data.element_size() for p in model.parameters() if p.data.is_cuda)
        assert resident > 0

    def test_hot_update_repack_keeps_golden_gate(self, tmp_path):
        """CPU copy -> restore -> re-pack keeps the golden numbers."""
        tiny_dir = _make_tiny_model(tmp_path, seed=9)
        reference, _ = _dump_tiny_reference(tmp_path, tiny_dir)

        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from w4a16_golden import PROMPTS, gate_against, run_engine

        results = run_engine(tiny_dir, PROMPTS, backend=None)
        gate = gate_against(reference, results)
        assert gate['top1_agreement'] == 1.0
        assert gate['greedy_agreement'] >= 0.8
        before = [r['greedy_ids'] for r in results]

        # Hot update with the same weights, fed as raw ct checkpoint
        # tensors, must preserve the outputs bit for bit.
        from lmdeploy import pipeline
        from lmdeploy.messages import PytorchEngineConfig

        engine_config = PytorchEngineConfig(session_len=256,
                                            max_batch_size=8,
                                            cache_max_entry_count=0.1,
                                            enable_prefix_caching=False,
                                            distributed_executor_backend='uni')
        pipe = pipeline(tiny_dir, backend_config=engine_config, log_level='WARNING')
        try:
            agent = pipe.async_engine.engine.executor.model_agent
            agent._load_updated_weight_bucket(_ct_bucket(tiny_dir), 'update_weights_golden_test')
            agent._finalize_updated_weights()
            from lmdeploy.messages import GenerationConfig

            gen_config = GenerationConfig(max_new_tokens=16, do_sample=False, ignore_eos=True, random_seed=0)
            after = [list(r.token_ids) for r in pipe(PROMPTS, gen_config=gen_config, do_preprocess=False)]
        finally:
            pipe.close()
        assert after == before

    def test_hot_update_changes_outputs_and_roundtrips(self, tmp_path):
        """A different-seed re-feed proves the update lands; an A-B-A roundtrip
        proves the CPU copies and the re-pack preserve values."""
        tiny_dir_a = _make_tiny_model(tmp_path, seed=9)
        tiny_dir_b = _make_tiny_model(tmp_path, seed=10)
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from w4a16_golden import PROMPTS

        prompts = PROMPTS[:4]

        from lmdeploy import pipeline
        from lmdeploy.messages import GenerationConfig, PytorchEngineConfig

        engine_config = PytorchEngineConfig(session_len=256,
                                            max_batch_size=8,
                                            cache_max_entry_count=0.1,
                                            enable_prefix_caching=False,
                                            distributed_executor_backend='uni')
        gen_config = GenerationConfig(max_new_tokens=16, do_sample=False, ignore_eos=True, random_seed=0)
        pipe = pipeline(tiny_dir_a, backend_config=engine_config, log_level='WARNING')
        try:
            agent = pipe.async_engine.engine.executor.model_agent

            def run():
                return [list(r.token_ids) for r in pipe(prompts, gen_config=gen_config, do_preprocess=False)]

            def update(tiny_dir):
                # Two feedings of the same bucket must give the same ids:
                # the restore and re-pack path is deterministic.
                agent._load_updated_weight_bucket(_ct_bucket(tiny_dir), 'update_weights_golden_test')
                agent._finalize_updated_weights()

            before_a = run()
            update(tiny_dir_b)
            after_b = run()
            update(tiny_dir_b)
            after_b_again = run()
            update(tiny_dir_a)
            back_to_a = run()
        finally:
            pipe.close()
        assert after_b != before_a
        assert after_b_again == after_b
        # The A-B-A roundtrip returns to the original outputs exactly: the
        # spilled CPU copies and the re-pack preserved every value.
        assert back_to_a == before_a

    def test_cudagraph_replay_matches_eager(self, tmp_path):
        """Observation item from ticket 02: TM route under CUDA Graph capture.

        No hard gate; the case asserts the graph runner captures, replays,
        and reproduces the eager greedy ids on this route.
        """
        tiny_dir = _make_tiny_model(tmp_path, seed=9)
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from w4a16_golden import PROMPTS

        from lmdeploy import pipeline
        from lmdeploy.messages import GenerationConfig, PytorchEngineConfig

        prompts = PROMPTS[:4]
        gen_config = GenerationConfig(max_new_tokens=8, do_sample=False, ignore_eos=True, random_seed=0)
        engine_config = PytorchEngineConfig(session_len=256,
                                            max_batch_size=8,
                                            cache_max_entry_count=0.1,
                                            enable_prefix_caching=False,
                                            cudagraph_capture_batch_sizes=[4],
                                            distributed_executor_backend='uni')
        pipe = pipeline(tiny_dir, backend_config=engine_config, log_level='WARNING')
        try:
            graph_ids = [list(r.token_ids) for r in pipe(prompts, gen_config=gen_config, do_preprocess=False)]
        finally:
            pipe.close()
        engine_config.cudagraph_capture_batch_sizes = None
        pipe = pipeline(tiny_dir, backend_config=engine_config, log_level='WARNING')
        try:
            eager_ids = [list(r.token_ids) for r in pipe(prompts, gen_config=gen_config, do_preprocess=False)]
        finally:
            pipe.close()
        assert graph_ids == eager_ids


class TestRealModelGoldenGate:
    """Slow gate against the frozen real-model reference."""

    @pytest.mark.slow
    @pytest.mark.skipif(not os.path.isfile(REAL_REF_PATH),
                        reason=f'golden reference missing: {REAL_REF_PATH}')
    @pytest.mark.skipif(not os.path.isdir(REAL_MODEL_DIR),
                        reason=f'real model missing: {REAL_MODEL_DIR}')
    def test_real_model_tm_route_matches_reference(self):
        """The TM route reproduces the AWQ Triton reference on Llama-3.1-8B."""
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from w4a16_golden import PROMPTS, gate_against, run_engine

        with open(REAL_REF_PATH, encoding='utf-8') as f:
            reference = json.load(f)
        results = run_engine(REAL_MODEL_DIR, PROMPTS, backend=None)
        gate = gate_against(reference, results)
        assert gate['top1_agreement'] >= 0.97, gate
        assert gate['greedy_agreement'] >= 0.97, gate
        assert gate['worst_truncated_kl'] <= gate['kl_threshold'], gate
