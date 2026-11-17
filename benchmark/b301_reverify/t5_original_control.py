#!/usr/bin/env python
"""T5 (control): reproduce the original agent's evidence probe verbatim.

The original agent's self-written probe kernel is NOT in the repo
(bench_b301_bridge.py contains no @triton.jit; .scratch/tm-gemm-graft holds
docs only) - so the only concrete kernel claimed miscompiled is the production
fill_kv_cache. Here we run the original agent's own
`check_production_fill_anomaly` harness verbatim (imported, not rewritten) as
the reproduction control against my independent T4 harness.
"""
import sys

import torch

sys.path.insert(0, 'benchmark')
import importlib.util

spec = importlib.util.spec_from_file_location('bridge', 'benchmark/bench_b301_bridge.py')
bridge = importlib.util.module_from_spec(spec)
sys.modules['bridge'] = bridge
spec.loader.exec_module(bridge)  # imports lmdeploy + _tm; main() not executed

out = bridge.check_production_fill_anomaly()
torch.cuda.synchronize()
print('T5 control (original agent harness, verbatim):', out)
ok = out['blocks_with_valid_params'] == out['blocks_total']
print(f"T5 VERDICT: {'production fill intact -> claim NOT reproduced' if ok else 'claim REPRODUCED on original harness'}")
sys.exit(0 if ok else 1)
